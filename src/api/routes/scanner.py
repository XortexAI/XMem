"""
/v1/scanner/* routes â€” Scanner dashboard API for the XMem web UI.

Provides endpoints for:
  - Validating GitHub URLs and detecting public/private repos
  - Pre-scan time/token/cost estimates (heuristic)
  - Starting Phase 1 (AST scan) + Phase 2 (LLM enrichment) pipelines
  - Polling scan status (persisted in MongoDB)
  - Listing user repos (includes community index sharing flag)
  - Setting per-repo ``share_index_publicly`` (community catalog vs scanner-only)
  - Listing org/repos marked as community-shared
  - Community catalog (browsable public indexes + stars)
  - Catalog status (works for repos you opened from the community without a personal job row)
  - Chat with indexed codebases (streaming NDJSON; respects sharing rules)

These routes now REQUIRE API-key authentication via the
Authorization: Bearer <token> header. The username is derived from
the authenticated user, not from request parameters.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Query, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies import require_api_key
from src.config import settings

logger = logging.getLogger("xmem.api.routes.scanner")

router = APIRouter(prefix="/v1/scanner", tags=["scanner"])

_code_store_singleton: Any = None


def _get_code_store():
    global _code_store_singleton
    if _code_store_singleton is None:
        from src.scanner.code_store import CodeStore

        _code_store_singleton = CodeStore(
            uri=settings.mongodb_uri,
            database=settings.mongodb_database,
        )
    return _code_store_singleton


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Request schemas
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class ValidateUrlRequest(BaseModel):
    github_url: str = Field(..., min_length=1)
    pat: str = Field(default="")
    branch: str = Field(
        default="",
        description="Optional branch for estimates (defaults to repo default_branch)",
    )


class ScanRequest(BaseModel):
    github_url: str = Field(..., min_length=1)
    username: str = Field(..., min_length=1)
    pat: str = Field(default="")
    branch: str = Field(default="main")


class ChatRequest(BaseModel):
    org_id: str = Field(..., min_length=1)
    repo: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    username: str = Field(default="")
    top_k: int = Field(default=10, ge=1, le=50)


class ShareIndexRequest(BaseModel):
    """Mark whether anyone may query the shared catalog for this org/repo."""

    username: str = Field(..., min_length=1)
    org_id: str = Field(..., min_length=1)
    repo: str = Field(..., min_length=1)
    share_index_publicly: bool = Field(
        ...,
        description="If true, any scanner user may chat; if false, only users who indexed this repo.",
    )


class CommunityStarRequest(BaseModel):
    username: str = Field(..., min_length=1)
    org_id: str = Field(..., min_length=1)
    repo: str = Field(..., min_length=1)
    starred: bool = Field(..., description="True to star, false to remove star")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Helpers â€” GitHub API
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def _parse_github_url(url: str) -> tuple:
    """Extract (org, repo) from a GitHub URL."""
    url = url.strip().rstrip("/")
    m = re.search(r"github\.com[/:]([^/]+)/([^/.]+?)(?:\.git)?$", url)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(
        "Invalid GitHub URL. Expected format: https://github.com/org/repo"
    )


def _github_headers(pat: str = "") -> Dict[str, str]:
    h = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "XMem-Scanner/1.0",
    }
    if pat:
        h["Authorization"] = f"token {pat}"
    return h


def _github_get_json(path: str, pat: str = "") -> Dict[str, Any]:
    """GET api.github.com path (leading slash optional)."""
    path = path if path.startswith("/") else f"/{path}"
    url = f"https://api.github.com{path}"
    req = urllib.request.Request(url, headers=_github_headers(pat))
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _check_github_repo(org: str, repo: str, pat: str = "") -> dict:
    """Hit the GitHub API to check repo accessibility and metadata."""
    api_url = f"https://api.github.com/repos/{org}/{repo}"
    req = urllib.request.Request(api_url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "XMem-Scanner/1.0")
    if pat:
        req.add_header("Authorization", f"token {pat}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            size_kb = int(data.get("size") or 0)
            return {
                "accessible": True,
                "is_private": data.get("private", False),
                "default_branch": data.get("default_branch", "main"),
                "description": data.get("description") or "",
                "language": data.get("language") or "",
                "size_kb": size_kb,
            }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"accessible": False, "needs_pat": not pat}
        if e.code in (401, 403):
            return {"accessible": False, "auth_error": True}
        return {"accessible": False, "error": f"GitHub API error: {e.code}"}
    except Exception as e:
        return {"accessible": False, "error": str(e)}


def _get_branch_tip_sha(
    org: str, repo: str, branch: str, pat: str = "",
) -> Optional[str]:
    """Resolve the commit SHA at the tip of ``branch``."""
    ref = urllib.parse.quote(branch, safe="")
    path = f"/repos/{org}/{repo}/commits/{ref}"
    try:
        data = _github_get_json(path, pat)
        sha = data.get("sha")
        if isinstance(sha, str) and len(sha) >= 7:
            return sha
    except urllib.error.HTTPError as e:
        logger.warning("Could not resolve branch tip %s/%s@%s: %s", org, repo, branch, e)
    except Exception as e:
        logger.warning("Branch tip resolution failed: %s", e)
    return None


def _compute_scan_estimates(size_kb: int, branch_label: str) -> Dict[str, Any]:
    """Heuristic pre-scan estimates; all values are approximate."""
    s = settings
    size_mb = max(size_kb / 1024.0, 0.01)

    phase1_sec = (
        s.scanner_estimate_phase1_base_seconds
        + size_mb * s.scanner_estimate_phase1_seconds_per_mb
    )
    emb_calls = size_mb * s.scanner_estimate_embedding_calls_per_mb
    emb_tokens = int(emb_calls * s.scanner_estimate_avg_tokens_per_embedding)
    llm_tokens = int(size_mb * s.scanner_estimate_llm_tokens_per_mb)

    cost: Optional[float] = None
    if (
        s.scanner_estimate_embedding_cost_per_1m_tokens is not None
        and s.scanner_estimate_llm_cost_per_1m_tokens is not None
    ):
        cost = (
            (emb_tokens / 1_000_000.0)
            * float(s.scanner_estimate_embedding_cost_per_1m_tokens)
            + (llm_tokens / 1_000_000.0)
            * float(s.scanner_estimate_llm_cost_per_1m_tokens)
        )

    return {
        "estimate_disclaimer": (
            "Estimates are approximate. Actual time and token use depend on "
            "file types, parsers, and API pricing."
        ),
        "branch_used_for_label": branch_label,
        "repo_size_kb": size_kb,
        "estimated_phase1_seconds": round(phase1_sec, 1),
        "estimated_embedding_api_calls": int(emb_calls),
        "estimated_embedding_tokens": emb_tokens,
        "estimated_phase2_llm_tokens": llm_tokens,
        "estimated_cost_usd": round(cost, 4) if cost is not None else None,
    }


def _scanner_chat_allowed(store, username: str, org_id: str, repo: str) -> Optional[str]:
    """Return None if allowed, else an error message."""
    snap = store.get_catalog_repo_snapshot(username, org_id, repo)
    if snap.get("phase1_status") != "complete":
        return "Index is not ready yet. Wait for scanning to finish."
    if snap.get("share_index_publicly"):
        return None
    if snap.get("source") == "user_job" and snap.get("phase1_status") == "complete":
        return None
    return (
        "This index is private. Sign in with a username that scanned this repo, "
        "or ask the owner to enable community sharing for this index."
    )


def _can_reuse_index(
    org: str, repo: str, remote_sha: Optional[str],
) -> Tuple[bool, bool]:
    """Return (fully_reusable, phase2_only).

    fully_reusable: Phase 1+2 done for this commit.
    phase2_only: Phase 1 data exists at this commit but enrichment pending.
    """
    if not remote_sha:
        return False, False

    store = _get_code_store()
    last = store.get_last_scan(org, repo)
    if not last or last.get("status") != "completed":
        return False, False
    if last.get("last_commit_sha") != remote_sha:
        return False, False

    pending = store.count_unenriched(org, repo)
    if pending["symbols"] == 0 and pending["files"] == 0:
        return True, False
    return False, True


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Background scan execution
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def _run_phase1(job_id: str, username: str, started_at: float, org: str, repo: str, url: str, branch: str, pat: str) -> dict:
    from src.scanner_v1.indexer import IndexerV1
    from src.scanner_v1.store import CodeStoreV1
    from src.scanner_v1.embedder import Embedder
    from src.pipelines.ingest import embed_text
    
    store = CodeStoreV1(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=None,
        embedding_dimension=settings.pinecone_dimension,
    )
    store.connect()
    # Ensure vector and fulltext indexes are explicitly created
    # so search indices exist even if Neo4j was freshly wiped.
    store.setup_schema()

    def _embed(text: str):
        return list(embed_text(text))

    embedder = Embedder(summary_embed_fn=_embed)
    indexer = IndexerV1(org_id=org, store=store, embedder=embedder)
    def _progress(stats: dict):
        _persist_job(
            job_id, username, org, repo, branch, url, started_at,
            phase1_status="running",
            phase2_status="pending",
            phase1_result={"stats": stats}
        )

    try:
        return indexer.scan_repo(
            repo_name=repo,
            repo_url=url,
            branch=branch,
            token=pat or None,
            force_full=True,
            progress_cb=_progress,
        )
    finally:
        indexer.close()


def _run_phase2(job_id: str, username: str, started_at: float, org: str, repo: str, url: str, branch: str, phase1_result: Optional[dict]) -> dict:
    from src.scanner_v1.enricher import EnricherV1
    from src.scanner_v1.store import CodeStoreV1
    from src.scanner_v1.embedder import Embedder
    from src.pipelines.ingest import embed_text
    from src.models import get_model

    store = CodeStoreV1(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=None,
        embedding_dimension=settings.pinecone_dimension,
    )
    store.connect()

    def _embed(text: str):
        return list(embed_text(text))

    embedder = Embedder(summary_embed_fn=_embed)
    
    model = get_model()
    def _llm_call(prompt: str) -> str:
        response = model.invoke(prompt)
        return getattr(response, "content", None) or str(response)

    enricher = EnricherV1(
        org_id=org, 
        store=store, 
        embedder=embedder, 
        llm_call=_llm_call
    )
    def _progress(stats: dict):
        _persist_job(
            job_id, username, org, repo, branch, url, started_at,
            phase1_status="complete",
            phase2_status="running",
            phase1_result=phase1_result,
            phase2_result=stats
        )

    try:
        return enricher.enrich_repo(repo, progress_cb=_progress)
    finally:
        enricher.close()


def _persist_job(
    job_id: str,
    username: str,
    org: str,
    repo: str,
    branch: str,
    url: str,
    started_at: float,
    phase1_status: str,
    phase2_status: str,
    error: Optional[str] = None,
    phase1_result: Optional[dict] = None,
    phase2_result: Optional[dict] = None,
) -> None:
    store = _get_code_store()
    store.upsert_scanner_job(
        job_id=job_id,
        username=username,
        org=org,
        repo=repo,
        branch=branch,
        url=url,
        phase1_status=phase1_status,
        phase2_status=phase2_status,
        started_at=started_at,
        error=error,
        phase1_result=phase1_result,
        phase2_result=phase2_result,
    )


async def _run_scan_pipeline(
    job_id: str,
    username: str,
    org: str,
    repo: str,
    url: str,
    branch: str,
    pat: str,
):
    """Run Phase 1 then Phase 2 in a background thread."""
    loop = asyncio.get_running_loop()
    store = _get_code_store()
    started = store.get_scanner_job(job_id)
    started_at = started["started_at"] if started else time.time()

    try:
        result = await loop.run_in_executor(
            None, lambda: _run_phase1(job_id, username, started_at, org, repo, url, branch, pat),
        )
        _persist_job(
            job_id, username, org, repo, branch, url, started_at,
            phase1_status="complete",
            phase2_status="running",
            phase1_result=result,
        )
        logger.info("Phase 1 complete for %s/%s", org, repo)
    except Exception as e:
        logger.error("Phase 1 failed for %s/%s: %s", org, repo, e)
        _persist_job(
            job_id, username, org, repo, branch, url, started_at,
            phase1_status="failed",
            phase2_status="pending",
            error=str(e),
        )
        return

    try:
        enrich_result = await loop.run_in_executor(
            None, lambda: _run_phase2(job_id, username, started_at, org, repo, url, branch, result),
        )
        _persist_job(
            job_id, username, org, repo, branch, url, started_at,
            phase1_status="complete",
            phase2_status="complete",
            phase1_result=result,
            phase2_result=enrich_result,
        )
        logger.info("Phase 2 complete for %s/%s", org, repo)
    except Exception as e:
        logger.error("Phase 2 failed for %s/%s: %s", org, repo, e)
        _persist_job(
            job_id, username, org, repo, branch, url, started_at,
            phase1_status="complete",
            phase2_status="failed",
            error=str(e),
            phase1_result=result,
        )


async def _run_phase2_pipeline_only(
    job_id: str,
    username: str,
    org: str,
    repo: str,
    url: str,
    branch: str,
):
    loop = asyncio.get_running_loop()
    store = _get_code_store()
    started = store.get_scanner_job(job_id)
    started_at = started["started_at"] if started else time.time()

    try:
        enrich_result = await loop.run_in_executor(
            None, lambda: _run_phase2(job_id, username, started_at, org, repo, url, branch, started.get("phase1_result") if started else None),
        )
        _persist_job(
            job_id, username, org, repo, branch, url, started_at,
            phase1_status="complete",
            phase2_status="complete",
            phase1_result=started.get("phase1_result") if started else None,
            phase2_result=enrich_result,
        )
        logger.info("Phase 2-only complete for %s/%s", org, repo)
    except Exception as e:
        logger.error("Phase 2-only failed for %s/%s: %s", org, repo, e)
        _persist_job(
            job_id, username, org, repo, branch, url, started_at,
            phase1_status="complete",
            phase2_status="failed",
            error=str(e),
            phase1_result=started.get("phase1_result") if started else None,
        )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Routes
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


@router.post(
    "/validate-url",
    summary="Validate a GitHub URL and check accessibility",
)
async def validate_url(req: ValidateUrlRequest, user: dict = Depends(require_api_key)):
    try:
        org, repo = _parse_github_url(req.github_url)
    except ValueError as e:
        return JSONResponse(
            {"status": "error", "error": str(e)}, status_code=400,
        )

    loop = asyncio.get_running_loop()
    info = await loop.run_in_executor(
        None, lambda: _check_github_repo(org, repo, req.pat),
    )

    payload: Dict[str, Any] = {"status": "ok", "org": org, "repo": repo, **info}

    if info.get("accessible") and info.get("size_kb") is not None:
        branch_label = (req.branch or info.get("default_branch") or "main").strip()
        estimates = _compute_scan_estimates(int(info["size_kb"]), branch_label)
        payload["estimates"] = estimates

    return JSONResponse(payload)


@router.post("/scan", summary="Start scanning a GitHub repository")
async def start_scan(req: ScanRequest):
    try:
        org, repo = _parse_github_url(req.github_url)
    except ValueError as e:
        return JSONResponse(
            {"status": "error", "error": str(e)}, status_code=400,
        )

    job_id = f"{username}:{org}:{repo}"
    store = _get_code_store()

    existing = store.get_scanner_job(job_id)
    if existing and existing.get("phase1_status") == "running":
        # Ensure it's not a zombie job from a previous server crash
        last_updated = existing.get("updated_at")
        since_last_update = (datetime.now(timezone.utc) - last_updated).total_seconds() if last_updated else 0
        if last_updated and since_last_update < 30:
            return JSONResponse({
                "status": "ok",
            "job_id": job_id,
            "org": org,
            "repo": repo,
            "message": "Scan already in progress",
            "phase1_status": "running",
            "phase2_status": existing.get("phase2_status", "pending"),
        })

    clone_url = req.github_url.strip().rstrip("/")
    if not clone_url.endswith(".git"):
        clone_url += ".git"

    branch = (req.branch or "main").strip()
    loop = asyncio.get_running_loop()
    remote_sha = await loop.run_in_executor(
        None, lambda: _get_branch_tip_sha(org, repo, branch, req.pat),
    )

    full_reuse, phase2_only = _can_reuse_index(org, repo, remote_sha)

    if full_reuse:
        now = time.time()
        store.upsert_scanner_job(
            job_id=job_id,
            username=username,
            org=org,
            repo=repo,
            branch=branch,
            url=clone_url,
            phase1_status="complete",
            phase2_status="complete",
            started_at=now,
            error=None,
            phase1_result=None,
            phase2_result=None,
        )
        store.upsert_user_repo_entry(
            username, org, repo, branch,
            last_seen_commit=remote_sha,
        )
        return JSONResponse({
            "status": "ok",
            "job_id": job_id,
            "org": org,
            "repo": repo,
            "reused": True,
            "message": (
                "This revision is already indexed in the shared catalog. "
                "Connected without re-scanning."
            ),
            "commit_sha": remote_sha,
            "phase1_status": "complete",
            "phase2_status": "complete",
        })

    started_at = time.time()
    store.upsert_scanner_job(
        job_id=job_id,
        username=username,
        org=org,
        repo=repo,
        branch=branch,
        url=clone_url,
        phase1_status="running" if not phase2_only else "complete",
        phase2_status="running" if phase2_only else "pending",
        started_at=started_at,
        error=None,
    )
    store.upsert_user_repo_entry(username, org, repo, branch)

    if phase2_only:
        asyncio.create_task(
            _run_phase2_pipeline_only(
                job_id, username, org, repo, clone_url, branch,
            ),
        )
        return JSONResponse({
            "status": "ok",
            "job_id": job_id,
            "org": org,
            "repo": repo,
            "reused": False,
            "phase2_only": True,
            "message": "Index exists; running Phase 2 (LLM enrichment) only.",
            "phase1_status": "complete",
            "phase2_status": "running",
        })

    asyncio.create_task(
        _run_scan_pipeline(
            job_id, username, org, repo, clone_url, branch, req.pat,
        ),
    )

    return JSONResponse({
        "status": "ok",
        "job_id": job_id,
        "org": org,
        "repo": repo,
        "reused": False,
        "phase1_status": "running",
        "phase2_status": "pending",
    })


@router.get("/status", summary="Get scan status for a repository")
async def scan_status(
    org_id: str = Query(...),
    repo: str = Query(...),
    user: dict = Depends(require_api_key),
):
    username = user.get("username") or user.get("name") or user["id"]
    job_id = f"{username}:{org_id}:{repo}"
    store = _get_code_store()
    job = store.get_scanner_job(job_id)

    if not job:
        return JSONResponse({
            "status": "ok",
            "phase1_status": "not_started",
            "phase2_status": "not_started",
        })

    elapsed = time.time() - float(job.get("started_at", time.time()))
    resp: Dict[str, Any] = {
        "status": "ok",
        "phase1_status": job.get("phase1_status", "not_started"),
        "phase2_status": job.get("phase2_status", "not_started"),
        "elapsed_seconds": round(elapsed, 1),
        "error": job.get("error"),
    }

    pr = job.get("phase1_result")
    if isinstance(pr, dict) and pr.get("stats"):
        resp["stats"] = pr["stats"]
        
    p2 = job.get("phase2_result")
    if isinstance(p2, dict):
        resp["phase2_stats"] = p2

    resp["share_index_publicly"] = store.get_scanner_index_visibility(org_id, repo)

    return JSONResponse(resp)


@router.get(
    "/catalog-status",
    summary="Scan status for a repo (your job if any, else shared catalog state)",
)
async def catalog_status(
    username: str = Query(...),
    org_id: str = Query(...),
    repo: str = Query(...),
):
    store = _get_code_store()
    snap = store.get_catalog_repo_snapshot(username, org_id, repo)
    return JSONResponse(snap)


@router.get("/community", summary="Browsable public indexes with star counts")
async def community_catalog(
    username: str = Query(default=""),
    q: str = Query(default="", description="Filter by org or repo name"),
    sort: str = Query(default="stars"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0, le=100_000),
):
    if sort not in ("stars", "recent"):
        sort = "stars"
    store = _get_code_store()
    items, total = store.list_community_catalog_page(
        username=username.strip(),
        q=q,
        sort=sort,
        limit=limit,
        offset=offset,
    )
    return JSONResponse(
        jsonable_encoder({
            "status": "ok",
            "total": total,
            "items": items,
        })
    )


@router.post("/community/star", summary="Star or unstar a public catalog repo")
async def community_star(req: CommunityStarRequest):
    store = _get_code_store()
    if not store.get_scanner_index_visibility(req.org_id, req.repo):
        return JSONResponse(
            {
                "status": "error",
                "error": "This repository is not in the public catalog.",
            },
            status_code=403,
        )
    last = store.get_last_scan(req.org_id, req.repo)
    if not last or last.get("status") != "completed":
        return JSONResponse(
            {
                "status": "error",
                "error": "This repository does not have a completed index yet.",
            },
            status_code=400,
        )
    count = store.set_community_star(
        username.strip(),
        req.org_id,
        req.repo,
        req.starred,
    )
    return JSONResponse(
        {
            "status": "ok",
            "org": req.org_id,
            "repo": req.repo,
            "star_count": count,
            "starred": req.starred,
        },
    )


@router.get("/repos", summary="List scanned repositories for a user")
async def list_repos(username: str = Query(...)):
    store = _get_code_store()
    jobs = store.list_scanner_jobs_for_user(username)
    seen: set[tuple[str, str]] = set()
    repos: list[Dict[str, Any]] = []
    for j in jobs:
        o, r = j.get("org"), j.get("repo")
        if not o or not r:
            continue
        seen.add((o, r))
        repos.append({
            "org": o,
            "repo": r,
            "phase1_status": j.get("phase1_status", "not_started"),
            "phase2_status": j.get("phase2_status", "not_started"),
        })
    for row in store.list_user_repos_for_user(username):
        o, r = row.get("github_org"), row.get("repo")
        if not o or not r:
            continue
        if (o, r) in seen:
            continue
        seen.add((o, r))
        repos.append({
            "org": o,
            "repo": r,
            "phase1_status": "not_started",
            "phase2_status": "not_started",
        })
    pairs = [(r["org"], r["repo"]) for r in repos]
    vis_map = store.get_scanner_index_visibility_batch(pairs)
    for r in repos:
        r["share_index_publicly"] = vis_map.get((r["org"], r["repo"]), True)
    return JSONResponse({"status": "ok", "repos": repos})


@router.get("/public-indexes", summary="List org/repo pairs shared with the community")
async def public_indexes(limit: int = Query(default=100, ge=1, le=500), user: dict = Depends(require_api_key)):
    store = _get_code_store()
    rows = store.list_public_scanner_indexes(limit=limit)
    items = [
        {"org": row["org_id"], "repo": row["repo"]}
        for row in rows
        if row.get("org_id") and row.get("repo")
    ]
    return JSONResponse({"status": "ok", "indexes": items})


@router.post("/index-visibility", summary="Set whether this index is shared with all scanner users")
async def set_index_visibility(req: ShareIndexRequest, user: dict = Depends(require_api_key)):
    username = user.get("username") or user.get("name") or user["id"]
    store = _get_code_store()
    if not store.user_has_completed_scanner_job(username, req.org_id, req.repo):
        return JSONResponse(
            {
                "status": "error",
                "error": "You can only change sharing after Phase 1 has completed for this repository.",
            },
            status_code=403,
        )
    store.set_scanner_index_visibility(
        req.org_id,
        req.repo,
        req.share_index_publicly,
        username,
    )
    return JSONResponse(
        {
            "status": "ok",
            "org": req.org_id,
            "repo": req.repo,
            "share_index_publicly": req.share_index_publicly,
        },
    )


@router.post("/chat", summary="Chat with an indexed codebase (streaming)")
async def chat_with_repo(req: ChatRequest, user: dict = Depends(require_api_key)):
    from src.api.dependencies import get_code_pipeline

    username = user.get("username") or user.get("name") or user["id"]
    store = _get_code_store()
    denied = _scanner_chat_allowed(store, username, req.org_id, req.repo)
    if denied:
        return JSONResponse(
            {"status": "error", "error": denied},
            status_code=403,
        )

    pipeline = get_code_pipeline(org_id=req.org_id, repo=req.repo)

    return StreamingResponse(
        pipeline.run_stream(
            query=req.query,
            user_id=username,
            repo=req.repo,
            top_k=req.top_k,
        ),
        media_type="application/x-ndjson",
    )
