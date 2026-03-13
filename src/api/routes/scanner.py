"""
/v1/scanner/* routes — Scanner dashboard API for the XMem web UI.

Provides endpoints for:
  - Validating GitHub URLs and detecting public/private repos
  - Starting Phase 1 (AST scan) + Phase 2 (LLM enrichment) pipelines
  - Polling scan status
  - Listing user repos
  - Chat with indexed codebases (streaming NDJSON)

These routes have NO API-key authentication — they are designed
for the public scanner dashboard. Rate limiting is handled by the
global middleware.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("xmem.api.routes.scanner")

router = APIRouter(prefix="/v1/scanner", tags=["scanner"])

_scan_jobs: Dict[str, Dict[str, Any]] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Request schemas
# ═══════════════════════════════════════════════════════════════════════════

class ValidateUrlRequest(BaseModel):
    github_url: str = Field(..., min_length=1)
    pat: str = Field(default="")


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


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_github_url(url: str) -> tuple:
    """Extract (org, repo) from a GitHub URL."""
    url = url.strip().rstrip("/")
    m = re.search(r"github\.com[/:]([^/]+)/([^/.]+?)(?:\.git)?$", url)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(
        "Invalid GitHub URL. Expected format: https://github.com/org/repo"
    )


def _check_github_repo(org: str, repo: str, pat: str = "") -> dict:
    """Hit the GitHub API to check repo accessibility."""
    api_url = f"https://api.github.com/repos/{org}/{repo}"
    req = urllib.request.Request(api_url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "XMem-Scanner/1.0")
    if pat:
        req.add_header("Authorization", f"token {pat}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return {
                "accessible": True,
                "is_private": data.get("private", False),
                "default_branch": data.get("default_branch", "main"),
                "description": data.get("description") or "",
                "language": data.get("language") or "",
            }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"accessible": False, "needs_pat": not pat}
        if e.code in (401, 403):
            return {"accessible": False, "auth_error": True}
        return {"accessible": False, "error": f"GitHub API error: {e.code}"}
    except Exception as e:
        return {"accessible": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# Background scan execution
# ═══════════════════════════════════════════════════════════════════════════

def _run_phase1(org: str, repo: str, url: str, branch: str, pat: str) -> dict:
    from src.scanner.indexer import Indexer

    indexer = Indexer(org_id=org)
    try:
        return indexer.scan_repo(
            repo_name=repo,
            repo_url=url,
            branch=branch,
            token=pat or None,
            force_full=True,
        )
    finally:
        indexer.close()


def _run_phase2(org: str, repo: str) -> dict:
    from src.scanner.enricher import Enricher

    enricher = Enricher(org_id=org)
    try:
        return enricher.enrich_repo(repo)
    finally:
        enricher.close()


async def _run_scan_pipeline(
    job_id: str, org: str, repo: str, url: str, branch: str, pat: str,
):
    """Run Phase 1 then Phase 2 in a background thread."""
    loop = asyncio.get_running_loop()

    try:
        result = await loop.run_in_executor(
            None, lambda: _run_phase1(org, repo, url, branch, pat),
        )
        _scan_jobs[job_id]["phase1_status"] = "complete"
        _scan_jobs[job_id]["phase1_result"] = result
        _scan_jobs[job_id]["phase2_status"] = "running"
        logger.info("Phase 1 complete for %s/%s", org, repo)
    except Exception as e:
        logger.error("Phase 1 failed for %s/%s: %s", org, repo, e)
        _scan_jobs[job_id]["phase1_status"] = "failed"
        _scan_jobs[job_id]["error"] = str(e)
        return

    try:
        enrich_result = await loop.run_in_executor(
            None, lambda: _run_phase2(org, repo),
        )
        _scan_jobs[job_id]["phase2_status"] = "complete"
        _scan_jobs[job_id]["phase2_result"] = enrich_result
        logger.info("Phase 2 complete for %s/%s", org, repo)
    except Exception as e:
        logger.error("Phase 2 failed for %s/%s: %s", org, repo, e)
        _scan_jobs[job_id]["phase2_status"] = "failed"
        _scan_jobs[job_id]["error"] = str(e)


# ═══════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/validate-url",
    summary="Validate a GitHub URL and check accessibility",
)
async def validate_url(req: ValidateUrlRequest):
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

    return JSONResponse({"status": "ok", "org": org, "repo": repo, **info})


@router.post("/scan", summary="Start scanning a GitHub repository")
async def start_scan(req: ScanRequest):
    try:
        org, repo = _parse_github_url(req.github_url)
    except ValueError as e:
        return JSONResponse(
            {"status": "error", "error": str(e)}, status_code=400,
        )

    job_id = f"{req.username}:{org}:{repo}"

    existing = _scan_jobs.get(job_id)
    if existing and existing.get("phase1_status") == "running":
        return JSONResponse({
            "status": "ok",
            "job_id": job_id,
            "org": org,
            "repo": repo,
            "message": "Scan already in progress",
            "phase1_status": "running",
            "phase2_status": "pending",
        })

    clone_url = req.github_url.strip().rstrip("/")
    if not clone_url.endswith(".git"):
        clone_url += ".git"

    _scan_jobs[job_id] = {
        "username": req.username,
        "org": org,
        "repo": repo,
        "url": clone_url,
        "branch": req.branch,
        "phase1_status": "running",
        "phase2_status": "pending",
        "started_at": time.time(),
        "phase1_result": None,
        "phase2_result": None,
        "error": None,
    }

    asyncio.create_task(
        _run_scan_pipeline(job_id, org, repo, clone_url, req.branch, req.pat),
    )

    return JSONResponse({
        "status": "ok",
        "job_id": job_id,
        "org": org,
        "repo": repo,
        "phase1_status": "running",
        "phase2_status": "pending",
    })


@router.get("/status", summary="Get scan status for a repository")
async def scan_status(
    username: str = Query(...),
    org_id: str = Query(...),
    repo: str = Query(...),
):
    job_id = f"{username}:{org_id}:{repo}"
    job = _scan_jobs.get(job_id)

    if not job:
        return JSONResponse({
            "status": "ok",
            "phase1_status": "not_started",
            "phase2_status": "not_started",
        })

    elapsed = time.time() - job["started_at"]
    resp: Dict[str, Any] = {
        "status": "ok",
        "phase1_status": job["phase1_status"],
        "phase2_status": job["phase2_status"],
        "elapsed_seconds": round(elapsed, 1),
        "error": job.get("error"),
    }

    if job.get("phase1_result"):
        resp["stats"] = job["phase1_result"].get("stats")

    return JSONResponse(resp)


@router.get("/repos", summary="List scanned repositories for a user")
async def list_repos(username: str = Query(...)):
    repos = []
    for _, job in _scan_jobs.items():
        if job["username"] == username:
            repos.append({
                "org": job["org"],
                "repo": job["repo"],
                "phase1_status": job["phase1_status"],
                "phase2_status": job["phase2_status"],
            })
    return JSONResponse({"status": "ok", "repos": repos})


@router.post("/chat", summary="Chat with an indexed codebase (streaming)")
async def chat_with_repo(req: ChatRequest):
    from src.api.dependencies import get_code_pipeline

    pipeline = get_code_pipeline(org_id=req.org_id, repo=req.repo)

    return StreamingResponse(
        pipeline.run_stream(
            query=req.query,
            user_id=req.username,
            repo=req.repo,
            top_k=req.top_k,
        ),
        media_type="application/x-ndjson",
    )
