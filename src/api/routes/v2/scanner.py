from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from src.api.dependencies import enforce_rate_limit, require_api_key
from src.api.routes import scanner as scanner_v1
from src.api.routes.v2.secrets import store_scanner_pat
from src.api.routes.v2.shared import _error, _wrap, accepted_job, elapsed_ms, job_status_data, read_user_job
from src.api.routes.v2.temporal_client import start_job_workflow
from src.api.schemas import APIResponse
from src.jobs.durable import (
    CANCELLED,
    DEAD_LETTER,
    FAILED,
    QUEUED,
    SUCCEEDED,
    get_default_job_store,
    new_attempt_id,
)

logger = logging.getLogger("xmem.api.routes.v2.scanner")

router = APIRouter(
    prefix="/v2/scanner",
    tags=["scanner-v2"],
    dependencies=[Depends(enforce_rate_limit)],
)


def _as_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _scanner_phase_status_from_durable(
    durable_job: Dict[str, Any],
    phase2_only: bool,
) -> tuple[str, str]:
    status = durable_job.get("status")
    if status == SUCCEEDED:
        return "complete", "complete"
    if status in {FAILED, DEAD_LETTER}:
        return ("complete", "failed") if phase2_only else ("failed", "pending")
    if status == CANCELLED:
        return ("complete", "cancelled") if phase2_only else ("cancelled", "pending")
    return ("complete", "running") if phase2_only else ("running", "pending")


@router.post("/scan", summary="Start a durable v2 GitHub repository scan")
async def start_scan_v2(req: scanner_v1.ScanRequest, request: Request, user: dict = Depends(require_api_key)):
    start = time.perf_counter()
    try:
        org, repo = scanner_v1._parse_github_url(req.github_url)
    except ValueError as exc:
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=400)

    username = user.get("username") or user.get("name") or user["id"]
    scanner_job_id = f"{username}:{org}:{repo}"
    store = scanner_v1._get_code_store()
    existing = store.get_scanner_job(scanner_job_id)
    if existing and existing.get("phase1_status") == "running":
        last_updated = existing.get("updated_at")
        since_last_update = (
            datetime.now(timezone.utc) - _as_aware_utc(last_updated)
        ).total_seconds() if last_updated else 0
        if last_updated and since_last_update < 30:
            return JSONResponse({
                "status": "ok",
                "job_id": scanner_job_id,
                "org": org,
                "repo": repo,
                "message": "Scan already in progress",
                "phase1_status": "running",
                "phase2_status": existing.get("phase2_status", "pending"),
                "durable_job_id": existing.get("durable_job_id"),
            })

    clone_url = req.github_url.strip().rstrip("/")
    if not clone_url.endswith(".git"):
        clone_url += ".git"
    branch = (req.branch or "main").strip()
    loop = asyncio.get_running_loop()
    try:
        remote_sha = await loop.run_in_executor(
            None,
            lambda: scanner_v1._get_branch_tip_sha(org, repo, branch, req.pat),
        )
    except Exception as exc:
        error = str(exc) or exc.__class__.__name__
        logger.warning(
            "Failed to inspect GitHub branch %s/%s:%s: %s",
            org,
            repo,
            branch,
            error,
        )
        return JSONResponse({
            "status": "error",
            "org": org,
            "repo": repo,
            "branch": branch,
            "error": f"Failed to inspect GitHub branch: {error}",
        }, status_code=400)
    full_reuse, phase2_only = scanner_v1._can_reuse_index(org, repo, remote_sha)

    if full_reuse:
        now = time.time()
        store.upsert_scanner_job(
            job_id=scanner_job_id,
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
        store.upsert_user_repo_entry(username, org, repo, branch, last_seen_commit=remote_sha)
        return JSONResponse({
            "status": "ok",
            "job_id": scanner_job_id,
            "org": org,
            "repo": repo,
            "reused": True,
            "message": "This revision is already indexed in the shared catalog. Connected without re-scanning.",
            "commit_sha": remote_sha,
            "phase1_status": "complete",
            "phase2_status": "complete",
        })

    durable_type = "scanner_phase2" if phase2_only else "scanner_scan"
    durable_payload = {
        "scanner_job_id": scanner_job_id,
        "username": username,
        "org": org,
        "repo": repo,
        "branch": branch,
        "github_url": clone_url,
        "force_full": req.force_full,
        "remote_sha": remote_sha,
    }
    durable_store = get_default_job_store()
    durable_job, created = await asyncio.to_thread(
        durable_store.enqueue,
        job_type=durable_type,
        payload=durable_payload,
        idempotency_fields={
            "user_id": username,
            "org": org,
            "repo": repo,
            "branch": branch,
            "remote_sha": remote_sha,
            "phase2_only": phase2_only,
            "force_full": req.force_full,
        },
        user_id=username,
        timeout_seconds=scanner_v1.SCANNER_DURABLE_TIMEOUT_SECONDS,
        max_attempts=2,
    )
    temporal_payload = dict(durable_payload)
    if req.pat:
        try:
            temporal_payload["github_credential_ref"] = await asyncio.to_thread(
                store_scanner_pat,
                durable_job["job_id"],
                req.pat,
            )
            await asyncio.to_thread(
                durable_store.update_payload,
                durable_job["job_id"],
                temporal_payload,
            )
            durable_job = await asyncio.to_thread(durable_store.get, durable_job["job_id"]) or durable_job
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            await asyncio.to_thread(durable_store.mark_failed, durable_job["job_id"], error)
            return JSONResponse({
                "status": "error",
                "durable_job_id": durable_job["job_id"],
                "org": org,
                "repo": repo,
                "error": f"Failed to store scanner credentials: {error}",
            }, status_code=503)
    elif isinstance(durable_job.get("payload"), dict) and durable_job["payload"].get("github_credential_ref"):
        temporal_payload["github_credential_ref"] = durable_job["payload"]["github_credential_ref"]

    should_start = created or (durable_job.get("status") == QUEUED and not durable_job.get("workflow_id"))
    if should_start:
        workflow_id = durable_job.get("workflow_id") or f"{durable_job['job_id']}:{new_attempt_id()}"
        reserved = await asyncio.to_thread(
            durable_store.reserve_workflow_start,
            durable_job["job_id"],
            workflow_id,
        )
        durable_job = await asyncio.to_thread(durable_store.get, durable_job["job_id"]) or durable_job
    else:
        reserved = False

    started_at = time.time()
    phase1_status, phase2_status = _scanner_phase_status_from_durable(durable_job, phase2_only)
    scanner_error = durable_job.get("error") if "failed" in {phase1_status, phase2_status} else None
    store.upsert_scanner_job(
        job_id=scanner_job_id,
        username=username,
        org=org,
        repo=repo,
        branch=branch,
        url=clone_url,
        phase1_status=phase1_status,
        phase2_status=phase2_status,
        started_at=started_at,
        error=scanner_error,
        durable_job_id=durable_job["job_id"],
        retry_count=int(durable_job.get("retry_count") or 0),
        timeout_seconds=float(durable_job.get("timeout_seconds") or scanner_v1.SCANNER_DURABLE_TIMEOUT_SECONDS),
    )
    store.upsert_user_repo_entry(username, org, repo, branch)

    if reserved:
        try:
            await start_job_workflow(durable_job, payload=temporal_payload)
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            await asyncio.to_thread(durable_store.mark_failed, durable_job["job_id"], error)
            failed_job = await asyncio.to_thread(durable_store.get, durable_job["job_id"]) or durable_job
            store.upsert_scanner_job(
                job_id=scanner_job_id,
                username=username,
                org=org,
                repo=repo,
                branch=branch,
                url=clone_url,
                phase1_status="complete" if phase2_only else "failed",
                phase2_status="failed" if phase2_only else "pending",
                started_at=started_at,
                error=error,
                durable_job_id=durable_job["job_id"],
                retry_count=int(failed_job.get("retry_count") or 0),
                timeout_seconds=float(failed_job.get("timeout_seconds") or scanner_v1.SCANNER_DURABLE_TIMEOUT_SECONDS),
            )
            return JSONResponse({
                "status": "error",
                "job_id": scanner_job_id,
                "durable_job_id": durable_job["job_id"],
                "org": org,
                "repo": repo,
                "error": f"Failed to start durable scanner workflow: {error}",
            }, status_code=503)
        durable_job = await asyncio.to_thread(durable_store.get, durable_job["job_id"]) or durable_job
        phase1_status, phase2_status = _scanner_phase_status_from_durable(durable_job, phase2_only)

    return accepted_job(
        request,
        durable_job,
        created,
        f"/v2/scanner/jobs/{durable_job['job_id']}/status",
        elapsed_ms(start),
        {
            "scanner_job_id": scanner_job_id,
            "org": org,
            "repo": repo,
            "commit_sha": remote_sha,
            "reused": False,
            "phase2_only": phase2_only,
            "phase1_status": phase1_status,
            "phase2_status": phase2_status,
        },
    )


@router.get("/status", summary="Get v2 durable scan status for a repository")
async def scan_status_v2(
    org_id: str = Query(...),
    repo: str = Query(...),
    user: dict = Depends(require_api_key),
):
    username = user.get("username") or user.get("name") or user["id"]
    scanner_job_id = f"{username}:{org_id}:{repo}"
    store = scanner_v1._get_code_store()
    job = store.get_scanner_job(scanner_job_id)
    if not job:
        return JSONResponse({
            "status": "ok",
            "phase1_status": "not_started",
            "phase2_status": "not_started",
        })

    elapsed = time.time() - float(job.get("started_at") or time.time())
    resp: Dict[str, Any] = {
        "status": "ok",
        "job_id": scanner_job_id,
        "phase1_status": job.get("phase1_status", "not_started"),
        "phase2_status": job.get("phase2_status", "not_started"),
        "elapsed_seconds": round(elapsed, 1),
        "error": job.get("error"),
        "error_state": job.get("error_state"),
        "retry_count": job.get("retry_count", 0),
        "timeout_seconds": job.get("timeout_seconds"),
        "durable_job_id": job.get("durable_job_id"),
    }
    if job.get("durable_job_id"):
        durable = await asyncio.to_thread(get_default_job_store().get, job["durable_job_id"])
        resp["durable_job"] = job_status_data(durable) if durable else None
    if isinstance(job.get("phase1_result"), dict) and job["phase1_result"].get("stats"):
        resp["stats"] = job["phase1_result"]["stats"]
    if isinstance(job.get("phase2_result"), dict):
        resp["phase2_stats"] = job["phase2_result"]
    resp["share_index_publicly"] = store.get_scanner_index_visibility(org_id, repo)
    return JSONResponse(resp)


@router.get("/jobs/{job_id}/status", response_model=APIResponse, summary="Poll a v2 scanner durable job")
async def scanner_job_status(job_id: str, request: Request, user: dict = Depends(require_api_key)):
    start = time.perf_counter()
    user_id = user.get("username") or user.get("name") or user["id"]
    job = await read_user_job(job_id, user_id)
    if not job:
        return _error(request, "Job not found.", 404, elapsed_ms(start))
    return _wrap(request, job_status_data(job), elapsed_ms(start))
