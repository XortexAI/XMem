from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, Query, Request

from src.api.dependencies import enforce_rate_limit, require_api_key
from src.api.routes import memory as memory_v1
from src.api.routes import scanner as scanner_v1
from src.api.routes.v2.shared import (
    _error,
    _wrap,
    elapsed_ms,
    job_status_data,
    read_user_job,
)
from src.api.routes.v2.temporal_client import cancel_job_workflow, start_job_workflow
from src.api.schemas import APIResponse
from src.billing import (
    InsufficientCredits,
    get_default_billing_service,
    release_job_billing,
    release_job_reservation,
)
from src.jobs.durable import CANCELLED, DEAD_LETTER, QUEUED, RUNNING, get_default_job_store

router = APIRouter(
    prefix="/v2/jobs",
    tags=["jobs-v2"],
    dependencies=[Depends(enforce_rate_limit)],
)


def _mark_scanner_job_cancelled(job: dict) -> None:
    scanner_job_types = {
        "scanner_scan",
        "scanner_phase2",
        "scanner_scan_resume",
        "scanner_phase2_resume",
    }
    if job.get("job_type") not in scanner_job_types:
        return
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    scanner_job_id = payload.get("scanner_job_id")
    if not scanner_job_id:
        return

    store = scanner_v1._get_code_store()
    existing = store.get_scanner_job(scanner_job_id)
    if not existing:
        return

    phase1_status = existing.get("phase1_status", "not_started")
    phase2_status = existing.get("phase2_status", "not_started")
    if phase1_status == "running":
        phase1_status = "cancelled"
    if phase2_status == "running":
        phase2_status = "cancelled"

    store.upsert_scanner_job(
        job_id=scanner_job_id,
        username=(
            existing.get("username")
            or job.get("user_id")
            or payload.get("username")
            or ""
        ),
        org=existing.get("org") or payload.get("org") or "",
        repo=existing.get("repo") or payload.get("repo") or "",
        branch=existing.get("branch") or payload.get("branch") or "main",
        url=existing.get("url") or payload.get("github_url") or "",
        phase1_status=phase1_status,
        phase2_status=phase2_status,
        started_at=float(existing.get("started_at") or time.time()),
        error="Scan cancelled.",
        phase1_result=existing.get("phase1_result"),
        phase2_result=existing.get("phase2_result"),
        durable_job_id=job.get("job_id"),
        retry_count=int(job.get("retry_count") or 0),
        timeout_seconds=float(job.get("timeout_seconds") or 0) or None,
    )


@router.get("/{job_id}/status", response_model=APIResponse, summary="Poll a durable v2 job")
async def get_job_status(job_id: str, request: Request, user: dict = Depends(require_api_key)):
    start = time.perf_counter()
    job = await read_user_job(job_id, memory_v1._current_user_id(user))
    if not job:
        return _error(request, "Job not found.", 404, elapsed_ms(start))
    return _wrap(request, job_status_data(job), elapsed_ms(start))


@router.get("/dead-letter", response_model=APIResponse, summary="List your dead-lettered v2 jobs")
async def list_dead_letter_jobs(
    request: Request,
    user: dict = Depends(require_api_key),
    limit: int = Query(default=50, ge=1, le=200),
):
    start = time.perf_counter()
    user_id = memory_v1._current_user_id(user)
    jobs = await asyncio.to_thread(
        get_default_job_store().list_by_status,
        DEAD_LETTER,
        user_id,
        limit,
    )
    return _wrap(request, [job_status_data(job) for job in jobs], elapsed_ms(start))


@router.post("/{job_id}/retry", response_model=APIResponse, summary="Retry a failed or dead-lettered v2 job")
async def retry_job(job_id: str, request: Request, user: dict = Depends(require_api_key)):
    start = time.perf_counter()
    user_id = memory_v1._current_user_id(user)
    job = await read_user_job(job_id, user_id)
    if not job:
        return _error(request, "Job not found.", 404, elapsed_ms(start))
    if job.get("status") not in {"failed", "dead_letter", "cancelled"}:
        return _error(request, "Only failed, dead-lettered, or cancelled jobs can be retried.", 409, elapsed_ms(start))

    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    billing_account_id = payload.get("billing_account_id")
    billing_reservation_created = False
    try:
        if billing_account_id:
            billing_service = get_default_billing_service()
            estimate = billing_service.estimate_required_credits(job.get("job_type") or "", payload)
            reservation = await asyncio.to_thread(
                billing_service.reserve_credits,
                billing_account_id,
                job_id,
                estimate.reserved_credits,
            )
            payload["billing_reservation_id"] = reservation.reservation_id
            payload["billing_estimate"] = estimate.model_dump()
            billing_reservation_created = reservation.created
            await asyncio.to_thread(get_default_job_store().update_payload, job_id, payload)
        await asyncio.to_thread(get_default_job_store().reset_for_retry, job_id, True)
        job = await asyncio.to_thread(get_default_job_store().get, job_id)
        await start_job_workflow(job)
    except InsufficientCredits as exc:
        return _error(request, str(exc), 402, elapsed_ms(start))
    except Exception as exc:
        release_error = None
        if billing_reservation_created and billing_account_id:
            try:
                await asyncio.to_thread(release_job_reservation, billing_account_id, job_id)
            except Exception as release_exc:
                release_error = str(release_exc) or release_exc.__class__.__name__
        error = str(exc) or exc.__class__.__name__
        if release_error:
            error = f"{error}; billing reservation release failed: {release_error}"
        await asyncio.to_thread(get_default_job_store().mark_failed, job_id, error)
        return _error(request, f"Retry failed to start workflow: {error}", 503, elapsed_ms(start))
    job = await asyncio.to_thread(get_default_job_store().get, job_id)
    return _wrap(request, job_status_data(job), elapsed_ms(start))


@router.post("/{job_id}/cancel", response_model=APIResponse, summary="Cancel a running v2 job")
async def cancel_job(job_id: str, request: Request, user: dict = Depends(require_api_key)):
    start = time.perf_counter()
    user_id = memory_v1._current_user_id(user)
    job = await read_user_job(job_id, user_id)
    if not job:
        return _error(request, "Job not found.", 404, elapsed_ms(start))
    if job.get("status") not in {QUEUED, RUNNING, CANCELLED}:
        return _error(request, "Only queued, running, or already-cancelled jobs can be cancelled.", 409, elapsed_ms(start))
    if job.get("status") != CANCELLED:
        cancel_signal_sent = False
        try:
            await cancel_job_workflow(job)
            cancel_signal_sent = True
            await asyncio.to_thread(get_default_job_store().mark_cancelled, job_id)
            await asyncio.to_thread(_mark_scanner_job_cancelled, job)
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            if cancel_signal_sent:
                return _error(request, f"Cancel reached workflow but failed to persist status: {error}", 503, elapsed_ms(start))
            return _error(request, f"Cancel failed to reach workflow: {error}", 503, elapsed_ms(start))
        job = await asyncio.to_thread(get_default_job_store().get, job_id)
    await asyncio.to_thread(release_job_billing, job, "cancelled")
    job = await asyncio.to_thread(get_default_job_store().get, job_id)
    return _wrap(request, job_status_data(job), elapsed_ms(start))
