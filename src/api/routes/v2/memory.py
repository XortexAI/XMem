from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.api.dependencies import enforce_rate_limit, require_api_key, require_ready
from src.api.routes import memory as memory_v1
from src.api.routes.v2.shared import (
    _error,
    _wrap,
    accepted_job,
    elapsed_ms,
    job_status_data,
    read_user_job,
)
from src.api.routes.v2.temporal_client import start_job_workflow
from src.api.schemas import APIResponse, BatchIngestRequest, IngestRequest, ScrapeRequest, StatusEnum
from src.billing import InsufficientCredits, get_default_billing_service
from src.config import settings
from src.jobs.durable import QUEUED, get_default_job_store, idempotency_key, new_attempt_id, stable_hash

router = APIRouter(
    prefix="/v2/memory",
    tags=["memory-v2"],
    dependencies=[Depends(require_ready), Depends(enforce_rate_limit)],
)

scrape_router = APIRouter(
    prefix="/v2/memory",
    tags=["memory-v2"],
    dependencies=[Depends(enforce_rate_limit)],
)


def _content_hash(payload: Dict[str, Any]) -> str:
    return stable_hash(payload)


def _durable_job_id(job_type: str, fields: Dict[str, Any]) -> str:
    return f"{job_type}:{idempotency_key(job_type, fields)}"


class WorkflowStartFailed(RuntimeError):
    def __init__(self, job: Dict[str, Any], error: str) -> None:
        super().__init__(error)
        self.job = job


def _workflow_start_error(
    request: Request,
    job: Dict[str, Any],
    detail: str,
    status_url: str,
    elapsed: float,
) -> JSONResponse:
    body = APIResponse(
        status=StatusEnum.ERROR,
        request_id=getattr(request.state, "request_id", None),
        data={
            "job_id": job["job_id"],
            "job_type": job.get("job_type"),
            "status": job.get("status"),
            "status_url": status_url,
        },
        error=detail,
        elapsed_ms=elapsed,
    )
    return JSONResponse(content=body.model_dump(), status_code=503)


async def _enqueue_and_start(
    *,
    job_type: str,
    payload: Dict[str, Any],
    idempotency_fields: Dict[str, Any],
    user_id: str,
    timeout_seconds: float,
    max_attempts: int,
) -> tuple[Dict[str, Any], bool]:
    store = get_default_job_store()
    job, created = await asyncio.to_thread(
        store.enqueue,
        job_type=job_type,
        payload=payload,
        idempotency_fields=idempotency_fields,
        user_id=user_id,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )
    should_start = created or (job.get("status") == QUEUED and not job.get("workflow_id"))
    if should_start:
        workflow_id = job.get("workflow_id") or f"{job['job_id']}:{new_attempt_id()}"
        reserved = await asyncio.to_thread(
            store.reserve_workflow_start,
            job["job_id"],
            workflow_id,
        )
        job = await asyncio.to_thread(store.get, job["job_id"]) or job
    else:
        reserved = False
    if reserved:
        try:
            await start_job_workflow(job)
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            await asyncio.to_thread(store.mark_failed, job["job_id"], error)
            job = await asyncio.to_thread(store.get, job["job_id"]) or job
            raise WorkflowStartFailed(job, error) from exc
        job = await asyncio.to_thread(store.get, job["job_id"]) or job
    return job, created


@router.post("/ingest", response_model=APIResponse, summary="Start an async durable memory ingest job")
async def ingest_memory_v2(req: IngestRequest, request: Request, user: dict = Depends(require_api_key)):
    start = time.perf_counter()
    user_id = memory_v1._current_user_id(user, req.user_id)
    job_user_id = memory_v1._current_user_id(user)
    payload = req.model_dump()
    payload["user_id"] = user_id
    payload["timeout_seconds"] = float(settings.memory_ingest_timeout_seconds)
    idempotency_fields = {
        "user_id": user_id,
        "org_id": payload.get("org_id", "default"),
        "content_hash": _content_hash({
            "user_query": req.user_query,
            "agent_response": req.agent_response or "",
            "session_datetime": req.session_datetime,
            "image_url": req.image_url,
            "effort_level": req.effort_level,
        }),
    }
    job_id = _durable_job_id("memory_ingest", idempotency_fields)
    billing_service = get_default_billing_service()
    billing_reservation_created = False

    try:
        account, estimate, reservation = await asyncio.to_thread(
            billing_service.reserve_job_credits,
            user=user,
            job_id=job_id,
            job_type="memory_ingest",
            payload=payload,
        )
        payload["billing_account_id"] = account["id"]
        payload["billing_reservation_id"] = reservation.reservation_id
        payload["billing_estimate"] = estimate.model_dump()
        billing_reservation_created = reservation.created
        job, created = await _enqueue_and_start(
            job_type="memory_ingest",
            payload=payload,
            idempotency_fields=idempotency_fields,
            user_id=job_user_id,
            timeout_seconds=float(settings.memory_ingest_timeout_seconds),
            max_attempts=3,
        )
        return accepted_job(
            request,
            job,
            created,
            f"/v2/memory/ingest/{job['job_id']}/status",
            elapsed_ms(start),
        )
    except WorkflowStartFailed as exc:
        if billing_reservation_created and payload.get("billing_account_id"):
            await asyncio.to_thread(
                billing_service.release_job_reservation,
                payload["billing_account_id"],
                job_id,
            )
        return _workflow_start_error(
            request,
            exc.job,
            str(exc),
            f"/v2/memory/ingest/{exc.job['job_id']}/status",
            elapsed_ms(start),
        )
    except InsufficientCredits as exc:
        return _error(request, str(exc), 402, elapsed_ms(start))
    except Exception as exc:
        if billing_reservation_created and payload.get("billing_account_id"):
            await asyncio.to_thread(
                billing_service.release_job_reservation,
                payload["billing_account_id"],
                job_id,
            )
        return _error(request, str(exc), 500, elapsed_ms(start))


@router.get("/ingest/{job_id}/status", response_model=APIResponse, summary="Poll an async memory ingest job")
async def ingest_job_status(job_id: str, request: Request, user: dict = Depends(require_api_key)):
    start = time.perf_counter()
    job = await read_user_job(job_id, memory_v1._current_user_id(user))
    if not job:
        return _error(request, "Job not found.", 404, elapsed_ms(start))
    return _wrap(request, job_status_data(job), elapsed_ms(start))


@router.get("/jobs/{job_id}/status", response_model=APIResponse, summary="Poll an async memory job")
async def memory_job_status(job_id: str, request: Request, user: dict = Depends(require_api_key)):
    return await ingest_job_status(job_id, request, user)


@router.post("/batch-ingest", response_model=APIResponse, summary="Start an async durable batch memory ingest job")
async def batch_ingest_memory_v2(req: BatchIngestRequest, request: Request, user: dict = Depends(require_api_key)):
    start = time.perf_counter()
    user_id = memory_v1._current_user_id(user)
    items = [memory_v1._scoped_ingest_payload(user, item) for item in req.items]
    payload = {
        "user_id": user_id,
        "items": items,
        "timeout_seconds": max(
            float(settings.memory_ingest_timeout_seconds),
            min(len(req.items) * float(settings.memory_ingest_timeout_seconds), 3600.0),
        ),
    }
    idempotency_fields = {
        "user_id": user_id,
        "content_hash": _content_hash({"items": items}),
    }
    job_id = _durable_job_id("memory_batch_ingest", idempotency_fields)
    billing_service = get_default_billing_service()
    billing_reservation_created = False

    try:
        account, estimate, reservation = await asyncio.to_thread(
            billing_service.reserve_job_credits,
            user=user,
            job_id=job_id,
            job_type="memory_batch_ingest",
            payload=payload,
        )
        payload["billing_account_id"] = account["id"]
        payload["billing_reservation_id"] = reservation.reservation_id
        payload["billing_estimate"] = estimate.model_dump()
        billing_reservation_created = reservation.created
        job, created = await _enqueue_and_start(
            job_type="memory_batch_ingest",
            payload=payload,
            idempotency_fields=idempotency_fields,
            user_id=user_id,
            timeout_seconds=payload["timeout_seconds"],
            max_attempts=3,
        )
        return accepted_job(
            request,
            job,
            created,
            f"/v2/memory/jobs/{job['job_id']}/status",
            elapsed_ms(start),
        )
    except WorkflowStartFailed as exc:
        if billing_reservation_created and payload.get("billing_account_id"):
            await asyncio.to_thread(
                billing_service.release_job_reservation,
                payload["billing_account_id"],
                job_id,
            )
        return _workflow_start_error(
            request,
            exc.job,
            str(exc),
            f"/v2/memory/jobs/{exc.job['job_id']}/status",
            elapsed_ms(start),
        )
    except InsufficientCredits as exc:
        return _error(request, str(exc), 402, elapsed_ms(start))
    except Exception as exc:
        if billing_reservation_created and payload.get("billing_account_id"):
            await asyncio.to_thread(
                billing_service.release_job_reservation,
                payload["billing_account_id"],
                job_id,
            )
        return _error(request, str(exc), 500, elapsed_ms(start))


@scrape_router.post("/scrape", response_model=APIResponse, summary="Start an async durable scrape job")
async def scrape_chat_link_v2(req: ScrapeRequest, request: Request):
    start = time.perf_counter()
    payload = req.model_dump()
    normalized_url = req.url.strip()

    try:
        job, created = await _enqueue_and_start(
            job_type="memory_scrape",
            payload=payload,
            idempotency_fields={"url_hash": _content_hash({"url": normalized_url})},
            user_id="anonymous",
            timeout_seconds=60.0,
            max_attempts=2,
        )
        return accepted_job(
            request,
            job,
            created,
            f"/v2/memory/scrape/{job['job_id']}/status",
            elapsed_ms(start),
        )
    except WorkflowStartFailed as exc:
        return _workflow_start_error(
            request,
            exc.job,
            str(exc),
            f"/v2/memory/scrape/{exc.job['job_id']}/status",
            elapsed_ms(start),
        )
    except Exception as exc:
        return _error(request, str(exc), 500, elapsed_ms(start))


@scrape_router.get("/scrape/{job_id}/status", response_model=APIResponse, summary="Poll an async scrape job")
async def scrape_job_status(job_id: str, request: Request):
    start = time.perf_counter()
    job = await asyncio.to_thread(get_default_job_store().get, job_id)
    if not job or job.get("user_id") != "anonymous":
        return _error(request, "Job not found.", 404, elapsed_ms(start))
    return _wrap(request, job_status_data(job), elapsed_ms(start))
