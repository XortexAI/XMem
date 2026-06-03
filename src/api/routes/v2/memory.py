from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.api.dependencies import (
    enforce_rate_limit,
    get_retrieval_pipeline,
    require_api_key,
    require_ready,
)
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
from src.api.schemas import (
    APIResponse,
    BatchIngestRequest,
    HybridSearchRequest,
    HybridSearchResponse,
    IngestRequest,
    ScrapeRequest,
    SourceRecord,
    StatusEnum,
)
from src.billing import InsufficientCredits, get_default_billing_service
from src.config import settings
from src.jobs.durable import QUEUED, get_default_job_store, idempotency_key, new_attempt_id, stable_hash
from src.storage.original import ORIGINAL_CHUNK_DOMAIN, original_config_snapshot

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


def _attach_original_storage_config(payload: Dict[str, Any]) -> None:
    payload["original_storage_enabled"] = bool(settings.original_storage_enabled)
    payload["original_storage_fail_closed"] = bool(settings.original_storage_fail_closed)
    payload["original_storage_timeout_seconds"] = float(
        settings.original_storage_timeout_seconds
    )
    payload["original_batch_item_concurrency"] = int(
        settings.original_batch_item_concurrency
    )
    payload["original_config"] = original_config_snapshot()


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
    _attach_original_storage_config(payload)
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
        "original_storage_enabled": bool(settings.original_storage_enabled),
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
    _attach_original_storage_config(payload)
    idempotency_fields = {
        "user_id": user_id,
        "content_hash": _content_hash({"items": items}),
        "original_storage_enabled": bool(settings.original_storage_enabled),
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


async def _search_original_chunks(
    query: str,
    user_id: str,
    top_k: int,
) -> list[SourceRecord]:
    pipeline = get_retrieval_pipeline()
    raw = await pipeline.vector_store.search_by_text(
        query_text=query,
        top_k=top_k,
        filters={"user_id": user_id, "domain": ORIGINAL_CHUNK_DOMAIN},
    )
    results: list[SourceRecord] = []
    for item in raw:
        score = float(item.score or 0.0)
        if score < float(settings.hybrid_search_min_score):
            continue
        results.append(
            SourceRecord(
                domain=ORIGINAL_CHUNK_DOMAIN,
                content=item.content,
                score=round(score, 3),
                metadata={"id": item.id, **item.metadata},
            )
        )
    return results


@router.post(
    "/hybrid-search",
    response_model=APIResponse,
    summary="v2-only hybrid search across extracted memories and original chunks",
)
async def hybrid_search_memory_v2(
    req: HybridSearchRequest,
    request: Request,
    user: dict = Depends(require_api_key),
):
    start = time.perf_counter()
    pipeline = get_retrieval_pipeline()
    user_id = memory_v1._current_user_id(user, req.user_id)
    memory_top_k = req.memory_top_k or int(settings.hybrid_search_memory_top_k)
    original_top_k = req.original_top_k or int(settings.hybrid_search_original_top_k)

    try:
        memory_results: list[SourceRecord] = []
        if "profile" in req.domains:
            profile_results = await asyncio.to_thread(
                memory_v1._search_profile,
                pipeline,
                user_id,
            )
            memory_results.extend(profile_results)
        if "temporal" in req.domains:
            temporal_results = await asyncio.to_thread(
                memory_v1._search_temporal,
                pipeline,
                req.query,
                user_id,
                memory_top_k,
            )
            memory_results.extend(temporal_results)
        if "summary" in req.domains:
            memory_results.extend(
                await memory_v1._search_summary(
                    pipeline,
                    req.query,
                    user_id,
                    memory_top_k,
                )
            )

        original_chunks: list[SourceRecord] = []
        if req.include_original_chunks and settings.original_storage_enabled:
            original_chunks = await _search_original_chunks(
                req.query,
                user_id,
                original_top_k,
            )

        all_results = memory_results + original_chunks
        data = HybridSearchResponse(
            memory_results=memory_results,
            original_chunks=original_chunks,
            results=all_results,
            total=len(all_results),
            original_storage_enabled=bool(settings.original_storage_enabled),
        )
        return _wrap(request, data, elapsed_ms(start))
    except Exception as exc:
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
