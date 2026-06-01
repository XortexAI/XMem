from __future__ import annotations

import time
from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from src.api.routes.memory import _error, _wrap
from src.jobs.durable import QUEUED, serialize_job


def job_status_data(job: Dict[str, Any]) -> Dict[str, Any]:
    public = serialize_job(job) or {}
    return {
        "job_id": public.get("job_id"),
        "job_type": public.get("job_type"),
        "status": public.get("status"),
        "retry_count": public.get("retry_count", 0),
        "attempt_count": public.get("attempt_count", 0),
        "max_attempts": public.get("max_attempts", 0),
        "timeout_seconds": public.get("timeout_seconds"),
        "workflow_id": public.get("workflow_id"),
        "run_id": public.get("run_id"),
        "progress": public.get("progress"),
        "error": public.get("error"),
        "error_state": public.get("error_state"),
        "result": public.get("result"),
        "created_at": public.get("created_at"),
        "updated_at": public.get("updated_at"),
        "started_at": public.get("started_at"),
        "completed_at": public.get("completed_at"),
        "dead_lettered_at": public.get("dead_lettered_at"),
        "cancelled_at": public.get("cancelled_at"),
    }


def accepted_job(
    request: Request,
    job: Dict[str, Any],
    created: bool,
    status_url: str,
    elapsed_ms: float,
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    data = {
        "job_id": job["job_id"],
        "status": job.get("status", QUEUED),
        "created": created,
        "status_url": status_url,
    }
    if extra:
        data.update(extra)
    return _wrap(request, data, elapsed_ms)


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


async def read_user_job(job_id: str, user_id: str) -> Dict[str, Any] | None:
    import asyncio

    from src.jobs.durable import get_default_job_store

    job = await asyncio.to_thread(get_default_job_store().get, job_id)
    if not job or job.get("user_id") != user_id:
        return None
    return job


__all__ = [
    "accepted_job",
    "elapsed_ms",
    "job_status_data",
    "read_user_job",
    "_error",
    "_wrap",
]
