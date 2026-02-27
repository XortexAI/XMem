"""
Health-check endpoint — unauthenticated, used by load balancers and probes.
"""

from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.dependencies import get_init_error, get_startup_time, is_ready
from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness probe",
)
async def health_check():
    error = get_init_error()
    ready = is_ready()
    uptime = round(time.time() - get_startup_time(), 1) if get_startup_time() else None

    resp = HealthResponse(
        status="ready" if ready else ("error" if error else "loading"),
        pipelines_ready=ready,
        uptime_seconds=uptime,
        error=error,
    )
    code = 200 if ready else 503
    return JSONResponse(content=resp.model_dump(), status_code=code)
