"""Background job status routes."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.dependencies import enforce_rate_limit, require_api_key
from src.api.schemas import APIResponse, JobStatusResponse, StatusEnum
from src.jobs import get_job_store, serialize_job

router = APIRouter(
    prefix="/v1/jobs",
    tags=["jobs"],
    dependencies=[Depends(enforce_rate_limit)],
)


@router.get("/{job_id}", summary="Get durable background job status")
async def get_job_status(
    job_id: str,
    request: Request,
    user: dict = Depends(require_api_key),
):
    start = time.perf_counter()
    owner_id = user.get("username") or user.get("name") or user["id"]
    try:
        store = get_job_store()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    job = store.get(job_id, owner_id=owner_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    data = JobStatusResponse(**serialize_job(job))
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    body = APIResponse(
        status=StatusEnum.OK,
        request_id=getattr(request.state, "request_id", None),
        data=data.model_dump(mode="json"),
        elapsed_ms=elapsed,
    )
    return JSONResponse(content=body.model_dump(mode="json"))
