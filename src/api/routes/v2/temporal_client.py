from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from src.config import settings
from src.jobs.durable import get_default_job_store, new_attempt_id

logger = logging.getLogger("xmem.api.routes.v2.temporal_client")

_temporal_client = None
_temporal_client_lock = asyncio.Lock()


WORKFLOW_BY_JOB_TYPE = {
    "memory_ingest": "MemoryIngestWorkflow",
    "memory_batch_ingest": "MemoryBatchIngestWorkflow",
    "memory_scrape": "MemoryScrapeWorkflow",
    "scanner_scan": "ScannerScanWorkflow",
    "scanner_phase2": "ScannerPhase2Workflow",
    "scanner_scan_resume": "ScannerScanWorkflow",
    "scanner_phase2_resume": "ScannerPhase2Workflow",
}


class TemporalUnavailable(RuntimeError):
    """Raised when Temporal is required but the SDK/server is unavailable."""


async def get_temporal_client():
    global _temporal_client

    try:
        from temporalio.client import Client
    except Exception as exc:  # pragma: no cover - depends on optional SDK import
        raise TemporalUnavailable(
            "temporalio is not installed. Install project dependencies first."
        ) from exc

    if _temporal_client is None:
        async with _temporal_client_lock:
            if _temporal_client is None:
                _temporal_client = await Client.connect(
                    settings.temporal_address,
                    namespace=settings.temporal_namespace,
                )
    return _temporal_client


def workflow_name_for_job(job_type: str) -> str:
    try:
        return WORKFLOW_BY_JOB_TYPE[job_type]
    except KeyError as exc:
        raise ValueError(f"No v2 Temporal workflow is registered for {job_type!r}.") from exc


async def start_job_workflow(
    job: Dict[str, Any],
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Start the Temporal workflow that owns this durable job."""

    workflow_name = workflow_name_for_job(str(job["job_type"]))
    workflow_id = job.get("workflow_id") or f"{job['job_id']}:{new_attempt_id()}"
    client = await get_temporal_client()
    start_options: Dict[str, Any] = {
        "id": workflow_id,
        "task_queue": settings.temporal_task_queue,
    }
    try:
        from temporalio.common import WorkflowIDConflictPolicy

        start_options["id_conflict_policy"] = WorkflowIDConflictPolicy.FAIL
    except Exception:  # pragma: no cover - older SDKs/fallback imports.
        logger.debug("Temporal WorkflowIDConflictPolicy is unavailable; using SDK default.")

    handle = await client.start_workflow(
        workflow_name,
        {
            "job_id": job["job_id"],
            "job_type": job["job_type"],
            "payload": payload if payload is not None else job.get("payload") or {},
        },
        **start_options,
    )

    run_id: Optional[str] = getattr(handle, "first_execution_run_id", None)
    await _record_temporal_ids(job["job_id"], workflow_id, run_id)
    return {
        "workflow_id": workflow_id,
        "run_id": run_id,
    }


async def cancel_job_workflow(job: Dict[str, Any]) -> None:
    workflow_id = job.get("workflow_id")
    if not workflow_id:
        return
    client = await get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    try:
        await handle.cancel()
    except Exception as exc:
        message = str(exc).lower()
        if "not found" in message or "already completed" in message:
            logger.info(
                "Temporal workflow %s was already absent or complete during cancellation: %s",
                workflow_id,
                exc,
            )
            return
        raise


async def _record_temporal_ids(
    job_id: str,
    workflow_id: str,
    run_id: Optional[str],
) -> None:
    await asyncio.to_thread(
        get_default_job_store().record_workflow,
        job_id,
        workflow_id,
        run_id,
    )
