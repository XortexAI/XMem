from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Dict, List

try:  # pragma: no cover - fallback makes app imports independent of SDK install.
    from temporalio import workflow
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import CancelledError
except Exception:  # pragma: no cover

    class CancelledError(BaseException):  # type: ignore[no-redef]
        pass

    class RetryPolicy:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _WorkflowFallback:
        def defn(self, cls=None, **_kwargs):
            if cls is None:
                return lambda wrapped: wrapped
            return cls

        def run(self, fn=None, **_kwargs):
            if fn is None:
                return lambda wrapped: wrapped
            return fn

        async def execute_activity(self, *args, **kwargs):
            raise RuntimeError("temporalio is not installed")

    workflow = _WorkflowFallback()


ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
)


def _activity_timeout(seconds: float) -> timedelta:
    return timedelta(seconds=max(float(seconds or 1.0), 1.0))


async def _execute(name: str, arg: Any, timeout_seconds: float) -> Any:
    return await workflow.execute_activity(
        name,
        arg,
        start_to_close_timeout=_activity_timeout(timeout_seconds),
        retry_policy=ACTIVITY_RETRY,
    )


def _original_enabled(payload: Dict[str, Any]) -> bool:
    return bool(payload.get("original_storage_enabled"))


def _original_timeout(payload: Dict[str, Any]) -> float:
    return float(payload.get("original_storage_timeout_seconds") or 180.0)


def _start_original_task(job_id: str, payload: Dict[str, Any]):
    if not _original_enabled(payload):
        return None
    return asyncio.create_task(
        _execute(
            "memory_store_original_activity",
            {**payload, "job_id": job_id},
            _original_timeout(payload),
        )
    )


async def _await_original_task(task, payload: Dict[str, Any]) -> Dict[str, Any]:
    if task is None:
        return {"status": "disabled", "indexed_chunks": 0}
    try:
        return await task
    except Exception as exc:
        if bool(payload.get("original_storage_fail_closed")):
            raise
        return {
            "status": "failed",
            "error": str(exc) or exc.__class__.__name__,
            "indexed_chunks": 0,
        }


async def _mark_dead(job_id: str, exc: BaseException) -> Dict[str, Any]:
    error = str(exc) or exc.__class__.__name__
    await _execute(
        "mark_job_dead_letter_activity",
        {"job_id": job_id, "error": error},
        30,
    )
    return {"status": "dead_letter", "error": error}


def _routes(classifications: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    routes: Dict[str, List[str]] = {
        "profile": [],
        "temporal": [],
        "image": [],
        "code": [],
    }
    for item in classifications:
        source = item.get("source")
        query = item.get("query") or ""
        if source == "profile":
            routes["profile"].append(query)
        elif source == "event":
            routes["temporal"].append(query)
        elif source == "image":
            routes["image"].append(query)
        elif source == "code":
            routes["code"].append(query)
    return routes


def _is_trivial(payload: Dict[str, Any], routes: Dict[str, List[str]]) -> bool:
    words = str(payload.get("user_query") or "").strip().split()
    return len(words) < 4 and not any(routes.values())


@workflow.defn(name="MemoryIngestWorkflow")
class MemoryIngestWorkflow:
    @workflow.run
    async def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        job_id = input["job_id"]
        payload = input["payload"]
        billing_activity = {
            "billing_job_id": job_id,
            "billing_account_id": payload.get("billing_account_id"),
        }
        timeout = float(payload.get("timeout_seconds") or 120.0)
        try:
            await _execute("mark_job_running_activity", job_id, 30)
            original_task = _start_original_task(job_id, payload)
            if payload.get("effort_level") == "high":
                result = await _execute(
                    "memory_run_pipeline_activity",
                    {**payload, **billing_activity},
                    timeout,
                )
                result["original_storage"] = await _await_original_task(
                    original_task,
                    payload,
                )
                await _execute(
                    "mark_job_succeeded_activity",
                    {"job_id": job_id, "result": result},
                    30,
                )
                return result

            classified = await _execute(
                "memory_classify_activity", {**payload, **billing_activity}, timeout
            )
            classifications = classified.get("classification") or []
            routes = _routes(classifications)
            result: Dict[str, Any] = {
                "model": classified.get("model", ""),
                "classification": classifications,
                "profile": None,
                "temporal": None,
                "summary": None,
                "image": None,
                "code": None,
            }
            await _execute(
                "mark_job_progress_activity",
                {
                    "job_id": job_id,
                    "progress": {
                        "step": "classified",
                        "classification_count": len(classifications),
                    },
                },
                30,
            )

            if not _is_trivial(payload, routes):
                summary = await _execute(
                    "memory_domain_activity",
                    {
                        "domain": "summary",
                        "user_id": payload["user_id"],
                        **billing_activity,
                        "user_query": payload.get("user_query", ""),
                        "agent_response": payload.get("agent_response", ""),
                    },
                    timeout,
                )
                result["summary"] = summary.get("result")

            if routes["profile"]:
                profile = await _execute(
                    "memory_domain_activity",
                    {
                        "domain": "profile",
                        "user_id": payload["user_id"],
                        "queries": routes["profile"],
                        **billing_activity,
                    },
                    timeout,
                )
                result["profile"] = profile.get("result")

            if routes["temporal"]:
                temporal = await _execute(
                    "memory_domain_activity",
                    {
                        "domain": "temporal",
                        "user_id": payload["user_id"],
                        **billing_activity,
                        "queries": routes["temporal"],
                        "session_datetime": payload.get("session_datetime", ""),
                    },
                    timeout,
                )
                result["temporal"] = temporal.get("result")

            if payload.get("image_url"):
                image = await _execute(
                    "memory_domain_activity",
                    {
                        "domain": "image",
                        "user_id": payload["user_id"],
                        **billing_activity,
                        "classifier_output": " ".join(routes["image"])
                        or "Analyze this image for memory-relevant details.",
                        "image_url": payload.get("image_url", ""),
                    },
                    timeout,
                )
                result["image"] = image.get("result")

            if routes["code"]:
                code = await _execute(
                    "memory_domain_activity",
                    {
                        "domain": "snippet",
                        "user_id": payload["user_id"],
                        "queries": routes["code"],
                        **billing_activity,
                    },
                    timeout,
                )
                result["code"] = code

            result["original_storage"] = await _await_original_task(
                original_task,
                payload,
            )
            await _execute(
                "mark_job_succeeded_activity", {"job_id": job_id, "result": result}, 30
            )
            return result
        except CancelledError:
            raise
        except Exception as exc:
            return await _mark_dead(job_id, exc)


@workflow.defn(name="MemoryBatchIngestWorkflow")
class MemoryBatchIngestWorkflow:
    @workflow.run
    async def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        job_id = input["job_id"]
        payload = input["payload"]
        billing_activity = {
            "billing_job_id": job_id,
            "billing_account_id": payload.get("billing_account_id"),
        }
        try:
            await _execute("mark_job_running_activity", job_id, 30)
            items = list(payload.get("items") or [])
            total_timeout = float(payload.get("timeout_seconds") or 3600.0)
            item_timeout = max(total_timeout / max(len(items), 1), 1.0)
            concurrency = max(
                int(payload.get("original_batch_item_concurrency") or 1),
                1,
            )
            results: List[Any] = [None] * len(items)
            completed = 0

            async def _run_item(index: int, item: Dict[str, Any]):
                item_payload = dict(item)
                item_payload["user_id"] = (
                    item_payload.get("user_id") or payload["user_id"]
                )
                for key in (
                    "original_storage_enabled",
                    "original_storage_fail_closed",
                    "original_storage_timeout_seconds",
                    "original_config",
                ):
                    if key in payload and key not in item_payload:
                        item_payload[key] = payload[key]

                original_task = _start_original_task(job_id, item_payload)
                try:
                    item_result = await _execute(
                        "memory_run_pipeline_activity",
                        {**item_payload, **billing_activity},
                        item_timeout,
                    )
                    item_result["original_storage"] = await _await_original_task(
                        original_task,
                        item_payload,
                    )
                    original_task = None
                    return index, item_result
                finally:
                    if original_task and not original_task.done():
                        original_task.cancel()
                        try:
                            await original_task
                        except BaseException:
                            pass

            for start in range(0, len(items), concurrency):
                window = [
                    asyncio.create_task(_run_item(index, item))
                    for index, item in enumerate(
                        items[start:start + concurrency],
                        start=start,
                    )
                ]
                for index, item_result in await asyncio.gather(*window):
                    results[index] = item_result
                    completed += 1
                    await _execute(
                        "mark_job_progress_activity",
                        {
                            "job_id": job_id,
                            "progress": {
                                "step": "batch_ingest",
                                "completed": completed,
                                "total": len(items),
                            },
                        },
                        30,
                    )
            result = {"results": results}
            await _execute(
                "mark_job_succeeded_activity", {"job_id": job_id, "result": result}, 30
            )
            return result
        except CancelledError:
            raise
        except Exception as exc:
            return await _mark_dead(job_id, exc)


@workflow.defn(name="MemoryScrapeWorkflow")
class MemoryScrapeWorkflow:
    @workflow.run
    async def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        job_id = input["job_id"]
        try:
            await _execute("mark_job_running_activity", job_id, 30)
            result = await _execute("memory_scrape_activity", input["payload"], 60)
            await _execute(
                "mark_job_succeeded_activity", {"job_id": job_id, "result": result}, 30
            )
            return result
        except CancelledError:
            raise
        except Exception as exc:
            return await _mark_dead(job_id, exc)


@workflow.defn(name="ScannerScanWorkflow")
class ScannerScanWorkflow:
    @workflow.run
    async def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        job_id = input["job_id"]
        try:
            await _execute("mark_job_running_activity", job_id, 30)
            activity_payload = dict(input["payload"])
            activity_payload["durable_job_id"] = job_id
            result = await _execute("scanner_scan_activity", activity_payload, 1800)
            await _execute(
                "mark_job_succeeded_activity", {"job_id": job_id, "result": result}, 30
            )
            return result
        except CancelledError:
            raise
        except Exception as exc:
            return await _mark_dead(job_id, exc)


@workflow.defn(name="ScannerPhase2Workflow")
class ScannerPhase2Workflow:
    @workflow.run
    async def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        job_id = input["job_id"]
        try:
            await _execute("mark_job_running_activity", job_id, 30)
            result = await _execute("scanner_phase2_activity", input["payload"], 1800)
            await _execute(
                "mark_job_succeeded_activity", {"job_id": job_id, "result": result}, 30
            )
            return result
        except CancelledError:
            raise
        except Exception as exc:
            return await _mark_dead(job_id, exc)


ALL_WORKFLOWS = [
    MemoryIngestWorkflow,
    MemoryBatchIngestWorkflow,
    MemoryScrapeWorkflow,
    ScannerScanWorkflow,
    ScannerPhase2Workflow,
]
