from __future__ import annotations

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
            if payload.get("effort_level") == "high":
                result = await _execute(
                    "memory_run_pipeline_activity",
                    {**payload, **billing_activity},
                    timeout,
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
            results = []
            for index, item in enumerate(items):
                item_payload = dict(item)
                item_payload["user_id"] = (
                    item_payload.get("user_id") or payload["user_id"]
                )
                item_payload.update(billing_activity)
                item_result = await _execute(
                    "memory_run_pipeline_activity",
                    item_payload,
                    item_timeout,
                )
                results.append(item_result)
                await _execute(
                    "mark_job_progress_activity",
                    {
                        "job_id": job_id,
                        "progress": {
                            "step": "batch_ingest",
                            "completed": index + 1,
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
