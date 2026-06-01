from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from fastapi.encoders import jsonable_encoder

from src.api.dependencies import get_ingest_pipeline
from src.api.routes import memory as memory_v1
from src.billing.context import use_billing_context
from src.billing.service import commit_job_billing, release_job_billing
from src.jobs.durable import get_default_job_store

try:  # pragma: no cover - no-op fallback keeps imports working without SDK.
    from temporalio import activity
except Exception:  # pragma: no cover

    class _ActivityFallback:
        def defn(self, fn=None, **_kwargs):
            if fn is None:
                return lambda wrapped: wrapped
            return fn

    activity = _ActivityFallback()


_scrape_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xmem-scrape")


def _domain_payload(result: Dict[str, Any], domain: str) -> Dict[str, Any] | None:
    value = memory_v1._build_domain_result(
        result.get(f"{domain}_judge"),
        result.get(f"{domain}_weaver"),
    )
    return value.model_dump() if value else None


@activity.defn
async def mark_job_running_activity(job_id: str) -> None:
    await asyncio.to_thread(get_default_job_store().mark_running, job_id)


@activity.defn
async def mark_job_progress_activity(payload: Dict[str, Any]) -> None:
    await asyncio.to_thread(
        get_default_job_store().update_progress,
        payload["job_id"],
        payload.get("progress") or {},
    )


@activity.defn
async def mark_job_succeeded_activity(payload: Dict[str, Any]) -> None:
    job = await asyncio.to_thread(get_default_job_store().get, payload["job_id"])
    result = payload.get("result") or {}
    if job:
        result = await asyncio.to_thread(commit_job_billing, job, result)
    await asyncio.to_thread(
        get_default_job_store().mark_succeeded,
        payload["job_id"],
        result,
    )


@activity.defn
async def mark_job_dead_letter_activity(payload: Dict[str, Any]) -> None:
    job = await asyncio.to_thread(get_default_job_store().get, payload["job_id"])
    if job:
        await asyncio.to_thread(release_job_billing, job, "dead_letter")
    await asyncio.to_thread(
        get_default_job_store().mark_dead_letter,
        payload["job_id"],
        payload.get("error") or "Workflow failed",
    )


@activity.defn
async def memory_classify_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = get_ingest_pipeline()
    state = {
        "user_query": payload.get("user_query", ""),
        "image_url": payload.get("image_url", ""),
    }
    with use_billing_context(
        job_id=payload.get("billing_job_id") or payload.get("job_id"),
        billing_account_id=payload.get("billing_account_id"),
        user_id=payload.get("user_id", ""),
    ):
        result = await pipeline._node_classify(state)
    classification = result.get("classification_result")
    return {
        "model": memory_v1._model_name(pipeline.model),
        "classification": jsonable_encoder(
            getattr(classification, "classifications", []) or []
        ),
    }


@activity.defn
async def memory_domain_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = get_ingest_pipeline()
    domain = payload["domain"]
    user_id = payload["user_id"]

    with use_billing_context(
        job_id=payload.get("billing_job_id") or payload.get("job_id"),
        billing_account_id=payload.get("billing_account_id"),
        user_id=user_id,
    ):
        if domain == "profile":
            result = await pipeline._node_extract_profile(
                {
                    "profile_queries": payload.get("queries", []),
                    "user_id": user_id,
                }
            )
            return {"domain": domain, "result": _domain_payload(result, "profile")}

        if domain == "temporal":
            result = await pipeline._node_extract_temporal(
                {
                    "temporal_queries": payload.get("queries", []),
                    "session_datetime": payload.get("session_datetime", ""),
                    "user_id": user_id,
                }
            )
            return {"domain": domain, "result": _domain_payload(result, "temporal")}

        if domain == "summary":
            result = await pipeline._node_extract_summary(
                {
                    "user_query": payload.get("user_query", ""),
                    "agent_response": payload.get("agent_response", ""),
                    "user_id": user_id,
                }
            )
            return {"domain": domain, "result": _domain_payload(result, "summary")}

        if domain == "image":
            result = await pipeline._node_extract_image(
                {
                    "classifier_output": payload.get("classifier_output", ""),
                    "image_url": payload.get("image_url", ""),
                    "user_id": user_id,
                }
            )
            return {"domain": domain, "result": _domain_payload(result, "image")}

        if domain == "code":
            result = await pipeline._node_extract_code(
                {
                    "code_queries": payload.get("queries", []),
                    "user_id": user_id,
                }
            )
            return {"domain": domain, "stored": bool(result)}

        if domain == "snippet":
            result = await pipeline._node_extract_snippet(
                {
                    "code_queries": payload.get("queries", []),
                    "user_id": user_id,
                }
            )
            return {"domain": domain, "stored": bool(result)}

    raise ValueError(f"Unsupported memory domain activity: {domain}")


@activity.defn
async def memory_run_pipeline_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    with use_billing_context(
        job_id=payload.get("billing_job_id") or payload.get("job_id"),
        billing_account_id=payload.get("billing_account_id"),
        user_id=payload.get("user_id", ""),
    ):
        return await memory_v1._run_ingest_payload(payload, payload["user_id"])


@activity.defn
async def memory_scrape_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    def _run_scrape() -> Dict[str, Any]:
        html, final_url = memory_v1._render_chat_share_sync(payload["url"])
        provider, extraction_method, pairs = memory_v1._extract_chat_pairs(
            final_url or payload["url"],
            html,
            payload["url"],
        )
        result = {
            "provider": provider,
            "url": payload["url"],
            "final_url": final_url,
            "pairs": pairs,
            "pair_count": len(pairs),
            "html_length": len(html),
            "extraction_method": extraction_method,
        }
        if not pairs:
            raise ValueError(memory_v1._chat_share_error_message(result))
        return memory_v1.ScrapeResponse(pairs=pairs).model_dump()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_scrape_executor, _run_scrape)


@activity.defn
async def scanner_scan_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    from src.api.routes import scanner as scanner_v1
    from src.api.routes.v2.secrets import resolve_scanner_pat

    try:
        pat = await asyncio.to_thread(
            resolve_scanner_pat, payload.get("github_credential_ref") or ""
        )
    except Exception as exc:
        error = str(exc) or exc.__class__.__name__
        await asyncio.to_thread(_mark_scanner_scan_failed, scanner_v1, payload, error)
        raise

    await scanner_v1._run_scan_job(
        payload["scanner_job_id"],
        payload["username"],
        payload["org"],
        payload["repo"],
        payload["github_url"],
        payload.get("branch") or "main",
        pat,
        bool(payload.get("force_full", False)),
    )
    job = scanner_v1._get_code_store().get_scanner_job(payload["scanner_job_id"]) or {}
    return {
        "scanner_job_id": payload["scanner_job_id"],
        "phase1_status": job.get("phase1_status"),
        "phase2_status": job.get("phase2_status"),
    }


def _mark_scanner_scan_failed(scanner_v1, payload: Dict[str, Any], error: str) -> None:
    store = scanner_v1._get_code_store()
    scanner_job_id = payload["scanner_job_id"]
    existing = store.get_scanner_job(scanner_job_id) or {}
    store.upsert_scanner_job(
        job_id=scanner_job_id,
        username=payload["username"],
        org=payload["org"],
        repo=payload["repo"],
        branch=payload.get("branch") or existing.get("branch") or "main",
        url=payload.get("github_url") or existing.get("url") or "",
        phase1_status="failed",
        phase2_status=existing.get("phase2_status", "pending"),
        started_at=float(existing.get("started_at") or time.time()),
        error=error,
        durable_job_id=payload.get("durable_job_id") or existing.get("durable_job_id"),
        retry_count=int(existing.get("retry_count") or 0),
        timeout_seconds=existing.get("timeout_seconds"),
    )


@activity.defn
async def scanner_phase2_activity(payload: Dict[str, Any]) -> Dict[str, Any]:
    from src.api.routes import scanner as scanner_v1

    await scanner_v1._run_phase2_job(
        payload["scanner_job_id"],
        payload["username"],
        payload["org"],
        payload["repo"],
        payload["github_url"],
        payload.get("branch") or "main",
    )
    job = scanner_v1._get_code_store().get_scanner_job(payload["scanner_job_id"]) or {}
    return {
        "scanner_job_id": payload["scanner_job_id"],
        "phase1_status": job.get("phase1_status"),
        "phase2_status": job.get("phase2_status"),
    }


ALL_ACTIVITIES = [
    mark_job_running_activity,
    mark_job_progress_activity,
    mark_job_succeeded_activity,
    mark_job_dead_letter_activity,
    memory_classify_activity,
    memory_domain_activity,
    memory_run_pipeline_activity,
    memory_scrape_activity,
    scanner_scan_activity,
    scanner_phase2_activity,
]
