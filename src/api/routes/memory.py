"""
/v1/memory/* routes — production endpoints for XMem memory operations.

All routes require a valid Bearer API key and respect the per-key rate limit.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.api.dependencies import (
    enforce_rate_limit,
    get_ingest_pipeline,
    get_retrieval_pipeline,
    require_ready,
)
from src.api.schemas import (
    APIResponse,
    DomainResult,
    IngestRequest,
    IngestResponse,
    OperationDetail,
    RetrieveRequest,
    RetrieveResponse,
    SearchRequest,
    SearchResponse,
    SourceRecord,
    StatusEnum,
    WeaverSummary,
)
from src.pipelines.retrieval import RetrievalPipeline

logger = logging.getLogger("xmem.api.routes.memory")

router = APIRouter(
    prefix="/v1/memory",
    tags=["memory"],
    dependencies=[Depends(require_ready), Depends(enforce_rate_limit)],
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _model_name(model: Any) -> str:
    return getattr(model, "model", getattr(model, "model_name", "unknown"))


def _build_domain_result(judge: Any, weaver: Any) -> DomainResult | None:
    if not judge or not getattr(judge, "operations", None):
        return None
    ops = [
        OperationDetail(type=op.type.value, content=op.content, reason=op.reason)
        for op in judge.operations
    ]
    ws = None
    if weaver:
        ws = WeaverSummary(
            succeeded=weaver.succeeded, skipped=weaver.skipped, failed=weaver.failed,
        )
    return DomainResult(confidence=judge.confidence, operations=ops, weaver=ws)


def _wrap(request: Request, data: Any, elapsed_ms: float) -> JSONResponse:
    body = APIResponse(
        status=StatusEnum.OK,
        request_id=getattr(request.state, "request_id", None),
        data=data.model_dump() if hasattr(data, "model_dump") else data,
        elapsed_ms=elapsed_ms,
    )
    resp = JSONResponse(content=body.model_dump())
    remaining = getattr(request.state, "rate_limit_remaining", None)
    if remaining is not None:
        resp.headers["X-RateLimit-Remaining"] = str(remaining)
    return resp


def _error(request: Request, detail: str, code: int, elapsed_ms: float = 0) -> JSONResponse:
    body = APIResponse(
        status=StatusEnum.ERROR,
        request_id=getattr(request.state, "request_id", None),
        error=detail,
        elapsed_ms=elapsed_ms,
    )
    return JSONResponse(content=body.model_dump(), status_code=code)


# ═══════════════════════════════════════════════════════════════════════════
# POST /v1/memory/ingest
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/ingest",
    response_model=APIResponse,
    summary="Ingest a conversation turn into long-term memory",
)
async def ingest_memory(req: IngestRequest, request: Request):
    start = time.perf_counter()
    pipeline = get_ingest_pipeline()

    try:
        result = await pipeline.run(
            user_query=req.user_query,
            agent_response=req.agent_response or "Acknowledged.",
            user_id=req.user_id,
            session_datetime=req.session_datetime,
            image_url=req.image_url,
        )
        data = IngestResponse(
            model=_model_name(pipeline.model),
            classification=_safe_classifications(result),
            profile=_build_domain_result(result.get("profile_judge"), result.get("profile_weaver")),
            temporal=_build_domain_result(result.get("temporal_judge"), result.get("temporal_weaver")),
            summary=_build_domain_result(result.get("summary_judge"), result.get("summary_weaver")),
            image=_build_domain_result(result.get("image_judge"), result.get("image_weaver")),
        )
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        return _wrap(request, data, elapsed)

    except Exception as exc:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("Ingest failed for user=%s", req.user_id)
        return _error(request, str(exc), 500, elapsed)


def _safe_classifications(result: Dict[str, Any]) -> list:
    cr = result.get("classification_result")
    if cr and getattr(cr, "classifications", None):
        return cr.classifications
    return []


# ═══════════════════════════════════════════════════════════════════════════
# POST /v1/memory/retrieve
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/retrieve",
    response_model=APIResponse,
    summary="Retrieve an LLM-generated answer backed by stored memories",
)
async def retrieve_memory(req: RetrieveRequest, request: Request):
    start = time.perf_counter()
    pipeline = get_retrieval_pipeline()

    try:
        result = await pipeline.run(query=req.query, user_id=req.user_id, top_k=req.top_k)
        data = RetrieveResponse(
            model=_model_name(pipeline.model),
            answer=result.answer,
            sources=[
                SourceRecord(
                    domain=s.domain, content=s.content,
                    score=round(s.score, 3), metadata=s.metadata,
                )
                for s in result.sources
            ],
            confidence=result.confidence,
        )
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        return _wrap(request, data, elapsed)

    except Exception as exc:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("Retrieve failed for user=%s", req.user_id)
        return _error(request, str(exc), 500, elapsed)


# ═══════════════════════════════════════════════════════════════════════════
# POST /v1/memory/search
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/search",
    response_model=APIResponse,
    summary="Raw semantic search across memory domains (no LLM answer)",
)
async def search_memory(req: SearchRequest, request: Request):
    start = time.perf_counter()
    pipeline = get_retrieval_pipeline()

    try:
        all_results: List[SourceRecord] = []

        if "profile" in req.domains:
            all_results.extend(_search_profile(pipeline, req.user_id))
        if "temporal" in req.domains:
            all_results.extend(_search_temporal(pipeline, req.query, req.user_id, req.top_k))
        if "summary" in req.domains:
            all_results.extend(await _search_summary(pipeline, req.query, req.user_id, req.top_k))

        data = SearchResponse(results=all_results, total=len(all_results))
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        return _wrap(request, data, elapsed)

    except Exception as exc:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("Search failed for user=%s", req.user_id)
        return _error(request, str(exc), 500, elapsed)


def _search_profile(pipeline: RetrievalPipeline, user_id: str) -> List[SourceRecord]:
    try:
        raw = pipeline.vector_store.search_by_metadata(
            filters={"user_id": user_id, "domain": "profile"}, top_k=100,
        )
        return [SourceRecord(domain="profile", content=r.content, score=r.score, metadata=r.metadata) for r in raw]
    except Exception as exc:
        logger.warning("Profile search error: %s", exc)
        return []


def _search_temporal(pipeline: RetrievalPipeline, query: str, user_id: str, top_k: int) -> List[SourceRecord]:
    try:
        events = pipeline.neo4j.search_events_by_embedding(
            user_id=user_id, query_text=query, top_k=top_k, similarity_threshold=0.15,
        )
        results = []
        for ev in events:
            parts = []
            if ev.get("date"):
                d = ev["date"]
                if ev.get("year"):
                    d += f", {ev['year']}"
                parts.append(f"Date: {d}")
            if ev.get("event_name"):
                parts.append(f"Event: {ev['event_name']}")
            if ev.get("desc"):
                parts.append(f"Description: {ev['desc']}")
            if ev.get("time"):
                parts.append(f"Time: {ev['time']}")
            results.append(SourceRecord(
                domain="temporal", content=" | ".join(parts),
                score=ev.get("similarity_score", 0.0), metadata=ev,
            ))
        return results
    except Exception as exc:
        logger.warning("Temporal search error: %s", exc)
        return []


async def _search_summary(pipeline: RetrievalPipeline, query: str, user_id: str, top_k: int) -> List[SourceRecord]:
    try:
        raw = await pipeline.vector_store.search_by_text(
            query_text=query, top_k=top_k,
            filters={"user_id": user_id, "domain": "summary"},
        )
        return [
            SourceRecord(domain="summary", content=r.content, score=r.score, metadata={"id": r.id, **r.metadata})
            for r in raw
        ]
    except Exception as exc:
        logger.warning("Summary search error: %s", exc)
        return []
