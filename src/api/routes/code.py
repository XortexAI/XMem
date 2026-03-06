"""
/v1/code/* routes — code retrieval endpoints for XMem IDE mode.

Provides codebase querying, directory tree browsing, and repo listing
backed by the CodeRetrievalPipeline, MongoDB, and Neo4j.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.dependencies import (
    get_code_pipeline,
    require_api_key,
)
from src.api.schemas import (
    APIResponse,
    CodeQueryRequest,
    CodeQueryResponse,
    DirectoryNode,
    DirectoryTreeResponse,
    RepoListResponse,
    SourceRecord,
    StatusEnum,
)

logger = logging.getLogger("xmem.api.routes.code")

router = APIRouter(
    prefix="/v1/code",
    tags=["code"],
    dependencies=[Depends(require_api_key)],
)


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
# POST /v1/code/query
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/query",
    response_model=APIResponse,
    summary="Query a codebase using the code retrieval pipeline",
)
async def code_query(req: CodeQueryRequest, request: Request):
    start = time.perf_counter()
    pipeline = get_code_pipeline(org_id=req.org_id, repo=req.repo)

    try:
        result = await pipeline.run(
            query=req.query,
            user_id=req.user_id or "",
            repo=req.repo,
            top_k=req.top_k,
        )
        data = CodeQueryResponse(
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
        logger.exception("Code query failed for org=%s repo=%s", req.org_id, req.repo)
        return _error(request, str(exc), 500, elapsed)


# ═══════════════════════════════════════════════════════════════════════════
# POST /v1/code/query_stream
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/query_stream",
    summary="Query a codebase with streaming response",
)
async def code_query_stream(req: CodeQueryRequest, request: Request):
    pipeline = get_code_pipeline(org_id=req.org_id, repo=req.repo)
    return StreamingResponse(
        pipeline.run_stream(
            query=req.query,
            user_id=req.user_id or "",
            repo=req.repo,
            top_k=req.top_k,
        ),
        media_type="application/x-ndjson",
    )


# ═══════════════════════════════════════════════════════════════════════════
# GET /v1/code/directory-tree
# ═══════════════════════════════════════════════════════════════════════════

def _build_tree(file_paths: List[str], repo: str) -> DirectoryNode:
    """Build a hierarchical tree from flat file paths."""
    root = DirectoryNode(name=repo, type="directory", path="", children=[])
    nodes: Dict[str, DirectoryNode] = {"": root}

    for fp in sorted(file_paths):
        parts = fp.split("/")
        for depth in range(len(parts)):
            partial = "/".join(parts[: depth + 1])
            if partial in nodes:
                continue

            is_file = depth == len(parts) - 1
            node = DirectoryNode(
                name=parts[depth],
                type="file" if is_file else "directory",
                path=partial,
                children=[],
            )
            parent_path = "/".join(parts[:depth])
            parent = nodes.get(parent_path, root)
            parent.children.append(node)
            nodes[partial] = node

    return root


@router.get(
    "/directory-tree",
    response_model=APIResponse,
    summary="Get the directory tree for an indexed repository",
)
async def directory_tree(
    request: Request,
    org_id: str = Query(..., min_length=1),
    repo: str = Query(..., min_length=1),
):
    start = time.perf_counter()

    try:
        pipeline = get_code_pipeline(org_id=org_id, repo=repo)
        cursor = pipeline.code_store.files.find(
            {"org_id": org_id, "repo": repo},
            {"file_path": 1, "_id": 0},
        )
        file_paths = [doc["file_path"] for doc in cursor]

        tree = _build_tree(file_paths, repo)
        data = DirectoryTreeResponse(repo=repo, tree=tree)
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        return _wrap(request, data, elapsed)

    except Exception as exc:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("Directory tree failed for org=%s repo=%s", org_id, repo)
        return _error(request, str(exc), 500, elapsed)


# ═══════════════════════════════════════════════════════════════════════════
# GET /v1/code/repos
# ═══════════════════════════════════════════════════════════════════════════

@router.get(
    "/repos",
    response_model=APIResponse,
    summary="List indexed repositories for an organisation",
)
async def list_repos(
    request: Request,
    org_id: str = Query(..., min_length=1),
):
    start = time.perf_counter()

    try:
        pipeline = get_code_pipeline(org_id=org_id)
        repos = pipeline.code_store.files.distinct("repo", {"org_id": org_id})
        data = RepoListResponse(repos=sorted(repos))
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        return _wrap(request, data, elapsed)

    except Exception as exc:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("List repos failed for org=%s", org_id)
        return _error(request, str(exc), 500, elapsed)
