"""
FastAPI application factory for the XMem production API.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import (
    mark_failed,
    mark_ready,
    set_pipelines,
    set_startup_time,
)
from src.api.middleware import (
    MaxBodySizeMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from src.api.routes.code import router as code_router
from src.api.routes.health import router as health_router
from src.api.routes.memory import router as memory_router
from src.api.routes.scanner import router as scanner_router
from src.api.schemas import APIResponse, StatusEnum
from src.config import settings

logger = logging.getLogger("xmem.api")

for _name in [
    "httpx", "neo4j", "neo4j.notifications", "google_genai",
    "google_genai.models", "sentence_transformers",
    "huggingface_hub", "urllib3",
]:
    logging.getLogger(_name).setLevel(logging.WARNING)


def _init_pipelines_sync() -> tuple:
    from src.pipelines.ingest import IngestPipeline
    from src.pipelines.retrieval import RetrievalPipeline

    logger.info("[boot] Creating IngestPipeline ...")
    ingest = IngestPipeline()
    logger.info("[boot] Creating RetrievalPipeline ...")
    retrieval = RetrievalPipeline()
    return ingest, retrieval


async def _boot_pipelines() -> None:
    loop = asyncio.get_running_loop()
    try:
        ingest, retrieval = await loop.run_in_executor(None, _init_pipelines_sync)
        set_pipelines(ingest, retrieval)
        mark_ready()
        logger.info("[boot] All pipelines ready.")
    except Exception as exc:
        logger.error("[boot] Pipeline init failed: %s", exc)
        traceback.print_exc()
        mark_failed(str(exc))


def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        set_startup_time(time.time())
        boot_task = asyncio.create_task(_boot_pipelines())
        yield
        await boot_task
        from src.api.dependencies import _ingest_pipeline, _retrieval_pipeline
        if _ingest_pipeline:
            _ingest_pipeline.close()
        if _retrieval_pipeline:
            _retrieval_pipeline.close()
        logger.info("Pipelines shut down.")

    app = FastAPI(
        title="XMem API",
        description="Production API for XMem long-term memory — ingest, retrieve, and search.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        # allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms", "X-RateLimit-Remaining"],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(MaxBodySizeMiddleware)
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health_router)
    app.include_router(memory_router)
    app.include_router(code_router)
    app.include_router(scanner_router)

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled exception [%s]", request_id)
        body = APIResponse(
            status=StatusEnum.ERROR, request_id=request_id, error="Internal server error.",
        )
        return JSONResponse(content=body.model_dump(), status_code=500)

    return app
