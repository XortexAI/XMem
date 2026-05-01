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
from fastapi.responses import JSONResponse, Response

from src.api.dependencies import (
    mark_failed,
    mark_ready,
    set_pipelines,
    set_startup_time,
)
from src.api.middleware import (
    AnalyticsMiddleware,
    MaxBodySizeMiddleware,
    PrometheusMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from src.api.routes.api_keys import router as api_keys_router
from src.api.routes.auth import router as auth_router
from src.api.routes.code import router as code_router
from src.api.routes.enterprise import router as enterprise_router
from src.api.routes.health import router as health_router
from src.api.routes.memory import router as memory_router
from src.api.routes.memory import scrape_router as memory_scrape_router
from src.api.routes.memory_graph import router as memory_graph_router
from src.api.routes.scanner import router as scanner_router
from src.api.routes.telemetry import router as telemetry_router
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

        # Report to Sentry
        from src.config.monitoring import capture_exception
        capture_exception(exc)


def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application."""

    # ── Sentry (must be before app creation) ──────────────────────────
    from src.config.monitoring import init_sentry
    init_sentry()

    # ── Analytics collector ───────────────────────────────────────────
    if settings.enable_analytics:
        from src.config.analytics import analytics
        analytics.start()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        set_startup_time(time.time())

        # Capture the event loop for the WebSocket log handler
        from src.api.routes.admin import _set_event_loop
        _set_event_loop(asyncio.get_running_loop())

        boot_task = asyncio.create_task(_boot_pipelines())
        yield
        await boot_task
        from src.api.dependencies import _ingest_pipeline, _retrieval_pipeline
        if _ingest_pipeline:
            _ingest_pipeline.close()
        if _retrieval_pipeline:
            _retrieval_pipeline.close()
        # Shut down the warm Playwright browser pool
        try:
            from src.api.routes.memory import _browser_instance, _pw_instance
            if _browser_instance:
                _browser_instance.close()
            if _pw_instance:
                _pw_instance.stop()
        except Exception:
            pass
        # Stop analytics flush thread
        if settings.enable_analytics:
            from src.config.analytics import analytics as _analytics
            _analytics.stop()
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
        allow_origins=settings.cors_origins,
        allow_origin_regex=r".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms", "X-RateLimit-Remaining"],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(MaxBodySizeMiddleware)

    # Prometheus metrics middleware
    if settings.enable_prometheus:
        app.add_middleware(PrometheusMiddleware)

    # Analytics middleware (fire-and-forget)
    if settings.enable_analytics:
        app.add_middleware(AnalyticsMiddleware)

    app.add_middleware(RequestContextMiddleware)

    # ── Routes ────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(memory_scrape_router)
    app.include_router(memory_router)
    app.include_router(memory_graph_router)
    app.include_router(code_router)
    app.include_router(scanner_router)
    app.include_router(auth_router)
    app.include_router(api_keys_router)
    app.include_router(enterprise_router)
    app.include_router(telemetry_router)

    # Admin dashboard routes
    from src.api.routes.admin import router as admin_router
    app.include_router(admin_router)

    # Serve admin static assets (CSS, JS)
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    admin_static = Path(__file__).resolve().parent.parent.parent / "admin"
    if admin_static.exists():
        app.mount("/admin-assets", StaticFiles(directory=str(admin_static)), name="admin-static")

    # ── Prometheus /metrics endpoint ──────────────────────────────────
    if settings.enable_prometheus:
        @app.get("/metrics", include_in_schema=False)
        async def prometheus_metrics():
            from src.config.metrics import metrics_endpoint_content
            body, content_type = metrics_endpoint_content()
            return Response(content=body, media_type=content_type)

    # ── Sentry debug endpoint ─────────────────────────────────────────
    @app.get("/sentry-debug", include_in_schema=False)
    async def sentry_debug():
        """Intentionally raises an error to verify Sentry is capturing exceptions."""
        try:
            division_by_zero = 1 / 0
        except ZeroDivisionError as exc:
            from src.config.monitoring import capture_exception
            capture_exception(exc)
            return JSONResponse({
                "status": "ok",
                "message": "Sentry test error captured successfully. Check your Sentry dashboard.",
                "error_type": "ZeroDivisionError",
            })


    # ── Global exception handler ──────────────────────────────────────
    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled exception [%s]", request_id)

        # Report to Sentry
        from src.config.monitoring import capture_exception
        capture_exception(exc)

        body = APIResponse(
            status=StatusEnum.ERROR, request_id=request_id, error="Internal server error.",
        )
        return JSONResponse(content=body.model_dump(), status_code=500)

    return app
