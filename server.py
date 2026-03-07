"""
Xmem Test Frontend — FastAPI server.

Exposes the ingest and retrieval pipelines as API endpoints.
Captures pipeline logs so the frontend can show each step.

Usage:
    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# ── Project root setup ────────────────────────────────────────────
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from the project root BEFORE importing src modules
# (src.config.Settings requires PINECONE_API_KEY, NEO4J_PASSWORD, etc.)
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── Suppress noisy loggers globally ───────────────────────────────
for name in [
    "httpx", "neo4j", "neo4j.notifications", "google_genai",
    "google_genai.models", "sentence_transformers",
    "huggingface_hub", "urllib3",
]:
    logging.getLogger(name).setLevel(logging.WARNING)

from src.pipelines.ingest import IngestPipeline
from src.pipelines.retrieval import RetrievalPipeline
from src.config import settings


# ═══════════════════════════════════════════════════════════════════
# Log capture — collects pipeline log messages during a run
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# Rate Limiting Middleware
# ═══════════════════════════════════════════════════════════════════

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.rate_limit_records: Dict[str, List[float]] = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        if client_ip not in self.rate_limit_records:
            self.rate_limit_records[client_ip] = []

        # Filter timestamps older than 60 seconds
        self.rate_limit_records[client_ip] = [
            t for t in self.rate_limit_records[client_ip]
            if current_time - t < 60
        ]

        if len(self.rate_limit_records[client_ip]) >= settings.rate_limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "detail": f"Rate limit exceeded: {settings.rate_limit} per minute"
                },
                headers={"Retry-After": "60"}
            )

        self.rate_limit_records[client_ip].append(current_time)
        return await call_next(request)

class StepCapture(logging.Handler):
    """Captures log records into a list of step dicts."""

    def __init__(self) -> None:
        super().__init__()
        self.steps: List[Dict[str, Any]] = []
        self._start = time.time()

    def emit(self, record: logging.LogRecord) -> None:
        self.steps.append({
            "t": round(time.time() - self._start, 3),
            "level": record.levelname,
            "source": record.name.replace("xmem.", ""),
            "msg": record.getMessage(),
        })


# ═══════════════════════════════════════════════════════════════════
# App setup
# ═══════════════════════════════════════════════════════════════════

ingest_pipeline: IngestPipeline | None = None
retrieval_pipeline: RetrievalPipeline | None = None
_pipelines_ready = asyncio.Event()
_init_error: str | None = None


def _init_pipelines_sync() -> None:
    """Heavy synchronous init — runs in a thread so it doesn't block the event loop."""
    global ingest_pipeline, retrieval_pipeline, _init_error
    try:
        print("[init] Creating IngestPipeline...", flush=True)
        ingest_pipeline = IngestPipeline()
        print("[init] IngestPipeline created.", flush=True)

        print("[init] Creating RetrievalPipeline...", flush=True)
        retrieval_pipeline = RetrievalPipeline()
        print("[init] RetrievalPipeline created.", flush=True)
    except Exception as exc:
        _init_error = str(exc)
        print(f"[init] FAILED: {exc}", flush=True)
        import traceback
        traceback.print_exc()


async def _init_pipelines_bg() -> None:
    """Kick off pipeline init in a thread and signal readiness when done."""
    loop = asyncio.get_running_loop()
    print("[init] Starting background pipeline init...", flush=True)
    await loop.run_in_executor(None, _init_pipelines_sync)
    _pipelines_ready.set()
    if _init_error:
        print(f"[init] Pipelines failed: {_init_error}", flush=True)
    else:
        print("[init] Pipelines ready!", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_task = asyncio.create_task(_init_pipelines_bg())
    yield
    await init_task
    if ingest_pipeline:
        ingest_pipeline.close()
    if retrieval_pipeline:
        retrieval_pipeline.close()
    print("Pipelines closed")


app = FastAPI(title="Xmem Test Frontend", lifespan=lifespan)

app.add_middleware(RateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════
# Request / Response models
# ═══════════════════════════════════════════════════════════════════

class IngestRequest(BaseModel):
    user_query: str
    agent_response: str = ""
    user_id: str = "demo_user"


class RetrieveRequest(BaseModel):
    query: str
    user_id: str = "demo_user"


# ═══════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    if _init_error:
        return JSONResponse({"status": "error", "data": {"status": "error", "pipelines_ready": False, "version": "", "error": _init_error}}, status_code=503)
    if _pipelines_ready.is_set():
        return JSONResponse({"status": "ok", "data": {"status": "ready", "pipelines_ready": True, "version": "1.0.0"}})
    return JSONResponse({"status": "ok", "data": {"status": "loading", "pipelines_ready": False, "version": "1.0.0"}}, status_code=503)


@app.get("/")
async def serve_ui():
    return FileResponse(Path(__file__).parent / "frontend" / "index.html")


@app.post("/api/ingest")
async def api_ingest(req: IngestRequest):
    """Run the ingest pipeline, return results + captured steps."""
    if _init_error:
        return JSONResponse({"error": f"Pipeline init failed: {_init_error}"}, status_code=503)
    if not _pipelines_ready.is_set():
        return JSONResponse({"error": "Pipelines are still loading, please retry in a minute."}, status_code=503)
    capture = StepCapture()
    capture.setLevel(logging.INFO)

    # Attach capture to all xmem loggers
    xmem_loggers = _get_xmem_loggers()
    for lg in xmem_loggers:
        lg.addHandler(capture)

    try:
        result = await ingest_pipeline.run(
            user_query=req.user_query,
            agent_response=req.agent_response or "Acknowledged.",
            user_id=req.user_id,
        )

        # Build structured response
        response: Dict[str, Any] = {
            "model": _get_model_name(ingest_pipeline.model),
            "steps": capture.steps,
            "classification": [],
            "profile": None,
            "temporal": None,
            "summary": None,
        }

        # Classification
        cr = result.get("classification_result")
        if cr and cr.classifications:
            response["classification"] = cr.classifications

        # Profile
        pj = result.get("profile_judge")
        pw = result.get("profile_weaver")
        if pj and pj.operations:
            response["profile"] = {
                "confidence": pj.confidence,
                "operations": [
                    {"type": op.type.value, "content": op.content, "reason": op.reason}
                    for op in pj.operations
                ],
                "weaver": {
                    "succeeded": pw.succeeded if pw else 0,
                    "skipped": pw.skipped if pw else 0,
                    "failed": pw.failed if pw else 0,
                },
            }

        # Temporal
        tj = result.get("temporal_judge")
        tw = result.get("temporal_weaver")
        if tj and tj.operations:
            response["temporal"] = {
                "confidence": tj.confidence,
                "operations": [
                    {"type": op.type.value, "content": op.content, "reason": op.reason}
                    for op in tj.operations
                ],
                "weaver": {
                    "succeeded": tw.succeeded if tw else 0,
                    "skipped": tw.skipped if tw else 0,
                    "failed": tw.failed if tw else 0,
                },
            }

        # Summary
        sj = result.get("summary_judge")
        sw = result.get("summary_weaver")
        if sj and sj.operations:
            response["summary"] = {
                "confidence": sj.confidence,
                "operations": [
                    {"type": op.type.value, "content": op.content, "reason": op.reason}
                    for op in sj.operations
                ],
                "weaver": {
                    "succeeded": sw.succeeded if sw else 0,
                    "skipped": sw.skipped if sw else 0,
                    "failed": sw.failed if sw else 0,
                },
            }

        return JSONResponse(response)

    except Exception as e:
        return JSONResponse({"error": str(e), "steps": capture.steps}, status_code=500)

    finally:
        for lg in xmem_loggers:
            lg.removeHandler(capture)


@app.post("/api/retrieve")
async def api_retrieve(req: RetrieveRequest):
    """Run the retrieval pipeline, return results + captured steps."""
    if _init_error:
        return JSONResponse({"error": f"Pipeline init failed: {_init_error}"}, status_code=503)
    if not _pipelines_ready.is_set():
        return JSONResponse({"error": "Pipelines are still loading, please retry in a minute."}, status_code=503)
    capture = StepCapture()
    capture.setLevel(logging.INFO)

    xmem_loggers = _get_xmem_loggers()
    for lg in xmem_loggers:
        lg.addHandler(capture)

    try:
        result = await retrieval_pipeline.run(
            query=req.query,
            user_id=req.user_id,
        )

        response = {
            "model": _get_model_name(retrieval_pipeline.model),
            "steps": capture.steps,
            "answer": result.answer,
            "sources": [
                {
                    "domain": s.domain,
                    "content": s.content,
                    "score": round(s.score, 3),
                }
                for s in result.sources
            ],
            "confidence": result.confidence,
        }
        return JSONResponse(response)

    except Exception as e:
        return JSONResponse({"error": str(e), "steps": capture.steps}, status_code=500)

    finally:
        for lg in xmem_loggers:
            lg.removeHandler(capture)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _get_model_name(model) -> str:
    """Extract the model name string from a LangChain model instance."""
    return getattr(model, "model", getattr(model, "model_name", "unknown"))


def _get_xmem_loggers() -> List[logging.Logger]:
    """Return all xmem.* loggers so we can attach/detach handlers."""
    names = [
        "xmem.pipelines.ingest",
        "xmem.pipelines.retrieval",
        "xmem.agents.classifier",
        "xmem.agents.profiler",
        "xmem.agents.temporal",
        "xmem.agents.summarizer",
        "xmem.agents.judge",
        "xmem.weaver",
        "xmem.graph.neo4j",
    ]
    loggers = []
    for n in names:
        lg = logging.getLogger(n)
        lg.setLevel(logging.INFO)
        loggers.append(lg)
    return loggers
