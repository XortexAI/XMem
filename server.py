"""
Xmem Test Frontend — FastAPI server.

Exposes the ingest and retrieval pipelines as API endpoints.
Captures pipeline logs so the frontend can show each step.

Usage:
    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

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


# ═══════════════════════════════════════════════════════════════════
# Log capture — collects pipeline log messages during a run
# ═══════════════════════════════════════════════════════════════════

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
SKIP_PIPELINES = os.getenv("XMEM_SKIP_PIPELINES", "").lower() in {"1", "true", "yes"}


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
    if SKIP_PIPELINES:
        print("[init] Skipping pipeline init because XMEM_SKIP_PIPELINES=1", flush=True)
        _pipelines_ready.set()
        return

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
    session_datetime: str = ""
    image_url: str = ""
    effort_level: str = "low"


class RetrieveRequest(BaseModel):
    query: str
    user_id: str = "demo_user"


class ScrapeRequest(BaseModel):
    url: str


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


@app.post("/v1/memory/scrape")
async def scrape_chat_link(req: ScrapeRequest):
    start = time.perf_counter()
    url = req.url

    try:
        result = await _scrape_chat_share(url)
        pairs = result["pairs"]
        elapsed = round((time.perf_counter() - start) * 1000, 2)

        if not pairs:
            return JSONResponse(
                {
                    "status": "error",
                    "data": None,
                    "error": "Failed to extract messages from the provided link.",
                    "elapsed_ms": elapsed,
                },
                status_code=400,
            )

        return JSONResponse(
            {
                "status": "ok",
                "data": {"pairs": pairs},
                "elapsed_ms": elapsed,
            },
        )
    except Exception as exc:
        return JSONResponse(
            {
                "status": "error",
                "data": None,
                "error": str(exc) or repr(exc),
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
            },
            status_code=500,
        )


@app.post("/v1/memory/parse_transcript")
async def parse_transcript(file: UploadFile = File(...)):
    start = time.perf_counter()
    
    try:
        content_bytes = await file.read()
        text = content_bytes.decode("utf-8", errors="ignore")
        
        if not text.strip():
            return JSONResponse(
                {
                    "status": "error",
                    "data": None,
                    "error": "Uploaded file is empty.",
                    "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
                },
                status_code=400,
            )
            
        format_detected, pairs = _parse_transcript_text(text)
        
        if not pairs:
            print("[parse_transcript] Format detection failed, trying LLM fallback", flush=True)
            pairs = await _parse_transcript_with_llm(text)
            
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        
        if not pairs:
            return JSONResponse(
                {
                    "status": "error",
                    "data": None,
                    "error": "Could not extract message pairs from the transcript.",
                    "elapsed_ms": elapsed,
                },
                status_code=400,
            )
            
        return JSONResponse(
            {
                "status": "ok",
                "data": {"pairs": pairs},
                "elapsed_ms": elapsed,
            },
        )
    except UnicodeDecodeError:
        return JSONResponse(
            {
                "status": "error",
                "data": None,
                "error": "Could not decode file. Please upload a text file.",
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
            },
            status_code=400,
        )
    except Exception as exc:
        return JSONResponse(
            {
                "status": "error",
                "data": None,
                "error": str(exc) or repr(exc),
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
            },
            status_code=500,
        )


@app.post("/v1/memory/ingest")
async def v1_ingest_memory(req: IngestRequest):
    """V1-compatible ingest route used by the /context importer."""
    start = time.perf_counter()

    if _init_error:
        return JSONResponse(
            {
                "status": "error",
                "data": None,
                "error": f"Pipeline init failed: {_init_error}",
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
            },
            status_code=503,
        )
    if not _pipelines_ready.is_set() or ingest_pipeline is None:
        return JSONResponse(
            {
                "status": "error",
                "data": None,
                "error": "Ingest pipeline is not ready.",
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
            },
            status_code=503,
        )

    try:
        result = await ingest_pipeline.run(
            user_query=req.user_query,
            agent_response=req.agent_response or "Acknowledged.",
            user_id=req.user_id,
            session_datetime=req.session_datetime,
            image_url=req.image_url,
            effort_level=req.effort_level,
        )

        data = {
            "model": _get_model_name(ingest_pipeline.model),
            "classification": _safe_classifications(result),
            "profile": _build_memory_domain(result.get("profile_judge"), result.get("profile_weaver")),
            "temporal": _build_memory_domain(result.get("temporal_judge"), result.get("temporal_weaver")),
            "summary": _build_memory_domain(result.get("summary_judge"), result.get("summary_weaver")),
            "image": _build_memory_domain(result.get("image_judge"), result.get("image_weaver")),
        }

        return JSONResponse(
            {
                "status": "ok",
                "data": data,
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
    except Exception as exc:
        return JSONResponse(
            {
                "status": "error",
                "data": None,
                "error": str(exc) or repr(exc),
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
            },
            status_code=500,
        )


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


def _safe_classifications(result: Dict[str, Any]) -> list:
    cr = result.get("classification_result")
    if cr and getattr(cr, "classifications", None):
        return cr.classifications
    return []


def _build_memory_domain(judge: Any, weaver: Any) -> dict[str, Any] | None:
    if not judge or not getattr(judge, "operations", None):
        return None

    return {
        "confidence": getattr(judge, "confidence", 0.0),
        "operations": [
            {
                "type": getattr(op.type, "value", op.type),
                "content": op.content,
                "reason": op.reason,
            }
            for op in judge.operations
        ],
        "weaver": {
            "succeeded": getattr(weaver, "succeeded", 0) if weaver else 0,
            "skipped": getattr(weaver, "skipped", 0) if weaver else 0,
            "failed": getattr(weaver, "failed", 0) if weaver else 0,
        },
    }


def _detect_chat_provider(url: str) -> str:
    lowered = url.lower()
    if "chatgpt.com" in lowered or "chat.openai.com" in lowered or "openai.com" in lowered:
        return "chatgpt"
    if "claude.ai" in lowered:
        return "claude"
    if "gemini.google.com" in lowered or "g.co/gemini" in lowered:
        return "gemini"
    return "unknown"


async def _render_chat_share(url: str) -> tuple[str, str]:
    return await asyncio.to_thread(_render_chat_share_sync, url)


def _render_chat_share_sync(url: str) -> tuple[str, str]:
    html = ""
    final_url = url

    with sync_playwright() as p:
        browser = None
        launch_errors = []
        for channel in (None, "msedge", "chrome"):
            try:
                kwargs = {"headless": True}
                if channel:
                    kwargs["channel"] = channel
                browser = p.chromium.launch(**kwargs)
                break
            except Exception as exc:
                launch_errors.append(f"{channel or 'bundled chromium'}: {exc}")

        if browser is None:
            raise RuntimeError(
                "Could not launch Playwright browser. Tried bundled Chromium, "
                f"Edge, and Chrome. Errors: {' | '.join(launch_errors)}"
            )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            ignore_https_errors=True,
        )
        try:
            page = context.new_page()

            def _block_heavy_assets(route):
                if route.request.resource_type in {"image", "media", "font"}:
                    route.abort()
                    return
                route.continue_()

            page.route("**/*", _block_heavy_assets)

            try:
                page.goto(url, wait_until="networkidle", timeout=20000)
            except Exception as exc:
                print(f"[scrape] navigation warning: {exc}", flush=True)

            provider = _detect_chat_provider(page.url or url)
            selector = {
                "chatgpt": "div[data-message-author-role]",
                "claude": "script",
                "gemini": "message-content, div.user-query, div.model-response",
            }.get(provider)
            if selector:
                try:
                    page.wait_for_selector(selector, timeout=12000)
                except Exception as exc:
                    print(f"[scrape] timed out waiting for {provider} content: {exc}", flush=True)

            page.wait_for_timeout(2000)
            final_url = page.url
            html = page.content()
        finally:
            context.close()
            browser.close()

    return html, final_url


def _extract_chat_pairs(url: str, html: str) -> tuple[str, str, list[dict[str, str]]]:
    provider = _detect_chat_provider(url)
    soup = BeautifulSoup(html, "html.parser")
    pairs: list[dict[str, str]] = []
    extraction_method = "none"

    if provider == "chatgpt":
        user_msgs = soup.find_all("div", {"data-message-author-role": "user"})
        asst_msgs = soup.find_all("div", {"data-message-author-role": "assistant"})
        for user_msg, assistant_msg in zip(user_msgs, asst_msgs):
            pairs.append({
                "user_query": user_msg.get_text(separator="\n").strip(),
                "agent_response": assistant_msg.get_text(separator="\n").strip(),
            })
        if pairs:
            extraction_method = "dom"

    elif provider == "claude":
        script_state = soup.find("script", string=re.compile(r"__PRELOADED_STATE__"))
        if script_state and script_state.string:
            try:
                match = re.search(
                    r"__PRELOADED_STATE__\s*=\s*(\{.*?\});",
                    script_state.string,
                    re.DOTALL,
                )
                if match:
                    data = json.loads(match.group(1))
                    messages = data.get("chat", {}).get("messages", [])
                    current_user = ""
                    for msg in messages:
                        if msg.get("sender") == "human":
                            current_user = msg.get("text", "")
                        elif msg.get("sender") == "assistant":
                            pairs.append({
                                "user_query": current_user,
                                "agent_response": msg.get("text", ""),
                            })
                            current_user = ""
                    if pairs:
                        extraction_method = "structured"
            except Exception as exc:
                print(f"[scrape] Claude parse warning: {exc}", flush=True)

    elif provider == "gemini":
        user_blocks = soup.select("message-content[role='user'], div.user-query")
        model_blocks = soup.select("message-content[role='model'], div.model-response")
        for user_block, model_block in zip(user_blocks, model_blocks):
            pairs.append({
                "user_query": user_block.get_text(separator="\n").strip(),
                "agent_response": model_block.get_text(separator="\n").strip(),
            })
        if pairs:
            extraction_method = "dom"

    if not pairs and provider == "unknown":
        paragraphs = [
            node.get_text(separator="\n", strip=True)
            for node in soup.find_all(["p", "div", "span"])
            if len(node.get_text(strip=True)) > 50
        ]
        unique_paras = []
        for paragraph in paragraphs:
            if paragraph not in unique_paras:
                unique_paras.append(paragraph)

        if unique_paras:
            pairs.append({
                "user_query": "Extracted text from link",
                "agent_response": "\n\n".join(unique_paras[:50])[:10000],
            })
            extraction_method = "fallback"

    return provider, extraction_method, pairs


def _parse_cursor_transcript(text: str) -> list[dict[str, str]]:
    """Parse a Cursor-exported markdown transcript into message pairs."""
    pairs: list[dict[str, str]] = []
    
    sections = text.split("---")
    
    start_idx = 0
    if sections and "Exported on" in sections[0]:
        start_idx = 1
    
    current_user_query = None
    
    for section in sections[start_idx:]:
        section = section.strip()
        if not section:
            continue
        
        if section.startswith("**User**"):
            content = section.replace("**User**", "", 1).strip()
            current_user_query = content
        
        elif section.startswith("**Cursor**") or section.startswith("**Assistant**"):
            content = section.replace("**Cursor**", "", 1).replace("**Assistant**", "", 1).strip()
            
            if current_user_query:
                pairs.append({
                    "user_query": current_user_query,
                    "agent_response": content,
                })
                current_user_query = None
    
    return pairs


async def _parse_transcript_with_llm(text: str) -> list[dict[str, str]]:
    """Use an LLM to parse transcript text when format detection fails."""
    from src.models import get_model
    
    max_chars = 50000
    if len(text) > max_chars:
        text = text[:max_chars]
    
    model = get_model(temperature=0.0)
    
    prompt = f"""You are parsing a chat transcript. Extract all user-agent message pairs from the following text.

Return a JSON array of objects with this structure:
[
  {{"user_query": "...", "agent_response": "..."}},
  ...
]

Only return valid JSON, nothing else.

Transcript:
{text}
"""
    
    try:
        response = await model.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            pairs = [
                {
                    "user_query": item.get("user_query", ""), 
                    "agent_response": item.get("agent_response", "")
                }
                for item in data
                if isinstance(item, dict)
            ]
            return pairs
    except Exception as exc:
        print(f"[parse_transcript] LLM parsing failed: {exc}", flush=True)
    
    return []


def _parse_transcript_text(text: str) -> tuple[str, list[dict[str, str]]]:
    """Parse transcript text and return (format, pairs)."""
    if "_Exported on" in text and "from Cursor" in text:
        pairs = _parse_cursor_transcript(text)
        if pairs:
            return "cursor", pairs
            
    return "unknown", []


async def _scrape_chat_share(url: str) -> dict[str, Any]:
    html, final_url = await _render_chat_share(url)
    provider, extraction_method, pairs = _extract_chat_pairs(final_url or url, html)

    return {
        "provider": provider,
        "url": url,
        "final_url": final_url,
        "pairs": pairs,
        "pair_count": len(pairs),
        "html_length": len(html),
        "extraction_method": extraction_method,
    }


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
