"""
/v1/memory/* routes â€” production endpoints for XMem memory operations.

All routes require a valid Bearer API key and respect the per-key rate limit.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request, UploadFile, File
from fastapi.responses import JSONResponse

from src.api.dependencies import (
    enforce_rate_limit,
    get_ingest_pipeline,
    get_retrieval_pipeline,
    require_api_key,
    require_ready,
)
from src.api.schemas import (
    APIResponse,
    BatchIngestRequest,
    BatchIngestResponse,
    DomainResult,
    IngestRequest,
    IngestResponse,
    OperationDetail,
    RetrieveRequest,
    RetrieveResponse,
    SearchRequest,
    SearchResponse,
    ScrapeRequest,
    ScrapeResponse,
    MessagePair,
    SourceRecord,
    StatusEnum,
    WeaverSummary,
)
from src.pipelines.retrieval import RetrievalPipeline

from bs4 import BeautifulSoup
import json
import re
from playwright.sync_api import sync_playwright

logger = logging.getLogger("xmem.api.routes.memory")

_ingest_semaphore = asyncio.Semaphore(5)

router = APIRouter(
    prefix="/v1/memory",
    tags=["memory"],
    dependencies=[Depends(require_ready), Depends(enforce_rate_limit)],
)

scrape_router = APIRouter(
    prefix="/v1/memory",
    tags=["memory"],
    dependencies=[Depends(enforce_rate_limit)],
)


# Helpers
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


# ── Warm browser pool ──────────────────────────────────────────────────────
# Launching Chromium from cold takes 3-5s. We keep a singleton alive and
# reuse it across scrape requests. The browser is thread-safe when each
# request uses its own BrowserContext.

import threading

_browser_lock = threading.Lock()
_pw_instance = None
_browser_instance = None


def _get_or_create_browser():
    """Return a long-lived Playwright browser, launching one if needed."""
    global _pw_instance, _browser_instance

    with _browser_lock:
        # If the browser is still alive, reuse it
        if _browser_instance is not None and _browser_instance.is_connected():
            return _browser_instance

        # Tear down stale Playwright context if any
        if _pw_instance is not None:
            try:
                _pw_instance.stop()
            except Exception:
                pass

        _pw_instance = sync_playwright().start()

        launch_errors = []
        for channel in (None, "msedge", "chrome"):
            try:
                kwargs = {"headless": True}
                if channel:
                    kwargs["channel"] = channel
                _browser_instance = _pw_instance.chromium.launch(**kwargs)
                logger.info("[scrape] Playwright browser launched (channel=%s)", channel or "bundled")
                return _browser_instance
            except Exception as exc:
                launch_errors.append(f"{channel or 'bundled chromium'}: {exc}")

        raise RuntimeError(
            "Could not launch Playwright browser. Tried bundled Chromium, "
            f"Edge, and Chrome. Errors: {' | '.join(launch_errors)}"
        )


def _render_chat_share_sync(url: str) -> tuple[str, str]:
    html = ""
    final_url = url

    browser = _get_or_create_browser()

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
            rtype = route.request.resource_type
            if rtype in {"image", "media", "font", "stylesheet"}:
                route.abort()
                return
            route.continue_()

        page.route("**/*", _block_heavy_assets)

        try:
            # domcontentloaded is much faster than networkidle — we don't
            # need to wait for analytics / tracking pixels to finish.
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
        except Exception as exc:
            logger.warning("Timeout or error during navigation: %s", exc)

        provider = _detect_chat_provider(page.url or url)
        selector = {
            "chatgpt": "div[data-message-author-role]",
            "claude": "script",
            "gemini": "message-content, div.user-query, div.model-response",
        }.get(provider)
        if selector:
            try:
                page.wait_for_selector(selector, timeout=8000)
            except Exception as exc:
                logger.warning("Timed out waiting for %s content: %s", provider, exc)

        # No hardcoded sleep — the selector wait above already guarantees
        # the chat content DOM nodes are present.

        final_url = page.url
        html = page.content()
    finally:
        context.close()
        # NOTE: we intentionally do NOT close the browser — it's pooled.

    return html, final_url


def _extract_chat_pairs(url: str, html: str) -> tuple[str, str, List[MessagePair]]:
    provider = _detect_chat_provider(url)
    soup = BeautifulSoup(html, "html.parser")
    pairs: List[MessagePair] = []
    extraction_method = "none"

    if provider == "chatgpt":
        user_msgs = soup.find_all("div", {"data-message-author-role": "user"})
        asst_msgs = soup.find_all("div", {"data-message-author-role": "assistant"})
        for u, a in zip(user_msgs, asst_msgs):
            pairs.append(MessagePair(
                user_query=u.get_text(separator="\n").strip(),
                agent_response=a.get_text(separator="\n").strip(),
            ))
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
                            pairs.append(MessagePair(
                                user_query=current_user,
                                agent_response=msg.get("text", ""),
                            ))
                            current_user = ""
                    if pairs:
                        extraction_method = "structured"
            except Exception as exc:
                logger.warning("Failed to parse Claude preloaded state: %s", exc)

    elif provider == "gemini":
        user_blocks = soup.select("message-content[role='user'], div.user-query")
        model_blocks = soup.select("message-content[role='model'], div.model-response")
        for u, m in zip(user_blocks, model_blocks):
            pairs.append(MessagePair(
                user_query=u.get_text(separator="\n").strip(),
                agent_response=m.get_text(separator="\n").strip(),
            ))
        if pairs:
            extraction_method = "dom"

    if not pairs and provider == "unknown":
        paragraphs = [
            p.get_text(separator="\n", strip=True)
            for p in soup.find_all(["p", "div", "span"])
            if len(p.get_text(strip=True)) > 50
        ]
        unique_paras = []
        for paragraph in paragraphs:
            if paragraph not in unique_paras:
                unique_paras.append(paragraph)

        if unique_paras:
            text = "\n\n".join(unique_paras[:50])
            pairs.append(MessagePair(
                user_query="Extracted text from link",
                agent_response=text[:10000],
            ))
            extraction_method = "fallback"

    return provider, extraction_method, pairs


def _parse_cursor_transcript(text: str) -> List[MessagePair]:
    """Parse a Cursor-exported markdown transcript into message pairs.
    
    Cursor transcripts have the format:
    _Exported on ... from Cursor_
    ---
    **User**
    <user message>
    ---
    **Cursor**
    <agent response>
    ---
    ...
    """
    pairs: List[MessagePair] = []
    
    # Split by --- separator
    sections = text.split("---")
    
    # Skip the first section if it's the header (contains "Exported on")
    start_idx = 0
    if sections and "Exported on" in sections[0]:
        start_idx = 1
    
    current_user_query = None
    
    for section in sections[start_idx:]:
        section = section.strip()
        if not section:
            continue
        
        # Check if this is a User message
        if section.startswith("**User**"):
            # Extract the user message (remove the **User** header)
            content = section.replace("**User**", "", 1).strip()
            current_user_query = content
        
        # Check if this is a Cursor/Agent message
        elif section.startswith("**Cursor**") or section.startswith("**Assistant**"):
            # Extract the agent response
            content = section.replace("**Cursor**", "", 1).replace("**Assistant**", "", 1).strip()
            
            # If we have a user query, create a pair
            if current_user_query:
                pairs.append(MessagePair(
                    user_query=current_user_query,
                    agent_response=content,
                ))
                current_user_query = None
    
    return pairs


def _parse_antigravity_transcript(text: str) -> List[MessagePair]:
    """Parse an Antigravity-exported markdown transcript into message pairs.

    Antigravity transcripts exported from the Antigravity coding assistant
    follow this format::

        # Chat Conversation

        Note: _This is purely the output of the chat conversation..._

        ### User Input

        <user message>

        ### Planner Response

        <agent response>

        ### User Input

        ...

    Multiple consecutive ``### Planner Response`` blocks (e.g. when the agent
    used tools between messages) are concatenated into a single agent response.
    """
    pairs: List[MessagePair] = []

    # Normalise line endings
    text = text.replace("\r\n", "\n")

    # Split into blocks by H3 headings (### ...)
    # We keep the heading so we know which role each block belongs to.
    blocks = re.split(r"(?m)^(###\s+.+)$", text)

    current_user_query: str | None = None
    planner_chunks: List[str] = []

    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue

        if re.match(r"###\s+User Input", block, re.IGNORECASE):
            # Flush any pending planner chunks as a completed pair
            if current_user_query and planner_chunks:
                pairs.append(MessagePair(
                    user_query=current_user_query,
                    agent_response="\n\n".join(planner_chunks).strip(),
                ))
                planner_chunks = []
            # The next block (index i+1) is the content of this user turn
            current_user_query = None  # will be filled by the content block below

        elif re.match(r"###\s+Planner Response", block, re.IGNORECASE):
            # The next content block belongs to the agent
            pass  # content handled in the else branch below

        else:
            # This is a content block — figure out which role it belongs to by
            # looking at the previous heading token.
            if i > 0:
                prev_heading = blocks[i - 1].strip() if i >= 1 else ""
                if re.match(r"###\s+User Input", prev_heading, re.IGNORECASE):
                    # New user turn — flush previous pair first
                    if current_user_query and planner_chunks:
                        pairs.append(MessagePair(
                            user_query=current_user_query,
                            agent_response="\n\n".join(planner_chunks).strip(),
                        ))
                        planner_chunks = []
                    current_user_query = block

                elif re.match(r"###\s+Planner Response", prev_heading, re.IGNORECASE):
                    # Accumulate (multiple tool-use steps = multiple planner blocks)
                    if block:
                        planner_chunks.append(block)

    # Flush last pair
    if current_user_query and planner_chunks:
        pairs.append(MessagePair(
            user_query=current_user_query,
            agent_response="\n\n".join(planner_chunks).strip(),
        ))

    return pairs


async def _parse_transcript_with_llm(text: str) -> List[MessagePair]:
    """Use an LLM to parse transcript text when format detection fails."""
    from src.models import get_model
    
    # Limit text size to avoid token issues
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
        
        # Try to extract JSON from the response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            pairs = [
                MessagePair(user_query=item.get("user_query", ""), agent_response=item.get("agent_response", ""))
                for item in data
                if isinstance(item, dict)
            ]
            return pairs
    except Exception as exc:
        logger.warning("LLM transcript parsing failed: %s", exc)
    
    return []


def _parse_transcript_text(text: str) -> tuple[str, List[MessagePair]]:
    """Parse transcript text and return (format, pairs)."""

    # Detect Cursor format
    if "_Exported on" in text and "from Cursor" in text:
        pairs = _parse_cursor_transcript(text)
        if pairs:
            return "cursor", pairs

    # Detect Antigravity format
    if "# Chat Conversation" in text and ("### User Input" in text or "### Planner Response" in text):
        pairs = _parse_antigravity_transcript(text)
        if pairs:
            return "antigravity", pairs

    return "unknown", []


async def _scrape_chat_share(url: str) -> Dict[str, Any]:
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


# POST /v1/memory/ingest
@router.post(
    "/ingest",
    response_model=APIResponse,
    summary="Ingest a conversation turn into long-term memory",
)
async def ingest_memory(req: IngestRequest, request: Request, user: dict = Depends(require_api_key)):
    start = time.perf_counter()
    pipeline = get_ingest_pipeline()

    # Get username from authenticated user
    user_id = user.get("username") or user.get("name") or user["id"]

    try:
        async with _ingest_semaphore:
            result = await asyncio.wait_for(
                pipeline.run(
                    user_query=req.user_query,
                    agent_response=req.agent_response or "Acknowledged.",
                    user_id=user_id,
                    session_datetime=req.session_datetime,
                    image_url=req.image_url,
                    effort_level=req.effort_level,
                ),
                timeout=120.0
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
        logger.exception("Ingest failed for user=%s", user_id)
        return _error(request, str(exc), 500, elapsed)


def _safe_classifications(result: Dict[str, Any]) -> list:
    cr = result.get("classification_result")
    if cr and getattr(cr, "classifications", None):
        return cr.classifications
    return []


# POST /v1/memory/batch-ingest
@router.post(
    "/batch-ingest",
    response_model=APIResponse,
    summary="Ingest multiple conversation turns into long-term memory sequentially",
)
async def batch_ingest_memory(req: BatchIngestRequest, request: Request, user: dict = Depends(require_api_key)):
    start = time.perf_counter()
    pipeline = get_ingest_pipeline()
    user_id = user.get("username") or user.get("name") or user["id"]

    results = []

    for item in req.items:
        result = await asyncio.wait_for(
            pipeline.run(
                user_query=item.user_query,
                agent_response=item.agent_response or "Acknowledged.",
                user_id=user_id,
                session_datetime=item.session_datetime,
                image_url=item.image_url,
                effort_level=item.effort_level,
            ),
            timeout=120.0
        )
        
        data = IngestResponse(
            model=_model_name(pipeline.model),
            classification=_safe_classifications(result),
            profile=_build_domain_result(result.get("profile_judge"), result.get("profile_weaver")),
            temporal=_build_domain_result(result.get("temporal_judge"), result.get("temporal_weaver")),
            summary=_build_domain_result(result.get("summary_judge"), result.get("summary_weaver")),
            image=_build_domain_result(result.get("image_judge"), result.get("image_weaver")),
        )
        results.append(data)

    response_data = BatchIngestResponse(results=results)
    
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    return _wrap(request, response_data, elapsed)



# POST /v1/memory/retrieve
@router.post(
    "/retrieve",
    response_model=APIResponse,
    summary="Retrieve an LLM-generated answer backed by stored memories",
)
async def retrieve_memory(req: RetrieveRequest, request: Request, user: dict = Depends(require_api_key)):
    start = time.perf_counter()
    pipeline = get_retrieval_pipeline()
    
    # Get username from authenticated user
    user_id = user.get("username") or user.get("name") or user["id"]

    try:
        result = await pipeline.run(query=req.query, user_id=user_id, top_k=req.top_k)
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
        logger.exception("Retrieve failed for user=%s", user_id)
        return _error(request, str(exc), 500, elapsed)


# POST /v1/memory/search
@router.post(
    "/search",
    response_model=APIResponse,
    summary="Raw semantic search across memory domains (no LLM answer)",
)
async def search_memory(req: SearchRequest, request: Request, user: dict = Depends(require_api_key)):
    start = time.perf_counter()
    pipeline = get_retrieval_pipeline()
    
    # Get username from authenticated user
    user_id = user.get("username") or user.get("name") or user["id"]

    try:
        all_results: List[SourceRecord] = []

        if "profile" in req.domains:
            all_results.extend(_search_profile(pipeline, user_id))
        if "temporal" in req.domains:
            all_results.extend(_search_temporal(pipeline, req.query, user_id, req.top_k))
        if "summary" in req.domains:
            all_results.extend(await _search_summary(pipeline, req.query, user_id, req.top_k))

        data = SearchResponse(results=all_results, total=len(all_results))
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        return _wrap(request, data, elapsed)

    except Exception as exc:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("Search failed for user=%s", user_id)
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


# POST /v1/memory/scrape
@scrape_router.post(
    "/scrape",
    response_model=APIResponse,
    summary="Scrape a shared AI chat link into message pairs",
)
async def scrape_chat_link(req: ScrapeRequest, request: Request):
    start = time.perf_counter()
    url = req.url
    
    try:
        result = await _scrape_chat_share(url)
        pairs = result["pairs"]

        if not pairs:
            return _error(request, "Failed to extract messages from the provided link.", 400)

        data = ScrapeResponse(pairs=pairs)
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        return _wrap(request, data, elapsed)

    except Exception as exc:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("Scrape failed for url=%s", url)
        return _error(request, str(exc) or repr(exc), 500, elapsed)



# POST /v1/memory/parse_transcript
@scrape_router.post(
    "/parse_transcript",
    response_model=APIResponse,
    summary="Parse an uploaded chat transcript file into message pairs",
)
async def parse_transcript(
    request: Request,
    file: UploadFile = File(..., description="Chat transcript file (.txt, .md, .json)")
):
    start = time.perf_counter()
    
    try:
        # Read file content
        content_bytes = await file.read()
        text = content_bytes.decode("utf-8", errors="ignore")
        
        if not text.strip():
            return _error(request, "Uploaded file is empty.", 400)
        
        # Try to parse the transcript
        format_detected, pairs = _parse_transcript_text(text)
        
        # If no pairs found, try LLM fallback
        if not pairs:
            logger.info("Format detection failed, trying LLM fallback")
            pairs = await _parse_transcript_with_llm(text)
        
        if not pairs:
            return _error(request, "Could not extract message pairs from the transcript.", 400)
        
        data = ScrapeResponse(pairs=pairs)
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        return _wrap(request, data, elapsed)
    
    except UnicodeDecodeError:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        return _error(request, "Could not decode file. Please upload a text file.", 400, elapsed)
    except Exception as exc:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("Transcript parsing failed for file=%s", file.filename)
        return _error(request, str(exc) or repr(exc), 500, elapsed)
