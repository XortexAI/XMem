"""
Retrieval Pipeline — two-step agentic retrieval.

Step 1: The LLM decides WHAT to fetch by making tool calls:
    - search_profile(topic, sub_topic) → Pinecone metadata lookup
    - search_temporal(query)           → Neo4j semantic search
    - search_summary(query)            → Pinecone semantic search (domain=summary)

Step 2: We execute the tool calls, collect the results, and send them
        back to the LLM to generate a final answer.

Usage:
    pipeline = RetrievalPipeline()
    result = await pipeline.run(
        query="When is my dentist appointment?",
        user_id="user_001",
    )
    print(result.answer)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from src.config.metrics import METRICS
from src.config import settings
from src.graph.neo4j_client import Neo4jClient
from src.prompts.retrieval import ANSWER_PROMPT, build_system_prompt
from src.schemas.retrieval import RetrievalResult, SourceRecord
from src.schemas.code import snippets_namespace
from src.storage.base import BaseVectorStore
from src.storage.factory import get_vector_store

load_dotenv()

logger = logging.getLogger("xmem.pipelines.retrieval")

DEFAULT_RAW_SEARCH_DOMAINS = ("profile", "temporal", "summary", "snippet")
PROFILE_CATALOG_CACHE_TTL_SECONDS = 60.0
RETRIEVAL_PLAN_CACHE_TTL_SECONDS = 60.0
RETRIEVAL_PLAN_CACHE_MAX = 256


class _LatencyTracker:
    """In-process latency percentiles for retrieval modes."""

    def __init__(self, max_samples: int = 512) -> None:
        self._max_samples = max_samples
        self._samples: Dict[str, deque[float]] = {}

    def record(self, mode: str, elapsed_ms: float) -> Dict[str, Any]:
        samples = self._samples.setdefault(mode, deque(maxlen=self._max_samples))
        samples.append(elapsed_ms)
        stats = self.snapshot(mode)
        stats["current_ms"] = round(elapsed_ms, 2)
        return stats

    def snapshot(self, mode: str) -> Dict[str, Any]:
        samples = list(self._samples.get(mode, ()))
        return {
            "mode": mode,
            "samples": len(samples),
            "p50_ms": self._percentile(samples, 50),
            "p95_ms": self._percentile(samples, 95),
            "p99_ms": self._percentile(samples, 99),
        }

    @staticmethod
    def _percentile(samples: List[float], percentile: int) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        index = round((len(ordered) - 1) * (percentile / 100))
        return round(ordered[index], 2)


# ═══════════════════════════════════════════════════════════════════════════
# Tool schemas — These are the "function signatures" exposed to the LLM
# ═══════════════════════════════════════════════════════════════════════════


class SearchProfile(BaseModel):
    """Look up user profile facts by topic.
    Use when the question asks about a specific attribute like job, name, hobby, food preference, etc.
    You MUST use a topic value from the AVAILABLE PROFILES list below.
    This returns ALL sub-topics under that topic for full context."""

    topic: str = Field(description="Profile topic, e.g. 'work', 'interest', 'personal'")


class SearchTemporal(BaseModel):
    """Search for date-based events like appointments, birthdays, milestones.
    Use when the question involves 'when', dates, schedules, or events."""

    query: str = Field(
        description="Short search query describing the event, e.g. 'dentist appointment'"
    )


class SearchSummary(BaseModel):
    """Search general conversation summaries for broad context.
    Use as a fallback for questions that don't fit profile or temporal domains."""

    query: str = Field(
        description="Short search query, e.g. 'what does the user enjoy'"
    )


class SearchSnippet(BaseModel):
    """Search for personal code snippets previously saved by the user.
    Use when the question asks about a specific piece of code, script, or technical configuration the user wrote.
    """

    query: str = Field(
        description="Short search query, e.g. 'python database connection script'"
    )


TOOLS = [SearchProfile, SearchTemporal, SearchSummary, SearchSnippet]


# ═══════════════════════════════════════════════════════════════════════════
# Embedding helper (reuses the cached model from ingest)
# ═══════════════════════════════════════════════════════════════════════════


def _get_embed_fn() -> Callable[[str], List[float]]:
    from src.pipelines.ingest import embed_text

    return embed_text


# ═══════════════════════════════════════════════════════════════════════════
# RetrievalPipeline
# ═══════════════════════════════════════════════════════════════════════════


class RetrievalPipeline:
    """Two-step agentic retrieval: tool-call → fetch → answer."""

    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        vector_store: Optional[BaseVectorStore] = None,
        neo4j_client: Optional[Neo4jClient] = None,
    ) -> None:
        # ── LLM ───────────────────────────────────────────────────────
        if model is None:
            from src.models import get_model

            override = settings.retrieval_model
            self.model = get_model(model_name=override) if override else get_model()
        else:
            self.model = model

        # Bind tools so the LLM knows about them
        self.model_with_tools = self.model.bind_tools(TOOLS)

        # ── Vector store (Pinecone) ───────────────────────────────────
        if vector_store is None:
            self.vector_store = get_vector_store()
        else:
            self.vector_store = vector_store

        # ── Graph store (Neo4j) ───────────────────────────────────────
        embed_fn = _get_embed_fn()
        if neo4j_client is None:
            self.neo4j = Neo4jClient(
                uri=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
                embedding_fn=embed_fn,
            )
            self.neo4j.connect()
        else:
            self.neo4j = neo4j_client

        self.embed_fn = embed_fn
        self._snippet_stores: Dict[str, BaseVectorStore] = {}
        self._cached_profile_records: List[Any] = []
        self._profile_catalog_cache: Dict[
            str, Tuple[float, List[Dict[str, str]], List[Any]]
        ] = {}
        self._retrieval_plan_cache: Dict[
            Tuple[Any, ...], Tuple[float, List[Dict[str, Any]]]
        ] = {}
        self._latency_tracker = _LatencyTracker()

        logger.info("RetrievalPipeline initialized")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Run the two-step retrieval pipeline."""
        start = time.perf_counter()

        logger.info("=" * 60)
        logger.info("RETRIEVAL PIPELINE START")
        logger.info("  query: %s", query)
        logger.info("  user_id: %s", user_id)
        logger.info("=" * 60)

        # ── Step 0: Fetch available profile catalog for this user ─────
        profile_catalog, profile_records = self._fetch_profile_catalog(user_id)
        catalog_text = self._format_catalog(profile_catalog)
        logger.info("Available profiles: %s", catalog_text)

        # Store profile records so _search_profile can reuse them
        self._cached_profile_records = profile_records

        # ── Step 1: Ask LLM what to fetch (tool calls) ────────────────
        system_prompt = build_system_prompt(profile_catalog=catalog_text)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        plan_cache_key = self._retrieval_plan_cache_key(
            user_id=user_id,
            query=query,
            top_k=top_k,
            profile_catalog=profile_catalog,
        )
        cached_tool_calls = self._get_cached_retrieval_plan(plan_cache_key)
        if cached_tool_calls is not None:
            ai_response: AIMessage = AIMessage(content="")
            tool_calls = cached_tool_calls
            logger.info("Using cached retrieval plan (tool_calls=%d)", len(tool_calls))
        else:
            ai_response = await self.model_with_tools.ainvoke(messages)
            tool_calls = ai_response.tool_calls or []
            if tool_calls:
                self._remember_retrieval_plan(plan_cache_key, tool_calls)
            logger.info("LLM response received (tool_calls=%d)", len(tool_calls))

        # ── Step 2: Execute tool calls ────────────────────────────────
        sources: List[SourceRecord] = []
        tool_messages: List[ToolMessage] = []

        if tool_calls:
            called_tools = set()

            async def _process_tool_call(tc):
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]
                logger.info("  Tool call: %s(%s)", tool_name, tool_args)
                records = await self._execute_tool(
                    tool_name,
                    tool_args,
                    user_id,
                    top_k,
                )
                return tool_name, tool_args, tool_id, records

            tool_results = await asyncio.gather(
                *[_process_tool_call(tc) for tc in tool_calls]
            )

            for tool_name, tool_args, tool_id, records in tool_results:
                sources.extend(records)

                # Build ToolMessage for the LLM
                tool_result_text = self._format_tool_results(records)
                tool_messages.append(
                    ToolMessage(content=tool_result_text, tool_call_id=tool_id)
                )

                called_tools.add(tool_name.lower().replace("_", ""))

            # Auto-add summary context when only profile or temporal was requested
            if "searchsummary" not in called_tools:
                logger.info("  Auto-adding summary context (top_k=5)")
                extra = await self._search_summary(
                    query=query,
                    user_id=user_id,
                    top_k=20,
                )
                if extra:
                    sources.extend(extra)
                    extra_text = self._format_tool_results(extra)
                    # Append as a system-injected tool message
                    tool_messages.append(
                        ToolMessage(
                            content=f"[Auto-fetched summary context]\n{extra_text}",
                            tool_call_id=tool_calls[-1]["id"],
                        )
                    )

            # ── Step 3: Generate final answer (clean RAG prompt) ─────
            # Only send the retrieved context + user query to the LLM.
            # No need for the system prompt, tool schemas, or tool-call history.
            context_text = "\n".join(tm.content for tm in tool_messages)
            answer = await self._generate_answer(query=query, context_text=context_text)
        else:
            # No tool calls — LLM answered directly (shouldn't happen often)
            answer = ai_response.content
            logger.info("LLM answered without tool calls")

        answer = self._coerce_answer_text(answer)

        confidence = min(1.0, len(sources) * 0.2) if sources else 0.1
        self.record_latency("answer", (time.perf_counter() - start) * 1000)

        logger.info("=" * 60)
        logger.info("RETRIEVAL PIPELINE COMPLETE")
        logger.info("  sources: %d", len(sources))
        logger.info(
            "  answer: %s", answer[:100] + "..." if len(answer) > 100 else answer
        )
        logger.info("=" * 60)

        return RetrievalResult(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence,
        )

    async def raw_search(
        self,
        query: str,
        user_id: str,
        domains: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> Tuple[List[SourceRecord], Dict[str, Any]]:
        """Return ranked raw memory hits without LLM tool selection or synthesis."""
        start = time.perf_counter()
        selected_domains = domains or list(DEFAULT_RAW_SEARCH_DOMAINS)
        selected = [
            domain
            for domain in selected_domains
            if domain in DEFAULT_RAW_SEARCH_DOMAINS
        ]

        results: List[SourceRecord] = []
        if "profile" in selected:
            results.extend(self._raw_profile_records(user_id=user_id, top_k=top_k))

        async def _safe_search(domain: str, search_coro) -> List[SourceRecord]:
            try:
                return await search_coro
            except Exception as exc:
                logger.warning("Raw %s search failed: %s", domain, exc)
                return []

        tasks = []
        if "temporal" in selected:
            tasks.append(
                _safe_search("temporal", self._search_temporal(query, user_id, top_k))
            )
        if "summary" in selected:
            tasks.append(
                _safe_search("summary", self._search_summary(query, user_id, top_k))
            )
        if "snippet" in selected:
            tasks.append(
                _safe_search("snippet", self._search_snippet(query, user_id, top_k))
            )

        if tasks:
            for records in await asyncio.gather(*tasks):
                results.extend(records)

        results.sort(key=lambda record: record.score, reverse=True)
        latency = self.record_latency("raw", (time.perf_counter() - start) * 1000)
        logger.info(
            "Raw retrieval complete (domains=%s results=%d p50=%sms p95=%sms p99=%sms)",
            ",".join(selected),
            len(results),
            latency["p50_ms"],
            latency["p95_ms"],
            latency["p99_ms"],
        )
        return results, latency

    async def synthesize_answer(self, query: str, sources: List[SourceRecord]) -> str:
        """Generate an answer from already-retrieved raw sources."""
        context_text = self._format_tool_results(sources)
        return await self._generate_answer(query=query, context_text=context_text)

    def record_latency(self, mode: str, elapsed_ms: float) -> Dict[str, Any]:
        stats = self._latency_tracker.record(mode, elapsed_ms)
        try:
            METRICS.pipeline_stage_duration.labels(
                pipeline="retrieval",
                stage=mode,
            ).observe(elapsed_ms / 1000)
        except Exception:
            logger.debug("Failed to observe retrieval latency metric", exc_info=True)
        logger.info(
            "Retrieval latency mode=%s current=%sms p50=%sms p95=%sms p99=%sms",
            mode,
            stats["current_ms"],
            stats["p50_ms"],
            stats["p95_ms"],
            stats["p99_ms"],
        )
        return stats

    async def _generate_answer(self, query: str, context_text: str) -> str:
        answer_prompt = ANSWER_PROMPT.format(
            context=context_text,
            query=query,
        )
        final_response = await self.model.ainvoke([HumanMessage(content=answer_prompt)])
        return self._coerce_answer_text(final_response.content)

    @staticmethod
    def _coerce_answer_text(answer: Any) -> str:
        if isinstance(answer, list):
            parts = []
            for c in answer:
                if isinstance(c, dict) and "text" in c:
                    parts.append(c["text"])
                elif isinstance(c, str):
                    parts.append(c)
                else:
                    parts.append(str(c))
            return "\n".join(parts)
        return str(answer)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_id: str,
        top_k: int,
    ) -> List[SourceRecord]:
        """Dispatch a tool call to the appropriate data store."""

        # Normalise: Gemini may return e.g. "search_profile" or "SearchProfile"
        name = tool_name.lower().replace("_", "")

        if name == "searchprofile":
            return self._search_profile(
                topic=tool_args.get("topic", ""),
                user_id=user_id,
            )
        elif name == "searchtemporal":
            return await self._search_temporal(
                query=tool_args.get("query", ""),
                user_id=user_id,
                top_k=10,  # Return top 3 temporal events for better context
            )
        elif name == "searchsummary":
            return await self._search_summary(
                query=tool_args.get("query", ""),
                user_id=user_id,
                top_k=15,
            )
        elif name == "searchsnippet":
            return await self._search_snippet(
                query=tool_args.get("query", ""),
                user_id=user_id,
                top_k=5,
            )
        else:
            logger.warning("Unknown tool: %s (normalised: %s)", tool_name, name)
            return []

    # -- Profile: Pinecone metadata lookup ─────────────────────────────

    def _search_profile(
        self,
        topic: str,
        user_id: str,
    ) -> List[SourceRecord]:
        """Fetch ALL profile entries for a given topic.

        Uses the cached results from the catalog fetch so we don't
        need a second Pinecone call.  Filters by topic prefix in
        the `main_content` metadata key (e.g. topic='work' matches
        'work_company', 'work_title', etc.).
        """
        topic_prefix = topic.strip().lower().replace(" ", "_")

        records = []
        for r in getattr(self, "_cached_profile_records", []):
            main_content = r.metadata.get("main_content", "")
            if not main_content.startswith(topic_prefix):
                continue

            # Derive sub_topic from the main_content key
            parts = main_content.split("_", 1)
            sub_topic = parts[1] if len(parts) == 2 else ""

            records.append(
                SourceRecord(
                    domain="profile",
                    content=r.content,
                    score=r.score,
                    metadata={
                        "id": r.id,
                        "topic": topic,
                        "sub_topic": sub_topic,
                        **r.metadata,
                    },
                )
            )

        logger.info("  → Profile [%s]: %d results", topic, len(records))
        return records

    # -- Temporal: Neo4j semantic search ───────────────────────────────

    async def _search_temporal(
        self,
        query: str,
        user_id: str,
        top_k: int = 3,
    ) -> List[SourceRecord]:
        """Semantic search over temporal events in Neo4j."""
        import asyncio
        from functools import partial

        loop = asyncio.get_running_loop()
        events = await loop.run_in_executor(
            None,
            partial(
                self.neo4j.search_events_by_embedding,
                user_id=user_id,
                query_text=query,
                top_k=top_k,
                similarity_threshold=0.15,
            ),
        )

        records = []
        for ev in events:
            date_str = ev.get("date", "")
            event_name = ev.get("event_name", "")
            desc = ev.get("desc", "")
            year = ev.get("year", "")
            time = ev.get("time", "")

            # Readable text for the LLM
            content_parts = []
            if date_str:
                date_display = f"{date_str}"
                if year:
                    date_display += f", {year}"
                content_parts.append(f"Date: {date_display}")
            if event_name:
                content_parts.append(f"Event: {event_name}")
            if desc:
                content_parts.append(f"Description: {desc}")
            if time:
                content_parts.append(f"Time: {time}")

            content = " | ".join(content_parts)

            records.append(
                SourceRecord(
                    domain="temporal",
                    content=content,
                    score=ev.get("similarity_score", 0.0),
                    metadata=ev,
                )
            )

        logger.info("  → Temporal [%s]: %d results", query, len(records))
        return records

    # -- Summary: Pinecone semantic search ─────────────────────────────

    async def _search_summary(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
    ) -> List[SourceRecord]:
        """Semantic search over summary entries in Pinecone."""

        results = await self.vector_store.search_by_text(
            query_text=query,
            top_k=top_k,
            filters={
                "user_id": user_id,
                "domain": "summary",
            },
        )

        records = []
        for r in results:
            records.append(
                SourceRecord(
                    domain="summary",
                    content=r.content,
                    score=r.score,
                    metadata={"id": r.id, **r.metadata},
                )
            )

        logger.info("  → Summary [%s]: %d results", query, len(records))
        return records

    # -- Snippet: Pinecone semantic search ─────────────────────────────

    def _get_snippet_store(self, user_id: str) -> BaseVectorStore:
        if user_id not in self._snippet_stores:
            ns = snippets_namespace(user_id)
            self._snippet_stores[user_id] = get_vector_store(
                namespace=ns,
                create_if_not_exists=False,
            )
        return self._snippet_stores[user_id]

    async def _search_snippet(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> List[SourceRecord]:
        """Semantic search over user code snippets (sandboxed namespace)."""
        store = self._get_snippet_store(user_id)

        # In the sandboxed namespace, we can just search. We pass domain filter just in case.
        results = await store.search_by_text(
            query_text=query,
            top_k=top_k,
            filters={"domain": "snippet"},
        )

        records = []
        for r in results:
            content = r.content
            snippet = r.metadata.get("code_snippet", "")
            if snippet:
                # Add the actual code block directly into the context content for LLM
                lang = r.metadata.get("language", "")
                content += f"\n```{lang}\n{snippet}\n```"

            records.append(
                SourceRecord(
                    domain="snippet",
                    content=content,
                    score=r.score,
                    metadata={"id": r.id, **r.metadata},
                )
            )

        logger.info("  → Snippet [%s]: %d results", query, len(records))
        return records

    # ------------------------------------------------------------------
    # Profile catalog (tells the LLM what profile keys exist)
    # ------------------------------------------------------------------

    def _raw_profile_records(self, user_id: str, top_k: int) -> List[SourceRecord]:
        _, raw_records = self._fetch_profile_catalog(user_id)
        return [
            SourceRecord(
                domain="profile",
                content=r.content,
                score=r.score,
                metadata={"id": r.id, **r.metadata},
            )
            for r in raw_records[:top_k]
        ]

    def _retrieval_plan_cache_key(
        self,
        user_id: str,
        query: str,
        top_k: int,
        profile_catalog: List[Dict[str, str]],
    ) -> Tuple[Any, ...]:
        catalog_key = tuple(
            sorted(
                (entry.get("topic", ""), entry.get("sub_topic", ""))
                for entry in profile_catalog
            )
        )
        return (user_id, query.strip().lower(), top_k, catalog_key)

    def _get_cached_retrieval_plan(
        self,
        cache_key: Tuple[Any, ...],
    ) -> Optional[List[Dict[str, Any]]]:
        cached = self._retrieval_plan_cache.get(cache_key)
        if not cached:
            return None

        cached_at, tool_calls = cached
        if time.monotonic() - cached_at > RETRIEVAL_PLAN_CACHE_TTL_SECONDS:
            self._retrieval_plan_cache.pop(cache_key, None)
            return None

        return [
            {
                "name": call["name"],
                "args": dict(call.get("args", {})),
                "id": f"cached-plan-{idx}",
            }
            for idx, call in enumerate(tool_calls)
        ]

    def _remember_retrieval_plan(
        self,
        cache_key: Tuple[Any, ...],
        tool_calls: List[Dict[str, Any]],
    ) -> None:
        clean_calls = [
            {
                "name": str(call.get("name", "")),
                "args": dict(call.get("args", {})),
            }
            for call in tool_calls
            if call.get("name")
        ]
        if not clean_calls:
            return

        if len(self._retrieval_plan_cache) >= RETRIEVAL_PLAN_CACHE_MAX:
            oldest_key = next(iter(self._retrieval_plan_cache))
            self._retrieval_plan_cache.pop(oldest_key, None)

        self._retrieval_plan_cache[cache_key] = (
            time.monotonic(),
            clean_calls,
        )

    def _fetch_profile_catalog(self, user_id: str):
        """Fetch all profile entries for a user.

        Returns:
            (catalog, raw_results)
            catalog  — list of {topic, sub_topic} for the prompt
            raw_results — the full SearchResult list, cached for _search_profile
        """
        now = time.monotonic()
        cached = self._profile_catalog_cache.get(user_id)
        if cached and now - cached[0] <= PROFILE_CATALOG_CACHE_TTL_SECONDS:
            return cached[1], cached[2]

        try:
            results = self.vector_store.search_by_metadata(
                filters={"user_id": user_id, "domain": "profile"},
                top_k=100,
            )
        except Exception as exc:
            logger.warning("Failed to fetch profile catalog: %s", exc)
            return [], []

        catalog: List[Dict[str, str]] = []
        seen = set()

        for r in results:
            main_content = r.metadata.get("main_content", "")
            if not main_content or main_content in seen:
                continue
            seen.add(main_content)

            parts = main_content.split("_", 1)
            if len(parts) == 2:
                catalog.append(
                    {
                        "topic": parts[0],
                        "sub_topic": parts[1],
                    }
                )
            else:
                catalog.append(
                    {
                        "topic": main_content,
                        "sub_topic": "",
                    }
                )

        self._profile_catalog_cache[user_id] = (now, catalog, results)
        return catalog, results

    def _format_catalog(self, catalog: List[Dict[str, str]]) -> str:
        """Format profile catalog for the system prompt."""
        if not catalog:
            return "(No profile data stored yet)"

        lines = []
        for entry in catalog:
            t = entry["topic"]
            st = entry["sub_topic"]
            lines.append(f"  - {t} / {st}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_tool_results(self, records: List[SourceRecord]) -> str:
        """Format source records into text for the LLM."""
        if not records:
            return "No results found."

        lines = []
        for i, rec in enumerate(records, 1):
            score_str = f" (score: {rec.score:.2f})" if rec.score > 0 else ""
            lines.append(f"{i}. [{rec.domain}]{score_str} {rec.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close connections."""
        try:
            self.neo4j.close()
        except Exception:
            pass
        logger.info("RetrievalPipeline closed")
