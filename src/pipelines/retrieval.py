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

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from src.config import settings
from src.graph.neo4j_client import Neo4jClient
from src.prompts.retrieval import ANSWER_PROMPT, build_system_prompt
from src.schemas.retrieval import RetrievalResult, SourceRecord
from src.storage.pinecone import PineconeVectorStore

load_dotenv()

logger = logging.getLogger("xmem.pipelines.retrieval")


# ═══════════════════════════════════════════════════════════════════════════
# Tool schemas — These are the "function signatures" exposed to the LLM
# ═══════════════════════════════════════════════════════════════════════════

class SearchProfile(BaseModel):
    """Look up a specific user profile fact by topic and sub_topic.
    Use when the question asks about a specific attribute like job, name, hobby, food preference, etc.
    You MUST use topic and sub_topic values from the AVAILABLE PROFILES list."""

    topic: str = Field(description="Profile topic, e.g. 'work', 'interest', 'personal'")
    sub_topic: str = Field(description="Profile sub-topic, e.g. 'company', 'foods', 'name'")


class SearchTemporal(BaseModel):
    """Search for date-based events like appointments, birthdays, milestones.
    Use when the question involves 'when', dates, schedules, or events."""

    query: str = Field(description="Short search query describing the event, e.g. 'dentist appointment'")


class SearchSummary(BaseModel):
    """Search general conversation summaries for broad context.
    Use as a fallback for questions that don't fit profile or temporal domains."""

    query: str = Field(description="Short search query, e.g. 'what does the user enjoy'")


TOOLS = [SearchProfile, SearchTemporal, SearchSummary]


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
        vector_store: Optional[PineconeVectorStore] = None,
        neo4j_client: Optional[Neo4jClient] = None,
    ) -> None:
        # ── LLM ───────────────────────────────────────────────────────
        if model is None:
            from src.models.gemini import build_gemini_model

            self.model = build_gemini_model()
        else:
            self.model = model

        # Bind tools so the LLM knows about them
        self.model_with_tools = self.model.bind_tools(TOOLS)

        # ── Vector store (Pinecone) ───────────────────────────────────
        if vector_store is None:
            self.vector_store = PineconeVectorStore()
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

        logger.info("=" * 60)
        logger.info("RETRIEVAL PIPELINE START")
        logger.info("  query: %s", query)
        logger.info("  user_id: %s", user_id)
        logger.info("=" * 60)

        # ── Step 0: Fetch available profile catalog for this user ─────
        profile_catalog = self._fetch_profile_catalog(user_id)
        catalog_text = self._format_catalog(profile_catalog)
        logger.info("Available profiles: %s", catalog_text)

        # ── Step 1: Ask LLM what to fetch (tool calls) ────────────────
        system_prompt = build_system_prompt(profile_catalog=catalog_text)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        ai_response: AIMessage = await self.model_with_tools.ainvoke(messages)
        logger.info("LLM response received (tool_calls=%d)", len(ai_response.tool_calls or []))

        # ── Step 2: Execute tool calls ────────────────────────────────
        sources: List[SourceRecord] = []
        tool_messages: List[ToolMessage] = []

        if ai_response.tool_calls:
            for tc in ai_response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                logger.info("  Tool call: %s(%s)", tool_name, tool_args)

                records = await self._execute_tool(
                    tool_name, tool_args, user_id, top_k,
                )
                sources.extend(records)

                # Build ToolMessage for the LLM
                tool_result_text = self._format_tool_results(records)
                tool_messages.append(
                    ToolMessage(content=tool_result_text, tool_call_id=tool_id)
                )

            # ── Step 3: Send results back to LLM for final answer ─────
            messages.append(ai_response)
            messages.extend(tool_messages)
            messages.append(HumanMessage(content=(
                "Now answer the original question using the retrieved information above. "
                "Be concise and direct. Use 'you' when referring to the user."
            )))

            final_response = await self.model.ainvoke(messages)
            answer = final_response.content
        else:
            # No tool calls — LLM answered directly (shouldn't happen often)
            answer = ai_response.content
            logger.info("LLM answered without tool calls")

        if isinstance(answer, list):
            answer = "\n".join(str(c) for c in answer)

        confidence = min(1.0, len(sources) * 0.2) if sources else 0.1

        logger.info("=" * 60)
        logger.info("RETRIEVAL PIPELINE COMPLETE")
        logger.info("  sources: %d", len(sources))
        logger.info("  answer: %s", answer[:100] + "..." if len(answer) > 100 else answer)
        logger.info("=" * 60)

        return RetrievalResult(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence,
        )

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

        if tool_name == "SearchProfile":
            return self._search_profile(
                topic=tool_args.get("topic", ""),
                sub_topic=tool_args.get("sub_topic", ""),
                user_id=user_id,
            )
        elif tool_name == "SearchTemporal":
            return self._search_temporal(
                query=tool_args.get("query", ""),
                user_id=user_id,
                top_k=top_k,
            )
        elif tool_name == "SearchSummary":
            return await self._search_summary(
                query=tool_args.get("query", ""),
                user_id=user_id,
                top_k=top_k,
            )
        else:
            logger.warning("Unknown tool: %s", tool_name)
            return []

    # -- Profile: Pinecone metadata lookup ─────────────────────────────

    def _search_profile(
        self,
        topic: str,
        sub_topic: str,
        user_id: str,
    ) -> List[SourceRecord]:
        """Fetch profile fact by topic_subtopic metadata key."""

        meta_key = f"{topic}_{sub_topic}".replace(" ", "_").lower()

        filters = {
            "main_content": meta_key,
            "user_id": user_id,
            "domain": "profile",
        }

        results = self.vector_store.search_by_metadata(
            filters=filters,
            top_k=5,
        )

        records = []
        for r in results:
            records.append(SourceRecord(
                domain="profile",
                content=r.content,
                score=r.score,
                metadata={
                    "id": r.id,
                    "topic": topic,
                    "sub_topic": sub_topic,
                    **r.metadata,
                },
            ))

        logger.info("  → Profile [%s/%s]: %d results", topic, sub_topic, len(records))
        return records

    # -- Temporal: Neo4j semantic search ───────────────────────────────

    def _search_temporal(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> List[SourceRecord]:
        """Semantic search over temporal events in Neo4j."""

        events = self.neo4j.search_events_by_embedding(
            user_id=user_id,
            query_text=query,
            top_k=top_k,
            similarity_threshold=0.15,
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

            records.append(SourceRecord(
                domain="temporal",
                content=content,
                score=ev.get("similarity_score", 0.0),
                metadata=ev,
            ))

        logger.info("  → Temporal [%s]: %d results", query, len(records))
        return records

    # -- Summary: Pinecone semantic search ─────────────────────────────

    async def _search_summary(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
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
            records.append(SourceRecord(
                domain="summary",
                content=r.content,
                score=r.score,
                metadata={"id": r.id, **r.metadata},
            ))

        logger.info("  → Summary [%s]: %d results", query, len(records))
        return records

    # ------------------------------------------------------------------
    # Profile catalog (tells the LLM what profile keys exist)
    # ------------------------------------------------------------------

    def _fetch_profile_catalog(self, user_id: str) -> List[Dict[str, str]]:
        """Fetch all distinct profile topic/sub_topic pairs for a user.

        We use a metadata-only search with domain=profile to find all
        profile entries, then extract unique topic/subtopic from the
        main_content key.
        """
        try:
            results = self.vector_store.search_by_metadata(
                filters={"user_id": user_id, "domain": "profile"},
                top_k=100,  # get all profile entries
            )
        except Exception as exc:
            logger.warning("Failed to fetch profile catalog: %s", exc)
            return []

        catalog: List[Dict[str, str]] = []
        seen = set()

        for r in results:
            main_content = r.metadata.get("main_content", "")
            if not main_content or main_content in seen:
                continue
            seen.add(main_content)

            # Reverse the key: "work_company" → topic="work", sub_topic="company"
            parts = main_content.split("_", 1)
            if len(parts) == 2:
                catalog.append({
                    "topic": parts[0],
                    "sub_topic": parts[1],
                    "value_preview": r.content[:60],
                })
            else:
                catalog.append({
                    "topic": main_content,
                    "sub_topic": "",
                    "value_preview": r.content[:60],
                })

        return catalog

    def _format_catalog(self, catalog: List[Dict[str, str]]) -> str:
        """Format profile catalog for the system prompt."""
        if not catalog:
            return "(No profile data stored yet)"

        lines = []
        for entry in catalog:
            t = entry["topic"]
            st = entry["sub_topic"]
            preview = entry.get("value_preview", "")
            lines.append(f"  - {t} / {st}  (current: \"{preview}\")")
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
