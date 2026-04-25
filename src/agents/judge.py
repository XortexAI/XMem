"""
Judge Agent — decides ADD / UPDATE / DELETE / NOOP for incoming memory data.

Works across three domains (profile, temporal, summary).  For each new item
it receives, the judge:

1. Formats the item as a search query appropriate for the domain.
2. Fetches similar existing records:
   - profile → Pinecone (deterministic metadata filter on topic_subtopic)
   - summary → Pinecone (semantic similarity via vector store)
   - temporal → Neo4j (graph DB, via injected search callable)
3. Sends both the new items and retrieved neighbours to the LLM.
4. Parses the LLM's JSON response into a list of typed Operation objects.

The judge ONLY decides — it never performs any writes itself.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.prompts.judge import build_system_prompt, pack_judge_query
from src.schemas.judge import (
    JudgeDomain,
    JudgeResult,
    Operation,
    OperationType,
)
from src.storage.base import BaseVectorStore, SearchResult


# ---------------------------------------------------------------------------
# Formatting helpers (domain → search query / display string)
# ---------------------------------------------------------------------------

def _profile_items_to_strings(facts: List[dict]) -> List[str]:
    return [
        f"{f.get('topic', '')} / {f.get('sub_topic', '')} = {f.get('memo', '')}"
        for f in facts
    ]


def _temporal_item_to_string(event: dict) -> str:
    # Format: date | event_name | desc | year | time | date_expression
    # All 6 fields are always emitted (even if empty) so the parser
    # can positionally reconstruct the full dict.
    # str() ensures non-string values (e.g. year as int) don't break join.
    parts = [
        str(event.get("date", "") or ""),
        str(event.get("event_name", "") or ""),
        str(event.get("desc", "") or ""),
        str(event.get("year", "") or ""),
        str(event.get("time", "") or ""),
        str(event.get("date_expression", "") or ""),
    ]
    return " | ".join(parts)


def _image_item_to_string(observation: dict) -> str:
    return f"{observation.get('category', '')}: {observation.get('description', '')}"


def _format_similar_block(
    items_strings: List[str],
    matches_per_item: Dict[str, List[SearchResult]],
) -> str:
    if not matches_per_item:
        return "(No similar records found — store is empty or search returned nothing)"

    lines: list[str] = []
    for item_str in items_strings:
        matches = matches_per_item.get(item_str, [])
        if matches:
            lines.append(f'For item: "{item_str}"')
            for m in matches:
                lines.append(
                    f'  - ID: {m.id} | Score: {m.score:.2f} | "{m.content}"'
                )
        else:
            lines.append(f'For item: "{item_str}"')
            lines.append("  - (no similar records)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Type alias for the Neo4j event search callable that the pipeline injects.
#
# Signature:
#   async (event_name: str, user_id: str, top_k: int) -> List[SearchResult]
#
# The pipeline creates a small wrapper around graph_helper that returns
# SearchResult objects so the judge doesn't need to know about Neo4j.
# ---------------------------------------------------------------------------

GraphEventSearchFn = Callable[..., Any]


# ---------------------------------------------------------------------------
# JudgeAgent
# ---------------------------------------------------------------------------

class JudgeAgent(BaseAgent):
    def __init__(
        self,
        model: BaseChatModel,
        vector_store: Optional[BaseVectorStore] = None,
        graph_event_search: Optional[GraphEventSearchFn] = None,
        top_k: int = 1,
    ) -> None:
        super().__init__(
            model=model,
            name="judge",
            system_prompt=build_system_prompt(),
        )
        self.vector_store = vector_store
        self.graph_event_search = graph_event_search
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def arun(self, state: Dict[str, Any]) -> JudgeResult:
        domain_str = state.get("domain", "")
        try:
            domain = JudgeDomain(domain_str)
        except ValueError:
            self.logger.error("Unknown judge domain: %s", domain_str)
            return JudgeResult()

        new_items: list = state.get("new_items", [])
        user_id: str = state.get("user_id", "")

        if not new_items:
            self.logger.debug("No new items — returning empty result.")
            return JudgeResult()

        # 1. Convert items to searchable strings
        items_strings = self._items_to_strings(domain, new_items)

        # 2. Fetch similar existing records (domain-aware)
        matches_per_item = await self._fetch_similar(
            items_strings,
            new_items=new_items,
            user_id=user_id,
            domain=domain,
        )

        # 3. Build the prompt
        new_items_block = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(items_strings)
        )
        similar_block = _format_similar_block(items_strings, matches_per_item)
        user_message = pack_judge_query(new_items_block, similar_block, domain.value)

        # 4. Call LLM
        messages = self._build_messages(user_message)
        raw_content = await self._call_model(messages)

        # 5. Parse JSON response
        result = self._parse_response(raw_content, items_strings)

        # 6. Log
        self._log_result(domain, result)

        return result

    async def arun_deterministic(self, state: Dict[str, Any]) -> JudgeResult:
        """Build operations without an LLM for structured domains.

        Profile and temporal extraction already returns normalized structured
        data. For those domains, exact-key lookups are enough to decide
        add/update/no-op without paying for a second model call.
        """
        domain_str = state.get("domain", "")
        try:
            domain = JudgeDomain(domain_str)
        except ValueError:
            self.logger.error("Unknown deterministic judge domain: %s", domain_str)
            return JudgeResult()

        new_items: list = state.get("new_items", [])
        user_id: str = state.get("user_id", "")
        if not new_items:
            return JudgeResult()

        if domain == JudgeDomain.PROFILE:
            result = await self._deterministic_profile(new_items, user_id)
        elif domain == JudgeDomain.TEMPORAL:
            result = await self._deterministic_temporal(new_items, user_id)
        else:
            self.logger.warning(
                "Deterministic judge unsupported for %s; falling back to LLM judge.",
                domain.value,
            )
            return await self.arun(state)

        self._log_result(domain, result)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _items_to_strings(
        self, domain: JudgeDomain, items: list
    ) -> List[str]:
        if domain == JudgeDomain.PROFILE:
            return _profile_items_to_strings(items)
        elif domain == JudgeDomain.TEMPORAL:
            if isinstance(items, dict):
                return [_temporal_item_to_string(items)]
            return [_temporal_item_to_string(e) for e in items]
        elif domain == JudgeDomain.IMAGE:
            return [_image_item_to_string(i) for i in items]
        else:  # SUMMARY
            return [str(s) for s in items]

    async def _fetch_similar(
        self,
        items_strings: List[str],
        new_items: list,
        user_id: str,
        domain: JudgeDomain,
    ) -> Dict[str, List[SearchResult]]:
        if domain == JudgeDomain.TEMPORAL:
            return await self._fetch_similar_temporal(items_strings, new_items, user_id)
        else:
            return await self._fetch_similar_vector(items_strings, new_items, user_id, domain)

    # -- Profile / Summary: Pinecone (vector store) -----------------------

    async def _fetch_similar_vector(
        self,
        items_strings: List[str],
        new_items: list,
        user_id: str,
        domain: JudgeDomain,
    ) -> Dict[str, List[SearchResult]]:
        if not self.vector_store:
            self.logger.debug("No vector store attached — skipping similarity search.")
            return {}

        # Profile domain: use deterministic metadata lookup
        if domain == JudgeDomain.PROFILE:
            return await self._fetch_similar_profile_metadata(
                items_strings, new_items, user_id,
            )

        # Summary / other: parallel semantic search across all items
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        filters["domain"] = domain.value

        async def _search_one(item_str: str) -> tuple[str, List[SearchResult]]:
            try:
                results = await self._search_vector_store(
                    query_text=item_str,
                    filters=filters if filters else None,
                )
                return item_str, results
            except Exception as exc:
                self.logger.warning(
                    "Vector search failed for '%s': %s", item_str[:60], exc
                )
                return item_str, []

        pairs = await asyncio.gather(*(_search_one(s) for s in items_strings))
        return dict(pairs)

    # -- Profile: deterministic metadata lookup ----------------------------

    async def _fetch_similar_profile_metadata(
        self,
        items_strings: List[str],
        new_items: list,
        user_id: str,
    ) -> Dict[str, List[SearchResult]]:
        """Fetch existing profile records by exact topic_subtopic match (parallel).

        Builds a metadata key ``main_content = "topic_subtopic"`` from the
        incoming item and queries Pinecone with a metadata-only filter.
        This is faster and more reliable than semantic similarity.
        """
        search_fn = getattr(self.vector_store, "search_by_metadata", None)

        async def _lookup_one(idx: int, item_str: str) -> tuple[str, List[SearchResult]]:
            try:
                item = new_items[idx] if idx < len(new_items) else {}
                meta_key = _build_profile_metadata_key(item)

                if not meta_key:
                    return item_str, []

                filters: Dict[str, Any] = {"main_content": meta_key}
                if user_id:
                    filters["user_id"] = user_id
                filters["domain"] = JudgeDomain.PROFILE.value

                if search_fn is not None:
                    # search_by_metadata is sync — run in thread pool
                    results = await asyncio.to_thread(
                        search_fn, filters=filters, top_k=self.top_k,
                    )
                    return item_str, results if results else []
                else:
                    self.logger.debug(
                        "Vector store has no search_by_metadata — "
                        "falling back to semantic search for profile."
                    )
                    results = await self._search_vector_store(
                        query_text=item_str,
                        filters={"user_id": user_id, "domain": "profile"} if user_id else None,
                    )
                    return item_str, results
            except Exception as exc:
                self.logger.warning(
                    "Profile metadata search failed for '%s': %s",
                    item_str[:60], exc,
                )
                return item_str, []

        pairs = await asyncio.gather(
            *(_lookup_one(i, s) for i, s in enumerate(items_strings))
        )
        return dict(pairs)

    # -- Semantic search fallback (summary domain) -------------------------

    async def _search_vector_store(
        self,
        query_text: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Semantic search via the vector store's search_by_text method."""
        if not self.vector_store:
            return []

        search_fn = getattr(self.vector_store, "search_by_text", None)
        if search_fn is not None:
            return await search_fn(query_text, top_k=self.top_k, filters=filters)

        self.logger.debug(
            "Vector store has no search_by_text — skipping search for this item."
        )
        return []

    # -- Temporal: Neo4j (graph DB) ----------------------------------------

    async def _fetch_similar_temporal(
        self,
        items_strings: List[str],
        new_items: list,
        user_id: str,
    ) -> Dict[str, List[SearchResult]]:
        if not self.graph_event_search:
            self.logger.debug("No graph_event_search provided — skipping Neo4j lookup.")
            return {}

        async def _lookup_one(idx: int, item_str: str) -> tuple[str, List[SearchResult]]:
            try:
                event = new_items[idx] if idx < len(new_items) else {}
                event_name = event.get("event_name", "") if isinstance(event, dict) else ""

                if not event_name:
                    return item_str, []

                results = await self.graph_event_search(
                    event_name=event_name,
                    user_id=user_id,
                    top_k=self.top_k,
                )
                return item_str, results if results else []
            except Exception as exc:
                self.logger.warning(
                    "Neo4j event search failed for '%s': %s", item_str[:60], exc
                )
                return item_str, []

        pairs = await asyncio.gather(
            *(_lookup_one(i, s) for i, s in enumerate(items_strings))
        )
        return dict(pairs)

    # -- Deterministic operation builders ---------------------------------

    async def _deterministic_profile(
        self, new_items: list, user_id: str,
    ) -> JudgeResult:
        new_items = _dedupe_profile_items(new_items)
        item_strings = self._items_to_strings(JudgeDomain.PROFILE, new_items)
        matches_per_item = await self._fetch_similar_profile_metadata(
            item_strings, new_items, user_id,
        )

        operations: list[Operation] = []
        for item_str in item_strings:
            match = _first_match(matches_per_item.get(item_str, []))
            if match is None:
                operations.append(Operation(
                    type=OperationType.ADD,
                    content=item_str,
                    reason="No profile record with the same topic/sub_topic.",
                ))
                continue

            incoming_memo = _profile_memo_from_content(item_str)
            existing_memo = _profile_memo_from_match(match)
            if _norm_text(incoming_memo) == _norm_text(existing_memo):
                operations.append(Operation(
                    type=OperationType.NOOP,
                    content=item_str,
                    embedding_id=match.id,
                    reason="Existing profile fact is unchanged.",
                ))
            else:
                operations.append(Operation(
                    type=OperationType.UPDATE,
                    content=item_str,
                    embedding_id=match.id,
                    reason="Existing profile fact has new content.",
                ))

        return JudgeResult(operations=operations, confidence=1.0)

    async def _deterministic_temporal(
        self, new_items: list, user_id: str,
    ) -> JudgeResult:
        new_items = _dedupe_temporal_items(new_items)
        item_strings = self._items_to_strings(JudgeDomain.TEMPORAL, new_items)
        matches_per_item = await self._fetch_similar_temporal(
            item_strings, new_items, user_id,
        )

        operations: list[Operation] = []
        for item_str in item_strings:
            match = _first_match(matches_per_item.get(item_str, []))
            if match is None:
                operations.append(Operation(
                    type=OperationType.ADD,
                    content=item_str,
                    reason="No temporal event with the same event_name.",
                ))
                continue

            incoming = _temporal_fields_from_content(item_str)
            existing = _temporal_fields_from_match(match)
            if _same_temporal_event(incoming, existing):
                operations.append(Operation(
                    type=OperationType.NOOP,
                    content=item_str,
                    embedding_id=match.id,
                    reason="Existing temporal event is unchanged.",
                ))
            elif _norm_text(incoming.get("date")) != _norm_text(existing.get("date")):
                operations.extend([
                    Operation(
                        type=OperationType.DELETE,
                        content=match.content,
                        embedding_id=match.id,
                        reason="Existing temporal event moved to a different date.",
                    ),
                    Operation(
                        type=OperationType.ADD,
                        content=item_str,
                        reason="Re-created temporal event on the updated date.",
                    ),
                ])
            else:
                operations.append(Operation(
                    type=OperationType.UPDATE,
                    content=item_str,
                    embedding_id=match.id,
                    reason="Existing temporal event has new content.",
                ))

        return JudgeResult(operations=operations, confidence=1.0)

    # -- Response parsing --------------------------------------------------

    def _parse_response(
        self, raw: str, items_strings: List[str]
    ) -> JudgeResult:
        try:
            cleaned = raw.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]

            data = json.loads(cleaned.strip())

            operations: list[Operation] = []
            for op_dict in data.get("operations", []):
                op_type_str = op_dict.get("type", "ADD").upper()
                try:
                    op_type = OperationType(op_type_str)
                except ValueError:
                    op_type = OperationType.ADD

                operations.append(
                    Operation(
                        type=op_type,
                        content=op_dict.get("content", ""),
                        embedding_id=op_dict.get("embedding_id"),
                        reason=op_dict.get("reason", ""),
                    )
                )

            confidence = float(data.get("confidence", 0.0))
            return JudgeResult(operations=operations, confidence=confidence)

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            self.logger.error("Failed to parse judge response: %s", exc)
            fallback_ops = [
                Operation(
                    type=OperationType.ADD,
                    content=s,
                    reason="Fallback — JSON parse failed",
                )
                for s in items_strings
            ]
            return JudgeResult(operations=fallback_ops, confidence=0.5)

    def _log_result(self, domain: JudgeDomain, result: JudgeResult) -> None:
        if result.is_empty:
            self.logger.info("No operations produced.")
            return

        self.logger.info("=" * 50)
        self.logger.info("Judge Decision (%s):", domain.value)
        for i, op in enumerate(result.operations, 1):
            preview = (op.content[:50] + "...") if len(op.content) > 50 else op.content
            self.logger.info(
                "  %d. %s  →  %s  (id=%s)",
                i, op.type.value, preview, op.embedding_id or "new",
            )
        self.logger.info("Confidence: %.2f", result.confidence)
        self.logger.info("=" * 50)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_profile_metadata_key(item: dict) -> str:
    """Build the normalized metadata key for a profile item.

    Given ``{"topic": "food", "sub_topic": "preference", "memo": "..."}``,
    returns ``"food_preference"`` — the same format the Weaver stores
    as ``main_content`` in Pinecone metadata.

    Returns empty string if topic or sub_topic is missing.
    """
    if not isinstance(item, dict):
        return ""
    topic = item.get("topic", "").strip()
    sub_topic = item.get("sub_topic", "").strip()
    if not topic or not sub_topic:
        return ""
    return f"{topic}_{sub_topic}".replace(" ", "_").lower()


def _first_match(matches: List[SearchResult]) -> Optional[SearchResult]:
    return matches[0] if matches else None


def _dedupe_profile_items(items: list) -> list:
    deduped: dict[str, dict] = {}
    passthrough: list = []
    for item in items:
        key = _build_profile_metadata_key(item)
        if key:
            deduped[key] = item
        else:
            passthrough.append(item)
    return [*deduped.values(), *passthrough]


def _dedupe_temporal_items(items: list) -> list:
    deduped: dict[str, dict] = {}
    passthrough: list = []
    for item in items:
        if not isinstance(item, dict):
            passthrough.append(item)
            continue
        event_name = _norm_text(item.get("event_name"))
        if event_name:
            deduped[event_name] = item
        else:
            passthrough.append(item)
    return [*deduped.values(), *passthrough]


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _profile_memo_from_content(content: str) -> str:
    if " = " not in content:
        return content
    return content.split(" = ", 1)[1].strip()


def _profile_memo_from_match(match: SearchResult) -> str:
    metadata = match.metadata or {}
    subcontent = metadata.get("subcontent")
    if subcontent is not None:
        return str(subcontent)
    return _profile_memo_from_content(match.content)


def _temporal_fields_from_content(content: str) -> Dict[str, str]:
    parts = [p.strip() for p in content.split("|")]
    keys = ["date", "event_name", "desc", "year", "time", "date_expression"]
    return {key: parts[idx] if idx < len(parts) else "" for idx, key in enumerate(keys)}


def _temporal_fields_from_match(match: SearchResult) -> Dict[str, str]:
    metadata = match.metadata or {}
    if metadata:
        return {
            "date": str(metadata.get("date", "") or ""),
            "event_name": str(metadata.get("event_name", "") or ""),
            "desc": str(metadata.get("desc", "") or ""),
            "year": str(metadata.get("year", "") or ""),
            "time": str(metadata.get("time", "") or ""),
            "date_expression": str(metadata.get("date_expression", "") or ""),
        }
    return _temporal_fields_from_content(match.content)


def _same_temporal_event(incoming: Dict[str, str], existing: Dict[str, str]) -> bool:
    keys = ["date", "event_name", "desc", "year", "time", "date_expression"]
    return all(_norm_text(incoming.get(key)) == _norm_text(existing.get(key)) for key in keys)
