"""
Weaver — deterministic executor for Judge operations.

Takes a JudgeResult and executes each operation against the appropriate store:
  - profile / summary → Pinecone (vector store)
  - temporal → Neo4j (graph DB)

No LLM involved — just structured execution with guard rails.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union, cast

from src.schemas.judge import (
    JudgeDomain,
    JudgeResult,
    Operation,
    OperationType,
)
from src.schemas.weaver import ExecutedOp, OpStatus, WeaverResult
from src.storage.base import BaseVectorStore

logger = logging.getLogger("xmem.weaver")


# ---------------------------------------------------------------------------
# Type aliases for injected callables (Neo4j operations)
# ---------------------------------------------------------------------------
# The pipeline wraps the actual graph_helper functions into these callables
# so the weaver stays decoupled from Neo4j internals.

# async (user_id, date_str, event_data) -> str (new relationship id)
GraphCreateEventFn = Callable[..., Any]

# async (user_id, date_str, event_data) -> bool
GraphUpdateEventFn = Callable[..., Any]

# async (user_id, date_str, event_name) -> bool
GraphDeleteEventFn = Callable[..., Any]


# ---------------------------------------------------------------------------
# Weaver
# ---------------------------------------------------------------------------

class Weaver:
    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embed_fn: Optional[Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]] = None,
        graph_create_event: Optional[GraphCreateEventFn] = None,
        graph_update_event: Optional[GraphUpdateEventFn] = None,
        graph_delete_event: Optional[GraphDeleteEventFn] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embed_fn = embed_fn
        self.graph_create_event = graph_create_event
        self.graph_update_event = graph_update_event
        self.graph_delete_event = graph_delete_event

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute(
        self,
        judge_result: JudgeResult,
        domain: JudgeDomain,
        user_id: str,
    ) -> WeaverResult:
        result = WeaverResult()

        if judge_result.is_empty or not judge_result.has_writes:
            logger.info("Nothing to execute — all NOOPs or empty.")
            return result

        # Buffer for batched ADD operations (vector domains only)
        add_ops_buffer: List[Operation] = []

        for op in judge_result.operations:
            # Check if this operation is a candidate for batching
            # We only batch ADDs for vector domains (Profile, Summary, Image)
            is_vector_domain = domain != JudgeDomain.TEMPORAL
            is_add_op = op.type == OperationType.ADD

            if is_vector_domain and is_add_op:
                add_ops_buffer.append(op)
                continue

            # If we hit a non-batchable op, flush the buffer first
            if add_ops_buffer:
                batch_results = await self._vector_add_batch(add_ops_buffer, domain, user_id)
                result.executed.extend(batch_results)
                add_ops_buffer = []

            # Execute the current operation individually
            executed = await self._execute_one(op, domain, user_id)
            result.executed.append(executed)

        # Flush any remaining operations in the buffer
        if add_ops_buffer:
            batch_results = await self._vector_add_batch(add_ops_buffer, domain, user_id)
            result.executed.extend(batch_results)

        self._log_summary(domain, result)
        return result

    # ------------------------------------------------------------------
    # Per-operation dispatch
    # ------------------------------------------------------------------

    async def _execute_one(
        self,
        op: Operation,
        domain: JudgeDomain,
        user_id: str,
    ) -> ExecutedOp:
        # ── Guard rails ──────────────────────────────────────────────
        if op.type == OperationType.NOOP:
            return ExecutedOp(
                type=op.type, status=OpStatus.SKIPPED, content=op.content,
                embedding_id=op.embedding_id,
            )

        if op.type == OperationType.ADD and not op.content:
            logger.warning("ADD with empty content — skipping.")
            return ExecutedOp(
                type=op.type, status=OpStatus.SKIPPED,
                error="ADD requires content",
            )

        if op.type in (OperationType.UPDATE, OperationType.DELETE) and not op.embedding_id:
            logger.warning(
                "%s missing embedding_id — converting to ADD.", op.type.value,
            )
            op = op.model_copy(update={"type": OperationType.ADD, "embedding_id": None})

        # ── Route by domain ──────────────────────────────────────────
        if domain == JudgeDomain.TEMPORAL:
            return await self._execute_temporal(op, user_id)
        else:
            return await self._execute_vector(op, domain, user_id)

    # ------------------------------------------------------------------
    # Profile / Summary → Pinecone
    # ------------------------------------------------------------------

    async def _execute_vector(
        self,
        op: Operation,
        domain: JudgeDomain,
        user_id: str,
    ) -> ExecutedOp:
        if not self.vector_store:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No vector store attached",
            )

        try:
            if op.type == OperationType.ADD:
                return await self._vector_add(op, domain, user_id)
            elif op.type == OperationType.UPDATE:
                return await self._vector_update(op, domain, user_id)
            elif op.type == OperationType.DELETE:
                return await self._vector_delete(op)
            else:
                return ExecutedOp(
                    type=op.type, status=OpStatus.SKIPPED,
                    content=op.content,
                )
        except Exception as exc:
            logger.error("Vector op %s failed: %s", op.type.value, exc)
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, embedding_id=op.embedding_id,
                error=str(exc),
            )

    async def _vector_add(
        self, op: Operation, domain: JudgeDomain, user_id: str,
    ) -> ExecutedOp:
        if not self.embed_fn:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No embed_fn provided",
            )

        embedding = cast(List[float], self.embed_fn(op.content))
        metadata = {"user_id": user_id, "domain": domain.value}

        # Store structured metadata for deterministic lookups
        structured = _extract_structured_metadata(op.content)
        metadata.update(structured)

        ids = self.vector_store.add(
            texts=[op.content],
            embeddings=[embedding],
            metadata=[metadata],
        )
        new_id = ids[0] if ids else None
        return ExecutedOp(
            type=op.type, status=OpStatus.SUCCESS,
            content=op.content, new_id=new_id,
        )

    async def _vector_add_batch(
        self, ops: List[Operation], domain: JudgeDomain, user_id: str,
    ) -> List[ExecutedOp]:
        """Execute a batch of ADD operations for vector store."""
        if not self.vector_store:
            return [
                ExecutedOp(
                    type=op.type, status=OpStatus.FAILED,
                    content=op.content, error="No vector store attached",
                )
                for op in ops
            ]

        if not self.embed_fn:
            return [
                ExecutedOp(
                    type=op.type, status=OpStatus.FAILED,
                    content=op.content, error="No embed_fn provided",
                )
                for op in ops
            ]

        # Filter out empty content to avoid errors
        valid_ops = []
        skipped_results = []
        for op in ops:
            if not op.content:
                logger.warning("ADD with empty content — skipping.")
                skipped_results.append(
                    ExecutedOp(
                        type=op.type, status=OpStatus.SKIPPED,
                        error="ADD requires content",
                    )
                )
            else:
                valid_ops.append(op)

        if not valid_ops:
            return skipped_results

        contents = [op.content for op in valid_ops]

        try:
            # Batch embedding
            embeddings = cast(List[List[float]], self.embed_fn(contents))
        except Exception as exc:
            logger.error("Batch embedding failed: %s", exc)
            return skipped_results + [
                ExecutedOp(
                    type=op.type, status=OpStatus.FAILED,
                    content=op.content, error=f"Embedding failed: {exc}",
                )
                for op in valid_ops
            ]

        # Prepare metadata
        metadatas = []
        for content in contents:
            meta = {"user_id": user_id, "domain": domain.value}
            meta.update(_extract_structured_metadata(content))
            metadatas.append(meta)

        try:
            ids = self.vector_store.add(
                texts=contents,
                embeddings=embeddings,
                metadata=metadatas,
            )
        except Exception as exc:
            logger.error("Vector batch add failed: %s", exc)
            return skipped_results + [
                ExecutedOp(
                    type=op.type, status=OpStatus.FAILED,
                    content=op.content, error=f"Vector add failed: {exc}",
                )
                for op in valid_ops
            ]

        # Map IDs back to operations
        results = []
        for i, op in enumerate(valid_ops):
            new_id = ids[i] if ids and i < len(ids) else None
            results.append(ExecutedOp(
                type=op.type, status=OpStatus.SUCCESS,
                content=op.content, new_id=new_id,
            ))

        return skipped_results + results

    async def _vector_update(
        self, op: Operation, domain: JudgeDomain, user_id: str,
    ) -> ExecutedOp:
        if not self.embed_fn:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No embed_fn provided",
            )

        embedding = self.embed_fn(op.content)
        metadata = {"user_id": user_id, "domain": domain.value}

        # Store structured metadata for deterministic lookups
        structured = _extract_structured_metadata(op.content)
        metadata.update(structured)

        success = self.vector_store.update(
            id=op.embedding_id,
            text=op.content,
            embedding=embedding,
            metadata=metadata,
        )
        if success:
            return ExecutedOp(
                type=op.type, status=OpStatus.SUCCESS,
                content=op.content, embedding_id=op.embedding_id,
            )
        else:
            logger.warning(
                "UPDATE target %s not found — falling back to ADD.", op.embedding_id,
            )
            return await self._vector_add(op, domain, user_id)

    async def _vector_delete(self, op: Operation) -> ExecutedOp:
        success = self.vector_store.delete(ids=[op.embedding_id])
        return ExecutedOp(
            type=op.type,
            status=OpStatus.SUCCESS if success else OpStatus.FAILED,
            embedding_id=op.embedding_id,
        )

    # ------------------------------------------------------------------
    # Temporal → Neo4j
    # ------------------------------------------------------------------

    async def _execute_temporal(
        self, op: Operation, user_id: str,
    ) -> ExecutedOp:
        try:
            if op.type == OperationType.ADD:
                return await self._graph_add(op, user_id)
            elif op.type == OperationType.UPDATE:
                return await self._graph_update(op, user_id)
            elif op.type == OperationType.DELETE:
                return await self._graph_delete(op, user_id)
            else:
                return ExecutedOp(
                    type=op.type, status=OpStatus.SKIPPED,
                    content=op.content,
                )
        except Exception as exc:
            logger.error("Graph op %s failed: %s", op.type.value, exc)
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, embedding_id=op.embedding_id,
                error=str(exc),
            )

    async def _graph_add(self, op: Operation, user_id: str) -> ExecutedOp:
        if not self.graph_create_event:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No graph_create_event provided",
            )

        event_data = _parse_temporal_content(op.content)
        date_str = event_data.pop("date", "")
        if not date_str:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No date found in temporal content",
            )

        await self.graph_create_event(
            user_id=user_id, date_str=date_str, event_data=event_data,
        )
        return ExecutedOp(
            type=op.type, status=OpStatus.SUCCESS, content=op.content,
        )

    async def _graph_update(self, op: Operation, user_id: str) -> ExecutedOp:
        if not self.graph_update_event:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No graph_update_event provided",
            )

        event_data = _parse_temporal_content(op.content)
        date_str = event_data.pop("date", "")
        if not date_str:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No date found in temporal content",
            )

        await self.graph_update_event(
            user_id=user_id, date_str=date_str, event_data=event_data,
        )
        return ExecutedOp(
            type=op.type, status=OpStatus.SUCCESS,
            content=op.content, embedding_id=op.embedding_id,
        )

    async def _graph_delete(self, op: Operation, user_id: str) -> ExecutedOp:
        if not self.graph_delete_event:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                embedding_id=op.embedding_id,
                error="No graph_delete_event provided",
            )

        await self.graph_delete_event(
            user_id=user_id, embedding_id=op.embedding_id,
        )
        return ExecutedOp(
            type=op.type, status=OpStatus.SUCCESS,
            embedding_id=op.embedding_id,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_summary(self, domain: JudgeDomain, result: WeaverResult) -> None:
        logger.info("=" * 50)
        logger.info(
            "Weaver Done (%s): %d executed, %d succeeded, %d skipped, %d failed",
            domain.value, result.total, result.succeeded,
            result.skipped, result.failed,
        )
        for i, ex in enumerate(result.executed, 1):
            status_icon = {"success": "✓", "skipped": "⊘", "failed": "✗"}[ex.status.value]
            preview = (ex.content[:40] + "...") if len(ex.content) > 40 else ex.content
            logger.info(
                "  %d. %s [%s] %s", i, status_icon, ex.type.value, preview,
            )
            if ex.error:
                logger.info("     error: %s", ex.error)
        logger.info("=" * 50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_structured_metadata(content: str) -> Dict[str, str]:
    """Extract structured metadata from profile/summary content.

    For profile items formatted as ``topic / sub_topic = memo``:
        main_content  → ``topic_subtopic``   (lowered, underscored)
        subcontent    → ``memo``             (the actual fact)

    For content that doesn't match the profile format the full string is
    stored as ``subcontent`` and ``main_content`` is left empty so the
    record can still be retrieved via semantic search.
    """
    result: Dict[str, str] = {}

    if " = " in content and " / " in content.split(" = ", 1)[0]:
        # Profile format: "topic / sub_topic = memo"
        key_part, memo = content.split(" = ", 1)
        topic_subtopic = (
            key_part.strip()
            .replace(" / ", "_")
            .replace(" ", "_")
            .lower()
        )
        result["main_content"] = topic_subtopic
        result["subcontent"] = memo.strip()
    else:
        # Summary or free-text — no structured key available
        result["main_content"] = ""
        result["subcontent"] = content.strip()

    return result


def _parse_temporal_content(content: str) -> Dict[str, str]:
    """Parse 'date | event_name | desc' back into a dict."""
    parts = [p.strip() for p in content.split("|")]
    result: Dict[str, str] = {}
    if len(parts) >= 1:
        result["date"] = parts[0]
    if len(parts) >= 2:
        result["event_name"] = parts[1]
    if len(parts) >= 3:
        result["desc"] = parts[2]
    return result
