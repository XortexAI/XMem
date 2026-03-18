"""
Weaver — deterministic executor for Judge operations.

Takes a JudgeResult and executes each operation against the appropriate store:
  - profile / summary → Pinecone (vector store)
  - temporal → Neo4j (graph DB)
  - code → Pinecone (annotations namespace) + Neo4j (Annotation node + ANNOTATES edge)
  - snippet → Pinecone (user snippets namespace) — personal code knowledge, no graph

No LLM involved — just structured execution with guard rails.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

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

# async (org_id, content, annotation_type, ...) -> str (annotation_id)
GraphCreateAnnotationFn = Callable[..., Any]


# ---------------------------------------------------------------------------
# Weaver
# ---------------------------------------------------------------------------

class Weaver:
    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        graph_create_event: Optional[GraphCreateEventFn] = None,
        graph_update_event: Optional[GraphUpdateEventFn] = None,
        graph_delete_event: Optional[GraphDeleteEventFn] = None,
        code_vector_store: Optional[BaseVectorStore] = None,
        graph_create_annotation: Optional[GraphCreateAnnotationFn] = None,
        snippet_vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embed_fn = embed_fn
        self.graph_create_event = graph_create_event
        self.graph_update_event = graph_update_event
        self.graph_delete_event = graph_delete_event
        self.code_vector_store = code_vector_store
        self.graph_create_annotation = graph_create_annotation
        self.snippet_vector_store = snippet_vector_store

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

        # Optimization: Batch vector operations if possible
        if domain not in (JudgeDomain.TEMPORAL, JudgeDomain.CODE, JudgeDomain.SNIPPET) and self.vector_store:
            batched_executed = await self._execute_batched_vector(judge_result.operations, domain, user_id)
            result.executed.extend(batched_executed)
        else:
            import asyncio
            # ⚡ Bolt Optimization:
            # What: Execute non-batched operations concurrently using asyncio.gather instead of sequentially.
            # Why: These operations (like Neo4j writes) are I/O bound. Running them sequentially incurs N * latency.
            # Impact: Significantly reduces latency for non-batched domains (Temporal, Code, Snippet).
            executed_ops = await asyncio.gather(
                *(self._execute_one(op, domain, user_id) for op in judge_result.operations)
            )
            result.executed.extend(executed_ops)

        self._log_summary(domain, result)
        return result

    async def _execute_batched_vector(
        self,
        operations: List[Operation],
        domain: JudgeDomain,
        user_id: str,
    ) -> List[ExecutedOp]:
        """Batch ADD and DELETE operations to reduce vector store round-trips."""
        executed_ops: List[ExecutedOp] = []

        # Batch containers
        add_batch_ops: List[Operation] = []
        delete_batch_ops: List[Operation] = []

        if not self.embed_fn:
            logger.error("Cannot batch execute vector ops: No embed_fn provided")
            return [
                ExecutedOp(type=op.type, status=OpStatus.FAILED, error="No embed_fn provided")
                for op in operations
            ]

        async def flush_add_batch():
            if not add_batch_ops:
                return

            # Prepare data for batch add
            valid_ops = []
            texts = []
            embeddings = []
            metadatas = []

            import asyncio
            from functools import partial
            loop = asyncio.get_running_loop()

            async def _process_op(op: Operation):
                if not op.content:
                    logger.warning("ADD with empty content — skipping.")
                    return ExecutedOp(
                        type=op.type, status=OpStatus.SKIPPED,
                        error="ADD requires content",
                    ), None, None, None

                try:
                    # ⚡ Bolt Optimization:
                    # What: Run synchronous embed_fn in an executor concurrently.
                    # Why: embed_fn makes synchronous network calls (or heavy CPU work). Running it sequentially blocks the event loop.
                    # Impact: Prevents event loop blocking and reduces overall embedding latency for the batch.
                    emb = await loop.run_in_executor(None, partial(self.embed_fn, op.content))
                    meta = {"user_id": user_id, "domain": domain.value}
                    meta.update(_extract_structured_metadata(op.content))
                    return None, op, emb, meta
                except Exception as exc:
                    logger.error("Embedding generation failed for ADD: %s", exc)
                    return ExecutedOp(
                        type=op.type, status=OpStatus.FAILED,
                        content=op.content, error=str(exc)
                    ), None, None, None

            results = await asyncio.gather(*(_process_op(op) for op in add_batch_ops))

            for err_op, op, emb, meta in results:
                if err_op:
                    executed_ops.append(err_op)
                else:
                    valid_ops.append(op)
                    texts.append(op.content)
                    embeddings.append(emb)
                    metadatas.append(meta)

            if valid_ops:
                try:
                    ids = self.vector_store.add(
                        texts=texts,
                        embeddings=embeddings,
                        metadata=metadatas,
                    )
                    # Map IDs back to ops
                    for op, new_id in zip(valid_ops, ids):
                        executed_ops.append(ExecutedOp(
                            type=op.type, status=OpStatus.SUCCESS,
                            content=op.content, new_id=new_id,
                        ))
                except Exception as exc:
                    logger.error("Vector batch ADD failed: %s", exc)
                    for op in valid_ops:
                        executed_ops.append(ExecutedOp(
                            type=op.type, status=OpStatus.FAILED,
                            content=op.content, error=str(exc)
                        ))

            add_batch_ops.clear()

        async def flush_delete_batch():
            if not delete_batch_ops:
                return

            valid_ops = []
            ids_to_delete = []

            for op in delete_batch_ops:
                if not op.embedding_id:
                     logger.warning("DELETE missing embedding_id — skipping.")
                     executed_ops.append(ExecutedOp(
                        type=op.type, status=OpStatus.FAILED,
                         error="DELETE missing embedding_id"
                     ))
                     continue

                valid_ops.append(op)
                ids_to_delete.append(op.embedding_id)

            if valid_ops:
                try:
                    success = self.vector_store.delete(ids=ids_to_delete)
                    status = OpStatus.SUCCESS if success else OpStatus.FAILED
                    for op in valid_ops:
                        executed_ops.append(ExecutedOp(
                            type=op.type, status=status,
                            embedding_id=op.embedding_id
                        ))
                except Exception as exc:
                    logger.error("Vector batch DELETE failed: %s", exc)
                    for op in valid_ops:
                        executed_ops.append(ExecutedOp(
                            type=op.type, status=OpStatus.FAILED,
                            embedding_id=op.embedding_id, error=str(exc)
                        ))

            delete_batch_ops.clear()

        # Iterate and group
        for op in operations:
            # Pre-processing
            current_op = op

            if current_op.type == OperationType.NOOP:
                await flush_add_batch()
                await flush_delete_batch()
                executed_ops.append(ExecutedOp(
                    type=current_op.type, status=OpStatus.SKIPPED, content=current_op.content,
                    embedding_id=current_op.embedding_id,
                ))
                continue

            # Convert missing ID UPDATE/DELETE to ADD
            if current_op.type in (OperationType.UPDATE, OperationType.DELETE) and not current_op.embedding_id:
                logger.warning(
                    "%s missing embedding_id — converting to ADD.", current_op.type.value,
                )
                current_op = current_op.model_copy(update={"type": OperationType.ADD, "embedding_id": None})

            # Now route to batches
            if current_op.type == OperationType.ADD:
                await flush_delete_batch()
                add_batch_ops.append(current_op)

            elif current_op.type == OperationType.DELETE:
                await flush_add_batch()
                delete_batch_ops.append(current_op)

            elif current_op.type == OperationType.UPDATE:
                await flush_add_batch()
                await flush_delete_batch()
                # Execute individual UPDATE
                executed_ops.append(await self._execute_one(current_op, domain, user_id))

            else:
                 # Fallback for unknown types
                 await flush_add_batch()
                 await flush_delete_batch()
                 executed_ops.append(await self._execute_one(current_op, domain, user_id))

        # Final flush
        await flush_add_batch()
        await flush_delete_batch()

        return executed_ops

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
        elif domain == JudgeDomain.CODE:
            return await self._execute_code(op, user_id)
        elif domain == JudgeDomain.SNIPPET:
            return await self._execute_snippet(op, user_id)
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

        embedding = self.embed_fn(op.content)
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
        logger.info("_graph_add: content=%s | date_str=%s | event_data=%s", op.content, date_str, event_data)
        if not date_str:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No date found in temporal content",
            )

        await self.graph_create_event(
            user_id=user_id, date_str=date_str, event_data=event_data,
        )
        logger.info("_graph_add: SUCCESS for user=%s date=%s", user_id, date_str)
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
    # Code → Pinecone (annotations namespace) + Neo4j (Annotation node)
    # ------------------------------------------------------------------

    async def _execute_code(
        self, op: Operation, user_id: str,
    ) -> ExecutedOp:
        try:
            if op.type == OperationType.ADD:
                return await self._code_add(op, user_id)
            elif op.type == OperationType.UPDATE:
                return await self._code_update(op, user_id)
            elif op.type == OperationType.DELETE:
                return await self._code_delete(op)
            else:
                return ExecutedOp(
                    type=op.type, status=OpStatus.SKIPPED,
                    content=op.content,
                )
        except Exception as exc:
            logger.error("Code op %s failed: %s", op.type.value, exc)
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, embedding_id=op.embedding_id,
                error=str(exc),
            )

    async def _code_add(self, op: Operation, user_id: str) -> ExecutedOp:
        if not self.embed_fn:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No embed_fn provided",
            )

        parsed = _parse_code_annotation_content(op.content)
        store = self.code_vector_store or self.vector_store
        if not store:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No vector store for code domain",
            )

        import asyncio
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, self.embed_fn, op.content)
        metadata: Dict[str, Any] = {
            "user_id": user_id,
            "domain": "code",
            "annotation_type": parsed.get("annotation_type", ""),
            "target_symbol": parsed.get("target_symbol", ""),
            "target_file": parsed.get("target_file", ""),
            "repo": parsed.get("repo", ""),
            "severity": parsed.get("severity", ""),
        }

        from functools import partial
        ids = await loop.run_in_executor(
            None,
            partial(
                store.add,
                texts=[op.content],
                embeddings=[embedding],
                metadata=[metadata],
            )
        )
        new_id = ids[0] if ids else None

        # Also create Neo4j annotation node + ANNOTATES edge
        if self.graph_create_annotation:
            try:
                await self.graph_create_annotation(
                    content=parsed.get("content", op.content),
                    annotation_type=parsed.get("annotation_type", "explanation"),
                    severity=parsed.get("severity"),
                    author_id=user_id,
                    repo=parsed.get("repo"),
                    target_file=parsed.get("target_file"),
                    target_symbol=parsed.get("target_symbol"),
                )
            except Exception as exc:
                logger.warning("Neo4j annotation creation failed (vector succeeded): %s", exc)

        return ExecutedOp(
            type=op.type, status=OpStatus.SUCCESS,
            content=op.content, new_id=new_id,
        )

    async def _code_update(self, op: Operation, user_id: str) -> ExecutedOp:
        if not self.embed_fn:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No embed_fn provided",
            )

        store = self.code_vector_store or self.vector_store
        if not store:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No vector store for code domain",
            )

        parsed = _parse_code_annotation_content(op.content)
        import asyncio
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, self.embed_fn, op.content)
        metadata: Dict[str, Any] = {
            "user_id": user_id,
            "domain": "code",
            "annotation_type": parsed.get("annotation_type", ""),
            "target_symbol": parsed.get("target_symbol", ""),
            "target_file": parsed.get("target_file", ""),
            "repo": parsed.get("repo", ""),
            "severity": parsed.get("severity", ""),
        }

        from functools import partial
        success = await loop.run_in_executor(
            None,
            partial(
                store.update,
                id=op.embedding_id,
                text=op.content,
                embedding=embedding,
                metadata=metadata,
            )
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
            return await self._code_add(op, user_id)

    async def _code_delete(self, op: Operation) -> ExecutedOp:
        store = self.code_vector_store or self.vector_store
        if not store:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                embedding_id=op.embedding_id,
                error="No vector store for code domain",
            )
        import asyncio
        loop = asyncio.get_running_loop()
        from functools import partial
        success = await loop.run_in_executor(None, partial(store.delete, ids=[op.embedding_id]))
        return ExecutedOp(
            type=op.type,
            status=OpStatus.SUCCESS if success else OpStatus.FAILED,
            embedding_id=op.embedding_id,
        )

    # ------------------------------------------------------------------
    # Snippet → Pinecone (user snippets namespace, no Neo4j)
    # ------------------------------------------------------------------

    async def _execute_snippet(
        self, op: Operation, user_id: str,
    ) -> ExecutedOp:
        try:
            if op.type == OperationType.ADD:
                return await self._snippet_add(op, user_id)
            elif op.type == OperationType.UPDATE:
                return await self._snippet_update(op, user_id)
            elif op.type == OperationType.DELETE:
                return await self._snippet_delete(op)
            else:
                return ExecutedOp(
                    type=op.type, status=OpStatus.SKIPPED,
                    content=op.content,
                )
        except Exception as exc:
            logger.error("Snippet op %s failed: %s", op.type.value, exc)
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, embedding_id=op.embedding_id,
                error=str(exc),
            )

    async def _snippet_add(self, op: Operation, user_id: str) -> ExecutedOp:
        if not self.embed_fn:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No embed_fn provided",
            )

        store = self.snippet_vector_store or self.vector_store
        if not store:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No vector store for snippet domain",
            )

        parsed = _parse_snippet_content(op.content)
        searchable = parsed.get("content", op.content)
        import asyncio
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, self.embed_fn, searchable)

        metadata: Dict[str, Any] = {
            "user_id": user_id,
            "domain": "snippet",
            "code_snippet": parsed.get("code_snippet", ""),
            "language": parsed.get("language", ""),
            "snippet_type": parsed.get("snippet_type", "algorithm"),
            "tags": parsed.get("tags", ""),
            "source": "chat",
        }

        from functools import partial
        ids = await loop.run_in_executor(
            None,
            partial(
                store.add,
                texts=[searchable],
                embeddings=[embedding],
                metadata=[metadata],
            )
        )
        new_id = ids[0] if ids else None
        return ExecutedOp(
            type=op.type, status=OpStatus.SUCCESS,
            content=op.content, new_id=new_id,
        )

    async def _snippet_update(self, op: Operation, user_id: str) -> ExecutedOp:
        if not self.embed_fn:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No embed_fn provided",
            )

        store = self.snippet_vector_store or self.vector_store
        if not store:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                content=op.content, error="No vector store for snippet domain",
            )

        parsed = _parse_snippet_content(op.content)
        searchable = parsed.get("content", op.content)
        import asyncio
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, self.embed_fn, searchable)

        metadata: Dict[str, Any] = {
            "user_id": user_id,
            "domain": "snippet",
            "code_snippet": parsed.get("code_snippet", ""),
            "language": parsed.get("language", ""),
            "snippet_type": parsed.get("snippet_type", "algorithm"),
            "tags": parsed.get("tags", ""),
            "source": "chat",
        }

        from functools import partial
        success = await loop.run_in_executor(
            None,
            partial(
                store.update,
                id=op.embedding_id,
                text=searchable,
                embedding=embedding,
                metadata=metadata,
            )
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
            return await self._snippet_add(op, user_id)

    async def _snippet_delete(self, op: Operation) -> ExecutedOp:
        store = self.snippet_vector_store or self.vector_store
        if not store:
            return ExecutedOp(
                type=op.type, status=OpStatus.FAILED,
                embedding_id=op.embedding_id,
                error="No vector store for snippet domain",
            )
        import asyncio
        loop = asyncio.get_running_loop()
        from functools import partial
        success = await loop.run_in_executor(None, partial(store.delete, ids=[op.embedding_id]))
        return ExecutedOp(
            type=op.type,
            status=OpStatus.SUCCESS if success else OpStatus.FAILED,
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
    """Parse 'date | event_name | desc | year | time | date_expression' back into a dict."""
    parts = [p.strip() for p in content.split("|")]
    result: Dict[str, str] = {}
    if len(parts) >= 1:
        result["date"] = parts[0]
    if len(parts) >= 2:
        result["event_name"] = parts[1]
    if len(parts) >= 3:
        result["desc"] = parts[2]
    if len(parts) >= 4 and parts[3]:
        result["year"] = parts[3]
    if len(parts) >= 5 and parts[4]:
        result["time"] = parts[4]
    if len(parts) >= 6 and parts[5]:
        result["date_expression"] = parts[5]
    return result


def _parse_snippet_content(content: str) -> Dict[str, str]:
    """Parse snippet content formatted as pipe-delimited fields.

    Expected format (from _node_extract_snippet):
        content | code_snippet | language | snippet_type | tags

    Falls back gracefully if the content doesn't match.
    """
    parts = [p.strip() for p in content.split(" | ")]
    result: Dict[str, str] = {}

    if len(parts) >= 5:
        result["content"] = parts[0]
        result["code_snippet"] = parts[1]
        result["language"] = parts[2]
        result["snippet_type"] = parts[3]
        result["tags"] = parts[4]
    elif len(parts) >= 3:
        result["content"] = parts[0]
        result["code_snippet"] = parts[1]
        result["language"] = parts[2]
        result["snippet_type"] = "algorithm"
        result["tags"] = ""
    else:
        result["content"] = content
        result["code_snippet"] = ""
        result["language"] = ""
        result["snippet_type"] = "algorithm"
        result["tags"] = ""

    return result


def _parse_code_annotation_content(content: str) -> Dict[str, str]:
    """Parse code annotation content formatted as pipe-delimited fields.

    Expected format (from judge):
        annotation_type | target_symbol | target_file | repo | severity | content

    Falls back gracefully if the content doesn't match the expected format
    (treats the entire string as the annotation content).
    """
    parts = [p.strip() for p in content.split("|")]
    result: Dict[str, str] = {}

    if len(parts) >= 6:
        result["annotation_type"] = parts[0] or "explanation"
        result["target_symbol"] = parts[1]
        result["target_file"] = parts[2]
        result["repo"] = parts[3]
        result["severity"] = parts[4]
        result["content"] = parts[5]
    elif len(parts) >= 2:
        result["annotation_type"] = parts[0] or "explanation"
        result["content"] = " | ".join(parts[1:])
    else:
        result["content"] = content
        result["annotation_type"] = "explanation"

    return result
