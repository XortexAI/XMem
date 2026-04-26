"""Enterprise chat orchestration for code, annotations, and memory."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from src.enterprise.annotation_service import EnterpriseAnnotationService
from src.enterprise.memory_service import EnterpriseMemoryService
from src.enterprise.schemas import (
    EnterpriseChatContext,
    EnterpriseMemoryContext,
)

if TYPE_CHECKING:
    from src.storage.team_annotation_store import TeamAnnotationStore

logger = logging.getLogger("xmem.enterprise.chat_orchestrator")

CodePipelineFactory = Callable[[str, str, Optional[str]], Any]


class EnterpriseChatOrchestrator:
    """Coordinate enterprise code chat, team annotations, and user memory."""

    def __init__(
        self,
        annotation_store: TeamAnnotationStore,
        annotation_service: Optional[EnterpriseAnnotationService] = None,
        memory_service: Optional[EnterpriseMemoryService] = None,
        code_pipeline_factory: Optional[CodePipelineFactory] = None,
    ) -> None:
        self.annotation_store = annotation_store
        self.annotation_service = annotation_service or EnterpriseAnnotationService(
            annotation_store=annotation_store,
        )
        self.memory_service = memory_service or EnterpriseMemoryService()
        self._code_pipeline_factory = code_pipeline_factory

    async def stream_chat(self, context: EnterpriseChatContext):
        """Yield NDJSON chunks for an enterprise chat response."""
        logger.info("="*60)
        logger.info("ENTERPRISE CHAT START")
        logger.info("  user=%s, role=%s, can_annotate=%s",
                     context.user.username, context.user.role,
                     context.user.can_create_annotations)
        logger.info("  project=%s, org=%s, repo=%s",
                     context.project.project_id, context.project.org_id,
                     context.project.repo)
        logger.info("  query=%s", context.request.query[:200])
        logger.info("="*60)

        try:
            relevant_annotations, manager_instructions, memory_context = (
                await self._load_context(context)
            )
        except Exception as exc:
            logger.error("ENTERPRISE CHAT: _load_context FAILED: %s", exc, exc_info=True)
            yield self._event("error", error=f"Context loading failed: {exc}")
            return

        logger.info("ENTERPRISE CHAT: context loaded — "
                     "annotations=%d, manager_instructions=%d, memory_sources=%d",
                     len(relevant_annotations), len(manager_instructions),
                     len(memory_context.sources))
        if memory_context.error:
            logger.warning("ENTERPRISE CHAT: memory context had error: %s",
                           memory_context.error)

        enhanced_query = self._build_enhanced_query(
            context=context,
            relevant_annotations=relevant_annotations,
            manager_instructions=manager_instructions,
            memory_context=memory_context,
        )
        logger.debug("ENTERPRISE CHAT: enhanced_query (first 500 chars):\n%s",
                      enhanced_query[:500])

        if relevant_annotations:
            logger.info("ENTERPRISE CHAT: yielding %d annotations to client",
                         len(relevant_annotations))
            yield self._event("annotations", annotations=relevant_annotations)

        if manager_instructions:
            logger.info("ENTERPRISE CHAT: yielding %d manager instructions",
                         len(manager_instructions))
            yield self._event(
                "manager_instructions",
                instructions=manager_instructions,
            )

        if memory_context.sources:
            logger.info("ENTERPRISE CHAT: yielding %d memory sources",
                         len(memory_context.sources))
            yield self._event("memory_sources", sources=memory_context.sources)

        try:
            pipeline = self._get_code_pipeline(context)
            logger.info("ENTERPRISE CHAT: code pipeline obtained — %s",
                         type(pipeline).__name__)
        except Exception as exc:
            logger.error("ENTERPRISE CHAT: _get_code_pipeline FAILED: %s",
                          exc, exc_info=True)
            yield self._event("error", error=f"Pipeline creation failed: {exc}")
            return

        answer_parts: List[str] = []
        chunk_count = 0

        logger.info("ENTERPRISE CHAT: starting pipeline.run_stream(repo=%s, user=%s, top_k=%d)",
                     context.project.repo, context.user.username,
                     context.request.top_k)

        try:
            async for chunk in pipeline.run_stream(
                query=enhanced_query,
                user_id=context.user.username,
                repo=context.project.repo,
                top_k=context.request.top_k,
            ):
                chunk_count += 1
                text_parts = self._extract_text_parts(chunk)
                answer_parts.extend(text_parts)
                if chunk_count <= 5 or chunk_count % 20 == 0:
                    logger.debug("ENTERPRISE CHAT: chunk #%d (len=%d): %s",
                                  chunk_count, len(chunk),
                                  chunk[:200] if len(chunk) > 200 else chunk)
                yield chunk
        except Exception as exc:
            logger.error("ENTERPRISE CHAT: pipeline.run_stream FAILED: %s",
                          exc, exc_info=True)
            yield self._event("error", error=f"Streaming failed: {exc}")
            return

        logger.info("ENTERPRISE CHAT: stream finished — %d chunks, %d answer chars",
                     chunk_count, sum(len(p) for p in answer_parts))

        answer_text = "".join(answer_parts).strip()
        logger.info("ENTERPRISE CHAT: answer_text length = %d", len(answer_text))

        annotation_task = None
        if context.user.can_create_annotations:
            logger.info("ENTERPRISE CHAT: scheduling annotation extraction")
            annotation_task = asyncio.create_task(
                self.annotation_service.extract_and_store(
                    context=context,
                    answer_text=answer_text,
                )
            )
        else:
            logger.info("ENTERPRISE CHAT: skipping annotation extraction "
                         "(can_create_annotations=%s)",
                         context.user.can_create_annotations)

        logger.info("ENTERPRISE CHAT: scheduling memory ingest")
        memory_task = asyncio.create_task(
            self.memory_service.ingest_conversation(
                query=context.request.query,
                answer_text=answer_text,
                user_id=context.user.username,
            )
        )

        if annotation_task is not None:
            try:
                result = await annotation_task
                created_ids, assigned_to_name = result if isinstance(result, tuple) else (result, "")
                logger.info("ENTERPRISE CHAT: annotation extraction returned %d ids (assigned=%s)",
                             len(created_ids) if created_ids else 0, assigned_to_name or "(none)")
                if created_ids:
                    yield self._event(
                        "annotations_created",
                        count=len(created_ids),
                        ids=created_ids,
                        assigned_to_name=assigned_to_name or None,
                    )
            except Exception as exc:
                logger.warning("Enterprise annotation write routing failed: %s",
                               exc, exc_info=True)

        try:
            ingest_result = await memory_task
            logger.info("ENTERPRISE CHAT: memory ingest result — success=%s, error=%s",
                         ingest_result.success, ingest_result.error)
            if not ingest_result.success and ingest_result.error:
                logger.debug(
                    "Enterprise memory ingest skipped/failed: %s",
                    ingest_result.error,
                )
        except Exception as exc:
            logger.warning("Enterprise memory write routing failed: %s",
                           exc, exc_info=True)

        logger.info("ENTERPRISE CHAT COMPLETE")
        logger.info("="*60)

    async def _load_context(
        self,
        context: EnterpriseChatContext,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], EnterpriseMemoryContext]:
        req = context.request
        logger.debug("ENTERPRISE CHAT: _load_context starting 3 concurrent tasks")

        annotations_task = asyncio.create_task(
            self._safe_search_annotations(context)
        )

        manager_task = asyncio.create_task(
            self._safe_get_manager_instructions(context)
        )

        memory_task = asyncio.create_task(
            self.memory_service.fetch_relevant_memory(
                query=req.query,
                user_id=context.user.username,
                top_k=min(req.top_k, 5),
            )
        )

        annotations, instructions, memory_context = await asyncio.gather(
            annotations_task,
            manager_task,
            memory_task,
        )

        logger.debug("ENTERPRISE CHAT: _load_context complete — "
                      "annotations=%d, instructions=%d, memory_sources=%d",
                      len(annotations), len(instructions),
                      len(memory_context.sources))

        return annotations, instructions, memory_context

    async def _safe_search_annotations(
        self,
        context: EnterpriseChatContext,
    ) -> List[Dict[str, Any]]:
        req = context.request
        try:
            return await self.annotation_store.search_relevant_for_query(
                project_id=context.project.project_id,
                query=req.query,
                file_path=req.file_path,
                symbol_name=req.symbol_name,
                top_k=5,
            )
        except Exception as exc:
            logger.warning("Enterprise annotation retrieval failed: %s", exc)
            return []

    async def _safe_get_manager_instructions(
        self,
        context: EnterpriseChatContext,
    ) -> List[Dict[str, Any]]:
        if context.user.role not in ("intern", "sde2"):
            return []

        try:
            return await self.annotation_store.get_manager_instructions(
                project_id=context.project.project_id,
                target_role=context.user.role,
                top_k=10,
            )
        except Exception as exc:
            logger.warning("Enterprise manager instruction retrieval failed: %s", exc)
            return []

    def _get_code_pipeline(self, context: EnterpriseChatContext) -> Any:
        logger.debug("ENTERPRISE CHAT: _get_code_pipeline(org=%s, repo=%s, project=%s)",
                      context.project.org_id, context.project.repo,
                      context.project.project_id)
        if self._code_pipeline_factory is not None:
            logger.debug("ENTERPRISE CHAT: using custom code_pipeline_factory")
            return self._code_pipeline_factory(
                context.project.org_id,
                context.project.repo,
                context.project.project_id,
            )

        from src.api.dependencies import get_code_pipeline

        logger.debug("ENTERPRISE CHAT: using default get_code_pipeline dependency")
        return get_code_pipeline(
            org_id=context.project.org_id,
            repo=context.project.repo,
            project_id=context.project.project_id,
        )

    def _build_enhanced_query(
        self,
        context: EnterpriseChatContext,
        relevant_annotations: List[Dict[str, Any]],
        manager_instructions: List[Dict[str, Any]],
        memory_context: EnterpriseMemoryContext,
    ) -> str:
        context_parts: List[str] = []

        if relevant_annotations:
            lines = ["Team Knowledge:"]
            for ann in relevant_annotations[:5]:
                lines.append(
                    "- "
                    f"[{ann.get('annotation_type', 'explanation')}] "
                    f"{self._truncate(ann.get('content', ''), 240)} "
                    f"(by {ann.get('author_name', 'unknown')}, "
                    f"{ann.get('author_role', 'member')})"
                )
            context_parts.append("\n".join(lines))

        if manager_instructions:
            lines = ["Manager Instructions:"]
            for instruction in manager_instructions[:10]:
                lines.append(
                    "- "
                    f"[{instruction.get('annotation_type', 'explanation')}] "
                    f"{self._truncate(instruction.get('content', ''), 260)} "
                    f"(from {instruction.get('author_name', 'manager')})"
                )
            context_parts.append("\n".join(lines))

        if memory_context.sources:
            lines = ["User Memory:"]
            for source in memory_context.sources[:5]:
                lines.append(
                    "- "
                    f"[{source.get('domain', 'memory')}] "
                    f"{self._truncate(source.get('content', ''), 260)}"
                )
            context_parts.append("\n".join(lines))

        if not context_parts:
            return context.request.query

        context_text = "\n\n".join(context_parts)
        return (
            f"{context.request.query}\n\n"
            "Enterprise context available for this chat:\n"
            f"{context_text}"
        )

    @staticmethod
    def _event(event_type: str, **payload: Any) -> str:
        return json.dumps(
            {"type": event_type, **payload},
            default=str,
        ) + "\n"

    @staticmethod
    def _extract_text_parts(chunk: str) -> List[str]:
        text_parts: List[str] = []
        for line in chunk.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "chunk" and payload.get("text"):
                text_parts.append(str(payload["text"]))
        return text_parts

    @staticmethod
    def _truncate(value: Any, limit: int) -> str:
        text = str(value or "").strip().replace("\n", " ")
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."
