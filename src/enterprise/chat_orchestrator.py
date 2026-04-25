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
        relevant_annotations, manager_instructions, memory_context = (
            await self._load_context(context)
        )

        enhanced_query = self._build_enhanced_query(
            context=context,
            relevant_annotations=relevant_annotations,
            manager_instructions=manager_instructions,
            memory_context=memory_context,
        )

        if relevant_annotations:
            yield self._event("annotations", annotations=relevant_annotations)

        if manager_instructions:
            yield self._event(
                "manager_instructions",
                instructions=manager_instructions,
            )

        if memory_context.sources:
            yield self._event("memory_sources", sources=memory_context.sources)

        pipeline = self._get_code_pipeline(context)
        answer_parts: List[str] = []

        async for chunk in pipeline.run_stream(
            query=enhanced_query,
            user_id=context.user.username,
            repo=context.project.repo,
            top_k=context.request.top_k,
        ):
            answer_parts.extend(self._extract_text_parts(chunk))
            yield chunk

        answer_text = "".join(answer_parts).strip()

        annotation_task = None
        if context.user.can_create_annotations:
            annotation_task = asyncio.create_task(
                self.annotation_service.extract_and_store(
                    context=context,
                    answer_text=answer_text,
                )
            )

        memory_task = asyncio.create_task(
            self.memory_service.ingest_conversation(
                query=context.request.query,
                answer_text=answer_text,
                user_id=context.user.username,
            )
        )

        if annotation_task is not None:
            try:
                created_ids = await annotation_task
                if created_ids:
                    yield self._event(
                        "annotations_created",
                        count=len(created_ids),
                        ids=created_ids,
                    )
            except Exception as exc:
                logger.warning("Enterprise annotation write routing failed: %s", exc)

        try:
            ingest_result = await memory_task
            if not ingest_result.success and ingest_result.error:
                logger.debug(
                    "Enterprise memory ingest skipped/failed: %s",
                    ingest_result.error,
                )
        except Exception as exc:
            logger.warning("Enterprise memory write routing failed: %s", exc)

    async def _load_context(
        self,
        context: EnterpriseChatContext,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], EnterpriseMemoryContext]:
        req = context.request

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
        if self._code_pipeline_factory is not None:
            return self._code_pipeline_factory(
                context.project.org_id,
                context.project.repo,
                context.project.project_id,
            )

        from src.api.dependencies import get_code_pipeline

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
