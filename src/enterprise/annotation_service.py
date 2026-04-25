"""Project-scoped enterprise annotation extraction and storage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from src.config import settings
from src.enterprise.schemas import EnterpriseChatContext

if TYPE_CHECKING:
    from src.storage.team_annotation_store import TeamAnnotationStore

logger = logging.getLogger("xmem.enterprise.annotation_service")

CodeAgentFactory = Callable[[], Any]
AnnotationCountIncrementer = Callable[[str, int], Any]


class EnterpriseAnnotationService:
    """Extract code annotations and persist them in project Pinecone namespaces."""

    def __init__(
        self,
        annotation_store: TeamAnnotationStore,
        code_agent_factory: Optional[CodeAgentFactory] = None,
        annotation_count_incrementer: Optional[AnnotationCountIncrementer] = None,
    ) -> None:
        self.annotation_store = annotation_store
        self._code_agent_factory = code_agent_factory
        self._annotation_count_incrementer = annotation_count_incrementer
        self._agent: Optional[Any] = None

    async def extract_and_store(
        self,
        context: EnterpriseChatContext,
        answer_text: str = "",
    ) -> List[str]:
        """Extract annotations from a completed chat turn and store them.

        Enterprise annotations are canonical in TeamAnnotationStore, which maps
        to Pinecone namespace ``annotations:{project_id}``.
        """
        if not context.user.can_create_annotations:
            logger.info(
                "Skipping annotation extraction for user %s in role %s",
                context.user.user_id,
                context.user.role,
            )
            return []

        agent = self._get_code_agent()
        conversation = self._pack_conversation(context.request.query, answer_text)
        result = await agent.arun({"classifier_output": conversation})

        if getattr(result, "is_empty", True):
            logger.info("No enterprise annotations extracted from chat turn")
            return []

        created_ids: List[str] = []
        for ann in result.annotations:
            content = (getattr(ann, "content", "") or "").strip()
            if not content:
                continue

            try:
                annotation_id = self.annotation_store.create_annotation(
                    project_id=context.project.project_id,
                    content=content,
                    author_id=context.user.user_id,
                    author_name=context.user.username or context.user.user_id,
                    author_role=context.user.role,
                    org_id=context.project.org_id,
                    repo=getattr(ann, "repo", None) or context.project.repo,
                    annotation_type=self._enum_value(
                        getattr(ann, "annotation_type", None),
                        default="explanation",
                    ),
                    file_path=getattr(ann, "target_file", None)
                    or context.request.file_path,
                    symbol_name=getattr(ann, "target_symbol", None)
                    or context.request.symbol_name,
                    severity=self._enum_value(getattr(ann, "severity", None)),
                )
                if annotation_id:
                    created_ids.append(annotation_id)
            except Exception as exc:
                logger.warning("Failed to store enterprise annotation: %s", exc)

        logger.info(
            "Stored %d enterprise annotation(s) for project %s",
            len(created_ids),
            context.project.project_id,
        )
        if created_ids and self._annotation_count_incrementer is not None:
            try:
                self._annotation_count_incrementer(
                    context.project.project_id,
                    len(created_ids),
                )
            except Exception as exc:
                logger.warning("Failed to increment project annotation count: %s", exc)

        return created_ids

    def _get_code_agent(self) -> Any:
        if self._code_agent_factory is not None:
            return self._code_agent_factory()

        if self._agent is None:
            from src.agents.code import CodeAgent
            from src.models import get_model

            override = getattr(settings, "code_model", None)
            model = get_model(model_name=override) if override else get_model()
            self._agent = CodeAgent(model=model)

        return self._agent

    @staticmethod
    def _pack_conversation(query: str, answer_text: str) -> str:
        if answer_text.strip():
            return (
                "User message:\n"
                f"{query.strip()}\n\n"
                "Assistant response:\n"
                f"{answer_text.strip()}"
            )
        return query.strip()

    @staticmethod
    def _enum_value(value: Any, default: Optional[str] = None) -> Optional[str]:
        if value is None:
            return default
        return getattr(value, "value", value) or default
