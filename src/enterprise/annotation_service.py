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
        logger.info("ANNOTATION SERVICE: extract_and_store called — "
                     "user=%s, role=%s, can_create=%s, answer_len=%d",
                     context.user.user_id, context.user.role,
                     context.user.can_create_annotations, len(answer_text))

        if not context.user.can_create_annotations:
            logger.info(
                "Skipping annotation extraction for user %s in role %s",
                context.user.user_id,
                context.user.role,
            )
            return []

        try:
            agent = self._get_code_agent()
            logger.info("ANNOTATION SERVICE: code agent obtained — %s", type(agent).__name__)
        except Exception as exc:
            logger.error("ANNOTATION SERVICE: _get_code_agent FAILED: %s", exc, exc_info=True)
            return []

        conversation = self._pack_conversation(context.request.query, answer_text)
        logger.debug("ANNOTATION SERVICE: packed conversation (first 300 chars): %s",
                      conversation[:300])

        try:
            result = await agent.arun({"classifier_output": conversation})
            logger.info("ANNOTATION SERVICE: agent returned — is_empty=%s",
                         getattr(result, 'is_empty', 'N/A'))
        except Exception as exc:
            logger.error("ANNOTATION SERVICE: agent.arun FAILED: %s", exc, exc_info=True)
            return []

        if getattr(result, "is_empty", True):
            logger.info("No enterprise annotations extracted from chat turn")
            return []

        created_ids: List[str] = []
        last_assigned_to_name: str = ""
        for ann in result.annotations:
            content = (getattr(ann, "content", "") or "").strip()
            if not content:
                continue

            ann_assigned = getattr(ann, "assigned_to_name", None)

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
                    assigned_to_name=ann_assigned,
                )
                if annotation_id:
                    created_ids.append(annotation_id)
                    if ann_assigned:
                        last_assigned_to_name = ann_assigned
            except Exception as exc:
                logger.warning("Failed to store enterprise annotation: %s", exc)

        logger.info(
            "Stored %d enterprise annotation(s) for project %s (assigned_to_name=%s)",
            len(created_ids),
            context.project.project_id,
            last_assigned_to_name or "(none)",
        )
        if created_ids and self._annotation_count_incrementer is not None:
            try:
                self._annotation_count_incrementer(
                    context.project.project_id,
                    len(created_ids),
                )
            except Exception as exc:
                logger.warning("Failed to increment project annotation count: %s", exc)

        return created_ids, last_assigned_to_name

    def _get_code_agent(self) -> Any:
        if self._code_agent_factory is not None:
            logger.debug("ANNOTATION SERVICE: using custom code_agent_factory")
            return self._code_agent_factory()

        if self._agent is None:
            logger.info("ANNOTATION SERVICE: initializing CodeAgent for annotation extraction")
            from src.agents.code import CodeAgent
            from src.models import get_model

            override = getattr(settings, "code_model", None)
            model = get_model(model_name=override) if override else get_model()
            self._agent = CodeAgent(model=model)
            logger.info("ANNOTATION SERVICE: CodeAgent initialized with model=%s", type(model).__name__)

        return self._agent

    @staticmethod
    def _pack_conversation(query: str, answer_text: str) -> str:
        # IMPORTANT: Only analyze the USER's query for annotations.
        # The assistant's answer is NOT user-authored knowledge — extracting it
        # would create annotation spam from every Q&A exchange.
        # Only the user's own words (tasks, bug reports, decisions) should be annotated.
        return query.strip()

    @staticmethod
    def _enum_value(value: Any, default: Optional[str] = None) -> Optional[str]:
        if value is None:
            return default
        return getattr(value, "value", value) or default
