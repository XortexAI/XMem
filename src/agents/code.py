"""
Code Agent — extracts code annotations (team knowledge) from developer
conversations.

Takes a query routed from the ClassifierAgent (source="code") and returns
a CodeAnnotationResult containing structured annotations targeting
specific symbols, files, or repositories.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.prompts.code import build_system_prompt, pack_code_query
from src.schemas.code import (
    AnnotationSeverity,
    AnnotationType,
    CodeAnnotationResult,
    ExtractedAnnotation,
)
from src.utils.json_parse import extract_json_from_response


class CodeAgent(BaseAgent):
    # Maximum input length before truncation
    MAX_INPUT_LENGTH = 16000

    def __init__(self, model: BaseChatModel) -> None:
        super().__init__(
            model=model,
            name="code",
            system_prompt=build_system_prompt(),
        )

    async def arun(self, state: Dict[str, Any]) -> CodeAnnotationResult:
        query = state.get("classifier_output", "")
        if not query:
            self.logger.debug("Empty query — returning empty code result.")
            return CodeAnnotationResult()

        # Truncate input if too long
        query = self._truncate_input(query)

        user_message = pack_code_query(query)
        messages = self._build_messages(user_message)
        raw_content = await self._call_model(messages)

        result = self._parse_response(raw_content)

        if not result.is_empty:
            self.logger.info("=" * 50)
            self.logger.info("Extracted Code Annotations:")
            for idx, ann in enumerate(result.annotations, 1):
                target = ann.target_symbol or ann.target_file or "(general)"
                self.logger.info(
                    "  %d. [%s] %s → %s",
                    idx, ann.annotation_type.value, target, ann.content[:60],
                )
            self.logger.info("Total annotations: %d", len(result.annotations))
            self.logger.info("=" * 50)
        else:
            self.logger.info("No code annotations extracted (trivial input).")

        return result

    def _truncate_input(self, query: str) -> str:
        """Truncate input if it exceeds MAX_INPUT_LENGTH."""
        if len(query) > self.MAX_INPUT_LENGTH:
            self.logger.warning(
                "Input length %d exceeds limit %d, truncating",
                len(query), self.MAX_INPUT_LENGTH,
            )
            return query[:self.MAX_INPUT_LENGTH]
        return query

    def _parse_response(self, raw: str) -> CodeAnnotationResult:
        """Parse JSON response into CodeAnnotationResult."""
        annotations: list[ExtractedAnnotation] = []

        try:
            data = extract_json_from_response(raw)
        except Exception as exc:
            self.logger.error("Failed to extract JSON from response: %s", exc)
            self.logger.debug("Raw response: %s", raw[:500])
            return CodeAnnotationResult()

        annotations_data = data.get("annotations", [])
        if not annotations_data:
            return CodeAnnotationResult()

        if not isinstance(annotations_data, list):
            self.logger.warning(
                "Expected annotations to be a list, got %s",
                type(annotations_data).__name__,
            )
            return CodeAnnotationResult()

        for idx, ann_dict in enumerate(annotations_data):
            if not isinstance(ann_dict, dict):
                self.logger.warning(
                    "Skipping annotation at index %d: not a dict", idx,
                )
                continue

            ann = self._parse_annotation(ann_dict, idx)
            if ann is not None:
                annotations.append(ann)

        return CodeAnnotationResult(annotations=annotations)

    def _parse_annotation(
        self, ann_dict: dict, idx: int
    ) -> Optional[ExtractedAnnotation]:
        """Parse a single annotation dict with validation."""
        # Validate required content field
        content = ann_dict.get("content", "").strip() if ann_dict.get("content") else ""
        if not content:
            self.logger.warning(
                "Skipping annotation at index %d: missing or empty content", idx,
            )
            return None

        # Parse annotation_type with fallback
        ann_type_str = ann_dict.get("annotation_type", "explanation")
        try:
            ann_type = AnnotationType(ann_type_str)
        except ValueError:
            self.logger.warning(
                "Invalid annotation_type '%s' at index %d, using default",
                ann_type_str, idx,
            )
            ann_type = AnnotationType.EXPLANATION

        # Parse severity (optional)
        severity = None
        if ann_dict.get("severity"):
            try:
                severity = AnnotationSeverity(ann_dict["severity"])
            except ValueError:
                self.logger.warning(
                    "Invalid severity '%s' at index %d, ignoring",
                    ann_dict["severity"], idx,
                )

        return ExtractedAnnotation(
            target_symbol=ann_dict.get("target_symbol"),
            target_file=ann_dict.get("target_file"),
            content=content,
            annotation_type=ann_type,
            severity=severity,
            repo=ann_dict.get("repo"),
            assigned_to_name=ann_dict.get("assigned_to_name"),
        )
