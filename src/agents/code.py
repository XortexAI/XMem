"""
Code Agent — extracts code annotations (team knowledge) from developer
conversations.

Takes a query routed from the ClassifierAgent (source="code") and returns
a CodeAnnotationResult containing structured annotations targeting
specific symbols, files, or repositories.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.prompts.code import build_system_prompt, pack_code_query
from src.schemas.code import (
    AnnotationSeverity,
    AnnotationType,
    CodeAnnotationResult,
    ExtractedAnnotation,
)


class CodeAgent(BaseAgent):
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

    def _parse_response(self, raw: str) -> CodeAnnotationResult:
        """Parse JSON response into CodeAnnotationResult."""
        try:
            cleaned = raw.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]

            data = json.loads(cleaned.strip())

            annotations_data = data.get("annotations", [])
            if not annotations_data:
                return CodeAnnotationResult()

            annotations = []
            for ann_dict in annotations_data:
                ann_type_str = ann_dict.get("annotation_type", "explanation")
                try:
                    ann_type = AnnotationType(ann_type_str)
                except ValueError:
                    ann_type = AnnotationType.EXPLANATION

                severity = None
                if ann_dict.get("severity"):
                    try:
                        severity = AnnotationSeverity(ann_dict["severity"])
                    except ValueError:
                        severity = None

                annotations.append(ExtractedAnnotation(
                    target_symbol=ann_dict.get("target_symbol"),
                    target_file=ann_dict.get("target_file"),
                    content=ann_dict.get("content", ""),
                    annotation_type=ann_type,
                    severity=severity,
                    repo=ann_dict.get("repo"),
                    assigned_to_name=ann_dict.get("assigned_to_name"),
                ))

            return CodeAnnotationResult(annotations=annotations)

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            self.logger.error("Failed to parse code agent response: %s", exc)
            self.logger.debug("Raw response: %s", raw[:500])
            return CodeAnnotationResult()
