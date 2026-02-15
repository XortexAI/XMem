"""
Classifier Agent — the entry-point router for Xmem.

Classifies user input into one or more intent categories (code, profile,
event) so downstream agents only receive the sub-queries relevant to them.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.prompts.classifier import build_system_prompt, pack_classification_query
from src.schemas.classification import ClassificationResult
from src.utils.text import parse_raw_response_to_classifications


class ClassifierAgent(BaseAgent):
    def __init__(self, model: BaseChatModel) -> None:
        super().__init__(
            model=model,
            name="classifier",
            system_prompt=build_system_prompt(),
        )

    async def arun(self, state: Dict[str, Any]) -> ClassificationResult:
        user_input = state.get("user_query")
        if not user_input:
            self.logger.debug("Empty query — returning empty classifications.")
            return ClassificationResult(classifications=[])

        user_message = pack_classification_query(user_input)
        messages = self._build_messages(user_message)
        raw_content = await self._call_model(messages)
        classifications = parse_raw_response_to_classifications(raw_content)

        if classifications:
            self.logger.info("=" * 50)
            self.logger.info("Extracted Classifications:")
            for idx, cls in enumerate(classifications, 1):
                self.logger.info(
                    "  %d. source=%s  query=%s", idx, cls["source"], cls["query"]
                )
            self.logger.info("Total classifications: %d", len(classifications))
            self.logger.info("=" * 50)
        else:
            self.logger.info("No actionable classifications found (trivial input).")

        return ClassificationResult(classifications=classifications)
