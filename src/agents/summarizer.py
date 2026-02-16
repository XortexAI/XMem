"""
Summarizer Agent — extracts memorable user facts from conversation turns.

Takes a user query and agent response, produces a concise bullet-point
summary of facts worth remembering about the user.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.prompts.summarizer import build_system_prompt, pack_summary_query
from src.schemas.summary import SummaryResult


class SummarizerAgent(BaseAgent):
    def __init__(self, model: BaseChatModel) -> None:
        super().__init__(
            model=model,
            name="summarizer",
            system_prompt=build_system_prompt(),
        )

    async def arun(
        self,
        state: Dict[str, Any],
    ) -> SummaryResult:
        user_query = state.get("user_query", "")
        agent_response = state.get("agent_response", "")

        if not user_query and not agent_response:
            self.logger.debug("Empty input — returning empty summary.")
            return SummaryResult()

        user_message = pack_summary_query(user_query, agent_response)
        messages = self._build_messages(user_message)
        raw_content = await self._call_model(messages)

        summary = raw_content.strip()

        # Treat empty-like responses as no summary
        if summary in ('""', "''", "empty", "(empty)", "(empty string)"):
            summary = ""

        result = SummaryResult(summary=summary)

        if not result.is_empty:
            self.logger.info("=" * 50)
            self.logger.info("Generated Summary:")
            self.logger.info(summary)
            self.logger.info("=" * 50)
        else:
            self.logger.info("No memorable facts extracted (trivial input).")

        return result
