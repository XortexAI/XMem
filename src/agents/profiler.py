"""
Profiler Agent — extracts structured user facts from conversational input.

Takes a query routed from the ClassifierAgent (source="profile")
and returns a ProfileResult containing topic/sub_topic/memo facts.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.prompts.profiler import build_system_prompt, pack_profiler_query
from src.schemas.profile import ProfileFact, ProfileResult
from src.utils.text import parse_raw_response_to_profiles


class ProfilerAgent(BaseAgent):
    def __init__(self, model: BaseChatModel) -> None:
        super().__init__(
            model=model,
            name="profiler",
            system_prompt=build_system_prompt(),
        )

    async def arun(
        self,
        state: Dict[str, Any],
    ) -> ProfileResult:
        query = state.get("classifier_output", "")
        if not query:
            self.logger.debug("Empty query — returning empty profile result.")
            return ProfileResult()

        user_message = pack_profiler_query(query)
        messages = self._build_messages(user_message)
        raw_content = await self._call_model(messages)

        raw_facts = parse_raw_response_to_profiles(raw_content)

        facts = [
            ProfileFact(topic=f["topic"], sub_topic=f["sub_topic"], memo=f["memo"])
            for f in raw_facts
        ]

        result = ProfileResult(facts=facts)

        if not result.is_empty:
            self.logger.info("=" * 50)
            self.logger.info("Extracted Profile Facts:")
            for idx, fact in enumerate(result.facts, 1):
                self.logger.debug(
                    "  %d. %s / %s = %s", idx, fact.topic, fact.sub_topic, fact.memo
                )
            self.logger.info("Total facts: %d", len(result.facts))
            self.logger.info("=" * 50)
        else:
            self.logger.info("No profile facts extracted (trivial input).")

        return result
