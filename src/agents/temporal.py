"""
Temporal Agent — extracts structured time-based events from user input.

Receives classified "event" queries from the ClassifierAgent, uses an LLM to
extract structured event data (date, event_name, year, desc, time), and
handles relative date expressions using a session context datetime.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.prompts.temporal import build_system_prompt, pack_temporal_query
from src.schemas.events import EventData, EventResult
from src.utils.text import parse_raw_response_to_event

_DAYS_IN_MONTH = {
    1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}


class TemporalAgent(BaseAgent):
    def __init__(self, model: BaseChatModel) -> None:
        super().__init__(
            model=model,
            name="temporal",
            system_prompt=build_system_prompt(),
        )

    async def arun(self, state: Dict[str, Any]) -> EventResult:
        query = state.get("classifier_output", "")
        context_date = state.get("session_datetime", "")

        if not query:
            self.logger.debug("Empty query — returning empty event result.")
            return EventResult()

        user_message = pack_temporal_query(query, context_date=context_date)
        messages = self._build_messages(user_message)
        raw_content = await self._call_model(messages)

        event_data = parse_raw_response_to_event(raw_content)

        if not event_data:
            self.logger.info("No valid event extracted (NO_EVENT or missing date).")
            return EventResult()

        # Validate MM-DD format
        date_str = event_data.get("date", "")
        if not self._validate_date_format(date_str):
            self.logger.warning("Invalid date format: %s", date_str)
            return EventResult()

        event = EventData(
            date=date_str,
            event_name=event_data.get("event_name"),
            year=event_data.get("year"),
            desc=event_data.get("desc"),
            time=event_data.get("time"),
            date_expression=event_data.get("date_expression"),
        )

        result = EventResult(event=event)

        self.logger.info("=" * 50)
        self.logger.info("Extracted Temporal Event:")
        self.logger.info("  Date: %s", event.date)
        self.logger.info("  Event: %s", event.event_name or "N/A")
        self.logger.info("  Year: %s", event.year or "N/A")
        self.logger.info("  Description: %s", event.desc or "N/A")
        self.logger.info("  Time: %s", event.time or "N/A")
        self.logger.info("  Original Expression: %s", event.date_expression or "N/A")
        self.logger.info("=" * 50)

        return result

    @staticmethod
    def _validate_date_format(date_str: str) -> bool:
        if not date_str or len(date_str) != 5:
            return False

        parts = date_str.split("-")
        if len(parts) != 2:
            return False

        try:
            month, day = int(parts[0]), int(parts[1])
        except ValueError:
            return False

        if month < 1 or month > 12:
            return False
        if day < 1 or day > _DAYS_IN_MONTH.get(month, 31):
            return False

        return True
