"""
Snippet Agent — extracts personal code snippets from developer
conversations (single-user / free tier).

Takes a query routed from the ClassifierAgent (source="code") and returns
a SnippetExtractionResult containing code blocks, descriptions, language,
type, and tags for storage in the user's personal snippet namespace.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.prompts.snippet import build_system_prompt, pack_snippet_query
from src.schemas.code import (
    ExtractedSnippet,
    SnippetExtractionResult,
    SnippetType,
)
from src.utils.json_parse import extract_json_from_response


class SnippetAgent(BaseAgent):
    # Maximum input length before truncation
    MAX_INPUT_LENGTH = 16000

    def __init__(self, model: BaseChatModel) -> None:
        super().__init__(
            model=model,
            name="snippet",
            system_prompt=build_system_prompt(),
        )

    async def arun(self, state: Dict[str, Any]) -> SnippetExtractionResult:
        query = state.get("classifier_output", "")
        if not query:
            self.logger.debug("Empty query — returning empty snippet result.")
            return SnippetExtractionResult()

        # Truncate input if too long
        query = self._truncate_input(query)

        user_message = pack_snippet_query(query)
        messages = self._build_messages(user_message)
        raw_content = await self._call_model(messages)

        result = self._parse_response(raw_content)

        if not result.is_empty:
            self.logger.info("=" * 50)
            self.logger.info("Extracted Code Snippets:")
            for idx, snip in enumerate(result.snippets, 1):
                has_code = "with code" if snip.code_snippet else "no code"
                self.logger.info(
                    "  %d. [%s] %s (%s, %s) tags=%s",
                    idx, snip.snippet_type.value, snip.content[:60],
                    snip.language, has_code, snip.tags,
                )
            self.logger.info("Total snippets: %d", len(result.snippets))
            self.logger.info("=" * 50)
        else:
            self.logger.info("No code snippets extracted (trivial input).")

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

    def _parse_response(self, raw: str) -> SnippetExtractionResult:
        """Parse JSON response into SnippetExtractionResult."""
        snippets: list[ExtractedSnippet] = []

        try:
            data = extract_json_from_response(raw)
        except Exception as exc:
            self.logger.error("Failed to extract JSON from response: %s", exc)
            self.logger.debug("Raw response: %s", raw[:500])
            return SnippetExtractionResult()

        snippets_data = data.get("snippets", [])
        if not snippets_data:
            return SnippetExtractionResult()

        if not isinstance(snippets_data, list):
            self.logger.warning(
                "Expected snippets to be a list, got %s",
                type(snippets_data).__name__,
            )
            return SnippetExtractionResult()

        for idx, s in enumerate(snippets_data):
            if not isinstance(s, dict):
                self.logger.warning(
                    "Skipping snippet at index %d: not a dict", idx,
                )
                continue

            snippet = self._parse_snippet(s, idx)
            if snippet is not None:
                snippets.append(snippet)

        return SnippetExtractionResult(snippets=snippets)

    def _parse_snippet(
        self, s: dict, idx: int
    ) -> Optional[ExtractedSnippet]:
        """Parse a single snippet dict with validation."""
        # Validate required content field
        content = s.get("content", "").strip() if s.get("content") else ""
        if not content:
            self.logger.warning(
                "Skipping snippet at index %d: missing or empty content", idx,
            )
            return None

        # Validate required language field
        language = s.get("language", "").strip() if s.get("language") else ""
        if not language:
            self.logger.warning(
                "Skipping snippet at index %d: missing language", idx,
            )
            return None

        # Parse snippet_type with fallback
        snippet_type_str = s.get("snippet_type", "algorithm")
        try:
            snippet_type = SnippetType(snippet_type_str)
        except ValueError:
            self.logger.warning(
                "Invalid snippet_type '%s' at index %d, using default",
                snippet_type_str, idx,
            )
            snippet_type = SnippetType.ALGORITHM

        # Parse tags
        tags = s.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        return ExtractedSnippet(
            content=content,
            code_snippet=s.get("code_snippet", ""),
            language=language,
            snippet_type=snippet_type,
            tags=tags[:10],
        )
