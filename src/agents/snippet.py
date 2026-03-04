"""
Snippet Agent — extracts personal code snippets from developer
conversations (single-user / free tier).

Takes a query routed from the ClassifierAgent (source="code") and returns
a SnippetExtractionResult containing code blocks, descriptions, language,
type, and tags for storage in the user's personal snippet namespace.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.prompts.snippet import build_system_prompt, pack_snippet_query
from src.schemas.code import (
    ExtractedSnippet,
    SnippetExtractionResult,
    SnippetType,
)


class SnippetAgent(BaseAgent):
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

    def _parse_response(self, raw: str) -> SnippetExtractionResult:
        """Parse JSON response into SnippetExtractionResult."""
        try:
            cleaned = raw.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]

            data = json.loads(cleaned.strip())

            snippets_data = data.get("snippets", [])
            if not snippets_data:
                return SnippetExtractionResult()

            snippets = []
            for s in snippets_data:
                content = s.get("content", "").strip()
                if not content:
                    continue

                snippet_type_str = s.get("snippet_type", "algorithm")
                try:
                    snippet_type = SnippetType(snippet_type_str)
                except ValueError:
                    snippet_type = SnippetType.ALGORITHM

                tags = s.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]

                snippets.append(ExtractedSnippet(
                    content=content,
                    code_snippet=s.get("code_snippet", ""),
                    language=s.get("language", ""),
                    snippet_type=snippet_type,
                    tags=tags[:10],
                ))

            return SnippetExtractionResult(snippets=snippets)

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            self.logger.error("Failed to parse snippet agent response: %s", exc)
            self.logger.debug("Raw response: %s", raw[:500])
            return SnippetExtractionResult()
