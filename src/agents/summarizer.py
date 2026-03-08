"""
Summarizer Agent — extracts memorable user facts from conversation turns.

Takes a user query and agent response, produces a concise bullet-point
summary of facts worth remembering about the user.

Effort-aware:
  LOW  — single LLM call on the full input (fast, cheaper)
  HIGH — chunks large inputs, summarizes each chunk independently,
         then merges + deduplicates bullets (slower, more accurate)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.config.effort import EffortConfig, EffortLevel, get_effort_config
from src.prompts.summarizer import build_system_prompt, pack_summary_query
from src.schemas.summary import SummaryResult


def _estimate_tokens(text: str) -> int:
    """Cheap word-count proxy for token estimation (~1.3 tokens/word)."""
    return int(len(text.split()) * 1.3)


def _chunk_text(text: str, chunk_size_tokens: int) -> List[str]:
    """Split *text* into chunks of approximately *chunk_size_tokens* tokens.

    Splits on sentence boundaries ('. ') when possible, falling back to
    word boundaries.  Never produces empty chunks.
    """
    words = text.split()
    # approximate words-per-chunk (reverse the 1.3 multiplier)
    words_per_chunk = max(1, int(chunk_size_tokens / 1.3))

    chunks: List[str] = []
    i = 0
    while i < len(words):
        end = min(i + words_per_chunk, len(words))
        segment = " ".join(words[i:end])

        # try to snap to the last sentence boundary inside the segment
        last_period = segment.rfind(". ")
        if last_period > len(segment) // 3:
            segment = segment[: last_period + 1]
            consumed = len(segment.split())
        else:
            consumed = end - i

        chunks.append(segment.strip())
        i += consumed

    return [c for c in chunks if c]


_MERGE_SYSTEM = (
    "You receive multiple bullet-point summaries extracted from different "
    "sections of the same conversation. Merge them into a single, "
    "deduplicated bullet-point list. Remove exact and near-duplicate bullets. "
    "Preserve ALL specific entities, dates, numbers, and names VERBATIM. "
    "Output ONLY the merged bullet list (each line starting with '- ')."
)


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
        effort_cfg = state.get("effort_config", get_effort_config(EffortLevel.LOW))

        if not user_query and not agent_response:
            self.logger.debug("Empty input — returning empty summary.")
            return SummaryResult()

        combined_len = _estimate_tokens(user_query) + _estimate_tokens(agent_response)

        if combined_len > effort_cfg.summary_chunk_threshold:
            return await self._chunked_summarize(
                user_query, agent_response, effort_cfg,
            )

        return await self._single_summarize(user_query, agent_response)

    # ── Single-pass (LOW or small input) ──────────────────────────────

    async def _single_summarize(
        self, user_query: str, agent_response: str,
    ) -> SummaryResult:
        user_message = pack_summary_query(user_query, agent_response)
        messages = self._build_messages(user_message)
        raw_content = await self._call_model(messages)
        return self._parse(raw_content)

    # ── Chunked (HIGH + large input) ──────────────────────────────────

    async def _chunked_summarize(
        self,
        user_query: str,
        agent_response: str,
        cfg: EffortConfig,
    ) -> SummaryResult:
        full_text = f"{user_query}\n\n{agent_response}"
        chunks = _chunk_text(full_text, cfg.summary_chunk_size)

        self.logger.info(
            "HIGH-effort chunked summarization: %d chunks "
            "(threshold=%d, chunk_size=%d)",
            len(chunks), cfg.summary_chunk_threshold, cfg.summary_chunk_size,
        )

        # Summarize each chunk concurrently
        tasks = [
            self._single_summarize(chunk, "")
            for chunk in chunks
        ]
        chunk_results: List[SummaryResult] = await asyncio.gather(*tasks)

        # Collect all non-empty bullet lines
        all_bullets: List[str] = []
        for cr in chunk_results:
            if not cr.is_empty:
                all_bullets.append(cr.summary.strip())

        if not all_bullets:
            self.logger.info("All chunks produced empty summaries.")
            return SummaryResult()

        merged_text = "\n".join(all_bullets)

        # Optional merge pass to deduplicate across chunks
        if cfg.summary_merge_pass and len(all_bullets) > 1:
            merged_text = await self._merge_bullets(merged_text)

        return self._parse(merged_text)

    async def _merge_bullets(self, combined: str) -> str:
        """LLM pass to deduplicate and merge bullets from multiple chunks."""
        messages = [
            {"role": "system", "content": _MERGE_SYSTEM},
            {"role": "user", "content": combined},
        ]
        return await self._call_model(messages)

    # ── Shared parsing ────────────────────────────────────────────────

    def _parse(self, raw_content: str) -> SummaryResult:
        summary = raw_content.strip()
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
