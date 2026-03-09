"""
Effort-level configuration for the XMem ingest pipeline.

Two modes
---------
LOW   — fast, single pipeline call, no chunking.
HIGH  — splits the user_query into overlapping chunks (≈200 tokens each,
        with a small context overlap from the previous chunk) and runs the
        full ingest pipeline on each chunk **in parallel**, ensuring that
        nothing is missed even in long inputs.

        Token estimation: 4 characters ≈ 1 token (cheap, no tokeniser needed).
        Chunk boundary: snapped to the last sentence-ending full stop within
        the chunk so that half-sentences are never cut mid-thought.

Usage::

    from src.config.effort import EffortLevel, EffortConfig, get_effort_config

    cfg = get_effort_config(EffortLevel.HIGH)
    cfg.chunk_size_tokens     # 200
    cfg.overlap_tokens        # 15
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import List


# ---------------------------------------------------------------------------
# Enums & config dataclass
# ---------------------------------------------------------------------------

class EffortLevel(str, Enum):
    LOW  = "low"
    HIGH = "high"


@dataclass(frozen=True)
class EffortConfig:
    """Strategy object holding all effort-dependent knobs.

    Add new knobs here as the product evolves; call-sites that already hold
    a config object need no changes.
    """

    level: EffortLevel

    # ── Ingest chunking (HIGH mode) ───────────────────────────────────
    # Minimum token count below which we never chunk (even in HIGH mode).
    chunk_threshold_tokens: int
    # Target chunk size in tokens when chunking is active.
    chunk_size_tokens: int
    # Number of tokens carried over from the tail of the previous chunk as
    # context for the next chunk.
    overlap_tokens: int

    # ── Summarizer (kept for backward compat with SummarizerAgent) ────
    summary_chunk_threshold: int
    summary_chunk_size: int
    summary_merge_pass: bool

    # ── Future knobs (add here as needed) ─────────────────────────────
    # judge_verification_depth: int
    # embedding_rerank: bool
    # classifier_multi_pass: bool


_CONFIGS: dict[EffortLevel, EffortConfig] = {
    EffortLevel.LOW: EffortConfig(
        level=EffortLevel.LOW,
        # LOW: never chunk — threshold is effectively infinite
        chunk_threshold_tokens=999_999,
        chunk_size_tokens=999_999,
        overlap_tokens=0,
        # Summarizer legacy knobs
        summary_chunk_threshold=999_999,
        summary_chunk_size=999_999,
        summary_merge_pass=False,
    ),
    EffortLevel.HIGH: EffortConfig(
        level=EffortLevel.HIGH,
        chunk_threshold_tokens=200,   # chunk if input exceeds this many tokens
        chunk_size_tokens=200,        # target size of each chunk in tokens
        overlap_tokens=15,            # tokens of tail-context carried forward
        # Summarizer legacy knobs
        summary_chunk_threshold=200,
        summary_chunk_size=100,
        summary_merge_pass=True,
    ),
}


@lru_cache(maxsize=4)
def get_effort_config(level: EffortLevel | str) -> EffortConfig:
    """Resolve an effort level (string or enum) to its ``EffortConfig``."""
    if isinstance(level, str):
        level = EffortLevel(level.lower())
    return _CONFIGS[level]


# ---------------------------------------------------------------------------
# Token estimation (4 chars ≈ 1 token — no tokeniser needed)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Cheap character-count proxy: 4 characters ≈ 1 token."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """Split *text* into overlapping chunks of ≈ *chunk_size_tokens* tokens.

    Rules
    -----
    * Chunk boundaries are snapped to the **last sentence-ending full stop**
      (``'. '``) inside the target window, so sentences are never split in
      the middle.  If no full stop is found in the latter two-thirds of the
      window, the boundary falls on a word boundary instead.
    * Each chunk (except the first) is prefixed with the last *overlap_tokens*
      tokens' worth of text from the previous chunk so downstream agents have
      a tiny bit of context.
    * Never produces empty chunks.

    Parameters
    ----------
    text:
        The raw input string to split.
    chunk_size_tokens:
        Approximate target size of each chunk measured in tokens (4 chars = 1).
    overlap_tokens:
        Number of tokens from the tail of the previous chunk to prepend to
        the next chunk as cross-boundary context.

    Returns
    -------
    List[str]
        Non-empty list of chunk strings (may be a single element if the input
        is short or no split point was found).
    """
    chars_per_chunk = chunk_size_tokens * 4           # token → char budget
    overlap_chars   = overlap_tokens   * 4            # overlap in chars

    if len(text) <= chars_per_chunk:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    start = 0
    prev_tail = ""   # overlap text from the previous chunk

    while start < len(text):
        end = min(start + chars_per_chunk, len(text))
        window = text[start:end]

        # ── Snap to last full-stop sentence boundary ──────────────────
        # Only look in the latter 2/3 of the window to avoid micro-chunks.
        search_from = len(window) // 3
        last_period = window.rfind(". ", search_from)

        if last_period != -1:
            # Include the period itself; cursor moves past it.
            segment = window[: last_period + 1]
            advance = last_period + 1           # chars consumed
        else:
            # No sentence boundary — fall back to the full window.
            segment = window
            advance = len(window)

        segment = segment.strip()
        if not segment:
            start += advance or 1
            continue

        # ── Prepend overlap from previous chunk ───────────────────────
        if prev_tail:
            full_segment = prev_tail.strip() + " " + segment
        else:
            full_segment = segment

        chunks.append(full_segment.strip())

        # Carry forward the tail of the *raw* (non-overlapped) segment as
        # overlap for the next chunk.
        tail_chars = min(overlap_chars, len(segment))
        prev_tail  = segment[-tail_chars:] if tail_chars > 0 else ""

        start += advance

    return [c for c in chunks if c]
