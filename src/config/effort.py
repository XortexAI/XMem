"""
Effort-level configuration for the XMem pipeline.

Two modes:
  LOW  — fast, fewer LLM calls, acceptable accuracy trade-off
  HIGH — slower, more LLM calls, maximum accuracy

Usage::

    from src.config.effort import EffortLevel, get_effort_config

    config = get_effort_config(EffortLevel.HIGH)
    config.summary_chunk_threshold   # 200
    config.summary_chunk_size        # 100

New knobs can be added to ``EffortConfig`` as the product evolves
(e.g. judge verification depth, embedding reranking, etc.) without
touching call-sites that already receive the config object.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache


class EffortLevel(str, Enum):
    LOW = "low"
    HIGH = "high"


@dataclass(frozen=True)
class EffortConfig:
    """Strategy object holding all effort-dependent knobs.

    Every agent that needs to vary its behavior reads from this
    instead of hard-coding thresholds, keeping the system scalable.
    """

    level: EffortLevel

    # ── Summarizer ────────────────────────────────────────────────
    # Token count above which large inputs get chunked before summarization.
    # LOW  → never chunk (threshold = infinity)
    # HIGH → chunk when input exceeds this many tokens
    summary_chunk_threshold: int
    # Target chunk size (in tokens) when chunking is triggered.
    summary_chunk_size: int
    # Whether to run a merge pass that de-duplicates bullets across chunks.
    summary_merge_pass: bool

    # ── Future knobs (add here as needed) ─────────────────────────
    # judge_verification_depth: int     # e.g. 1 for LOW, 3 for HIGH
    # embedding_rerank: bool            # False for LOW, True for HIGH
    # classifier_multi_pass: bool       # single vs multi-pass classification
    # profiler_cross_check: bool        # verify facts against existing profile


_CONFIGS = {
    EffortLevel.LOW: EffortConfig(
        level=EffortLevel.LOW,
        summary_chunk_threshold=999_999,   # effectively never chunk
        summary_chunk_size=999_999,
        summary_merge_pass=False,
    ),
    EffortLevel.HIGH: EffortConfig(
        level=EffortLevel.HIGH,
        summary_chunk_threshold=200,
        summary_chunk_size=100,
        summary_merge_pass=True,
    ),
}


@lru_cache(maxsize=4)
def get_effort_config(level: EffortLevel | str) -> EffortConfig:
    """Resolve an effort level (string or enum) to its config."""
    if isinstance(level, str):
        level = EffortLevel(level.lower())
    return _CONFIGS[level]
