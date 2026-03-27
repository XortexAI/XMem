"""
Retrieval schemas — data models for the retrieval pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RetrievalResult:
    """Final output of the retrieval pipeline."""

    query: str
    answer: str
    sources: List[SourceRecord] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def source_count(self) -> int:
        return len(self.sources)


@dataclass
class SourceRecord:
    """A single piece of evidence fetched from a data store."""

    domain: str              # "profile", "temporal", "summary"
    content: str             # the actual text
    score: float = 0.0       # similarity score (if applicable)
    metadata: Dict[str, Any] = field(default_factory=dict)
