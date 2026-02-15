"""
Text parsing and formatting utilities for structured LLM responses.

All pack_* functions convert structured data → prompt strings.
All parse_* functions convert raw LLM output → structured data.
"""

from __future__ import annotations

from typing import List

from src.config.constants import LLM_TAB_SEPARATOR
from src.schemas.classification import Classification


def attribute_unify(value: str) -> str:
    return value.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

_VALID_SOURCES = frozenset({"code", "profile", "event"})


def pack_classifications_into_string(
    classifications: List[Classification],
) -> str:
    """Serialise a list of classifications into the tab-separated prompt format.

    Example output::
        profile::My name is Alice
        code::Write me a hello-world script
    """
    lines = [
        f"{c['source']}{LLM_TAB_SEPARATOR}{c['query']}"
        for c in classifications
    ]
    return "\n".join(lines)


def parse_raw_response_to_classifications(content: str) -> List[Classification]:
    """Parse the raw LLM response into a list of Classification dicts.

    Expected line format::
        SOURCE::QUERY
    """
    classifications: List[Classification] = []

    for line in content.strip().splitlines():
        line = line.strip()
        if LLM_TAB_SEPARATOR not in line:
            continue

        parts = line.split(LLM_TAB_SEPARATOR, maxsplit=1)

        if len(parts) < 2:
            continue

        source = parts[0].strip().lower()
        query = parts[1].strip()

        if source in _VALID_SOURCES and query:
            classifications.append({"source": source, "query": query})

    return classifications
