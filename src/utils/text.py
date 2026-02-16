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


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------


def pack_profiles_into_string(facts: list) -> str:
    """Serialise a list of ProfileFact objects into the prompt format.

    Example output::
        work::company::Google
        basic_info::name::Alice

    """
    lines: list[str] = []
    for f in facts:
        topic = f.topic if hasattr(f, "topic") else f["topic"]
        sub = f.sub_topic if hasattr(f, "sub_topic") else f["sub_topic"]
        memo = f.memo if hasattr(f, "memo") else f["memo"]
        lines.append(
            f"{attribute_unify(topic)}{LLM_TAB_SEPARATOR}"
            f"{attribute_unify(sub)}{LLM_TAB_SEPARATOR}{memo.strip()}"
        )
    return "\n".join(lines) if lines else "NONE"


def parse_raw_response_to_profiles(content: str) -> list[dict]:
    """Parse the raw LLM response into a list of profile fact dicts.

    Expected line format (after the ``---`` separator)::
        TOPIC::SUB_TOPIC::MEMO

    Lines before ``---`` are treated as the LLM's "thinking" and ignored.
    """
    facts: list[dict] = []

    # Skip the thinking section (everything before '---')
    if "---" in content:
        content = content.split("---", maxsplit=1)[1]

    for line in content.strip().splitlines():
        line = line.strip()

        if LLM_TAB_SEPARATOR not in line:
            continue

        parts = line.split(LLM_TAB_SEPARATOR)

        if len(parts) >= 3:
            topic = parts[0].strip().lower()
            sub_topic = parts[1].strip().lower()
            memo = LLM_TAB_SEPARATOR.join(parts[2:]).strip()  # rejoin if memo had separator

            if topic and sub_topic and memo:
                facts.append({
                    "topic": topic,
                    "sub_topic": sub_topic,
                    "memo": memo,
                })

    return facts
