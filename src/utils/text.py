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


# ---------------------------------------------------------------------------
# Temporal / Event helpers
# ---------------------------------------------------------------------------


def parse_raw_response_to_event(content: str) -> dict | None:
    """Parse the raw LLM response for temporal event extraction.

    Expected format::

        DATE: MM-DD
        EVENT_NAME: <name>
        YEAR: <year or empty>
        DESC: <description>
        TIME: <time or empty>
        DATE_EXPRESSION: <original date expression>

    Or simply ``NO_EVENT``.

    Returns:
        Dict with event data or *None* if no event was found.
    """
    content = content.strip()

    if "NO_EVENT" in content.upper():
        return None

    event_data: dict = {
        "date": None,
        "event_name": None,
        "year": None,
        "desc": None,
        "time": None,
        "date_expression": None,
    }

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        upper = line.upper()

        if upper.startswith("DATE:") and not upper.startswith("DATE_EXPRESSION:"):
            value = line[5:].strip()
            if value:
                event_data["date"] = value

        elif upper.startswith("EVENT_NAME:"):
            value = line[11:].strip()
            if value:
                event_data["event_name"] = value

        elif upper.startswith("YEAR:"):
            value = line[5:].strip()
            if value:
                try:
                    event_data["year"] = int(value)
                except ValueError:
                    event_data["year"] = value

        elif upper.startswith("DESC:"):
            value = line[5:].strip()
            if value:
                event_data["desc"] = value

        elif upper.startswith("TIME:"):
            value = line[5:].strip()
            if value:
                event_data["time"] = value

        elif upper.startswith("DATE_EXPRESSION:"):
            value = line[16:].strip()
            if value:
                event_data["date_expression"] = value

    # Must have at least a date to be valid
    if not event_data["date"]:
        return None

    return event_data


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def parse_raw_response_to_image(content: str) -> dict:
    """Parse the raw LLM response for image analysis.

    Expected format::

        DESCRIPTION: <natural language summary>

        OBSERVATIONS:
        - [category] observation text (confidence: high/medium/low)
        - [category] observation text (confidence: high/medium/low)

    Returns:
        Dict with keys ``description`` (str) and ``observations`` (list of dicts).
        Each observation dict has ``category``, ``description``, and optional ``confidence``.
    """
    content = content.strip()

    result: dict = {
        "description": "",
        "observations": [],
    }

    # --- Extract DESCRIPTION line ---
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("DESCRIPTION:"):
            result["description"] = stripped[len("DESCRIPTION:"):].strip()
            break

    # --- Extract OBSERVATIONS section ---
    in_observations = False
    for line in content.splitlines():
        stripped = line.strip()

        if stripped.upper().startswith("OBSERVATIONS:"):
            in_observations = True
            continue

        if not in_observations or not stripped.startswith("-"):
            continue

        # Remove the leading "- "
        entry = stripped[1:].strip()

        # Parse "[category] description (confidence: level)"
        category = "other"
        confidence = None
        description = entry

        # Extract category from [brackets]
        if entry.startswith("[") and "]" in entry:
            bracket_end = entry.index("]")
            category = entry[1:bracket_end].strip().lower()
            description = entry[bracket_end + 1:].strip()

        # Extract confidence from (confidence: level)
        if "(confidence:" in description.lower():
            idx = description.lower().index("(confidence:")
            conf_part = description[idx:]
            description = description[:idx].strip()

            conf_part = conf_part.strip("()")
            if ":" in conf_part:
                confidence = conf_part.split(":", maxsplit=1)[1].strip().lower()

        if description:
            result["observations"].append({
                "category": category,
                "description": description,
                "confidence": confidence,
            })

    return result
