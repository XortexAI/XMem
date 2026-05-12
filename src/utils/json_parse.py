"""
Shared utilities for parsing LLM JSON responses.

Provides a common extraction function used by all agents that need to
parse JSON from LLM outputs (which are often wrapped in markdown fences).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger("xmem.utils.json_parse")

T = TypeVar("T")


def extract_json_from_response(raw: str) -> dict:
    """Extract JSON from an LLM response, stripping markdown code fences.

    Handles both `` ```json ... ``` `` and bare `` ``` ... ``` `` fences.
    Returns an empty dict if extraction or parsing fails.
    """
    try:
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in cleaned:
            parts = cleaned.split("```", 1)
            if len(parts) > 1:
                cleaned = parts[1].split("```", 1)[0]

        return json.loads(cleaned.strip())
    except (json.JSONDecodeError, ValueError, IndexError) as exc:
        logger.warning("Failed to extract JSON from LLM response: %s", exc)
        logger.debug("Raw response: %s", raw[:500])
        return {}


def parse_list_field(
    raw: str,
    field_name: str,
    *,
    model_cls: type[T],
    logger: Optional[logging.Logger] = None,
) -> list[T]:
    """Parse a list of Pydantic models from a JSON field in an LLM response.

    Args:
        raw:         The raw LLM output string.
        field_name:  The key in the JSON object whose value is a list.
        model_cls:   Pydantic model class to instantiate for each item.
        logger:      Optional logger for error reporting.

    Returns:
        A list of parsed model instances (empty list on any failure).
    """
    log = logger or logging.getLogger("xmem.utils.json_parse")
    try:
        data = extract_json_from_response(raw)
        items_data = data.get(field_name, [])
        if not items_data:
            return []

        results: list[T] = []
        for item_dict in items_data:
            if not isinstance(item_dict, dict):
                continue
            try:
                results.append(model_cls(**item_dict))
            except Exception as exc:
                log.warning("Failed to parse item in %s: %s", field_name, exc)

        return results

    except Exception as exc:
        log.error("Failed to parse %s from response: %s", field_name, exc)
        log.debug("Raw response: %s", raw[:500])
        return []


def parse_json_response(
    raw: str,
    result_cls: type[T],
    *,
    expected_field: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> T:
    """Parse an LLM response into a Pydantic result model.

    Args:
        raw:           The raw LLM output string.
        result_cls:    Pydantic model class for the result object.
        expected_field: If provided, extract the list from this field and
                        pass it as a constructor argument named "field_name".
                        The constructor must accept the field as a list of dicts.
        logger:        Optional logger for error reporting.

    Returns:
        An instance of result_cls, populated or empty on failure.
    """
    log = logger or logging.getLogger("xmem.utils.json_parse")
    try:
        data = extract_json_from_response(raw)

        if expected_field:
            field_data = data.get(expected_field, [])
            return result_cls(**{expected_field: field_data})

        return result_cls(**data)

    except Exception as exc:
        log.error("Failed to parse response into %s: %s", result_cls.__name__, exc)
        log.debug("Raw response: %s", raw[:500])
        return result_cls()