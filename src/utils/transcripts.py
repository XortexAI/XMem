"""
Deterministic parsers for transcript uploads used by the context importer.

The public API only needs user/assistant message pairs. Tool calls, tool
results, thinking blocks, and CLI bootstrap messages are intentionally ignored.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict


class ParsedMessagePair(TypedDict):
    user_query: str
    agent_response: str


_ASSISTANT_ROLES = {"assistant", "model", "claude", "gemini", "cursor"}
_USER_ROLES = {"user", "human"}
_SKIPPED_BLOCK_TYPES = {
    "tool_result",
    "tool_use",
    "thinking",
    "redacted_thinking",
    "server_tool_use",
    "web_search_tool_result",
}
_TOOL_MARKDOWN_RE = re.compile(
    r"(?ms)^\*\*Tool (?:Command|Response)\*\*:\s*\n```(?:json)?\n.*?\n```\s*"
)


def parse_transcript_text(text: str) -> tuple[str, list[ParsedMessagePair]]:
    """Parse transcript text and return ``(format, pairs)``.

    Supported deterministic formats:
    - Cursor markdown exports
    - Antigravity markdown exports
    - Claude Code JSONL session transcripts
    - Claude-style role-heading markdown/plain text exports
    - Gemini CLI ``/chat share`` JSON and markdown exports
    """

    normalized = text.replace("\r\n", "\n")

    if "_Exported on" in normalized and "from Cursor" in normalized:
        pairs = _parse_cursor_transcript(normalized)
        if pairs:
            return "cursor", pairs

    if "# Chat Conversation" in normalized and (
        "### User Input" in normalized or "### Planner Response" in normalized
    ):
        pairs = _parse_antigravity_transcript(normalized)
        if pairs:
            return "antigravity", pairs

    json_format, json_pairs = _parse_json_or_jsonl_transcript(normalized)
    if json_pairs:
        return json_format, json_pairs

    gemini_pairs = _parse_role_heading_transcript(
        normalized,
        assistant_roles={"model", "gemini"},
        skip_gemini_bootstrap=True,
    )
    if gemini_pairs and _looks_like_gemini_markdown(normalized):
        return "gemini", gemini_pairs

    claude_pairs = _parse_role_heading_transcript(
        normalized,
        assistant_roles={"assistant", "claude"},
        skip_gemini_bootstrap=False,
    )
    if claude_pairs and _looks_like_claude_export(normalized):
        return "claude_code", claude_pairs

    return "unknown", []


def _parse_cursor_transcript(text: str) -> list[ParsedMessagePair]:
    """Parse a Cursor-exported markdown transcript into message pairs."""
    pairs: list[ParsedMessagePair] = []
    sections = text.split("---")

    start_idx = 0
    if sections and "Exported on" in sections[0]:
        start_idx = 1

    current_user_query: str | None = None

    for section in sections[start_idx:]:
        section = section.strip()
        if not section:
            continue

        if section.startswith("**User**"):
            current_user_query = section.replace("**User**", "", 1).strip()
        elif section.startswith("**Cursor**") or section.startswith("**Assistant**"):
            content = (
                section.replace("**Cursor**", "", 1)
                .replace("**Assistant**", "", 1)
                .strip()
            )
            if current_user_query:
                pairs.append(
                    {
                        "user_query": current_user_query,
                        "agent_response": content,
                    }
                )
                current_user_query = None

    return pairs


def _parse_antigravity_transcript(text: str) -> list[ParsedMessagePair]:
    """Parse an Antigravity-exported markdown transcript into message pairs."""
    pairs: list[ParsedMessagePair] = []
    blocks = re.split(r"(?m)^(###\s+.+)$", text)

    current_user_query: str | None = None
    planner_chunks: list[str] = []

    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue

        if re.match(r"###\s+User Input", block, re.IGNORECASE):
            if current_user_query and planner_chunks:
                pairs.append(
                    {
                        "user_query": current_user_query,
                        "agent_response": "\n\n".join(planner_chunks).strip(),
                    }
                )
                planner_chunks = []
            current_user_query = None

        elif re.match(r"###\s+Planner Response", block, re.IGNORECASE):
            continue

        elif i > 0:
            prev_heading = blocks[i - 1].strip() if i >= 1 else ""
            if re.match(r"###\s+User Input", prev_heading, re.IGNORECASE):
                if current_user_query and planner_chunks:
                    pairs.append(
                        {
                            "user_query": current_user_query,
                            "agent_response": "\n\n".join(planner_chunks).strip(),
                        }
                    )
                    planner_chunks = []
                current_user_query = block

            elif re.match(r"###\s+Planner Response", prev_heading, re.IGNORECASE):
                if block:
                    planner_chunks.append(block)

    if current_user_query and planner_chunks:
        pairs.append(
            {
                "user_query": current_user_query,
                "agent_response": "\n\n".join(planner_chunks).strip(),
            }
        )

    return pairs


def _parse_json_or_jsonl_transcript(
    text: str,
) -> tuple[str, list[ParsedMessagePair]]:
    records = _load_jsonl_records(text)
    if records:
        return _detect_record_format(records), _pair_role_records(records)

    payload = _load_json_payload(text)
    records = _records_from_json_payload(payload)
    if records:
        return _detect_record_format(records), _pair_role_records(records)

    return "unknown", []


def _load_jsonl_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(non_empty_lines) < 2:
        return []

    for line in non_empty_lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(value, dict):
            return []
        records.append(value)

    return records


def _load_json_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _records_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    for key in ("history", "messages", "conversation", "items", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    return []


def _detect_record_format(records: list[dict[str, Any]]) -> str:
    if any(
        "sessionId" in record
        or "parentUuid" in record
        or ("message" in record and record.get("type") in {"user", "assistant"})
        for record in records
    ):
        return "claude_code"

    if any(
        record.get("role") == "model"
        or "parts" in record
        or (
            isinstance(record.get("message"), dict)
            and record["message"].get("role") == "model"
        )
        for record in records
    ):
        return "gemini"

    return "json"


def _pair_role_records(records: list[dict[str, Any]]) -> list[ParsedMessagePair]:
    pairs: list[ParsedMessagePair] = []
    current_user_query: str | None = None
    assistant_chunks: list[str] = []

    def flush_pair() -> None:
        nonlocal current_user_query, assistant_chunks
        if current_user_query and assistant_chunks:
            pairs.append(
                {
                    "user_query": current_user_query,
                    "agent_response": "\n\n".join(assistant_chunks).strip(),
                }
            )
        assistant_chunks = []

    for record in records:
        role = _record_role(record)
        if role not in _USER_ROLES and role not in _ASSISTANT_ROLES:
            continue

        if role in _USER_ROLES and _record_has_tool_result(record):
            continue

        text = _record_text(record)
        if not text or _is_gemini_cli_setup_text(text):
            continue

        if role in _USER_ROLES:
            flush_pair()
            current_user_query = text
        elif current_user_query:
            assistant_chunks.append(text)

    flush_pair()
    return pairs


def _record_role(record: dict[str, Any]) -> str:
    message = record.get("message")
    raw_role = ""

    if isinstance(message, dict):
        raw_role = str(message.get("role") or "")

    raw_role = raw_role or str(record.get("role") or record.get("type") or "")
    role = raw_role.lower()

    if role == "human":
        return "user"
    if role in {"assistant", "model", "user"}:
        return role
    return role


def _record_text(record: dict[str, Any]) -> str:
    message = record.get("message")
    source = message if isinstance(message, dict) else record

    if "content" in source:
        return _extract_text(source.get("content"))
    if "parts" in source:
        return _extract_text(source.get("parts"))
    if "text" in source:
        return _extract_text(source.get("text"))

    return ""


def _record_has_tool_result(record: dict[str, Any]) -> bool:
    message = record.get("message")
    source = message if isinstance(message, dict) else record
    return _has_tool_result(source.get("content")) or _has_tool_result(source.get("parts"))


def _extract_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return _clean_text(value)

    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
                continue

            if not isinstance(item, dict):
                continue

            block_type = str(item.get("type") or "").lower()
            if block_type in _SKIPPED_BLOCK_TYPES:
                continue
            if item.get("functionCall") or item.get("functionResponse"):
                continue

            if "text" in item:
                chunks.append(str(item["text"]))
            elif "content" in item:
                nested = _extract_text(item["content"])
                if nested:
                    chunks.append(nested)
            elif "parts" in item:
                nested = _extract_text(item["parts"])
                if nested:
                    chunks.append(nested)

        return _clean_text("\n\n".join(chunk for chunk in chunks if chunk))

    if isinstance(value, dict):
        block_type = str(value.get("type") or "").lower()
        if block_type in _SKIPPED_BLOCK_TYPES:
            return ""
        if value.get("functionCall") or value.get("functionResponse"):
            return ""
        if "text" in value:
            return _clean_text(str(value["text"]))
        if "content" in value:
            return _extract_text(value["content"])
        if "parts" in value:
            return _extract_text(value["parts"])

    return ""


def _has_tool_result(value: Any) -> bool:
    if isinstance(value, list):
        return any(_has_tool_result(item) for item in value)
    if isinstance(value, dict):
        block_type = str(value.get("type") or "").lower()
        if block_type == "tool_result" or value.get("functionResponse"):
            return True
        return _has_tool_result(value.get("content")) or _has_tool_result(
            value.get("parts")
        )
    return False


def _parse_role_heading_transcript(
    text: str,
    *,
    assistant_roles: set[str],
    skip_gemini_bootstrap: bool,
) -> list[ParsedMessagePair]:
    pattern = re.compile(
        r"(?im)^(?:#{1,6}\s*)?(USER|HUMAN|ASSISTANT|CLAUDE|MODEL|GEMINI)"
        r"(?:[^\S\n]+[^\n]*)?[^\S\n]*$"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        pattern = re.compile(
            r"(?im)^(USER|HUMAN|ASSISTANT|CLAUDE|MODEL|GEMINI)\s*:\s*"
        )
        matches = list(pattern.finditer(text))
        return _parse_inline_role_labels(text, matches, assistant_roles)

    pairs: list[ParsedMessagePair] = []
    current_user_query: str | None = None
    assistant_chunks: list[str] = []

    def flush_pair() -> None:
        nonlocal current_user_query, assistant_chunks
        if current_user_query and assistant_chunks:
            pairs.append(
                {
                    "user_query": current_user_query,
                    "agent_response": "\n\n".join(assistant_chunks).strip(),
                }
            )
        assistant_chunks = []

    for index, match in enumerate(matches):
        role = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = _clean_text(text[start:end])
        content = _strip_tool_markdown(content)
        if not content:
            continue
        if skip_gemini_bootstrap and _is_gemini_cli_setup_text(content):
            continue

        if role in _USER_ROLES:
            flush_pair()
            current_user_query = content
        elif role in assistant_roles and current_user_query:
            assistant_chunks.append(content)

    flush_pair()
    return pairs


def _parse_inline_role_labels(
    text: str,
    matches: list[re.Match[str]],
    assistant_roles: set[str],
) -> list[ParsedMessagePair]:
    pairs: list[ParsedMessagePair] = []
    current_user_query: str | None = None
    assistant_chunks: list[str] = []

    def flush_pair() -> None:
        nonlocal current_user_query, assistant_chunks
        if current_user_query and assistant_chunks:
            pairs.append(
                {
                    "user_query": current_user_query,
                    "agent_response": "\n\n".join(assistant_chunks).strip(),
                }
            )
        assistant_chunks = []

    for index, match in enumerate(matches):
        role = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = _strip_tool_markdown(_clean_text(text[start:end]))
        if not content:
            continue

        if role in _USER_ROLES:
            flush_pair()
            current_user_query = content
        elif role in assistant_roles and current_user_query:
            assistant_chunks.append(content)

    flush_pair()
    return pairs


def _strip_tool_markdown(text: str) -> str:
    return _clean_text(_TOOL_MARKDOWN_RE.sub("", text))


def _clean_text(text: str) -> str:
    return text.strip().strip("-").strip()


def _is_gemini_cli_setup_text(text: str) -> bool:
    return text.startswith("This is the Gemini CLI. We are setting up the context")


def _looks_like_gemini_markdown(text: str) -> bool:
    return bool(
        re.search(r"(?im)^#{1,6}\s*USER\b", text)
        and re.search(r"(?im)^#{1,6}\s*(MODEL|GEMINI)\b", text)
    )


def _looks_like_claude_export(text: str) -> bool:
    if "Claude Code" in text or ".claude/projects" in text:
        return True
    return bool(
        re.search(r"(?im)^(?:#{1,6}\s*)?(USER|HUMAN)\b", text)
        and re.search(r"(?im)^(?:#{1,6}\s*)?(ASSISTANT|CLAUDE)\b", text)
    )
