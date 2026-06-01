"""Dataset loading and normalization for LongMemEval records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx

from .config import DEFAULT_DATASET_URLS


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str


@dataclass(frozen=True)
class ConversationSession:
    session_id: str
    date: str = ""
    turns: list[ConversationTurn] = field(default_factory=list)


@dataclass(frozen=True)
class LongMemEvalExample:
    question_id: str
    question: str
    answer: str
    question_type: str = ""
    sessions: list[ConversationSession] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def user_id_suffix(self) -> str:
        safe = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in self.question_id
        )
        return safe.strip("_") or "example"


@dataclass(frozen=True)
class IngestItem:
    user_query: str
    agent_response: str
    user_id: str
    session_datetime: str = ""
    effort_level: str = "low"


def download_dataset(variant: str, destination: Path) -> Path:
    """Download a known LongMemEval dataset variant to destination."""

    if variant not in DEFAULT_DATASET_URLS:
        known = ", ".join(sorted(DEFAULT_DATASET_URLS))
        raise ValueError(
            f"Unknown dataset variant '{variant}'. Known variants: {known}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream(
        "GET",
        DEFAULT_DATASET_URLS[variant],
        follow_redirects=True,
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return destination


def load_examples(path: Path) -> list[LongMemEvalExample]:
    """Load LongMemEval examples from JSON or JSONL."""

    raw_records = _read_records(path)
    examples = [
        _parse_example(record, index)
        for index, record in enumerate(raw_records)
    ]
    return [example for example in examples if example.question]


def select_examples(
    examples: Iterable[LongMemEvalExample],
    *,
    offset: int = 0,
    limit: int | None = None,
    question_type: str | None = None,
) -> list[LongMemEvalExample]:
    selected = list(examples)
    if question_type:
        selected = [
            example
            for example in selected
            if example.question_type.lower() == question_type.lower()
        ]
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def build_ingest_items(
    example: LongMemEvalExample,
    *,
    user_id: str,
    effort_level: str = "low",
) -> list[IngestItem]:
    """Convert LongMemEval sessions into XMem conversation-turn ingest items."""

    items: list[IngestItem] = []
    for session in example.sessions:
        for user_query, agent_response in _iter_message_pairs(session.turns):
            if not user_query.strip() and not agent_response.strip():
                continue
            items.append(
                IngestItem(
                    user_query=user_query.strip() or "[empty user message]",
                    agent_response=agent_response.strip(),
                    user_id=user_id,
                    session_datetime=session.date,
                    effort_level=effort_level,
                )
            )
    return items


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "examples", "records", "questions"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if all(isinstance(v, dict) for v in payload.values()):
            return list(payload.values())
        return [payload]
    raise ValueError(f"Unsupported dataset payload in {path}")


def _parse_example(record: dict[str, Any], index: int) -> LongMemEvalExample:
    question_id = str(
        record.get("question_id")
        or record.get("id")
        or record.get("sample_id")
        or f"example-{index}"
    )
    sessions = _parse_sessions(record)
    return LongMemEvalExample(
        question_id=question_id,
        question=str(record.get("question") or record.get("query") or "").strip(),
        answer=str(record.get("answer") or record.get("gold_answer") or "").strip(),
        question_type=str(
            record.get("question_type")
            or record.get("category")
            or record.get("type")
            or ""
        ).strip(),
        sessions=sessions,
        metadata={
            key: value
            for key, value in record.items()
            if key not in {"haystack_sessions", "sessions", "conversation", "messages"}
        },
    )


def _parse_sessions(record: dict[str, Any]) -> list[ConversationSession]:
    raw_sessions = (
        record.get("haystack_sessions")
        or record.get("sessions")
        or record.get("conversation")
        or record.get("messages")
        or []
    )
    dates = record.get("haystack_dates") or record.get("session_dates") or []
    if isinstance(raw_sessions, dict):
        raw_sessions = list(raw_sessions.values())
    if _looks_like_turn(raw_sessions):
        raw_sessions = [raw_sessions]

    sessions: list[ConversationSession] = []
    for idx, raw_session in enumerate(raw_sessions):
        date = ""
        if isinstance(dates, list) and idx < len(dates):
            date = str(dates[idx] or "")
        session_id = f"session-{idx + 1}"
        turns_source = raw_session
        if isinstance(raw_session, dict):
            session_id = str(
                raw_session.get("session_id")
                or raw_session.get("id")
                or session_id
            )
            date = str(raw_session.get("date") or raw_session.get("created_at") or date)
            turns_source = (
                raw_session.get("messages")
                or raw_session.get("turns")
                or raw_session.get("conversation")
                or []
            )
        turns = _parse_turns(turns_source)
        if turns:
            sessions.append(
                ConversationSession(session_id=session_id, date=date, turns=turns)
            )
    return sessions


def _parse_turns(raw_turns: Any) -> list[ConversationTurn]:
    if not isinstance(raw_turns, list):
        return []
    turns: list[ConversationTurn] = []
    for raw_turn in raw_turns:
        if isinstance(raw_turn, str):
            turns.append(ConversationTurn(role="user", content=raw_turn))
            continue
        if not isinstance(raw_turn, dict):
            continue
        role = str(
            raw_turn.get("role")
            or raw_turn.get("speaker")
            or raw_turn.get("sender")
            or raw_turn.get("from")
            or ""
        ).lower()
        content = (
            raw_turn.get("content")
            or raw_turn.get("text")
            or raw_turn.get("message")
            or raw_turn.get("utterance")
            or ""
        )
        turns.append(ConversationTurn(role=_normalize_role(role), content=str(content)))
    return [turn for turn in turns if turn.content.strip()]


def _looks_like_turn(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and any(
        key in first
        for key in ("role", "speaker", "content", "text")
    )


def _normalize_role(role: str) -> str:
    if role in {"assistant", "ai", "agent", "bot", "gpt"}:
        return "assistant"
    if role in {"system"}:
        return "system"
    return "user"


def _iter_message_pairs(
    turns: Iterable[ConversationTurn],
) -> Iterable[tuple[str, str]]:
    pending_user: str | None = None
    pending_assistant: list[str] = []

    for turn in turns:
        role = _normalize_role(turn.role)
        content = turn.content.strip()
        if not content:
            continue
        if role == "system":
            continue
        if role == "user":
            if pending_user is not None:
                yield pending_user, "\n\n".join(pending_assistant)
                pending_assistant = []
            pending_user = content
            continue
        if pending_user is None:
            pending_user = "[assistant context]"
        pending_assistant.append(content)

    if pending_user is not None:
        yield pending_user, "\n\n".join(pending_assistant)
