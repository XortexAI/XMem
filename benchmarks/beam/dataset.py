"""Dataset loading and normalization for BEAM."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from benchmarks.common.io import download_file, read_records

from .config import DEFAULT_DATASET_URLS


@dataclass(frozen=True)
class BeamTurn:
    role: str
    content: str
    time_anchor: str = ""
    message_id: str = ""


@dataclass(frozen=True)
class BeamConversation:
    conversation_id: str
    chat_sessions: list[list[BeamTurn]] = field(default_factory=list)


@dataclass(frozen=True)
class BeamExample:
    question_id: str
    conversation_id: str
    question: str
    answer: str
    question_type: str = ""
    rubric: list[str] = field(default_factory=list)
    split: str = "1M"
    chat_sessions: list[list[BeamTurn]] = field(default_factory=list)

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


def download_dataset(split: str, destination: Path) -> Path:
    if split not in DEFAULT_DATASET_URLS:
        known = ", ".join(sorted(DEFAULT_DATASET_URLS))
        raise ValueError(f"Unknown BEAM split '{split}'. Known splits: {known}")
    return download_file(
        DEFAULT_DATASET_URLS[split],
        destination,
        timeout_seconds=300.0,
    )


def load_examples(path: Path, *, split: str = "1M") -> list[BeamExample]:
    records = read_records(path)
    examples: list[BeamExample] = []
    for conv_index, record in enumerate(records):
        conversation_id = str(
            record.get("conversation_id") or f"conversation-{conv_index}"
        )
        chat_sessions = _parse_chat(record.get("chat") or [])
        questions = _parse_probing_questions(record.get("probing_questions") or [])
        for q_index, question_record in enumerate(questions):
            question = _first_text(
                question_record,
                ("question", "query", "prompt", "user_question"),
            )
            if not question:
                continue
            answer = _first_text(
                question_record,
                ("answer", "ideal_response", "gold_answer", "reference"),
            )
            question_type = _first_text(
                question_record,
                ("question_type", "type", "ability", "category"),
            )
            examples.append(
                BeamExample(
                    question_id=str(
                        question_record.get("question_id")
                        or question_record.get("id")
                        or f"{conversation_id}-q-{q_index}"
                    ),
                    conversation_id=conversation_id,
                    question=question,
                    answer=answer,
                    question_type=question_type or "unknown",
                    rubric=[
                        str(item)
                        for item in question_record.get("rubric") or []
                    ],
                    split=split,
                    chat_sessions=chat_sessions,
                )
            )
    return examples


def select_examples(
    examples: Iterable[BeamExample],
    *,
    offset: int = 0,
    limit: int | None = None,
    question_type: str | None = None,
) -> list[BeamExample]:
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
    example: BeamExample,
    *,
    user_id: str,
    effort_level: str = "low",
) -> list[IngestItem]:
    items: list[IngestItem] = []
    for session in example.chat_sessions:
        for user_turns, agent_turn in _exchange_pairs(session):
            first_user_turn = user_turns[0]
            items.append(
                IngestItem(
                    user_query="\n".join(_format_turn(turn) for turn in user_turns),
                    agent_response=_format_turn(agent_turn),
                    user_id=user_id,
                    session_datetime=first_user_turn.time_anchor,
                    effort_level=effort_level,
                )
            )
    return items


def _parse_chat(raw_chat: Any) -> list[list[BeamTurn]]:
    raw_chat = _coerce_literal(raw_chat)
    if not isinstance(raw_chat, list):
        return []
    if raw_chat and isinstance(raw_chat[0], dict):
        raw_chat = [raw_chat]

    sessions: list[list[BeamTurn]] = []
    for raw_session in raw_chat:
        if not isinstance(raw_session, list):
            continue
        turns = [_parse_turn(item) for item in raw_session]
        turns = [turn for turn in turns if turn and turn.content]
        if turns:
            sessions.append(turns)
    return sessions


def _parse_turn(raw_turn: Any) -> BeamTurn | None:
    if not isinstance(raw_turn, dict):
        return None
    return BeamTurn(
        role=str(raw_turn.get("role") or "message"),
        content=str(raw_turn.get("content") or raw_turn.get("text") or "").strip(),
        time_anchor=str(raw_turn.get("time_anchor") or ""),
        message_id=str(raw_turn.get("id") or raw_turn.get("index") or ""),
    )


def _parse_probing_questions(raw_questions: Any) -> list[dict[str, Any]]:
    raw_questions = _coerce_literal(raw_questions)
    if isinstance(raw_questions, dict):
        questions: list[dict[str, Any]] = []
        for question_type, value in raw_questions.items():
            if isinstance(value, dict):
                item = dict(value)
                item["question_type"] = str(question_type)
                questions.append(item)
            elif isinstance(value, list):
                for question in value:
                    if isinstance(question, dict):
                        item = dict(question)
                        item["question_type"] = str(question_type)
                        questions.append(item)
        return questions
    if isinstance(raw_questions, list):
        return [item for item in raw_questions if isinstance(item, dict)]
    return []


def _coerce_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except ValueError:
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return value


def _first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return str(value).strip()
    return ""


def _format_turn(turn: BeamTurn | None) -> str:
    if turn is None:
        return ""
    prefix = turn.role
    if turn.message_id:
        prefix = f"{prefix} ({turn.message_id})"
    if turn.time_anchor:
        prefix = f"{prefix} [{turn.time_anchor}]"
    return f"{prefix}: {turn.content}"


def _exchange_pairs(
    turns: list[BeamTurn],
) -> list[tuple[list[BeamTurn], BeamTurn]]:
    if any(_is_assistant_role(turn.role) for turn in turns):
        return _role_aware_exchange_pairs(turns)

    pairs: list[tuple[list[BeamTurn], BeamTurn]] = []
    for index in range(0, len(turns) - 1, 2):
        pairs.append(([turns[index]], turns[index + 1]))
    return pairs


def _role_aware_exchange_pairs(
    turns: list[BeamTurn],
) -> list[tuple[list[BeamTurn], BeamTurn]]:
    pairs: list[tuple[list[BeamTurn], BeamTurn]] = []
    pending_user_turns: list[BeamTurn] = []
    for turn in turns:
        if _is_assistant_role(turn.role):
            if pending_user_turns:
                pairs.append((pending_user_turns, turn))
                pending_user_turns = []
            continue
        pending_user_turns.append(turn)
    return pairs


def _is_assistant_role(role: str) -> bool:
    normalized = role.lower().replace("_", " ").replace("-", " ")
    return any(
        label in normalized
        for label in ("assistant", "agent", "bot", "ai", "gpt")
    )
