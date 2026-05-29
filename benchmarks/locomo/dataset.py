"""Dataset loading and normalization for LoCoMo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from benchmarks.common.io import download_file, read_records

from .config import DEFAULT_DATASET_URL


@dataclass(frozen=True)
class ConversationTurn:
    speaker: str
    content: str
    dialog_id: str = ""


@dataclass(frozen=True)
class ConversationSession:
    session_id: str
    date: str = ""
    turns: list[ConversationTurn] = field(default_factory=list)


@dataclass(frozen=True)
class LoCoMoExample:
    question_id: str
    sample_id: str
    question: str
    answer: str
    category: str = ""
    evidence: list[str] = field(default_factory=list)
    sessions: list[ConversationSession] = field(default_factory=list)

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


def download_dataset(destination: Path) -> Path:
    return download_file(DEFAULT_DATASET_URL, destination)


def load_examples(path: Path) -> list[LoCoMoExample]:
    records = read_records(path)
    examples: list[LoCoMoExample] = []
    for sample_index, record in enumerate(records):
        sample_id = str(record.get("sample_id") or f"sample-{sample_index}")
        sessions = _parse_sessions(record.get("conversation") or {})
        qa_items = record.get("qa") or []
        if isinstance(qa_items, dict):
            qa_items = list(qa_items.values())
        for qa_index, qa in enumerate(qa_items):
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question") or "").strip()
            if not question:
                continue
            examples.append(
                LoCoMoExample(
                    question_id=str(
                        qa.get("question_id")
                        or qa.get("id")
                        or f"{sample_id}-qa-{qa_index}"
                    ),
                    sample_id=sample_id,
                    question=question,
                    answer=str(qa.get("answer") or "").strip(),
                    category=str(qa.get("category") or "unknown").strip(),
                    evidence=[str(item) for item in qa.get("evidence") or []],
                    sessions=sessions,
                )
            )
    return examples


def select_examples(
    examples: Iterable[LoCoMoExample],
    *,
    offset: int = 0,
    limit: int | None = None,
    category: str | None = None,
) -> list[LoCoMoExample]:
    selected = list(examples)
    if category:
        selected = [
            example
            for example in selected
            if example.category.lower() == category.lower()
        ]
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def build_ingest_items(
    example: LoCoMoExample,
    *,
    user_id: str,
    effort_level: str = "low",
) -> list[IngestItem]:
    items: list[IngestItem] = []
    for session in example.sessions:
        for user_turns, agent_turn in _exchange_pairs(session.turns):
            items.append(
                IngestItem(
                    user_query="\n".join(_format_turn(turn) for turn in user_turns),
                    agent_response=_format_turn(agent_turn),
                    user_id=user_id,
                    session_datetime=session.date,
                    effort_level=effort_level,
                )
            )
    return items


def _parse_sessions(conversation: Any) -> list[ConversationSession]:
    if not isinstance(conversation, dict):
        return []

    session_numbers = sorted({
        _session_number(key)
        for key, value in conversation.items()
        if key.startswith("session_")
        and not key.endswith("_date_time")
        and isinstance(value, list)
    })
    sessions: list[ConversationSession] = []
    for number in session_numbers:
        session_key = f"session_{number}"
        date_key = f"session_{number}_date_time"
        turns = _parse_turns(conversation.get(session_key) or [])
        if turns:
            sessions.append(
                ConversationSession(
                    session_id=session_key,
                    date=str(conversation.get(date_key) or ""),
                    turns=turns,
                )
            )
    return sessions


def _parse_turns(raw_turns: list[Any]) -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    for raw_turn in raw_turns:
        if not isinstance(raw_turn, dict):
            continue
        content = str(raw_turn.get("text") or raw_turn.get("content") or "").strip()
        caption = str(raw_turn.get("blip_caption") or "").strip()
        if caption:
            content = f"{content}\nImage caption: {caption}".strip()
        if not content:
            continue
        turns.append(
            ConversationTurn(
                speaker=str(raw_turn.get("speaker") or "speaker").strip(),
                content=content,
                dialog_id=str(raw_turn.get("dia_id") or raw_turn.get("id") or ""),
            )
        )
    return turns


def _format_turn(turn: ConversationTurn | None) -> str:
    if turn is None:
        return ""
    prefix = f"{turn.speaker}"
    if turn.dialog_id:
        prefix = f"{prefix} ({turn.dialog_id})"
    return f"{prefix}: {turn.content}"


def _exchange_pairs(
    turns: list[ConversationTurn],
) -> list[tuple[list[ConversationTurn], ConversationTurn]]:
    if any(_is_assistant_speaker(turn.speaker) for turn in turns):
        return _role_aware_exchange_pairs(turns)

    pairs: list[tuple[list[ConversationTurn], ConversationTurn]] = []
    for index in range(0, len(turns) - 1, 2):
        pairs.append(([turns[index]], turns[index + 1]))
    return pairs


def _role_aware_exchange_pairs(
    turns: list[ConversationTurn],
) -> list[tuple[list[ConversationTurn], ConversationTurn]]:
    pairs: list[tuple[list[ConversationTurn], ConversationTurn]] = []
    pending_user_turns: list[ConversationTurn] = []
    for turn in turns:
        if _is_assistant_speaker(turn.speaker):
            if pending_user_turns:
                pairs.append((pending_user_turns, turn))
                pending_user_turns = []
            continue
        pending_user_turns.append(turn)
    return pairs


def _is_assistant_speaker(speaker: str) -> bool:
    normalized = speaker.lower().replace("_", " ").replace("-", " ")
    return any(
        label in normalized
        for label in ("assistant", "agent", "bot", "ai", "gpt")
    )


def _session_number(key: str) -> int:
    try:
        return int(key.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0
