from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.code import (
    AnnotationSeverity,
    AnnotationType,
    ExtractedAnnotation,
    SnippetRecord,
    SnippetType,
    annotations_namespace,
    snippets_namespace,
    symbols_namespace,
)
from src.schemas.events import EventData, EventResult
from src.schemas.judge import JudgeResult, Operation, OperationType
from src.schemas.profile import ProfileFact, ProfileResult
from src.schemas.retrieval import RetrievalResult, SourceRecord
from src.schemas.summary import SummaryResult
from src.schemas.weaver import ExecutedOp, OpStatus, WeaverResult


def test_judge_and_weaver_properties_validate_confidence_and_counts():
    with pytest.raises(ValidationError):
        JudgeResult(confidence=1.1)

    judge = JudgeResult(
        operations=[
            Operation(type=OperationType.NOOP),
            Operation(type=OperationType.ADD, content="new fact"),
        ],
        confidence=0.8,
    )
    assert not judge.is_empty
    assert judge.has_writes

    weaver = WeaverResult(
        executed=[
            ExecutedOp(type=OperationType.ADD, status=OpStatus.SUCCESS),
            ExecutedOp(type=OperationType.DELETE, status=OpStatus.FAILED),
            ExecutedOp(type=OperationType.NOOP, status=OpStatus.SKIPPED),
        ]
    )
    assert (weaver.total, weaver.succeeded, weaver.failed, weaver.skipped) == (3, 1, 1, 1)


def test_profile_event_summary_and_retrieval_models_expose_empty_helpers():
    assert ProfileResult().is_empty
    assert not ProfileResult(facts=[ProfileFact(topic="work", sub_topic="company", memo="XMem")]).is_empty

    with pytest.raises(ValidationError):
        EventData()

    events = EventResult(events=[EventData(date="05-11", event_name="Launch")])
    assert not events.is_empty
    assert events.event.event_name == "Launch"

    assert SummaryResult(summary=" ").is_empty
    retrieval = RetrievalResult(
        query="q",
        answer="a",
        sources=[SourceRecord(domain="summary", content="memory", score=0.7)],
    )
    assert retrieval.source_count == 1


def test_code_schema_enums_and_namespace_helpers():
    annotation = ExtractedAnnotation(
        target_file="src/payments.py",
        content="Retry path can duplicate charges",
        annotation_type=AnnotationType.BUG_REPORT,
        severity=AnnotationSeverity.HIGH,
    )
    snippet = SnippetRecord(
        user_id="user-1",
        content="Binary search helper",
        snippet_type=SnippetType.ALGORITHM,
        tags=["python", "search"],
    )

    assert annotation.annotation_type.value == "bug_report"
    assert snippet.tags == ["python", "search"]
    assert symbols_namespace("acme", "payments") == "acme:payments:symbols"
    assert annotations_namespace("acme") == "acme:annotations"
    assert snippets_namespace("user-1") == "user-1:snippets"
