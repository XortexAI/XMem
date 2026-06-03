from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.pipelines.weaver import Weaver
from src.schemas.judge import JudgeDomain, JudgeResult, Operation, OperationType
from src.schemas.weaver import OpStatus

NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_weaver_batches_vector_add_update_and_delete(vector_store, fast_embed):
    vector_store.seed(
        "profile-1",
        "work / company = OldCo",
        {"user_id": "user-1", "domain": "profile", "main_content": "work_company"},
    )
    weaver = Weaver(vector_store=vector_store, embed_fn=fast_embed)
    result = await weaver.execute(
        JudgeResult(
            operations=[
                Operation(type=OperationType.ADD, content="work / title = Engineer"),
                Operation(type=OperationType.UPDATE, embedding_id="profile-1", content="work / company = XMem"),
                Operation(type=OperationType.DELETE, embedding_id="missing"),
            ],
            confidence=1.0,
        ),
        JudgeDomain.PROFILE,
        "user-1",
    )

    assert result.succeeded == 3
    assert vector_store.add_calls[0]["texts"] == ["work / title = Engineer"]
    assert vector_store.records["profile-1"]["content"] == "work / company = XMem"
    assert vector_store.records["profile-1"]["metadata"]["main_content"] == "work_company"


@pytest.mark.asyncio
async def test_weaver_temporal_uses_injected_graph_callables():
    calls = []

    async def create_event(**kwargs):
        calls.append(("create", kwargs))

    async def update_event(**kwargs):
        calls.append(("update", kwargs))

    async def delete_event(**kwargs):
        calls.append(("delete", kwargs))

    weaver = Weaver(
        graph_create_event=create_event,
        graph_update_event=update_event,
        graph_delete_event=delete_event,
    )
    result = await weaver.execute(
        JudgeResult(
            operations=[
                Operation(type=OperationType.ADD, content="05-11 | Demo | Launch | 2026 | 10:00 | today"),
                Operation(type=OperationType.DELETE, embedding_id="05-10_Demo"),
            ],
            confidence=1.0,
        ),
        JudgeDomain.TEMPORAL,
        "user-1",
    )

    assert result.succeeded == 2
    assert calls[0] == (
        "create",
        {
            "user_id": "user-1",
            "date_str": "05-11",
            "event_data": {
                "event_name": "Demo",
                "desc": "Launch",
                "year": "2026",
                "time": "10:00",
                "date_expression": "today",
            },
        },
    )
    assert calls[1] == ("delete", {"user_id": "user-1", "embedding_id": "05-10_Demo"})


@pytest.mark.asyncio
async def test_weaver_code_and_snippet_paths_store_structured_metadata(vector_store, fast_embed):
    graph_annotations = []

    async def create_annotation(**kwargs):
        graph_annotations.append(kwargs)
        return "ann-1"

    weaver = Weaver(
        vector_store=vector_store,
        code_vector_store=vector_store,
        snippet_vector_store=vector_store,
        embed_fn=fast_embed,
        graph_create_annotation=create_annotation,
    )
    code = await weaver.execute(
        JudgeResult(operations=[
            Operation(
                type=OperationType.ADD,
                content="bug_report | Payment.process | src/payments.py | payments | high | Duplicate charge risk",
            )
        ], confidence=1.0),
        JudgeDomain.CODE,
        "user-1",
    )
    snippet = await weaver.execute(
        JudgeResult(operations=[
            Operation(type=OperationType.ADD, content="Binary search | def bs(): pass | python | algorithm | search,array")
        ], confidence=1.0),
        JudgeDomain.SNIPPET,
        "user-1",
    )

    assert code.executed[0].status == OpStatus.SUCCESS
    assert snippet.executed[0].status == OpStatus.SUCCESS
    assert graph_annotations[0]["target_symbol"] == "Payment.process"
    assert vector_store.add_calls[-1]["metadata"][0]["domain"] == "snippet"
    assert vector_store.add_calls[-1]["metadata"][0]["language"] == "python"


@pytest.mark.asyncio
async def test_weaver_forget_lifecycle_metadata_stored(vector_store, fast_embed):
    """extra_metadata carrying forget/TTL fields is written into every ADD record."""
    expires_at = (NOW + timedelta(days=30)).isoformat()
    lifecycle = {
        "forget": True,
        "expires_at": expires_at,
        "lifecycle_state": "active",
    }
    weaver = Weaver(
        vector_store=vector_store,
        embed_fn=fast_embed,
        _now=lambda: NOW,
    )
    result = await weaver.execute(
        JudgeResult(
            operations=[
                Operation(type=OperationType.ADD, content="work / title = Engineer"),
                Operation(type=OperationType.ADD, content="work / company = XMem"),
            ],
            confidence=1.0,
        ),
        JudgeDomain.PROFILE,
        "user-1",
        extra_metadata=lifecycle,
    )

    assert result.succeeded == 2
    for record in vector_store.records.values():
        meta = record["metadata"]
        assert meta.get("forget") is True
        assert meta.get("expires_at") == expires_at
        assert meta.get("lifecycle_state") == "active"
        # Core metadata must still be present
        assert meta.get("user_id") == "user-1"
        assert meta.get("domain") == "profile"


@pytest.mark.asyncio
async def test_weaver_extra_metadata_none_no_regression(vector_store, fast_embed):
    """extra_metadata=None must not change any existing behaviour."""
    weaver = Weaver(vector_store=vector_store, embed_fn=fast_embed)
    result = await weaver.execute(
        JudgeResult(
            operations=[Operation(type=OperationType.ADD, content="work / title = Engineer")],
            confidence=1.0,
        ),
        JudgeDomain.PROFILE,
        "user-1",
    )
    assert result.succeeded == 1
    meta = list(vector_store.records.values())[0]["metadata"]
    assert "forget" not in meta
    assert "expires_at" not in meta
