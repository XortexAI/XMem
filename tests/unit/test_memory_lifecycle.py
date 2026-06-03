"""
Unit tests for memory lifecycle (forget/TTL) — Issue #166 PR #1.

All tests use InMemoryVectorStore; no live services are called.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.pipelines.lifecycle import build_lifecycle_metadata, is_retrievable
from src.pipelines.weaver import Weaver
from src.schemas.judge import JudgeDomain, JudgeResult, Operation, OperationType
from tests.conftest import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
FUTURE = NOW + timedelta(days=30)
PAST = NOW - timedelta(days=1)


def _fixed_now(dt: datetime):
    return lambda: dt


# ---------------------------------------------------------------------------
# 1. is_retrievable — legacy records always pass through
# ---------------------------------------------------------------------------

def test_is_retrievable_legacy_record():
    """Records stored before lifecycle was introduced must be retrievable."""
    assert is_retrievable({}, NOW) is True
    assert is_retrievable({"user_id": "u1", "domain": "summary"}, NOW) is True


def test_is_retrievable_forgotten_state():
    assert is_retrievable({"lifecycle_state": "forgotten"}, NOW) is False


def test_is_retrievable_ttl_not_expired():
    meta = {
        "forget": True,
        "expires_at": FUTURE.isoformat(),
        "lifecycle_state": "active",
    }
    assert is_retrievable(meta, NOW) is True


def test_is_retrievable_ttl_expired():
    meta = {
        "forget": True,
        "expires_at": PAST.isoformat(),
        "lifecycle_state": "active",
    }
    assert is_retrievable(meta, NOW) is False


def test_is_retrievable_forget_true_no_expires_at():
    """forget=true with no expires_at should NOT hide the record (missing key = legacy-safe)."""
    meta = {"forget": True}
    assert is_retrievable(meta, NOW) is True


def test_is_retrievable_forget_false_ignores_expires_at():
    """forget=false means the record is always active regardless of expires_at."""
    meta = {"forget": False, "expires_at": PAST.isoformat()}
    assert is_retrievable(meta, NOW) is True


# ---------------------------------------------------------------------------
# 2. build_lifecycle_metadata
# ---------------------------------------------------------------------------

def test_build_lifecycle_metadata_sets_correct_fields():
    meta = build_lifecycle_metadata(now=NOW, ttl_days=7.0)
    assert meta["forget"] is True
    assert meta["lifecycle_state"] == "active"

    # expires_at must be NOW + 7 days
    expected = (NOW + timedelta(days=7.0)).isoformat()
    assert meta["expires_at"] == expected

    assert meta["created_at"] == NOW.isoformat()
    assert meta["updated_at"] == NOW.isoformat()


def test_build_lifecycle_metadata_with_reason():
    meta = build_lifecycle_metadata(now=NOW, ttl_days=1.0, reason="gdpr")
    assert meta.get("forget_reason") == "gdpr"


# ---------------------------------------------------------------------------
# 3. Weaver — extra_metadata merged onto stored records
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_weaver_extra_metadata_merged_on_add(vector_store, fast_embed):
    """extra_metadata is merged onto the record stored by a batched ADD."""
    lifecycle_meta = {
        "forget": True,
        "expires_at": FUTURE.isoformat(),
        "lifecycle_state": "active",
    }
    weaver = Weaver(
        vector_store=vector_store,
        embed_fn=fast_embed,
        _now=_fixed_now(NOW),
    )
    result = await weaver.execute(
        JudgeResult(
            operations=[Operation(type=OperationType.ADD, content="I like hiking")],
            confidence=1.0,
        ),
        JudgeDomain.SUMMARY,
        "u1",
        extra_metadata=lifecycle_meta,
    )

    assert result.succeeded == 1
    stored_meta = list(vector_store.records.values())[0]["metadata"]
    assert stored_meta["forget"] is True
    assert stored_meta["expires_at"] == FUTURE.isoformat()
    assert stored_meta["user_id"] == "u1"
    assert stored_meta["domain"] == "summary"


@pytest.mark.asyncio
async def test_weaver_extra_metadata_none_is_noop(vector_store, fast_embed):
    """extra_metadata=None leaves metadata exactly as before — no regression."""
    weaver = Weaver(vector_store=vector_store, embed_fn=fast_embed)
    result = await weaver.execute(
        JudgeResult(
            operations=[Operation(type=OperationType.ADD, content="I like coffee")],
            confidence=1.0,
        ),
        JudgeDomain.SUMMARY,
        "u1",
    )
    assert result.succeeded == 1
    stored_meta = list(vector_store.records.values())[0]["metadata"]
    assert "forget" not in stored_meta
    assert "expires_at" not in stored_meta


# ---------------------------------------------------------------------------
# 4. _search_summary filters expired records
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_summary_filters_expired_records(vector_store):
    """Expired forget records are excluded; non-forget records are unaffected."""
    # A normal record — always retrievable
    vector_store.seed("live-1", "normal summary", {
        "user_id": "u1", "domain": "summary",
    }, score=0.9)

    # A forget record whose TTL has not yet expired — still retrievable
    vector_store.seed("forget-live", "temporary note (live)", {
        "user_id": "u1", "domain": "summary",
        "forget": True,
        "expires_at": FUTURE.isoformat(),
        "lifecycle_state": "active",
    }, score=0.8)

    # A forget record whose TTL has passed — must be hidden
    vector_store.seed("forget-expired", "temporary note (expired)", {
        "user_id": "u1", "domain": "summary",
        "forget": True,
        "expires_at": PAST.isoformat(),
        "lifecycle_state": "active",
    }, score=0.7)

    from src.pipelines.retrieval import RetrievalPipeline
    from tests.conftest import FakeChatModel, FakeNeo4jClient

    pipeline = RetrievalPipeline(
        model=FakeChatModel(),
        vector_store=vector_store,
        neo4j_client=FakeNeo4jClient(),
        _now=_fixed_now(NOW),
    )
    records = await pipeline._search_summary("query", "u1", top_k=10)

    ids = [r.metadata["id"] for r in records]
    assert "live-1" in ids
    assert "forget-live" in ids
    assert "forget-expired" not in ids


@pytest.mark.asyncio
async def test_search_summary_no_regression_without_lifecycle_keys(vector_store):
    """Legacy records with no lifecycle keys must still be returned."""
    for i in range(3):
        vector_store.seed(f"legacy-{i}", f"legacy content {i}", {
            "user_id": "u1", "domain": "summary",
        })

    from src.pipelines.retrieval import RetrievalPipeline
    from tests.conftest import FakeChatModel, FakeNeo4jClient

    pipeline = RetrievalPipeline(
        model=FakeChatModel(),
        vector_store=vector_store,
        neo4j_client=FakeNeo4jClient(),
        _now=_fixed_now(NOW),
    )
    records = await pipeline._search_summary("query", "u1", top_k=10)
    assert len(records) == 3


# ---------------------------------------------------------------------------
# 5. conftest merge fix — InMemoryVectorStore.update() merges, not replaces
# ---------------------------------------------------------------------------

def test_fake_update_merges_metadata():
    """update() must merge metadata, not replace it.

    This test FAILS on the old replace-fake and PASSES after the merge fix,
    locking the correction so future regressions surface immediately.
    """
    store = InMemoryVectorStore()
    store.seed("r1", "content", {"a": 1, "b": 2})

    store.update("r1", metadata={"b": 99, "c": 3})

    meta = store.records["r1"]["metadata"]
    assert meta["a"] == 1, "Pre-existing key must be preserved"
    assert meta["b"] == 99, "Updated key must reflect new value"
    assert meta["c"] == 3, "New key must be added"


# ---------------------------------------------------------------------------
# 6. is_retrievable — None guard + datetime short-circuit
# ---------------------------------------------------------------------------

def test_is_retrievable_none_metadata():
    """is_retrievable(None, now) must return True (legacy-safe)."""
    assert is_retrievable(None, NOW) is True


def test_is_retrievable_datetime_expires_at():
    """is_retrievable must handle a real datetime object (isinstance short-circuit)."""
    meta_past = {"forget": True, "expires_at": PAST, "lifecycle_state": "active"}
    meta_future = {"forget": True, "expires_at": FUTURE, "lifecycle_state": "active"}
    assert is_retrievable(meta_past, NOW) is False
    assert is_retrievable(meta_future, NOW) is True
