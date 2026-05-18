from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.database.control_plane_store import (
    ControlPlaneStore,
    _memory_rate_limits,
    _memory_records,
)


def _memory_store() -> ControlPlaneStore:
    store = object.__new__(ControlPlaneStore)
    store._connected = False
    store._in_memory = True
    store.records = None
    store.rate_limits = None
    return store


class FakeRecordsCollection:
    def __init__(self):
        self.docs = []

    def insert_one(self, record):
        self.docs.append(dict(record))

    def find_one_and_delete(self, query):
        for index, record in enumerate(self.docs):
            if (
                record["record_type"] == query["record_type"]
                and record["token_hash"] == query["token_hash"]
                and record["expires_at"] > query["expires_at"]["$gt"]
            ):
                return self.docs.pop(index)
        return None


class FakeRateLimitCollection:
    def __init__(self):
        self.docs = {}
        self.calls = []

    def find_one_and_update(self, filter_doc, pipeline, upsert=False, return_document=None):
        identity = filter_doc["identity"]
        filter_stage = pipeline[0]["$set"]["hits"]["$filter"]
        allowed_expr = pipeline[1]["$set"]["allowed"]["$lt"]
        update_stage = pipeline[2]["$set"]

        assert filter_stage["input"] == {"$ifNull": ["$hits", []]}
        assert update_stage["hits"]["$cond"][0] == "$allowed"
        assert upsert is True
        assert return_document is not None

        cutoff = filter_stage["cond"]["$gt"][1]
        max_requests = allowed_expr[1]
        now = update_stage["hits"]["$cond"][1]["$concatArrays"][1][0]
        expires_at = update_stage["expires_at"]

        current = self.docs.get(identity, {"identity": identity, "hits": []})
        hits = [float(hit) for hit in current.get("hits", []) if float(hit) > cutoff]
        allowed = len(hits) < max_requests
        if allowed:
            hits.append(now)

        updated = {
            "identity": identity,
            "hits": hits,
            "allowed": allowed,
            "expires_at": expires_at,
        }
        self.docs[identity] = updated
        self.calls.append({"filter": filter_doc, "pipeline": pipeline})
        return dict(updated)


def _mongo_store(
    records: FakeRecordsCollection = None,
    rate_limits: FakeRateLimitCollection = None,
) -> ControlPlaneStore:
    store = object.__new__(ControlPlaneStore)
    store._connected = True
    store._in_memory = False
    store.records = records or FakeRecordsCollection()
    store.rate_limits = rate_limits or FakeRateLimitCollection()
    store._client = None
    return store


def test_control_plane_store_does_not_connect_during_init(monkeypatch):
    def fail_connect(self):
        raise AssertionError("should not connect during init")

    monkeypatch.setattr(ControlPlaneStore, "_try_connect", fail_connect)

    ControlPlaneStore(uri="mongodb://example.invalid:27017", database="xmem-test")


def test_single_use_tokens_are_consumed_once():
    _memory_records.clear()
    store = _memory_store()

    created = store.create_single_use_token(
        record_type="mcp_temp_token",
        user_id="user-1",
        prefix="xm-temp-",
        ttl_seconds=600,
    )

    assert created["token"].startswith("xm-temp-")
    assert store.consume_single_use_token("mcp_temp_token", created["token"]) == "user-1"
    assert store.consume_single_use_token("mcp_temp_token", created["token"]) is None


def test_mongo_single_use_tokens_are_consumed_once():
    records = FakeRecordsCollection()
    store = _mongo_store(records=records)

    created = store.create_single_use_token(
        record_type="oauth_auth_code",
        user_id="user-1",
        prefix="",
        ttl_seconds=600,
    )

    assert store.consume_single_use_token("oauth_auth_code", created["token"]) == "user-1"
    assert store.consume_single_use_token("oauth_auth_code", created["token"]) is None


def test_expired_single_use_tokens_are_rejected():
    _memory_records.clear()
    store = _memory_store()

    created = store.create_single_use_token(
        record_type="oauth_auth_code",
        user_id="user-1",
        prefix="",
        ttl_seconds=600,
    )
    key = store._record_key("oauth_auth_code", created["token"])
    _memory_records[key]["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)

    assert store.consume_single_use_token("oauth_auth_code", created["token"]) is None


def test_admin_sessions_can_be_created_read_and_deleted():
    _memory_records.clear()
    store = _memory_store()

    created = store.create_admin_session({"username": "admin", "role": "superadmin"}, ttl_seconds=60)

    assert store.get_admin_session(created["token"]) == {"username": "admin", "role": "superadmin"}
    store.delete_admin_session(created["token"])
    assert store.get_admin_session(created["token"]) is None


@pytest.mark.asyncio
async def test_rate_limit_counter_is_shared_by_identity():
    _memory_rate_limits.clear()
    store = _memory_store()

    assert await store.check_rate_limit("user-1", max_requests=2, window_seconds=60) == (True, 1)
    assert await store.check_rate_limit("user-1", max_requests=2, window_seconds=60) == (True, 0)
    assert await store.check_rate_limit("user-1", max_requests=2, window_seconds=60) == (False, 0)


@pytest.mark.asyncio
async def test_mongo_rate_limit_counter_uses_atomic_update_pipeline():
    rate_limits = FakeRateLimitCollection()
    store = _mongo_store(rate_limits=rate_limits)

    assert await store.check_rate_limit("user-1", max_requests=2, window_seconds=60) == (True, 1)
    assert await store.check_rate_limit("user-1", max_requests=2, window_seconds=60) == (True, 0)
    assert await store.check_rate_limit("user-1", max_requests=2, window_seconds=60) == (False, 0)

    assert len(rate_limits.calls) == 3
    assert len(rate_limits.docs["user-1"]["hits"]) == 2


def test_mongo_rate_limit_update_prunes_expired_hits():
    rate_limits = FakeRateLimitCollection()
    rate_limits.docs["user-1"] = {
        "identity": "user-1",
        "hits": [10.0, 80.0],
    }
    store = _mongo_store(rate_limits=rate_limits)

    assert store._check_rate_limit_mongo(
        identity="user-1",
        max_requests=2,
        window_seconds=60,
        now=100.0,
        cutoff=40.0,
    ) == (True, 0)

    assert rate_limits.docs["user-1"]["hits"] == [80.0, 100.0]


def test_control_plane_store_refuses_memory_fallback_in_production(monkeypatch):
    store = _memory_store()
    monkeypatch.setattr("src.database.control_plane_store.settings.environment", "production")

    with pytest.raises(RuntimeError, match="MongoDB is required"):
        store._enable_in_memory_fallback(ConnectionError("offline"))
