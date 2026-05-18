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


def test_control_plane_store_refuses_memory_fallback_in_production(monkeypatch):
    store = _memory_store()
    monkeypatch.setattr("src.database.control_plane_store.settings.environment", "production")

    with pytest.raises(RuntimeError, match="MongoDB is required"):
        store._enable_in_memory_fallback(ConnectionError("offline"))
