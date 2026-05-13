from __future__ import annotations

from src.database.control_plane_store import (
    ControlPlaneStore,
    _memory_rate_limits,
    _memory_records,
)


def _force_memory(self):
    self._connected = False
    self._in_memory = True
    self.records = None
    self.rate_limits = None


def test_single_use_tokens_are_durable_shape_and_single_use(monkeypatch):
    _memory_records.clear()
    monkeypatch.setattr(ControlPlaneStore, "_try_connect", _force_memory)
    store = ControlPlaneStore()

    created = store.create_single_use_token(
        "oauth_auth_code",
        user_id="user-1",
        prefix="",
        ttl_seconds=600,
    )

    assert (
        store.consume_single_use_token("oauth_auth_code", created["token"]) == "user-1"
    )
    assert store.consume_single_use_token("oauth_auth_code", created["token"]) is None


def test_admin_sessions_can_be_created_read_and_deleted(monkeypatch):
    _memory_records.clear()
    monkeypatch.setattr(ControlPlaneStore, "_try_connect", _force_memory)
    store = ControlPlaneStore()

    session = store.create_admin_session({"username": "admin"}, ttl_seconds=60)

    assert store.get_admin_session(session["token"]) == {"username": "admin"}
    store.delete_admin_session(session["token"])
    assert store.get_admin_session(session["token"]) is None


async def test_rate_limits_use_shared_store_shape(monkeypatch):
    _memory_rate_limits.clear()
    monkeypatch.setattr(ControlPlaneStore, "_try_connect", _force_memory)
    store = ControlPlaneStore()

    assert await store.check_rate_limit(
        "user-1", max_requests=1, window_seconds=60
    ) == (True, 0)
    assert await store.check_rate_limit(
        "user-1", max_requests=1, window_seconds=60
    ) == (False, 0)
