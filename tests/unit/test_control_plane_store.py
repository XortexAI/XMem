from __future__ import annotations

from datetime import datetime

from src.config import settings
from src.database.control_plane_store import (
    ControlPlaneStore,
    _in_memory_admin_sessions,
    _in_memory_auth_codes,
    _in_memory_rate_limits,
    _in_memory_temp_tokens,
)


def _force_control_memory(self):
    self._connected = False
    self._in_memory = True
    self.temp_tokens = None
    self.auth_codes = None
    self.admin_sessions = None
    self.rate_limits = None


def test_control_plane_store_handles_tokens_sessions_and_rate_limits_in_memory(monkeypatch):
    monkeypatch.setattr(settings, "environment", "dev", raising=False)
    monkeypatch.setattr(ControlPlaneStore, "_try_connect", _force_control_memory)
    _in_memory_temp_tokens.clear()
    _in_memory_auth_codes.clear()
    _in_memory_admin_sessions.clear()
    _in_memory_rate_limits.clear()

    store = ControlPlaneStore()

    token, expires_at = store.create_temp_token("user-1", ttl_minutes=10)
    assert token.startswith("xm-temp-")
    assert expires_at > datetime.utcnow()
    assert store.consume_temp_token(token) == "user-1"
    assert store.consume_temp_token(token) is None

    code = store.create_auth_code("user-2", ttl_minutes=10)
    assert store.consume_auth_code(code) == "user-2"
    assert store.consume_auth_code(code) is None

    session_token = store.create_admin_session({"username": "admin", "role": "superadmin"}, ttl_hours=24)
    assert store.get_admin_session(session_token)["username"] == "admin"
    store.delete_admin_session(session_token)
    assert store.get_admin_session(session_token) is None

    allowed, remaining = store.check_rate_limit("user-3", max_requests=2, window_seconds=60)
    assert allowed and remaining == 1
    allowed, remaining = store.check_rate_limit("user-3", max_requests=2, window_seconds=60)
    assert allowed and remaining == 0
    allowed, remaining = store.check_rate_limit("user-3", max_requests=2, window_seconds=60)
    assert not allowed and remaining == 0
