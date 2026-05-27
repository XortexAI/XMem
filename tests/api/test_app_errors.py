from __future__ import annotations

from src.api import app as api_app


def test_public_exception_message_redacts_connection_details_in_production(monkeypatch):
    monkeypatch.setattr(api_app.settings, "environment", "production", raising=False)

    message = api_app._public_exception_message(
        ConnectionError("postgresql://user:password@internal-db:5432/xmem")
    )

    assert "password" not in message
    assert "internal-db" not in message
    assert "backend service is unavailable" in message


def test_public_exception_message_redacts_value_error_in_production(monkeypatch):
    monkeypatch.setattr(api_app.settings, "environment", "production", raising=False)

    message = api_app._public_exception_message(
        ValueError("vector mismatch for postgresql://user:password@internal-db:5432/xmem")
    )

    assert message == "Invalid request."
    assert "password" not in message
    assert "internal-db" not in message


def test_public_exception_message_keeps_timeout_detail_in_local(monkeypatch):
    monkeypatch.setattr(api_app.settings, "environment", "development", raising=False)

    message = api_app._public_exception_message(TimeoutError("LLM timed out after 180 seconds"))

    assert message == "LLM timed out after 180 seconds"
