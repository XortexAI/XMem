from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import require_user
from src.api.routes import connectors


@pytest.fixture(autouse=True)
def _reset_connector_state() -> None:
    connectors._pending_states.clear()


def _user() -> dict:
    return {"id": "user-1", "email": "user@example.com", "username": "user"}


def _client() -> TestClient:
    app = FastAPI()
    app.dependency_overrides[require_user] = _user
    app.include_router(connectors.router)
    return TestClient(app)


def test_lists_supported_connectors() -> None:
    response = _client().get("/api/connectors")

    assert response.status_code == 200
    body = response.json()
    ids = {item["id"] for item in body["connectors"]}
    assert ids == {"notion", "google-drive"}
    assert {item["state"] for item in body["connectors"]} == {"not_connected"}


def test_oauth_start_requires_configured_client_id(monkeypatch) -> None:
    monkeypatch.delenv("NOTION_CLIENT_ID", raising=False)

    response = _client().post("/api/connectors/notion/oauth/start")

    assert response.status_code == 503
    assert "client ID is not configured" in response.json()["detail"]


def test_oauth_start_builds_authorization_url_without_secret(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_ID", "drive-client")
    monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_SECRET", "do-not-leak")
    monkeypatch.setenv(
        "GOOGLE_DRIVE_REDIRECT_URI",
        "http://localhost:8000/api/connectors/google-drive/oauth/callback",
    )

    response = _client().post("/api/connectors/google-drive/oauth/start")

    assert response.status_code == 200
    body = response.json()
    assert body["connector_id"] == "google-drive"
    assert "accounts.google.com" in body["authorization_url"]
    assert "client_id=drive-client" in body["authorization_url"]
    assert "do-not-leak" not in body["authorization_url"]
    assert body["state"]


def test_callback_validates_state_without_marking_connected(monkeypatch) -> None:
    monkeypatch.setenv("NOTION_CLIENT_ID", "notion-client")
    client = _client()

    started = client.post("/api/connectors/notion/oauth/start")
    state = started.json()["state"]

    callback = client.get(f"/api/connectors/notion/oauth/callback?code=abc&state={state}")
    assert callback.status_code == 200
    assert callback.json()["status"] == "pending"

    status = client.get("/api/connectors/notion/status")
    assert status.status_code == 200
    assert status.json()["state"] == "not_connected"

    disconnected = client.post("/api/connectors/notion/disconnect")
    assert disconnected.status_code == 200
    assert disconnected.json() == {"connector_id": "notion", "disconnected": False}


def test_callback_handles_provider_denial_and_consumes_state(monkeypatch) -> None:
    monkeypatch.setenv("NOTION_CLIENT_ID", "notion-client")
    client = _client()

    started = client.post("/api/connectors/notion/oauth/start")
    state = started.json()["state"]

    callback = client.get(f"/api/connectors/notion/oauth/callback?error=access_denied&state={state}")

    assert callback.status_code == 400
    assert "access_denied" in callback.json()["detail"]
    retry = client.get(f"/api/connectors/notion/oauth/callback?code=abc&state={state}")
    assert retry.status_code == 400
