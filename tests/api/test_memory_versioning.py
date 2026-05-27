from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.routes import memory


class FakeIngestPipeline:
    model = SimpleNamespace(model="fake-ingest")

    def __init__(self):
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"classification_result": SimpleNamespace(classifications=[])}


class FakeRetrievalPipeline:
    model = SimpleNamespace(model="fake-retrieval")


class FakeJobStore:
    def __init__(self):
        self.jobs = {}

    def enqueue(self, *, job_type, payload, user_id, timeout_seconds, max_attempts, **_):
        job = {
            "job_id": f"{job_type}:fake",
            "job_type": job_type,
            "payload": payload,
            "user_id": user_id,
            "status": "queued",
            "timeout_seconds": timeout_seconds,
            "max_attempts": max_attempts,
            "retry_count": 0,
        }
        self.jobs[job["job_id"]] = job
        return job, True

    def get(self, job_id):
        return self.jobs.get(job_id)


def _build_app(monkeypatch, user=None):
    ingest = FakeIngestPipeline()
    deps._init_error = None
    deps._pipelines_ready.set()
    deps.set_pipelines(ingest, FakeRetrievalPipeline())
    auth_user = user or {"id": "user-1", "username": "hunter"}

    async def fake_user():
        return auth_user

    async def fake_ready():
        return None

    async def fake_rate_limit():
        return {"id": "user-1", "username": "hunter"}

    app = FastAPI()
    app.dependency_overrides[deps.require_api_key] = fake_user
    app.dependency_overrides[deps.require_ready] = fake_ready
    app.dependency_overrides[deps.enforce_rate_limit] = fake_rate_limit
    app.include_router(memory.scrape_router)
    app.include_router(memory.router)
    app.include_router(memory.v2_scrape_router)
    app.include_router(memory.v2_router)
    return app, ingest


def test_static_key_user_id_override_is_local_only(monkeypatch):
    static_user = {"id": "static-key", "name": "Static Key User", "email": "static@xmem.ai"}

    monkeypatch.setattr(memory.settings, "environment", "development", raising=False)
    assert memory._current_user_id(static_user, "friendly-user") == "friendly-user"

    monkeypatch.setattr(memory.settings, "environment", "production", raising=False)
    assert memory._current_user_id(static_user, "friendly-user") == "Static Key User"


def test_v1_ingest_keeps_synchronous_response_contract(monkeypatch):
    app, ingest = _build_app(monkeypatch)
    payload = {
        "user_query": "remember this",
        "agent_response": "done",
        "user_id": "body-user",
    }

    response = TestClient(app).post("/v1/memory/ingest", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["data"]["model"] == "fake-ingest"
    assert "job_id" not in body["data"]
    assert ingest.calls[0]["user_id"] == "hunter"


def test_v2_ingest_returns_durable_job_envelope(monkeypatch):
    app, ingest = _build_app(monkeypatch)
    store = FakeJobStore()
    scheduled = []
    monkeypatch.setattr(memory, "get_default_job_store", lambda: store)
    monkeypatch.setattr(
        memory,
        "_schedule_job",
        lambda job, handler: scheduled.append(job["job_id"]),
    )
    payload = {
        "user_query": "remember this",
        "agent_response": "done",
        "user_id": "body-user",
    }

    response = TestClient(app).post("/v2/memory/ingest", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["data"] == {
        "job_id": "memory_ingest:fake",
        "status": "queued",
        "created": True,
        "status_url": "/v2/memory/ingest/memory_ingest:fake/status",
    }
    assert scheduled == ["memory_ingest:fake"]
    assert ingest.calls == []


def test_v1_batch_ingest_scopes_each_item_for_local_static_key(monkeypatch):
    monkeypatch.setattr(memory.settings, "environment", "development", raising=False)
    static_user = {"id": "static-key", "name": "Static Key User", "email": "static@xmem.ai"}
    app, ingest = _build_app(monkeypatch, user=static_user)
    payload = {
        "items": [
            {"user_query": "remember alpha", "agent_response": "done", "user_id": "alice"},
            {"user_query": "remember beta", "agent_response": "done", "user_id": "bob"},
        ],
    }

    response = TestClient(app).post("/v1/memory/batch-ingest", json=payload)

    assert response.status_code == 200
    assert [call["user_id"] for call in ingest.calls] == ["alice", "bob"]


def test_v2_batch_ingest_queues_scoped_items_for_local_static_key(monkeypatch):
    monkeypatch.setattr(memory.settings, "environment", "development", raising=False)
    static_user = {"id": "static-key", "name": "Static Key User", "email": "static@xmem.ai"}
    app, ingest = _build_app(monkeypatch, user=static_user)
    store = FakeJobStore()
    scheduled = []
    monkeypatch.setattr(memory, "get_default_job_store", lambda: store)
    monkeypatch.setattr(
        memory,
        "_schedule_job",
        lambda job, handler: scheduled.append(job["job_id"]),
    )
    payload = {
        "items": [
            {"user_query": "remember alpha", "agent_response": "done", "user_id": "alice"},
            {"user_query": "remember beta", "agent_response": "done", "user_id": "bob"},
        ],
    }

    response = TestClient(app).post("/v2/memory/batch-ingest", json=payload)

    assert response.status_code == 200
    job = store.jobs["memory_batch_ingest:fake"]
    assert job["user_id"] == "Static Key User"
    assert [item["user_id"] for item in job["payload"]["items"]] == ["alice", "bob"]
    assert scheduled == ["memory_batch_ingest:fake"]
    assert ingest.calls == []
