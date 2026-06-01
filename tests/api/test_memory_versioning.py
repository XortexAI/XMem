from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.routes import memory
from src.api.routes.v2 import jobs as jobs_v2
from src.api.routes.v2 import memory as memory_v2
from src.jobs import durable


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
            "attempt_count": 0,
            "workflow_id": None,
        }
        self.jobs[job["job_id"]] = job
        return job, True

    def get(self, job_id):
        return self.jobs.get(job_id)

    def mark_failed(self, job_id, error):
        job = self.jobs[job_id]
        job["status"] = "failed"
        job["error"] = error
        return "failed"

    def mark_cancelled(self, job_id):
        job = self.jobs[job_id]
        job["status"] = "cancelled"
        job["cancelled_at"] = "now"
        job["completed_at"] = "now"

    def reset_for_retry(self, job_id, clear_workflow=False):
        job = self.jobs[job_id]
        job["status"] = "queued"
        job["error"] = None
        if clear_workflow:
            job["workflow_id"] = None
            job["run_id"] = None

    def update_payload(self, job_id, payload):
        self.jobs[job_id]["payload"] = payload

    def reserve_workflow_start(self, job_id, workflow_id):
        job = self.jobs[job_id]
        if job["status"] != "queued" or job.get("workflow_id"):
            return False
        job["workflow_id"] = workflow_id
        return True


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
    app.include_router(memory_v2.scrape_router)
    app.include_router(memory_v2.router)
    app.include_router(jobs_v2.router)
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
    async def fake_start_job_workflow(job):
        scheduled.append(job["job_id"])

    monkeypatch.setattr(memory_v2, "get_default_job_store", lambda: store)
    monkeypatch.setattr(
        memory_v2,
        "start_job_workflow",
        fake_start_job_workflow,
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


def test_v2_ingest_start_failure_returns_durable_job_handle(monkeypatch):
    app, ingest = _build_app(monkeypatch)
    store = FakeJobStore()

    async def fake_start_job_workflow(job):
        raise RuntimeError("temporal unavailable")

    monkeypatch.setattr(memory_v2, "get_default_job_store", lambda: store)
    monkeypatch.setattr(
        memory_v2,
        "start_job_workflow",
        fake_start_job_workflow,
    )
    payload = {
        "user_query": "remember this",
        "agent_response": "done",
        "user_id": "body-user",
    }

    response = TestClient(app).post("/v2/memory/ingest", json=payload)

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["data"]["job_id"] == "memory_ingest:fake"
    assert body["data"]["status"] == "failed"
    assert body["data"]["status_url"] == "/v2/memory/ingest/memory_ingest:fake/status"
    assert store.jobs["memory_ingest:fake"]["error"] == "temporal unavailable"
    assert ingest.calls == []


def test_v2_retry_start_failure_releases_fresh_billing_reservation(monkeypatch):
    app, _ = _build_app(monkeypatch)
    store = FakeJobStore()
    store.jobs["job-1"] = {
        "job_id": "job-1",
        "job_type": "memory_ingest",
        "payload": {"billing_account_id": "acct-1", "user_id": "hunter"},
        "user_id": "hunter",
        "status": "failed",
        "timeout_seconds": 30,
        "max_attempts": 3,
        "retry_count": 1,
        "attempt_count": 1,
        "workflow_id": "old-workflow",
    }
    released = []

    class FakeEstimate:
        reserved_credits = 100

        def model_dump(self):
            return {"reserved_credits": self.reserved_credits}

    class FakeBillingService:
        def estimate_required_credits(self, job_type, payload):
            return FakeEstimate()

        def reserve_credits(self, account_id, job_id, estimated_credits):
            return SimpleNamespace(reservation_id="reservation-1", created=True)

    async def fake_start_job_workflow(job):
        raise RuntimeError("temporal unavailable")

    def fake_release(account_id, job_id):
        released.append((account_id, job_id))

    monkeypatch.setattr(jobs_v2, "get_default_job_store", lambda: store)
    monkeypatch.setattr(durable, "get_default_job_store", lambda: store)
    monkeypatch.setattr(jobs_v2, "get_default_billing_service", lambda: FakeBillingService())
    monkeypatch.setattr(jobs_v2, "release_job_reservation", fake_release)
    monkeypatch.setattr(jobs_v2, "start_job_workflow", fake_start_job_workflow)

    response = TestClient(app).post("/v2/jobs/job-1/retry")

    assert response.status_code == 503
    assert released == [("acct-1", "job-1")]
    assert store.jobs["job-1"]["status"] == "failed"
    assert store.jobs["job-1"]["error"] == "temporal unavailable"


def test_v2_retry_payload_update_failure_releases_fresh_billing_reservation(monkeypatch):
    app, _ = _build_app(monkeypatch)
    store = FakeJobStore()
    store.jobs["job-1"] = {
        "job_id": "job-1",
        "job_type": "memory_ingest",
        "payload": {"billing_account_id": "acct-1", "user_id": "hunter"},
        "user_id": "hunter",
        "status": "failed",
        "timeout_seconds": 30,
        "max_attempts": 3,
        "retry_count": 1,
        "attempt_count": 1,
        "workflow_id": "old-workflow",
    }
    released = []

    class FakeEstimate:
        reserved_credits = 100

        def model_dump(self):
            return {"reserved_credits": self.reserved_credits}

    class FakeBillingService:
        def estimate_required_credits(self, job_type, payload):
            return FakeEstimate()

        def reserve_credits(self, account_id, job_id, estimated_credits):
            return SimpleNamespace(reservation_id="reservation-1", created=True)

    def fail_update_payload(job_id, payload):
        raise RuntimeError("payload write failed")

    monkeypatch.setattr(jobs_v2, "get_default_job_store", lambda: store)
    monkeypatch.setattr(durable, "get_default_job_store", lambda: store)
    monkeypatch.setattr(jobs_v2, "get_default_billing_service", lambda: FakeBillingService())
    monkeypatch.setattr(store, "update_payload", fail_update_payload)
    monkeypatch.setattr(
        jobs_v2,
        "release_job_reservation",
        lambda account_id, job_id: released.append((account_id, job_id)),
    )

    response = TestClient(app).post("/v2/jobs/job-1/retry")

    assert response.status_code == 503
    assert released == [("acct-1", "job-1")]
    assert store.jobs["job-1"]["status"] == "failed"
    assert store.jobs["job-1"]["error"] == "payload write failed"


def test_v2_retry_release_failure_still_marks_job_failed(monkeypatch):
    app, _ = _build_app(monkeypatch)
    store = FakeJobStore()
    store.jobs["job-1"] = {
        "job_id": "job-1",
        "job_type": "memory_ingest",
        "payload": {"billing_account_id": "acct-1", "user_id": "hunter"},
        "user_id": "hunter",
        "status": "failed",
        "timeout_seconds": 30,
        "max_attempts": 3,
        "retry_count": 1,
        "attempt_count": 1,
        "workflow_id": "old-workflow",
    }

    class FakeEstimate:
        reserved_credits = 100

        def model_dump(self):
            return {"reserved_credits": self.reserved_credits}

    class FakeBillingService:
        def estimate_required_credits(self, job_type, payload):
            return FakeEstimate()

        def reserve_credits(self, account_id, job_id, estimated_credits):
            return SimpleNamespace(reservation_id="reservation-1", created=True)

    async def fake_start_job_workflow(job):
        raise RuntimeError("temporal unavailable")

    def fail_release(account_id, job_id):
        raise RuntimeError("mongo unavailable")

    monkeypatch.setattr(jobs_v2, "get_default_job_store", lambda: store)
    monkeypatch.setattr(durable, "get_default_job_store", lambda: store)
    monkeypatch.setattr(jobs_v2, "get_default_billing_service", lambda: FakeBillingService())
    monkeypatch.setattr(jobs_v2, "release_job_reservation", fail_release)
    monkeypatch.setattr(jobs_v2, "start_job_workflow", fake_start_job_workflow)

    response = TestClient(app).post("/v2/jobs/job-1/retry")

    assert response.status_code == 503
    assert store.jobs["job-1"]["status"] == "failed"
    assert store.jobs["job-1"]["error"] == (
        "temporal unavailable; billing reservation release failed: mongo unavailable"
    )


def test_v2_retry_start_failure_keeps_reused_billing_reservation(monkeypatch):
    app, _ = _build_app(monkeypatch)
    store = FakeJobStore()
    store.jobs["job-1"] = {
        "job_id": "job-1",
        "job_type": "memory_ingest",
        "payload": {"billing_account_id": "acct-1", "user_id": "hunter"},
        "user_id": "hunter",
        "status": "failed",
        "timeout_seconds": 30,
        "max_attempts": 3,
        "retry_count": 1,
        "attempt_count": 1,
        "workflow_id": "old-workflow",
    }
    released = []

    class FakeEstimate:
        reserved_credits = 100

        def model_dump(self):
            return {"reserved_credits": self.reserved_credits}

    class FakeBillingService:
        def estimate_required_credits(self, job_type, payload):
            return FakeEstimate()

        def reserve_credits(self, account_id, job_id, estimated_credits):
            return SimpleNamespace(reservation_id="reservation-1", created=False)

    async def fake_start_job_workflow(job):
        raise RuntimeError("temporal unavailable")

    monkeypatch.setattr(jobs_v2, "get_default_job_store", lambda: store)
    monkeypatch.setattr(durable, "get_default_job_store", lambda: store)
    monkeypatch.setattr(jobs_v2, "get_default_billing_service", lambda: FakeBillingService())
    monkeypatch.setattr(
        jobs_v2,
        "release_job_reservation",
        lambda account_id, job_id: released.append((account_id, job_id)),
    )
    monkeypatch.setattr(jobs_v2, "start_job_workflow", fake_start_job_workflow)

    response = TestClient(app).post("/v2/jobs/job-1/retry")

    assert response.status_code == 503
    assert released == []
    assert store.jobs["job-1"]["status"] == "failed"


def test_v2_cancel_mark_failure_keeps_billing_reserved_after_signal(monkeypatch):
    app, _ = _build_app(monkeypatch)
    store = FakeJobStore()
    store.jobs["job-1"] = {
        "job_id": "job-1",
        "job_type": "memory_ingest",
        "payload": {"billing_account_id": "acct-1", "user_id": "hunter"},
        "user_id": "hunter",
        "status": "running",
        "timeout_seconds": 30,
        "max_attempts": 3,
        "retry_count": 0,
        "attempt_count": 1,
        "workflow_id": "workflow-1",
    }
    released = []
    cancelled = []

    async def fake_cancel_job_workflow(job):
        cancelled.append(job["job_id"])

    def fail_mark_cancelled(job_id):
        raise RuntimeError("cancel status write failed")

    monkeypatch.setattr(jobs_v2, "get_default_job_store", lambda: store)
    monkeypatch.setattr(durable, "get_default_job_store", lambda: store)
    monkeypatch.setattr(jobs_v2, "cancel_job_workflow", fake_cancel_job_workflow)
    monkeypatch.setattr(store, "mark_cancelled", fail_mark_cancelled)
    monkeypatch.setattr(
        jobs_v2,
        "release_job_billing",
        lambda job, reason: released.append((job["job_id"], reason)),
    )

    response = TestClient(app).post("/v2/jobs/job-1/cancel")

    assert response.status_code == 503
    assert cancelled == ["job-1"]
    assert released == []
    assert store.jobs["job-1"]["status"] == "running"


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
    async def fake_start_job_workflow(job):
        scheduled.append(job["job_id"])

    monkeypatch.setattr(memory_v2, "get_default_job_store", lambda: store)
    monkeypatch.setattr(
        memory_v2,
        "start_job_workflow",
        fake_start_job_workflow,
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
