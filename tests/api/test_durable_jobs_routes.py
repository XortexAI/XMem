from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.routes import jobs as jobs_routes
from src.api.routes import memory as memory_routes
from src.api.routes.jobs import router as jobs_router
from src.api.routes.memory import router as memory_router
from src.api.routes.memory import v2_router as memory_v2_router
from src.jobs import JobStatus, JobStore


class FakeIngestPipeline:
    model = SimpleNamespace(model="fake-ingest")

    def __init__(self) -> None:
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"classification_result": SimpleNamespace(classifications=[])}


class FakeJobStore:
    def __init__(self) -> None:
        self.enqueued = []

    def enqueue(self, **kwargs):
        self.enqueued.append(kwargs)
        return {
            "job_id": f"job-{len(self.enqueued)}",
            "job_type": kwargs["job_type"],
            "status": "pending",
            "idempotency_key": kwargs.get("idempotency_key") or "idem-1",
        }

    def get(self, job_id: str, owner_id: str | None = None):
        if job_id != "job-1" or owner_id != "Static Key User":
            return None
        return {
            "job_id": job_id,
            "job_type": "memory.ingest",
            "status": "pending",
            "retry_count": 0,
            "max_retries": 3,
            "timeout_seconds": 120.0,
        }


def _app(monkeypatch, pipeline: FakeIngestPipeline, store: FakeJobStore) -> FastAPI:
    monkeypatch.setattr(deps.settings, "api_keys", ["test-static-key"], raising=False)
    deps._init_error = None
    deps._pipelines_ready.set()
    deps._ingest_pipeline = pipeline
    monkeypatch.setattr(memory_routes, "get_job_store", lambda: store)
    monkeypatch.setattr(jobs_routes, "get_job_store", lambda: store)
    app = FastAPI()
    app.include_router(memory_router)
    app.include_router(memory_v2_router)
    app.include_router(jobs_router)
    return app


def _auth() -> dict[str, str]:
    return {"Authorization": "Bearer test-static-key"}


def _ingest_payload() -> dict[str, str]:
    return {
        "user_query": "remember this",
        "agent_response": "ok",
        "user_id": "ignored-by-auth",
    }


def test_v1_ingest_remains_synchronous_and_backward_compatible(monkeypatch):
    pipeline = FakeIngestPipeline()
    store = FakeJobStore()
    client = TestClient(_app(monkeypatch, pipeline, store))

    response = client.post("/v1/memory/ingest", json=_ingest_payload(), headers=_auth())

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["data"]["model"] == "fake-ingest"
    assert "job_id" not in body["data"]
    assert len(pipeline.calls) == 1
    assert store.enqueued == []


def test_v2_ingest_enqueues_durable_job(monkeypatch):
    pipeline = FakeIngestPipeline()
    store = FakeJobStore()
    client = TestClient(_app(monkeypatch, pipeline, store))

    response = client.post("/v2/memory/ingest", json=_ingest_payload(), headers=_auth())

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["job_id"] == "job-1"
    assert body["data"]["job_type"] == "memory.ingest"
    assert body["data"]["status_url"] == "/v1/jobs/job-1"
    assert pipeline.calls == []
    assert store.enqueued[0]["job_type"] == "memory.ingest"
    assert store.enqueued[0]["owner_id"] == "Static Key User"


def test_v2_batch_ingest_enqueues_with_batch_timeout(monkeypatch):
    pipeline = FakeIngestPipeline()
    store = FakeJobStore()
    client = TestClient(_app(monkeypatch, pipeline, store))

    response = client.post(
        "/v2/memory/batch-ingest",
        json={"items": [_ingest_payload(), _ingest_payload()]},
        headers=_auth(),
    )

    assert response.status_code == 200
    assert response.json()["data"]["job_type"] == "memory.batch_ingest"
    assert store.enqueued[0]["timeout_seconds"] == 240.0


def test_job_status_is_scoped_to_authenticated_owner(monkeypatch):
    pipeline = FakeIngestPipeline()
    store = FakeJobStore()
    client = TestClient(_app(monkeypatch, pipeline, store))

    ok = client.get("/v1/jobs/job-1", headers=_auth())
    missing = client.get("/v1/jobs/other", headers=_auth())

    assert ok.status_code == 200
    assert ok.json()["data"]["job_id"] == "job-1"
    assert missing.status_code == 404


class FakeCollection:
    def __init__(self) -> None:
        self.docs = []

    def create_index(self, *_args, **_kwargs):
        return None

    def find_one(self, query, projection=None):
        for doc in self.docs:
            if _matches(doc, query):
                return _project(doc, projection)
        return None

    def insert_one(self, doc):
        self.docs.append(dict(doc))

    def update_one(self, query, update, upsert=False):
        doc = self.find_one(query)
        if doc is None:
            if not upsert:
                return None
            doc = dict(query)
            self.docs.append(doc)
        doc.update(update.get("$set", {}))
        return None

    def find_one_and_update(self, query, update, sort=None, return_document=None):
        docs = [doc for doc in self.docs if _matches(doc, query)]
        if sort:
            for key, direction in reversed(sort):
                docs.sort(key=lambda item: item.get(key), reverse=direction < 0)
        if not docs:
            return None
        docs[0].update(update.get("$set", {}))
        return docs[0]

    def find(self, query):
        return FakeCursor([doc for doc in self.docs if _matches(doc, query)])


class FakeCursor(list):
    def sort(self, sort_spec):
        for key, direction in reversed(sort_spec):
            super().sort(key=lambda item: item.get(key), reverse=direction < 0)
        return self


def _matches(doc, query):
    for key, expected in query.items():
        actual = doc.get(key)
        if isinstance(expected, dict):
            if "$lte" in expected and not (actual <= expected["$lte"]):
                return False
            if "$ne" in expected and not (actual != expected["$ne"]):
                return False
        elif actual != expected:
            return False
    return True


def _project(doc, projection):
    if projection is None:
        return doc
    projected = dict(doc)
    if projection and projection.get("_id") is False:
        projected.pop("_id", None)
    return projected


def _store(monkeypatch) -> JobStore:
    jobs = FakeCollection()
    dead_letters = FakeCollection()
    store = JobStore.__new__(JobStore)
    store.jobs = jobs
    store.dead_letters = dead_letters
    monkeypatch.setattr("src.jobs.settings.job_timeout_seconds", 120.0, raising=False)
    monkeypatch.setattr("src.jobs.settings.job_lease_seconds", 300.0, raising=False)
    monkeypatch.setattr("src.jobs.settings.job_max_retries", 1, raising=False)
    monkeypatch.setattr("src.jobs.settings.job_retry_backoff_seconds", 5.0, raising=False)
    return store


def test_job_store_idempotency_and_dead_letter_payload(monkeypatch):
    store = _store(monkeypatch)
    payload = {"b": 2, "a": 1}

    first = store.enqueue(job_type="memory.ingest", owner_id="user-1", payload=payload)
    second = store.enqueue(job_type="memory.ingest", owner_id="user-1", payload={"a": 1, "b": 2})

    assert first["job_id"] == second["job_id"]
    assert first["idempotency_key"] == second["idempotency_key"]

    running = store.claim_next("worker-1")
    running["retry_count"] = 1
    running["max_retries"] = 1
    store.fail_or_retry(running, "boom")

    assert store.dead_letters.docs[0]["job"]["payload"] == payload
    assert store.dead_letters.docs[0]["job"]["status"] == JobStatus.DEAD_LETTER.value


def test_claim_next_respects_per_job_lease(monkeypatch):
    store = _store(monkeypatch)
    now = datetime.now(timezone.utc)
    store.jobs.docs.extend(
        [
            {
                "job_id": "long-running",
                "status": JobStatus.RUNNING.value,
                "started_at": now - timedelta(seconds=400),
                "run_after": now,
                "created_at": now,
                "lease_seconds": 600.0,
            },
            {
                "job_id": "stale-running",
                "status": JobStatus.RUNNING.value,
                "started_at": now - timedelta(seconds=400),
                "run_after": now,
                "created_at": now,
                "lease_seconds": 300.0,
            },
        ]
    )

    claimed = store.claim_next("worker-1")

    assert claimed["job_id"] == "stale-running"
    assert store.jobs.docs[0].get("worker_id") != "worker-1"
