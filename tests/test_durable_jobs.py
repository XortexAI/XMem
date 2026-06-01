import asyncio
import os
import threading

import pytest

os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("NEO4J_PASSWORD", "test-neo4j-password")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

from src.jobs.durable import (
    CANCELLED,
    DEAD_LETTER,
    FAILED,
    QUEUED,
    RUNNING,
    SUCCEEDED,
    DurableJobStore,
    idempotency_key,
    redact_payload,
    run_job,
    utc_now,
)


def test_idempotency_key_is_stable_for_equivalent_payloads():
    left = idempotency_key("memory_ingest", {"b": 2, "a": {"z": 1, "y": 0}})
    right = idempotency_key("memory_ingest", {"a": {"y": 0, "z": 1}, "b": 2})

    assert left == right


def test_redact_payload_masks_nested_secret_fields():
    payload = {
        "github_url": "https://github.com/acme/repo",
        "pat": "ghp_secret",
        "nested": {
            "Authorization": "Bearer token",
            "client_secret": "secret",
            "ok": "visible",
        },
    }

    redacted = redact_payload(payload)

    assert redacted["pat"] == "[redacted]"
    assert redacted["nested"]["Authorization"] == "[redacted]"
    assert redacted["nested"]["client_secret"] == "[redacted]"
    assert redacted["nested"]["ok"] == "visible"


class FakeJobStore:
    def __init__(self, job):
        self.job = job
        self.lock = threading.Lock()

    def get(self, job_id):
        assert job_id == self.job["job_id"]
        return dict(self.job)

    def claim_for_run(self, job_id):
        assert job_id == self.job["job_id"]
        with self.lock:
            if self.job["status"] != QUEUED:
                return False
            self.job["status"] = RUNNING
            self.job["attempt_count"] = self.job.get("attempt_count", 0) + 1
            self.job["retry_count"] = max(self.job["attempt_count"] - 1, 0)
            return True

    def mark_succeeded(self, job_id, result=None):
        assert job_id == self.job["job_id"]
        self.job["status"] = SUCCEEDED
        self.job["result"] = dict(result or {})

    def mark_failed(self, job_id, error):
        assert job_id == self.job["job_id"]
        attempt_count = self.job.get("attempt_count", 0)
        status = (
            DEAD_LETTER
            if attempt_count >= self.job.get("max_attempts", 1)
            else FAILED
        )
        self.job["status"] = status
        self.job["retry_count"] = max(attempt_count - 1, 0)
        self.job["error"] = error
        self.job["error_state"] = {
            "message": error,
            "failed_at": utc_now(),
            "attempt": attempt_count,
            "retry_count": self.job["retry_count"],
        }
        return status

    def reset_for_retry(self, job_id):
        assert job_id == self.job["job_id"]
        self.job["status"] = QUEUED


class _UpdateResult:
    def __init__(self, modified_count):
        self.modified_count = modified_count


class _FakeCollection:
    def __init__(self, doc):
        self.doc = dict(doc)

    def _matches(self, query):
        for key, expected in query.items():
            actual = self.doc.get(key)
            if isinstance(expected, dict) and "$nin" in expected:
                if actual in expected["$nin"]:
                    return False
            elif actual != expected:
                return False
        return True

    def _apply(self, update):
        for key, value in update.get("$inc", {}).items():
            self.doc[key] = self.doc.get(key, 0) + value
        for key, value in update.get("$set", {}).items():
            self.doc[key] = value

    def find_one(self, query):
        return dict(self.doc) if self._matches(query) else None

    def find_one_and_update(self, query, update, return_document=False):
        if not self._matches(query):
            return None
        before = dict(self.doc)
        self._apply(update)
        return dict(self.doc) if return_document else before

    def update_one(self, query, update):
        if not self._matches(query):
            return _UpdateResult(0)
        self._apply(update)
        return _UpdateResult(1)


def _durable_store_with_doc(doc):
    store = DurableJobStore.__new__(DurableJobStore)
    store.jobs = _FakeCollection(doc)
    return store


def test_terminal_jobs_are_not_overwritten_by_late_workflow_updates():
    store = _durable_store_with_doc({
        "job_id": "job-1",
        "status": CANCELLED,
        "attempt_count": 0,
        "retry_count": 0,
    })

    store.mark_running("job-1")
    store.mark_succeeded("job-1", {"ok": True})
    status = store.mark_failed("job-1", "late failure")

    job = store.get("job-1")
    assert status == CANCELLED
    assert job["status"] == CANCELLED
    assert job["attempt_count"] == 0
    assert "result" not in job
    assert "error" not in job


def test_mark_cancelled_does_not_overwrite_succeeded_job():
    store = _durable_store_with_doc({
        "job_id": "job-1",
        "status": SUCCEEDED,
        "result": {"ok": True},
    })

    store.mark_cancelled("job-1")

    job = store.get("job-1")
    assert job["status"] == SUCCEEDED
    assert job["result"] == {"ok": True}
    assert "cancelled_at" not in job


def test_mark_dead_letter_does_not_overwrite_cancelled_job():
    store = _durable_store_with_doc({
        "job_id": "job-1",
        "status": CANCELLED,
        "cancelled_at": utc_now(),
    })

    store.mark_dead_letter("job-1", "late failure")

    job = store.get("job-1")
    assert job["status"] == CANCELLED
    assert "dead_lettered_at" not in job
    assert "error" not in job


def test_reserve_workflow_start_claims_queued_job_once():
    store = _durable_store_with_doc({
        "job_id": "job-1",
        "status": QUEUED,
        "workflow_id": None,
    })

    assert store.reserve_workflow_start("job-1", "workflow-1") is True
    assert store.reserve_workflow_start("job-1", "workflow-2") is False

    job = store.get("job-1")
    assert job["workflow_id"] == "workflow-1"


@pytest.mark.asyncio
async def test_cancel_job_workflow_reraises_transient_cancel_errors(monkeypatch):
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).parents[1] / "src/api/routes/v2/temporal_client.py"
    spec = importlib.util.spec_from_file_location("temporal_client_under_test", module_path)
    temporal_client = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(temporal_client)

    class FakeHandle:
        async def cancel(self):
            raise RuntimeError("connection refused")

    class FakeClient:
        def get_workflow_handle(self, workflow_id):
            assert workflow_id == "workflow-1"
            return FakeHandle()

    async def fake_get_temporal_client():
        return FakeClient()

    monkeypatch.setattr(
        temporal_client,
        "get_temporal_client",
        fake_get_temporal_client,
    )

    with pytest.raises(RuntimeError, match="connection refused"):
        await temporal_client.cancel_job_workflow({"workflow_id": "workflow-1"})


@pytest.mark.asyncio
async def test_run_job_retries_then_succeeds():
    store = FakeJobStore({
        "job_id": "job-1",
        "status": QUEUED,
        "retry_count": 0,
        "attempt_count": 0,
        "max_attempts": 2,
        "timeout_seconds": 1,
    })
    attempts = 0

    async def handler():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient failure")
        return {"ok": True}

    await run_job(store, "job-1", handler, retry_base_seconds=0)

    assert attempts == 2
    assert store.job["status"] == SUCCEEDED
    assert store.job["attempt_count"] == 2
    assert store.job["retry_count"] == 1
    assert store.job["result"]["ok"] is True


@pytest.mark.asyncio
async def test_run_job_dead_letters_after_max_attempts():
    store = FakeJobStore({
        "job_id": "job-2",
        "status": QUEUED,
        "retry_count": 0,
        "attempt_count": 0,
        "max_attempts": 1,
        "timeout_seconds": 1,
    })

    async def handler():
        raise RuntimeError("permanent failure")

    await run_job(store, "job-2", handler, retry_base_seconds=0)

    assert store.job["status"] == DEAD_LETTER
    assert store.job["attempt_count"] == 1
    assert store.job["retry_count"] == 0
    assert store.job["error"] == "permanent failure"


@pytest.mark.asyncio
async def test_duplicate_runners_only_execute_handler_once():
    store = FakeJobStore({
        "job_id": "job-3",
        "status": QUEUED,
        "retry_count": 0,
        "attempt_count": 0,
        "max_attempts": 1,
        "timeout_seconds": 1,
    })
    started = asyncio.Event()
    release = asyncio.Event()
    attempts = 0

    async def handler():
        nonlocal attempts
        attempts += 1
        started.set()
        await release.wait()
        return {"ok": True}

    first = asyncio.create_task(run_job(store, "job-3", handler, retry_base_seconds=0))
    await started.wait()
    second = asyncio.create_task(run_job(store, "job-3", handler, retry_base_seconds=0))

    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, second)

    assert attempts == 1
    assert store.job["status"] == SUCCEEDED
    assert store.job["attempt_count"] == 1
    assert store.job["retry_count"] == 0
