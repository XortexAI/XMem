import os

import pytest

os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("NEO4J_PASSWORD", "test-neo4j-password")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

from src.jobs.durable import (
    DEAD_LETTER,
    FAILED,
    QUEUED,
    RUNNING,
    SUCCEEDED,
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

    def get(self, job_id):
        assert job_id == self.job["job_id"]
        return dict(self.job)

    def mark_running(self, job_id):
        assert job_id == self.job["job_id"]
        self.job["status"] = RUNNING
        self.job["retry_count"] = self.job.get("retry_count", 0) + 1

    def mark_succeeded(self, job_id, result=None):
        assert job_id == self.job["job_id"]
        self.job["status"] = SUCCEEDED
        self.job["result"] = dict(result or {})

    def mark_failed(self, job_id, error):
        assert job_id == self.job["job_id"]
        status = (
            DEAD_LETTER
            if self.job.get("retry_count", 0) >= self.job.get("max_attempts", 1)
            else FAILED
        )
        self.job["status"] = status
        self.job["error"] = error
        self.job["error_state"] = {
            "message": error,
            "failed_at": utc_now(),
            "attempt": self.job.get("retry_count", 0),
        }
        return status

    def reset_for_retry(self, job_id):
        assert job_id == self.job["job_id"]
        self.job["status"] = QUEUED


@pytest.mark.asyncio
async def test_run_job_retries_then_succeeds():
    store = FakeJobStore({
        "job_id": "job-1",
        "status": QUEUED,
        "retry_count": 0,
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
    assert store.job["retry_count"] == 2
    assert store.job["result"]["ok"] is True


@pytest.mark.asyncio
async def test_run_job_dead_letters_after_max_attempts():
    store = FakeJobStore({
        "job_id": "job-2",
        "status": QUEUED,
        "retry_count": 0,
        "max_attempts": 1,
        "timeout_seconds": 1,
    })

    async def handler():
        raise RuntimeError("permanent failure")

    await run_job(store, "job-2", handler, retry_base_seconds=0)

    assert store.job["status"] == DEAD_LETTER
    assert store.job["retry_count"] == 1
    assert store.job["error"] == "permanent failure"
