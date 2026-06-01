import os
import threading

import pytest
from cryptography.fernet import Fernet

os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("NEO4J_PASSWORD", "test-neo4j-password")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault(
    "XMEM_SECRET_ENCRYPTION_KEY",
    Fernet.generate_key().decode("utf-8"),
)

pytest.importorskip("fastapi")


def test_scanner_secret_round_trips_through_public_store_accessor(monkeypatch):
    from src.api.routes.v2 import secrets

    class FakeCollection:
        def __init__(self):
            self.docs = {}

        def create_index(self, *_args, **_kwargs):
            return None

        def update_one(self, query, update, upsert=False):
            assert upsert is True
            self.docs[query["secret_ref"]] = update["$set"]

        def find_one(self, query):
            return self.docs.get(query["secret_ref"])

    class FakeStore:
        def __init__(self):
            self.collection = FakeCollection()

        def get_collection(self, name):
            assert name == "durable_job_secrets"
            return self.collection

    fake_store = FakeStore()
    monkeypatch.setattr(secrets, "get_default_job_store", lambda: fake_store)

    secret_ref = secrets.store_scanner_pat("job-1", "ghp_test")

    assert secret_ref == "scanner_pat:job-1"
    assert secrets.resolve_scanner_pat(secret_ref) == "ghp_test"


def test_missing_scanner_secret_ref_raises(monkeypatch):
    from src.api.routes.v2 import secrets

    class FakeCollection:
        def create_index(self, *_args, **_kwargs):
            return None

        def find_one(self, _query):
            return None

    class FakeStore:
        def get_collection(self, name):
            assert name == "durable_job_secrets"
            return FakeCollection()

    monkeypatch.setattr(secrets, "get_default_job_store", lambda: FakeStore())

    with pytest.raises(ValueError, match="could not be found"):
        secrets.resolve_scanner_pat("scanner_pat:missing")


def test_scanner_cancel_marks_running_phase_cancelled(monkeypatch):
    from src.api.routes.v2 import jobs

    class FakeScannerStore:
        def __init__(self):
            self.updated = None
            self.existing = {
                "job_id": "user:org:repo",
                "username": "user",
                "org": "org",
                "repo": "repo",
                "branch": "main",
                "url": "https://github.com/org/repo.git",
                "phase1_status": "running",
                "phase2_status": "pending",
                "started_at": 123.0,
            }

        def get_scanner_job(self, job_id):
            assert job_id == "user:org:repo"
            return self.existing

        def upsert_scanner_job(self, **kwargs):
            self.updated = kwargs

    fake_store = FakeScannerStore()
    monkeypatch.setattr(jobs.scanner_v1, "_get_code_store", lambda: fake_store)

    jobs._mark_scanner_job_cancelled({
        "job_id": "durable-1",
        "job_type": "scanner_scan",
        "payload": {
            "scanner_job_id": "user:org:repo",
            "org": "org",
            "repo": "repo",
        },
        "retry_count": 0,
    })

    assert fake_store.updated["phase1_status"] == "cancelled"
    assert fake_store.updated["phase2_status"] == "pending"
    assert fake_store.updated["durable_job_id"] == "durable-1"


def test_scanner_phase_status_reflects_existing_failed_durable_job():
    from src.api.routes.v2 import scanner

    assert scanner._scanner_phase_status_from_durable(
        {"status": "failed"},
        phase2_only=False,
    ) == ("failed", "pending")
    assert scanner._scanner_phase_status_from_durable(
        {"status": "failed"},
        phase2_only=True,
    ) == ("complete", "failed")


@pytest.mark.asyncio
async def test_cancel_job_workflow_ignores_completed_workflow(monkeypatch):
    from src.api.routes.v2 import temporal_client

    class FakeHandle:
        async def cancel(self):
            raise RuntimeError("workflow already completed")

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

    await temporal_client.cancel_job_workflow({"workflow_id": "workflow-1"})


@pytest.mark.asyncio
async def test_memory_scrape_activity_uses_dedicated_single_thread(monkeypatch):
    from src.api.routes import memory as memory_v1
    from src.api.routes.v2 import activities

    thread_names = []

    def fake_render(url):
        thread_names.append(threading.current_thread().name)
        return "<html></html>", url

    def fake_extract(_url, _html, _source_url):
        return "chatgpt", "dom", [
            memory_v1.MessagePair(user_query="hi", agent_response="hello"),
        ]

    monkeypatch.setattr(activities.memory_v1, "_render_chat_share_sync", fake_render)
    monkeypatch.setattr(activities.memory_v1, "_extract_chat_pairs", fake_extract)

    result = await activities.memory_scrape_activity({
        "url": "https://chatgpt.com/share/test",
    })

    assert result["pairs"][0]["user_query"] == "hi"
    assert thread_names
    assert all(name.startswith("xmem-scrape") for name in thread_names)
