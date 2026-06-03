from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.routes.v2 import memory as memory_v2
from src.storage.original import ORIGINAL_CHUNK_DOMAIN


class FakeIngestPipeline:
    model = SimpleNamespace(model="fake-ingest")


class FakeRetrievalPipeline:
    model = SimpleNamespace(model="fake-retrieval")

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.neo4j = SimpleNamespace()


def test_v2_hybrid_search_returns_memory_and_original_chunks(
    monkeypatch,
    vector_store,
):
    vector_store.seed(
        "summary-1",
        "Extracted memory about deterministic retries.",
        {"user_id": "hunter", "domain": "summary"},
        score=0.91,
    )
    vector_store.seed(
        "original-1",
        "Raw source chunk mentioning S3 and original preservation.",
        {
            "user_id": "hunter",
            "domain": ORIGINAL_CHUNK_DOMAIN,
            "original_doc_id": "doc-1",
            "s3_key": "originals/test/user/doc-1.json",
        },
        score=0.83,
    )
    vector_store.seed(
        "other-user",
        "Wrong user source chunk.",
        {"user_id": "someone-else", "domain": ORIGINAL_CHUNK_DOMAIN},
        score=0.99,
    )

    deps._init_error = None
    deps._pipelines_ready.set()
    deps.set_pipelines(FakeIngestPipeline(), FakeRetrievalPipeline(vector_store))
    monkeypatch.setattr(memory_v2.settings, "original_storage_enabled", True, raising=False)

    async def fake_user():
        return {"id": "user-1", "username": "hunter"}

    async def fake_ready():
        return None

    async def fake_rate_limit():
        return None

    app = FastAPI()
    app.dependency_overrides[deps.require_api_key] = fake_user
    app.dependency_overrides[deps.require_ready] = fake_ready
    app.dependency_overrides[deps.enforce_rate_limit] = fake_rate_limit
    app.include_router(memory_v2.router)

    response = TestClient(app).post(
        "/v2/memory/hybrid-search",
        json={
            "query": "where is the original S3 preservation plan?",
            "user_id": "hunter",
            "domains": ["summary"],
            "memory_top_k": 5,
            "original_top_k": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["original_storage_enabled"] is True
    assert [r["domain"] for r in data["memory_results"]] == ["summary"]
    assert [r["domain"] for r in data["original_chunks"]] == [ORIGINAL_CHUNK_DOMAIN]
    assert data["total"] == 2
    assert data["original_chunks"][0]["metadata"]["original_doc_id"] == "doc-1"
