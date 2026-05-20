from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from src.api.routes.health import router as health_router
from src.api.routes.memory import router as memory_router
from src.schemas.retrieval import RetrievalResult
from src.storage.base import SearchResult


class FakeIngestPipeline:
    model = SimpleNamespace(model="fake-ingest")

    async def run(self, **kwargs):
        return {"classification_result": SimpleNamespace(classifications=[])}

    def close(self):
        pass


class FakeVectorStore:
    async def search_by_text(self, query_text: str, top_k: int = 10, filters=None):
        domain = (filters or {}).get("domain", "summary")
        return [
            SearchResult(
                id=f"{domain}-1",
                content=f"{domain} hit for {query_text}",
                score=0.88,
                metadata={"domain": domain},
            )
        ]


class FakeNeo4j:
    def search_events_by_embedding(self, user_id: str, query_text: str, top_k: int = 3, similarity_threshold: float = 0.0):
        return [
            {
                "date": "May 20",
                "year": "2026",
                "event_name": "launch",
                "desc": f"Temporal hit for {query_text}",
                "similarity_score": 0.7,
            }
        ]


class FakeSearchModel:
    model = "fake-retrieval"

    async def ainvoke(self, messages):
        return SimpleNamespace(content="synthesized answer")


class FakeRetrievalPipeline:
    model = FakeSearchModel()
    vector_store = FakeVectorStore()
    neo4j = FakeNeo4j()

    async def run(self, query: str, user_id: str, top_k: int = 5):
        return RetrievalResult(query=query, answer=f"answer for {user_id}", sources=[], confidence=0.1)

    def _fetch_profile_catalog(self, user_id: str):
        records = [
            SearchResult(
                id="profile-1",
                content="Profile hit",
                score=0.95,
                metadata={"domain": "profile", "user_id": user_id},
            )
        ]
        return [{"topic": "work", "sub_topic": "role"}], records

    async def _search_snippet(self, query: str, user_id: str, top_k: int = 5):
        from src.api.schemas import SourceRecord

        return [
            SourceRecord(
                domain="snippet",
                content=f"Snippet hit for {query}",
                score=0.66,
                metadata={"user_id": user_id},
            )
        ]

    def close(self):
        pass


@pytest.fixture
def dependency_app(monkeypatch):
    monkeypatch.setattr(deps.settings, "api_keys", ["test-static-key"], raising=False)
    deps._init_error = None
    deps._pipelines_ready.set()
    deps.set_pipelines(FakeIngestPipeline(), FakeRetrievalPipeline())

    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health_router)
    app.include_router(memory_router)

    @app.get("/protected")
    async def protected(user: dict = Depends(deps.require_api_key)):
        return {"user_id": user["id"], "email": user["email"]}

    @app.get("/pipeline")
    async def pipeline(_ready=Depends(deps.require_ready)):
        return {"ingest": deps.get_ingest_pipeline().model.model}

    return app


def test_health_route_uses_readiness_state(dependency_app):
    deps.set_startup_time(0)

    response = TestClient(dependency_app).get("/health")

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "ready"


def test_auth_dependency_rejects_missing_and_accepts_static_bearer_key(dependency_app):
    client = TestClient(dependency_app)

    missing = client.get("/protected")
    assert missing.status_code == 401

    ok = client.get("/protected", headers={"Authorization": "Bearer test-static-key"})
    assert ok.status_code == 200
    assert ok.json()["email"] == "static@xmem.ai"
    assert ok.headers["x-content-type-options"] == "nosniff"
    assert "x-request-id" in ok.headers


def test_dependency_injection_returns_configured_pipeline(dependency_app):
    response = TestClient(dependency_app).get("/pipeline")

    assert response.status_code == 200
    assert response.json() == {"ingest": "fake-ingest"}


@pytest.mark.asyncio
async def test_rate_limiter_blocks_after_limit(monkeypatch):
    limiter = deps._SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    assert await limiter.check("user-1") == (True, 0)
    assert await limiter.check("user-1") == (False, 0)

def test_memory_search_route_returns_ranked_raw_results_and_latency(dependency_app):
    response = TestClient(dependency_app).post(
        "/v1/memory/search",
        headers={"Authorization": "Bearer test-static-key"},
        json={
            "query": "launch python",
            "user_id": "request-user",
            "domains": ["summary", "profile", "temporal", "snippet"],
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["mode"] == "raw"
    assert data["answer"] is None
    assert data["total"] == 4
    assert [record["domain"] for record in data["results"]] == ["profile", "summary", "temporal", "snippet"]
    assert "raw_p50" in data["latency_ms"]
    assert "summary" in data["latency_ms"]


def test_memory_search_route_synthesizes_answer_when_requested(dependency_app):
    response = TestClient(dependency_app).post(
        "/v1/memory/search",
        headers={"Authorization": "Bearer test-static-key"},
        json={
            "query": "what happened",
            "user_id": "request-user",
            "domains": ["summary"],
            "top_k": 1,
            "answer": True,
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["mode"] == "answer"
    assert data["answer"] == "synthesized answer"
    assert data["confidence"] > 0
    assert "answer" in data["latency_ms"]
    assert "answer_p95" in data["latency_ms"]
