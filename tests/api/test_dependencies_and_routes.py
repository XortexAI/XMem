from __future__ import annotations
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from src.api.routes.health import router as health_router
from src.api.routes.memory import router as memory_router
from src.schemas.retrieval import RetrievalResult, SourceRecord
from src.storage.base import SearchResult


class FakeIngestPipeline:
    model = SimpleNamespace(model="fake-ingest")

    async def run(self, **kwargs):
        return {"classification_result": SimpleNamespace(classifications=[])}

    def close(self):
        pass


class FakeRetrievalPipeline:
    model = SimpleNamespace(model="fake-retrieval")

    def __init__(self):
        self.vector_store = SimpleNamespace(
            search_by_metadata=self._search_by_metadata,
            search_by_text=self._search_by_text,
        )
        self.neo4j = SimpleNamespace(search_events_by_embedding=self._search_events_by_embedding)

    def raw_retrieval_plan(self, domains, answer=False):
        ordered = ("profile", "temporal", "summary", "snippet", "code")
        return tuple(domain for domain in ordered if domain in set(domains))

    async def run(self, query: str, user_id: str, top_k: int = 5):
        return RetrievalResult(
            query=query,
            answer=f"answer for {user_id}",
            sources=[SourceRecord(domain="summary", content="answer source", score=0.7)],
            confidence=0.7,
        )

    def _search_by_metadata(self, filters, top_k=10):
        return [
            SearchResult(
                id="profile-1",
                content="Profile fact",
                score=0.4,
                metadata={"domain": "profile", "user_id": filters.get("user_id")},
            )
        ][:top_k]

    async def _search_by_text(self, query_text, top_k=10, filters=None):
        domain = (filters or {}).get("domain", "summary")
        return [
            SearchResult(
                id=f"{domain}-1",
                content=f"{domain} hit for {query_text}",
                score=0.8,
                metadata={"domain": domain},
            )
        ][:top_k]

    def _search_events_by_embedding(self, user_id, query_text, top_k=3, similarity_threshold=0.0):
        return [
            {
                "event_name": "Demo event",
                "desc": query_text,
                "date": "2026-05-21",
                "similarity_score": 0.9,
            }
        ][:top_k]

    async def _search_snippet(self, query: str, user_id: str, top_k: int = 5):
        return [SourceRecord(domain="snippet", content=f"snippet hit for {query}", score=0.6)]

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


def test_memory_search_returns_raw_hits_latency_and_optional_answer(dependency_app):
    client = TestClient(dependency_app)

    response = client.post(
        "/v1/memory/search",
        headers={"Authorization": "Bearer test-static-key"},
        json={
            "query": "fast lookup",
            "user_id": "ignored-for-auth-user",
            "domains": ["profile", "temporal", "summary", "snippet", "code"],
            "answer": True,
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["answer"].startswith("answer for Static Key User")
    assert data["total"] == 5
    assert {item["domain"] for item in data["results"]} == {
        "profile",
        "temporal",
        "summary",
        "snippet",
        "code",
    }
    assert {"profile", "temporal", "summary", "snippet", "code", "answer"} <= set(data["latency_ms"])
    assert data["latency_stats"]["summary"]["count"] >= 1


@pytest.mark.asyncio
async def test_rate_limiter_blocks_after_limit(monkeypatch):
    limiter = deps._SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    assert await limiter.check("user-1") == (True, 0)
    assert await limiter.check("user-1") == (False, 0)
