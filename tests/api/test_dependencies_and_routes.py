from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.routes import memory as memory_routes
from src.api.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from src.api.routes.health import router as health_router
from src.schemas.retrieval import RetrievalResult, SourceRecord as RetrievalSourceRecord


class FakeIngestPipeline:
    model = SimpleNamespace(model="fake-ingest")

    async def run(self, **kwargs):
        return {"classification_result": SimpleNamespace(classifications=[])}

    def close(self):
        pass


class FakeRetrievalPipeline:
    model = SimpleNamespace(model="fake-retrieval")

    async def run(self, query: str, user_id: str, top_k: int = 5):
        return RetrievalResult(
            query=query, answer=f"answer for {user_id}", sources=[], confidence=0.1
        )

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


@pytest.mark.asyncio
async def test_search_code_uses_public_raw_search(monkeypatch):
    class FakeCodePipeline:
        def __init__(self):
            self.calls = []

        async def raw_search(self, query: str, user_id: str, repo: str, top_k: int):
            self.calls.append(
                {
                    "query": query,
                    "user_id": user_id,
                    "repo": repo,
                    "top_k": top_k,
                }
            )
            return [
                RetrievalSourceRecord(
                    domain="file_code",
                    content="def handler(): pass",
                    score=0.8,
                    metadata={"file_path": "src/app.py"},
                )
            ]

    fake_pipeline = FakeCodePipeline()
    monkeypatch.setattr(
        memory_routes,
        "get_code_pipeline",
        lambda org_id, repo: fake_pipeline,
    )

    results = await memory_routes._search_code(
        query="handler",
        user_id="alice",
        org_id="acme",
        repo="sample",
        top_k=2,
    )

    assert fake_pipeline.calls == [
        {
            "query": "handler",
            "user_id": "alice",
            "repo": "sample",
            "top_k": 2,
        }
    ]
    assert len(results) == 1
    assert results[0].domain == "code"
    assert results[0].content == "def handler(): pass"
    assert results[0].metadata == {
        "source_domain": "file_code",
        "file_path": "src/app.py",
    }
