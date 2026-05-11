from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from src.api.routes.health import router as health_router
from src.database.control_plane_store import _in_memory_rate_limits
from src.schemas.retrieval import RetrievalResult


class FakeIngestPipeline:
    model = SimpleNamespace(model="fake-ingest")

    async def run(self, **kwargs):
        return {"classification_result": SimpleNamespace(classifications=[])}

    def close(self):
        pass


class FakeRetrievalPipeline:
    model = SimpleNamespace(model="fake-retrieval")

    async def run(self, query: str, user_id: str, top_k: int = 5):
        return RetrievalResult(query=query, answer=f"answer for {user_id}", sources=[], confidence=0.1)

    def close(self):
        pass


@pytest.fixture
def dependency_app(monkeypatch):
    monkeypatch.setattr(deps.settings, "environment", "dev", raising=False)
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
    monkeypatch.setattr(deps.settings, "environment", "dev", raising=False)
    _in_memory_rate_limits.clear()
    assert await deps._rate_limiter.check("user-1") == (True, deps.settings.rate_limit - 1)
    limiter = deps._rate_limiter
    for _ in range(deps.settings.rate_limit - 1):
        await limiter.check("user-2")
    assert await limiter.check("user-2") == (True, 0)
    assert await limiter.check("user-2") == (False, 0)
