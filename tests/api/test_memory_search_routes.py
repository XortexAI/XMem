from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.middleware import RequestContextMiddleware
from src.api.routes.memory import router as memory_router
from src.api.routes.memory import search_router as memory_search_router
from src.schemas.retrieval import SourceRecord


class FakeSearchPipeline:
    model = SimpleNamespace(model="fake-retrieval")

    def __init__(self) -> None:
        self.answer_calls = 0
        self.search_calls: list[dict[str, object]] = []
        self.latencies: dict[str, list[float]] = {}

    async def search_raw(
        self, query: str, user_id: str, domains: list[str], top_k: int
    ):
        assert query == "latency"
        assert user_id == "Static Key User"
        assert top_k == 3
        self.search_calls.append(
            {"query": query, "user_id": user_id, "domains": domains, "top_k": top_k}
        )

        fixtures = {
            "summary": SourceRecord(
                domain="summary",
                content="Low-latency summary",
                score=0.9,
            ),
            "profile": SourceRecord(
                domain="profile",
                content="work / company = XMem",
                score=0.7,
            ),
            "code": SourceRecord(
                domain="code",
                content="[file=src/retry.py; symbol=RetryLoop] Timeout retry note",
                score=0.8,
                metadata={"target_file": "src/retry.py"},
            ),
        }
        return [fixtures[domain] for domain in domains if domain in fixtures]

    async def answer_from_sources(self, query: str, sources: list[SourceRecord]) -> str:
        self.answer_calls += 1
        return "Alice is working on low-latency retrieval."

    def record_latency(self, mode: str, elapsed_ms: float) -> None:
        self.latencies.setdefault(mode, []).append(elapsed_ms)

    def get_latency_snapshot(self):
        return {
            mode: {
                "count": len(samples),
                "p50": samples[-1],
                "p95": samples[-1],
                "p99": samples[-1],
            }
            for mode, samples in self.latencies.items()
        }


@pytest.fixture
def memory_search_app(monkeypatch):
    pipeline = FakeSearchPipeline()
    monkeypatch.setattr(deps.settings, "api_keys", ["test-static-key"], raising=False)
    deps._init_error = None
    deps._pipelines_ready.set()
    deps.set_pipelines(SimpleNamespace(), pipeline)

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    app.include_router(memory_search_router)
    app.include_router(memory_router)
    return app, pipeline


def test_memory_search_route_returns_raw_hits_without_answer(memory_search_app):
    app, pipeline = memory_search_app
    response = TestClient(app).post(
        "/v1/memory/search",
        headers={"Authorization": "Bearer test-static-key"},
        json={
            "query": "latency",
            "user_id": "ignored-by-auth",
            "domains": ["profile", "summary"],
            "top_k": 3,
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["data"]["total"] == 2
    assert payload["data"]["answer"] == ""
    assert payload["data"]["latency"]["raw"]["count"] == 1
    assert pipeline.search_calls[0]["domains"] == ["profile", "summary"]
    assert pipeline.answer_calls == 0


def test_root_search_alias_can_synthesize_answer(memory_search_app):
    app, pipeline = memory_search_app
    response = TestClient(app).post(
        "/search",
        headers={"Authorization": "Bearer test-static-key"},
        json={
            "query": "latency",
            "user_id": "ignored-by-auth",
            "domains": ["profile", "summary"],
            "top_k": 3,
            "answer": True,
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["data"]["answer"] == "Alice is working on low-latency retrieval."
    assert payload["data"]["model"] == "fake-retrieval"
    assert payload["data"]["latency"]["answer"]["count"] == 1
    assert pipeline.answer_calls == 1


def test_memory_search_route_accepts_code_domain(memory_search_app):
    app, pipeline = memory_search_app
    response = TestClient(app).post(
        "/v1/memory/search",
        headers={"Authorization": "Bearer test-static-key"},
        json={
            "query": "latency",
            "user_id": "ignored-by-auth",
            "domains": ["code"],
            "top_k": 3,
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["data"]["total"] == 1
    assert payload["data"]["results"][0]["domain"] == "code"
    assert payload["data"]["results"][0]["metadata"]["target_file"] == "src/retry.py"
    assert pipeline.search_calls[0]["domains"] == ["code"]
