from __future__ import annotations

import asyncio

import pytest

from src.pipelines.retrieval import (
    _RETRIEVAL_PLAN_CACHE_LIMIT,
    RetrievalPipeline,
)
from src.schemas.retrieval import SourceRecord
from tests.conftest import FakeChatModel, FakeLLMResponse


@pytest.mark.asyncio
async def test_retrieval_pipeline_executes_tool_calls_and_generates_answer(
    vector_store, neo4j_client
):
    vector_store.seed(
        "profile-1",
        "work / company = XMem",
        {"user_id": "alice", "domain": "profile", "main_content": "work_company"},
    )
    vector_store.seed(
        "summary-1",
        "Alice is building tests for XMem.",
        {"user_id": "alice", "domain": "summary"},
    )
    neo4j_client.seed_event(
        user_id="alice",
        date="05-11",
        event_name="Launch",
        desc="Testing architecture launch",
        year="2026",
        similarity_score=0.8,
    )
    model = FakeChatModel(
        tool_responses=[
            FakeLLMResponse(
                "",
                tool_calls=[
                    {
                        "name": "search_profile",
                        "args": {"topic": "work"},
                        "id": "call-profile",
                    },
                    {
                        "name": "search_temporal",
                        "args": {"query": "launch"},
                        "id": "call-event",
                    },
                ],
            )
        ],
        responses=["Alice works at XMem and has a launch on 05-11."],
    )
    pipeline = RetrievalPipeline(
        model=model, vector_store=vector_store, neo4j_client=neo4j_client
    )

    result = await pipeline.run("Where does Alice work and what is upcoming?", "alice")

    assert "XMem" in result.answer
    assert {source.domain for source in result.sources} == {
        "profile",
        "temporal",
        "summary",
    }
    assert result.confidence > 0.1


@pytest.mark.asyncio
async def test_retrieval_pipeline_caches_catalog_and_retrieval_plan(
    vector_store, neo4j_client
):
    vector_store.seed(
        "profile-1",
        "work / company = XMem",
        {"user_id": "alice", "domain": "profile", "main_content": "work_company"},
    )
    model = FakeChatModel(
        tool_responses=[
            FakeLLMResponse(
                "",
                tool_calls=[
                    {
                        "name": "search_profile",
                        "args": {"topic": "work"},
                        "id": "call-profile",
                    },
                ],
            )
        ],
        responses=["Alice works at XMem.", "Alice still works at XMem."],
    )
    pipeline = RetrievalPipeline(
        model=model, vector_store=vector_store, neo4j_client=neo4j_client
    )

    first = await pipeline.run("Where does Alice work?", "alice")
    second = await pipeline.run("Where does Alice work?", "alice")

    assert "XMem" in first.answer
    assert "XMem" in second.answer
    assert len(pipeline.model_with_tools.calls) == 1


@pytest.mark.asyncio
async def test_raw_search_returns_ranked_hits_without_tool_selection(
    vector_store, neo4j_client
):
    vector_store.seed(
        "profile-1",
        "work / company = XMem",
        {"user_id": "alice", "domain": "profile", "main_content": "work_company"},
        score=0.7,
    )
    vector_store.seed(
        "summary-1",
        "Alice is tuning low-latency retrieval.",
        {"user_id": "alice", "domain": "summary"},
        score=0.9,
    )
    vector_store.seed(
        "code-1",
        "RetryLoop can spin when the first retrieval attempt times out.",
        {
            "user_id": "alice",
            "domain": "code",
            "annotation_type": "bug_report",
            "target_symbol": "RetryLoop",
            "target_file": "src/retry.py",
            "repo": "xmem",
            "severity": "high",
        },
        score=0.95,
    )
    neo4j_client.seed_event(
        user_id="alice",
        date="05-11",
        event_name="Latency review",
        desc="Measured raw search latency",
        year="2026",
        similarity_score=0.8,
    )
    model = FakeChatModel()
    pipeline = RetrievalPipeline(
        model=model, vector_store=vector_store, neo4j_client=neo4j_client
    )

    results = await pipeline.search_raw(
        "latency",
        "alice",
        ["profile", "temporal", "summary", "code"],
        top_k=5,
    )

    assert [record.score for record in results] == sorted(
        [record.score for record in results],
        reverse=True,
    )
    assert {record.domain for record in results} == {
        "profile",
        "temporal",
        "summary",
        "code",
    }
    code_hit = next(record for record in results if record.domain == "code")
    assert "file=src/retry.py" in code_hit.content
    assert code_hit.metadata["target_symbol"] == "RetryLoop"
    assert not pipeline.model_with_tools.calls


@pytest.mark.asyncio
async def test_raw_search_runs_requested_domains_concurrently(
    vector_store, neo4j_client
):
    model = FakeChatModel()
    pipeline = RetrievalPipeline(
        model=model, vector_store=vector_store, neo4j_client=neo4j_client
    )
    started: list[str] = []
    release = asyncio.Event()

    async def fake_domain(name: str, score: float):
        started.append(name)
        if len(started) == 3:
            release.set()
        await asyncio.wait_for(release.wait(), timeout=0.1)
        return [SourceRecord(domain=name, content=f"{name} hit", score=score)]

    pipeline._search_profile_raw = lambda *_args: fake_domain("profile", 0.7)
    pipeline._search_temporal = lambda *_args: fake_domain("temporal", 0.8)
    pipeline._search_summary = lambda *_args: fake_domain("summary", 0.9)

    results = await pipeline.search_raw(
        "latency",
        "alice",
        ["profile", "temporal", "summary"],
        top_k=5,
    )

    assert set(started) == {"profile", "temporal", "summary"}
    assert [record.domain for record in results] == ["summary", "temporal", "profile"]


@pytest.mark.asyncio
async def test_profile_catalog_fetch_does_not_block_event_loop(
    vector_store, neo4j_client
):
    import threading

    class BlockingVectorStore:
        def search_by_metadata(self, filters, top_k=10):
            threading.Event().wait(0.05)
            return vector_store.search_by_metadata(filters, top_k=top_k)

        async def search_by_text(self, *args, **kwargs):
            return await vector_store.search_by_text(*args, **kwargs)

    vector_store.seed(
        "profile-1",
        "work / company = XMem",
        {"user_id": "alice", "domain": "profile", "main_content": "work_company"},
    )
    model = FakeChatModel()
    pipeline = RetrievalPipeline(
        model=model,
        vector_store=BlockingVectorStore(),
        neo4j_client=neo4j_client,
    )
    ticks: list[str] = []

    async def ticker():
        await asyncio.sleep(0.01)
        ticks.append("tick")

    tick_task = asyncio.create_task(ticker())
    catalog, records = await pipeline._get_profile_catalog("alice")
    await tick_task

    assert ticks == ["tick"]
    assert catalog == [{"topic": "work", "sub_topic": "company"}]
    assert len(records) == 1


def test_retrieval_plan_cache_evicts_oldest_entry(vector_store, neo4j_client):
    model = FakeChatModel()
    pipeline = RetrievalPipeline(
        model=model, vector_store=vector_store, neo4j_client=neo4j_client
    )
    first_key = ("alice", "query-0", 5, "catalog-0")

    for index in range(_RETRIEVAL_PLAN_CACHE_LIMIT + 1):
        key = ("alice", f"query-{index}", 5, f"catalog-{index}")
        pipeline._cache_retrieval_plan(key, FakeLLMResponse(f"response-{index}"))

    assert len(pipeline._retrieval_plan_cache) == _RETRIEVAL_PLAN_CACHE_LIMIT
    assert pipeline._get_cached_retrieval_plan(first_key) is None


@pytest.mark.asyncio
async def test_answer_from_sources_skips_tool_selection(vector_store, neo4j_client):
    model = FakeChatModel(responses=["Alice works at XMem."])
    pipeline = RetrievalPipeline(
        model=model, vector_store=vector_store, neo4j_client=neo4j_client
    )

    answer = await pipeline.answer_from_sources(
        "Where does Alice work?",
        [],
    )

    assert answer == "Alice works at XMem."
    assert not pipeline.model_with_tools.calls


@pytest.mark.asyncio
async def test_retrieval_tool_dispatch_handles_unknown_and_snippet(
    vector_store, neo4j_client
):
    model = FakeChatModel()
    pipeline = RetrievalPipeline(
        model=model, vector_store=vector_store, neo4j_client=neo4j_client
    )

    class SnippetStore:
        async def search_by_text(self, **kwargs):
            return [
                type(
                    "R",
                    (),
                    {
                        "id": "snip-1",
                        "content": "Binary search helper",
                        "score": 0.9,
                        "metadata": {
                            "code_snippet": "def bs(): pass",
                            "language": "python",
                        },
                    },
                )()
            ]

    snippet_store = SnippetStore()
    pipeline._snippet_stores["user-1"] = snippet_store

    assert await pipeline._execute_tool("missing_tool", {}, "user-1", 5) == []
    snippets = await pipeline._execute_tool(
        "SearchSnippet", {"query": "binary search"}, "user-1", 5
    )
    assert snippets[0].domain == "snippet"
    assert "def bs" in snippets[0].content
