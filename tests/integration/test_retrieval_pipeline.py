from __future__ import annotations

import pytest

from src.pipelines.retrieval import RetrievalPipeline
from src.schemas.retrieval import SourceRecord
from tests.conftest import FakeChatModel, FakeLLMResponse


@pytest.mark.asyncio
async def test_retrieval_pipeline_executes_tool_calls_and_generates_answer(vector_store, neo4j_client):
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
            FakeLLMResponse("", tool_calls=[
                {"name": "search_profile", "args": {"topic": "work"}, "id": "call-profile"},
                {"name": "search_temporal", "args": {"query": "launch"}, "id": "call-event"},
            ])
        ],
        responses=["Alice works at XMem and has a launch on 05-11."],
    )
    pipeline = RetrievalPipeline(model=model, vector_store=vector_store, neo4j_client=neo4j_client)

    result = await pipeline.run("Where does Alice work and what is upcoming?", "alice")

    assert "XMem" in result.answer
    assert {source.domain for source in result.sources} == {"profile", "temporal", "summary"}
    assert result.confidence > 0.1


@pytest.mark.asyncio
async def test_retrieval_tool_dispatch_handles_unknown_and_snippet(vector_store, neo4j_client):
    model = FakeChatModel()
    pipeline = RetrievalPipeline(model=model, vector_store=vector_store, neo4j_client=neo4j_client)
    class SnippetStore:
        async def search_by_text(self, **kwargs):
            return [type("R", (), {
                "id": "snip-1",
                "content": "Binary search helper",
                "score": 0.9,
                "metadata": {"code_snippet": "def bs(): pass", "language": "python"},
            })()]

    snippet_store = SnippetStore()
    pipeline._snippet_stores["user-1"] = snippet_store

    assert await pipeline._execute_tool("missing_tool", {}, "user-1", 5) == []
    snippets = await pipeline._execute_tool("SearchSnippet", {"query": "binary search"}, "user-1", 5)
    assert snippets[0].domain == "snippet"
    assert "def bs" in snippets[0].content


@pytest.mark.asyncio
async def test_raw_search_returns_ranked_hits_without_llm_planning(vector_store, neo4j_client):
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
    neo4j_client.seed_event(
        user_id="alice",
        date="05-11",
        event_name="Launch",
        desc="Raw search launch",
        year="2026",
        similarity_score=0.8,
    )

    class SnippetStore:
        async def search_by_text(self, **kwargs):
            return [type("R", (), {
                "id": "snip-1",
                "content": "Search helper",
                "score": 0.95,
                "metadata": {"code_snippet": "def search(): pass", "language": "python"},
            })()]

    model = FakeChatModel(responses=["unused"])
    pipeline = RetrievalPipeline(model=model, vector_store=vector_store, neo4j_client=neo4j_client)
    pipeline._snippet_stores["alice"] = SnippetStore()

    result = await pipeline.raw_search(
        query="search launch",
        user_id="alice",
        domains=["profile", "summary", "temporal", "snippet"],
        top_k=5,
    )

    assert [source.domain for source in result.sources] == [
        "snippet", "summary", "temporal", "profile",
    ]
    assert result.answer == ""
    assert model.calls == []
    assert pipeline.model_with_tools.calls == []
    assert "raw_search" in pipeline.latency_snapshot()


@pytest.mark.asyncio
async def test_raw_search_can_synthesize_answer_after_hits(vector_store, neo4j_client):
    vector_store.seed(
        "summary-1",
        "Alice is building raw search.",
        {"user_id": "alice", "domain": "summary"},
    )
    model = FakeChatModel(responses=["Alice is building raw search."])
    pipeline = RetrievalPipeline(model=model, vector_store=vector_store, neo4j_client=neo4j_client)

    result = await pipeline.raw_search(
        query="What is Alice building?",
        user_id="alice",
        domains=["summary"],
        include_answer=True,
    )

    assert result.answer == "Alice is building raw search."
    assert len(model.calls) == 1
    assert pipeline.model_with_tools.calls == []
    assert "raw_search_answer" in pipeline.latency_snapshot()


@pytest.mark.asyncio
async def test_answer_from_sources_handles_missing_or_non_finite_source_scores(vector_store, neo4j_client):
    model = FakeChatModel(responses=["Answer synthesized from a missing-score source."])
    pipeline = RetrievalPipeline(model=model, vector_store=vector_store, neo4j_client=neo4j_client)

    answer = await pipeline.answer_from_sources(
        query="What is Alice building?",
        sources=[
            SourceRecord(
                domain="summary",
                content="Alice is building raw search.",
                score=None,
            ),
            SourceRecord(
                domain="summary",
                content="Alice is filtering non-finite scores.",
                score=float("inf"),
            ),
            SourceRecord(
                domain="summary",
                content="Alice is handling invalid score math.",
                score=float("nan"),
            ),
        ],
    )

    assert answer == "Answer synthesized from a missing-score source."
    assert len(model.calls) == 1
    prompt = model.calls[0][0].content
    assert "[summary] Alice is building raw search." in prompt
    assert "(score: inf" not in prompt
    assert "(score: nan" not in prompt


@pytest.mark.asyncio
async def test_retrieval_pipeline_reuses_cached_tool_plan(vector_store, neo4j_client):
    vector_store.seed(
        "profile-1",
        "work / company = XMem",
        {"user_id": "alice", "domain": "profile", "main_content": "work_company"},
    )
    model = FakeChatModel(
        tool_responses=[
            FakeLLMResponse("", tool_calls=[
                {"name": "search_profile", "args": {"topic": "work"}, "id": "call-profile"},
            ])
        ],
        responses=["first answer", "second answer"],
    )
    pipeline = RetrievalPipeline(model=model, vector_store=vector_store, neo4j_client=neo4j_client)

    first = await pipeline.run("Where does Alice work?", "alice")
    second = await pipeline.run("Where does Alice work?", "alice")

    assert first.answer == "first answer"
    assert second.answer == "second answer"
    assert len(pipeline.model_with_tools.calls) == 1
