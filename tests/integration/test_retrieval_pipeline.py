from __future__ import annotations

import pytest

from src.pipelines.retrieval import RetrievalPipeline
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
