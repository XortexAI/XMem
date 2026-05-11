from __future__ import annotations

import pytest

from src.agents.judge import JudgeAgent
from src.pipelines.retrieval import RetrievalPipeline
from src.pipelines.weaver import Weaver
from src.schemas.judge import JudgeDomain
from tests.conftest import FakeChatModel, FakeLLMResponse


@pytest.mark.asyncio
async def test_repository_scan_to_memory_ingestion_to_retrieval_to_agent_response(vector_store, neo4j_client, fast_embed):
    # Repository scan: a sample codebase produces a durable code summary.
    scanned_summary = "sample/src/app.py exposes handler() for health checks."
    vector_store.add(
        texts=[scanned_summary],
        embeddings=[fast_embed(scanned_summary)],
        metadata=[{"user_id": "alice", "domain": "summary", "repo": "sample"}],
    )

    # Memory ingestion: structured profile extraction is judged and written without an LLM.
    judge = JudgeAgent(model=FakeChatModel(), vector_store=vector_store)
    decision = await judge.arun_deterministic({
        "domain": "profile",
        "new_items": [{"topic": "work", "sub_topic": "company", "memo": "XMem"}],
        "user_id": "alice",
    })
    written = await Weaver(vector_store=vector_store, embed_fn=fast_embed).execute(
        decision,
        JudgeDomain.PROFILE,
        "alice",
    )
    assert written.succeeded == 1

    # Retrieval query: mocked tool-calling model asks for profile context; summary is auto-added.
    model = FakeChatModel(
        tool_responses=[
            FakeLLMResponse("", tool_calls=[
                {"name": "search_profile", "args": {"topic": "work"}, "id": "call-profile"}
            ])
        ],
        responses=["Alice works at XMem. The indexed sample repo has src/app.py with handler()."],
    )
    retrieval = RetrievalPipeline(model=model, vector_store=vector_store, neo4j_client=neo4j_client)
    result = await retrieval.run("Where does Alice work and what code was scanned?", "alice")

    assert "XMem" in result.answer
    assert "src/app.py" in result.answer
    assert {source.domain for source in result.sources} == {"profile", "summary"}
