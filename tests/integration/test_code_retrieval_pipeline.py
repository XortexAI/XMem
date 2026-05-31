from __future__ import annotations

import pytest

from src.pipelines.code_retrieval import CodeRetrievalPipeline, _rrf_fuse
from src.schemas.retrieval import SourceRecord
from tests.conftest import FakeChatModel


class FakeCodeStore:
    def __init__(self):
        self.files = {
            ("acme", "sample", "src/app.py"): "def handler():\n    return 'ok'\n",
        }

    def get_file_content(self, org_id: str, repo: str, file_path: str):
        return self.files.get((org_id, repo, file_path))

    def close(self):
        pass


def test_rrf_fuse_ranks_items_seen_in_multiple_lists_higher():
    fused = _rrf_fuse(
        [
            [{"qualified_name": "A"}, {"qualified_name": "B"}],
            [{"qualified_name": "B"}, {"qualified_name": "C"}],
        ]
    )

    assert fused[0]["qualified_name"] == "B"
    assert fused[0]["rrf_score"] > fused[-1]["rrf_score"]


@pytest.mark.asyncio
async def test_code_retrieval_fast_path_reads_sample_codebase_file(monkeypatch):
    monkeypatch.setattr(
        "src.pipelines.code_retrieval._get_embed_fn",
        lambda: (lambda text: [1.0, 0.0, 0.0]),
    )
    pipeline = CodeRetrievalPipeline(
        org_id="acme",
        repos=["sample"],
        model=FakeChatModel(),
        store=FakeCodeStore(),
    )

    result = await pipeline.run("src/app.py", user_id="alice", repo="sample")

    assert result.confidence == 1.0
    assert "def handler" in result.answer
    assert result.sources[0].domain == "file_code"


@pytest.mark.asyncio
async def test_code_retrieval_raw_search_queries_symbols_and_files(monkeypatch):
    monkeypatch.setattr(
        "src.pipelines.code_retrieval._get_embed_fn",
        lambda: (lambda text: [1.0, 0.0, 0.0]),
    )
    pipeline = CodeRetrievalPipeline(
        org_id="acme",
        repos=["sample"],
        model=FakeChatModel(),
        store=FakeCodeStore(),
    )
    calls = []

    async def fake_execute_tool(tool_name, tool_args, repo, top_k, user_id=""):
        calls.append(
            {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "repo": repo,
                "top_k": top_k,
                "user_id": user_id,
            }
        )
        return [
            SourceRecord(
                domain=tool_name,
                content=f"{tool_name} result",
                score=0.9,
                metadata={"repo": repo},
            )
        ]

    monkeypatch.setattr(pipeline, "_execute_tool", fake_execute_tool)

    results = await pipeline.raw_search(
        "handler", user_id="alice", repo="sample", top_k=3
    )

    assert [call["tool_name"] for call in calls] == ["search_symbols", "search_files"]
    assert all(
        call["tool_args"] == {"query": "handler", "repo": "sample"} for call in calls
    )
    assert all(call["repo"] == "sample" for call in calls)
    assert all(call["top_k"] == 3 for call in calls)
    assert all(call["user_id"] == "alice" for call in calls)
    assert [source.domain for source in results] == ["search_symbols", "search_files"]


@pytest.mark.asyncio
async def test_code_retrieval_raw_search_dedupes_and_limits_results(monkeypatch):
    monkeypatch.setattr(
        "src.pipelines.code_retrieval._get_embed_fn",
        lambda: (lambda text: [1.0, 0.0, 0.0]),
    )
    pipeline = CodeRetrievalPipeline(
        org_id="acme",
        repos=["sample"],
        model=FakeChatModel(),
        store=FakeCodeStore(),
    )

    async def fake_execute_tool(tool_name, tool_args, repo, top_k, user_id=""):
        if tool_name == "search_symbols":
            return [
                SourceRecord(
                    domain="symbol",
                    content="old handler",
                    score=0.4,
                    metadata={
                        "repo": repo,
                        "qualified_name": "handler",
                        "file_path": "src/app.py",
                    },
                ),
                SourceRecord(
                    domain="symbol",
                    content="helper",
                    score=0.3,
                    metadata={
                        "repo": repo,
                        "qualified_name": "helper",
                        "file_path": "src/app.py",
                    },
                ),
                SourceRecord(
                    domain="symbol",
                    content="new handler",
                    score=0.9,
                    metadata={
                        "repo": repo,
                        "qualified_name": "handler",
                        "file_path": "src/app.py",
                    },
                ),
                SourceRecord(
                    domain="symbol",
                    content="extra",
                    score=0.2,
                    metadata={
                        "repo": repo,
                        "qualified_name": "extra",
                        "file_path": "src/extra.py",
                    },
                ),
            ]
        return [
            SourceRecord(
                domain="file",
                content="app file",
                score=0.8,
                metadata={"repo": repo, "file_path": "src/app.py"},
            ),
            SourceRecord(
                domain="file",
                content="util file",
                score=0.7,
                metadata={"repo": repo, "file_path": "src/util.py"},
            ),
        ]

    monkeypatch.setattr(pipeline, "_execute_tool", fake_execute_tool)

    results = await pipeline.raw_search(
        "handler", user_id="alice", repo="sample", top_k=3
    )

    assert len(results) == 3
    assert [source.domain for source in results] == ["symbol", "file", "symbol"]
    assert [source.content for source in results].count("new handler") == 1
    assert all(source.content != "old handler" for source in results)


@pytest.mark.asyncio
async def test_code_retrieval_raw_search_keeps_successful_tool_results(monkeypatch):
    monkeypatch.setattr(
        "src.pipelines.code_retrieval._get_embed_fn",
        lambda: (lambda text: [1.0, 0.0, 0.0]),
    )
    pipeline = CodeRetrievalPipeline(
        org_id="acme",
        repos=["sample"],
        model=FakeChatModel(),
        store=FakeCodeStore(),
    )

    async def fake_execute_tool(tool_name, tool_args, repo, top_k, user_id=""):
        if tool_name == "search_symbols":
            raise RuntimeError("symbol index unavailable")
        return [
            SourceRecord(
                domain="file",
                content="app file",
                score=0.8,
                metadata={"repo": repo, "file_path": "src/app.py"},
            )
        ]

    monkeypatch.setattr(pipeline, "_execute_tool", fake_execute_tool)

    results = await pipeline.raw_search(
        "handler", user_id="alice", repo="sample", top_k=3
    )

    assert len(results) == 1
    assert results[0].domain == "file"
    assert results[0].content == "app file"
