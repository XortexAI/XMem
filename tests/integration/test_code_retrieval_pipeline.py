from __future__ import annotations

import pytest

from src.pipelines.code_retrieval import CodeRetrievalPipeline, _rrf_fuse
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
    fused = _rrf_fuse([
        [{"qualified_name": "A"}, {"qualified_name": "B"}],
        [{"qualified_name": "B"}, {"qualified_name": "C"}],
    ])

    assert fused[0]["qualified_name"] == "B"
    assert fused[0]["rrf_score"] > fused[-1]["rrf_score"]


@pytest.mark.asyncio
async def test_code_retrieval_fast_path_reads_sample_codebase_file(monkeypatch):
    monkeypatch.setattr("src.pipelines.code_retrieval._get_embed_fn", lambda: (lambda text: [1.0, 0.0, 0.0]))
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
