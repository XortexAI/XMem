from __future__ import annotations

import json

import pytest

from src.storage import original


class FakeS3OriginalStore:
    writes = []

    def __init__(self, cfg):
        self.cfg = cfg

    def put_json(self, key, body):
        self.__class__.writes.append((key, body))
        return {"bucket": self.cfg.bucket, "key": key, "etag": '"fake-etag"'}


@pytest.mark.asyncio
async def test_preserve_original_stores_s3_json_and_indexes_deterministic_chunks(
    monkeypatch,
    vector_store,
):
    FakeS3OriginalStore.writes = []
    monkeypatch.setattr(original, "S3OriginalStore", FakeS3OriginalStore)
    monkeypatch.setattr(original.settings, "environment", "test", raising=False)

    payload = {
        "user_id": "alice@example.com",
        "user_query": "Please remember the migration plan. " * 80,
        "agent_response": "Use deterministic ids and retries. " * 80,
        "session_datetime": "2026-06-03T10:00:00Z",
        "job_id": "memory_ingest:test",
        "original_config": {
            "enabled": True,
            "provider": "s3",
            "bucket": "xmem-originals-test",
            "region": "us-east-1",
            "prefix": "originals",
            "chunk_size_tokens": 80,
            "chunk_overlap_tokens": 10,
            "index_batch_size": 2,
            "embed_concurrency": 2,
            "index_concurrency": 1,
            "max_bytes": 10_000_000,
            "include_agent_response": True,
            "include_image_url": False,
        },
    }

    result = await original.preserve_original(
        payload,
        vector_store=vector_store,
        embed_fn=lambda _text: [0.0, 0.0, 0.0],
    )
    second = await original.preserve_original(
        payload,
        vector_store=vector_store,
        embed_fn=lambda _text: [0.0, 0.0, 0.0],
    )

    assert result["status"] == "stored"
    assert result["original_doc_id"] == second["original_doc_id"]
    assert result["indexed_chunks"] > 1
    assert "alice@example.com" not in result["s3_key"]
    assert all(
        record_id.startswith(f"original:{result['original_doc_id']}:chunk:")
        for record_id in vector_store.records
    )
    assert all(
        record["metadata"]["domain"] == original.ORIGINAL_CHUNK_DOMAIN
        for record in vector_store.records.values()
    )

    key, body = FakeS3OriginalStore.writes[0]
    stored = json.loads(body)
    assert key == result["s3_key"]
    assert stored["original_doc_id"] == result["original_doc_id"]
    assert stored["user_id_hash"]
    assert "user_id" not in stored


@pytest.mark.asyncio
async def test_preserve_original_disabled_is_noop(vector_store):
    result = await original.preserve_original(
        {
            "user_id": "alice",
            "user_query": "remember this",
            "original_config": {"enabled": False},
        },
        vector_store=vector_store,
        embed_fn=lambda _text: [0.0, 0.0, 0.0],
    )

    assert result == {"status": "disabled", "indexed_chunks": 0}
    assert vector_store.records == {}
