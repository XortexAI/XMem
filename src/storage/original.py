"""Original document preservation for v2 memory ingest.

This module stores the raw input in an object store and indexes searchable
chunks in the configured vector backend. It is intentionally independent from
the extraction pipeline so v2 workflows can run both branches in parallel.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional

from src.config import settings
from src.config.effort import chunk_text, estimate_tokens
from src.storage.base import BaseVectorStore
from src.storage.factory import get_vector_store

logger = logging.getLogger("xmem.storage.original")

ORIGINAL_CHUNK_DOMAIN = "original_chunk"


class OriginalStorageError(RuntimeError):
    """Raised when original preservation cannot complete."""


@dataclass(frozen=True)
class OriginalConfig:
    enabled: bool
    provider: str
    environment: str
    bucket: Optional[str]
    region: str
    prefix: str
    endpoint_url: Optional[str]
    kms_key_id: Optional[str]
    multipart_threshold_bytes: int
    multipart_chunk_bytes: int
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    index_batch_size: int
    embed_concurrency: int
    index_concurrency: int
    max_bytes: int
    include_agent_response: bool
    include_image_url: bool


def original_config_snapshot() -> Dict[str, Any]:
    """Capture non-secret original-storage config into a durable job payload."""
    return {
        "enabled": bool(settings.original_storage_enabled),
        "provider": settings.original_storage_provider,
        "environment": settings.environment,
        "bucket": settings.original_s3_bucket,
        "region": settings.original_s3_region,
        "prefix": settings.original_s3_prefix,
        "endpoint_url": settings.original_s3_endpoint_url,
        "kms_key_id": settings.original_s3_kms_key_id,
        "multipart_threshold_bytes": int(settings.original_s3_multipart_threshold_bytes),
        "multipart_chunk_bytes": int(settings.original_s3_multipart_chunk_bytes),
        "chunk_size_tokens": int(settings.original_chunk_size_tokens),
        "chunk_overlap_tokens": int(settings.original_chunk_overlap_tokens),
        "index_batch_size": int(settings.original_index_batch_size),
        "embed_concurrency": int(settings.original_embed_concurrency),
        "index_concurrency": int(settings.original_index_concurrency),
        "max_bytes": int(settings.original_max_bytes),
        "include_agent_response": bool(settings.original_include_agent_response),
        "include_image_url": bool(settings.original_include_image_url),
    }


def _config_from_payload(payload: Mapping[str, Any]) -> OriginalConfig:
    raw = dict(payload.get("original_config") or {})
    if not raw:
        raw = original_config_snapshot()
    return OriginalConfig(
        enabled=bool(raw.get("enabled", settings.original_storage_enabled)),
        provider=str(raw.get("provider") or "s3").strip().lower(),
        environment=str(raw.get("environment") or settings.environment),
        bucket=raw.get("bucket") or None,
        region=str(raw.get("region") or "us-east-1"),
        prefix=str(raw.get("prefix") or "originals").strip("/"),
        endpoint_url=raw.get("endpoint_url") or None,
        kms_key_id=raw.get("kms_key_id") or None,
        multipart_threshold_bytes=max(int(raw.get("multipart_threshold_bytes") or 1), 1),
        multipart_chunk_bytes=max(int(raw.get("multipart_chunk_bytes") or 1), 1),
        chunk_size_tokens=max(int(raw.get("chunk_size_tokens") or 350), 1),
        chunk_overlap_tokens=max(int(raw.get("chunk_overlap_tokens") or 0), 0),
        index_batch_size=max(int(raw.get("index_batch_size") or 64), 1),
        embed_concurrency=max(int(raw.get("embed_concurrency") or 4), 1),
        index_concurrency=max(int(raw.get("index_concurrency") or 2), 1),
        max_bytes=max(int(raw.get("max_bytes") or 1), 1),
        include_agent_response=bool(raw.get("include_agent_response", True)),
        include_image_url=bool(raw.get("include_image_url", False)),
    )


class S3OriginalStore:
    def __init__(self, cfg: OriginalConfig) -> None:
        if not cfg.bucket:
            raise OriginalStorageError("ORIGINAL_S3_BUCKET is required.")
        if cfg.provider != "s3":
            raise OriginalStorageError(
                f"Unsupported ORIGINAL_STORAGE_PROVIDER={cfg.provider!r}."
            )
        self.cfg = cfg
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            kwargs: Dict[str, Any] = {
                "region_name": self.cfg.region,
                "config": Config(read_timeout=60),
            }
            if self.cfg.endpoint_url:
                kwargs["endpoint_url"] = self.cfg.endpoint_url
            if settings.aws_access_key_id and settings.aws_secret_access_key:
                kwargs["aws_access_key_id"] = settings.aws_access_key_id
                kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
            if settings.aws_session_token:
                kwargs["aws_session_token"] = settings.aws_session_token

            self._client = boto3.client("s3", **kwargs)
        return self._client

    def put_json(self, key: str, body: bytes) -> Dict[str, Any]:
        extra_args = {
            "ContentType": "application/json",
            "ServerSideEncryption": "AES256",
        }
        if self.cfg.kms_key_id:
            extra_args["ServerSideEncryption"] = "aws:kms"
            extra_args["SSEKMSKeyId"] = self.cfg.kms_key_id

        if len(body) >= self.cfg.multipart_threshold_bytes:
            from boto3.s3.transfer import TransferConfig

            transfer_config = TransferConfig(
                multipart_threshold=self.cfg.multipart_threshold_bytes,
                multipart_chunksize=self.cfg.multipart_chunk_bytes,
            )
            self.client.upload_fileobj(
                io.BytesIO(body),
                self.cfg.bucket,
                key,
                ExtraArgs=extra_args,
                Config=transfer_config,
            )
            return {"bucket": self.cfg.bucket, "key": key, "etag": None}

        response = self.client.put_object(
            Bucket=self.cfg.bucket,
            Key=key,
            Body=body,
            **extra_args,
        )
        return {
            "bucket": self.cfg.bucket,
            "key": key,
            "etag": response.get("ETag"),
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_hash(user_id: str) -> str:
    return _sha256_text(user_id)[:24]


def _content_parts(payload: Mapping[str, Any], cfg: OriginalConfig) -> List[str]:
    parts = []
    user_query = str(payload.get("user_query") or "").strip()
    agent_response = str(payload.get("agent_response") or "").strip()
    image_url = str(payload.get("image_url") or "").strip()

    if user_query:
        parts.append(f"[User]\n{user_query}")
    if cfg.include_agent_response and agent_response:
        parts.append(f"[Assistant]\n{agent_response}")
    if cfg.include_image_url and image_url:
        parts.append(f"[Image URL]\n{image_url}")
    return parts


def _build_original_json(
    payload: Mapping[str, Any],
    cfg: OriginalConfig,
    created_at: str,
    original_doc_id: str,
    content_sha256: str,
) -> Dict[str, Any]:
    obj: Dict[str, Any] = {
        "schema_version": 1,
        "original_doc_id": original_doc_id,
        "user_id_hash": _user_hash(str(payload.get("user_id") or "")),
        "source_type": payload.get("source_type") or "conversation",
        "content_sha256": content_sha256,
        "created_at": created_at,
        "session_datetime": payload.get("session_datetime") or "",
        "request_id": payload.get("request_id") or "",
        "job_id": payload.get("job_id") or "",
        "user_query": payload.get("user_query") or "",
    }
    if cfg.include_agent_response:
        obj["agent_response"] = payload.get("agent_response") or ""
    if cfg.include_image_url:
        obj["image_url"] = payload.get("image_url") or ""
    return obj


def _chunk_original_text(text: str, cfg: OriginalConfig) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if estimate_tokens(text) <= cfg.chunk_size_tokens:
        return [text]
    return chunk_text(
        text,
        chunk_size_tokens=cfg.chunk_size_tokens,
        overlap_tokens=cfg.chunk_overlap_tokens,
    )


async def _embed_chunks(
    chunks: List[str],
    embed_fn: Callable[[str], List[float]],
    concurrency: int,
) -> List[List[float]]:
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()

    async def _embed(chunk: str) -> List[float]:
        async with semaphore:
            return await loop.run_in_executor(None, embed_fn, chunk)

    return await asyncio.gather(*[_embed(chunk) for chunk in chunks])


async def _index_chunks(
    *,
    vector_store: BaseVectorStore,
    chunks: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    batch_size: int,
    concurrency: int,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()

    async def _upsert_batch(start: int) -> None:
        end = min(start + batch_size, len(chunks))
        async with semaphore:
            await loop.run_in_executor(
                None,
                partial(
                    vector_store.add,
                    texts=chunks[start:end],
                    embeddings=embeddings[start:end],
                    ids=ids[start:end],
                    metadata=metadatas[start:end],
                ),
            )

    await asyncio.gather(*[_upsert_batch(i) for i in range(0, len(chunks), batch_size)])


async def preserve_original(
    payload: Mapping[str, Any],
    *,
    vector_store: Optional[BaseVectorStore] = None,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
) -> Dict[str, Any]:
    """Store the raw original and index searchable chunks.

    The operation is retry-safe: S3 keys and vector IDs are deterministic for
    the same user/content pair, so Temporal can replay the activity.
    """
    cfg = _config_from_payload(payload)
    if not cfg.enabled:
        return {"status": "disabled", "indexed_chunks": 0}

    parts = _content_parts(payload, cfg)
    combined_text = "\n\n".join(parts).strip()
    if not combined_text:
        return {"status": "skipped", "reason": "empty_original", "indexed_chunks": 0}

    user_id = str(payload.get("user_id") or "default")
    content_sha256 = _sha256_text(combined_text)
    identity = {
        "user_id": user_id,
        "content_sha256": content_sha256,
        "session_datetime": payload.get("session_datetime") or "",
        "source_type": payload.get("source_type") or "conversation",
    }
    original_doc_id = _sha256_text(json.dumps(identity, sort_keys=True))[:32]
    created_at = _utc_now_iso()
    original_obj = _build_original_json(
        payload, cfg, created_at, original_doc_id, content_sha256
    )
    body = json.dumps(original_obj, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(body) > cfg.max_bytes:
        raise OriginalStorageError(
            f"Original object is {len(body)} bytes; max is {cfg.max_bytes}."
        )

    key = (
        f"{cfg.prefix}/{cfg.environment}/"
        f"{_user_hash(user_id)}/{original_doc_id}.json"
    )
    store = S3OriginalStore(cfg)
    write_result = await asyncio.to_thread(store.put_json, key, body)

    chunks = _chunk_original_text(combined_text, cfg)
    if not chunks:
        return {
            "status": "stored",
            "original_doc_id": original_doc_id,
            "bucket": write_result["bucket"],
            "s3_key": write_result["key"],
            "indexed_chunks": 0,
            "content_sha256": content_sha256,
        }

    if vector_store is None:
        vector_store = get_vector_store(namespace=settings.pinecone_namespace)
    if embed_fn is None:
        from src.pipelines.ingest import embed_text

        embed_fn = embed_text

    embeddings = await _embed_chunks(chunks, embed_fn, cfg.embed_concurrency)
    chunk_count = len(chunks)
    ids = [
        f"original:{original_doc_id}:chunk:{idx}"
        for idx in range(chunk_count)
    ]
    metadatas = [
        {
            "user_id": user_id,
            "domain": ORIGINAL_CHUNK_DOMAIN,
            "original_doc_id": original_doc_id,
            "s3_key": write_result["key"],
            "bucket": write_result["bucket"],
            "chunk_index": idx,
            "chunk_count": chunk_count,
            "content_sha256": content_sha256,
            "source_type": str(payload.get("source_type") or "conversation"),
            "created_at": created_at,
        }
        for idx in range(chunk_count)
    ]
    await _index_chunks(
        vector_store=vector_store,
        chunks=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
        batch_size=cfg.index_batch_size,
        concurrency=cfg.index_concurrency,
    )

    logger.info(
        "Preserved original_doc_id=%s chunks=%d s3_key=%s",
        original_doc_id,
        chunk_count,
        write_result["key"],
    )
    return {
        "status": "stored",
        "original_doc_id": original_doc_id,
        "bucket": write_result["bucket"],
        "s3_key": write_result["key"],
        "etag": write_result.get("etag"),
        "indexed_chunks": chunk_count,
        "content_sha256": content_sha256,
    }
