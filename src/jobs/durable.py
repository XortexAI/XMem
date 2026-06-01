"""Mongo-backed durable job records and async runner helpers.

The API routes still own the actual domain work. This module provides the
shared persistence contract: idempotency keys, status transitions, retry
counts, timeouts, error state, and dead-letter marking.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Tuple

logger = logging.getLogger("xmem.jobs.durable")

JobHandler = Callable[[], Awaitable[Dict[str, Any] | None]]

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
DEAD_LETTER = "dead_letter"
CANCELLED = "cancelled"

TERMINAL_STATUSES = {SUCCEEDED, DEAD_LETTER, CANCELLED}
REDACTED_KEYS = {
    "authorization",
    "cookie",
    "gh_token",
    "pat",
    "password",
    "secret",
    "token",
}

_default_store: Optional["DurableJobStore"] = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalise(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalise(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _normalise(value),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def idempotency_key(job_type: str, fields: Mapping[str, Any]) -> str:
    return stable_hash({"job_type": job_type, "fields": fields})


def redact_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(secret_key in lowered for secret_key in REDACTED_KEYS):
                redacted[str(key)] = "[redacted]"
            else:
                redacted[str(key)] = redact_payload(item)
        return redacted
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    return value


def serialize_job(doc: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if doc is None:
        return None

    def convert(value: Any) -> Any:
        if value.__class__.__name__ == "ObjectId":
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Mapping):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(dict(doc))


class DurableJobStore:
    """Persistence layer for queue/workflow jobs."""

    def __init__(
        self,
        uri: Optional[str] = None,
        database: Optional[str] = None,
        collection: str = "durable_jobs",
    ) -> None:
        from pymongo import MongoClient
        from src.config import settings

        uri = uri or settings.mongodb_uri
        database = database or settings.mongodb_database

        self._client = MongoClient(uri)
        self._db = self._client[database]
        self.jobs = self._db[collection]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.jobs.create_index([("job_id", 1)], unique=True)
        self.jobs.create_index([("job_type", 1), ("idempotency_key", 1)], unique=True)
        self.jobs.create_index([("user_id", 1), ("updated_at", -1)])
        self.jobs.create_index([("status", 1), ("updated_at", 1)])
        self.jobs.create_index([("workflow_id", 1)])
        secrets = self.get_collection("durable_job_secrets")
        secrets.create_index([("secret_ref", 1)], unique=True)
        secrets.create_index([("job_id", 1)])

    def get_collection(self, name: str):
        return self._db[name]

    def enqueue(
        self,
        *,
        job_type: str,
        payload: Mapping[str, Any],
        idempotency_fields: Mapping[str, Any],
        user_id: str,
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
    ) -> Tuple[Dict[str, Any], bool]:
        key = idempotency_key(job_type, idempotency_fields)
        job_id = f"{job_type}:{key}"
        now = utc_now()
        doc = {
            "job_id": job_id,
            "job_type": job_type,
            "idempotency_key": key,
            "user_id": user_id,
            "payload": redact_payload(payload),
            "status": QUEUED,
            "retry_count": 0,
            "attempt_count": 0,
            "max_attempts": max_attempts,
            "timeout_seconds": timeout_seconds,
            "error": None,
            "error_state": None,
            "result": None,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "dead_lettered_at": None,
            "cancelled_at": None,
            "workflow_id": None,
            "run_id": None,
            "progress": None,
        }
        try:
            self.jobs.insert_one(doc)
            return doc, True
        except Exception as exc:
            if exc.__class__.__name__ != "DuplicateKeyError":
                raise
            existing = self.get(job_id)
            if existing is None:
                existing = self.jobs.find_one({
                    "job_type": job_type,
                    "idempotency_key": key,
                })
            if existing is None:
                raise
            return existing, False

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.find_one({"job_id": job_id})


    def record_workflow(
        self,
        job_id: str,
        workflow_id: str,
        run_id: Optional[str] = None,
    ) -> None:
        self.jobs.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "updated_at": utc_now(),
                },
            },
        )

    def reserve_workflow_start(self, job_id: str, workflow_id: str) -> bool:
        result = self.jobs.update_one(
            {"job_id": job_id, "status": QUEUED, "workflow_id": None},
            {
                "$set": {
                    "workflow_id": workflow_id,
                    "updated_at": utc_now(),
                },
            },
        )
        return getattr(result, "modified_count", 0) == 1

    def mark_running(self, job_id: str) -> None:
        now = utc_now()
        job = self.jobs.find_one_and_update(
            {"job_id": job_id, "status": {"$nin": list(TERMINAL_STATUSES)}},
            {
                "$inc": {"attempt_count": 1},
                "$set": {
                    "status": RUNNING,
                    "started_at": now,
                    "updated_at": now,
                    "error": None,
                    "error_state": None,
                },
            },
            return_document=True,
        )
        if job is None:
            return

        attempt_count = int(job.get("attempt_count") or 0)
        self.jobs.update_one(
            {
                "job_id": job_id,
                "attempt_count": attempt_count,
                "status": {"$nin": list(TERMINAL_STATUSES)},
            },
            {"$set": {"retry_count": max(attempt_count - 1, 0)}},
        )

    def update_progress(self, job_id: str, progress: Mapping[str, Any]) -> None:
        self.jobs.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "progress": _normalise(progress),
                    "updated_at": utc_now(),
                },
            },
        )

    def update_payload(self, job_id: str, payload: Mapping[str, Any]) -> None:
        self.jobs.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "payload": redact_payload(payload),
                    "updated_at": utc_now(),
                },
            },
        )

    def mark_dead_letter(self, job_id: str, error: str) -> None:
        now = utc_now()
        self.jobs.update_one(
            {"job_id": job_id, "status": {"$nin": list(TERMINAL_STATUSES)}},
            {
                "$set": {
                    "status": DEAD_LETTER,
                    "error": error,
                    "error_state": {
                        "message": error,
                        "failed_at": now,
                    },
                    "dead_lettered_at": now,
                    "completed_at": now,
                    "updated_at": now,
                },
            },
        )

    def mark_cancelled(self, job_id: str) -> None:
        now = utc_now()
        self.jobs.update_one(
            {"job_id": job_id, "status": {"$nin": list(TERMINAL_STATUSES)}},
            {
                "$set": {
                    "status": CANCELLED,
                    "cancelled_at": now,
                    "completed_at": now,
                    "updated_at": now,
                },
            },
        )

    def list_by_status(
        self,
        status: str,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[Dict[str, Any]]:
        query: Dict[str, Any] = {"status": status}
        if user_id:
            query["user_id"] = user_id
        cursor = self.jobs.find(query).sort("updated_at", -1).limit(limit)
        return list(cursor)

    def claim_for_run(self, job_id: str) -> bool:
        job = self.get(job_id) or {}
        attempt_count = int(job.get("attempt_count") or 0) + 1
        result = self.jobs.update_one(
            {"job_id": job_id, "status": QUEUED},
            {
                "$set": {
                    "status": RUNNING,
                    "attempt_count": attempt_count,
                    "retry_count": max(attempt_count - 1, 0),
                    "started_at": utc_now(),
                    "updated_at": utc_now(),
                    "error": None,
                    "error_state": None,
                },
            },
        )
        return result.modified_count == 1

    def mark_succeeded(
        self,
        job_id: str,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        now = utc_now()
        self.jobs.update_one(
            {"job_id": job_id, "status": {"$nin": list(TERMINAL_STATUSES)}},
            {
                "$set": {
                    "status": SUCCEEDED,
                    "result": _normalise(result or {}),
                    "error": None,
                    "error_state": None,
                    "completed_at": now,
                    "updated_at": now,
                },
            },
        )

    def mark_failed(self, job_id: str, error: str) -> str:
        job = self.get(job_id) or {}
        if job.get("status") in TERMINAL_STATUSES:
            return str(job.get("status"))
        attempt_count = int(job.get("attempt_count") or 0)
        retry_count = max(attempt_count - 1, 0)
        max_attempts = int(job.get("max_attempts") or 1)
        status = DEAD_LETTER if attempt_count >= max_attempts else FAILED
        update: Dict[str, Any] = {
            "status": status,
            "retry_count": retry_count,
            "error": error,
            "error_state": {
                "message": error,
                "failed_at": utc_now(),
                "attempt": attempt_count,
                "retry_count": retry_count,
            },
            "updated_at": utc_now(),
        }
        if status == DEAD_LETTER:
            update["dead_lettered_at"] = utc_now()
            update["completed_at"] = utc_now()
        result = self.jobs.update_one(
            {"job_id": job_id, "status": {"$nin": list(TERMINAL_STATUSES)}},
            {"$set": update},
        )
        if getattr(result, "modified_count", 0) != 1:
            current = self.get(job_id) or {}
            return str(current.get("status") or status)
        return status

    def reset_for_retry(self, job_id: str, clear_workflow: bool = False) -> None:
        update = {
            "status": QUEUED,
            "updated_at": utc_now(),
            "error": None,
            "error_state": None,
            "completed_at": None,
            "dead_lettered_at": None,
            "cancelled_at": None,
        }
        if clear_workflow:
            update["workflow_id"] = None
            update["run_id"] = None
        self.jobs.update_one(
            {"job_id": job_id},
            {"$set": update},
        )


def get_default_job_store() -> DurableJobStore:
    global _default_store
    if _default_store is None:
        _default_store = DurableJobStore()
    return _default_store


async def run_job(
    store: DurableJobStore,
    job_id: str,
    handler: JobHandler,
    *,
    retry_base_seconds: float = 1.0,
) -> None:
    """Run one queued job and persist lifecycle transitions."""
    while True:
        job = await asyncio.to_thread(store.get, job_id)
        if not job:
            logger.warning("Durable job %s disappeared before execution", job_id)
            return
        if job.get("status") in TERMINAL_STATUSES:
            return

        timeout_seconds = float(job.get("timeout_seconds") or 120.0)
        claimed = await asyncio.to_thread(store.claim_for_run, job_id)
        if not claimed:
            logger.info(
                "Durable job %s was already claimed; skipping duplicate runner",
                job_id,
            )
            return

        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(handler(), timeout=timeout_seconds)
            payload = dict(result or {})
            payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
            await asyncio.to_thread(store.mark_succeeded, job_id, payload)
            return
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            status = await asyncio.to_thread(store.mark_failed, job_id, error)
            if status == DEAD_LETTER:
                logger.exception("Durable job %s dead-lettered: %s", job_id, error)
                return
            job = await asyncio.to_thread(store.get, job_id) or {}
            attempt_count = int(job.get("attempt_count") or 1)
            delay = min(retry_base_seconds * (2 ** max(attempt_count - 1, 0)), 30.0)
            logger.warning(
                "Durable job %s failed; retrying in %.1fs: %s",
                job_id,
                delay,
                error,
            )
            await asyncio.sleep(delay)
            await asyncio.to_thread(store.reset_for_retry, job_id)


def new_attempt_id() -> str:
    return uuid.uuid4().hex
