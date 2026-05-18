"""Durable background jobs for request-time memory work."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional

from pymongo import ASCENDING, ReturnDocument

from src.config import settings

logger = logging.getLogger("xmem.jobs")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


JobHandler = Callable[[Dict[str, Any]], Awaitable[Any]]


class JobStore:
    """Mongo-backed durable queue with deterministic idempotency keys."""

    def __init__(self, uri: str | None = None, database: str | None = None) -> None:
        from pymongo import MongoClient

        self._client = MongoClient(uri or settings.mongodb_uri, serverSelectionTimeoutMS=5000)
        self._client.admin.command("ping")
        self._db = self._client[database or settings.mongodb_database]
        self.jobs = self._db["jobs"]
        self.dead_letters = self._db["job_dead_letters"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.jobs.create_index([("idempotency_key", ASCENDING)], unique=True, sparse=True)
        self.jobs.create_index([("status", ASCENDING), ("run_after", ASCENDING)])
        self.jobs.create_index([("owner_id", ASCENDING), ("created_at", ASCENDING)])
        self.dead_letters.create_index([("job_id", ASCENDING)], unique=True)

    @staticmethod
    def make_idempotency_key(job_type: str, owner_id: str, payload: Dict[str, Any]) -> str:
        stable = json.dumps(payload, default=str, separators=(",", ":"), sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(stable).hexdigest()
        return f"{job_type}:{owner_id}:{digest}"

    def enqueue(
        self,
        *,
        job_type: str,
        owner_id: str,
        payload: Dict[str, Any],
        idempotency_key: str | None = None,
        timeout_seconds: float | None = None,
        lease_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> Dict[str, Any]:
        now = _now()
        job_id = str(uuid.uuid4())
        key = idempotency_key or self.make_idempotency_key(job_type, owner_id, payload)
        effective_timeout = settings.job_timeout_seconds if timeout_seconds is None else timeout_seconds
        effective_lease = (
            max(settings.job_lease_seconds, effective_timeout)
            if lease_seconds is None
            else lease_seconds
        )
        doc = {
            "job_id": job_id,
            "job_type": job_type,
            "owner_id": owner_id,
            "payload": payload,
            "idempotency_key": key,
            "status": JobStatus.PENDING.value,
            "retry_count": 0,
            "max_retries": settings.job_max_retries if max_retries is None else max_retries,
            "timeout_seconds": effective_timeout,
            "lease_seconds": effective_lease,
            "error": None,
            "result": None,
            "run_after": now,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
        }

        existing = self.jobs.find_one({"idempotency_key": key})
        if existing:
            return existing

        try:
            self.jobs.insert_one(doc)
            return doc
        except Exception:
            existing = self.jobs.find_one({"idempotency_key": key})
            if existing:
                return existing
            raise

    def get(self, job_id: str, owner_id: str | None = None) -> Optional[Dict[str, Any]]:
        query: Dict[str, Any] = {"job_id": job_id}
        if owner_id is not None:
            query["owner_id"] = owner_id
        return self.jobs.find_one(query, {"_id": False})

    def claim_next(self, worker_id: str) -> Optional[Dict[str, Any]]:
        now = _now()
        update = {
            "$set": {
                "status": JobStatus.RUNNING.value,
                "worker_id": worker_id,
                "started_at": now,
                "updated_at": now,
            }
        }
        pending = self.jobs.find_one_and_update(
            {"status": JobStatus.PENDING.value, "run_after": {"$lte": now}},
            update,
            sort=[("run_after", ASCENDING), ("created_at", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )
        if pending:
            return pending

        running = self.jobs.find(
            {"status": JobStatus.RUNNING.value, "started_at": {"$ne": None}},
        ).sort([("started_at", ASCENDING)])
        for job in running:
            try:
                lease_seconds = float(job.get("lease_seconds") or settings.job_lease_seconds)
            except (TypeError, ValueError):
                lease_seconds = settings.job_lease_seconds
            stale_before = now - timedelta(seconds=lease_seconds)
            if job.get("started_at") >= stale_before:
                continue
            claimed = self.jobs.find_one_and_update(
                {
                    "job_id": job["job_id"],
                    "status": JobStatus.RUNNING.value,
                    "started_at": job.get("started_at"),
                },
                update,
                return_document=ReturnDocument.AFTER,
            )
            if claimed:
                return claimed
        return None

    def succeed(self, job_id: str, result: Any) -> None:
        now = _now()
        self.jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": JobStatus.SUCCEEDED.value,
                "result": result,
                "error": None,
                "finished_at": now,
                "updated_at": now,
            }},
        )

    def fail_or_retry(self, job: Dict[str, Any], error: str) -> None:
        now = _now()
        retry_count = int(job.get("retry_count") or 0) + 1
        max_retries = int(job.get("max_retries") or 0)
        job_id = job["job_id"]

        if retry_count > max_retries:
            update = {
                "$set": {
                    "status": JobStatus.DEAD_LETTER.value,
                    "retry_count": retry_count,
                    "error": error,
                    "finished_at": now,
                    "updated_at": now,
                }
            }
            self.jobs.update_one({"job_id": job_id}, update)
            dead_letter_job = {k: v for k, v in job.items() if k != "_id"}
            dead_letter_job.update(update["$set"])
            self.dead_letters.update_one(
                {"job_id": job_id},
                {"$set": {"job_id": job_id, "job": dead_letter_job, "error": error, "created_at": now}},
                upsert=True,
            )
            return

        delay = min(settings.job_retry_backoff_seconds * (2 ** (retry_count - 1)), 300)
        self.jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": JobStatus.PENDING.value,
                "retry_count": retry_count,
                "error": error,
                "run_after": now + timedelta(seconds=delay),
                "updated_at": now,
            }},
        )


class JobWorker:
    """Small async poller that executes jobs from the durable queue."""

    def __init__(self, store: JobStore, handlers: Dict[str, JobHandler]) -> None:
        self.store = store
        self.handlers = handlers
        self.worker_id = str(uuid.uuid4())
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopping.set()
        if self._task:
            await self._task

    async def _run(self) -> None:
        while not self._stopping.is_set():
            from src.api.dependencies import is_ready

            if not is_ready():
                await asyncio.sleep(settings.job_poll_interval_seconds)
                continue

            job = await asyncio.to_thread(self.store.claim_next, self.worker_id)
            if not job:
                await asyncio.sleep(settings.job_poll_interval_seconds)
                continue

            handler = self.handlers.get(job.get("job_type"))
            if handler is None:
                await asyncio.to_thread(
                    self.store.fail_or_retry,
                    job,
                    f"No handler registered for job type {job.get('job_type')!r}",
                )
                continue

            try:
                result = await asyncio.wait_for(
                    handler(dict(job.get("payload") or {})),
                    timeout=float(job.get("timeout_seconds") or settings.job_timeout_seconds),
                )
                await asyncio.to_thread(self.store.succeed, job["job_id"], result)
            except Exception as exc:
                logger.exception("Job %s failed", job.get("job_id"))
                await asyncio.to_thread(self.store.fail_or_retry, job, str(exc) or repr(exc))


_job_store: JobStore | None = None
_job_worker: JobWorker | None = None


def init_jobs(handlers: Dict[str, JobHandler]) -> JobStore | None:
    global _job_store, _job_worker
    if not settings.job_worker_enabled:
        return None
    try:
        _job_store = JobStore()
        _job_worker = JobWorker(_job_store, handlers)
        _job_worker.start()
        logger.info("[jobs] Worker started.")
        return _job_store
    except Exception as exc:
        logger.warning("[jobs] Worker disabled; MongoDB job store unavailable: %s", exc)
        _job_store = None
        _job_worker = None
        return None


async def shutdown_jobs() -> None:
    global _job_worker
    if _job_worker is not None:
        await _job_worker.stop()
        _job_worker = None


def get_job_store() -> JobStore:
    if _job_store is None:
        raise RuntimeError("Job store is not available.")
    return _job_store


def serialize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return _public_job(job)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _public_job(job: Dict[str, Any]) -> Dict[str, Any]:
    public = {k: v for k, v in job.items() if k != "_id"}
    public.pop("payload", None)
    return public
