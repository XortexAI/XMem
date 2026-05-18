"""Durable control-plane storage for short-lived auth and rate-limit state."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import string
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Optional

from src.config import settings

logger = logging.getLogger("xmem.database.control_plane_store")

_memory_records: dict[str, dict[str, Any]] = {}
_memory_rate_limits: dict[str, list[float]] = defaultdict(list)
_memory_lock = RLock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _random_token(prefix: str, length: int = 32) -> str:
    alphabet = string.ascii_letters + string.digits
    return f"{prefix}{''.join(secrets.choice(alphabet) for _ in range(length))}"


class ControlPlaneStore:
    """MongoDB-backed store for OAuth codes, temp tokens, sessions, and rate limits."""

    def __init__(self, uri: Optional[str] = None, database: Optional[str] = None) -> None:
        self._uri = uri or settings.mongodb_uri
        self._database = database or settings.mongodb_database
        self._connection_lock = RLock()
        self._client = None
        self._db = None
        self.records = None
        self.rate_limits = None
        self._connected = False
        self._in_memory = False

    def _requires_durable_storage(self) -> bool:
        return settings.environment.lower() in {"production", "prod"}

    def _ensure_ready(self) -> None:
        if self._connected or self._in_memory:
            return

        with self._connection_lock:
            if self._connected or self._in_memory:
                return
            self._try_connect()

    def _enable_in_memory_fallback(self, error: Exception) -> None:
        message = f"MongoDB connection failed for control-plane storage: {error}"
        if self._requires_durable_storage():
            logger.error("%s; refusing in-memory fallback in production", message)
            raise RuntimeError(
                "MongoDB is required for control-plane storage when ENVIRONMENT=production"
            ) from error

        logger.warning("%s; using in-memory storage", message)
        self._connected = False
        self._in_memory = True
        self.records = None
        self.rate_limits = None

    def _try_connect(self) -> None:
        provider = (settings.app_store_provider or "mongo").strip().lower()
        if provider == "memory":
            if self._requires_durable_storage():
                raise RuntimeError(
                    "MongoDB is required for control-plane storage when ENVIRONMENT=production"
                )
            self._connected = False
            self._in_memory = True
            logger.info("Using in-memory control-plane storage")
            return

        try:
            from pymongo import MongoClient

            self._client = MongoClient(self._uri, serverSelectionTimeoutMS=3000)
            self._client.admin.command("ping")
            self._db = self._client[self._database]
            self.records = self._db["control_plane_records"]
            self.rate_limits = self._db["control_plane_rate_limits"]
            self._connected = True
            self._in_memory = False
            self._ensure_indexes()
            logger.info("Connected to MongoDB for control-plane storage")
        except Exception as exc:
            self._enable_in_memory_fallback(exc)

    def _ensure_indexes(self) -> None:
        if not self._connected:
            return
        try:
            from pymongo import ASCENDING

            self.records.create_index(
                [("record_type", ASCENDING), ("token_hash", ASCENDING)],
                unique=True,
            )
            self.records.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)
            self.rate_limits.create_index([("identity", ASCENDING)], unique=True)
            self.rate_limits.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)
        except Exception as exc:
            logger.warning("Failed to create control-plane indexes: %s", exc)

    def create_single_use_token(
        self,
        record_type: str,
        user_id: str,
        prefix: str,
        ttl_seconds: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        self._ensure_ready()
        token = _random_token(prefix)
        now = _utc_now()
        expires_at = now + timedelta(seconds=ttl_seconds)
        record = {
            "record_type": record_type,
            "token_hash": _hash_secret(token),
            "user_id": user_id,
            "metadata": metadata or {},
            "created_at": now,
            "expires_at": expires_at,
        }

        if self._in_memory:
            with _memory_lock:
                _memory_records[self._record_key(record_type, token)] = record
        else:
            self.records.insert_one(record)

        return {"token": token, "expires_at": expires_at}

    async def create_single_use_token_async(
        self,
        record_type: str,
        user_id: str,
        prefix: str,
        ttl_seconds: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.create_single_use_token,
            record_type,
            user_id,
            prefix,
            ttl_seconds,
            metadata,
        )

    def consume_single_use_token(self, record_type: str, token: str) -> Optional[str]:
        self._ensure_ready()
        now = _utc_now()
        if self._in_memory:
            with _memory_lock:
                record = _memory_records.pop(self._record_key(record_type, token), None)
            if not record or _as_utc(record["expires_at"]) <= now:
                return None
            return record["user_id"]

        record = self.records.find_one_and_delete(
            {
                "record_type": record_type,
                "token_hash": _hash_secret(token),
                "expires_at": {"$gt": now},
            }
        )
        return record["user_id"] if record else None

    async def consume_single_use_token_async(
        self,
        record_type: str,
        token: str,
    ) -> Optional[str]:
        return await asyncio.to_thread(self.consume_single_use_token, record_type, token)

    def create_admin_session(self, user: dict[str, Any], ttl_seconds: int) -> dict[str, Any]:
        self._ensure_ready()
        token = _random_token("adm-", 48)
        now = _utc_now()
        expires_at = now + timedelta(seconds=ttl_seconds)
        record = {
            "record_type": "admin_session",
            "token_hash": _hash_secret(token),
            "user": user,
            "created_at": now,
            "expires_at": expires_at,
        }

        if self._in_memory:
            with _memory_lock:
                _memory_records[self._record_key("admin_session", token)] = record
        else:
            self.records.insert_one(record)

        return {"token": token, "expires_at": expires_at}

    async def create_admin_session_async(
        self,
        user: dict[str, Any],
        ttl_seconds: int,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self.create_admin_session, user, ttl_seconds)

    def get_admin_session(self, token: str) -> Optional[dict[str, Any]]:
        self._ensure_ready()
        now = _utc_now()
        if self._in_memory:
            key = self._record_key("admin_session", token)
            with _memory_lock:
                record = _memory_records.get(key)
                if record and _as_utc(record["expires_at"]) <= now:
                    _memory_records.pop(key, None)
                    record = None
            return record["user"] if record else None

        record = self.records.find_one(
            {
                "record_type": "admin_session",
                "token_hash": _hash_secret(token),
                "expires_at": {"$gt": now},
            }
        )
        return record["user"] if record else None

    async def get_admin_session_async(self, token: str) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(self.get_admin_session, token)

    def delete_admin_session(self, token: str) -> None:
        self._ensure_ready()
        if self._in_memory:
            with _memory_lock:
                _memory_records.pop(self._record_key("admin_session", token), None)
            return

        self.records.delete_one(
            {"record_type": "admin_session", "token_hash": _hash_secret(token)}
        )

    async def delete_admin_session_async(self, token: str) -> None:
        await asyncio.to_thread(self.delete_admin_session, token)

    async def check_rate_limit(
        self,
        identity: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        now = time.time()
        cutoff = now - window_seconds

        if self._in_memory:
            return self._check_rate_limit_memory(identity, max_requests, now, cutoff)

        return await asyncio.to_thread(
            self._check_rate_limit_sync,
            identity,
            max_requests,
            window_seconds,
            now,
            cutoff,
        )

    def _check_rate_limit_sync(
        self,
        identity: str,
        max_requests: int,
        window_seconds: int,
        now: float,
        cutoff: float,
    ) -> tuple[bool, int]:
        self._ensure_ready()
        if self._in_memory:
            return self._check_rate_limit_memory(identity, max_requests, now, cutoff)
        return self._check_rate_limit_mongo(
            identity,
            max_requests,
            window_seconds,
            now,
            cutoff,
        )

    def _check_rate_limit_memory(
        self,
        identity: str,
        max_requests: int,
        now: float,
        cutoff: float,
    ) -> tuple[bool, int]:
        with _memory_lock:
            hits = [hit for hit in _memory_rate_limits[identity] if hit > cutoff]
            if len(hits) >= max_requests:
                _memory_rate_limits[identity] = hits
                return False, 0
            hits.append(now)
            _memory_rate_limits[identity] = hits
            return True, max(max_requests - len(hits), 0)

    def _check_rate_limit_mongo(
        self,
        identity: str,
        max_requests: int,
        window_seconds: int,
        now: float,
        cutoff: float,
    ) -> tuple[bool, int]:
        from pymongo import ReturnDocument

        record = self.rate_limits.find_one_and_update(
            {"identity": identity},
            [
                {
                    "$set": {
                        "identity": identity,
                        "hits": {
                            "$filter": {
                                "input": {"$ifNull": ["$hits", []]},
                                "as": "hit",
                                "cond": {"$gt": ["$$hit", cutoff]},
                            }
                        },
                    }
                },
                {"$set": {"allowed": {"$lt": [{"$size": "$hits"}, max_requests]}}},
                {
                    "$set": {
                        "hits": {
                            "$cond": [
                                "$allowed",
                                {"$concatArrays": ["$hits", [now]]},
                                "$hits",
                            ]
                        },
                        "expires_at": _utc_now() + timedelta(seconds=window_seconds),
                    }
                },
            ],
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

        if not record or not record.get("allowed"):
            return False, 0

        hits = record.get("hits", [])
        return True, max(max_requests - len(hits), 0)

    def _record_key(self, record_type: str, token: str) -> str:
        return f"{record_type}:{_hash_secret(token)}"

    def close(self) -> None:
        if self._client:
            self._client.close()


control_plane_store = ControlPlaneStore()
