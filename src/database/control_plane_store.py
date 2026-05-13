"""Durable control-plane state for auth, sessions, and rate limits."""

from __future__ import annotations

import hashlib
import logging
import secrets
import string
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from src.config import settings

logger = logging.getLogger("xmem.database.control_plane_store")

_memory_records: Dict[str, Dict[str, Any]] = {}
_memory_rate_limits: dict[str, list[float]] = defaultdict(list)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _production_requires_durable_storage() -> bool:
    parsed = urlparse(settings.mongodb_uri)
    host = parsed.hostname or ""
    return settings.environment.lower() == "production" and host not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


def _random_token(prefix: str, length: int = 32) -> str:
    alphabet = string.ascii_letters + string.digits
    return prefix + "".join(secrets.choice(alphabet) for _ in range(length))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class ControlPlaneStore:
    """MongoDB-backed store for short-lived control-plane state."""

    def __init__(self, uri: str = None, database: str = None) -> None:
        self._uri = uri or settings.mongodb_uri
        self._database = database or settings.mongodb_database
        self._client = None
        self._db = None
        self.records = None
        self.rate_limits = None
        self._connected = False
        self._in_memory = False
        self._try_connect()

    def _try_connect(self) -> None:
        try:
            from pymongo import MongoClient

            self._client = MongoClient(self._uri, serverSelectionTimeoutMS=5000)
            self._client.admin.command("ping")
            self._db = self._client[self._database]
            self.records = self._db["control_plane_records"]
            self.rate_limits = self._db["rate_limits"]
            self._connected = True
            self._in_memory = False
            self._ensure_indexes()
            logger.info("Connected to MongoDB for control-plane storage")
        except Exception as exc:
            if _production_requires_durable_storage():
                raise RuntimeError(
                    "Durable MongoDB storage is required in production"
                ) from exc
            logger.warning(
                "MongoDB unavailable; using in-memory control-plane storage: %s", exc
            )
            self._connected = False
            self._in_memory = True

    def _ensure_indexes(self) -> None:
        if not self._connected:
            return
        try:
            from pymongo import ASCENDING

            self.records.create_index(
                [("record_type", ASCENDING), ("token_hash", ASCENDING)], unique=True
            )
            self.records.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)
            self.rate_limits.create_index([("identity", ASCENDING)], unique=True)
            self.rate_limits.create_index(
                [("expires_at", ASCENDING)], expireAfterSeconds=0
            )
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
        token = _random_token(prefix)
        now = _now()
        expires_at = now + timedelta(seconds=ttl_seconds)
        record = {
            "record_type": record_type,
            "token_hash": _hash(token),
            "user_id": user_id,
            "metadata": metadata or {},
            "created_at": now,
            "expires_at": expires_at,
            "exchanged": False,
        }

        if self._in_memory:
            _memory_records[f"{record_type}:{record['token_hash']}"] = record
        else:
            self.records.insert_one(record)

        return {"token": token, "expires_at": expires_at}

    def consume_single_use_token(self, record_type: str, token: str) -> Optional[str]:
        key = f"{record_type}:{_hash(token)}"
        now = _now()

        if self._in_memory:
            record = _memory_records.pop(key, None)
            if not record or record["expires_at"] <= now or record.get("exchanged"):
                return None
            return record["user_id"]

        record = self.records.find_one_and_update(
            {
                "record_type": record_type,
                "token_hash": _hash(token),
                "exchanged": False,
                "expires_at": {"$gt": now},
            },
            {"$set": {"exchanged": True, "exchanged_at": now}},
        )
        return record["user_id"] if record else None

    def create_admin_session(
        self, user: dict[str, Any], ttl_seconds: int
    ) -> dict[str, Any]:
        token = _random_token("adm-", 48)
        now = _now()
        expires_at = now + timedelta(seconds=ttl_seconds)
        record = {
            "record_type": "admin_session",
            "token_hash": _hash(token),
            "user": user,
            "created_at": now,
            "expires_at": expires_at,
        }

        if self._in_memory:
            _memory_records[f"admin_session:{record['token_hash']}"] = record
        else:
            self.records.insert_one(record)

        return {"token": token, "expires_at": expires_at}

    def get_admin_session(self, token: str) -> Optional[dict[str, Any]]:
        key = f"admin_session:{_hash(token)}"
        now = _now()

        if self._in_memory:
            record = _memory_records.get(key)
            if not record or record["expires_at"] <= now:
                _memory_records.pop(key, None)
                return None
            return record["user"]

        record = self.records.find_one(
            {
                "record_type": "admin_session",
                "token_hash": _hash(token),
                "expires_at": {"$gt": now},
            }
        )
        return record["user"] if record else None

    def delete_admin_session(self, token: str) -> None:
        key = f"admin_session:{_hash(token)}"
        if self._in_memory:
            _memory_records.pop(key, None)
            return
        self.records.delete_one(
            {"record_type": "admin_session", "token_hash": _hash(token)}
        )

    async def check_rate_limit(
        self,
        identity: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        now = time.time()
        cutoff = now - window_seconds

        if self._in_memory:
            timestamps = [hit for hit in _memory_rate_limits[identity] if hit > cutoff]
            if len(timestamps) >= max_requests:
                _memory_rate_limits[identity] = timestamps
                return False, 0
            timestamps.append(now)
            _memory_rate_limits[identity] = timestamps
            return True, max_requests - len(timestamps)

        record = self.rate_limits.find_one({"identity": identity}) or {}
        hits = [hit for hit in record.get("hits", []) if hit > cutoff]
        if len(hits) >= max_requests:
            self.rate_limits.update_one(
                {"identity": identity},
                {
                    "$set": {
                        "hits": hits,
                        "expires_at": _now() + timedelta(seconds=window_seconds),
                    }
                },
                upsert=True,
            )
            return False, 0

        hits.append(now)
        self.rate_limits.update_one(
            {"identity": identity},
            {
                "$set": {
                    "hits": hits,
                    "expires_at": _now() + timedelta(seconds=window_seconds),
                }
            },
            upsert=True,
        )
        return True, max(max_requests - len(hits), 0)

    def close(self) -> None:
        if self._client:
            self._client.close()
