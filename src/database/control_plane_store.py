"""Shared durable storage for auth codes, temp tokens, admin sessions, and rate limits."""

from __future__ import annotations

import hashlib
import logging
import secrets
import string
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from src.config import settings

logger = logging.getLogger("xmem.database.control_plane_store")

_in_memory_temp_tokens: Dict[str, Dict[str, Any]] = {}
_in_memory_auth_codes: Dict[str, Dict[str, Any]] = {}
_in_memory_admin_sessions: Dict[str, Dict[str, Any]] = {}
_in_memory_rate_limits: Dict[str, Dict[str, Any]] = {}


def _hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _random_token(prefix: str, length: int) -> str:
    alphabet = string.ascii_letters + string.digits
    random_part = "".join(secrets.choice(alphabet) for _ in range(length))
    return f"{prefix}{random_part}"


class ControlPlaneStore:
    """MongoDB-backed storage for cross-instance auth/session state.

    The store keeps an in-memory fallback for development and tests, but
    production mode refuses to serve from memory if MongoDB is unavailable.
    """

    def __init__(self, uri: str = None, database: str = None) -> None:
        self._uri = uri or settings.mongodb_uri
        self._database = database or settings.mongodb_database
        self._client = None
        self._db = None
        self._connected = False
        self._in_memory = False

        self.temp_tokens = None
        self.auth_codes = None
        self.admin_sessions = None
        self.rate_limits = None

        self._try_connect()

    def _try_connect(self) -> None:
        try:
            from pymongo import MongoClient

            self._client = MongoClient(self._uri, serverSelectionTimeoutMS=5000)
            self._client.admin.command("ping")
            self._db = self._client[self._database]
            self.temp_tokens = self._db["mcp_temp_tokens"]
            self.auth_codes = self._db["oauth_auth_codes"]
            self.admin_sessions = self._db["admin_sessions"]
            self.rate_limits = self._db["rate_limits"]
            self._connected = True
            self._in_memory = False
            self._ensure_indexes()
            logger.info("Connected to MongoDB for control-plane storage")
        except Exception as exc:
            self._connected = False
            self._in_memory = True
            logger.warning("MongoDB connection failed, using in-memory control-plane storage: %s", exc)

    def _ensure_indexes(self) -> None:
        if not self._connected:
            return
        try:
            from pymongo import ASCENDING

            self.temp_tokens.create_index([("token_hash", ASCENDING)], unique=True)
            self.temp_tokens.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)

            self.auth_codes.create_index([("code_hash", ASCENDING)], unique=True)
            self.auth_codes.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)

            self.admin_sessions.create_index([("session_hash", ASCENDING)], unique=True)
            self.admin_sessions.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)

            self.rate_limits.create_index([("identity", ASCENDING), ("window_key", ASCENDING)], unique=True)
            self.rate_limits.create_index([("window_expires_at", ASCENDING)], expireAfterSeconds=0)
        except Exception as exc:
            logger.warning("Failed to create control-plane indexes: %s", exc)

    def _require_durable_storage(self) -> None:
        if self._in_memory and settings.environment.lower() == "production":
            raise RuntimeError("MongoDB is required for control-plane state in production")

    def create_temp_token(self, user_id: str, ttl_minutes: int, prefix: str = "xm-temp-") -> Tuple[str, datetime]:
        token = _random_token(prefix, 32)
        expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
        token_hash = _hash_secret(token)
        doc = {
            "token_hash": token_hash,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
        }

        if self._in_memory:
            self._require_durable_storage()
            _in_memory_temp_tokens[token_hash] = doc
            return token, expires_at

        try:
            self.temp_tokens.insert_one(doc)
            return token, expires_at
        except Exception as exc:
            logger.error("Failed to create temp token: %s", exc)
            raise RuntimeError("Failed to create temp token") from exc

    def consume_temp_token(self, token: str) -> Optional[str]:
        token_hash = _hash_secret(token)

        if self._in_memory:
            self._require_durable_storage()
            token_doc = _in_memory_temp_tokens.get(token_hash)
            if not token_doc:
                return None
            if datetime.utcnow() > token_doc["expires_at"]:
                _in_memory_temp_tokens.pop(token_hash, None)
                return None
            _in_memory_temp_tokens.pop(token_hash, None)
            return token_doc["user_id"]

        try:
            now = datetime.utcnow()
            token_doc = self.temp_tokens.find_one_and_delete(
                {"token_hash": token_hash, "expires_at": {"$gt": now}},
            )
            if not token_doc:
                return None
            return token_doc["user_id"]
        except Exception as exc:
            logger.error("Failed to consume temp token: %s", exc)
            raise RuntimeError("Failed to consume temp token") from exc

    def create_auth_code(self, user_id: str, ttl_minutes: int = 10) -> str:
        code = _random_token("", 32)
        code_hash = _hash_secret(code)
        expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
        doc = {
            "code_hash": code_hash,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
        }

        if self._in_memory:
            self._require_durable_storage()
            _in_memory_auth_codes[code_hash] = doc
            return code

        try:
            self.auth_codes.insert_one(doc)
            return code
        except Exception as exc:
            logger.error("Failed to create auth code: %s", exc)
            raise RuntimeError("Failed to create auth code") from exc

    def consume_auth_code(self, code: str) -> Optional[str]:
        code_hash = _hash_secret(code)

        if self._in_memory:
            self._require_durable_storage()
            code_doc = _in_memory_auth_codes.get(code_hash)
            if not code_doc:
                return None
            if datetime.utcnow() > code_doc["expires_at"]:
                _in_memory_auth_codes.pop(code_hash, None)
                return None
            _in_memory_auth_codes.pop(code_hash, None)
            return code_doc["user_id"]

        try:
            now = datetime.utcnow()
            code_doc = self.auth_codes.find_one_and_delete(
                {"code_hash": code_hash, "expires_at": {"$gt": now}},
            )
            if not code_doc:
                return None
            return code_doc["user_id"]
        except Exception as exc:
            logger.error("Failed to consume auth code: %s", exc)
            raise RuntimeError("Failed to consume auth code") from exc

    def create_admin_session(self, user: Dict[str, Any], ttl_hours: int = 24) -> str:
        token = _hash_secret(f"{user.get('username', 'admin')}:{secrets.token_hex(32)}:{time.time()}")
        expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
        doc = {
            "session_hash": token,
            "user": user,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
        }

        if self._in_memory:
            self._require_durable_storage()
            _in_memory_admin_sessions[token] = doc
            return token

        try:
            self.admin_sessions.insert_one(doc)
            return token
        except Exception as exc:
            logger.error("Failed to create admin session: %s", exc)
            raise RuntimeError("Failed to create admin session") from exc

    def get_admin_session(self, token: str) -> Optional[Dict[str, Any]]:
        session_hash = token

        if self._in_memory:
            self._require_durable_storage()
            session_doc = _in_memory_admin_sessions.get(session_hash)
            if not session_doc:
                return None
            if datetime.utcnow() > session_doc["expires_at"]:
                _in_memory_admin_sessions.pop(session_hash, None)
                return None
            return session_doc["user"]

        try:
            now = datetime.utcnow()
            session_doc = self.admin_sessions.find_one(
                {"session_hash": session_hash, "expires_at": {"$gt": now}},
            )
            if not session_doc:
                return None
            return session_doc["user"]
        except Exception as exc:
            logger.error("Failed to fetch admin session: %s", exc)
            raise RuntimeError("Failed to fetch admin session") from exc

    def delete_admin_session(self, token: str) -> None:
        session_hash = token

        if self._in_memory:
            self._require_durable_storage()
            _in_memory_admin_sessions.pop(session_hash, None)
            return

        try:
            self.admin_sessions.delete_one({"session_hash": session_hash})
        except Exception as exc:
            logger.error("Failed to delete admin session: %s", exc)
            raise RuntimeError("Failed to delete admin session") from exc

    def check_rate_limit(self, identity: str, max_requests: int, window_seconds: int = 60) -> tuple[bool, int]:
        window_key = int(time.time() // window_seconds)
        window_expires_at = datetime.utcnow() + timedelta(seconds=window_seconds)

        if self._in_memory:
            self._require_durable_storage()
            bucket = _in_memory_rate_limits.get(identity)
            if not bucket or bucket.get("window_key") != window_key:
                bucket = {"window_key": window_key, "count": 0}
            if bucket["count"] >= max_requests:
                return False, 0
            bucket["count"] += 1
            _in_memory_rate_limits[identity] = bucket
            return True, max_requests - bucket["count"]

        try:
            from pymongo import ReturnDocument

            doc = self.rate_limits.find_one_and_update(
                {"identity": identity, "window_key": window_key},
                {
                    "$setOnInsert": {
                        "identity": identity,
                        "window_key": window_key,
                        "count": 0,
                        "window_started_at": datetime.utcnow(),
                        "window_expires_at": window_expires_at,
                    },
                    "$inc": {"count": 1},
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            count = int(doc.get("count", 0))
            if count > max_requests:
                return False, 0
            return True, max_requests - count
        except Exception as exc:
            logger.error("Failed to check rate limit: %s", exc)
            raise RuntimeError("Failed to check rate limit") from exc


control_plane_store = ControlPlaneStore()
