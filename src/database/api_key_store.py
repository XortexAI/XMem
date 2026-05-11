"""API Key management store for MongoDB."""

import hashlib
import logging
import secrets
import string
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.config import settings

logger = logging.getLogger("xmem.database.api_key_store")

# API key prefix for identification
API_KEY_PREFIX = "xmem_"
API_KEY_LENGTH = 48  # Total length including prefix

# In-memory fallback
_in_memory_api_keys: Dict[str, Dict[str, Any]] = {}


class APIKeyStore:
    """MongoDB-backed storage for API key management with in-memory fallback."""

    def __init__(
        self,
        uri: str = None,
        database: str = None,
    ) -> None:
        self._uri = uri or settings.mongodb_uri
        self._database = database or settings.mongodb_database
        self._client = None
        self._db = None
        self.api_keys = None
        self._connected = False
        self._in_memory = False

        # Try to connect
        self._try_connect()

    def _requires_durable_storage(self) -> bool:
        """Return True when in-memory API key fallback must not be used."""
        return settings.environment.lower() in {"production", "prod"}

    def _enable_in_memory_fallback(self, error: Exception) -> None:
        """Switch to in-memory storage unless the environment forbids it."""
        message = f"MongoDB connection failed for API key storage: {error}"
        if self._requires_durable_storage():
            logger.error("%s; refusing in-memory fallback in production", message)
            raise RuntimeError(
                "MongoDB is required for API key storage when ENVIRONMENT=production"
            ) from error

        logger.warning("%s; using in-memory storage", message)
        self._connected = False
        self._in_memory = True
        self.api_keys = None

    def _try_connect(self) -> None:
        """Attempt to connect to MongoDB, fall back to in-memory if unavailable."""
        try:
            from pymongo import MongoClient

            self._client = MongoClient(self._uri, serverSelectionTimeoutMS=5000)
            self._client.admin.command('ping')
            self._db = self._client[self._database]
            self.api_keys = self._db["api_keys"]
            self._connected = True
            self._in_memory = False
            logger.info("Connected to MongoDB for API key storage")
            self._ensure_indexes()
        except Exception as e:
            self._enable_in_memory_fallback(e)

    def _ensure_indexes(self) -> None:
        """Create necessary indexes for the api_keys collection."""
        if not self._connected:
            return
        try:
            from pymongo import ASCENDING
            self.api_keys.create_index([("user_id", ASCENDING)])
            self.api_keys.create_index([("key_hash", ASCENDING)], unique=True)
            self.api_keys.create_index([("is_active", ASCENDING)])
            self.api_keys.create_index([("expires_at", ASCENDING)])
            self.api_keys.create_index([("org_id", ASCENDING), ("project_id", ASCENDING)])
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    def _generate_api_key(self) -> str:
        """Generate a new secure API key."""
        alphabet = string.ascii_letters + string.digits
        random_part = "".join(secrets.choice(alphabet) for _ in range(API_KEY_LENGTH - len(API_KEY_PREFIX)))
        return f"{API_KEY_PREFIX}{random_part}"

    def _hash_key(self, key: str) -> str:
        """Create SHA-256 hash of the API key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def _normalize_scopes(self, scopes: Optional[List[str]]) -> List[str]:
        """Return a stable non-empty scope list for storage."""
        normalized = sorted({scope.strip() for scope in (scopes or ["*"]) if scope and scope.strip()})
        return normalized or ["*"]

    def _is_expired(self, key_doc: Dict[str, Any]) -> bool:
        expires_at = key_doc.get("expires_at")
        return bool(expires_at and datetime.utcnow() > expires_at)

    def _scope_allowed(self, key_doc: Dict[str, Any], required_scope: Optional[str]) -> bool:
        if not required_scope:
            return True
        scopes = key_doc.get("scopes") or ["*"]
        return "*" in scopes or required_scope in scopes

    def _binding_allowed(
        self,
        key_doc: Dict[str, Any],
        org_id: Optional[str],
        project_id: Optional[str],
    ) -> bool:
        bound_org = key_doc.get("org_id")
        bound_project = key_doc.get("project_id")
        if org_id is not None and bound_org not in (None, org_id):
            return False
        if project_id is not None and bound_project not in (None, project_id):
            return False
        return True

    def _deactivate_expired_key(self, key_doc: Dict[str, Any]) -> None:
        key_doc["is_active"] = False
        if self._in_memory:
            return
        try:
            self.api_keys.update_one(
                {"_id": key_doc["_id"]},
                {"$set": {"is_active": False}},
            )
        except Exception as e:
            logger.warning(f"Failed to deactivate expired API key: {e}")

    def create_api_key(
        self,
        user_id: str,
        name: str = "Default",
        scopes: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new API key for a user."""
        key = self._generate_api_key()
        key_hash = self._hash_key(key)
        key_prefix = key[:8]
        now = datetime.utcnow()
        normalized_scopes = self._normalize_scopes(scopes)

        if self._in_memory:
            key_id = f"mem_{len(_in_memory_api_keys)}"
            key_doc = {
                "_id": key_id,
                "user_id": user_id,
                "key_hash": key_hash,
                "key_prefix": key_prefix,
                "name": name,
                "scopes": normalized_scopes,
                "expires_at": expires_at,
                "org_id": org_id,
                "project_id": project_id,
                "created_at": now,
                "last_used": None,
                "is_active": True,
            }
            _in_memory_api_keys[key_id] = key_doc
            logger.info(f"Created new API key (memory) for user {user_id}")
            return {
                "key": key,
                "key_id": key_id,
                "name": name,
                "scopes": normalized_scopes,
                "expires_at": expires_at,
                "org_id": org_id,
                "project_id": project_id,
                "created_at": now,
            }

        try:
            key_doc = {
                "user_id": user_id,
                "key_hash": key_hash,
                "key_prefix": key_prefix,
                "name": name,
                "scopes": normalized_scopes,
                "expires_at": expires_at,
                "org_id": org_id,
                "project_id": project_id,
                "created_at": now,
                "last_used": None,
                "is_active": True,
            }
            result = self.api_keys.insert_one(key_doc)
            logger.info(f"Created new API key for user {user_id}")
            return {
                "key": key,
                "key_id": str(result.inserted_id),
                "name": name,
                "scopes": normalized_scopes,
                "expires_at": expires_at,
                "org_id": org_id,
                "project_id": project_id,
                "created_at": now,
            }
        except Exception as e:
            logger.error(f"Database error creating API key: {e}")
            self._enable_in_memory_fallback(e)
            return self.create_api_key(
                user_id=user_id,
                name=name,
                scopes=scopes,
                expires_at=expires_at,
                org_id=org_id,
                project_id=project_id,
            )

    def get_user_api_keys(self, user_id: str, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """Get all API keys for a user."""
        if self._in_memory:
            keys = [
                {**k, "id": str(k["_id"])}
                for k in _in_memory_api_keys.values()
                if k["user_id"] == user_id and (include_inactive or k.get("is_active", True))
            ]
            for key in keys:
                key.pop("key_hash", None)
            return keys

        try:
            from pymongo import ASCENDING
            query = {"user_id": user_id}
            if not include_inactive:
                query["is_active"] = True

            keys = list(self.api_keys.find(query).sort("created_at", ASCENDING))
            for key in keys:
                key.pop("key_hash", None)
                key["id"] = str(key.pop("_id"))
            return keys
        except Exception as e:
            logger.error(f"Database error getting API keys: {e}")
            return []

    def validate_api_key(
        self,
        key: str,
        required_scope: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Validate an API key and return associated user info."""
        key_hash = self._hash_key(key)

        if self._in_memory:
            for key_doc in _in_memory_api_keys.values():
                if key_doc.get("key_hash") == key_hash and key_doc.get("is_active", True):
                    if self._is_expired(key_doc):
                        self._deactivate_expired_key(key_doc)
                        return None
                    if not self._scope_allowed(key_doc, required_scope):
                        return None
                    if not self._binding_allowed(key_doc, org_id, project_id):
                        return None
                    key_doc["last_used"] = datetime.utcnow()
                    result = {**key_doc, "id": str(key_doc["_id"])}
                    result.pop("key_hash", None)
                    return result
            return None

        try:
            key_doc = self.api_keys.find_one({
                "key_hash": key_hash,
                "is_active": True,
            })
            if key_doc:
                if self._is_expired(key_doc):
                    self._deactivate_expired_key(key_doc)
                    return None
                if not self._scope_allowed(key_doc, required_scope):
                    return None
                if not self._binding_allowed(key_doc, org_id, project_id):
                    return None
                now = datetime.utcnow()
                self.api_keys.update_one(
                    {"_id": key_doc["_id"]},
                    {"$set": {"last_used": now}}
                )
                key_doc["last_used"] = now
                key_doc["id"] = str(key_doc.pop("_id"))
                key_doc.pop("key_hash", None)
            return key_doc
        except Exception as e:
            logger.error(f"Database error validating API key: {e}")
            return None

    def revoke_api_key(self, user_id: str, key_id: str) -> bool:
        """Revoke (deactivate) an API key."""
        if self._in_memory:
            if key_id in _in_memory_api_keys:
                if _in_memory_api_keys[key_id].get("user_id") == user_id:
                    _in_memory_api_keys[key_id]["is_active"] = False
                    return True
            return False

        try:
            from bson import ObjectId
            result = self.api_keys.update_one(
                {"_id": ObjectId(key_id), "user_id": user_id},
                {"$set": {"is_active": False}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to revoke API key {key_id}: {e}")
            return False

    def update_api_key_name(
        self,
        user_id: str,
        key_id: str,
        new_name: str,
    ) -> bool:
        """Update the name of an API key."""
        if self._in_memory:
            if key_id in _in_memory_api_keys:
                if _in_memory_api_keys[key_id].get("user_id") == user_id:
                    _in_memory_api_keys[key_id]["name"] = new_name
                    return True
            return False

        try:
            from bson import ObjectId
            result = self.api_keys.update_one(
                {"_id": ObjectId(key_id), "user_id": user_id},
                {"$set": {"name": new_name}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update API key name {key_id}: {e}")
            return False

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
