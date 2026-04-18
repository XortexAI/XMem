"""User management store for MongoDB."""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from src.config import settings

logger = logging.getLogger("xmem.database.user_store")

# In-memory fallback for when MongoDB is unavailable
_in_memory_users: Dict[str, Dict[str, Any]] = {}


class UserStore:
    """MongoDB-backed storage for user management with in-memory fallback."""

    def __init__(
        self,
        uri: str = None,
        database: str = None,
    ) -> None:
        self._uri = uri or settings.mongodb_uri
        self._database = database or settings.mongodb_database
        self._client = None
        self._db = None
        self.users = None
        self._connected = False
        self._in_memory = False

        # Try to connect, but don't fail if MongoDB is unavailable
        self._try_connect()

    def _try_connect(self) -> None:
        """Attempt to connect to MongoDB, fall back to in-memory if unavailable."""
        try:
            from pymongo import MongoClient, ASCENDING
            from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

            self._client = MongoClient(self._uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self._client.admin.command('ping')
            self._db = self._client[self._database]
            self.users = self._db["users"]
            self._connected = True
            self._in_memory = False
            logger.info("Connected to MongoDB for user storage")
            self._ensure_indexes()
        except Exception as e:
            logger.warning(f"MongoDB connection failed, using in-memory storage: {e}")
            self._connected = False
            self._in_memory = True
            self.users = None

    def _ensure_indexes(self) -> None:
        """Create necessary indexes for the users collection."""
        if not self._connected:
            return
        try:
            from pymongo import ASCENDING
            # Unique index on google_id
            self.users.create_index([("google_id", ASCENDING)], unique=True)
            # Unique index on email
            self.users.create_index([("email", ASCENDING)], unique=True)
            # Unique index on username (sparse allows null/duplicate values)
            self.users.create_index([("username", ASCENDING)], unique=True, sparse=True)
            # Index on created_at for sorting
            self.users.create_index([("created_at", ASCENDING)])
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    def get_or_create_user(
        self,
        google_id: str,
        email: str,
        name: str,
        picture: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get existing user or create a new one from Google OAuth data."""
        now = datetime.utcnow()

        if self._in_memory:
            # In-memory fallback
            if google_id in _in_memory_users:
                _in_memory_users[google_id]["last_login"] = now
                logger.info(f"User logged in (memory): {email}")
                return _in_memory_users[google_id]

            user_doc = {
                "_id": google_id,  # Use google_id as _id for simplicity
                "google_id": google_id,
                "email": email,
                "name": name,
                "picture": picture,
                "created_at": now,
                "last_login": now,
            }
            _in_memory_users[google_id] = user_doc
            logger.info(f"New user created (memory): {email}")
            return user_doc

        # MongoDB path
        try:
            from pymongo.errors import DuplicateKeyError

            existing = self.users.find_one({"google_id": google_id})

            if existing:
                self.users.update_one(
                    {"_id": existing["_id"]},
                    {"$set": {"last_login": now}}
                )
                existing["last_login"] = now
                logger.info(f"User logged in: {email}")
                return existing

            user_doc = {
                "google_id": google_id,
                "email": email,
                "name": name,
                "picture": picture,
                "created_at": now,
                "last_login": now,
            }

            result = self.users.insert_one(user_doc)
            user_doc["_id"] = result.inserted_id
            logger.info(f"New user created: {email}")
            return user_doc
        except Exception as e:
            logger.error(f"Database error in get_or_create_user: {e}")
            # Fall back to in-memory
            self._in_memory = True
            return self.get_or_create_user(google_id, email, name, picture)

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by MongoDB ID."""
        if self._in_memory:
            for user in _in_memory_users.values():
                if str(user.get("_id")) == user_id:
                    return user
            return None

        try:
            from bson import ObjectId
            return self.users.find_one({"_id": ObjectId(user_id)})
        except Exception:
            return None

    def get_user_by_google_id(self, google_id: str) -> Optional[Dict[str, Any]]:
        """Get user by Google ID."""
        if self._in_memory:
            return _in_memory_users.get(google_id)

        try:
            return self.users.find_one({"google_id": google_id})
        except Exception:
            return None

    def update_user(
        self,
        user_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update user fields."""
        if self._in_memory:
            for user in _in_memory_users.values():
                if str(user.get("_id")) == user_id:
                    user.update(updates)
                    return True
            return False

        try:
            from bson import ObjectId
            result = self.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            return False

    def set_username(self, user_id: str, username: str) -> bool:
        """Set a unique username for a user.

        Args:
            user_id: MongoDB ObjectId as string
            username: Desired username (must be unique)

        Returns:
            True if username was set successfully
        """
        # Validate username format (alphanumeric, underscore, hyphen, 3-30 chars)
        import re
        if not re.match(r'^[a-zA-Z0-9_-]{3,30}$', username):
            logger.warning(f"Invalid username format: {username}")
            return False

        if self._in_memory:
            # Check if username is taken
            for user in _in_memory_users.values():
                if user.get("username") == username and str(user.get("_id")) != user_id:
                    return False
            # Set username
            for user in _in_memory_users.values():
                if str(user.get("_id")) == user_id:
                    user["username"] = username
                    return True
            return False

        try:
            from bson import ObjectId
            from pymongo.errors import DuplicateKeyError

            result = self.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"username": username}}
            )
            if result.modified_count > 0:
                logger.info(f"Username set for user {user_id}: {username}")
                return True
            return False
        except DuplicateKeyError:
            logger.warning(f"Username already taken: {username}")
            return False
        except Exception as e:
            logger.error(f"Failed to set username for user {user_id}: {e}")
            return False

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username.

        Args:
            username: Unique username

        Returns:
            User document or None if not found
        """
        if self._in_memory:
            for user in _in_memory_users.values():
                if user.get("username") == username:
                    return user
            return None

        try:
            return self.users.find_one({"username": username})
        except Exception:
            return None

    def is_username_available(self, username: str) -> bool:
        """Check if a username is available.

        Args:
            username: Desired username

        Returns:
            True if username is available
        """
        # Validate username format
        import re
        if not re.match(r'^[a-zA-Z0-9_-]{3,30}$', username):
            return False

        if self._in_memory:
            for user in _in_memory_users.values():
                if user.get("username") == username:
                    return False
            return True

        try:
            existing = self.users.find_one({"username": username})
            return existing is None
        except Exception:
            return True  # Assume available if DB error

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
