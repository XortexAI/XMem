"""
Shared FastAPI dependencies — authentication, rate limiting, pipeline access.

All security-critical logic lives here so route handlers stay thin.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from collections import defaultdict
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.config import settings
from src.database.api_key_store import APIKeyStore
from src.database.user_store import UserStore
from src.pipelines.ingest import IngestPipeline
from src.pipelines.retrieval import RetrievalPipeline

logger = logging.getLogger("xmem.api.deps")

# Initialize stores
_user_store = UserStore()
_api_key_store = APIKeyStore()

# ═══════════════════════════════════════════════════════════════════════════
# Pipeline singletons (initialised at app startup via lifespan)
# ═══════════════════════════════════════════════════════════════════════════

_ingest_pipeline: Optional[IngestPipeline] = None
_retrieval_pipeline: Optional[RetrievalPipeline] = None
_code_pipelines: dict[str, "CodeRetrievalPipeline"] = {}  # keyed by "org_id:repo"
_pipelines_ready = asyncio.Event()
_init_error: Optional[str] = None
_startup_time: float = 0.0


def set_pipelines(
    ingest: IngestPipeline,
    retrieval: RetrievalPipeline,
) -> None:
    global _ingest_pipeline, _retrieval_pipeline
    _ingest_pipeline = ingest
    _retrieval_pipeline = retrieval


def mark_ready() -> None:
    _pipelines_ready.set()


def mark_failed(error: str) -> None:
    global _init_error
    _init_error = error
    _pipelines_ready.set()


def set_startup_time(t: float) -> None:
    global _startup_time
    _startup_time = t


def get_startup_time() -> float:
    return _startup_time


async def require_ready() -> None:
    """Block until pipelines finish initialising; raise 503 on failure."""
    if not _pipelines_ready.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipelines are still loading. Retry shortly.",
        )
    if _init_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pipeline initialisation failed: {_init_error}",
        )


def get_ingest_pipeline() -> IngestPipeline:
    if _ingest_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingest pipeline not available.",
        )
    return _ingest_pipeline


def get_retrieval_pipeline() -> RetrievalPipeline:
    if _retrieval_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval pipeline not available.",
        )
    return _retrieval_pipeline


def get_code_pipeline(org_id: str, repo: str = "") -> "CodeRetrievalPipeline":
    """Lazily create and cache a CodeRetrievalPipeline per org+repo."""
    from src.pipelines.code_retrieval import CodeRetrievalPipeline

    cache_key = f"{org_id}:{repo}"
    if cache_key not in _code_pipelines:
        repos = [repo] if repo else []
        _code_pipelines[cache_key] = CodeRetrievalPipeline(org_id=org_id, repos=repos)
        logger.info("Created CodeRetrievalPipeline for %s", cache_key)
    return _code_pipelines[cache_key]


def get_init_error() -> Optional[str]:
    return _init_error


def is_ready() -> bool:
    return _pipelines_ready.is_set() and _init_error is None


# ═══════════════════════════════════════════════════════════════════════════
# Bearer-token authentication
# ═══════════════════════════════════════════════════════════════════════════

_bearer_scheme = HTTPBearer(auto_error=False)


def _constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


async def require_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    """Validate Bearer token against configured API keys, MongoDB API keys, or JWT.

    Checks:
    1. Static API keys from settings (for backward compatibility)
    2. User-generated API keys from MongoDB
    3. JWT access tokens

    Returns the user dictionary.
    """
    if credentials is None:
        logger.warning("Missing Authorization header from %s", request.client.host)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide a Bearer token in the Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    user = None

    # 1. Check if it's a JWT access token
    if not token.startswith("xmem_"):
        user = await get_current_user(credentials)
        if user:
            request.state.user = user
            return user

    # 2. Check MongoDB for user-generated API keys
    key_doc = _api_key_store.validate_api_key(token)
    if key_doc:
        user_id = key_doc.get("user_id")
        if user_id:
            user = _user_store.get_user_by_id(user_id)
            if user:
                user["id"] = str(user.pop("_id"))
                request.state.user = user
                return user

    # 3. Check static keys first (backward compatibility)
    configured_keys = settings.api_keys
    for key in configured_keys:
        if _constant_time_compare(token, key):
            # Return a dummy user for static keys
            dummy_user = {"id": hashlib.sha256(token.encode()).hexdigest()[:16], "name": "Static Key User", "email": "static@xmem.ai"}
            request.state.user = dummy_user
            return dummy_user

    logger.warning("Invalid API key attempt from %s", request.client.host)
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key or token.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# JWT Authentication (for user sessions)
# ═══════════════════════════════════════════════════════════════════════════

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[dict]:
    """Extract and validate JWT token from Authorization header.

    Returns the user dictionary if token is valid, None otherwise.
    This is used as a dependency for routes that require authentication.
    """
    if credentials is None:
        return None

    token = credentials.credentials

    # Check if it's a JWT token (starts with 'ey' for standard JWT)
    # or our user-generated API key (starts with 'xmem_')
    if token.startswith("xmem_"):
        # This is an API key, not a JWT - skip JWT validation
        return None

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )

        user_id: str = payload.get("sub")
        if user_id is None:
            return None

        # Verify token type is 'access'
        token_type = payload.get("type")
        if token_type != "access":
            return None

        # Get fresh user data from database
        user = _user_store.get_user_by_id(user_id)
        if not user:
            return None

        # Create a copy to avoid mutating the in-memory cache
        user_copy = dict(user)
        # Convert ObjectId to string for JSON serialization
        if "_id" in user_copy:
            user_copy["id"] = str(user_copy.pop("_id"))
        elif "id" not in user_copy:
            user_copy["id"] = user_id

        return user_copy

    except JWTError:
        return None
    except Exception as e:
        logger.error(f"Error validating JWT: {e}")
        return None


async def require_user(current_user: Optional[dict] = Depends(get_current_user)) -> dict:
    """Dependency that requires a valid JWT token.

    Raises HTTPException if user is not authenticated.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


# ═══════════════════════════════════════════════════════════════════════════
# Sliding-window rate limiter (in-process, per-key)
# ═══════════════════════════════════════════════════════════════════════════

class _SlidingWindowRateLimiter:
    """Thread-safe sliding-window counter keyed by API identity."""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> tuple[bool, int]:
        """Return (allowed, remaining) for *key*."""
        now = time.monotonic()
        cutoff = now - self.window

        async with self._lock:
            timestamps = self._hits[key]
            self._hits[key] = [t for t in timestamps if t > cutoff]

            if len(self._hits[key]) >= self.max_requests:
                return False, 0

            self._hits[key].append(now)
            remaining = self.max_requests - len(self._hits[key])
            return True, remaining


_rate_limiter = _SlidingWindowRateLimiter(
    max_requests=settings.rate_limit,
    window_seconds=60,
)


async def enforce_rate_limit(
    request: Request,
    user: dict = Depends(require_api_key),
) -> dict:
    """Raise 429 if the caller has exceeded their per-minute quota."""
    identity = user.get("id", "anonymous")
    allowed, remaining = await _rate_limiter.check(identity)

    request.state.rate_limit_remaining = remaining

    if not allowed:
        logger.warning(
            "Rate limit exceeded for %s (%s)",
            identity,
            request.client.host,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": str(settings.rate_limit),
                "X-RateLimit-Remaining": "0",
            },
        )

    return user
