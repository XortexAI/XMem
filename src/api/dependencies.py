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

from src.config import settings
from src.pipelines.ingest import IngestPipeline
from src.pipelines.retrieval import RetrievalPipeline

logger = logging.getLogger("xmem.api.deps")

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
) -> str:
    """Validate Bearer token against the configured API key list.

    If no keys are configured (dev mode) authentication is skipped.
    Returns the (hashed) key identity for rate-limit bucketing.
    """
    configured_keys = settings.api_keys
    if not configured_keys:
        return "anonymous"

    if credentials is None:
        logger.warning("Missing Authorization header from %s", request.client.host)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide a Bearer token in the Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    for key in configured_keys:
        if _constant_time_compare(token, key):
            return hashlib.sha256(token.encode()).hexdigest()[:16]

    logger.warning("Invalid API key attempt from %s", request.client.host)
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key.",
    )


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
    identity: str = Depends(require_api_key),
) -> str:
    """Raise 429 if the caller has exceeded their per-minute quota."""
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

    return identity
