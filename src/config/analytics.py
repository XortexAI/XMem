"""
Fire-and-forget analytics collector for XMem.

Events are pushed into an in-memory queue and flushed to MongoDB
by a background thread.  The ``track()`` call is synchronous, never
blocks on I/O, and never raises — production latency is unaffected.

Usage::

    from src.config.analytics import analytics
    analytics.track("api_call", {"endpoint": "/v1/memory/ingest", "user_id": "u1", "latency_ms": 123})
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("xmem.analytics")

_FLUSH_INTERVAL_SECONDS = 15
_MAX_BATCH_SIZE = 200
_MAX_QUEUE_SIZE = 10_000  # drop events rather than OOM


class AnalyticsCollector:
    """Non-blocking analytics collector backed by MongoDB."""

    def __init__(self) -> None:
        self._queue: deque[Dict[str, Any]] = deque(maxlen=_MAX_QUEUE_SIZE)
        self._lock = threading.Lock()
        self._collection = None
        self._mongo_client = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background flush thread.  Safe to call multiple times."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._flush_loop, daemon=True, name="analytics-flush")
        self._thread.start()
        atexit.register(self.stop)
        logger.info("Analytics collector started.")

    def stop(self) -> None:
        """Flush remaining events and stop the background thread."""
        self._running = False
        self._flush_batch()  # one final flush
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Analytics collector stopped.")

    def track(self, event_type: str, data: Dict[str, Any] | None = None) -> None:
        """Push an analytics event — non-blocking, never raises."""
        try:
            event = {
                "event": event_type,
                "ts": datetime.now(timezone.utc),
                **(data or {}),
            }
            self._queue.append(event)
        except Exception:
            pass  # never let analytics break the app

    # ── LLM-specific helpers ──────────────────────────────────────────

    def track_llm_call(
        self,
        *,
        provider: str,
        model: str,
        agent: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        success: bool = True,
        error: str = "",
    ) -> None:
        """Convenience: track an LLM API call with token usage."""
        self.track("llm_call", {
            "provider": provider,
            "model": model,
            "agent": agent,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens or (input_tokens + output_tokens),
            "success": success,
            "error": error,
        })

    def track_api_call(
        self,
        *,
        method: str,
        path: str,
        status: int,
        latency_ms: float,
        user_id: str = "",
        request_id: str = "",
    ) -> None:
        """Convenience: track an HTTP API call."""
        self.track("api_call", {
            "method": method,
            "path": path,
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "user_id": user_id,
            "request_id": request_id,
        })

    # ── Background flush ──────────────────────────────────────────────

    def _flush_loop(self) -> None:
        """Background loop: flush buffered events every N seconds."""
        while self._running:
            time.sleep(_FLUSH_INTERVAL_SECONDS)
            try:
                self._flush_batch()
            except Exception as exc:
                logger.debug("Analytics flush error: %s", exc)

    def _flush_batch(self) -> None:
        """Drain up to ``_MAX_BATCH_SIZE`` events and insert into MongoDB."""
        batch: list[Dict[str, Any]] = []
        with self._lock:
            while self._queue and len(batch) < _MAX_BATCH_SIZE:
                batch.append(self._queue.popleft())

        if not batch:
            return

        collection = self._get_collection()
        if collection is None:
            return  # silently drop — don't crash

        try:
            collection.insert_many(batch, ordered=False)
        except Exception as exc:
            logger.debug("Analytics insert_many failed: %s", exc)

    def _get_collection(self):
        """Lazy-connect to MongoDB and return the analytics collection."""
        if self._collection is not None:
            return self._collection

        try:
            from pymongo import MongoClient, ASCENDING
            from src.config import settings

            self._mongo_client = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=3000,
            )
            self._mongo_client.admin.command("ping")

            db = self._mongo_client[settings.mongodb_database]
            self._collection = db["analytics"]

            # TTL index: auto-delete events after 90 days
            self._collection.create_index("ts", expireAfterSeconds=90 * 24 * 3600)
            # Query indexes
            self._collection.create_index([("event", ASCENDING), ("ts", ASCENDING)])
            self._collection.create_index([("user_id", ASCENDING), ("ts", ASCENDING)])

            logger.info("Analytics collection connected.")
            return self._collection
        except Exception as exc:
            logger.debug("Analytics MongoDB connection failed: %s", exc)
            return None


# Module-level singleton
analytics = AnalyticsCollector()
