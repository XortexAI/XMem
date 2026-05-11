"""
Per-user ingestion coordinator — serialises ingestion for each user.

Guarantees that only one ingestion pipeline runs at a time for any given
``user_id``, while allowing different users to proceed in parallel.
Requests for the same user are processed in strict FIFO order.

This is the **in-memory** implementation (Option 1).  A future distributed
lock (Redis, etc.) can be swapped in by implementing the same ``acquire()``
context-manager interface.

Usage::

    from src.api.ingestion_coordinator import UserIngestionCoordinator

    coordinator = UserIngestionCoordinator()

    async with coordinator.acquire(user_id):
        result = await pipeline.run(...)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict

logger = logging.getLogger("xmem.api.ingestion_coordinator")


class UserIngestionCoordinator:
    """Per-user FIFO ingestion lock.

    Internally maintains a ``dict[str, asyncio.Lock]`` keyed by ``user_id``.
    Locks are created lazily on first access and removed once no tasks are
    waiting or holding them, preventing unbounded memory growth.

    Thread-safety note
    ------------------
    All mutations to the internal registry are protected by a single
    ``asyncio.Lock`` (the *registry lock*).  Since this code runs on the
    asyncio event loop, ``asyncio.Lock`` is sufficient — no OS-level
    threading primitives are needed.
    """

    def __init__(self) -> None:
        # Maps user_id -> (asyncio.Lock, active_count)
        # active_count tracks how many tasks are either holding or waiting
        # for the lock so we know when it's safe to clean up.
        self._locks: Dict[str, asyncio.Lock] = {}
        self._waiters: Dict[str, int] = {}
        self._registry_lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, user_id: str) -> AsyncIterator[None]:
        """Acquire the per-user ingestion lock.

        Usage::

            async with coordinator.acquire("user_123"):
                # Only one coroutine per user_id reaches here at a time.
                await do_work()

        The lock is automatically released (and cleaned up if idle) when
        the ``async with`` block exits, even if an exception is raised.
        """
        # ── Get-or-create the user lock ──────────────────────────────
        async with self._registry_lock:
            if user_id not in self._locks:
                self._locks[user_id] = asyncio.Lock()
                self._waiters[user_id] = 0
            self._waiters[user_id] += 1
            user_lock = self._locks[user_id]

        logger.debug("User %s: waiting for ingestion lock (waiters=%d)", user_id, self._waiters.get(user_id, 0))

        try:
            async with user_lock:
                logger.debug("User %s: ingestion lock acquired", user_id)
                yield
        finally:
            # ── Cleanup: remove the lock if nobody else is waiting ────
            async with self._registry_lock:
                self._waiters[user_id] -= 1
                if self._waiters[user_id] <= 0:
                    self._locks.pop(user_id, None)
                    self._waiters.pop(user_id, None)
                    logger.debug("User %s: ingestion lock cleaned up", user_id)

    @property
    def active_users(self) -> int:
        """Return the number of users with active or pending ingestion locks."""
        return len(self._locks)
