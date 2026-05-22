"""
Tests for the UserIngestionCoordinator.

Validates per-user serialisation, cross-user parallelism, FIFO ordering,
lock cleanup, and exception safety.
"""

import asyncio
import time

import importlib.util
import os
import sys

import pytest

# Import the coordinator module directly from its file to avoid pulling in
# src.api.__init__ → src.api.app → src.config.Settings (requires env vars).
_coordinator_path = os.path.join(
    os.path.dirname(__file__), os.pardir, "src", "api", "ingestion_coordinator.py"
)
_spec = importlib.util.spec_from_file_location(
    "ingestion_coordinator", os.path.abspath(_coordinator_path)
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
UserIngestionCoordinator = _mod.UserIngestionCoordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _timed_task(coordinator: UserIngestionCoordinator, user_id: str, duration: float, log: list):
    """Acquire the user lock, record (start, end) timestamps, and sleep."""
    async with coordinator.acquire(user_id):
        start = time.monotonic()
        await asyncio.sleep(duration)
        end = time.monotonic()
        log.append((user_id, start, end))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sequential_for_same_user():
    """Concurrent tasks for the same user must execute one at a time (non-overlapping)."""
    coordinator = UserIngestionCoordinator()
    log: list = []

    tasks = [
        asyncio.create_task(_timed_task(coordinator, "alice", 0.05, log))
        for _ in range(5)
    ]
    await asyncio.gather(*tasks)

    assert len(log) == 5

    # Sort by start time and verify no overlaps
    log.sort(key=lambda x: x[1])
    for i in range(1, len(log)):
        prev_end = log[i - 1][2]
        curr_start = log[i][1]
        assert curr_start >= prev_end - 0.001, (
            f"Task {i} started at {curr_start:.4f} before task {i-1} ended at {prev_end:.4f}"
        )


@pytest.mark.asyncio
async def test_parallel_for_different_users():
    """Tasks for different users should run concurrently (overlapping in time)."""
    coordinator = UserIngestionCoordinator()
    log: list = []

    users = ["alice", "bob", "charlie"]
    tasks = [
        asyncio.create_task(_timed_task(coordinator, user, 0.1, log))
        for user in users
    ]
    await asyncio.gather(*tasks)

    assert len(log) == 3

    # All three should start roughly at the same time (within 20ms of each other)
    starts = sorted(entry[1] for entry in log)
    spread = starts[-1] - starts[0]
    assert spread < 0.05, (
        f"Different users should run in parallel, but start-time spread was {spread:.4f}s"
    )


@pytest.mark.asyncio
async def test_fifo_ordering():
    """Tasks for the same user should complete in submission order (FIFO)."""
    coordinator = UserIngestionCoordinator()
    completion_order: list = []

    async def _ordered_task(task_id: int):
        async with coordinator.acquire("user_fifo"):
            await asyncio.sleep(0.02)
            completion_order.append(task_id)

    # Create tasks in order 0, 1, 2, 3, 4
    tasks = []
    for i in range(5):
        tasks.append(asyncio.create_task(_ordered_task(i)))
        # Small delay to ensure submission order is deterministic
        await asyncio.sleep(0.005)

    await asyncio.gather(*tasks)

    assert completion_order == [0, 1, 2, 3, 4], (
        f"Expected FIFO order [0,1,2,3,4], got {completion_order}"
    )


@pytest.mark.asyncio
async def test_lock_cleanup():
    """After all tasks complete, internal lock dict should be empty."""
    coordinator = UserIngestionCoordinator()

    async with coordinator.acquire("cleanup_user"):
        assert coordinator.active_users == 1

    # After context exit, lock should be cleaned up
    assert coordinator.active_users == 0
    assert "cleanup_user" not in coordinator._locks
    assert "cleanup_user" not in coordinator._waiters


@pytest.mark.asyncio
async def test_exception_safety():
    """If a task raises inside the lock, the lock must still be released for subsequent tasks."""
    coordinator = UserIngestionCoordinator()
    results: list = []

    async def _failing_task():
        async with coordinator.acquire("error_user"):
            raise ValueError("deliberate test error")

    async def _succeeding_task():
        async with coordinator.acquire("error_user"):
            results.append("success")

    # First task fails
    with pytest.raises(ValueError, match="deliberate test error"):
        await _failing_task()

    # Second task should still be able to acquire the lock and succeed
    await _succeeding_task()

    assert results == ["success"]
    assert coordinator.active_users == 0


@pytest.mark.asyncio
async def test_concurrent_same_user_does_not_deadlock():
    """Many concurrent acquires for the same user must all complete without deadlock."""
    coordinator = UserIngestionCoordinator()
    counter = {"value": 0}

    async def _increment():
        async with coordinator.acquire("stress_user"):
            counter["value"] += 1
            await asyncio.sleep(0.001)

    tasks = [asyncio.create_task(_increment()) for _ in range(20)]
    await asyncio.gather(*tasks)

    assert counter["value"] == 20
    assert coordinator.active_users == 0


@pytest.mark.asyncio
async def test_mixed_users_serialization():
    """Two users interleaving: each user's tasks are serial, but different users overlap."""
    coordinator = UserIngestionCoordinator()
    log: list = []

    # 3 tasks for alice, 3 tasks for bob — all launched concurrently
    tasks = []
    for i in range(3):
        tasks.append(asyncio.create_task(_timed_task(coordinator, "alice", 0.03, log)))
        tasks.append(asyncio.create_task(_timed_task(coordinator, "bob", 0.03, log)))

    await asyncio.gather(*tasks)

    assert len(log) == 6

    # Verify per-user serialisation
    for user in ("alice", "bob"):
        user_entries = sorted([e for e in log if e[0] == user], key=lambda x: x[1])
        for i in range(1, len(user_entries)):
            prev_end = user_entries[i - 1][2]
            curr_start = user_entries[i][1]
            assert curr_start >= prev_end - 0.001, (
                f"{user} task {i} overlapped with task {i-1}"
            )
