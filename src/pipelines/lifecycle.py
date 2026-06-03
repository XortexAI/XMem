"""
Memory lifecycle — pure, deterministic helper functions for forget/TTL.

is_retrievable() is the single retrieval-time gate. build_lifecycle_metadata()
stamps the lifecycle fields onto a metadata dict for storage.

All functions are side-effect-free so they can be tested without live services.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Optional


def is_retrievable(metadata: Optional[Mapping[str, Any]], now: datetime) -> bool:
    """Return True when a record should appear in retrieval results.

    Rules (applied in order):
    1. ``lifecycle_state == "forgotten"`` → hidden (manual soft-forget).
    2. ``forget is True`` and ``expires_at`` is present and in the past → hidden (TTL expired).
    3. Everything else (including all legacy records with no lifecycle keys) → retrievable.

    Missing keys default to the legacy-safe value so records stored before
    lifecycle was introduced are never hidden.
    """
    if not metadata:
        return True

    if metadata.get("lifecycle_state", "active") == "forgotten":
        return False

    if metadata.get("forget"):
        expires_raw = metadata.get("expires_at")
        if expires_raw:
            try:
                if isinstance(expires_raw, datetime):
                    expires_at = expires_raw
                else:
                    expires_at = datetime.fromisoformat(str(expires_raw))
                # Make both sides timezone-aware or both naive for comparison
                if expires_at.tzinfo is None and now.tzinfo is not None:
                    from datetime import timezone
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                elif expires_at.tzinfo is not None and now.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=None)
                if expires_at < now:
                    return False
            except (ValueError, TypeError):
                pass

    return True


def build_lifecycle_metadata(
    now: datetime,
    ttl_days: float,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the lifecycle metadata dict to merge onto a forget=true record.

    Called once at v2 ingestion time when the caller sets ``forget=true``.
    The result is stored as part of the vector record's metadata so the
    retrieval-time filter can enforce the TTL without any background sweeper.
    """
    from datetime import timedelta
    expires_at = now + timedelta(days=ttl_days)
    meta: Dict[str, Any] = {
        "forget": True,
        "expires_at": expires_at.isoformat(),
        "lifecycle_state": "active",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    if reason:
        meta["forget_reason"] = reason
    return meta
