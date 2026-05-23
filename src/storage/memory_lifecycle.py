"""Memory lifecycle metadata helpers.

These helpers keep duplicate detection, version lineage, and soft-forget
metadata consistent across vector-store implementations.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

CONTENT_HASH_KEY = "content_hash"
PARENT_MEMORY_ID_KEY = "parent_memory_id"
VERSION_KEY = "version"
IS_CURRENT_KEY = "is_current"
FORGOTTEN_AT_KEY = "forgotten_at"
FORGET_REASON_KEY = "forget_reason"


def normalize_memory_content(content: str) -> str:
    """Normalize memory text before hashing to catch whitespace-only duplicates."""

    return re.sub(r"\s+", " ", content.strip()).casefold()


def compute_memory_hash(content: str) -> str:
    """Return a stable SHA-256 digest for normalized memory content."""

    normalized = normalize_memory_content(content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_lifecycle_metadata(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    parent_memory_id: Optional[str] = None,
    version: int = 1,
    is_current: bool = True,
) -> Dict[str, Any]:
    """Merge caller metadata with lifecycle fields without losing custom keys."""

    merged = dict(metadata or {})
    merged[CONTENT_HASH_KEY] = compute_memory_hash(content)
    merged[PARENT_MEMORY_ID_KEY] = parent_memory_id
    merged[VERSION_KEY] = version
    merged[IS_CURRENT_KEY] = is_current
    merged[FORGOTTEN_AT_KEY] = None
    merged[FORGET_REASON_KEY] = None
    return merged


def is_retrievable_memory(metadata: Optional[Dict[str, Any]]) -> bool:
    """Return False for superseded or soft-forgotten memory records."""

    meta = metadata or {}
    return meta.get(IS_CURRENT_KEY, True) is not False and not meta.get(FORGOTTEN_AT_KEY)
