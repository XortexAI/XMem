from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class MemoryLifecycle(BaseModel):
    """Boundary model for forget/TTL lifecycle metadata stored on vector records.

    All fields are optional and defaulted so legacy records (no lifecycle keys)
    validate cleanly and are treated as active + retrievable.
    """
    forget: bool = False
    expires_at: Optional[datetime] = None
    # NOTE: the manual soft-forget write path (lifecycle_state="forgotten" + forgotten_at)
    # is READ by is_retrievable() but not yet WRITTEN by any endpoint — lands in PR #2.
    lifecycle_state: Literal["active", "forgotten"] = "active"
    forgotten_at: Optional[datetime] = None
    forget_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


RESERVED_LIFECYCLE_KEYS: frozenset[str] = frozenset(MemoryLifecycle.model_fields)

# Identity/routing keys that caller-supplied extra_metadata must NEVER overwrite.
# Denylist (not allowlist): main_content/subcontent are deterministically set by
# _extract_structured_metadata before the merge, and an allowlist tied to lifecycle
# would silently drop PR #2 versioning keys (version/parent_memory_id/is_current)
# that flow through the same extra_metadata channel.
PROTECTED_METADATA_KEYS: frozenset[str] = frozenset({"user_id", "domain"})
