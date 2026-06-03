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
    lifecycle_state: Literal["active", "forgotten"] = "active"
    forgotten_at: Optional[datetime] = None
    forget_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


RESERVED_LIFECYCLE_KEYS: frozenset[str] = frozenset(MemoryLifecycle.model_fields)
