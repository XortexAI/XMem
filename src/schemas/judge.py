from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class OperationType(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class JudgeDomain(str, Enum):
    PROFILE = "profile"
    TEMPORAL = "temporal"
    SUMMARY = "summary"


class Operation(BaseModel):
    type: OperationType
    content: str = Field(default="", description="The text to add or update to")
    embedding_id: Optional[str] = Field(
        None, description="ID of the existing vector to update/delete"
    )
    reason: str = Field(default="", description="Why this operation was chosen")


class JudgeResult(BaseModel):
    operations: List[Operation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def is_empty(self) -> bool:
        return len(self.operations) == 0

    @property
    def has_writes(self) -> bool:
        return any(op.type != OperationType.NOOP for op in self.operations)
