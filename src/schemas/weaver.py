from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from src.schemas.judge import OperationType


class OpStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


class ExecutedOp(BaseModel):
    type: OperationType
    status: OpStatus
    content: str = ""
    embedding_id: Optional[str] = None
    new_id: Optional[str] = Field(None, description="ID assigned on ADD")
    error: Optional[str] = None


class WeaverResult(BaseModel):
    executed: List[ExecutedOp] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.executed)

    @property
    def succeeded(self) -> int:
        return sum(1 for op in self.executed if op.status == OpStatus.SUCCESS)

    @property
    def skipped(self) -> int:
        return sum(1 for op in self.executed if op.status == OpStatus.SKIPPED)

    @property
    def failed(self) -> int:
        return sum(1 for op in self.executed if op.status == OpStatus.FAILED)
