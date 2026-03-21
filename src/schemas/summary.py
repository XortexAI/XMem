from __future__ import annotations
from pydantic import BaseModel, Field


class SummaryResult(BaseModel):
    summary: str = Field(
        default="",
        description="Bullet-point summary of memorable user facts extracted from the conversation",
    )

    @property
    def is_empty(self) -> bool:
        return not self.summary.strip()
