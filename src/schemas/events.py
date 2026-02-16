from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class EventData(BaseModel):
    date: str = Field(..., description="Date in MM-DD format (e.g. 03-15)")
    event_name: Optional[str] = Field(None, description="Short event name (2-5 words)")
    year: Optional[int] = Field(None, description="Year if mentioned or inferable")
    desc: Optional[str] = Field(None, description="Brief event description")
    time: Optional[str] = Field(None, description="Time if mentioned (e.g. 2:30 PM)")
    date_expression: Optional[str] = Field(
        None, description="Original date expression from the input"
    )


class EventResult(BaseModel):
    event: Optional[EventData] = Field(
        default=None,
        description="Extracted event data, or None if no event found",
    )

    @property
    def is_empty(self) -> bool:
        return self.event is None
