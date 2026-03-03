from __future__ import annotations

from typing import List, Optional

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
    events: List[EventData] = Field(
        default_factory=list,
        description="List of extracted event data",
    )

    @property
    def is_empty(self) -> bool:
        return len(self.events) == 0

    @property
    def event(self) -> Optional[EventData]:
        """Backward-compatible accessor — returns the first event or None."""
        return self.events[0] if self.events else None
