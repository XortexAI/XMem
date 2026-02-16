from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


class ProfileFact(BaseModel):
    topic: str = Field(..., description="High-level topic (e.g. work, education, interest)")
    sub_topic: str = Field(..., description="Specific attribute within the topic (e.g. company, degree)")
    memo: str = Field(..., description="The extracted value or fact")


class ProfileResult(BaseModel):
    facts: List[ProfileFact] = Field(
        default_factory=list,
        description="List of structured facts extracted from the user input",
    )

    @property
    def is_empty(self) -> bool:
        return len(self.facts) == 0
