from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class Classification(TypedDict):
    """A classification of a part of the user query."""
    source: Literal["code", "profile", "event"]
    query: str


class ClassificationResult(BaseModel):
    """Result of classifying a user query into agent-specific sub-questions."""
    classifications: List[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )