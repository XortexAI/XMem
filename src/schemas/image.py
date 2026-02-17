from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class ImageAnalysis(BaseModel):
    """A single extracted fact or observation from an image."""
    category: str = Field(..., description="Category of the observation (e.g. object, text, scene, person, document)")
    description: str = Field(..., description="Concise description of what was observed")
    confidence: Optional[str] = Field(default=None, description="Confidence qualifier: high, medium, low")


class ImageResult(BaseModel):
    """Result of analysing an image through the ImageAgent."""
    description: str = Field(
        default="",
        description="Natural-language summary of the image contents",
    )
    observations: List[ImageAnalysis] = Field(
        default_factory=list,
        description="Structured list of observations extracted from the image",
    )

    @property
    def is_empty(self) -> bool:
        return not self.description.strip() and len(self.observations) == 0
