"""
Request / response models for the XMem v1 API.

Every public endpoint has an explicit pair of Pydantic models so that
OpenAPI docs, input validation, and serialization are fully type-safe.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Shared envelope ────────────────────────────────────────────────────────

class StatusEnum(str, Enum):
    OK = "ok"
    ERROR = "error"


class APIResponse(BaseModel):
    """Standard wrapper returned by every endpoint."""
    status: StatusEnum = StatusEnum.OK
    request_id: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None
    elapsed_ms: Optional[float] = None


# ── Health ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    pipelines_ready: bool
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None
    error: Optional[str] = None


# ── Ingest (save memory) ──────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Store a new memory from a conversation turn."""
    user_query: str = Field(
        ..., min_length=1, max_length=10_000,
        description="The user's message to memorize",
    )
    agent_response: str = Field(
        default="", max_length=10_000,
        description="The assistant's reply (used for summary extraction)",
    )
    user_id: str = Field(
        ..., min_length=1, max_length=256, pattern=r"^[\w.\-@]+$",
        description="Unique user identifier (alphanumeric, dots, hyphens, underscores, @)",
    )
    session_datetime: str = Field(
        default="",
        description="ISO-8601 datetime context for temporal event extraction",
    )
    image_url: str = Field(
        default="", max_length=50_000,
        description="URL or base64 data-URI of an attached image",
    )
    effort_level: str = Field(
        default="low",
        description="'low' (fast, single pass) or 'high' (chunked parallel extraction)",
    )

    @field_validator("user_query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class OperationDetail(BaseModel):
    type: str
    content: str
    reason: str


class WeaverSummary(BaseModel):
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0


class DomainResult(BaseModel):
    confidence: float = 0.0
    operations: List[OperationDetail] = Field(default_factory=list)
    weaver: Optional[WeaverSummary] = None


class IngestResponse(BaseModel):
    model: str = ""
    classification: List[Any] = Field(default_factory=list)
    profile: Optional[DomainResult] = None
    temporal: Optional[DomainResult] = None
    summary: Optional[DomainResult] = None
    image: Optional[DomainResult] = None


# ── Retrieve (answer a question from memory) ──────────────────────────────

class RetrieveRequest(BaseModel):
    """Ask a question answered from stored memories."""
    query: str = Field(
        ..., min_length=1, max_length=5_000,
        description="The question to answer from memory",
    )
    user_id: str = Field(
        ..., min_length=1, max_length=256, pattern=r"^[\w.\-@]+$",
    )
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class SourceRecord(BaseModel):
    domain: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    model: str = ""
    answer: str = ""
    sources: List[SourceRecord] = Field(default_factory=list)
    confidence: float = 0.0


# ── Search (raw vector / graph search without LLM answer) ─────────────────

class SearchRequest(BaseModel):
    """Raw semantic search across memory domains."""
    query: str = Field(
        ..., min_length=1, max_length=5_000,
    )
    user_id: str = Field(
        ..., min_length=1, max_length=256, pattern=r"^[\w.\-@]+$",
    )
    domains: List[str] = Field(
        default=["profile", "temporal", "summary"],
        description="Which memory domains to search",
    )
    top_k: int = Field(default=10, ge=1, le=100)

    @field_validator("domains")
    @classmethod
    def validate_domains(cls, v: List[str]) -> List[str]:
        allowed = {"profile", "temporal", "summary"}
        for d in v:
            if d not in allowed:
                raise ValueError(f"Invalid domain '{d}'. Allowed: {allowed}")
        return v


class SearchResponse(BaseModel):
    results: List[SourceRecord] = Field(default_factory=list)
    total: int = 0


# ── Scrape (extract from shared chat links) ────────────────────────────────

class ScrapeRequest(BaseModel):
    """Request to scrape a shared AI chat link."""
    url: str = Field(
        ..., min_length=1, max_length=2000,
        description="Public share link (ChatGPT, Claude, Gemini)"
    )

class MessagePair(BaseModel):
    user_query: str
    agent_response: str

class ScrapeResponse(BaseModel):
    pairs: List[MessagePair] = Field(default_factory=list)
    error: Optional[str] = None


# ── Code retrieval (IDE mode) ─────────────────────────────────────────────

class CodeQueryRequest(BaseModel):
    """Query a codebase via the code retrieval pipeline."""
    org_id: str = Field(..., min_length=1, max_length=256)
    repo: str = Field(..., min_length=1, max_length=256)
    query: str = Field(..., min_length=1, max_length=5_000)
    user_id: str = Field(default="", max_length=256)
    top_k: int = Field(default=10, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def strip_code_query(cls, v: str) -> str:
        return v.strip()


class CodeQueryResponse(BaseModel):
    answer: str = ""
    sources: List[SourceRecord] = Field(default_factory=list)
    confidence: float = 0.0


class DirectoryNode(BaseModel):
    name: str
    type: str = "file"
    path: str = ""
    children: List["DirectoryNode"] = Field(default_factory=list)


class DirectoryTreeResponse(BaseModel):
    repo: str
    tree: DirectoryNode


class RepoListResponse(BaseModel):
    repos: List[str] = Field(default_factory=list)
