"""Internal schemas for enterprise chat orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EnterpriseProjectContext:
    project_id: str
    org_id: str
    repo: str


@dataclass(frozen=True)
class EnterpriseUserContext:
    user_id: str
    username: str
    email: Optional[str] = None
    role: str = "member"
    can_create_annotations: bool = False


@dataclass(frozen=True)
class EnterpriseChatRequest:
    query: str
    file_path: Optional[str] = None
    symbol_name: Optional[str] = None
    top_k: int = 10


@dataclass(frozen=True)
class EnterpriseChatContext:
    project: EnterpriseProjectContext
    user: EnterpriseUserContext
    request: EnterpriseChatRequest


@dataclass
class EnterpriseMemoryContext:
    answer: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class EnterpriseIngestResult:
    success: bool = False
    error: Optional[str] = None
