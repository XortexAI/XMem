"""
Code domain schemas — Pydantic models for symbols, files, directories,
annotations, and snippets.

Maps directly to the Pinecone namespace structure:
  {org_id}:{repo}:symbols      → SymbolRecord
  {org_id}:{repo}:files        → FileRecord
  {org_id}:{repo}:directories  → DirectoryRecord
  {org_id}:annotations         → AnnotationRecord
  {user_id}:snippets           → SnippetRecord
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SymbolType(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    ENUM = "enum"


class AnnotationType(str, Enum):
    BUG_REPORT = "bug_report"
    FIX = "fix"
    FEATURE_IDEA = "feature_idea"
    EXPLANATION = "explanation"
    WARNING = "warning"


class AnnotationSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnnotationStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    OUTDATED = "outdated"
    SUPERSEDED = "superseded"


class ComplexityBucket(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Pinecone record models (what gets stored in each namespace)
# ---------------------------------------------------------------------------

class SummarySource(str, Enum):
    """Tracks how a summary was generated."""
    AST = "ast"          # deterministic: docstring or signature (Phase 1)
    LLM = "llm"          # enriched by LLM (Phase 2)
    MANUAL = "manual"    # manually written by a developer


class SymbolRecord(BaseModel):
    """A function, method, class, interface, or enum indexed in the symbols namespace."""
    org_id: str
    repo: str
    file_path: str
    symbol_name: str
    symbol_type: SymbolType
    language: str

    summary: str
    summary_source: SummarySource = SummarySource.AST
    signature: str = ""
    docstring: str = ""

    start_line: int = 0
    end_line: int = 0
    commit_sha: str = ""
    branch: str = "main"

    parent_class: Optional[str] = None
    parent_file_id: Optional[str] = None
    imports: List[str] = Field(default_factory=list)

    signature_hash: str = ""
    indexed_at: str = ""
    scanner_version: str = "1.0.0"

    is_public: bool = True
    is_entrypoint: bool = False
    complexity_bucket: ComplexityBucket = ComplexityBucket.MEDIUM
    line_count: int = 0


class FileRecord(BaseModel):
    """File-level summary indexed in the files namespace."""
    org_id: str
    repo: str
    file_path: str
    language: str

    summary: str
    summary_source: SummarySource = SummarySource.AST
    symbol_count: int = 0
    symbol_names: List[str] = Field(default_factory=list)
    total_lines: int = 0

    commit_sha: str = ""
    indexed_at: str = ""


class DirectoryRecord(BaseModel):
    """Directory-level summary indexed in the directories namespace."""
    org_id: str
    repo: str
    dir_path: str
    languages: List[str] = Field(default_factory=list)

    summary: str
    summary_source: SummarySource = SummarySource.AST
    file_count: int = 0
    total_symbols: int = 0

    indexed_at: str = ""


class AnnotationRecord(BaseModel):
    """Team knowledge annotation targeting a symbol, file, or directory."""
    org_id: str
    repo: Optional[str] = None

    target_file: Optional[str] = None
    target_symbol: Optional[str] = None
    target_symbol_id: Optional[str] = None

    content: str
    annotation_type: AnnotationType = AnnotationType.EXPLANATION
    severity: Optional[AnnotationSeverity] = None

    author_id: Optional[str] = None
    author_name: Optional[str] = None
    source_session_id: Optional[str] = None
    source_message_snippet: Optional[str] = None

    status: AnnotationStatus = AnnotationStatus.ACTIVE
    superseded_by: Optional[str] = None
    created_at: str = ""
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None


class SnippetRecord(BaseModel):
    """Personal code snippet saved by a user."""
    user_id: str
    title: str
    content: str
    language: str = ""
    tags: List[str] = Field(default_factory=list)
    repo: Optional[str] = None
    file_path: Optional[str] = None
    created_at: str = ""


# ---------------------------------------------------------------------------
# Agent output models (what the code agent extracts from conversations)
# ---------------------------------------------------------------------------

class ExtractedAnnotation(BaseModel):
    """An annotation the code agent extracts from a conversation turn."""
    target_symbol: Optional[str] = Field(
        None, description="Fully qualified symbol name, e.g. 'PaymentProcessor.process'"
    )
    target_file: Optional[str] = Field(
        None, description="File path the annotation applies to"
    )
    content: str = Field(
        ..., description="The actual insight / knowledge being annotated"
    )
    annotation_type: AnnotationType = Field(
        AnnotationType.EXPLANATION, description="Category of the annotation"
    )
    severity: Optional[AnnotationSeverity] = Field(
        None, description="Severity level (for bugs/warnings)"
    )
    repo: Optional[str] = Field(
        None, description="Repository the annotation targets"
    )


class CodeAnnotationResult(BaseModel):
    """Result of the code agent's annotation extraction."""
    annotations: List[ExtractedAnnotation] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.annotations) == 0


# ---------------------------------------------------------------------------
# Namespace helpers
# ---------------------------------------------------------------------------

def symbols_namespace(org_id: str, repo: str) -> str:
    return f"{org_id}:{repo}:symbols"


def files_namespace(org_id: str, repo: str) -> str:
    return f"{org_id}:{repo}:files"


def directories_namespace(org_id: str, repo: str) -> str:
    return f"{org_id}:{repo}:directories"


def annotations_namespace(org_id: str) -> str:
    return f"{org_id}:annotations"


def snippets_namespace(user_id: str) -> str:
    return f"{user_id}:snippets"
