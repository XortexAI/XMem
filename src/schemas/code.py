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

import hashlib
from enum import Enum
from typing import Any, List, Optional

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
    INSTRUCTION = "instruction"
    ARCHITECTURE = "architecture"
    BEST_PRACTICE = "best_practice"
    TODO = "todo"
    TECHNICAL_DEBT = "technical_debt"


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


class SnippetType(str, Enum):
    """Category of a personal code snippet."""
    ALGORITHM = "algorithm"
    CONFIG = "config"
    PATTERN = "pattern"
    FIX = "fix"
    UTILITY = "utility"
    EXPLANATION = "explanation"


class SnippetSource(str, Enum):
    """How the snippet entered the system."""
    CHAT = "chat"
    PASTE = "paste"
    FILE_UPLOAD = "file_upload"


class SnippetRecord(BaseModel):
    """Personal code snippet saved by a user (stored in {user_id}:snippets namespace)."""
    user_id: str
    content: str                                          # NL description (embedded for search)
    code_snippet: str = ""                                # the actual code
    language: str = ""
    snippet_type: SnippetType = SnippetType.ALGORITHM
    tags: List[str] = Field(default_factory=list)
    source: SnippetSource = SnippetSource.CHAT
    source_session_id: Optional[str] = None
    created_at: str = ""
    last_accessed: str = ""
    access_count: int = 0


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
    assigned_to_name: Optional[str] = Field(
        None, description="The name or role of the person this is assigned to (e.g. 'intern', 'john', etc.)"
    )


class CodeAnnotationResult(BaseModel):
    """Result of the code agent's annotation extraction."""
    annotations: List[ExtractedAnnotation] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.annotations) == 0


# ---------------------------------------------------------------------------
# Single-user snippet extraction models
# ---------------------------------------------------------------------------

class ExtractedSnippet(BaseModel):
    """A code snippet the snippet agent extracts from a user conversation."""
    content: str = Field(
        ..., description="Natural language description of what the code does"
    )
    code_snippet: Optional[str] = Field(
        default="", description="The actual code (if present in the conversation)"
    )
    language: Optional[str] = Field(
        default="", description="Programming language (python, cpp, java, etc.)"
    )
    snippet_type: SnippetType = Field(
        default=SnippetType.ALGORITHM,
        description="Category: algorithm, config, pattern, fix, utility, explanation",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Relevant tags for retrieval, e.g. ['dsa', 'binary-search', 'interview']",
    )


class SnippetExtractionResult(BaseModel):
    """Result of the snippet agent's extraction."""
    snippets: List[ExtractedSnippet] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.snippets) == 0


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


# ---------------------------------------------------------------------------
# Pinecone identity helpers for code/snippet memory
# ---------------------------------------------------------------------------

def normalize_code_text(value: str | None) -> str:
    """Normalize code only for stable identity keys, not for display/storage."""
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line.rstrip() for line in text.split("\n"))


def normalize_lookup_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def stable_hash(*parts: Any) -> str:
    joined = "\x1f".join(normalize_lookup_text(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def snippet_fields_from_storage_content(content: str) -> dict[str, str]:
    """Parse the pipe-delimited snippet string emitted by the ingest pipeline."""
    parts = [p.strip() for p in content.split(" | ")]
    if len(parts) >= 5:
        return {
            "content": parts[0],
            "code_snippet": parts[1].replace("\\n", "\n"),
            "language": parts[2],
            "snippet_type": parts[3] or SnippetType.ALGORITHM.value,
            "tags": parts[4],
        }
    if len(parts) >= 3:
        return {
            "content": parts[0],
            "code_snippet": parts[1].replace("\\n", "\n"),
            "language": parts[2],
            "snippet_type": SnippetType.ALGORITHM.value,
            "tags": "",
        }
    return {
        "content": content,
        "code_snippet": "",
        "language": "",
        "snippet_type": SnippetType.ALGORITHM.value,
        "tags": "",
    }


def snippet_identity_hash(fields: dict[str, Any]) -> str:
    code = normalize_code_text(fields.get("code_snippet"))
    content = normalize_lookup_text(fields.get("content"))
    language = normalize_lookup_text(fields.get("language"))
    identity_body = code or content
    return stable_hash(language, identity_body)


def snippet_search_text(fields: dict[str, Any]) -> str:
    """Return the text embedded for semantic snippet search."""
    content = str(fields.get("content") or "").strip()
    language = str(fields.get("language") or "").strip()
    tags = fields.get("tags") or ""
    tags_text = ", ".join(tags) if isinstance(tags, list) else str(tags)
    parts = [content]
    if language:
        parts.append(f"language: {language}")
    if tags_text:
        parts.append(f"tags: {tags_text}")
    return "\n".join(part for part in parts if part)


def snippet_pinecone_metadata(user_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    tags = fields.get("tags") or ""
    if isinstance(tags, list):
        tags = ",".join(str(tag).strip() for tag in tags if str(tag).strip())
    return {
        "user_id": user_id,
        "domain": "snippet",
        "snippet_hash": snippet_identity_hash(fields),
        "code_snippet": str(fields.get("code_snippet") or ""),
        "language": normalize_lookup_text(fields.get("language")),
        "snippet_type": str(fields.get("snippet_type") or SnippetType.ALGORITHM.value),
        "tags": str(tags),
        "source": str(fields.get("source") or SnippetSource.CHAT.value),
    }


def code_annotation_fields_from_storage_content(content: str) -> dict[str, str]:
    """Parse the pipe-delimited code annotation string emitted by ingest."""
    parts = [p.strip() for p in content.split("|")]
    if len(parts) >= 6:
        return {
            "annotation_type": parts[0] or AnnotationType.EXPLANATION.value,
            "target_symbol": parts[1],
            "target_file": parts[2],
            "repo": parts[3],
            "severity": parts[4],
            "content": " | ".join(parts[5:]).strip(),
        }
    if len(parts) >= 2:
        return {
            "annotation_type": parts[0] or AnnotationType.EXPLANATION.value,
            "target_symbol": "",
            "target_file": "",
            "repo": "",
            "severity": "",
            "content": " | ".join(parts[1:]).strip(),
        }
    return {
        "annotation_type": AnnotationType.EXPLANATION.value,
        "target_symbol": "",
        "target_file": "",
        "repo": "",
        "severity": "",
        "content": content,
    }


def code_annotation_identity_key(fields: dict[str, Any]) -> str:
    target = fields.get("target_symbol") or fields.get("target_file") or ""
    return "|".join([
        normalize_lookup_text(fields.get("repo")),
        normalize_lookup_text(target),
        normalize_lookup_text(fields.get("annotation_type")),
    ])


def code_annotation_content_hash(fields: dict[str, Any]) -> str:
    return stable_hash(
        code_annotation_identity_key(fields),
        fields.get("severity"),
        fields.get("content"),
    )


def code_annotation_pinecone_metadata(
    user_id: str, fields: dict[str, Any],
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "domain": "code",
        "annotation_key": code_annotation_identity_key(fields),
        "annotation_hash": code_annotation_content_hash(fields),
        "annotation_type": str(fields.get("annotation_type") or ""),
        "target_symbol": str(fields.get("target_symbol") or ""),
        "target_file": str(fields.get("target_file") or ""),
        "repo": str(fields.get("repo") or ""),
        "severity": str(fields.get("severity") or ""),
    }
