"""
Neo4j schema constants for scanner_v1.

All labels, relationship types, property keys, and vector-index names
live here so the rest of the package never hard-codes Cypher strings.

Why this file exists instead of reusing src/graph/cypher_schema.py:
  - v0 schema assumes Pinecone is doing vector search and Neo4j is a
    pure graph. v1 collapses the two, so every Symbol / File node now
    carries one or two embedding properties plus a vector index.
  - Keeping v1 constants isolated means scanner/ (v0) and scanner_v1/
    can coexist and write to disjoint parts of the same Neo4j instance
    (or separate databases) without stepping on each other.

Conventions:
  LABEL_*  — node labels
  REL_*    — relationship types
  PROP_*   — property keys
  IDX_*    — index names (vector + fulltext + constraint)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Scanner version
# ---------------------------------------------------------------------------
# Stamped onto every node written by scanner_v1 so future migrations can
# tell "this row was written by which scanner build" without touching git
# history.  Bump on any write-shape change; the enricher and retriever
# should tolerate rows with older versions.

SCANNER_VERSION = "1.0.0-alpha"


# ---------------------------------------------------------------------------
# Node labels
# ---------------------------------------------------------------------------

LABEL_REPOSITORY = "RepositoryV1"
LABEL_DIRECTORY  = "DirectoryV1"
LABEL_FILE       = "FileV1"
LABEL_SYMBOL     = "SymbolV1"
LABEL_SCAN       = "ScanV1"     # tracks scan state (replaces Mongo scan records)

# NOTE: labels are suffixed "V1" so v0 and v1 can share a Neo4j instance.
# Once v0 is retired the suffix can be dropped in a one-shot rename pass.


# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------

REL_CONTAINS_DIR  = "CONTAINS_DIR"    # Repo → Dir, Dir → Dir
REL_CONTAINS_FILE = "CONTAINS_FILE"   # Dir → File
REL_DEFINES       = "DEFINES"         # File → Symbol
REL_CALLS         = "CALLS"           # Symbol → Symbol
REL_IMPORTS       = "IMPORTS"         # File → File
REL_EXTENDS       = "EXTENDS"         # Symbol → Symbol (class inheritance)
REL_IMPLEMENTS    = "IMPLEMENTS"      # Symbol → Symbol (interface)
REL_HAS_SCAN      = "HAS_SCAN"        # Repo → Scan


# ---------------------------------------------------------------------------
# Property keys — shared
# ---------------------------------------------------------------------------

PROP_ORG_ID    = "org_id"
PROP_REPO      = "repo"
PROP_BRANCH    = "branch"
PROP_COMMIT    = "commit_sha"

# File / Symbol addressing
PROP_FILE_PATH    = "file_path"
PROP_DIR_PATH     = "dir_path"
PROP_SYMBOL_NAME  = "symbol_name"     # fully-qualified, e.g. "ClassName.method"
PROP_SYMBOL_TYPE  = "symbol_type"     # function | method | class | interface | enum
PROP_LANGUAGE     = "language"

# Content / change detection
PROP_CONTENT_HASH  = "content_hash"   # sha256 of raw text; drives incremental
PROP_SIGNATURE_HASH = "signature_hash" # sha256 of signature only (for API-level change detection)

# Raw code (stored on the node for v1; future: object storage pointer)
PROP_RAW_CODE    = "raw_code"
PROP_RAW_CONTENT = "raw_content"      # file-level

# Summaries
PROP_SUMMARY         = "summary"
PROP_SUMMARY_SOURCE  = "summary_source"   # ast | llm | manual
PROP_SIGNATURE       = "signature"
PROP_DOCSTRING       = "docstring"

# Scanner provenance — which scanner build wrote this row
PROP_SCANNER_VERSION = "scanner_version"

# Structural metadata
PROP_START_LINE        = "start_line"
PROP_END_LINE          = "end_line"
PROP_LINE_COUNT        = "line_count"
PROP_TOTAL_LINES       = "total_lines"
PROP_PARENT_CLASS      = "parent_class"
PROP_IS_PUBLIC         = "is_public"
PROP_IS_ENTRYPOINT     = "is_entrypoint"
PROP_COMPLEXITY_BUCKET = "complexity_bucket"
PROP_SYMBOL_COUNT      = "symbol_count"

# ─── The two embedding lanes (the whole point of v1) ──────────────────────
PROP_SUMMARY_EMBEDDING = "summary_embedding"
PROP_CODE_EMBEDDING    = "code_embedding"

# Scan state
PROP_SCAN_ID       = "scan_id"
PROP_SCAN_STATUS   = "scan_status"        # running | complete | failed
PROP_SCAN_STARTED  = "started_at"
PROP_SCAN_FINISHED = "finished_at"
PROP_SCAN_ERROR    = "error_message"
PROP_SCAN_STATS    = "stats"              # JSON-encoded dict

# Timestamps
PROP_INDEXED_AT = "indexed_at"


# ---------------------------------------------------------------------------
# Edge property keys
# ---------------------------------------------------------------------------
# Edges carry their own metadata — flags the indexer sets today and the
# retriever / graph tools read later.  Keeping the names here means the
# store layer and the retrieval layer can't drift.

# CALLS edge — shortcoming #2 lands here.  Until the AST parser emits
# qualified callee names, v1 still resolves by flat name and marks the
# ambiguous ones so retrieval can down-weight them.
PROP_IS_DIRECT    = "is_direct"       # resolved to a concrete Symbol node (vs. unresolved name)
PROP_IS_AMBIGUOUS = "is_ambiguous"    # flat name matched > 1 candidate symbol

# IMPORTS edge — distinguishes module imports from selective imports so
# the retriever can treat "from x import y" differently from "import x".
PROP_IMPORT_TYPE  = "import_type"     # module | symbol | wildcard


# ---------------------------------------------------------------------------
# Enum-like string values
# ---------------------------------------------------------------------------
# Anywhere the schema says "one of these strings", the exact string lives
# here.  Typos in string literals are the #1 silent bug in Cypher code —
# centralizing them lets the type checker help.

# summary_source: who wrote the current summary
SUMMARY_SOURCE_AST    = "ast"       # extracted from docstring at parse time
SUMMARY_SOURCE_LLM    = "llm"       # rewritten by the enricher
SUMMARY_SOURCE_MANUAL = "manual"    # human edit (reserved — not used yet)

# scan_status: lifecycle of a ScanV1 node
SCAN_STATUS_RUNNING  = "running"
SCAN_STATUS_COMPLETE = "complete"
SCAN_STATUS_FAILED   = "failed"

# symbol_type: what kind of definition this Symbol node represents
SYMBOL_TYPE_FUNCTION  = "function"
SYMBOL_TYPE_METHOD    = "method"
SYMBOL_TYPE_CLASS     = "class"
SYMBOL_TYPE_INTERFACE = "interface"
SYMBOL_TYPE_ENUM      = "enum"

# complexity_bucket: coarse size bucket used by the retriever to decide
# how much code to pull into an LLM context window.
COMPLEXITY_LOW    = "low"       # ≤ 20 lines
COMPLEXITY_MEDIUM = "medium"    # 21 – 100 lines
COMPLEXITY_HIGH   = "high"      # > 100 lines

# import_type values (paired with PROP_IMPORT_TYPE above)
IMPORT_TYPE_MODULE   = "module"
IMPORT_TYPE_SYMBOL   = "symbol"
IMPORT_TYPE_WILDCARD = "wildcard"


# ---------------------------------------------------------------------------
# Vector index names
# ---------------------------------------------------------------------------
# Neo4j 5.11+ native vector indexes.  Each lane gets its own index so the
# retriever can query them independently, then fuse.
#
# Dimensions come from settings at setup time (not hard-coded here) so the
# same file works whether the embedder is SentenceTransformer, jina-code,
# voyage-code-3, etc.

IDX_SYMBOL_SUMMARY_VEC = "symbol_summary_vec_idx"
IDX_SYMBOL_CODE_VEC    = "symbol_code_vec_idx"
IDX_FILE_SUMMARY_VEC   = "file_summary_vec_idx"

# Constraint / lookup indexes
IDX_SYMBOL_UNIQUE = "symbol_unique_idx"   # (org_id, repo, file_path, symbol_name)
IDX_FILE_UNIQUE   = "file_unique_idx"     # (org_id, repo, file_path)
IDX_REPO_UNIQUE   = "repo_unique_idx"     # (org_id, repo)

# Optional fulltext index over identifiers and raw code (BM25 lane for the
# hybrid retrieval story — v1 sets it up but the retriever uses it later).
IDX_SYMBOL_FULLTEXT = "symbol_fulltext_idx"
IDX_FILE_FULLTEXT   = "file_fulltext_idx"


# ---------------------------------------------------------------------------
# Vector index config defaults
# ---------------------------------------------------------------------------
# Neo4j vector index options: similarity_function ∈ {"cosine", "euclidean"}.
# Keep "cosine" to match v0 Pinecone behavior.

DEFAULT_VECTOR_SIMILARITY = "cosine"


# ---------------------------------------------------------------------------
# Vector index specs
# ---------------------------------------------------------------------------
# Each vector index is described once — index name, the label + property
# it covers, and the similarity function.  Dimensions are injected at
# setup time (from the embedder), not baked in, so swapping embedding
# models doesn't touch this file.
#
# store.setup_schema iterates ALL_VECTOR_INDEXES and calls as_options()
# with whatever dimensionality the live embedder reports.

@dataclass(frozen=True)
class VectorIndexSpec:
    """Describes one Neo4j vector index.

    `name`       — index name (one of IDX_*_VEC above).
    `label`      — node label the index covers.
    `property`   — node property holding the embedding array.
    `similarity` — "cosine" or "euclidean".
    """
    name: str
    label: str
    property: str
    similarity: str = DEFAULT_VECTOR_SIMILARITY

    def as_options(self, dimensions: int) -> Dict[str, Any]:
        """Return the options dict Neo4j expects for this index.

        Shape matches Neo4j 5.11+ CREATE VECTOR INDEX syntax:
            {indexConfig: {`vector.dimensions`: N,
                           `vector.similarity_function`: "cosine"}}
        """
        return {
            "indexConfig": {
                "vector.dimensions": int(dimensions),
                "vector.similarity_function": self.similarity,
            }
        }


SYMBOL_SUMMARY_VEC = VectorIndexSpec(
    name=IDX_SYMBOL_SUMMARY_VEC,
    label=LABEL_SYMBOL,
    property=PROP_SUMMARY_EMBEDDING,
)

SYMBOL_CODE_VEC = VectorIndexSpec(
    name=IDX_SYMBOL_CODE_VEC,
    label=LABEL_SYMBOL,
    property=PROP_CODE_EMBEDDING,
)

FILE_SUMMARY_VEC = VectorIndexSpec(
    name=IDX_FILE_SUMMARY_VEC,
    label=LABEL_FILE,
    property=PROP_SUMMARY_EMBEDDING,
)

# Iterate this tuple in store.setup_schema — adding a new lane is a
# one-line change here and nothing else.
ALL_VECTOR_INDEXES = (
    SYMBOL_SUMMARY_VEC,
    SYMBOL_CODE_VEC,
    FILE_SUMMARY_VEC,
)


# ---------------------------------------------------------------------------
# Fulltext analyzer
# ---------------------------------------------------------------------------
# CRITICAL: Neo4j's default "standard" analyzer applies English stemming
# and tokenization that mangles code identifiers — "getUserById" becomes
# ["getuserbyid"], "_private_helper" gets the leading underscore stripped,
# camelCase is not split, and stop-words like "is"/"as"/"do" (all real
# function names!) are dropped entirely.
#
# "whitespace" keeps identifiers intact at the cost of no stemming.
# Good enough for v1; a custom analyzer that splits camelCase AND
# keeps the original form is a later upgrade.

FULLTEXT_ANALYZER_CODE = "whitespace"
