"""
Dual-lane embedder — the fix for shortcoming #1.

v0 embeds only `qualified_name + signature + docstring[:500]`.  That vector
never sees the code, so queries phrased in code vocabulary (identifiers,
literals, library names, error strings) systematically miss.

v1 produces TWO vectors per symbol:

  summary_embedding  ← built from (qualified_name + signature + summary)
                       The enricher may later overwrite this with a vector
                       of the LLM-written summary. Used for natural-language
                       queries ("function that handles retry").

  code_embedding     ← built from the actual code body (optionally
                       trimmed / AST-chunked). Produced ONCE at scan time
                       and never touched by the enricher. Used for
                       identifier / literal / API-shaped queries
                       ("stripe.Charge.create", "ECONNRESET", "balance -=").

Retrieval (next phase) queries both indexes and fuses results.

Design knobs baked into this file:
  - Same model for both lanes in v1 (simpler).  The interface leaves room
    to plug a code-native model (jina-code, voyage-code-3) into embed_code()
    without touching the rest of the scanner.
  - Single-vector per symbol even for long functions in v1.  AST-chunking
    long bodies is a future concern; for now we truncate at MAX_CODE_CHARS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol

logger = logging.getLogger("xmem.scanner_v1.embedder")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Hard cap on how much of a symbol body we feed to the code-lane embedder.
# Most embedding models have an 8k–32k token window; chars ≈ tokens × 3–4.
# Start conservative and measure.
MAX_CODE_CHARS = 36000

# Hard cap on the summary-lane input.
MAX_SUMMARY_CHARS = 1200

# When a body is longer than MAX_CODE_CHARS, how to truncate.
#   "head"        → first N chars
#   "head_tail"   → first N/2 + last N/2 (keeps the return + any trailing logic)
TRUNCATE_STRATEGY = "head_tail"


# ---------------------------------------------------------------------------
# EmbedFn protocol
# ---------------------------------------------------------------------------

class EmbedFn(Protocol):
    """Any callable that turns a string into a float vector.
    Matches the signature of src/pipelines/ingest.embed_text."""
    def __call__(self, text: str) -> List[float]: ...


# ---------------------------------------------------------------------------
# Input shapes
# ---------------------------------------------------------------------------

@dataclass
class SymbolEmbedInput:
    """Everything the embedder needs to build both lanes for one symbol.
    The indexer constructs this from a ParsedSymbol."""
    qualified_name: str
    symbol_type: str
    signature: str
    docstring: str
    summary: str        # AST summary or enricher summary
    raw_code: str
    language: str = ""
    parent_class: Optional[str] = None


@dataclass
class FileEmbedInput:
    """File-level embedding input. Only a summary lane at the file level;
    file bodies are indexed via their symbols."""
    file_path: str
    language: str
    summary: str
    symbol_names: List[str]


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------
# Pure functions so the user can unit-test them without touching Neo4j or
# the embedding model.
# ---------------------------------------------------------------------------

def build_summary_text(sym: SymbolEmbedInput) -> str:
    """Compose the string that will be embedded into the summary lane.

    Includes: qualified_name, signature, docstring (if any), and the
    (possibly enriched) summary.  Length-capped at MAX_SUMMARY_CHARS so
    the model sees a bounded input even when the docstring is huge.
    """
    parts = [sym.qualified_name, sym.signature]
    if sym.docstring:
        parts.append(sym.docstring[:500])
    if sym.summary:
        parts.append(sym.summary)
    text = " ".join(p for p in parts if p)
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[:MAX_SUMMARY_CHARS]
    return text


def build_code_text(sym: SymbolEmbedInput) -> str:
    """Compose the string that will be embedded into the code lane.

    Maximizes lexical fidelity to the actual source:

        {qualified_name}
        {signature}
        {raw_code, truncated per TRUNCATE_STRATEGY}

    The natural-language summary is deliberately excluded — the whole
    point of the code lane is that it preserves source vocabulary.
    """
    parts = [sym.qualified_name, sym.signature, _truncate_code(sym.raw_code)]
    return "\n".join(p for p in parts if p)


def build_file_summary_text(f: FileEmbedInput) -> str:
    """Compose the string embedded into the file summary lane.
    file_path + language + summary + a short symbol-name list."""
    parts = [f.file_path, f.language]
    if f.summary:
        parts.append(f.summary)
    parts.extend(f.symbol_names)
    text = " ".join(p for p in parts if p)
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[:MAX_SUMMARY_CHARS]
    return text


def _truncate_code(raw_code: str, max_chars: int = MAX_CODE_CHARS) -> str:
    """Apply TRUNCATE_STRATEGY to raw code.
    - head:      first max_chars characters
    - head_tail: first max_chars/2 + "\\n...\\n" + last max_chars/2
                 (keeps the function opening and its return/trailing logic)
    """
    if len(raw_code) <= max_chars:
        return raw_code
    if TRUNCATE_STRATEGY == "head":
        return raw_code[:max_chars]
    if TRUNCATE_STRATEGY == "head_tail":
        half = max_chars // 2
        return raw_code[:half] + "\n...\n" + raw_code[-half:]
    raise ValueError(f"Unknown truncate strategy: {TRUNCATE_STRATEGY}")


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """Produces summary-lane and code-lane vectors.

    Holds one EmbedFn per lane so a code-native model can be swapped into
    the code lane without affecting the summary lane.  In v1 both default
    to the same function; migration to a code-specific model is a
    one-line change in __init__.
    """

    def __init__(
        self,
        summary_embed_fn: EmbedFn,
        code_embed_fn: Optional[EmbedFn] = None,
    ) -> None:
        # code_embed_fn defaults to summary_embed_fn in v1 — same model,
        # different input text. Passing a distinct code-native model
        # (jina-code, voyage-code-3) is a one-line swap at the call site.
        self.summary_embed_fn: EmbedFn = summary_embed_fn
        self.code_embed_fn: EmbedFn = code_embed_fn or summary_embed_fn

    # ── Symbol-level ──────────────────────────────────────────────────

    def embed_symbol(self, sym: SymbolEmbedInput) -> "SymbolEmbedding":
        """Build both lanes for one symbol.
        Returns a SymbolEmbedding carrying both vectors plus the source
        strings (handy for debugging / retrieval-quality inspection)."""
        summary_text = build_summary_text(sym)
        code_text = build_code_text(sym)
        return SymbolEmbedding(
            qualified_name=sym.qualified_name,
            summary_text=summary_text,
            summary_vector=self.summary_embed_fn(summary_text),
            code_text=code_text,
            code_vector=self.code_embed_fn(code_text),
        )

    def embed_symbols_batch(
        self, symbols: List[SymbolEmbedInput],
    ) -> List["SymbolEmbedding"]:
        """Batch version.  The EmbedFn protocol is per-string, so today
        this is a simple loop — the seam exists so a future batch-aware
        embed fn (SentenceTransformer.encode-style) can be plugged in
        here without rewriting the indexer."""
        return [self.embed_symbol(sym) for sym in symbols]

    # ── File-level ────────────────────────────────────────────────────

    def embed_file(self, f: FileEmbedInput) -> "FileEmbedding":
        """Build the single summary-lane vector for a file."""
        summary_text = build_file_summary_text(f)
        return FileEmbedding(
            file_path=f.file_path,
            summary_text=summary_text,
            summary_vector=self.summary_embed_fn(summary_text),
        )

    def embed_files_batch(
        self, files: List[FileEmbedInput],
    ) -> List["FileEmbedding"]:
        return [self.embed_file(f) for f in files]

    # ── Enricher hook ─────────────────────────────────────────────────

    def re_embed_summary(self, text: str) -> List[float]:
        """Single-text helper used by the enricher when it rewrites a
        summary and needs a fresh vector for just the summary lane.
        The code lane is never re-embedded."""
        return self.summary_embed_fn(text)


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

@dataclass
class SymbolEmbedding:
    qualified_name: str
    summary_text: str
    summary_vector: List[float]
    code_text: str
    code_vector: List[float]


@dataclass
class FileEmbedding:
    file_path: str
    summary_text: str
    summary_vector: List[float]
