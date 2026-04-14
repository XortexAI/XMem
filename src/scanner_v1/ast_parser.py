"""
AST parser for scanner_v1.

Thin re-export of src.scanner.ast_parser. v0's parser already returns
everything v1 needs: ParsedSymbol (with raw_code, signature, docstring,
content_hash), ParsedImport, ParsedCall, ParsedFile. No v1-specific
change required today.

This shim exists so scanner_v1 is self-contained and so future
parser-level fixes (notably shortcoming #2 — qualified call resolution,
and shortcoming #3 — multi-language import resolution) can live here
without editing the v0 module.

Anything importable from src.scanner.ast_parser is importable here:

    from src.scanner_v1.ast_parser import parse_file, ParsedSymbol, ...

Override pattern: when v1 needs to change a function, replace the
re-export below with a real definition and leave the rest passthrough.
"""

from __future__ import annotations

from src.scanner.ast_parser import (
    ParsedSymbol,
    ParsedImport,
    ParsedCall,
    ParsedFile,
    parse_file,
    compute_content_hash,
)

__all__ = [
    "ParsedSymbol",
    "ParsedImport",
    "ParsedCall",
    "ParsedFile",
    "parse_file",
    "compute_content_hash",
]


# ---------------------------------------------------------------------------
# v1-specific overrides (none yet)
# ---------------------------------------------------------------------------
# Planned future additions (tied to later shortcoming fixes):
#
#   - Qualified call resolution: enrich ParsedCall with
#     `callee_qualified_name` so the indexer's call-edge builder can
#     stop collapsing same-named methods across classes.
#
#   - Multi-language import resolution helpers: per-language path
#     resolvers (TS tsconfig paths, Go module paths, Python relative
#     imports) that the indexer can plug in without knowing the
#     language details.
#
#   - AST-aware chunking for long function bodies so the code embedding
#     lane can produce multiple vectors per symbol instead of one
#     truncated blob.
