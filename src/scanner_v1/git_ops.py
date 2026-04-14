"""
Git operations for scanner_v1.

Currently a thin re-export of src.scanner.git_ops — v0's git wrapper is
pure subprocess / pathlib code with no coupling to Mongo, Pinecone, or
Neo4j, so there's nothing v1-specific to change yet.

This file exists so scanner_v1 is self-contained and so future git-level
optimizations (shallow clone, sparse checkout, partial-clone filters for
faster incremental scans, provider-specific diff API fast-paths) have a
home without touching the v0 module.

Anything importable from src.scanner.git_ops is importable from here:

    from src.scanner_v1.git_ops import clone_or_pull, get_diff, ...

If v1 needs to override a function, replace the re-export with a real
definition in this file and leave the rest passthrough.
"""

from __future__ import annotations

# Re-export everything public from v0.  Keep this list explicit so it's
# obvious what v1 depends on — and so adding/removing functions in v0
# shows up as a diff here.
from src.scanner.git_ops import (
    FileChange,
    FileChangeType,
    DiffResult,
    clone_or_pull,
    get_diff,
    get_head_sha,
    get_language,
    list_all_files,
    should_skip_file,
)

__all__ = [
    "FileChange",
    "FileChangeType",
    "DiffResult",
    "clone_or_pull",
    "get_diff",
    "get_head_sha",
    "get_language",
    "list_all_files",
    "should_skip_file",
]


# ---------------------------------------------------------------------------
# v1-specific overrides (none yet)
# ---------------------------------------------------------------------------
# Candidate future additions:
#
#   def shallow_clone(...) -> str:
#       """Depth-1 clone for first-time full scans of huge repos."""
#       ...
#
#   def sparse_checkout(...) -> None:
#       """Restrict checkout to a path filter (monorepo single-package scans)."""
#       ...
#
#   def get_diff_by_api(provider: str, ...) -> DiffResult:
#       """Use GitHub/GitLab compare API instead of local git diff for
#       remote-only incremental scans (no working tree needed)."""
#       ...
