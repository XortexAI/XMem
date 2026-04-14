"""
IndexerV1 — orchestrates a repo scan against the single Neo4j store.

High-level flow (largely mirrors v0 so the diff is readable):

  1. clone_or_pull → HEAD sha
  2. start_scan (Scan node in Neo4j)
  3. determine file set:
       - incremental: git diff last_sha..head_sha
       - full:        walk all tracked files
  4. per file:
       a. read + content_hash — skip unchanged
       b. ast_parser.parse_file → ParsedFile
       c. build SymbolEmbedInput for each symbol
       d. Embedder.embed_symbols_batch → SymbolEmbedding list
          (DUAL vectors — summary lane + code lane)
       e. buffer file row (with file summary embedding)
       f. buffer symbol rows (with both lanes)
       g. diff against previous symbol hashes → delete removed symbols
  5. per deleted file: store.delete_file (cascading)
  6. build directory index (kept minimal for v1)
  7. build_call_edges  (same resolution algorithm as v0 for now —
                        will be fixed in a later shortcoming pass)
  8. build_import_edges (same as v0 — likewise a future fix)
  9. complete_scan with stats

Key differences vs v0:
  - ONE downstream store. No fan-out loops, no three-store reconciliation.
  - Batched writes end-to-end (fixes v0's per-symbol Pinecone roundtrip).
  - Deletions cascade via store.delete_file — no ghost nodes.
  - Symbols carry TWO embeddings.  The code lane is the fix for
    shortcoming #1 (embedded text doesn't contain code).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.scanner_v1.ast_parser import (
    ParsedFile,
    ParsedSymbol,
    parse_file,
    compute_content_hash,
)
from src.scanner_v1.git_ops import (
    clone_or_pull,
    get_diff,
    get_language,
    list_all_files,
    should_skip_file,
)

from src.scanner_v1.store import CodeStoreV1, build_symbol_row
from src.scanner_v1.embedder import (
    Embedder,
    SymbolEmbedInput,
    FileEmbedInput,
    SymbolEmbedding,
)
from src.scanner_v1 import schemas as S

logger = logging.getLogger("xmem.scanner_v1.indexer")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

SYMBOL_FLUSH_THRESHOLD = 200   # flush symbol buffer after N entries
FILE_FLUSH_THRESHOLD   = 50
CALL_EDGE_BATCH        = 500
IMPORT_EDGE_BATCH      = 500


# ---------------------------------------------------------------------------
# IndexerV1
# ---------------------------------------------------------------------------

class IndexerV1:
    """Single-store scanner orchestrator."""

    def __init__(
        self,
        org_id: str,
        store: CodeStoreV1,
        embedder: Embedder,
        clone_root: str = "/tmp/xmem_repos_v1",
    ) -> None:
        self.org_id = org_id
        self.store = store
        self.embedder = embedder
        self.clone_root = clone_root

        # Write buffers — flushed in UNWIND batches.
        self._file_rows: List[Dict[str, Any]] = []
        self._symbol_rows: List[Dict[str, Any]] = []

        # Stats counter bumped by _stat().
        self._stats: Dict[str, int] = defaultdict(int)

    # =================================================================
    # PUBLIC ENTRY
    # =================================================================

    def scan_repo(
        self,
        repo_name: str,
        repo_url: str,
        branch: str = "main",
        token: Optional[str] = None,
        force_full: bool = False,
    ) -> Dict[str, Any]:
        """Run a full or incremental scan."""
        start_time = time.time()
        self._stats = defaultdict(int)
        self._file_rows.clear()
        self._symbol_rows.clear()

        logger.info("=" * 70)
        logger.info("SCAN START: %s/%s (branch=%s)", self.org_id, repo_name, branch)
        logger.info("=" * 70)

        local_path = str(Path(self.clone_root) / self.org_id / repo_name)

        # 1. Clone or pull.
        head_sha = clone_or_pull(repo_url, local_path, branch=branch, token=token)

        # 2. Upsert Repository node BEFORE start_scan — start_scan MATCHes it.
        self.store.upsert_repository(self.org_id, repo_name, branch=branch)
        scan_id = self.store.start_scan(self.org_id, repo_name, head_sha)

        try:
            # 3. Determine file set.
            last_sha = self.store.get_last_commit_sha(self.org_id, repo_name)

            if force_full or not last_sha:
                logger.info("Running FULL scan (no previous scan or force_full=True)")
                files_to_process = self._get_all_source_files(local_path)
                files_to_delete: List[str] = []
            else:
                logger.info(
                    "Running INCREMENTAL scan (diff %s..%s)",
                    last_sha[:8], head_sha[:8],
                )
                diff = get_diff(local_path, last_sha, head_sha)
                files_to_process = [
                    f for f in diff.changed_files if not should_skip_file(f)
                ]
                files_to_delete = [
                    f for f in diff.deleted if not should_skip_file(f)
                ]

            logger.info(
                "Files to process: %d, Files to delete: %d",
                len(files_to_process), len(files_to_delete),
            )

            # 4. Process changed/new files.
            parsed_files: List[ParsedFile] = []
            for file_path in files_to_process:
                try:
                    parsed = self._process_file(
                        local_path, repo_name, file_path, head_sha,
                    )
                    if parsed:
                        parsed_files.append(parsed)
                except Exception as e:
                    logger.error("Failed to process %s: %s", file_path, e)
                    self._stat("files_errored")

            # Flush anything left in the buffers before we start edge work —
            # edge writes MATCH on Symbol nodes and need them to exist.
            self._flush_all()

            # 5. Deleted files.
            for file_path in files_to_delete:
                try:
                    self._delete_file(repo_name, file_path)
                except Exception as e:
                    logger.error("Failed to delete %s: %s", file_path, e)

            # 6. Directory rollup.
            self._build_directory_index(repo_name, parsed_files)

            # 7. Call edges (Symbol → Symbol).
            self._build_call_edges(repo_name, parsed_files, local_path)

            # 8. Import edges (File → File).
            self._build_import_edges(repo_name, parsed_files, local_path)

            # 9. Complete scan.
            duration = time.time() - start_time
            self.store.complete_scan(
                self.org_id, repo_name, scan_id,
                commit_sha=head_sha,
                stats=dict(self._stats),
                duration_seconds=duration,
            )

            logger.info("=" * 70)
            logger.info(
                "SCAN COMPLETE: %s/%s in %.1fs",
                self.org_id, repo_name, duration,
            )
            for k, v in sorted(self._stats.items()):
                logger.info("  %s: %d", k, v)
            logger.info("=" * 70)

            return {
                "org_id": self.org_id,
                "repo": repo_name,
                "scan_id": scan_id,
                "commit_sha": head_sha,
                "duration_seconds": round(duration, 1),
                "stats": dict(self._stats),
            }

        except Exception as e:
            logger.error("Scan failed: %s", e)
            self.store.fail_scan(self.org_id, repo_name, scan_id, str(e))
            raise

    # =================================================================
    # FILE PROCESSING
    # =================================================================

    def _process_file(
        self,
        local_path: str,
        repo_name: str,
        file_path: str,
        commit_sha: str,
    ) -> Optional[ParsedFile]:
        """Parse one file and buffer its upsert rows for batch flush."""
        language = get_language(file_path)
        if not language:
            return None

        full_path = Path(local_path) / file_path
        if not full_path.exists():
            return None

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Cannot read %s: %s", file_path, e)
            return None

        # Short-circuit on unchanged content — avoids parse + embed cost.
        file_hash = compute_content_hash(content)
        old_hash = self.store.get_file_hash(self.org_id, repo_name, file_path)
        if old_hash == file_hash:
            self._stat("files_skipped_unchanged")
            return None

        self._stat("files_processed")

        # Parse.
        parsed = parse_file(file_path, content, language)
        if parsed.parse_errors:
            self._stat("parse_errors", len(parsed.parse_errors))

        # Old symbol hashes for incremental symbol processing.
        old_symbol_hashes = self.store.get_file_symbol_hashes(
            self.org_id, repo_name, file_path,
        )
        current_symbol_names: Set[str] = set()

        # Pick the symbols that need re-embedding.
        changed_symbols: List[ParsedSymbol] = []
        for sym in parsed.symbols:
            current_symbol_names.add(sym.qualified_name)
            if old_symbol_hashes.get(sym.qualified_name) == sym.content_hash:
                self._stat("symbols_skipped_unchanged")
                continue
            changed_symbols.append(sym)

        # Embed all changed symbols in ONE call (fixes v0's per-symbol roundtrip).
        sym_inputs = [
            SymbolEmbedInput(
                qualified_name=sym.qualified_name,
                symbol_type=sym.symbol_type,
                signature=sym.signature,
                docstring=sym.docstring,
                summary=sym.summary,
                raw_code=sym.raw_code,
                language=language,
                parent_class=sym.parent_class,
            )
            for sym in changed_symbols
        ]
        sym_embeddings = self.embedder.embed_symbols_batch(sym_inputs)

        # File-level embedding (summary lane only).
        file_input = FileEmbedInput(
            file_path=file_path,
            language=language,
            summary=parsed.summary,
            symbol_names=parsed.symbol_names,
        )
        file_embedding = self.embedder.embed_file(file_input)

        # Buffer writes.
        self._buffer_file_row(
            repo_name=repo_name,
            parsed=parsed,
            file_embedding=file_embedding,
            content=content,
            content_hash=file_hash,
            commit_sha=commit_sha,
        )

        for sym, emb in zip(changed_symbols, sym_embeddings):
            self._buffer_symbol_row(
                repo_name=repo_name,
                file_path=file_path,
                language=language,
                symbol=sym,
                embedding=emb,
                commit_sha=commit_sha,
            )
            self._stat("symbols_indexed")

        # Deleted symbols (present before, absent now).
        removed = set(old_symbol_hashes.keys()) - current_symbol_names
        for sym_name in removed:
            self._delete_symbol(repo_name, file_path, sym_name)
            self._stat("symbols_deleted")

        return parsed

    def _buffer_file_row(
        self,
        repo_name: str,
        parsed: ParsedFile,
        file_embedding,
        content: str,
        content_hash: str,
        commit_sha: str,
    ) -> None:
        row = {
            "org_id": self.org_id,
            "repo": repo_name,
            "file_path": parsed.file_path,
            "language": parsed.language,
            "raw_content": content,
            "summary": parsed.summary,
            "summary_source": S.SUMMARY_SOURCE_AST,
            "summary_embedding": file_embedding.summary_vector,
            "total_lines": int(parsed.total_lines),
            "symbol_count": len(parsed.symbols),
            "symbol_names": list(parsed.symbol_names),
            "content_hash": content_hash,
            "commit_sha": commit_sha,
        }
        self._file_rows.append(row)
        if len(self._file_rows) >= FILE_FLUSH_THRESHOLD:
            self._flush_file_buffer()

    def _buffer_symbol_row(
        self,
        repo_name: str,
        file_path: str,
        language: str,
        symbol: ParsedSymbol,
        embedding: SymbolEmbedding,
        commit_sha: str,
    ) -> None:
        sym_dict = {
            "qualified_name": symbol.qualified_name,
            "symbol_name": symbol.name,
            "symbol_type": symbol.symbol_type,
            "language": language,
            "signature": symbol.signature,
            "docstring": symbol.docstring,
            "summary": symbol.summary,
            "summary_source": S.SUMMARY_SOURCE_AST,
            "raw_code": symbol.raw_code,
            "content_hash": symbol.content_hash,
            "signature_hash": symbol.content_hash,
            "start_line": symbol.start_line,
            "end_line": symbol.end_line,
            "line_count": symbol.line_count,
            "parent_class": symbol.parent_class,
            "is_public": symbol.is_public,
            "is_entrypoint": symbol.is_entrypoint,
            "complexity_bucket": symbol.complexity_bucket,
        }
        row = build_symbol_row(
            self.org_id, repo_name, file_path,
            sym_dict, embedding, commit_sha,
        )
        self._symbol_rows.append(row)
        if len(self._symbol_rows) >= SYMBOL_FLUSH_THRESHOLD:
            # Symbols MATCH File — flush files first so the MATCH succeeds.
            self._flush_file_buffer()
            self._flush_symbol_buffer()

    def _flush_file_buffer(self) -> None:
        if not self._file_rows:
            return
        self.store.upsert_files_batch(self._file_rows)
        self._stat("files_written", len(self._file_rows))
        self._file_rows.clear()

    def _flush_symbol_buffer(self) -> None:
        if not self._symbol_rows:
            return
        self.store.upsert_symbols_batch(self._symbol_rows)
        self._stat("symbols_written", len(self._symbol_rows))
        self._symbol_rows.clear()

    def _flush_all(self) -> None:
        """Order matters: files before symbols."""
        self._flush_file_buffer()
        self._flush_symbol_buffer()

    # =================================================================
    # DIRECTORY INDEX
    # =================================================================

    def _build_directory_index(
        self,
        repo_name: str,
        parsed_files: List[ParsedFile],
    ) -> None:
        """Aggregate parsed files into directory rollups.
        v1 directories are purely structural — no embeddings (fixes v0
        shortcoming #5 where directories embedded meaningless file lists)."""
        dir_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "languages": set(),
            "file_count": 0,
            "total_symbols": 0,
            "file_names": [],
        })

        for pf in parsed_files:
            dir_path = str(Path(pf.file_path).parent) + "/"
            if dir_path == "./":
                dir_path = "/"
            bucket = dir_data[dir_path]
            bucket["languages"].add(pf.language)
            bucket["file_count"] += 1
            bucket["total_symbols"] += len(pf.symbols)
            bucket["file_names"].append(Path(pf.file_path).name)

        for dir_path, data in dir_data.items():
            names = data["file_names"][:20]
            summary = f"Directory {dir_path}: {', '.join(names)}"
            if len(data["file_names"]) > 20:
                summary += f" and {len(data['file_names']) - 20} more files"

            self.store.upsert_directory(
                org_id=self.org_id,
                repo=repo_name,
                dir_path=dir_path,
                languages=list(data["languages"]),
                file_count=data["file_count"],
                total_symbols=data["total_symbols"],
                summary=summary,
            )
            self._stat("directories_indexed")

    # =================================================================
    # GRAPH EDGES
    # =================================================================
    # v1 uses the same flat-name resolution as v0; shortcoming #2 fixes
    # this by teaching the parser to emit qualified callee names.
    # Until then, flat-name ambiguities are flagged on the edge via
    # is_ambiguous so retrieval can down-weight them.
    # -----------------------------------------------------------------

    def _build_call_edges(
        self,
        repo_name: str,
        parsed_files: List[ParsedFile],
        local_path: str,
    ) -> None:
        """Build CALLS edges from parsed ParsedCall data."""
        # Flat name → list of (file_path, qualified_name).
        # Multiple entries = ambiguous; v1 flags these on the edge.
        name_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for pf in parsed_files:
            for sym in pf.symbols:
                name_index[sym.name].append((pf.file_path, sym.qualified_name))

        rows: List[Dict[str, Any]] = []
        for pf in parsed_files:
            # Per-file caller lookup: flat name → qualified_name within this file.
            caller_lookup = {sym.name: sym.qualified_name for sym in pf.symbols}

            for call in pf.calls:
                caller_qname = caller_lookup.get(call.caller_name)
                if not caller_qname:
                    continue

                candidates = name_index.get(call.callee_name, [])
                if not candidates:
                    continue

                callee_file, callee_qname = candidates[0]
                is_ambiguous = len(candidates) > 1

                rows.append({
                    "org_id": self.org_id,
                    "repo": repo_name,
                    "caller_file": pf.file_path,
                    "caller_qname": caller_qname,
                    "callee_file": callee_file,
                    "callee_qname": callee_qname,
                    "is_direct": bool(call.is_direct),
                    "is_ambiguous": is_ambiguous,
                })

                if len(rows) >= CALL_EDGE_BATCH:
                    self.store.add_calls_edges_batch(rows)
                    self._stat("call_edges_created", len(rows))
                    rows = []

        if rows:
            self.store.add_calls_edges_batch(rows)
            self._stat("call_edges_created", len(rows))

    def _build_import_edges(
        self,
        repo_name: str,
        parsed_files: List[ParsedFile],
        local_path: str,
    ) -> None:
        """Build IMPORTS edges from parsed ParsedImport data."""
        known_files = {pf.file_path for pf in parsed_files}

        rows: List[Dict[str, Any]] = []
        for pf in parsed_files:
            for imp in pf.imports:
                resolved = self._resolve_import_to_file(
                    imp.module, pf.file_path, known_files,
                )
                if not resolved or resolved == pf.file_path:
                    continue

                import_type = (
                    S.IMPORT_TYPE_SYMBOL if imp.names else S.IMPORT_TYPE_MODULE
                )
                rows.append({
                    "org_id": self.org_id,
                    "repo": repo_name,
                    "importer_file": pf.file_path,
                    "imported_file": resolved,
                    "import_type": import_type,
                })

                if len(rows) >= IMPORT_EDGE_BATCH:
                    self.store.add_imports_edges_batch(rows)
                    self._stat("import_edges_created", len(rows))
                    rows = []

        if rows:
            self.store.add_imports_edges_batch(rows)
            self._stat("import_edges_created", len(rows))

    def _resolve_import_to_file(
        self,
        module: str,
        source_file: str,
        known_files: Set[str],
    ) -> Optional[str]:
        """Python default-layout resolver, same as v0.
        Multi-language resolution is a future shortcoming fix."""
        if not module:
            return None

        candidates = [
            module.replace(".", "/") + ".py",
            module.replace(".", "/") + "/__init__.py",
        ]
        for candidate in candidates:
            if candidate in known_files:
                return candidate
        return None

    # =================================================================
    # DELETION
    # =================================================================

    def _delete_file(self, repo_name: str, file_path: str) -> None:
        """Cascading DETACH DELETE — file + all its symbols in one call.
        Replaces v0's multi-store deletion dance that left Neo4j dirty."""
        self.store.delete_file(self.org_id, repo_name, file_path)
        self._stat("files_deleted")

    def _delete_symbol(
        self, repo_name: str, file_path: str, symbol_name: str,
    ) -> None:
        self.store.delete_symbol(self.org_id, repo_name, file_path, symbol_name)

    # =================================================================
    # HELPERS
    # =================================================================

    def _get_all_source_files(self, local_path: str) -> List[str]:
        """Walk all parseable files under local_path."""
        all_files = list_all_files(local_path)
        return [f for f in all_files if not should_skip_file(f)]

    def _stat(self, key: str, n: int = 1) -> None:
        self._stats[key] += n

    def close(self) -> None:
        """Release store handles."""
        self.store.close()
