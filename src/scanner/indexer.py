"""
Indexer — orchestrates the full scan pipeline.

Flow:
  1. Clone or pull the repo from GitHub
  2. Compute git diff against last scan (incremental) or scan all files (full)
  3. For each changed file:
     a. Parse with AST (no LLM)
     b. Check content hashes against MongoDB (skip unchanged symbols)
     c. Store raw code in MongoDB
     d. Embed summaries and upsert to Pinecone (symbols, files, directories namespaces)
     e. Upsert nodes and edges to Neo4j knowledge graph
  4. For each deleted file: clean up MongoDB + Pinecone + Neo4j
  5. Record scan state in MongoDB

Design:
  - Incremental by default (only processes git diff)
  - Falls back to full scan if no previous scan exists
  - All summaries and embeddings are built from AST/docstrings (no LLM)
  - Batched writes for Pinecone and MongoDB (cost + latency efficient)
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from src.scanner.ast_parser import ParsedFile, ParsedSymbol, parse_file, compute_content_hash
from src.scanner.code_store import CodeStore
from src.scanner.git_ops import (
    DiffResult,
    clone_or_pull,
    get_diff,
    get_head_sha,
    get_language,
    list_all_files,
    should_skip_file,
)
from src.graph.code_graph_client import CodeGraphClient
from src.schemas.code import (
    symbols_namespace,
    files_namespace,
    directories_namespace,
)
from src.storage.pinecone import PineconeVectorStore
from src.config import settings

logger = logging.getLogger("xmem.scanner.indexer")

PINECONE_BATCH = 50


def _symbol_pinecone_id(org_id: str, repo: str, file_path: str, symbol_name: str) -> str:
    key = f"{org_id}:{repo}:{file_path}:{symbol_name}"
    return hashlib.sha256(key.encode()).hexdigest()


def _file_pinecone_id(org_id: str, repo: str, file_path: str) -> str:
    key = f"{org_id}:{repo}:{file_path}"
    return hashlib.sha256(key.encode()).hexdigest()


def _dir_pinecone_id(org_id: str, repo: str, dir_path: str) -> str:
    key = f"{org_id}:{repo}:{dir_path}"
    return hashlib.sha256(key.encode()).hexdigest()


class Indexer:
    """Orchestrates the full scan: parse → MongoDB → Pinecone → Neo4j."""

    def __init__(
        self,
        org_id: str,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        code_store: Optional[CodeStore] = None,
        code_graph: Optional[CodeGraphClient] = None,
        clone_root: str = "/tmp/xmem_repos",
    ) -> None:
        self.org_id = org_id
        self.clone_root = clone_root

        # Embedding function (SentenceTransformer — runs locally, no API cost)
        if embed_fn is None:
            from src.pipelines.ingest import embed_text
            self.embed_fn = embed_text
        else:
            self.embed_fn = embed_fn

        # MongoDB
        if code_store is None:
            self.code_store = CodeStore(
                uri=settings.mongodb_uri,
                database=settings.mongodb_database,
            )
        else:
            self.code_store = code_store

        # Neo4j
        if code_graph is None:
            self.code_graph = CodeGraphClient(
                uri=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
                embedding_fn=self.embed_fn,
            )
            self.code_graph.connect()
            self.code_graph.setup()
        else:
            self.code_graph = code_graph

        # Pinecone stores (lazily created per namespace)
        self._pinecone_stores: Dict[str, PineconeVectorStore] = {}

        # Stats
        self._stats: Dict[str, int] = defaultdict(int)

    def _get_pinecone(self, namespace: str) -> PineconeVectorStore:
        if namespace not in self._pinecone_stores:
            self._pinecone_stores[namespace] = PineconeVectorStore(
                api_key=settings.pinecone_api_key,
                index_name=settings.pinecone_index_name,
                dimension=settings.pinecone_dimension,
                metric=settings.pinecone_metric,
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
                namespace=namespace,
                create_if_not_exists=False,
            )
        return self._pinecone_stores[namespace]

    # ======================================================================
    # PUBLIC API
    # ======================================================================

    def scan_repo(
        self,
        repo_name: str,
        repo_url: str,
        branch: str = "main",
        token: Optional[str] = None,
        force_full: bool = False,
    ) -> Dict[str, Any]:
        """Run a full or incremental scan of a repository.

        Args:
            repo_name: Short name (e.g. "payment-service")
            repo_url: Git clone URL
            branch: Branch to scan
            token: GitHub token for private repos
            force_full: Skip incremental, re-scan everything

        Returns:
            Dict with scan stats and timing.
        """
        start_time = time.time()
        self._stats = defaultdict(int)

        logger.info("=" * 70)
        logger.info("SCAN START: %s/%s (branch=%s)", self.org_id, repo_name, branch)
        logger.info("=" * 70)

        local_path = str(Path(self.clone_root) / self.org_id / repo_name)

        # 1. Clone or pull
        try:
            head_sha = clone_or_pull(repo_url, local_path, branch=branch, token=token)
        except Exception as e:
            self.code_store.fail_scan(self.org_id, repo_name, str(e))
            logger.error("Git operation failed: %s", e)
            raise

        self.code_store.start_scan(self.org_id, repo_name, head_sha)

        try:
            # 2. Determine which files to process
            last_sha = self.code_store.get_last_commit_sha(self.org_id, repo_name)

            if force_full or not last_sha:
                logger.info("Running FULL scan (no previous scan or force_full=True)")
                files_to_process = self._get_all_source_files(local_path)
                files_to_delete: List[str] = []
            else:
                logger.info("Running INCREMENTAL scan (diff %s..%s)", last_sha[:8], head_sha[:8])
                diff = get_diff(local_path, last_sha, head_sha)
                files_to_process = [
                    f for f in diff.changed_files if not should_skip_file(f)
                ]
                files_to_delete = [
                    f for f in diff.deleted if not should_skip_file(f)
                ]

            logger.info("Files to process: %d, Files to delete: %d",
                         len(files_to_process), len(files_to_delete))

            # 3. Process changed/new files
            self.code_graph.upsert_repository(self.org_id, repo_name)
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
                    self._stats["files_errored"] += 1

            # 4. Process deleted files
            for file_path in files_to_delete:
                try:
                    self._delete_file(repo_name, file_path)
                except Exception as e:
                    logger.error("Failed to delete %s: %s", file_path, e)

            # 5. Build directory-level summaries
            self._build_directory_index(repo_name, parsed_files)

            # 6. Build call graph edges from parsed data
            self._build_call_edges(repo_name, parsed_files, local_path)

            # 7. Build import edges
            self._build_import_edges(repo_name, parsed_files, local_path)

            # 8. Complete scan
            duration = time.time() - start_time
            self.code_store.complete_scan(
                self.org_id, repo_name, head_sha,
                stats=dict(self._stats),
                duration_seconds=duration,
            )

            logger.info("=" * 70)
            logger.info("SCAN COMPLETE: %s/%s in %.1fs", self.org_id, repo_name, duration)
            for k, v in sorted(self._stats.items()):
                logger.info("  %s: %d", k, v)
            logger.info("=" * 70)

            return {
                "org_id": self.org_id,
                "repo": repo_name,
                "commit_sha": head_sha,
                "duration_seconds": round(duration, 1),
                "stats": dict(self._stats),
            }

        except Exception as e:
            self.code_store.fail_scan(self.org_id, repo_name, str(e))
            logger.error("Scan failed: %s", e)
            raise

    # ======================================================================
    # FILE PROCESSING
    # ======================================================================

    def _process_file(
        self,
        local_path: str,
        repo_name: str,
        file_path: str,
        commit_sha: str,
    ) -> Optional[ParsedFile]:
        """Process a single file: parse → diff → store → index."""
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

        # Quick check: has the file actually changed?
        file_hash = compute_content_hash(content)
        old_hash = self.code_store.get_file_hash(self.org_id, repo_name, file_path)
        if old_hash == file_hash:
            self._stats["files_skipped_unchanged"] += 1
            return None

        self._stats["files_processed"] += 1

        # Parse with AST (no LLM)
        parsed = parse_file(file_path, content, language)

        if parsed.parse_errors:
            self._stats["parse_errors"] += len(parsed.parse_errors)

        # Store raw file content in MongoDB (summary_source="ast")
        self.code_store.upsert_file(
            org_id=self.org_id, repo=repo_name,
            file_path=file_path, language=language,
            raw_content=content, summary=parsed.summary,
            summary_source="ast",
            total_lines=parsed.total_lines,
            content_hash=file_hash, commit_sha=commit_sha,
        )

        # Get old symbol hashes for incremental symbol processing
        old_symbol_hashes = self.code_store.get_file_symbol_hashes(
            self.org_id, repo_name, file_path,
        )

        # Track which symbols still exist (to detect deletions)
        current_symbol_names: Set[str] = set()

        # Process each symbol
        for symbol in parsed.symbols:
            current_symbol_names.add(symbol.qualified_name)

            # Skip if symbol code hasn't changed
            old_hash = old_symbol_hashes.get(symbol.qualified_name)
            if old_hash == symbol.content_hash:
                self._stats["symbols_skipped_unchanged"] += 1
                continue

            self._index_symbol(repo_name, file_path, language, symbol, commit_sha)
            self._stats["symbols_indexed"] += 1

        # Delete symbols that no longer exist in this file
        removed_symbols = set(old_symbol_hashes.keys()) - current_symbol_names
        for sym_name in removed_symbols:
            self._delete_symbol(repo_name, file_path, sym_name)
            self._stats["symbols_deleted"] += 1

        # Index file-level record in Pinecone
        self._index_file_record(repo_name, file_path, language, parsed, commit_sha)

        # Upsert file node in Neo4j
        self.code_graph.upsert_file(
            org_id=self.org_id, repo=repo_name,
            file_path=file_path, language=language,
            summary=parsed.summary,
            symbol_count=len(parsed.symbols),
            symbol_names=parsed.symbol_names,
            total_lines=parsed.total_lines,
            commit_sha=commit_sha,
        )

        return parsed

    def _index_symbol(
        self,
        repo_name: str,
        file_path: str,
        language: str,
        symbol: ParsedSymbol,
        commit_sha: str,
    ) -> None:
        """Store a symbol in MongoDB + Pinecone + Neo4j."""
        # MongoDB: raw code (summary_source="ast" — Phase 2 enricher will upgrade to "llm")
        self.code_store.upsert_symbol(
            org_id=self.org_id, repo=repo_name,
            file_path=file_path, symbol_name=symbol.qualified_name,
            symbol_type=symbol.symbol_type, language=language,
            raw_code=symbol.raw_code, signature=symbol.signature,
            docstring=symbol.docstring, summary=symbol.summary,
            summary_source="ast",
            start_line=symbol.start_line, end_line=symbol.end_line,
            line_count=symbol.line_count, parent_class=symbol.parent_class,
            is_public=symbol.is_public, is_entrypoint=symbol.is_entrypoint,
            complexity_bucket=symbol.complexity_bucket,
            content_hash=symbol.content_hash, commit_sha=commit_sha,
        )

        # Pinecone: embed the searchable text (summary + signature + docstring)
        ns = symbols_namespace(self.org_id, repo_name)
        store = self._get_pinecone(ns)
        vec_id = _symbol_pinecone_id(self.org_id, repo_name, file_path, symbol.qualified_name)

        embedding = self.embed_fn(symbol.searchable_text)

        metadata = {
            "org_id": self.org_id,
            "repo": repo_name,
            "file_path": file_path,
            "symbol_name": symbol.qualified_name,
            "symbol_type": symbol.symbol_type,
            "language": language,
            "signature": symbol.signature[:500],
            "is_public": symbol.is_public,
            "is_entrypoint": symbol.is_entrypoint,
            "complexity_bucket": symbol.complexity_bucket,
            "line_count": symbol.line_count,
            "start_line": symbol.start_line,
            "end_line": symbol.end_line,
        }

        store.add(
            texts=[symbol.summary],
            embeddings=[embedding],
            ids=[vec_id],
            metadata=[metadata],
        )

        # Neo4j: symbol node
        self.code_graph.upsert_symbol(
            org_id=self.org_id, repo=repo_name,
            file_path=file_path, symbol_name=symbol.qualified_name,
            symbol_type=symbol.symbol_type, summary=symbol.summary,
            signature=symbol.signature, docstring=symbol.docstring,
            start_line=symbol.start_line, end_line=symbol.end_line,
            commit_sha=commit_sha, parent_class=symbol.parent_class,
            is_public=symbol.is_public, is_entrypoint=symbol.is_entrypoint,
            complexity_bucket=symbol.complexity_bucket,
            line_count=symbol.line_count, language=language,
            signature_hash=symbol.content_hash,
        )

    def _index_file_record(
        self,
        repo_name: str,
        file_path: str,
        language: str,
        parsed: ParsedFile,
        commit_sha: str,
    ) -> None:
        """Index a file-level record in Pinecone (files namespace)."""
        ns = files_namespace(self.org_id, repo_name)
        store = self._get_pinecone(ns)
        vec_id = _file_pinecone_id(self.org_id, repo_name, file_path)

        embedding = self.embed_fn(parsed.searchable_text)

        metadata = {
            "org_id": self.org_id,
            "repo": repo_name,
            "file_path": file_path,
            "language": language,
            "symbol_count": len(parsed.symbols),
            "total_lines": parsed.total_lines,
            "commit_sha": commit_sha,
        }

        store.add(
            texts=[parsed.summary],
            embeddings=[embedding],
            ids=[vec_id],
            metadata=[metadata],
        )
        self._stats["files_indexed_pinecone"] += 1

    # ======================================================================
    # DIRECTORY INDEX
    # ======================================================================

    def _build_directory_index(
        self,
        repo_name: str,
        parsed_files: List[ParsedFile],
    ) -> None:
        """Build directory-level summaries from parsed files."""
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

            dir_data[dir_path]["languages"].add(pf.language)
            dir_data[dir_path]["file_count"] += 1
            dir_data[dir_path]["total_symbols"] += len(pf.symbols)
            dir_data[dir_path]["file_names"].append(Path(pf.file_path).name)

        ns = directories_namespace(self.org_id, repo_name)
        store = self._get_pinecone(ns)

        for dir_path, data in dir_data.items():
            languages = list(data["languages"])
            file_names = data["file_names"][:20]
            summary = f"Directory {dir_path}: {', '.join(file_names)}"
            if len(data["file_names"]) > 20:
                summary += f" and {len(data['file_names']) - 20} more files"

            # Pinecone
            vec_id = _dir_pinecone_id(self.org_id, repo_name, dir_path)
            embedding = self.embed_fn(summary)
            metadata = {
                "org_id": self.org_id,
                "repo": repo_name,
                "dir_path": dir_path,
                "languages": languages,
                "file_count": data["file_count"],
                "total_symbols": data["total_symbols"],
            }
            store.add(
                texts=[summary],
                embeddings=[embedding],
                ids=[vec_id],
                metadata=[metadata],
            )

            # Neo4j
            self.code_graph.upsert_directory(
                org_id=self.org_id, repo=repo_name,
                dir_path=dir_path, summary=summary,
                languages=languages,
                file_count=data["file_count"],
                total_symbols=data["total_symbols"],
            )

            self._stats["directories_indexed"] += 1

    # ======================================================================
    # CALL & IMPORT EDGES
    # ======================================================================

    def _build_call_edges(
        self,
        repo_name: str,
        parsed_files: List[ParsedFile],
        local_path: str,
    ) -> None:
        """Build CALLS edges in Neo4j from parsed call data."""
        all_symbols: Dict[str, str] = {}
        for pf in parsed_files:
            for sym in pf.symbols:
                all_symbols[sym.name] = pf.file_path
                all_symbols[sym.qualified_name] = pf.file_path

        for pf in parsed_files:
            for call in pf.calls:
                if call.callee_name not in all_symbols:
                    continue

                caller_file = pf.file_path
                callee_file = all_symbols[call.callee_name]

                try:
                    self.code_graph.add_calls_edge(
                        org_id=self.org_id, repo=repo_name,
                        caller_name=call.caller_name,
                        caller_file=caller_file,
                        callee_name=call.callee_name,
                        callee_file=callee_file,
                        is_direct=call.is_direct,
                    )
                    self._stats["call_edges_created"] += 1
                except Exception as e:
                    logger.debug("Failed to create call edge %s→%s: %s",
                                 call.caller_name, call.callee_name, e)

    def _build_import_edges(
        self,
        repo_name: str,
        parsed_files: List[ParsedFile],
        local_path: str,
    ) -> None:
        """Build IMPORTS edges in Neo4j from parsed import data."""
        known_files = {pf.file_path for pf in parsed_files}

        for pf in parsed_files:
            for imp in pf.imports:
                resolved = self._resolve_import_to_file(
                    imp.module, pf.file_path, known_files,
                )
                if resolved and resolved != pf.file_path:
                    try:
                        self.code_graph.add_imports_edge(
                            org_id=self.org_id, repo=repo_name,
                            importer_file=pf.file_path,
                            imported_file=resolved,
                            import_type="relative" if imp.is_relative else "direct",
                        )
                        self._stats["import_edges_created"] += 1
                    except Exception as e:
                        logger.debug("Failed to create import edge %s→%s: %s",
                                     pf.file_path, resolved, e)

    def _resolve_import_to_file(
        self,
        module: str,
        source_file: str,
        known_files: Set[str],
    ) -> Optional[str]:
        """Try to resolve a module import to a known file path.

        Handles common Python patterns:
          ``src.utils.retry`` → ``src/utils/retry.py``
          ``src.utils``       → ``src/utils/__init__.py``
        """
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

    # ======================================================================
    # DELETION
    # ======================================================================

    def _delete_file(self, repo_name: str, file_path: str) -> None:
        """Clean up all data for a deleted file from MongoDB + Pinecone + Neo4j."""
        logger.info("Deleting indexed data for %s", file_path)

        # Get symbols before deletion (need IDs for Pinecone cleanup)
        old_hashes = self.code_store.get_file_symbol_hashes(
            self.org_id, repo_name, file_path,
        )

        # MongoDB
        self.code_store.delete_symbols_for_file(self.org_id, repo_name, file_path)
        self.code_store.delete_file(self.org_id, repo_name, file_path)

        # Pinecone: delete symbol vectors
        sym_ns = symbols_namespace(self.org_id, repo_name)
        sym_store = self._get_pinecone(sym_ns)
        sym_ids = [
            _symbol_pinecone_id(self.org_id, repo_name, file_path, name)
            for name in old_hashes.keys()
        ]
        if sym_ids:
            try:
                sym_store.delete(ids=sym_ids)
            except Exception as e:
                logger.warning("Failed to delete symbol vectors: %s", e)

        # Pinecone: delete file vector
        file_ns = files_namespace(self.org_id, repo_name)
        file_store = self._get_pinecone(file_ns)
        file_vec_id = _file_pinecone_id(self.org_id, repo_name, file_path)
        try:
            file_store.delete(ids=[file_vec_id])
        except Exception as e:
            logger.warning("Failed to delete file vector: %s", e)

        self._stats["files_deleted"] += 1

    def _delete_symbol(
        self, repo_name: str, file_path: str, symbol_name: str,
    ) -> None:
        """Delete a single symbol from MongoDB + Pinecone."""
        self.code_store.delete_symbol(self.org_id, repo_name, file_path, symbol_name)

        sym_ns = symbols_namespace(self.org_id, repo_name)
        sym_store = self._get_pinecone(sym_ns)
        vec_id = _symbol_pinecone_id(self.org_id, repo_name, file_path, symbol_name)
        try:
            sym_store.delete(ids=[vec_id])
        except Exception as e:
            logger.warning("Failed to delete symbol vector %s: %s", symbol_name, e)

    # ======================================================================
    # HELPERS
    # ======================================================================

    def _get_all_source_files(self, local_path: str) -> List[str]:
        """Get all parseable source files in the repo."""
        all_files = list_all_files(local_path)
        return [f for f in all_files if not should_skip_file(f)]

    def close(self) -> None:
        self.code_store.close()
        self.code_graph.close()
