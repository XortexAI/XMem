"""
MongoDB Code Store — raw code storage and scan state management.

Collections:
  raw_symbols  — raw source code for each function/class (keyed by content hash)
  raw_files    — raw content for each file
  scan_runs    — tracks nightly scan state (last SHA, timestamps, stats)

The raw code is stored here so the retrieval pipeline can fetch exact
function bodies via ``get_symbol_code()`` without hitting the LLM or
loading entire files into context.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

logger = logging.getLogger("xmem.scanner.code_store")

BATCH_SIZE = 500


def _symbol_id(org_id: str, repo: str, file_path: str, symbol_name: str) -> str:
    """Deterministic ID for a symbol record."""
    key = f"{org_id}:{repo}:{file_path}:{symbol_name}"
    return hashlib.sha256(key.encode()).hexdigest()


def _file_id(org_id: str, repo: str, file_path: str) -> str:
    key = f"{org_id}:{repo}:{file_path}"
    return hashlib.sha256(key.encode()).hexdigest()


class CodeStore:
    """MongoDB-backed storage for raw code and scan state."""

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        database: str = "xmem",
    ) -> None:
        self._client = MongoClient(uri)
        self._db = self._client[database]

        self.symbols = self._db["raw_symbols"]
        self.files = self._db["raw_files"]
        self.scan_runs = self._db["scan_runs"]

        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.symbols.create_index(
            [("org_id", 1), ("repo", 1), ("file_path", 1), ("symbol_name", 1)],
            unique=True,
        )
        self.symbols.create_index([("org_id", 1), ("repo", 1), ("content_hash", 1)])

        self.files.create_index(
            [("org_id", 1), ("repo", 1), ("file_path", 1)],
            unique=True,
        )

        self.scan_runs.create_index(
            [("org_id", 1), ("repo", 1)],
            unique=True,
        )

    # ======================================================================
    # SYMBOL CRUD
    # ======================================================================

    def upsert_symbol(
        self,
        org_id: str,
        repo: str,
        file_path: str,
        symbol_name: str,
        symbol_type: str,
        language: str,
        raw_code: str,
        signature: str = "",
        docstring: str = "",
        summary: str = "",
        summary_source: str = "ast",
        start_line: int = 0,
        end_line: int = 0,
        line_count: int = 0,
        parent_class: Optional[str] = None,
        is_public: bool = True,
        is_entrypoint: bool = False,
        complexity_bucket: str = "medium",
        content_hash: str = "",
        commit_sha: str = "",
    ) -> str:
        """Upsert a symbol record. Returns the symbol _id."""
        doc_id = _symbol_id(org_id, repo, file_path, symbol_name)

        self.symbols.update_one(
            {"_id": doc_id},
            {"$set": {
                "org_id": org_id,
                "repo": repo,
                "file_path": file_path,
                "symbol_name": symbol_name,
                "symbol_type": symbol_type,
                "language": language,
                "raw_code": raw_code,
                "signature": signature,
                "docstring": docstring,
                "summary": summary,
                "summary_source": summary_source,
                "start_line": start_line,
                "end_line": end_line,
                "line_count": line_count,
                "parent_class": parent_class,
                "is_public": is_public,
                "is_entrypoint": is_entrypoint,
                "complexity_bucket": complexity_bucket,
                "content_hash": content_hash,
                "commit_sha": commit_sha,
                "indexed_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )
        return doc_id

    def upsert_symbols_bulk(
        self,
        org_id: str,
        repo: str,
        symbols: List[Dict[str, Any]],
    ) -> int:
        """Bulk upsert symbols. Returns count of upserted documents."""
        if not symbols:
            return 0

        ops = []
        for sym in symbols:
            doc_id = _symbol_id(
                org_id, repo,
                sym["file_path"], sym["symbol_name"],
            )
            ops.append(UpdateOne(
                {"_id": doc_id},
                {"$set": {
                    **sym,
                    "org_id": org_id,
                    "repo": repo,
                    "indexed_at": datetime.now(timezone.utc),
                }},
                upsert=True,
            ))

        total = 0
        for i in range(0, len(ops), BATCH_SIZE):
            batch = ops[i:i + BATCH_SIZE]
            try:
                result = self.symbols.bulk_write(batch, ordered=False)
                total += result.upserted_count + result.modified_count
            except BulkWriteError as e:
                total += e.details.get("nUpserted", 0) + e.details.get("nModified", 0)
                logger.warning("Bulk write had %d errors", len(e.details.get("writeErrors", [])))

        return total

    def get_symbol_code(
        self, org_id: str, repo: str, file_path: str, symbol_name: str,
    ) -> Optional[str]:
        """Retrieve the raw code for a specific symbol."""
        doc_id = _symbol_id(org_id, repo, file_path, symbol_name)
        doc = self.symbols.find_one({"_id": doc_id}, {"raw_code": 1})
        return doc["raw_code"] if doc else None

    def get_symbol_hash(
        self, org_id: str, repo: str, file_path: str, symbol_name: str,
    ) -> Optional[str]:
        """Get the content hash for change detection."""
        doc_id = _symbol_id(org_id, repo, file_path, symbol_name)
        doc = self.symbols.find_one({"_id": doc_id}, {"content_hash": 1})
        return doc["content_hash"] if doc else None

    def get_file_symbol_hashes(
        self, org_id: str, repo: str, file_path: str,
    ) -> Dict[str, str]:
        """Get all symbol content hashes for a file (for incremental scanning)."""
        cursor = self.symbols.find(
            {"org_id": org_id, "repo": repo, "file_path": file_path},
            {"symbol_name": 1, "content_hash": 1},
        )
        return {doc["symbol_name"]: doc["content_hash"] for doc in cursor}

    def delete_symbols_for_file(
        self, org_id: str, repo: str, file_path: str,
    ) -> int:
        """Delete all symbols belonging to a file (used on file deletion)."""
        result = self.symbols.delete_many({
            "org_id": org_id, "repo": repo, "file_path": file_path,
        })
        return result.deleted_count

    def delete_symbol(
        self, org_id: str, repo: str, file_path: str, symbol_name: str,
    ) -> bool:
        doc_id = _symbol_id(org_id, repo, file_path, symbol_name)
        result = self.symbols.delete_one({"_id": doc_id})
        return result.deleted_count > 0

    # ======================================================================
    # FILE CRUD
    # ======================================================================

    def upsert_file(
        self,
        org_id: str,
        repo: str,
        file_path: str,
        language: str,
        raw_content: str,
        summary: str = "",
        summary_source: str = "ast",
        total_lines: int = 0,
        content_hash: str = "",
        commit_sha: str = "",
    ) -> str:
        doc_id = _file_id(org_id, repo, file_path)
        self.files.update_one(
            {"_id": doc_id},
            {"$set": {
                "org_id": org_id,
                "repo": repo,
                "file_path": file_path,
                "language": language,
                "raw_content": raw_content,
                "summary": summary,
                "summary_source": summary_source,
                "total_lines": total_lines,
                "content_hash": content_hash,
                "commit_sha": commit_sha,
                "indexed_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )
        return doc_id

    def get_file_content(
        self, org_id: str, repo: str, file_path: str,
    ) -> Optional[str]:
        doc_id = _file_id(org_id, repo, file_path)
        doc = self.files.find_one({"_id": doc_id}, {"raw_content": 1})
        return doc["raw_content"] if doc else None

    def get_file_hash(
        self, org_id: str, repo: str, file_path: str,
    ) -> Optional[str]:
        doc_id = _file_id(org_id, repo, file_path)
        doc = self.files.find_one({"_id": doc_id}, {"content_hash": 1})
        return doc["content_hash"] if doc else None

    def delete_file(self, org_id: str, repo: str, file_path: str) -> bool:
        doc_id = _file_id(org_id, repo, file_path)
        result = self.files.delete_one({"_id": doc_id})
        return result.deleted_count > 0

    # ======================================================================
    # SCAN STATE — Tracks nightly scan progress
    # ======================================================================

    def get_last_scan(self, org_id: str, repo: str) -> Optional[Dict[str, Any]]:
        """Get the last scan run for a repo."""
        return self.scan_runs.find_one({"org_id": org_id, "repo": repo})

    def get_last_commit_sha(self, org_id: str, repo: str) -> Optional[str]:
        doc = self.get_last_scan(org_id, repo)
        return doc.get("last_commit_sha") if doc else None

    def start_scan(self, org_id: str, repo: str, commit_sha: str) -> None:
        """Mark a scan as started."""
        self.scan_runs.update_one(
            {"org_id": org_id, "repo": repo},
            {"$set": {
                "org_id": org_id,
                "repo": repo,
                "current_commit_sha": commit_sha,
                "status": "running",
                "started_at": datetime.now(timezone.utc),
                "error": None,
            }},
            upsert=True,
        )

    def complete_scan(
        self,
        org_id: str,
        repo: str,
        commit_sha: str,
        stats: Dict[str, int],
        duration_seconds: float,
    ) -> None:
        """Mark a scan as completed successfully."""
        self.scan_runs.update_one(
            {"org_id": org_id, "repo": repo},
            {"$set": {
                "last_commit_sha": commit_sha,
                "status": "completed",
                "completed_at": datetime.now(timezone.utc),
                "duration_seconds": duration_seconds,
                "stats": stats,
            }},
        )

    def fail_scan(self, org_id: str, repo: str, error: str) -> None:
        """Mark a scan as failed."""
        self.scan_runs.update_one(
            {"org_id": org_id, "repo": repo},
            {"$set": {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc),
                "error": error,
            }},
        )

    # ======================================================================
    # ENRICHMENT — querying and updating for Phase 2 LLM enrichment
    # ======================================================================

    def get_unenriched_symbols(
        self,
        org_id: str,
        repo: str,
        limit: int = 100,
        prioritize_public: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get symbols that still have AST-generated summaries (not yet LLM-enriched).

        Prioritizes public/entrypoint symbols first so the most user-facing
        code gets enriched before internal helpers.
        """
        query = {
            "org_id": org_id,
            "repo": repo,
            "summary_source": {"$ne": "llm"},
        }
        sort_key: List = []
        if prioritize_public:
            sort_key = [
                ("is_entrypoint", -1),
                ("is_public", -1),
                ("line_count", -1),
            ]
        else:
            sort_key = [("indexed_at", 1)]

        cursor = self.symbols.find(query).sort(sort_key).limit(limit)
        return list(cursor)

    def get_unenriched_files(
        self,
        org_id: str,
        repo: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get files that still have AST-generated summaries."""
        query = {
            "org_id": org_id,
            "repo": repo,
            "summary_source": {"$ne": "llm"},
        }
        cursor = self.files.find(query).sort("total_lines", -1).limit(limit)
        return list(cursor)

    def update_symbol_summary(
        self,
        doc_id: str,
        summary: str,
        summary_source: str = "llm",
    ) -> bool:
        """Update a symbol's summary after LLM enrichment."""
        result = self.symbols.update_one(
            {"_id": doc_id},
            {"$set": {
                "summary": summary,
                "summary_source": summary_source,
                "enriched_at": datetime.now(timezone.utc),
            }},
        )
        return result.modified_count > 0

    def update_file_summary(
        self,
        doc_id: str,
        summary: str,
        summary_source: str = "llm",
    ) -> bool:
        """Update a file's summary after LLM enrichment."""
        result = self.files.update_one(
            {"_id": doc_id},
            {"$set": {
                "summary": summary,
                "summary_source": summary_source,
                "enriched_at": datetime.now(timezone.utc),
            }},
        )
        return result.modified_count > 0

    def count_unenriched(self, org_id: str, repo: str) -> Dict[str, int]:
        """Count how many symbols/files still need LLM enrichment."""
        sym_count = self.symbols.count_documents({
            "org_id": org_id, "repo": repo,
            "summary_source": {"$ne": "llm"},
        })
        file_count = self.files.count_documents({
            "org_id": org_id, "repo": repo,
            "summary_source": {"$ne": "llm"},
        })
        return {"symbols": sym_count, "files": file_count}

    # ======================================================================
    # STATS
    # ======================================================================

    def get_repo_stats(self, org_id: str, repo: str) -> Dict[str, int]:
        symbol_count = self.symbols.count_documents({"org_id": org_id, "repo": repo})
        file_count = self.files.count_documents({"org_id": org_id, "repo": repo})
        return {"symbols": symbol_count, "files": file_count}

    # ======================================================================
    # CLEANUP
    # ======================================================================

    def delete_repo(self, org_id: str, repo: str) -> Dict[str, int]:
        """Delete all data for a repository."""
        s = self.symbols.delete_many({"org_id": org_id, "repo": repo}).deleted_count
        f = self.files.delete_many({"org_id": org_id, "repo": repo}).deleted_count
        self.scan_runs.delete_one({"org_id": org_id, "repo": repo})
        return {"symbols_deleted": s, "files_deleted": f}

    def close(self) -> None:
        self._client.close()
