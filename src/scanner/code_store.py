"""
MongoDB Code Store — raw code storage and scan state management.

Collections:
  raw_symbols  — raw source code for each function/class (keyed by content hash)
  raw_files    — raw content for each file
  scan_runs    — tracks nightly scan state (last SHA, timestamps, stats)
  scanner_jobs — dashboard scan job state (persists across API restarts)
  scanner_user_repos — per-user repo rows for listing (shared index; key by user + org + repo)
  scanner_index_visibility — per org/repo: whether the shared catalog index may be queried by any scanner user
  scanner_community_stars — per-user stars on public catalog repos (reach / discovery)

The raw code is stored here so the retrieval pipeline can fetch exact
function bodies via ``get_symbol_code()`` without hitting the LLM or
loading entire files into context.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

logger = logging.getLogger("xmem.scanner.code_store")

BATCH_SIZE = 500


def _phase_rank(status: str) -> int:
    s = (status or "not_started").lower()
    order = (
        "not_started",
        "pending",
        "running",
        "failed",
        "complete",
    )
    try:
        return order.index(s)
    except ValueError:
        return 0


def _pick_best_scanner_job(
    jobs: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Prefer jobs closest to fully indexed (Phase 2 complete)."""
    if not jobs:
        return None
    return max(
        jobs,
        key=lambda j: (
            _phase_rank(j.get("phase1_status")),
            _phase_rank(j.get("phase2_status")),
            j.get("updated_at") or datetime.min.replace(tzinfo=timezone.utc),
        ),
    )


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
        self.scanner_jobs = self._db["scanner_jobs"]
        self.scanner_user_repos = self._db["scanner_user_repos"]
        self.scanner_index_visibility = self._db["scanner_index_visibility"]
        self.scanner_community_stars = self._db["scanner_community_stars"]

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

        self.scanner_jobs.create_index([("job_id", 1)], unique=True)
        self.scanner_jobs.create_index([("username", 1), ("updated_at", -1)])

        self.scanner_user_repos.create_index(
            [("username", 1), ("github_org", 1), ("repo", 1)],
            unique=True,
        )
        self.scanner_user_repos.create_index([("username", 1)])

        self.scanner_index_visibility.create_index(
            [("org_id", 1), ("repo", 1)],
            unique=True,
        )

        self.scanner_community_stars.create_index(
            [("username", 1), ("org_id", 1), ("repo", 1)],
            unique=True,
        )
        self.scanner_community_stars.create_index([("org_id", 1), ("repo", 1)])

    # ======================================================================
    # SCANNER DASHBOARD — job + per-user repo listing
    # ======================================================================

    def upsert_scanner_job(
        self,
        job_id: str,
        username: str,
        org: str,
        repo: str,
        branch: str,
        url: str,
        phase1_status: str,
        phase2_status: str,
        started_at: float,
        error: Optional[str] = None,
        phase1_result: Optional[Dict[str, Any]] = None,
        phase2_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist or update scanner dashboard job state."""
        doc: Dict[str, Any] = {
            "job_id": job_id,
            "username": username,
            "org": org,
            "repo": repo,
            "branch": branch,
            "url": url,
            "phase1_status": phase1_status,
            "phase2_status": phase2_status,
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc),
            "error": error,
        }
        if phase1_result is not None:
            doc["phase1_result"] = phase1_result
        if phase2_result is not None:
            doc["phase2_result"] = phase2_result

        self.scanner_jobs.update_one(
            {"job_id": job_id},
            {"$set": doc},
            upsert=True,
        )

    def get_scanner_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.scanner_jobs.find_one({"job_id": job_id})

    def list_scanner_jobs_for_user(self, username: str) -> List[Dict[str, Any]]:
        cursor = self.scanner_jobs.find({"username": username}).sort(
            "updated_at", -1,
        )
        return list(cursor)

    def upsert_user_repo_entry(
        self,
        username: str,
        github_org: str,
        repo: str,
        branch: str,
        last_seen_commit: Optional[str] = None,
    ) -> None:
        """Record that this user has this repo in their list (shared index)."""
        set_doc: Dict[str, Any] = {
            "username": username,
            "github_org": github_org,
            "repo": repo,
            "branch": branch,
            "updated_at": datetime.now(timezone.utc),
        }
        if last_seen_commit:
            set_doc["last_seen_commit"] = last_seen_commit

        self.scanner_user_repos.update_one(
            {
                "username": username,
                "github_org": github_org,
                "repo": repo,
            },
            {
                "$set": set_doc,
                "$setOnInsert": {
                    "created_at": datetime.now(timezone.utc),
                },
            },
            upsert=True,
        )

    def list_user_repos_for_user(self, username: str) -> List[Dict[str, Any]]:
        """All repo rows bookmarked for this user (for listing merge)."""
        cursor = self.scanner_user_repos.find({"username": username}).sort(
            "updated_at", -1,
        )
        return list(cursor)

    def get_scanner_index_visibility(self, org_id: str, repo: str) -> bool:
        """If unset, treat as public (legacy behavior)."""
        doc = self.scanner_index_visibility.find_one(
            {"org_id": org_id, "repo": repo},
            {"share_index_publicly": 1},
        )
        if not doc:
            return True
        return bool(doc.get("share_index_publicly", True))

    def get_scanner_index_visibility_batch(
        self, pairs: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], bool]:
        """Return share_index_publicly per (org, repo); default True when unset."""
        if not pairs:
            return {}
        or_clauses: List[Dict[str, str]] = [
            {"org_id": o, "repo": r} for o, r in pairs
        ]
        cursor = self.scanner_index_visibility.find(
            {"$or": or_clauses},
            {"org_id": 1, "repo": 1, "share_index_publicly": 1},
        )
        found = {
            (d["org_id"], d["repo"]): bool(d.get("share_index_publicly", True))
            for d in cursor
        }
        return {pair: found.get(pair, True) for pair in pairs}

    def set_scanner_index_visibility(
        self,
        org_id: str,
        repo: str,
        share_index_publicly: bool,
        set_by_username: str,
    ) -> None:
        self.scanner_index_visibility.update_one(
            {"org_id": org_id, "repo": repo},
            {
                "$set": {
                    "org_id": org_id,
                    "repo": repo,
                    "share_index_publicly": share_index_publicly,
                    "updated_at": datetime.now(timezone.utc),
                    "updated_by_username": set_by_username,
                },
            },
            upsert=True,
        )

    def list_public_scanner_indexes(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Repos explicitly marked community-shared (for discovery)."""
        cursor = (
            self.scanner_index_visibility.find(
                {"share_index_publicly": True},
                {"org_id": 1, "repo": 1, "updated_at": 1},
            )
            .sort("updated_at", -1)
            .limit(limit)
        )
        return list(cursor)

    def user_has_completed_scanner_job(
        self, username: str, org_id: str, repo: str,
    ) -> bool:
        """True if this username has a dashboard job with Phase 1 complete."""
        if not (username and username.strip()):
            return False
        job_id = f"{username}:{org_id}:{repo}"
        job = self.get_scanner_job(job_id)
        return bool(job and job.get("phase1_status") == "complete")

    # ======================================================================
    # COMMUNITY — stars + browsable public catalog
    # ======================================================================

    def set_community_star(
        self, username: str, org_id: str, repo: str, starred: bool,
    ) -> int:
        """Star or unstar. Returns total star count for this org/repo."""
        if not (username and username.strip()):
            return self.get_community_star_count(org_id, repo)
        filt = {"username": username, "org_id": org_id, "repo": repo}
        if starred:
            self.scanner_community_stars.update_one(
                filt,
                {
                    "$set": {
                        **filt,
                        "created_at": datetime.now(timezone.utc),
                    },
                },
                upsert=True,
            )
        else:
            self.scanner_community_stars.delete_one(filt)
        return self.get_community_star_count(org_id, repo)

    def get_community_star_count(self, org_id: str, repo: str) -> int:
        return self.scanner_community_stars.count_documents(
            {"org_id": org_id, "repo": repo},
        )

    def get_community_star_counts_batch(
        self, pairs: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], int]:
        if not pairs:
            return {}
        or_clauses = [{"org_id": o, "repo": r} for o, r in pairs]
        pipeline = [
            {"$match": {"$or": or_clauses}},
            {
                "$group": {
                    "_id": {"o": "$org_id", "r": "$repo"},
                    "n": {"$sum": 1},
                }
            },
        ]
        out: Dict[Tuple[str, str], int] = {}
        for row in self.scanner_community_stars.aggregate(pipeline):
            _id = row["_id"]
            out[(_id["o"], _id["r"])] = int(row["n"])
        return {pair: out.get(pair, 0) for pair in pairs}

    def user_starred_community_repos_batch(
        self, username: str, pairs: List[Tuple[str, str]],
    ) -> set:
        """Which (org, repo) pairs this user has starred."""
        if not pairs or not (username and username.strip()):
            return set()
        or_clauses = [
            {"org_id": o, "repo": r} for o, r in pairs
        ]
        cursor = self.scanner_community_stars.find(
            {"username": username, "$or": or_clauses},
            {"org_id": 1, "repo": 1},
        )
        return {(d["org_id"], d["repo"]) for d in cursor}

    def list_completed_scan_pairs(self, max_docs: int = 8000) -> List[Dict[str, Any]]:
        """scan_runs rows with a completed index (Phase 1 catalog)."""
        cursor = (
            self.scan_runs.find(
                {"status": "completed"},
                {"org_id": 1, "repo": 1, "completed_at": 1},
            )
            .sort("completed_at", -1)
            .limit(max_docs)
        )
        return list(cursor)

    def list_scanner_jobs_for_org_repo(
        self, org_id: str, repo: str, limit: int = 30,
    ) -> List[Dict[str, Any]]:
        return list(
            self.scanner_jobs.find({"org": org, "repo": repo})
            .sort("updated_at", -1)
            .limit(limit),
        )

    def get_catalog_repo_snapshot(
        self, username: str, org_id: str, repo: str,
    ) -> Dict[str, Any]:
        """Status for UI: prefer this user's job, else best global job for org/repo."""
        share = self.get_scanner_index_visibility(org_id, repo)
        job_id = f"{username}:{org_id}:{repo}"
        user_job = self.get_scanner_job(job_id)
        if user_job:
            elapsed = time.time() - float(user_job.get("started_at", time.time()))
            pr = user_job.get("phase1_result")
            p2 = user_job.get("phase2_result")
            out: Dict[str, Any] = {
                "status": "ok",
                "source": "user_job",
                "phase1_status": user_job.get("phase1_status", "not_started"),
                "phase2_status": user_job.get("phase2_status", "not_started"),
                "elapsed_seconds": round(elapsed, 1),
                "error": user_job.get("error"),
                "share_index_publicly": share,
            }
            if isinstance(pr, dict) and pr.get("stats"):
                out["stats"] = pr["stats"]
            if isinstance(p2, dict):
                out["phase2_stats"] = p2
            return out

        jobs = self.list_scanner_jobs_for_org_repo(org_id, repo)
        best = _pick_best_scanner_job(jobs)
        if best:
            pr = best.get("phase1_result")
            p2 = best.get("phase2_result")
            out = {
                "status": "ok",
                "source": "catalog",
                "phase1_status": best.get("phase1_status", "not_started"),
                "phase2_status": best.get("phase2_status", "not_started"),
                "error": best.get("error"),
                "share_index_publicly": share,
            }
            if isinstance(pr, dict) and pr.get("stats"):
                out["stats"] = pr["stats"]
            if isinstance(p2, dict):
                out["phase2_stats"] = p2
            return out

        scan = self.get_last_scan(org_id, repo)
        if scan and scan.get("status") == "completed":
            st = scan.get("stats") if isinstance(scan.get("stats"), dict) else None
            out = {
                "status": "ok",
                "source": "scan_runs",
                "phase1_status": "complete",
                "phase2_status": "complete",
                "share_index_publicly": share,
            }
            if st:
                out["stats"] = st
            return out

        return {
            "status": "ok",
            "source": "none",
            "phase1_status": "not_started",
            "phase2_status": "not_started",
            "share_index_publicly": share,
        }

    def list_community_catalog_page(
        self,
        username: str,
        q: str = "",
        sort: str = "stars",
        limit: int = 50,
        offset: int = 0,
        max_scan_pool: int = 5000,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Completed scans that are community-visible, with star counts."""
        rows = self.list_completed_scan_pairs(max_docs=max_scan_pool)
        seen: set[Tuple[str, str]] = set()
        unique_pairs: List[Tuple[str, str]] = []
        completed_at: Dict[Tuple[str, str], Any] = {}
        for r in rows:
            o = r.get("org_id")
            rp = r.get("repo")
            if not o or not rp:
                continue
            key = (o, rp)
            if key in seen:
                continue
            seen.add(key)
            unique_pairs.append(key)
            completed_at[key] = r.get("completed_at")

        vis = self.get_scanner_index_visibility_batch(unique_pairs)
        public_pairs = [p for p in unique_pairs if vis.get(p, True)]

        qn = (q or "").strip().lower()
        if qn:
            public_pairs = [
                p for p in public_pairs if qn in f"{p[0]}/{p[1]}".lower()
            ]

        star_counts = self.get_community_star_counts_batch(public_pairs)
        starred = self.user_starred_community_repos_batch(username, public_pairs)

        items: List[Dict[str, Any]] = []
        for p in public_pairs:
            o, rp = p
            items.append(
                {
                    "org": o,
                    "repo": rp,
                    "star_count": star_counts.get(p, 0),
                    "starred_by_me": p in starred,
                    "completed_at": completed_at.get(p),
                },
            )

        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if sort == "recent":
            items.sort(
                key=lambda x: x.get("completed_at") or epoch,
                reverse=True,
            )
        else:
            items.sort(
                key=lambda x: (x["star_count"], f"{x['org']}/{x['repo']}"),
                reverse=True,
            )

        total = len(items)
        page = items[offset : offset + limit]
        return page, total

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
        self.scanner_index_visibility.delete_one({"org_id": org_id, "repo": repo})
        self.scanner_community_stars.delete_many({"org_id": org_id, "repo": repo})
        return {"symbols_deleted": s, "files_deleted": f}

    def close(self) -> None:
        self._client.close()
