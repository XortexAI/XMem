"""
Enricher — Phase 2 async LLM summary enrichment worker.

Runs AFTER the scanner (Phase 1) has indexed all code with deterministic
AST-based summaries.  The enricher picks up records where
``summary_source != "llm"``, sends the raw code to a cheap/fast LLM for a
richer natural-language summary, then propagates the enriched summary to
MongoDB, Pinecone, and Neo4j.

Design principles:
  - **Resumable**: Uses ``summary_source`` as a cursor — if interrupted,
    re-running picks up right where it left off.
  - **Prioritised**: Public symbols and entrypoints are enriched first so the
    most user-facing code gets better summaries earliest.
  - **Rate-limit safe**: Configurable delay between LLM calls with exponential
    backoff on transient errors.
  - **Batched writes**: Groups Pinecone and MongoDB updates.
  - **No destructive rewrites**: Only touches the ``summary`` and
    ``summary_source`` fields; raw code and structural data are untouched.

Usage:
    python -m src.scanner.runner --org zinnia --repo payment-service --enrich
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from src.scanner.code_store import CodeStore
from src.graph.code_graph_client import CodeGraphClient
from src.storage.pinecone import PineconeVectorStore
from src.schemas.code import symbols_namespace, files_namespace
from src.config import settings

logger = logging.getLogger("xmem.scanner.enricher")

SYMBOL_BATCH_SIZE = 50
FILE_BATCH_SIZE = 20
DEFAULT_DELAY_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 60


# ---------------------------------------------------------------------------
# Prompts — kept minimal to save tokens
# ---------------------------------------------------------------------------

_SYMBOL_PROMPT = """\
You are a code documentation expert. Given a code symbol (function, method, \
or class), write a concise 1-2 sentence summary that describes:
1. WHAT it does (purpose/behavior)
2. WHY it matters (business context if obvious)

Rules:
- Be specific — mention key inputs, outputs, side-effects.
- Do NOT repeat the function signature or parameter names literally.
- Do NOT use phrases like "This function..." — start directly with a verb.
- Max 200 characters.

---
Symbol: {qualified_name}
Type: {symbol_type}
Signature: {signature}
Docstring: {docstring}
Code:
```{language}
{raw_code}
```

Summary:"""

_FILE_PROMPT = """\
You are a code documentation expert. Given the symbols defined in a source \
file, write a concise 1-2 sentence summary that describes the file's purpose \
and the key capabilities it provides.

Rules:
- Be specific about domain/functionality.
- Do NOT list every symbol — highlight the most important ones.
- Max 250 characters.

---
File: {file_path}
Language: {language}
Symbols ({symbol_count}): {symbol_list}

Summary:"""


# ---------------------------------------------------------------------------
# Core enricher
# ---------------------------------------------------------------------------


def _symbol_pinecone_id(org_id: str, repo: str, file_path: str, symbol_name: str) -> str:
    key = f"{org_id}:{repo}:{file_path}:{symbol_name}"
    return hashlib.sha256(key.encode()).hexdigest()


def _file_pinecone_id(org_id: str, repo: str, file_path: str) -> str:
    key = f"{org_id}:{repo}:{file_path}"
    return hashlib.sha256(key.encode()).hexdigest()


class Enricher:
    """Phase 2 LLM enrichment worker for code summaries."""

    def __init__(
        self,
        org_id: str,
        llm_fn: Optional[Callable[[str], str]] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        code_store: Optional[CodeStore] = None,
        code_graph: Optional[CodeGraphClient] = None,
        delay: float = DEFAULT_DELAY_SECONDS,
        max_symbols: int = 0,
        max_files: int = 0,
    ) -> None:
        """
        Args:
            org_id: Organisation identifier.
            llm_fn: ``(prompt: str) -> str`` callable. If *None*, a default
                     Gemini Flash model is used.
            embed_fn: ``(text: str) -> List[float]`` callable for re-embedding.
            code_store: MongoDB code store. Created from settings if *None*.
            code_graph: Neo4j graph client. Created from settings if *None*.
            delay: Seconds to sleep between LLM calls (rate-limit safety).
            max_symbols: Cap on symbols to enrich per run (0 = unlimited).
            max_files: Cap on files to enrich per run (0 = unlimited).
        """
        self.org_id = org_id
        self.delay = delay
        self.max_symbols = max_symbols
        self.max_files = max_files

        # LLM
        if llm_fn is None:
            self._llm_fn = self._build_default_llm()
        else:
            self._llm_fn = llm_fn

        # Embeddings
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
        else:
            self.code_graph = code_graph

        # Pinecone (lazily created per namespace)
        self._pinecone_stores: Dict[str, PineconeVectorStore] = {}

        self._stats: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Default LLM factory
    # ------------------------------------------------------------------

    @staticmethod
    def _build_default_llm() -> Callable[[str], str]:
        """Build a lightweight LLM caller using the cheapest available model."""
        from src.models.gemini import build_gemini_model

        model = build_gemini_model(
            model_name=settings.code_model or "gemini-2.0-flash-lite",
            temperature=0.2,
        )

        def call(prompt: str) -> str:
            response = model.invoke([{"role": "user", "content": prompt}])
            content = response.content
            if isinstance(content, list):
                return "\n".join(str(c) for c in content)
            return str(content).strip()

        return call

    # ------------------------------------------------------------------
    # Pinecone helper
    # ------------------------------------------------------------------

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

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def enrich_repo(
        self,
        repo_name: str,
    ) -> Dict[str, Any]:
        """Run enrichment for a single repository.

        Returns dict with stats and timing.
        """
        start_time = time.time()
        self._stats = defaultdict(int)

        remaining = self.code_store.count_unenriched(self.org_id, repo_name)
        logger.info("=" * 70)
        logger.info(
            "ENRICHMENT START: %s/%s — %d symbols, %d files pending",
            self.org_id, repo_name,
            remaining["symbols"], remaining["files"],
        )
        logger.info("=" * 70)

        self._enrich_symbols(repo_name)
        self._enrich_files(repo_name)

        duration = time.time() - start_time

        logger.info("=" * 70)
        logger.info("ENRICHMENT COMPLETE: %s/%s in %.1fs", self.org_id, repo_name, duration)
        for k, v in sorted(self._stats.items()):
            logger.info("  %s: %d", k, v)
        logger.info("=" * 70)

        return {
            "org_id": self.org_id,
            "repo": repo_name,
            "duration_seconds": round(duration, 1),
            "stats": dict(self._stats),
        }

    # ==================================================================
    # SYMBOL ENRICHMENT
    # ==================================================================

    def _enrich_symbols(self, repo_name: str) -> None:
        """Enrich all un-enriched symbols for a repo."""
        total_done = 0

        while True:
            if self.max_symbols and total_done >= self.max_symbols:
                logger.info("Reached max_symbols cap (%d)", self.max_symbols)
                break

            batch_limit = SYMBOL_BATCH_SIZE
            if self.max_symbols:
                batch_limit = min(batch_limit, self.max_symbols - total_done)

            batch = self.code_store.get_unenriched_symbols(
                self.org_id, repo_name, limit=batch_limit,
            )
            if not batch:
                break

            for doc in batch:
                try:
                    self._enrich_one_symbol(repo_name, doc)
                    total_done += 1
                    self._stats["symbols_enriched"] += 1
                except Exception as e:
                    logger.error(
                        "Failed to enrich symbol %s: %s",
                        doc.get("symbol_name", "?"), e,
                    )
                    self._stats["symbols_failed"] += 1
                    self._handle_backoff(e)

    def _enrich_one_symbol(self, repo_name: str, doc: Dict[str, Any]) -> None:
        """Enrich a single symbol: LLM call → MongoDB → Pinecone → Neo4j."""
        symbol_name = doc["symbol_name"]
        file_path = doc["file_path"]
        language = doc.get("language", "python")
        raw_code = doc.get("raw_code", "")

        # Truncate very large functions to save tokens
        if len(raw_code) > 4000:
            raw_code = raw_code[:4000] + "\n# ... (truncated)"

        prompt = _SYMBOL_PROMPT.format(
            qualified_name=symbol_name,
            symbol_type=doc.get("symbol_type", "function"),
            signature=doc.get("signature", ""),
            docstring=(doc.get("docstring", "") or "")[:500],
            language=language,
            raw_code=raw_code,
        )

        summary = self._call_llm_safe(prompt)
        if not summary:
            return

        summary = summary.strip().strip('"').strip("'")
        if len(summary) > 300:
            summary = summary[:297] + "..."

        doc_id = doc["_id"]

        # 1. MongoDB
        self.code_store.update_symbol_summary(doc_id, summary, summary_source="llm")

        # 2. Pinecone — re-embed and upsert
        ns = symbols_namespace(self.org_id, repo_name)
        store = self._get_pinecone(ns)
        vec_id = _symbol_pinecone_id(self.org_id, repo_name, file_path, symbol_name)

        searchable = f"{symbol_name} {doc.get('signature', '')} {summary}"
        embedding = self.embed_fn(searchable)

        store.add(
            texts=[summary],
            embeddings=[embedding],
            ids=[vec_id],
            metadata=[{
                "org_id": self.org_id,
                "repo": repo_name,
                "file_path": file_path,
                "symbol_name": symbol_name,
                "symbol_type": doc.get("symbol_type", "function"),
                "language": language,
                "signature": (doc.get("signature", "") or "")[:500],
                "is_public": doc.get("is_public", True),
                "is_entrypoint": doc.get("is_entrypoint", False),
                "complexity_bucket": doc.get("complexity_bucket", "medium"),
                "line_count": doc.get("line_count", 0),
                "start_line": doc.get("start_line", 0),
                "end_line": doc.get("end_line", 0),
                "summary_source": "llm",
            }],
        )

        # 3. Neo4j — update summary on the Symbol node
        try:
            self.code_graph.upsert_symbol(
                org_id=self.org_id, repo=repo_name,
                file_path=file_path, symbol_name=symbol_name,
                symbol_type=doc.get("symbol_type", "function"),
                summary=summary,
                signature=doc.get("signature", ""),
                docstring=doc.get("docstring", ""),
                start_line=doc.get("start_line", 0),
                end_line=doc.get("end_line", 0),
                commit_sha=doc.get("commit_sha", ""),
                parent_class=doc.get("parent_class"),
                is_public=doc.get("is_public", True),
                is_entrypoint=doc.get("is_entrypoint", False),
                complexity_bucket=doc.get("complexity_bucket", "medium"),
                line_count=doc.get("line_count", 0),
                language=language,
                signature_hash=doc.get("content_hash", ""),
            )
        except Exception as e:
            logger.warning("Neo4j update failed for %s: %s", symbol_name, e)

        logger.debug("Enriched symbol: %s → %s", symbol_name, summary[:80])
        time.sleep(self.delay)

    # ==================================================================
    # FILE ENRICHMENT
    # ==================================================================

    def _enrich_files(self, repo_name: str) -> None:
        """Enrich all un-enriched files for a repo."""
        total_done = 0

        while True:
            if self.max_files and total_done >= self.max_files:
                logger.info("Reached max_files cap (%d)", self.max_files)
                break

            batch_limit = FILE_BATCH_SIZE
            if self.max_files:
                batch_limit = min(batch_limit, self.max_files - total_done)

            batch = self.code_store.get_unenriched_files(
                self.org_id, repo_name, limit=batch_limit,
            )
            if not batch:
                break

            for doc in batch:
                try:
                    self._enrich_one_file(repo_name, doc)
                    total_done += 1
                    self._stats["files_enriched"] += 1
                except Exception as e:
                    logger.error(
                        "Failed to enrich file %s: %s",
                        doc.get("file_path", "?"), e,
                    )
                    self._stats["files_failed"] += 1
                    self._handle_backoff(e)

    def _enrich_one_file(self, repo_name: str, doc: Dict[str, Any]) -> None:
        """Enrich a single file: LLM call → MongoDB → Pinecone → Neo4j."""
        file_path = doc["file_path"]
        language = doc.get("language", "python")

        # Get symbols in this file from MongoDB for context
        sym_cursor = self.code_store.symbols.find(
            {"org_id": self.org_id, "repo": repo_name, "file_path": file_path},
            {"symbol_name": 1, "symbol_type": 1, "signature": 1, "docstring": 1},
        )
        symbols = list(sym_cursor)
        symbol_names = [s.get("symbol_name", "") for s in symbols]

        symbol_list = ", ".join(
            f"{s.get('symbol_type', '?')} {s.get('symbol_name', '?')}"
            for s in symbols[:30]
        )
        if len(symbols) > 30:
            symbol_list += f" and {len(symbols) - 30} more"

        prompt = _FILE_PROMPT.format(
            file_path=file_path,
            language=language,
            symbol_count=len(symbols),
            symbol_list=symbol_list,
        )

        summary = self._call_llm_safe(prompt)
        if not summary:
            return

        summary = summary.strip().strip('"').strip("'")
        if len(summary) > 350:
            summary = summary[:347] + "..."

        doc_id = doc["_id"]

        # 1. MongoDB
        self.code_store.update_file_summary(doc_id, summary, summary_source="llm")

        # 2. Pinecone
        ns = files_namespace(self.org_id, repo_name)
        store = self._get_pinecone(ns)
        vec_id = _file_pinecone_id(self.org_id, repo_name, file_path)

        searchable = f"{file_path} {summary}"
        embedding = self.embed_fn(searchable)

        store.add(
            texts=[summary],
            embeddings=[embedding],
            ids=[vec_id],
            metadata=[{
                "org_id": self.org_id,
                "repo": repo_name,
                "file_path": file_path,
                "language": language,
                "symbol_count": len(symbols),
                "total_lines": doc.get("total_lines", 0),
                "commit_sha": doc.get("commit_sha", ""),
                "summary_source": "llm",
            }],
        )

        # 3. Neo4j
        try:
            self.code_graph.upsert_file(
                org_id=self.org_id, repo=repo_name,
                file_path=file_path, language=language,
                summary=summary,
                symbol_count=len(symbols),
                symbol_names=symbol_names[:50],
                total_lines=doc.get("total_lines", 0),
                commit_sha=doc.get("commit_sha", ""),
            )
        except Exception as e:
            logger.warning("Neo4j update failed for %s: %s", file_path, e)

        logger.debug("Enriched file: %s → %s", file_path, summary[:80])
        time.sleep(self.delay)

    # ==================================================================
    # LLM + error handling
    # ==================================================================

    def _call_llm_safe(self, prompt: str) -> Optional[str]:
        """Call LLM with error handling. Returns None on failure."""
        try:
            return self._llm_fn(prompt)
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            self._stats["llm_errors"] += 1
            return None

    def _handle_backoff(self, error: Exception) -> None:
        """Exponential backoff on repeated errors."""
        consecutive = self._stats.get("_consecutive_errors", 0) + 1
        self._stats["_consecutive_errors"] = consecutive

        backoff = min(self.delay * (2 ** consecutive), MAX_BACKOFF_SECONDS)
        logger.warning("Backing off %.1fs after %d consecutive errors", backoff, consecutive)
        time.sleep(backoff)

    # ==================================================================
    # CLEANUP
    # ==================================================================

    def close(self) -> None:
        self.code_store.close()
        self.code_graph.close()
