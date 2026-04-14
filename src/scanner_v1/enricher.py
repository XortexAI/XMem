"""
EnricherV1 — phase 2 async worker that upgrades AST summaries to LLM
summaries.

Critical difference from v0: the enricher only touches the SUMMARY
lane.  The code lane was built from raw source at scan time and is
considered permanent — the whole point of having a separate lane is
that it stays grounded in code vocabulary no matter what the LLM
writes about the function.

Flow:
  1. Query the store for Symbol / File nodes where summary_source != 'llm'.
  2. Prioritize: is_public first, then is_entrypoint, then the rest.
  3. For each symbol:
       a. build the LLM prompt from (qualified_name, signature, docstring,
          raw_code).
       b. LLM call → new summary text.
       c. embedder.re_embed_summary(new_text) → new vector.
       d. store.update_symbol_summary(new_text, 'llm', new_vector).
  4. Same shape for files.
  5. Rate-limit + exponential backoff on transient errors.
  6. Batched writes — flush every N updates.

Resumable: because `summary_source` is the cursor, killing and
restarting the worker picks up exactly where it left off.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from src.scanner_v1 import schemas as S
from src.scanner_v1.store import CodeStoreV1
from src.scanner_v1.embedder import Embedder

logger = logging.getLogger("xmem.scanner_v1.enricher")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

SYMBOL_BATCH_SIZE     = 50
FILE_BATCH_SIZE       = 20
DEFAULT_DELAY_SECONDS = 0.5
MAX_BACKOFF_SECONDS   = 60

# Maximum raw_code length included in the LLM prompt.  Keeps token
# budgets predictable — the enricher prompt is the single largest
# per-call cost.
MAX_PROMPT_CODE_CHARS = 12_000


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# Kept minimal to save tokens.  The v1 prompts are almost identical to
# v0 — the only philosophical change is that this output is used as
# display text + summary-lane embedding, NOT as the primary recall
# signal (the code lane handles that).
# ---------------------------------------------------------------------------

_SYMBOL_PROMPT = """\
You are a code documentation expert. Given a code symbol, write a
concise 1-2 sentence summary describing:
  1. WHAT it does.
  2. WHY it matters (business context if obvious).

Rules:
  - Be specific about inputs, outputs, side-effects.
  - Do NOT repeat the signature literally.
  - Start with a verb, not "This function...".
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
You are a code documentation expert. Given the symbols defined in a
source file, write a concise 1-2 sentence summary describing the
file's purpose and the key capabilities it provides.

Rules:
  - Be specific about domain / functionality.
  - Do NOT list every symbol — highlight the most important ones.
  - Max 250 characters.

---
File: {file_path}
Language: {language}
Symbols ({symbol_count}): {symbol_list}

Summary:"""


# ---------------------------------------------------------------------------
# EnricherV1
# ---------------------------------------------------------------------------

class EnricherV1:
    """Async LLM-driven summary enricher, v1."""

    def __init__(
        self,
        org_id: str,
        store: CodeStoreV1,
        embedder: Embedder,
        llm_call: Callable[[str], str],
        delay: float = DEFAULT_DELAY_SECONDS,
        max_symbols: int = 0,
        max_files: int = 0,
    ) -> None:
        # llm_call is a thin wrapper around the model of choice:
        # prompt in, summary string out.  Keeping it as a plain
        # callable makes unit testing and model swapping trivial.
        self.org_id = org_id
        self.store = store
        self.embedder = embedder
        self.llm_call = llm_call
        self.delay_seconds = delay
        self.max_symbols = max_symbols   # 0 = unlimited
        self.max_files = max_files       # 0 = unlimited

    def close(self) -> None:
        """Close the underlying store.  Safe to call multiple times."""
        self.store.close()

    # =================================================================
    # PUBLIC ENTRY
    # =================================================================

    def enrich_repo(
        self,
        repo: str,
        symbols: bool = True,
        files: bool = True,
    ) -> Dict[str, int]:
        """Run enrichment for a whole repo.
        Returns a stats dict."""
        stats: Dict[str, int] = {}
        t0 = time.time()

        if symbols:
            stats["symbols_enriched"] = self._enrich_symbols(repo)
        if files:
            stats["files_enriched"] = self._enrich_files(repo)

        stats["duration_seconds"] = round(time.time() - t0, 1)
        logger.info(
            "Enrichment complete for %s/%s: %s",
            self.org_id, repo, stats,
        )
        return stats

    # =================================================================
    # SYMBOLS
    # =================================================================

    def _enrich_symbols(self, repo: str) -> int:
        """Loop over un-enriched symbols, enrich in batches.
        Returns the number of symbols updated."""
        rows = self._fetch_unenriched_symbols(repo)
        if not rows:
            logger.info("No un-enriched symbols for %s/%s", self.org_id, repo)
            return 0

        total = len(rows)
        cap = self.max_symbols if self.max_symbols > 0 else total
        rows = rows[:cap]
        logger.info(
            "Enriching %d / %d symbols for %s/%s",
            len(rows), total, self.org_id, repo,
        )

        updated = 0
        for i, row in enumerate(rows, 1):
            qname = row.get("qualified_name", "?")
            try:
                prompt = self._build_symbol_prompt(row)
                new_summary = self._llm_with_backoff(prompt)
                self._update_symbol(row, new_summary)
                updated += 1
                if i % 25 == 0:
                    logger.info(
                        "  [symbols] %d / %d enriched", i, len(rows),
                    )
            except Exception as e:
                logger.warning(
                    "  Skipping symbol %s: %s", qname, e,
                )
            self._sleep_rate_limit()

        logger.info("Enriched %d symbols for %s/%s", updated, self.org_id, repo)
        return updated

    def _fetch_unenriched_symbols(self, repo: str) -> List[Dict[str, Any]]:
        """Query the store for Symbol nodes where summary_source != 'llm'.

        Priority order: is_public DESC, is_entrypoint DESC, then by
        qualified_name for determinism.
        """
        cypher = f"""
        MATCH (s:{S.LABEL_SYMBOL} {{org_id: $org_id, repo: $repo}})
        WHERE s.summary_source <> '{S.SUMMARY_SOURCE_LLM}'
           OR s.summary_source IS NULL
        RETURN s.org_id          AS org_id,
               s.repo            AS repo,
               s.file_path       AS file_path,
               s.qualified_name  AS qualified_name,
               s.symbol_type     AS symbol_type,
               s.signature       AS signature,
               s.docstring       AS docstring,
               s.summary         AS summary,
               s.raw_code        AS raw_code,
               s.language        AS language,
               s.is_public       AS is_public,
               s.is_entrypoint   AS is_entrypoint
        ORDER BY s.is_public DESC, s.is_entrypoint DESC, s.qualified_name
        """
        with self.store._session() as session:
            result = session.run(
                cypher, org_id=self.org_id, repo=repo,
            )
            return [r.data() for r in result]

    def _build_symbol_prompt(self, row: Dict[str, Any]) -> str:
        """Fill _SYMBOL_PROMPT from a Symbol row.
        Truncates raw_code to keep the prompt within budget."""
        raw_code = row.get("raw_code", "") or ""
        if len(raw_code) > MAX_PROMPT_CODE_CHARS:
            half = MAX_PROMPT_CODE_CHARS // 2
            raw_code = raw_code[:half] + "\n... (truncated) ...\n" + raw_code[-half:]

        return _SYMBOL_PROMPT.format(
            qualified_name=row.get("qualified_name", ""),
            symbol_type=row.get("symbol_type", "function"),
            signature=row.get("signature", ""),
            docstring=row.get("docstring", "") or "(none)",
            language=row.get("language", ""),
            raw_code=raw_code,
        )

    def _update_symbol(
        self,
        row: Dict[str, Any],
        new_summary: str,
    ) -> None:
        """Re-embed summary lane and write back via
        store.update_symbol_summary.  Never touches code_embedding."""
        new_vector = self.embedder.re_embed_summary(new_summary)
        self.store.update_symbol_summary(
            org_id=row["org_id"],
            repo=row["repo"],
            file_path=row["file_path"],
            symbol_name=row["qualified_name"],
            summary=new_summary,
            summary_source=S.SUMMARY_SOURCE_LLM,
            summary_embedding=new_vector,
        )

    # =================================================================
    # FILES
    # =================================================================

    def _enrich_files(self, repo: str) -> int:
        """Loop over un-enriched files, enrich one at a time.
        Returns the number of files updated."""
        rows = self._fetch_unenriched_files(repo)
        if not rows:
            logger.info("No un-enriched files for %s/%s", self.org_id, repo)
            return 0

        total = len(rows)
        cap = self.max_files if self.max_files > 0 else total
        rows = rows[:cap]
        logger.info(
            "Enriching %d / %d files for %s/%s",
            len(rows), total, self.org_id, repo,
        )

        updated = 0
        for i, row in enumerate(rows, 1):
            fpath = row.get("file_path", "?")
            try:
                prompt = self._build_file_prompt(row)
                new_summary = self._llm_with_backoff(prompt)
                self._update_file(row, new_summary)
                updated += 1
                if i % 10 == 0:
                    logger.info(
                        "  [files] %d / %d enriched", i, len(rows),
                    )
            except Exception as e:
                logger.warning("  Skipping file %s: %s", fpath, e)
            self._sleep_rate_limit()

        logger.info("Enriched %d files for %s/%s", updated, self.org_id, repo)
        return updated

    def _fetch_unenriched_files(self, repo: str) -> List[Dict[str, Any]]:
        """Query for File nodes where summary_source != 'llm'."""
        cypher = f"""
        MATCH (f:{S.LABEL_FILE} {{org_id: $org_id, repo: $repo}})
        WHERE f.summary_source <> '{S.SUMMARY_SOURCE_LLM}'
           OR f.summary_source IS NULL
        RETURN f.org_id        AS org_id,
               f.repo          AS repo,
               f.file_path     AS file_path,
               f.language      AS language,
               f.summary       AS summary,
               f.symbol_names  AS symbol_names,
               f.symbol_count  AS symbol_count
        ORDER BY f.file_path
        """
        with self.store._session() as session:
            result = session.run(
                cypher, org_id=self.org_id, repo=repo,
            )
            return [r.data() for r in result]

    def _build_file_prompt(self, row: Dict[str, Any]) -> str:
        """Fill _FILE_PROMPT from a File row."""
        symbol_names = row.get("symbol_names") or []
        # Show a reasonable subset of symbol names to keep the prompt short.
        symbol_list = ", ".join(symbol_names[:40])
        if len(symbol_names) > 40:
            symbol_list += f", ... (+{len(symbol_names) - 40} more)"

        return _FILE_PROMPT.format(
            file_path=row.get("file_path", ""),
            language=row.get("language", ""),
            symbol_count=row.get("symbol_count", len(symbol_names)),
            symbol_list=symbol_list or "(none)",
        )

    def _update_file(
        self,
        row: Dict[str, Any],
        new_summary: str,
    ) -> None:
        """Re-embed file summary and write back via
        store.update_file_summary."""
        new_vector = self.embedder.re_embed_summary(new_summary)
        self.store.update_file_summary(
            org_id=row["org_id"],
            repo=row["repo"],
            file_path=row["file_path"],
            summary=new_summary,
            summary_source=S.SUMMARY_SOURCE_LLM,
            summary_embedding=new_vector,
        )

    # =================================================================
    # RELIABILITY
    # =================================================================

    def _llm_with_backoff(self, prompt: str) -> str:
        """Wrap llm_call with exponential backoff on transient errors.

        Retries up to 4 times with doubling delay (1s → 2s → 4s → 8s),
        capped at MAX_BACKOFF_SECONDS.  Non-transient errors (ValueError,
        KeyError, etc.) are re-raised immediately.
        """
        max_retries = 4
        backoff = 1.0

        for attempt in range(max_retries + 1):
            try:
                result = self.llm_call(prompt)
                if not result or not result.strip():
                    raise ValueError("LLM returned empty summary")
                return result.strip()
            except (ValueError, KeyError, TypeError):
                # Non-transient — don't retry.
                raise
            except Exception as e:
                if attempt == max_retries:
                    raise
                logger.warning(
                    "  LLM call failed (attempt %d/%d): %s — retrying in %.0fs",
                    attempt + 1, max_retries, e, backoff,
                )
                time.sleep(min(backoff, MAX_BACKOFF_SECONDS))
                backoff *= 2

        # Should be unreachable, but satisfy the type checker.
        raise RuntimeError("LLM call exhausted retries")

    def _sleep_rate_limit(self) -> None:
        """Honor delay_seconds between calls."""
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
