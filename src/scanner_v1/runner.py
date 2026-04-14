"""
Runner — CLI entry point for scanner_v1.

Subcommands:
  setup    One-time schema setup (constraints + vector + fulltext indexes).
  scan     Run a full or incremental scan of a repository.
  enrich   Run the phase 2 LLM summary enrichment worker.
  reset    Delete everything for a given repo.

Usage:
  python -m src.scanner_v1.runner setup
  python -m src.scanner_v1.runner scan  --org acme --repo payments \\
                                        --url git@github.com:acme/payments.git
  python -m src.scanner_v1.runner scan  --org acme --config repos.json
  python -m src.scanner_v1.runner enrich --org acme --repo payments
  python -m src.scanner_v1.runner reset  --org acme --repo payments --yes

Wiring:
  - CodeStoreV1 is built from settings (neo4j_uri / username / password).
  - Embedder is built from src.pipelines.ingest.embed_text (same function
    used by v0). Both lanes share it in v1; a code-native model can be
    plugged into the code lane later without touching this file.
  - llm_call for enrich is built from src.models.get_model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from src.config import settings
from src.scanner_v1.store import CodeStoreV1
from src.scanner_v1.embedder import Embedder
from src.scanner_v1.indexer import IndexerV1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("xmem.scanner_v1.runner")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _build_store() -> CodeStoreV1:
    """Instantiate and connect CodeStoreV1 from settings.

    embedding_dimension must match the live embedder — pulled from
    settings.pinecone_dimension for now because the v0 embed_text was
    configured there. When a code-native model is plugged in this
    dimension will need to match whichever lane governs the index.
    """
    store = CodeStoreV1(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=None,
        embedding_dimension=settings.pinecone_dimension,
    )
    store.connect()
    return store


def _build_embedder() -> Embedder:
    """Build an Embedder wired to the default embed_fn.

    v1 uses the same fn for both lanes; plugging a code-native model
    into the code lane is a one-line swap at the `code_embed_fn=` arg.
    embed_text returns a tuple, so we wrap it to match List[float].
    """
    from src.pipelines.ingest import embed_text

    def _embed(text: str) -> List[float]:
        return list(embed_text(text))

    return Embedder(summary_embed_fn=_embed)


def _build_llm_call() -> Callable[[str], str]:
    """Return a callable(prompt) -> str for the enricher.
    Uses the default model from the registry."""
    from src.models import get_model

    model = get_model()

    def _call(prompt: str) -> str:
        response = model.invoke(prompt)
        # LangChain-style models return an object with `.content`;
        # fall back to str() if we got a raw string.
        return getattr(response, "content", None) or str(response)

    return _call


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_setup(args: argparse.Namespace) -> int:
    """Run CodeStoreV1.setup_schema once per Neo4j instance."""
    store = _build_store()
    try:
        logger.info("Running schema setup (constraints + vector + fulltext)...")
        store.setup_schema()
        logger.info("Schema setup complete.")
        return 0
    finally:
        store.close()


def cmd_scan(args: argparse.Namespace) -> int:
    """Build IndexerV1 and call scan_repo. Supports single repo or config.
    Returns non-zero if any repo failed."""
    if not args.config and not (args.repo and args.url):
        logger.error("scan requires --repo + --url, or --config")
        return 2

    store = _build_store()
    embedder = _build_embedder()
    indexer = IndexerV1(
        org_id=args.org,
        store=store,
        embedder=embedder,
        clone_root=args.clone_root,
    )
    token = os.environ.get("GITHUB_TOKEN")

    try:
        if args.config:
            repos = _load_repos_config(args.config)
            return _run_config_scan(indexer, repos, token, args.force_full)

        # Single repo.
        try:
            result = indexer.scan_repo(
                repo_name=args.repo,
                repo_url=args.url,
                branch=args.branch,
                token=token,
                force_full=args.force_full,
            )
            logger.info("Scan result: %s", result)
            return 0
        except Exception as e:
            logger.error("Scan failed: %s", e)
            return 1
    finally:
        indexer.close()


def _run_config_scan(
    indexer: IndexerV1,
    repos: List[Dict[str, Any]],
    token: Optional[str],
    force_full: bool,
) -> int:
    """Scan multiple repos from a config list. Returns 1 if any failed."""
    logger.info("=" * 70)
    logger.info("BATCH SCAN: %d repositories", len(repos))
    logger.info("=" * 70)

    results: List[Dict[str, Any]] = []
    total_start = time.time()

    for i, repo_cfg in enumerate(repos, 1):
        name = repo_cfg["name"]
        url = repo_cfg["url"]
        branch = repo_cfg.get("branch", "main")

        logger.info("[%d/%d] Scanning %s ...", i, len(repos), name)
        try:
            result = indexer.scan_repo(
                repo_name=name,
                repo_url=url,
                branch=branch,
                token=token,
                force_full=force_full,
            )
            result["status"] = "success"
            results.append(result)
        except Exception as e:
            logger.error("[%d/%d] FAILED: %s — %s", i, len(repos), name, e)
            results.append({"repo": name, "status": "failed", "error": str(e)})

    duration = time.time() - total_start
    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")

    logger.info("=" * 70)
    logger.info("BATCH SCAN COMPLETE in %.1fs", duration)
    logger.info("  Succeeded: %d / %d", succeeded, len(repos))
    logger.info("  Failed:    %d / %d", failed, len(repos))
    for r in results:
        icon = "OK" if r.get("status") == "success" else "FAIL"
        logger.info(
            "  [%s] %s (%.1fs)",
            icon, r.get("repo", "?"), r.get("duration_seconds", 0),
        )
    logger.info("=" * 70)

    return 1 if failed else 0


def cmd_enrich(args: argparse.Namespace) -> int:
    """Build EnricherV1 and call enrich_repo."""
    # Lazy import — enricher.py is a sibling file that lands in the
    # next implementation step. Importing at module top would break
    # every other subcommand until the file exists.
    from src.scanner_v1.enricher import EnricherV1

    store = _build_store()
    embedder = _build_embedder()
    llm_call = _build_llm_call()

    enricher = EnricherV1(
        org_id=args.org,
        store=store,
        embedder=embedder,
        llm_call=llm_call,
        delay=args.enrich_delay,
        max_symbols=args.max_symbols,
        max_files=args.max_files,
    )
    try:
        if args.config:
            repos = _load_repos_config(args.config)
            failures = 0
            for repo_cfg in repos:
                name = repo_cfg["name"]
                logger.info("Enriching %s ...", name)
                try:
                    enricher.enrich_repo(name)
                except Exception as e:
                    logger.error("Enrichment failed for %s: %s", name, e)
                    failures += 1
            return 1 if failures else 0

        if not args.repo:
            logger.error("enrich requires --repo or --config")
            return 2

        try:
            enricher.enrich_repo(args.repo)
            return 0
        except Exception as e:
            logger.error("Enrichment failed: %s", e)
            return 1
    finally:
        enricher.close()


def cmd_reset(args: argparse.Namespace) -> int:
    """Delete every node for (org_id, repo). Dangerous — requires --yes
    unless stdin is an interactive TTY where we can prompt."""
    if not args.yes:
        if sys.stdin.isatty():
            prompt = (
                f"This will DELETE all indexed data for "
                f"{args.org}/{args.repo}. Type 'yes' to confirm: "
            )
            if input(prompt).strip().lower() != "yes":
                logger.info("Aborted.")
                return 1
        else:
            logger.error(
                "reset requires --yes when stdin is not an interactive TTY",
            )
            return 2

    store = _build_store()
    try:
        count = store.delete_repository(args.org, args.repo)
        logger.info(
            "Deleted %d nodes for %s/%s", count, args.org, args.repo,
        )
        return 0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_repos_config(config_path: str) -> List[Dict[str, Any]]:
    """Load a repos.json file.

    Shape::

        {"repos": [{"name": "...", "url": "...", "branch": "main"}, ...]}
    """
    with open(config_path) as f:
        config = json.load(f)
    return config.get("repos", [])


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse tree."""
    parser = argparse.ArgumentParser(
        prog="python -m src.scanner_v1.runner",
        description="XMem scanner_v1 CLI — single-store Neo4j scanner.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── setup ────────────────────────────────────────────────────────
    p_setup = sub.add_parser(
        "setup", help="One-time schema setup (constraints + vector + fulltext).",
    )
    p_setup.set_defaults(func=cmd_setup)

    # ── scan ─────────────────────────────────────────────────────────
    p_scan = sub.add_parser("scan", help="Scan one or more repositories.")
    p_scan.add_argument("--org", required=True, help="Organization ID")
    p_scan.add_argument("--repo", help="Repository name (single repo)")
    p_scan.add_argument("--url", help="Git clone URL (single repo)")
    p_scan.add_argument("--branch", default="main", help="Branch (default: main)")
    p_scan.add_argument(
        "--config",
        help="Path to repos.json for multi-repo scan",
    )
    p_scan.add_argument(
        "--force-full", action="store_true",
        help="Force full scan (ignore incremental)",
    )
    p_scan.add_argument(
        "--clone-root", default="/tmp/xmem_repos_v1",
        help="Root dir for repo clones",
    )
    p_scan.set_defaults(func=cmd_scan)

    # ── enrich ───────────────────────────────────────────────────────
    p_enrich = sub.add_parser(
        "enrich", help="Run phase 2 LLM summary enrichment.",
    )
    p_enrich.add_argument("--org", required=True, help="Organization ID")
    p_enrich.add_argument("--repo", help="Repository name")
    p_enrich.add_argument("--config", help="Path to repos.json for multi-repo")
    p_enrich.add_argument(
        "--max-symbols", type=int, default=0,
        help="Cap on symbols per repo (0 = unlimited)",
    )
    p_enrich.add_argument(
        "--max-files", type=int, default=0,
        help="Cap on files per repo (0 = unlimited)",
    )
    p_enrich.add_argument(
        "--enrich-delay", type=float, default=0.5,
        help="Seconds between LLM calls (rate-limit safety)",
    )
    p_enrich.set_defaults(func=cmd_enrich)

    # ── reset ────────────────────────────────────────────────────────
    p_reset = sub.add_parser(
        "reset", help="Delete all indexed data for a repo.",
    )
    p_reset.add_argument("--org", required=True, help="Organization ID")
    p_reset.add_argument("--repo", required=True, help="Repository name")
    p_reset.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation (required in non-TTY).",
    )
    p_reset.set_defaults(func=cmd_reset)

    return parser


def main(argv: Optional[list] = None) -> int:
    """CLI entry. Returns exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
