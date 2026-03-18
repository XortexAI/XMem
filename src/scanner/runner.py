"""
Scanner Runner — CLI entry point for nightly (or on-demand) scans and
Phase 2 LLM summary enrichment.

Usage:
  # Scan a single repo (Phase 1 — deterministic AST indexing)
  python -m src.scanner.runner \\
    --org zinnia \\
    --repo payment-service \\
    --url https://github.com/zinnia/payment-service.git

  # Scan multiple repos from a config
  python -m src.scanner.runner \\
    --org zinnia \\
    --config repos.json

  # Force a full re-scan (ignore incremental)
  python -m src.scanner.runner \\
    --org zinnia \\
    --repo payment-service \\
    --url https://github.com/zinnia/payment-service.git \\
    --full

  # Enrich summaries with LLM (Phase 2 — runs after scan)
  python -m src.scanner.runner \\
    --org zinnia \\
    --repo payment-service \\
    --enrich

  # Scan + enrich in one go
  python -m src.scanner.runner \\
    --org zinnia \\
    --repo payment-service \\
    --url https://github.com/zinnia/payment-service.git \\
    --enrich

  # Enrich with caps (limit LLM calls)
  python -m src.scanner.runner \\
    --org zinnia \\
    --repo payment-service \\
    --enrich \\
    --max-symbols 200 \\
    --max-files 50

Environment variables:
  GITHUB_TOKEN        — for private repos
  MONGODB_URI         — MongoDB connection string
  PINECONE_API_KEY    — Pinecone API key
  NEO4J_URI           — Neo4j connection string
  NEO4J_PASSWORD      — Neo4j password
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("xmem.scanner.runner")


def scan_single_repo(
    org_id: str,
    repo_name: str,
    repo_url: str,
    branch: str = "main",
    force_full: bool = False,
    clone_root: str = "/tmp/xmem_repos",
) -> Dict[str, Any]:
    """Scan a single repository."""
    from src.scanner.indexer import Indexer

    token = os.environ.get("GITHUB_TOKEN")

    indexer = Indexer(org_id=org_id, clone_root=clone_root)
    try:
        result = indexer.scan_repo(
            repo_name=repo_name,
            repo_url=repo_url,
            branch=branch,
            token=token,
            force_full=force_full,
        )
        return result
    finally:
        indexer.close()


def enrich_single_repo(
    org_id: str,
    repo_name: str,
    max_symbols: int = 0,
    max_files: int = 0,
    delay: float = 0.5,
) -> Dict[str, Any]:
    """Run Phase 2 LLM enrichment on a single repository."""
    from src.scanner.enricher import Enricher

    enricher = Enricher(
        org_id=org_id,
        delay=delay,
        max_symbols=max_symbols,
        max_files=max_files,
    )
    try:
        return enricher.enrich_repo(repo_name)
    finally:
        enricher.close()


def scan_from_config(
    org_id: str,
    config_path: str,
    force_full: bool = False,
    clone_root: str = "/tmp/xmem_repos",
) -> List[Dict[str, Any]]:
    """Scan multiple repositories from a JSON config file.

    Config format::

        {
          "repos": [
            {
              "name": "payment-service",
              "url": "https://github.com/zinnia/payment-service.git",
              "branch": "main"
            },
            {
              "name": "auth-service",
              "url": "https://github.com/zinnia/auth-service.git"
            }
          ]
        }
    """
    with open(config_path) as f:
        config = json.load(f)

    results: List[Dict[str, Any]] = []
    repos = config.get("repos", [])

    logger.info("=" * 70)
    logger.info("NIGHTLY SCAN: %d repositories for org '%s'", len(repos), org_id)
    logger.info("=" * 70)

    total_start = time.time()

    for i, repo_config in enumerate(repos, 1):
        name = repo_config["name"]
        url = repo_config["url"]
        branch = repo_config.get("branch", "main")

        logger.info("[%d/%d] Scanning %s ...", i, len(repos), name)

        try:
            result = scan_single_repo(
                org_id=org_id,
                repo_name=name,
                repo_url=url,
                branch=branch,
                force_full=force_full,
                clone_root=clone_root,
            )
            result["status"] = "success"
            results.append(result)
        except Exception as e:
            logger.error("[%d/%d] FAILED: %s — %s", i, len(repos), name, e)
            results.append({
                "repo": name,
                "status": "failed",
                "error": str(e),
            })

    total_duration = time.time() - total_start

    # Summary
    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "failed")

    logger.info("=" * 70)
    logger.info("NIGHTLY SCAN COMPLETE in %.1fs", total_duration)
    logger.info("  Succeeded: %d / %d", succeeded, len(repos))
    logger.info("  Failed:    %d / %d", failed, len(repos))
    for r in results:
        status_icon = "OK" if r.get("status") == "success" else "FAIL"
        duration = r.get("duration_seconds", 0)
        logger.info("  [%s] %s (%.1fs)", status_icon, r.get("repo", "?"), duration)
    logger.info("=" * 70)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XMem Scanner Bot — index codebases and enrich summaries with LLM",
    )
    parser.add_argument("--org", required=True, help="Organization ID")
    parser.add_argument("--repo", help="Repository name (for single repo scan/enrich)")
    parser.add_argument("--url", help="Git clone URL (for single repo scan)")
    parser.add_argument("--branch", default="main", help="Branch to scan (default: main)")
    parser.add_argument("--config", help="Path to repos.json config file (for multi-repo scan)")
    parser.add_argument("--full", action="store_true", help="Force full scan (ignore incremental)")
    parser.add_argument("--clone-root", default="/tmp/xmem_repos", help="Root dir for repo clones")

    # Phase 2: enrichment flags
    parser.add_argument("--enrich", action="store_true",
                        help="Run Phase 2 LLM summary enrichment (after scan, or standalone)")
    parser.add_argument("--max-symbols", type=int, default=0,
                        help="Max symbols to enrich per repo (0 = unlimited)")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Max files to enrich per repo (0 = unlimited)")
    parser.add_argument("--enrich-delay", type=float, default=0.5,
                        help="Seconds between LLM calls (rate-limit safety, default: 0.5)")

    args = parser.parse_args()

    ran_scan = False

    # Phase 1: scan
    if args.config:
        results = scan_from_config(
            org_id=args.org,
            config_path=args.config,
            force_full=args.full,
            clone_root=args.clone_root,
        )
        failed = [r for r in results if r.get("status") == "failed"]
        if failed and not args.enrich:
            sys.exit(1)
        ran_scan = True
    elif args.repo and args.url:
        scan_single_repo(
            org_id=args.org,
            repo_name=args.repo,
            repo_url=args.url,
            branch=args.branch,
            force_full=args.full,
            clone_root=args.clone_root,
        )
        ran_scan = True

    # Phase 2: enrichment
    if args.enrich:
        if args.config:
            with open(args.config) as f:
                config = json.load(f)
            for repo_cfg in config.get("repos", []):
                name = repo_cfg["name"]
                logger.info("Enriching %s ...", name)
                try:
                    enrich_single_repo(
                        org_id=args.org,
                        repo_name=name,
                        max_symbols=args.max_symbols,
                        max_files=args.max_files,
                        delay=args.enrich_delay,
                    )
                except Exception as e:
                    logger.error("Enrichment failed for %s: %s", name, e)
        elif args.repo:
            enrich_single_repo(
                org_id=args.org,
                repo_name=args.repo,
                max_symbols=args.max_symbols,
                max_files=args.max_files,
                delay=args.enrich_delay,
            )
        else:
            parser.error("--enrich requires --repo or --config")
    elif not ran_scan:
        parser.error(
            "Provide --config for multi-repo scan, --repo + --url for single scan, "
            "or --repo + --enrich for standalone enrichment"
        )


if __name__ == "__main__":
    main()
