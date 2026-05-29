"""Command line entrypoint for the LoCoMo benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import DEFAULT_API_BASE_URL, DEFAULT_API_KEY_ENV, BenchmarkConfig
from .dataset import download_dataset
from .runner import LoCoMoRunner


def main() -> None:
    try:
        args = parse_args()
        dataset_path = prepare_dataset(args)
        config = BenchmarkConfig(
            dataset_path=dataset_path,
            output_dir=args.output_dir,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            api_timeout_seconds=args.api_timeout_seconds,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            batch_size=args.batch_size,
            ingest_api_version=args.ingest_api_version,
            poll_interval_seconds=args.poll_interval_seconds,
            poll_timeout_seconds=args.poll_timeout_seconds,
            top_k=args.top_k,
            effort_level=args.effort_level,
            user_prefix=args.user_prefix,
            limit=args.limit,
            offset=args.offset,
            category=args.category,
            skip_ingest=args.skip_ingest,
            resume=not args.no_resume,
            dry_run=args.dry_run,
        )
        print(json.dumps(asyncio.run(LoCoMoRunner(config).run()), indent=2))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def prepare_dataset(args: argparse.Namespace) -> Path:
    if args.download:
        try:
            return download_dataset(args.dataset_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download LoCoMo. Check network access, or pass "
                f"--dataset-path to a local file. Details: {exc}"
            ) from exc
    if not args.dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {args.dataset_path}. "
            "Run with --download, or pass --dataset-path."
        )
    return args.dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoCoMo against XMem.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("benchmarks/locomo/data/locomo10.json"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/locomo/results/latest"),
    )
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--api-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--ingest-api-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--effort-level", choices=("low", "high"), default="low")
    parser.add_argument("--user-prefix", default="locomo")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--category")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
