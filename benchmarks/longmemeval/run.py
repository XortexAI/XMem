"""Command line entrypoint for the LongMemEval benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_API_KEY_ENV,
    DEFAULT_DATASET_VARIANT,
    BenchmarkConfig,
)
from .dataset import download_dataset
from .runner import LongMemEvalRunner


def main() -> None:
    try:
        args = parse_args()
        dataset_path = prepare_dataset(args)
        config = build_config(args, dataset_path)
        summary = asyncio.run(LongMemEvalRunner(config).run())
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def prepare_dataset(args: argparse.Namespace) -> Path:
    if args.download:
        try:
            return download_dataset(args.variant, args.dataset_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download the LongMemEval dataset. "
                "Check your network connection, or download the dataset manually "
                f"and pass --dataset-path. Details: {exc}"
            ) from exc

    if not args.dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {args.dataset_path}. "
            "Run with --download, or pass --dataset-path to a local "
            "LongMemEval JSON/JSONL file."
        )
    return args.dataset_path


def build_config(args: argparse.Namespace, dataset_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
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
        question_type=args.question_type,
        skip_ingest=args.skip_ingest,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LongMemEval against the Python XMem API.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("benchmarks/longmemeval/data/longmemeval_s_cleaned.json"),
        help="Path to a LongMemEval JSON or JSONL file.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the selected LongMemEval dataset variant before running.",
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_DATASET_VARIANT,
        help="Dataset variant to download when --download is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/longmemeval/results/latest"),
        help="Directory for results.jsonl, predictions.jsonl, and summary.json.",
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
    parser.add_argument("--user-prefix", default="longmemeval")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--question-type")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and transform the dataset without calling XMem.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
