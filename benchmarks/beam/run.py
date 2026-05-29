"""Command line entrypoint for the BEAM benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_API_KEY_ENV,
    DEFAULT_SPLIT,
    BenchmarkConfig,
)
from .dataset import download_dataset
from .runner import BeamRunner


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
            question_type=args.question_type,
            sample_percent_per_question_type=args.sample_percent_per_question_type,
            sample_min_per_question_type=args.sample_min_per_question_type,
            sample_seed=args.sample_seed,
            split=args.split,
            skip_ingest=args.skip_ingest,
            resume=not args.no_resume,
            dry_run=args.dry_run,
        )
        print(json.dumps(asyncio.run(BeamRunner(config).run()), indent=2))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def prepare_dataset(args: argparse.Namespace) -> Path:
    if args.download:
        try:
            return download_dataset(args.split, args.dataset_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download BEAM. Check network access, or pass "
                f"--dataset-path to a local file. Details: {exc}"
            ) from exc
    if not args.dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {args.dataset_path}. "
            "Run with --download, or pass --dataset-path."
        )
    return args.dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BEAM against XMem.")
    parser.add_argument(
        "--split",
        choices=("100K", "500K", "1M"),
        default=DEFAULT_SPLIT,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("benchmarks/beam/data/1M-00000-of-00001.parquet"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/beam/results/latest"),
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
    parser.add_argument("--user-prefix", default="beam")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--question-type")
    parser.add_argument(
        "--sample-percent-per-question-type",
        type=float,
        help="Select a balanced percent from each BEAM question_type.",
    )
    parser.add_argument("--sample-min-per-question-type", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
