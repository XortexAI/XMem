"""Run the official LongMemEval question_type categories in parallel."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_API_BASE_URL, DEFAULT_API_KEY_ENV, DEFAULT_DATASET_VARIANT
from .dataset import build_ingest_items, download_dataset, load_examples


OFFICIAL_QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
)


@dataclass
class CategoryState:
    category: str
    total: int
    ingest_total: int = 0
    processed: int = 0
    ingest_processed: int = 0
    current_ingest_seen: int = 0
    done: bool = False
    returncode: int | None = None

    @property
    def left(self) -> int:
        return max(self.total - self.processed, 0)


async def main() -> None:
    try:
        args = parse_args()
        await run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


async def run(args: argparse.Namespace) -> None:
    validate_args(args)
    dataset_path = prepare_dataset(args)
    examples = load_examples(dataset_path)
    validate_independence(examples)
    counts = Counter(example.question_type for example in examples)
    ingest_counts = Counter(
        {
            category: sum(
                len(build_ingest_items(example, user_id="inspect"))
                for example in examples
                if example.question_type == category
            )
            for category in OFFICIAL_QUESTION_TYPES
        }
    )
    states = {
        category: CategoryState(
            category=category,
            total=counts[category],
            ingest_total=ingest_counts[category],
        )
        for category in OFFICIAL_QUESTION_TYPES
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    print_category_plan(states)

    if args.dry_run:
        print(
            "Dry run complete: dataset loaded, categories validated, "
            "and no API calls were made.",
            flush=True,
        )
        return

    if not os.getenv(args.api_key_env):
        raise RuntimeError(
            f"Missing API key. Set {args.api_key_env} before running the "
            "benchmark, for example: export XMEM_API_KEY='...'."
        )

    start_time = time.monotonic()
    semaphore = asyncio.Semaphore(args.max_parallel_categories)
    tasks = [
        asyncio.create_task(
            run_category(
                category,
                state,
                args=args,
                dataset_path=dataset_path,
                semaphore=semaphore,
            )
        )
        for category, state in states.items()
    ]
    reporter = asyncio.create_task(
        report_progress(states, start_time, args.status_seconds)
    )
    results = await asyncio.gather(*tasks, return_exceptions=True)
    reporter.cancel()
    await asyncio.gather(reporter, return_exceptions=True)

    failures = []
    for result in results:
        if isinstance(result, Exception):
            failures.append(str(result))
    if failures:
        raise RuntimeError("One or more category runs failed: " + " | ".join(failures))

    merge_predictions(args.output_root)
    print_status(states, start_time, final=True)


def prepare_dataset(args: argparse.Namespace) -> Path:
    if args.download:
        try:
            return download_dataset(args.variant, args.dataset_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download the LongMemEval dataset. "
                "Check network access, or download the dataset manually and "
                f"pass --dataset-path. Details: {exc}"
            ) from exc

    if not args.dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {args.dataset_path}. "
            "Run with --download, or pass --dataset-path to a local "
            "LongMemEval JSON/JSONL file."
        )
    return args.dataset_path


def validate_args(args: argparse.Namespace) -> None:
    if args.max_parallel_categories < 1:
        raise ValueError("--max-parallel-categories must be at least 1.")
    if args.max_parallel_categories > len(OFFICIAL_QUESTION_TYPES):
        raise ValueError(
            "--max-parallel-categories cannot exceed the six official "
            "LongMemEval question_type categories."
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.status_seconds < 1:
        raise ValueError("--status-seconds must be at least 1.")


async def run_category(
    category: str,
    state: CategoryState,
    *,
    args: argparse.Namespace,
    dataset_path: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        output_dir = args.output_root / category
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "runner.log"
        recent_lines: deque[str] = deque(maxlen=20)
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.longmemeval.run",
            "--dataset-path",
            str(dataset_path),
            "--api-base-url",
            args.api_base_url,
            "--api-key-env",
            args.api_key_env,
            "--output-dir",
            str(output_dir),
            "--question-type",
            category,
            "--batch-size",
            str(args.batch_size),
            "--ingest-api-version",
            args.ingest_api_version,
            "--top-k",
            str(args.top_k),
            "--effort-level",
            args.effort_level,
            "--user-prefix",
            f"{args.user_prefix}-{category}",
        ]
        if args.no_resume:
            cmd.append("--no-resume")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert process.stdout is not None
        with log_path.open("w", encoding="utf-8") as log_file:
            async for raw_line in process.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                recent_lines.append(line)
                log_file.write(line + "\n")
                log_file.flush()
                update_state_from_line(state, line)
                update_ingest_state_from_line(state, line)
                if args.verbose:
                    print(f"[{category}] {line}", flush=True)
        state.returncode = await process.wait()
        state.done = state.returncode == 0
        if state.returncode != 0:
            tail = "\n".join(recent_lines) or "(no child output captured)"
            raise RuntimeError(
                f"{category} failed with exit code {state.returncode}. "
                f"See {log_path}. Recent output:\n{tail}"
            )


async def report_progress(
    states: dict[str, CategoryState],
    start_time: float,
    status_seconds: float,
) -> None:
    try:
        while True:
            print_status(states, start_time)
            await asyncio.sleep(status_seconds)
    except asyncio.CancelledError:
        return


def update_state_from_line(state: CategoryState, line: str) -> None:
    if not line.startswith("[") or "/" not in line:
        return
    close = line.find("]")
    if close == -1:
        return
    progress = line[1:close]
    if "/" not in progress:
        return
    processed_text, total_text = progress.split("/", 1)
    if processed_text.isdigit() and total_text.isdigit():
        state.processed = max(state.processed, int(processed_text))
        state.total = int(total_text)


def update_ingest_state_from_line(state: CategoryState, line: str) -> None:
    if not line.startswith("[INGEST] processed="):
        return
    progress = line.split("processed=", 1)[1].strip()
    if "/" not in progress:
        return
    processed_text, total_text = progress.split("/", 1)
    if processed_text.isdigit() and total_text.isdigit():
        processed = int(processed_text)
        total = int(total_text)
        # Child output is per-question, so accumulate forward movement.
        if processed < state.current_ingest_seen:
            state.current_ingest_seen = 0
        delta = max(processed - state.current_ingest_seen, 0)
        state.ingest_processed = min(
            state.ingest_processed + delta,
            state.ingest_total,
        )
        state.current_ingest_seen = 0 if processed == total else processed


def print_status(
    states: dict[str, CategoryState],
    start_time: float,
    *,
    final: bool = False,
) -> None:
    processed = sum(state.processed for state in states.values())
    total = sum(state.total for state in states.values())
    left = max(total - processed, 0)
    ingest_processed = sum(state.ingest_processed for state in states.values())
    ingest_total = sum(state.ingest_total for state in states.values())
    ingest_left = max(ingest_total - ingest_processed, 0)
    elapsed = max(time.monotonic() - start_time, 0.001)
    rate = processed / elapsed if processed else 0.0
    eta = left / rate if rate else 0.0
    label = "FINAL" if final else "STATUS"
    print(
        f"[{label}] processed={processed}/{total} left={left} "
        f"ingested_pairs={ingest_processed}/{ingest_total} "
        f"pairs_left={ingest_left} elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta)}",
        flush=True,
    )
    for category in OFFICIAL_QUESTION_TYPES:
        state = states[category]
        print(
            f"  - {category}: {state.processed}/{state.total} questions left="
            f"{state.left}; ingest_pairs={state.ingest_processed}/"
            f"{state.ingest_total}",
            flush=True,
        )


def print_category_plan(states: dict[str, CategoryState]) -> None:
    print("LongMemEval category plan:", flush=True)
    for category in OFFICIAL_QUESTION_TYPES:
        state = states[category]
        print(
            f"  - {category}: {state.total} questions, "
            f"{state.ingest_total} ingest pairs",
            flush=True,
        )


def merge_predictions(output_root: Path) -> None:
    merged_path = output_root / "predictions.jsonl"
    missing_predictions = []
    with merged_path.open("w", encoding="utf-8") as merged:
        for category in OFFICIAL_QUESTION_TYPES:
            path = output_root / category / "predictions.jsonl"
            if not path.exists():
                missing_predictions.append(str(path))
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        payload = json.loads(line)
                        merged.write(
                            json.dumps(
                                {
                                    "question_id": payload["question_id"],
                                    "hypothesis": payload["hypothesis"],
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
    if missing_predictions:
        raise RuntimeError(
            "Missing category prediction files: "
            + ", ".join(missing_predictions)
        )
    print(f"Merged official predictions: {merged_path}", flush=True)


def validate_independence(examples: list[object]) -> None:
    question_ids = [getattr(example, "question_id") for example in examples]
    if len(question_ids) != len(set(question_ids)):
        raise RuntimeError("Dataset has duplicate question_id values.")
    categories = {getattr(example, "question_type") for example in examples}
    missing = set(OFFICIAL_QUESTION_TYPES) - categories
    if missing:
        raise RuntimeError(f"Dataset missing official categories: {sorted(missing)}")


def format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all official LongMemEval categories against XMem.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("benchmarks/longmemeval/data/longmemeval_s_cleaned.json"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--variant", default=DEFAULT_DATASET_VARIANT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/longmemeval/results/full-six-categories"),
    )
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--ingest-api-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--effort-level", choices=("low", "high"), default="low")
    parser.add_argument("--user-prefix", default="longmemeval")
    parser.add_argument("--max-parallel-categories", type=int, default=6)
    parser.add_argument("--status-seconds", type=float, default=30.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset/category setup without requiring an API key.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
