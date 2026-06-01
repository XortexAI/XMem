# LongMemEval Benchmark for XMem Python

This harness benchmarks the Python XMem service only. It targets the deployed
Python API at `https://api.xmem.in` by default and does not run or compare the
Go implementation.

LongMemEval evaluates long-term conversational memory across multi-session
recall, temporal reasoning, single-session recall, knowledge updates, and
preference tracking. The harness follows the same broad structure used by
open-source memory-layer benchmarks: load dataset records, ingest the haystack
conversation history into an isolated user namespace, retrieve an answer for
the benchmark question, write predictions, and compute lightweight local
metrics for quick iteration.

## Files

- `dataset.py`: Loads JSON/JSONL LongMemEval records and converts sessions to
  XMem conversation-turn ingest payloads.
- `client.py`: Async HTTP client for the Python XMem API.
- `runner.py`: Benchmark orchestration, batching, polling, resume support, and
  output writing.
- `metrics.py`: Local exact-match, contains, and token-F1 metrics plus summary
  aggregation.
- `run.py`: CLI entrypoint.

## Secrets

Do not commit API keys or provider credentials.

To generate XMem predictions, set an XMem API key:

```bash
export XMEM_API_KEY="..."
```

Use `--api-key-env` if your local environment uses a different variable name.

To score predictions with the official LongMemEval LLM-as-judge evaluator, set
an OpenAI API key before running the evaluator:

```bash
export OPENAI_API_KEY="..."
```

## Run a Smoke Check

Validate dataset parsing and payload construction without calling the service:

```bash
python -m benchmarks.longmemeval.run \
  --download \
  --dry-run \
  --limit 2
```

Validate all six official categories without requiring an API key:

```bash
python -m benchmarks.longmemeval.run_all_categories \
  --download \
  --dry-run
```

If the dataset is already available locally:

```bash
python -m benchmarks.longmemeval.run \
  --dataset-path benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
  --dry-run \
  --limit 2
```

## Run Against the Python API

```bash
export XMEM_API_KEY="..."

python -m benchmarks.longmemeval.run \
  --download \
  --api-base-url https://api.xmem.in \
  --limit 10 \
  --batch-size 25 \
  --output-dir benchmarks/longmemeval/results/run-001
```

The runner writes:

- `results.jsonl`: Full per-example benchmark records.
- `predictions.jsonl`: Official prediction file with only `question_id` and
  `hypothesis`.
- `summary.json`: Aggregate local metrics and latency.

The local metrics are intended for fast development feedback. For publication
quality reporting, run the generated `predictions.jsonl` through the official
LongMemEval evaluation flow or an agreed LLM-as-judge rubric using the same
model/settings across systems.

The benchmark runner itself only needs `XMEM_API_KEY` because it generates XMem
answers. The official/equivalent evaluator is a separate scoring step and needs
`OPENAI_API_KEY` when using an OpenAI judge model.

## Run All Official Categories

The dataset has six `question_type` categories. Each example has a unique
`question_id` and its own haystack sessions, and this runner isolates each
question into a separate XMem user namespace. That makes category-level
parallelism safe from memory leakage; the only practical constraint is API
throughput and rate limiting.

```bash
export XMEM_API_KEY="..."

python -m benchmarks.longmemeval.run_all_categories \
  --dataset-path benchmarks/longmemeval/data/longmemeval_s_cleaned.json \
  --api-base-url https://api.xmem.in \
  --output-root benchmarks/longmemeval/results/full-six-categories \
  --max-parallel-categories 6
```

The all-category runner prints live processed/left/ETA status and writes one
official merged prediction file at:

```text
benchmarks/longmemeval/results/full-six-categories/predictions.jsonl
```

Each category also gets a `runner.log` file under its output directory. If a
category process fails, the launcher prints the failing category, exit code, log
path, and the most recent child-process output.

## Useful Options

- `--limit N`: Run a small subset first.
- `--offset N`: Skip the first N selected examples.
- `--question-type TYPE`: Filter to one LongMemEval category.
- `--skip-ingest`: Reuse already-ingested user namespaces and only retrieve.
- `--no-resume`: Re-run examples even if they already exist in `results.jsonl`.
- `--ingest-api-version v1`: Use synchronous batch ingestion instead of the
  default durable `/v2/memory/batch-ingest` path.
- `--effort-level high`: Use high-effort XMem ingestion for long records.
- `--dry-run`: Validate dataset/category setup without API calls.
- `--verbose`: Print child runner output while the all-category launcher runs.

## Expected Failures

These errors are intentional and should be actionable:

- `Dataset file not found`: run with `--download`, or pass `--dataset-path`.
- `Missing API key`: set `XMEM_API_KEY`, or pass `--api-key-env` for a custom
  variable name.
- Official evaluator authentication errors: set `OPENAI_API_KEY` before running
  the LongMemEval scoring step.
- `Failed to download the LongMemEval dataset`: check network access, then retry
  or download the dataset manually.
- `<category> failed with exit code ...`: inspect that category's `runner.log`.

## Isolation Model

Each example is ingested into a user id derived from:

```text
<user-prefix>-<question-id>
```

This prevents facts from one benchmark question from leaking into another. Use a
new `--user-prefix` for fully fresh runs.
