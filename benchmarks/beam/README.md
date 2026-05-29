# BEAM 1M Benchmark for XMem Python

This harness benchmarks the Python XMem API on the BEAM dataset, defaulting to
the `1M` split from `Mohammadta/BEAM` on Hugging Face. It does not run or
compare the Go implementation.

BEAM rows contain long `chat` histories and stringified `probing_questions`.
The dataset card lists ten memory ability types: abstention, contradiction
resolution, event ordering, information extraction, instruction following,
knowledge update, multi-session reasoning, preference following, summarization,
and temporal reasoning.

## Dependencies

BEAM is distributed as parquet. Install `pyarrow` before reading the downloaded
dataset:

```bash
pip install pyarrow
```

## Smoke Check

```bash
python -m benchmarks.beam.run \
  --split 1M \
  --download \
  --dry-run \
  --limit 1
```

This downloads the BEAM 1M parquet file, validates parsing, counts ingest items,
and does not call XMem.

## Run Against XMem

```bash
export XMEM_API_KEY="..."

python -m benchmarks.beam.run \
  --split 1M \
  --dataset-path benchmarks/beam/data/1M-00000-of-00001.parquet \
  --api-base-url https://api.xmem.in \
  --output-dir benchmarks/beam/results/beam-1m
```

Outputs:

- `results.jsonl`: full per-question records with local proxy metrics
- `predictions.jsonl`: `question_id` and `hypothesis`
- `summary.json`: local exact/contains/token-F1 grouped by BEAM question type

Use BEAM's official/equivalent evaluator for publication-quality accuracy.
