# LoCoMo Benchmark for XMem Python

This harness benchmarks the Python XMem API on LoCoMo only. It does not run or
compare the Go implementation.

LoCoMo contains ten long conversations. Each sample includes chronological
sessions under `conversation` and annotated question-answer items under `qa`
with `question`, `answer`, `category`, and optional evidence dialog ids.

## Smoke Check

```bash
python -m benchmarks.locomo.run --download --dry-run --limit 2
```

This downloads `locomo10.json`, validates parsing, counts ingest items, and
does not call XMem.

## Run Against XMem

```bash
export XMEM_API_KEY="..."

python -m benchmarks.locomo.run \
  --dataset-path benchmarks/locomo/data/locomo10.json \
  --api-base-url https://api.xmem.in \
  --output-dir benchmarks/locomo/results/run-001
```

Outputs:

- `results.jsonl`: full per-question records with local proxy metrics
- `predictions.jsonl`: `question_id` and `hypothesis`
- `summary.json`: local exact/contains/token-F1 grouped by LoCoMo category

Use LoCoMo's official/equivalent evaluator for publication-quality accuracy.
