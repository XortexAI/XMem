# Benchmarks

## v2 Durable Ingest Staging Check

Use `scripts/benchmark_v2_ingest.py` to measure the new durable ingest route in
a staging environment without putting credentials in shell history, PR comments,
or logs. The script reads the Bearer token from an environment variable, posts a
single low-effort ingest payload to `/v2/memory/ingest`, then polls the returned
status URL until the durable job reaches a terminal state or the timeout expires.

```bash
export XMEM_API_KEY="..."
python scripts/benchmark_v2_ingest.py --base-url https://staging.example.com
```

For a production-like sample, pass an explicit payload file:

```bash
python scripts/benchmark_v2_ingest.py \
  --base-url https://staging.example.com \
  --payload-file ./benchmark-payload.json
```

The output intentionally excludes the API key and reports:

- `enqueue_latency_ms`: latency for the initial `POST /v2/memory/ingest` request.
- `completion_latency_ms`: elapsed time until the durable job is terminal.
- `final_status`: `succeeded`, `dead_letter`, or the last observed nonterminal
  status if the timeout expires.
- `poll_count`: number of status polling requests made after enqueue.

This separates the expected fast enqueue path from the total pipeline completion
time, which is the useful comparison for the v1 synchronous route.
