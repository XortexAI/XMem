# 2026-03-07: Concurrency Optimization in IngestPipeline

## What
Added `asyncio.Semaphore(5)` to limit the maximum concurrency when calling extraction agents (`profiler`, `temporal`, `code_agent`, `snippet_agent`) within `src/pipelines/ingest.py`. Wrapped the `arun` invocations inside inline `async def` helper functions (`_run_profiler`, `_run_temporal`, etc.) so that the `async with` context manager could correctly enforce the semaphore limit for generator expressions inside `asyncio.gather`.

## Why
The original code successfully parallelized the extraction LLM queries via `asyncio.gather`. However, without an upper bound, long lists of batched queries could simultaneously fire off tens or hundreds of requests to the LLM backend. This runs the severe risk of triggering 429 Too Many Requests errors and API rate limits, failing the extraction or causing lengthy retries.

## Impact
- **Safety**: Prevents unbounded concurrent requests to external API providers.
- **Latency Consistency**: Sub-queries execute in predictable batches of up to 5, providing steady performance.
- **Stability**: Prevents event loop starvation and backend timeouts on massive workloads.

## Measurement
A `benchmark.py` testing script successfully validated that for 15 requests, the execution was properly bounded to exactly `5` concurrent processes at peak, completing cleanly in ~0.3s (simulating batch latency) vs an expected unbound explosion of calls.
