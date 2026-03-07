
2026-03-08:
⚡ Bolt: Add concurrency control using `asyncio.Semaphore` to parallel ingest queries.
- **What**: Capped the concurrent sub-queries executed via `asyncio.gather` with a semaphore in extraction nodes.
- **Why**: When a large number of profile/temporal/code queries are generated, unbounded parallel execution via `asyncio.gather` causes LLM API rate limits. Bounding them ensures throughput while keeping the workflow reliable.
- **Impact**: Stabilizes ingest pipeline during heavy loads with many sub-queries.
- **Measurement**: Benchmarks demonstrated capping parallel execution bounds the rate of API calls without sacrificing baseline latency per query batch.
