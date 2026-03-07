
## 2026-03-07: Concurrency Optimization in IngestPipeline

Optimized LLM queries in extraction nodes (`_node_extract_profile`, `_node_extract_temporal`, `_node_extract_code`, `_node_extract_snippet`) of `IngestPipeline` by parallelizing them with `asyncio.gather` and adding an `asyncio.Semaphore(5)` to prevent unbounded concurrency that could lead to LLM API rate limiting. This limits concurrent execution of `agent.arun` calls.
