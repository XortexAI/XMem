## 2024-03-02 - Parallelizing LLM extraction calls
**Learning:** Ingest pipelines often loop sequentially over items (like batched queries) and `await` an LLM call per item. This causes severe bottlenecks due to blocking on network IO sequentially. Using `asyncio.gather` with an `asyncio.Semaphore` is a clean way to parallelize this and significantly improve throughput.
**Action:** Always scan for `for` loops in `async` functions that `await` expensive external calls like model inference, and refactor them to run concurrently using bounded concurrency via semaphores.
