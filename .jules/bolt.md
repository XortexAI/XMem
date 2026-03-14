## 2024-05-24 - Weaver Pipeline Vector Add Operations Bottleneck
**Learning:** The Weaver pipeline (`src/pipelines/weaver.py`) batched vector `ADD` operations was iterating sequentially to generate embeddings, which was a performance bottleneck specific to this codebase due to the nature of synchronous external calls embedded in an asynchronous flow without leveraging the thread pool for each item in the batch.
**Action:** Replaced the sequential iteration with `asyncio.gather` and `asyncio.get_running_loop().run_in_executor` to perform embedding generations concurrently.
