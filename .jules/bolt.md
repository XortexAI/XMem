## 2024-05-20 - Concurrent embeddings for batch writes in Weaver
**Learning:** The Weaver pipeline previously executed batch vector ADD operations by generating embeddings synchronously in a loop. This created a performance bottleneck by blocking the event loop with synchronous network calls.
**Action:** Use `asyncio.gather` combined with `loop.run_in_executor` to execute all embedding requests for a batch concurrently before passing them to the vector store.
