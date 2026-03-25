## 2026-03-25 - Concurrent Execution in Weaver Pipeline
**Learning:** Synchronous embedding generation in the batch flush blocked the async event loop. Furthermore, non-batched operations running sequentially introduced unnecessary latency.
**Action:** Use asyncio.gather and asyncio.get_running_loop().run_in_executor to execute independent operations and blocking synchronous network calls concurrently.
