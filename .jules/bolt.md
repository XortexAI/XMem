## 2025-02-27 - [Weaver Pipeline Concurrent Execution]
**Learning:** Non-batched `_execute_one` and embedding generation in `_execute_batched_vector` inside Weaver can be executed concurrently to avoid blocking the event loop and speed up execution.
**Action:** Use `asyncio.gather` combined with `asyncio.get_running_loop().run_in_executor` for synchronous functions like `embed_fn` in batched processing where applicable.
