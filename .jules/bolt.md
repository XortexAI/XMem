
## 2024-05-20 - Batch multi-repository searches in pipelines
**Learning:** In pipelines performing multi-repository operations (like `CodeRetrievalPipeline`), sequential iterations over configured repositories for asynchronous tasks (e.g., `await self._search_namespace`) introduce an N+1 query overhead. The performance degrades linearly with the number of attached repositories.
**Action:** When working with multiple repositories concurrently, avoid sequential bottlenecks by batching operations using `asyncio.gather(*tasks)` mapped over async helpers and flattening the results, exactly as optimized in `src/pipelines/code_retrieval.py` for `_search_symbols` and `_search_files`.
