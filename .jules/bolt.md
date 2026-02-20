## 2024-05-22 - Sequential Processing Bottleneck in Ingest Pipeline
**Learning:** The `IngestPipeline` was processing extracted profile and temporal queries sequentially in a loop, leading to linear latency scaling with the number of queries. Since these operations involve independent LLM/DB calls, they are ideal candidates for parallelization.
**Action:** Always check for loops iterating over independent async tasks (like LLM calls) and consider `asyncio.gather` to parallelize them.
