## 2024-04-11 - Parallelize Tool Executions in Retrieval Pipelines
**Learning:** Sequential execution of LLM tool calls in `RetrievalPipeline` and `CodeRetrievalPipeline` caused unnecessary blocking during query processing. Refactoring the loop to use `asyncio.gather` reduces latency.
**Action:** When working with LLM responses that request multiple tool calls, evaluate if the calls are independent. If so, process them concurrently with `asyncio.gather` and then update shared state sequentially after awaiting the results to ensure thread safety and optimal performance.
