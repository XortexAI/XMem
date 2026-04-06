## 2023-10-25 - Concurrent tool call execution in Retrieval Pipelines
**Learning:** Sequential execution of multiple tool calls in LangChain AIMessage (e.g. querying Pinecone and Neo4j simultaneously) creates a significant latency bottleneck in `RetrievalPipeline` and `CodeRetrievalPipeline`. A similar sequential bottleneck exists when searching across multiple repositories.
**Action:** Use `asyncio.gather` to execute independent tool calls and multi-namespace searches concurrently, then process results sequentially to safely update shared state.
