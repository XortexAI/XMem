
## 2024-05-28 - Pipeline Concurrent Tool Execution
**Learning:** LangGraph/LLM-based pipelines in this architecture often sequentialize independent tool calls (e.g., retrieving context from multiple namespaces). Given that these operations involve external vector and graph databases, running them sequentially acts as a significant latency bottleneck.
**Action:** Always scan for `for tc in ai_response.tool_calls:` loops and similar sequential aggregation patterns in pipeline execution logic. Replace them with concurrent execution using `asyncio.gather` and internal helper functions to drastically reduce total retrieval latency, especially when processing queries requiring broad context.
