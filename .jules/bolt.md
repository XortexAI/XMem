
## 2024-05-24 - [Optimize Code Retrieval LLM Tool Calling]
**Learning:** Sequential loops for LLM tool executions (`ai_response.tool_calls`) and multi-repository searches (`results.extend()` over `self.repos`) create a bottleneck as multiple independent I/O-bound requests wait for each other.
**Action:** Replace these sequential loops with `asyncio.gather` for concurrent execution, avoiding sequential bottlenecks, while ensuring the results are processed sequentially afterward to ensure safe appending to shared lists like `sources` and `tool_messages`.
