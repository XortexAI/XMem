# Bolt's Performance Journal

## 2026-03-10 - Concurrent Tool Calls in Pipeline Execution
**Learning:** Sequential await loops for multiple backend/network operations (like LLM tool calls targeting different data stores) create a significant performance bottleneck. In `RetrievalPipeline`, tools like `search_temporal` and `search_profile` were executed sequentially, causing total retrieval latency to scale linearly with the number of tools requested.
**Action:** Replace sequential loops with `asyncio.gather(*tasks)` for concurrent execution when operations are independent. Process the resulting list sequentially to ensure order preservation when appending to shared lists or building context strings.
