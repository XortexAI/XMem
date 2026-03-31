## 2024-05-24 - [Optimizing code retrieval search pipelines]
**Learning:** Sequential namespace lookups for cross-repository search (e.g., in `_search_symbols`, `_search_files`) create N+1 query overhead in Pinecone/Neo4j searches, blocking I/O bound loops.
**Action:** Always prefer `asyncio.gather` for independent `_search_namespace` calls when iterating across multiple repositories to bound latency to the slowest search rather than the sum of all searches.
