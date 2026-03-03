"""XMem Scanner Bot — indexes codebases into Pinecone + Neo4j + MongoDB.

Phase 1 (Indexer): Deterministic AST parsing → MongoDB + Pinecone + Neo4j
Phase 2 (Enricher): Async LLM enrichment of summaries
"""
