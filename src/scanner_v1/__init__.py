"""
scanner_v1 — next-generation scanner built around a single database.

Design differences from scanner/ (v0):
  1. ONE store instead of three (Mongo + Pinecone + Neo4j → Neo4j only).
     Neo4j 5.11+ has native vector indexes, so the graph, the raw code,
     and the embeddings all live on the same nodes. No fan-out writes,
     no cross-store reconciliation, transactional deletes.

  2. DUAL vectors per symbol — fixes the first real shortcoming
     ("the embedded text does not contain the code"):
       - summary_embedding : produced from the natural-language summary
                             (good for NL queries, display text comes from here)
       - code_embedding    : produced from the raw code body
                             (good for identifier / literal / API queries)
     Retrieval later fuses the two lanes (RRF or weighted sum).

  3. Batched writes end-to-end (the PINECONE_BATCH=50 constant in v0
     existed but was never used — per-symbol roundtrips made full scans
     slow). v1 flushes symbols/files in batches on a single store.

  4. Cascading deletes live in the store, not the indexer. v0's Neo4j
     never got cleaned up on file/symbol deletion; v1 does DETACH DELETE
     in one Cypher call.

Files:
  schemas.py   — Neo4j labels, relationship types, property names,
                 vector-index definitions.
  store.py     — unified Neo4j client. Replaces CodeStore +
                 PineconeVectorStore + CodeGraphClient.
  embedder.py  — dual-lane embedder. Exposes embed_summary() and
                 embed_code(); this is where the fix for shortcoming #1
                 actually lives.
  indexer.py   — orchestrator. Clone/pull → parse → embed → upsert →
                 edges → scan state. Much shorter than v0 because
                 there is only one downstream store.
  enricher.py  — phase 2 async worker. Re-writes summary text only and
                 updates only the summary embedding lane (code lane is
                 permanent, so enrichment never touches it).
  runner.py    — CLI entry point (scan / enrich subcommands).

Reused unchanged from scanner/ (v0):
  ast_parser.py — already produces raw_code, signature, docstring,
                  imports, and calls. No v1-specific change needed yet.
  git_ops.py    — clone/pull/diff. No change.

Pydantic record models (SymbolRecord, FileRecord, etc.) in
src/schemas/code.py are still reused for validation.
"""

__version__ = "1.0.0-alpha"
