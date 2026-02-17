"""
Test script for the retrieval pipeline.

Runs several queries against data already ingested by test_ingest.py.
Make sure you've run test_ingest.py first so there's data to retrieve!

Usage:
    python test_retrieval.py
"""

import asyncio
import logging

# ── Logging: only show our stuff ──────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(message)s")

for name in [
    "xmem.pipelines.retrieval",
]:
    logging.getLogger(name).setLevel(logging.INFO)

for name in [
    "httpx", "neo4j", "neo4j.notifications", "google_genai",
    "google_genai.models", "sentence_transformers",
    "src.storage.pinecone", "huggingface_hub",
]:
    logging.getLogger(name).setLevel(logging.WARNING)


from src.pipelines.retrieval import RetrievalPipeline


async def test_retrieval():
    print("\n" + "=" * 70)
    print("XMEM RETRIEVAL PIPELINE TEST")
    print("=" * 70 + "\n")

    print("🔧 Initializing pipeline...")
    pipeline = RetrievalPipeline()
    print("✅ Pipeline initialized\n")

    test_queries = [
        {
            "name": "Profile lookup",
            "query": "What food does the user like?",
            "user_id": "test_user_001",
        },
        {
            "name": "Temporal lookup",
            "query": "When is my dentist appointment?",
            "user_id": "test_user_001",
        },
        {
            "name": "General / summary",
            "query": "What do you know about me?",
            "user_id": "test_user_001",
        },
        {
            "name": "Profile + Temporal",
            "query": "Where do I work and when is my birthday?",
            "user_id": "test_user_001",
        },
    ]

    for i, test in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"TEST {i}/{len(test_queries)}: {test['name']}")
        print(f"{'─' * 70}")
        print(f"  Question: \"{test['query']}\"")
        print()

        try:
            result = await pipeline.run(
                query=test["query"],
                user_id=test["user_id"],
            )

            # Show sources
            if result.sources:
                print(f"\n  📦 SOURCES ({len(result.sources)}):")
                for s in result.sources:
                    score_str = f" ({s.score:.2f})" if s.score > 0 else ""
                    print(f"     [{s.domain}]{score_str}  {s.content[:80]}")

            # Show answer
            print(f"\n  💬 ANSWER:")
            for line in result.answer.strip().splitlines():
                print(f"     {line}")

            print(f"\n  ✅ Test {i} passed (confidence: {result.confidence:.2f})")

        except Exception as e:
            print(f"\n  ❌ Test {i} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("ALL RETRIEVAL TESTS COMPLETE")
    print(f"{'=' * 70}\n")

    pipeline.close()
    print("🔒 Pipeline closed")


if __name__ == "__main__":
    asyncio.run(test_retrieval())
