"""
Quick test script for the ingest pipeline.

Tests the full end-to-end flow:
- Classifier → Profile/Temporal/Summary extraction
- Judge (with real Pinecone + Neo4j lookups)
- Weaver (writes to real Pinecone + Neo4j)

Usage:
    python test_ingest.py
"""

import asyncio
import logging

# ── Only show logs from our agents, suppress everything else ──────
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)

# Enable only xmem agent/weaver/pipeline logs at INFO
for name in [
    "xmem.agents.classifier",
    "xmem.agents.profiler",
    "xmem.agents.temporal",
    "xmem.agents.summarizer",
    "xmem.agents.judge",
    "xmem.weaver",
    "xmem.pipelines.ingest",
    "xmem.graph.neo4j",
]:
    logging.getLogger(name).setLevel(logging.INFO)

# Suppress noisy third-party loggers
for name in [
    "httpx", "neo4j", "neo4j.notifications", "google_genai",
    "google_genai.models", "sentence_transformers",
    "src.storage.pinecone", "huggingface_hub",
]:
    logging.getLogger(name).setLevel(logging.WARNING)


from src.pipelines.ingest import IngestPipeline


async def test_ingest():
    print("\n" + "="*70)
    print("XMEM INGEST PIPELINE TEST")
    print("="*70 + "\n")

    # Initialize pipeline (connects to Pinecone + Neo4j)
    print("🔧 Initializing pipeline...")
    pipeline = IngestPipeline()
    print("✅ Pipeline initialized\n")

    # Test cases
    test_cases = [
        {
            "name": "Profile + Temporal",
            "user_query": "I just got promoted to Senior Engineer at Google! My birthday is on March 15th.",
            "agent_response": "Congratulations on your promotion!",
            "user_id": "test_user_001",
        },
        {
            "name": "Profile only",
            "user_query": "I love pizza and I'm vegetarian now.",
            "agent_response": "That's great! There are many delicious vegetarian pizza options.",
            "user_id": "test_user_001",
        },
        {
            "name": "Temporal only",
            "user_query": "I have a dentist appointment on April 20th at 2pm.",
            "agent_response": "I've noted your dentist appointment.",
            "user_id": "test_user_001",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"TEST {i}/{len(test_cases)}: {test['name']}")
        print(f"{'─'*70}")
        print(f"  Query: \"{test['user_query']}\"")
        print()

        try:
            result = await pipeline.run(
                user_query=test["user_query"],
                agent_response=test["agent_response"],
                user_id=test["user_id"],
            )

            # ── Classification ────────────────────────────────────────
            cr = result.get("classification_result")
            if cr and cr.classifications:
                print("\n  📋 CLASSIFIED AS:")
                for c in cr.classifications:
                    print(f"     → {c['source']:8s}  \"{c['query']}\"")

            # ── Profile ───────────────────────────────────────────────
            pj = result.get("profile_judge")
            pw = result.get("profile_weaver")
            if pj and pj.operations:
                print(f"\n  👤 PROFILE  (confidence: {pj.confidence})")
                for op in pj.operations:
                    print(f"     {op.type.value:6s}  {op.content}")
                if pw:
                    print(f"     ── weaver: {pw.succeeded}✓  {pw.skipped}⊘  {pw.failed}✗")

            # ── Temporal ──────────────────────────────────────────────
            tj = result.get("temporal_judge")
            tw = result.get("temporal_weaver")
            if tj and tj.operations:
                print(f"\n  📅 TEMPORAL  (confidence: {tj.confidence})")
                for op in tj.operations:
                    print(f"     {op.type.value:6s}  {op.content}")
                if tw:
                    print(f"     ── weaver: {tw.succeeded}✓  {tw.skipped}⊘  {tw.failed}✗")

            # ── Summary ───────────────────────────────────────────────
            sj = result.get("summary_judge")
            sw = result.get("summary_weaver")
            if sj and sj.operations:
                print(f"\n  📝 SUMMARY  (confidence: {sj.confidence})")
                for op in sj.operations:
                    print(f"     {op.type.value:6s}  {op.content}")
                if sw:
                    print(f"     ── weaver: {sw.succeeded}✓  {sw.skipped}⊘  {sw.failed}✗")

            print(f"\n  ✅ Test {i} passed")

        except Exception as e:
            print(f"\n  ❌ Test {i} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*70}\n")

    pipeline.close()
    print("🔒 Pipeline closed")

if __name__ == "__main__":
    asyncio.run(test_ingest())
