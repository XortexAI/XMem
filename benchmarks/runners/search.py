"""
XMEM Benchmark - SEARCH Phase (Refactored)

Answers LoCoMo QA pairs using stored memories (vector DB + profiles + Neo4j graph).
Records per-question latency, F1, BLEU-1, and category for the eval phase.

Usage: python benchmarks/runners/search.py
"""
import os
import sys
import json
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any

# Setup paths
runners_dir = os.path.dirname(os.path.abspath(__file__))
benchmarks_dir = os.path.dirname(runners_dir)
xmem_root = os.path.dirname(benchmarks_dir)
sys.path.insert(0, xmem_root)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(xmem_root, ".env"))

from benchmarks.metrics.utils import calculate_f1_score, calculate_bleu_scores

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("xmem.benchmark.search")

# ── Silence noisy internal loggers during benchmark ──────────────────────
# Only show WARNING+ from pipeline internals so the output stays clean.
for _noisy in (
    "xmem.pipelines.retrieval",
    "xmem.models",
    "src.storage.pinecone",
    "xmem.graph.neo4j",
    "neo4j",               # Neo4j driver notifications (e.g. 'year' property warnings)
    "httpx",
    "httpcore",
    "langchain_core",
    "langchain_google_genai",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# Category mapping
CAT_NAMES = {1: "Single Hop", 2: "Temporal", 3: "Multi-Hop", 4: "Open Domain", 5: "Adversarial"}


def determine_query_user(
    question: str,
    speaker_a: str,
    speaker_b: str,
    user_a_id: str,
    user_b_id: str,
    evidence: List[str] = None
) -> str:
    """
    Determine which user to query based on the question.

    Checks for:
    1. Exact full name match (e.g. "Melanie" in question)
    2. Prefix/nickname match (e.g. "Mel" matches "Melanie", "Nick" matches "Nicholas")

    Falls back to user_a if neither speaker is detected.
    """
    q_lower = question.lower()
    a_lower = speaker_a.lower()
    b_lower = speaker_b.lower()

    # 1. Exact full name match (most reliable)
    a_exact = a_lower in q_lower
    b_exact = b_lower in q_lower

    if a_exact and not b_exact:
        return user_a_id
    if b_exact and not a_exact:
        return user_b_id
    if a_exact and b_exact:
        # Both mentioned — pick the one that appears first
        return user_a_id if q_lower.index(a_lower) <= q_lower.index(b_lower) else user_b_id

    # 2. Prefix/nickname match — check if any word in the question
    #    is a prefix (3+ chars) of either speaker name
    #    e.g. "Mel" → prefix of "Melanie", "Nick" → prefix of "Nicholas"
    import re
    words = set(re.findall(r"[a-z]+", q_lower))

    a_prefix = any(
        a_lower.startswith(w) and len(w) >= 3
        for w in words
    )
    b_prefix = any(
        b_lower.startswith(w) and len(w) >= 3
        for w in words
    )

    if a_prefix and not b_prefix:
        return user_a_id
    if b_prefix and not a_prefix:
        return user_b_id

    return user_a_id


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (p / 100) * (len(sorted_vals) - 1)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


async def main():
    from src.pipelines.retrieval import RetrievalPipeline
    from src.graph.neo4j_client import Neo4jClient
    from src.storage.pinecone import PineconeVectorStore
    from src.config import settings

    print("=" * 80)
    print("XMEM BENCHMARK - SEARCH PHASE")
    print("=" * 80)

    # ── Load dataset ────────────────────────────────────────────────────
    data_path = os.path.join(benchmarks_dir, "dataset", "locomo_10%testing.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversation = data["conversation"]
    qa_pairs = data.get("qa", [])
    sample_id = data.get("sample_id", "conv-0")

    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    # User IDs must match what was used in add.py
    user_a_id = f"{speaker_a.lower()}_benchmark"
    user_b_id = f"{speaker_b.lower()}_benchmark"

    # Filter out adversarial (cat 5)
    qa_pairs = [q for q in qa_pairs if q.get("category", 0) != 5]

    from collections import Counter
    cat_counts = Counter(q.get("category") for q in qa_pairs)

    print(f"\n[CONFIG]")
    print(f"  Dataset:     {data_path}")
    print(f"  Sample ID:   {sample_id}")
    print(f"  Speakers:    {speaker_a} ({user_a_id}), {speaker_b} ({user_b_id})")
    print(f"  QA pairs:    {len(qa_pairs)} (excl. adversarial)")
    for cat in sorted(cat_counts):
        print(f"    Cat {cat} ({CAT_NAMES.get(cat, '?')}): {cat_counts[cat]}")

    # ── Init retrieval ──────────────────────────────────────────────────
    print("\n[SETUP]")
    
    retrieval = RetrievalPipeline()
    logger.info("Retrieval pipeline initialized")

    # Check stored data counts
    try:
        vector_store = PineconeVectorStore()
        fact_count = vector_store.get_stats().total_vector_count
        logger.info(f"Vector store has {fact_count} vectors")
    except Exception as e:
        fact_count = 0
        logger.warning(f"Could not get vector count: {e}")

    print(f"  Facts in vector store: {fact_count}")

    if fact_count == 0:
        print("\n  [ERROR] No data! Run add.py first.")
        return

    # ── Answer questions ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ANSWERING QUESTIONS")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    search_start = time.time()

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        expected = str(qa["answer"])
        category = qa.get("category", 0)
        evidence = qa.get("evidence", [])

        # Determine which user to query
        query_user = determine_query_user(
            question, speaker_a, speaker_b, user_a_id, user_b_id, evidence
        )

        # Rewrite query (replace names with "User")
        search_query = question
        for name in [speaker_a, speaker_b]:
            search_query = search_query.replace(name, "User")
            search_query = search_query.replace(f"{name}'s", "User's")

        print(f"\n--- Q{i+1}/{len(qa_pairs)} [Cat {category}: {CAT_NAMES.get(category, '?')}] ---")
        print(f"  Q: {question}")
        print(f"  Expected: {expected}")
        print(f"  Query user: {query_user}")

        q_start = time.time()
        try:
            # Use the retrieval pipeline to search and answer (60s timeout per question)
            retrieval_result = await asyncio.wait_for(
                retrieval.run(
                    query=search_query,
                    user_id=query_user,
                    top_k=10,
                ),
                timeout=60.0,
            )

            answer = retrieval_result.answer
            
            # Group sources by domain
            facts_found = [s for s in retrieval_result.sources if s.domain == "summary"]
            profiles_found = [s for s in retrieval_result.sources if s.domain == "profile"]
            events_found = [s for s in retrieval_result.sources if s.domain == "temporal"]

        except asyncio.TimeoutError:
            logger.warning(f"Q{i+1} timed out after 60s — skipping")
            answer = "TIMEOUT"
            facts_found, profiles_found, events_found = [], [], []
        except Exception as e:
            logger.error(f"Search error: {e}")
            answer = f"ERROR: {e}"
            facts_found, profiles_found, events_found = [], [], []

        q_end = time.time()
        q_latency = q_end - q_start

        # Compute F1 and BLEU-1 inline
        f1 = calculate_f1_score(answer, expected)
        bleu1 = calculate_bleu_scores(answer, expected).get("bleu1", 0.0)

        print(f"  Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")
        print(f"  F1={f1:.3f}  B1={bleu1:.3f}  Latency={q_latency:.2f}s")
        print(f"  Sources: {len(facts_found)} facts, {len(events_found)} events, {len(profiles_found)} profiles")

        results.append({
            "question_id": i + 1,
            "question": question,
            "search_query": search_query,
            "expected_answer": expected,
            "generated_answer": answer,
            "category": category,
            "evidence": evidence,
            "query_user": query_user,
            "f1_score": f1,
            "bleu1_score": bleu1,
            "latency_seconds": round(q_latency, 4),
            "facts_retrieved": [{"content": s.content, "score": s.score} for s in facts_found],
            "events_retrieved": [{"content": s.content, "score": s.score, **s.metadata} for s in events_found],
            "profiles_retrieved": [{"content": s.content, "score": s.score} for s in profiles_found],
            "num_facts": len(facts_found),
            "num_events": len(events_found),
            "num_profiles": len(profiles_found)
        })

    search_end = time.time()
    total_search_time = search_end - search_start

    # ── Compute aggregate metrics ───────────────────────────────────────
    latencies = [r["latency_seconds"] for r in results]
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    p50_latency = percentile(latencies_sorted, 50)
    p95_latency = percentile(latencies_sorted, 95)
    avg_latency = sum(latencies) / n if n else 0

    # Per-category aggregates
    from collections import defaultdict
    cat_metrics = defaultdict(lambda: {"f1": [], "bleu1": [], "latencies": [], "count": 0})
    for r in results:
        cat = r["category"]
        cat_metrics[cat]["f1"].append(r["f1_score"])
        cat_metrics[cat]["bleu1"].append(r["bleu1_score"])
        cat_metrics[cat]["latencies"].append(r["latency_seconds"])
        cat_metrics[cat]["count"] += 1

    # ── Save ────────────────────────────────────────────────────────────
    search_output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": data_path,
            "sample_id": sample_id,
            "total_questions": len(qa_pairs),
            "speakers": {
                "a": {"name": speaker_a, "user_id": user_a_id},
                "b": {"name": speaker_b, "user_id": user_b_id}
            },
            "vector_store_facts": fact_count,
            "category_counts": {str(k): v for k, v in cat_counts.items()}
        },
        "metrics": {
            "total_search_time_seconds": round(total_search_time, 2),
            "avg_latency_seconds": round(avg_latency, 4),
            "p50_latency_seconds": round(p50_latency, 4),
            "p95_latency_seconds": round(p95_latency, 4),
            "avg_f1": round(sum(r["f1_score"] for r in results) / n, 4) if n else 0,
            "avg_bleu1": round(sum(r["bleu1_score"] for r in results) / n, 4) if n else 0,
            "by_category": {
                str(cat): {
                    "name": CAT_NAMES.get(cat, "?"),
                    "count": m["count"],
                    "avg_f1": round(sum(m["f1"]) / m["count"], 4),
                    "avg_bleu1": round(sum(m["bleu1"]) / m["count"], 4),
                    "p50_latency": round(percentile(sorted(m["latencies"]), 50), 4),
                    "p95_latency": round(percentile(sorted(m["latencies"]), 95), 4),
                }
                for cat, m in cat_metrics.items()
            }
        },
        "results": results
    }

    results_dir = os.path.join(benchmarks_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "search_results.json")
    with open(output_file, "w") as f:
        json.dump(search_output, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SEARCH PHASE COMPLETE")
    print("=" * 80)
    print(f"  Questions answered:  {len(results)}")
    print(f"  Total search time:   {total_search_time:.2f}s")
    print(f"  Avg latency:         {avg_latency:.3f}s")
    print(f"  P50 latency:         {p50_latency:.3f}s")
    print(f"  P95 latency:         {p95_latency:.3f}s")
    print(f"\n  Avg F1:   {sum(r['f1_score'] for r in results)/n:.4f}" if n else "")
    print(f"  Avg B1:   {sum(r['bleu1_score'] for r in results)/n:.4f}" if n else "")

    print(f"\n  Per-Category:")
    print(f"  {'Category':<15} {'N':>4} {'F1':>8} {'B1':>8} {'P50':>8} {'P95':>8}")
    print(f"  {'-'*55}")
    for cat in sorted(cat_metrics):
        m = cat_metrics[cat]
        avg_f1 = sum(m["f1"]) / m["count"]
        avg_b1 = sum(m["bleu1"]) / m["count"]
        p50 = percentile(sorted(m["latencies"]), 50)
        p95 = percentile(sorted(m["latencies"]), 95)
        print(f"  {CAT_NAMES.get(cat, '?'):<15} {m['count']:>4} {avg_f1:>8.4f} {avg_b1:>8.4f} {p50:>7.3f}s {p95:>7.3f}s")

    print(f"\n  Saved → {output_file}")
    print(f"\n  Next: python benchmarks/runners/evaluate.py")

    # Cleanup
    retrieval.close()


if __name__ == "__main__":
    asyncio.run(main())
