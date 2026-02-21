import os
import sys
import json
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any
import argparse

# Setup paths
longmem_dir = os.path.dirname(os.path.abspath(__file__))
benchmarks_dir = os.path.dirname(longmem_dir)
xmem_root = os.path.dirname(benchmarks_dir)
sys.path.insert(0, xmem_root)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(xmem_root, ".env"))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("xmem.benchmark.longmemeval.search")

# ── Silence noisy internal loggers during benchmark ──────────────────────
for _noisy in (
    "xmem.pipelines.retrieval",
    "xmem.models",
    "src.storage.pinecone",
    "xmem.graph.neo4j",
    "neo4j",
    "httpx",
    "httpcore",
    "langchain_core",
    "langchain_google_genai",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


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
    
    parser = argparse.ArgumentParser(description="LongMemEval Benchmark - SEARCH Phase")
    parser.add_argument(
        "--data", type=str, default="longmemeval_oracle.json",
        help="Dataset file inside LongMemEval/data/ (default: longmemeval_oracle.json)"
    )
    parser.add_argument(
        "--start-question", type=int, default=1,
        help="Resume from this question ID (1-based index) to process"
    )
    parser.add_argument(
        "--evalset", type=int, default=50,
        help="Number of questions to process (default: 50)"
    )
    
    args = parser.parse_args()
    data_filename = args.data
    start_q = args.start_question
    eval_size = args.evalset

    print("=" * 80)
    print("LONGMEMEVAL BENCHMARK - SEARCH PHASE")
    print("=" * 80)

    # ── Load dataset ────────────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(xmem_root), "LongMemEval", "data", data_filename)
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_questions = len(data)
    end_q = min(start_q + eval_size - 1, total_questions)
    
    # Subset to process
    eval_data = data[start_q-1 : end_q]

    print(f"\n[CONFIG]")
    print(f"  Dataset:     {data_filename}")
    print(f"  Questions:   Processing Q{start_q} to Q{end_q} (out of {total_questions})")
    
    # Analyze question distribution
    from collections import Counter
    cat_counts = Counter(q["question_type"] for q in eval_data)
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat}: {count}")

    # ── Init retrieval ──────────────────────────────────────────────────
    print("\n[SETUP]")
    
    retrieval = RetrievalPipeline()
    logger.info("Retrieval pipeline initialized")
    
    base_user_id = f"longmemeval_{data_filename.replace('.json', '')}"

    # ── Answer questions ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ANSWERING QUESTIONS")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    hf_eval_output = []  # Specific format expected by LongMemEval evaluate_qa.py
    search_start = time.time()

    for idx, qa_entry in enumerate(eval_data, start_q):
        q_id = qa_entry["question_id"]
        q_type = qa_entry["question_type"]
        question = qa_entry["question"]
        expected = qa_entry["answer"]
        question_date = qa_entry["question_date"]
        
        # Each question has isolated history
        current_user_id = f"{base_user_id}_q{idx}"

        # Rewrite query to insert dynamic date for temporal context
        # XMEM temporal agent is very good if we say "Today is [Date]..."
        search_query = f"Context: Today's date is {question_date}.\\nQuestion: {question}"

        print(f"\n--- Q{idx} [{q_type}] ---")
        print(f"  Q ID: {q_id}")
        print(f"  Q:    {question}")
        print(f"  Exp:  {expected}")
        print(f"  User: {current_user_id}")
        
        # Check if this is an abstention question
        is_abstention = "_abs" in q_id

        q_start = time.time()
        try:
            # Use the retrieval pipeline to search and answer (90s timeout)
            retrieval_result = await asyncio.wait_for(
                retrieval.run(
                    query=search_query,
                    user_id=current_user_id,
                    top_k=20,  # Grab slightly more context since histories can be large
                ),
                timeout=90.0,
            )

            answer = retrieval_result.answer
            
            # Group sources by domain
            facts_found = [s for s in retrieval_result.sources if s.domain == "summary"]
            profiles_found = [s for s in retrieval_result.sources if s.domain == "profile"]
            events_found = [s for s in retrieval_result.sources if s.domain == "temporal"]

        except asyncio.TimeoutError:
            logger.warning(f"Q{idx} timed out after 90s — skipping")
            answer = "TIMEOUT"
            facts_found, profiles_found, events_found = [], [], []
        except Exception as e:
            logger.error(f"Search error: {e}")
            answer = f"ERROR: {e}"
            facts_found, profiles_found, events_found = [], [], []

        q_end = time.time()
        q_latency = q_end - q_start

        print(f"  Ans:  {answer[:120]}{'...' if len(answer) > 120 else ''}")
        print(f"  Lat:  {q_latency:.2f}s")
        print(f"  Src:  {len(facts_found)} facts, {len(events_found)} events, {len(profiles_found)} profiles")

        # Save for our own analysis format
        results.append({
            "original_index": idx,
            "question_id": q_id,
            "question": question,
            "search_query": search_query,
            "expected_answer": expected,
            "generated_answer": answer,
            "question_type": q_type,
            "is_abstention": is_abstention,
            "latency_seconds": round(q_latency, 4),
            "num_facts": len(facts_found),
            "num_events": len(events_found),
            "num_profiles": len(profiles_found)
        })
        
        # Save in the exact format required by LongMemEval's native GPT-4o evaluator
        hf_eval_output.append({
            "question_id": q_id,
            "hypothesis": answer
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

    # ── Save ────────────────────────────────────────────────────────────
    results_dir = os.path.join(longmem_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Native LongMemEval format for their HF scripts
    hf_output_file = os.path.join(results_dir, f"hypotheses_{data_filename.split('.')[0]}_{start_q}-{end_q}.jsonl")
    with open(hf_output_file, "w") as f:
        for item in hf_eval_output:
            f.write(json.dumps(item) + "\\n")
            
    # 2. XMEM detailed results format
    search_output = {
        "dataset": data_filename,
        "run_range": f"Q{start_q}-Q{end_q}",
        "total_questions": len(eval_data),
        "total_abstention": sum(1 for r in results if r["is_abstention"]),
        "metrics": {
            "total_search_time_seconds": round(total_search_time, 2),
            "avg_latency_seconds": round(avg_latency, 4),
            "p50_latency_seconds": round(p50_latency, 4),
            "p95_latency_seconds": round(p95_latency, 4),
        },
        "results": results
    }

    output_file = os.path.join(results_dir, f"search_results_{data_filename.split('.')[0]}_{start_q}-{end_q}.json")
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

    print(f"\n  Saved detailed results → {output_file}")
    print(f"  Saved LongMem hypotheses → {hf_output_file}")
    # print(f"\n  Next: python benchmarks/longmemeval/evaluate.py --data {data_filename} --start-question {start_q} --evalset {eval_size}")
    # Or native evaluation via LongMemEval
    print(f"\n  To evaluate using LongMemEval's metric:")
    print(f"  cd {os.path.join(xmem_root, 'LongMemEval', 'src', 'evaluation')}")
    print(f"  python evaluate_qa.py gpt-4o-mini {hf_output_file} {data_path}")

    # Cleanup
    retrieval.close()


if __name__ == "__main__":
    asyncio.run(main())
