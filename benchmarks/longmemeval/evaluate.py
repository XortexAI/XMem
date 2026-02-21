import os
import sys
import json
import asyncio
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List
import argparse

# Setup paths
longmem_dir = os.path.dirname(os.path.abspath(__file__))
benchmarks_dir = os.path.dirname(longmem_dir)
xmem_root = os.path.dirname(benchmarks_dir)
sys.path.insert(0, xmem_root)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(xmem_root, ".env"))

# Import LLM judge from XMEM general metrics
from benchmarks.metrics.llm_judge import evaluate_llm_judge_async, get_judge_model

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("xmem.benchmark.longmemeval.evaluate")


# ─── helpers ────────────────────────────────────────────────────────────────

def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (p / 100) * (len(sorted_vals) - 1)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def fmt(m: float, s: float, scale: float = 100) -> str:
    """Format as 'mean ± std' scaled to percentage."""
    return f"{m*scale:.2f} ± {s*scale:.2f}"


# ─── main ──────────────────────────────────────────────────────────────────

async def main():
    print("=" * 80)
    print("LONGMEMEVAL BENCHMARK - EVALUATION PHASE (XMEM LLM-as-Judge)")
    print("=" * 80)
    
    parser = argparse.ArgumentParser(description="LongMemEval Benchmark - EVAL Phase")
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

    # ── Load search results ─────────────────────────────────────────────
    results_dir = os.path.join(longmem_dir, "results")
    search_file = os.path.join(results_dir, f"search_results_{data_filename.split('.')[0]}_{start_q}-{start_q + eval_size - 1}.json")
    
    if not os.path.exists(search_file):
        print(f"\n  [ERROR] {search_file} not found! Run search.py first.")
        return

    with open(search_file, "r") as f:
        search_data = json.load(f)

    results = search_data["results"]
    search_metrics = search_data.get("metrics", {})

    print(f"\n[CONFIG]")
    print(f"  Input:      {search_file}")
    print(f"  Questions:  {len(results)}")
    print(f"  Judge model: gemini-2.5-flash (XMEM native judge)")
    
    print(f"\n  Note: To run the official LongMemEval GPT-4o evaluator:")
    print(f"  python LongMemEval/src/evaluation/evaluate_qa.py gpt-4o-mini \\")
    print(f"         benchmarks/longmemeval/results/hypotheses_{data_filename.split('.')[0]}_{start_q}-{start_q + eval_size - 1}.jsonl \\")
    print(f"         ../../../LongMemEval/data/{data_filename}")

    # ── Load add_results for compression metrics ────────────────────────
    add_results_file = os.path.join(results_dir, f"add_results_{data_filename.split('.')[0]}_{start_q}-{start_q + eval_size - 1}.json")
    add_metrics = {}
    if os.path.exists(add_results_file):
        with open(add_results_file, "r") as f:
            add_data = json.load(f)
            add_metrics = add_data.get("metrics", {})

    # ── Init judge ──────────────────────────────────────────────────────
    judge_model = get_judge_model()

    # ── Evaluate each question ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("EVALUATING ANSWERS")
    print("=" * 80)

    evaluated: List[Dict[str, Any]] = []

    for i, r in enumerate(results):
        question = r["question"]
        expected = r["expected_answer"]
        generated = r["generated_answer"]
        q_type = r["question_type"]
        is_abs = r.get("is_abstention", False)
        latency = r.get("latency_seconds", 0.0)

        expected_str = str(expected)
        generated_str = str(generated)

        print(f"\n--- Q{i+1}/{len(results)} [{q_type}] {'(ABSTENTION)' if is_abs else ''} ---")
        print(f"  Q: {question[:70]}...")
        print(f"  Expected: {expected_str[:50]}...")
        print(f"  Generated: {generated_str[:50]}...")

        # LLM judge
        try:
            j_score = await evaluate_llm_judge_async(question, expected_str, generated_str, model=judge_model)
        except Exception as e:
            logger.error(f"Judge error: {e}")
            j_score = 0.0

        verdict = "CORRECT" if j_score == 1.0 else "WRONG"
        print(f"  J={verdict}  Latency={latency:.2f}s")

        evaluated.append({
            **r,
            "llm_judge_score": j_score,
            "latency_seconds": latency
        })

    # ── Aggregate per category ──────────────────────────────────────────
    cat_data: Dict[str, Dict[str, list]] = defaultdict(lambda: {
        "j": [], "latencies": []
    })
    for r in evaluated:
        cat = r["question_type"]
        cat_data[cat]["j"].append(r["llm_judge_score"])
        cat_data[cat]["latencies"].append(r["latency_seconds"])

    all_j = [r["llm_judge_score"] for r in evaluated]
    all_lat = [r["latency_seconds"] for r in evaluated]

    # ── Print report ────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("FINAL BENCHMARK REPORT (Using native XMEM Judge)")
    print("=" * 90)

    # Table header
    print(f"\n{'Category':<28} {'N':>4} | {'J ↑':>10} | {'P50':>7} {'P95':>7}")
    print("-" * 75)

    for cat in sorted(cat_data.keys()):
        d = cat_data[cat]
        n = len(d["j"])
        j_m, j_s = mean(d["j"]), std(d["j"])
        p50 = percentile(sorted(d["latencies"]), 50)
        p95 = percentile(sorted(d["latencies"]), 95)
        print(f"{cat:<28} {n:>4} | {fmt(j_m,0):>10} | {p50:>6.2f}s {p95:>6.2f}s")

    print("-" * 75)
    overall_j_m, overall_j_s = mean(all_j), std(all_j)
    overall_p50 = percentile(sorted(all_lat), 50)
    overall_p95 = percentile(sorted(all_lat), 95)
    print(f"{'OVERALL':<28} {len(evaluated):>4} | {fmt(overall_j_m,0):>10} | {overall_p50:>6.2f}s {overall_p95:>6.2f}s")

    # ── Latency summary ────────────────────────────────────────────────
    total_search_time = search_metrics.get("total_search_time_seconds", sum(all_lat))
    add_latency = add_metrics.get("total_time_seconds", 0)
    total_e2e = total_search_time + add_latency

    print(f"\n{'─'*60}")
    print("LATENCY")
    print(f"{'─'*60}")
    print(f"  Search total:    {total_search_time:.2f}s")
    print(f"  Search P50:      {overall_p50:.3f}s")
    print(f"  Search P95:      {overall_p95:.3f}s")
    print(f"  Search Avg:      {mean(all_lat):.3f}s")
    if add_latency:
        print(f"  Add latency:     {add_latency:.2f}s")
        print(f"  Total (add+search): {total_e2e:.2f}s")

    # ── Save eval output ────────────────────────────────────────────────
    per_cat_summary = {}
    for cat in cat_data.keys():
        d = cat_data[cat]
        n = len(d["j"])
        per_cat_summary[cat] = {
            "name": cat,
            "count": n,
            "j_mean": round(mean(d["j"]), 4),
            "j_std": round(std(d["j"]), 4),
            "p50_latency": round(percentile(sorted(d["latencies"]), 50), 4),
            "p95_latency": round(percentile(sorted(d["latencies"]), 95), 4),
        }

    eval_output = {
        "dataset": data_filename,
        "run_range": f"Q{start_q}-Q{start_q + eval_size - 1}",
        "metadata": {
            "evaluation_timestamp": datetime.now().isoformat(),
            "judge_model": "gemini-2.5-flash"
        },
        "summary": {
            "total_questions": len(evaluated),
            "overall": {
                "j_mean": round(overall_j_m, 4),
                "j_std": round(overall_j_s, 4),
                "p50_latency": round(overall_p50, 4),
                "p95_latency": round(overall_p95, 4),
                "correct_count": sum(1 for r in evaluated if r["llm_judge_score"] == 1.0),
                "accuracy_pct": round(mean(all_j) * 100, 2)
            },
            "by_category": per_cat_summary,
            "latency": {
                "total_search_seconds": round(total_search_time, 2),
                "add_seconds": round(add_latency, 2) if add_latency else None,
                "total_e2e_seconds": round(total_e2e, 2) if add_latency else None,
                "search_p50": round(overall_p50, 4),
                "search_p95": round(overall_p95, 4),
                "search_avg": round(mean(all_lat), 4)
            }
        },
        "results": evaluated
    }

    output_file = os.path.join(results_dir, f"eval_results_{data_filename.split('.')[0]}_{start_q}-{start_q + eval_size - 1}.json")
    with open(output_file, "w") as f:
        json.dump(eval_output, f, indent=2)

    print(f"\n  Saved → {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
