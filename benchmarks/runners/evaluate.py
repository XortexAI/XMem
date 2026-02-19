"""
XMEM Benchmark - EVALUATION Phase (Refactored)

Takes search_results.json and:
1. Computes F1, BLEU-1 (already in search results – carried forward)
2. Runs LLM-as-Judge (J) on every Q/A
3. Aggregates scores per category (Single Hop, Multi-Hop, Open Domain, Temporal)
4. Computes latency p50/p95 and compression metrics
5. Prints a final report table

Usage: python benchmarks/runners/evaluate.py
"""
import os
import sys
import json
import asyncio
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List

# Setup paths
runners_dir = os.path.dirname(os.path.abspath(__file__))
benchmarks_dir = os.path.dirname(runners_dir)
xmem_root = os.path.dirname(benchmarks_dir)
sys.path.insert(0, xmem_root)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(xmem_root, ".env"))

from benchmarks.metrics.utils import calculate_f1_score, calculate_bleu_scores
from benchmarks.metrics.llm_judge import evaluate_llm_judge_async, get_judge_model

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("xmem.benchmark.evaluate")

# Category mapping
CAT_NAMES = {1: "Single Hop", 2: "Temporal", 3: "Multi-Hop", 4: "Open Domain"}
CAT_ORDER = [1, 3, 4, 2]  # display order matching LoCoMo paper


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
    print("XMEM BENCHMARK - EVALUATION PHASE (LLM-as-Judge)")
    print("=" * 80)

    # ── Load search results ─────────────────────────────────────────────
    results_dir = os.path.join(benchmarks_dir, "results")
    search_file = os.path.join(results_dir, "search_results.json")
    
    if not os.path.exists(search_file):
        print(f"\n  [ERROR] {search_file} not found! Run search.py first.")
        return

    with open(search_file, "r") as f:
        search_data = json.load(f)

    results = search_data["results"]
    search_metrics = search_data.get("metrics", {})
    metadata = search_data.get("metadata", {})

    print(f"\n[CONFIG]")
    print(f"  Input:      {search_file}")
    print(f"  Questions:  {len(results)}")
    print(f"  Judge model: gemini-2.5-flash")

    # ── Load add_results for compression metrics ────────────────────────
    add_results_file = os.path.join(results_dir, "add_results.json")
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
        category = r.get("category", 0)
        f1 = r.get("f1_score", calculate_f1_score(generated, expected))
        bleu1 = r.get("bleu1_score", calculate_bleu_scores(generated, expected).get("bleu1", 0.0))
        latency = r.get("latency_seconds", 0.0)

        print(f"\n--- Q{i+1}/{len(results)} [Cat {category}: {CAT_NAMES.get(category, '?')}] ---")
        print(f"  Q: {question[:70]}...")
        print(f"  Expected: {expected[:50]}...")
        print(f"  Generated: {generated[:50]}...")

        # LLM judge
        try:
            j_score = await evaluate_llm_judge_async(question, expected, generated, model=judge_model)
        except Exception as e:
            logger.error(f"Judge error: {e}")
            j_score = 0.0

        verdict = "CORRECT" if j_score == 1.0 else "WRONG"
        print(f"  F1={f1:.3f}  B1={bleu1:.3f}  J={verdict}  Latency={latency:.2f}s")

        evaluated.append({
            **r,
            "f1_score": f1,
            "bleu1_score": bleu1,
            "llm_judge_score": j_score,
            "latency_seconds": latency
        })

    # ── Aggregate per category ──────────────────────────────────────────
    cat_data: Dict[int, Dict[str, list]] = defaultdict(lambda: {
        "f1": [], "bleu1": [], "j": [], "latencies": []
    })
    for r in evaluated:
        cat = r["category"]
        cat_data[cat]["f1"].append(r["f1_score"])
        cat_data[cat]["bleu1"].append(r["bleu1_score"])
        cat_data[cat]["j"].append(r["llm_judge_score"])
        cat_data[cat]["latencies"].append(r["latency_seconds"])

    all_f1 = [r["f1_score"] for r in evaluated]
    all_b1 = [r["bleu1_score"] for r in evaluated]
    all_j = [r["llm_judge_score"] for r in evaluated]
    all_lat = [r["latency_seconds"] for r in evaluated]

    # ── Print report ────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("FINAL BENCHMARK REPORT")
    print("=" * 100)

    # Table header
    print(f"\n{'Category':<15} {'N':>4} | {'F1 ↑':>14} | {'B1 ↑':>14} | {'J ↑':>14} | {'P50':>7} {'P95':>7}")
    print("-" * 90)

    for cat in CAT_ORDER:
        if cat not in cat_data:
            continue
        d = cat_data[cat]
        n = len(d["f1"])
        f1_m, f1_s = mean(d["f1"]), std(d["f1"])
        b1_m, b1_s = mean(d["bleu1"]), std(d["bleu1"])
        j_m, j_s = mean(d["j"]), std(d["j"])
        p50 = percentile(sorted(d["latencies"]), 50)
        p95 = percentile(sorted(d["latencies"]), 95)
        print(f"{CAT_NAMES[cat]:<15} {n:>4} | {fmt(f1_m,f1_s):>14} | {fmt(b1_m,b1_s):>14} | {fmt(j_m,j_s):>14} | {p50:>6.2f}s {p95:>6.2f}s")

    print("-" * 90)
    overall_f1_m, overall_f1_s = mean(all_f1), std(all_f1)
    overall_b1_m, overall_b1_s = mean(all_b1), std(all_b1)
    overall_j_m, overall_j_s = mean(all_j), std(all_j)
    overall_p50 = percentile(sorted(all_lat), 50)
    overall_p95 = percentile(sorted(all_lat), 95)
    print(f"{'OVERALL':<15} {len(evaluated):>4} | {fmt(overall_f1_m,overall_f1_s):>14} | {fmt(overall_b1_m,overall_b1_s):>14} | {fmt(overall_j_m,overall_j_s):>14} | {overall_p50:>6.2f}s {overall_p95:>6.2f}s")

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

    # ── Compression / Token summary ─────────────────────────────────────
    if add_metrics:
        input_tok = add_metrics.get("input_tokens", 0)
        stored_tok = add_metrics.get("stored_tokens", 0)
        comp_ratio = add_metrics.get("compression_ratio", 0)
        reduction = add_metrics.get("reduction_percent", 0)

        print(f"\n{'─'*60}")
        print("TOKEN COMPRESSION")
        print(f"{'─'*60}")
        print(f"  Input tokens:       ~{input_tok}")
        print(f"  Stored tokens:      ~{stored_tok}")
        print(f"  Compression ratio:  {comp_ratio:.1f}x")
        print(f"  Reduction:          {reduction:.1f}%")

    # ── Detailed results table ──────────────────────────────────────────
    print(f"\n{'─'*100}")
    print("DETAILED RESULTS")
    print(f"{'─'*100}")
    print(f"{'#':<3} {'Category':<12} {'Question':<38} {'Expected':<18} {'F1':>5} {'B1':>5} {'J':>3} {'Lat':>6}")
    print("-" * 100)
    for r in evaluated:
        q = r["question"][:35] + "..." if len(r["question"]) > 38 else r["question"]
        e = str(r["expected_answer"])[:15] + "..." if len(str(r["expected_answer"])) > 18 else str(r["expected_answer"])
        cat_name = CAT_NAMES.get(r["category"], "?")[:10]
        j_sym = "Y" if r["llm_judge_score"] == 1.0 else "N"
        print(f"{r['question_id']:<3} {cat_name:<12} {q:<38} {e:<18} {r['f1_score']:>5.2f} {r['bleu1_score']:>5.2f} {j_sym:>3} {r['latency_seconds']:>5.2f}s")

    # ── Save eval output ────────────────────────────────────────────────
    per_cat_summary = {}
    for cat in CAT_ORDER:
        if cat not in cat_data:
            continue
        d = cat_data[cat]
        n = len(d["f1"])
        per_cat_summary[str(cat)] = {
            "name": CAT_NAMES[cat],
            "count": n,
            "f1_mean": round(mean(d["f1"]), 4),
            "f1_std": round(std(d["f1"]), 4),
            "bleu1_mean": round(mean(d["bleu1"]), 4),
            "bleu1_std": round(std(d["bleu1"]), 4),
            "j_mean": round(mean(d["j"]), 4),
            "j_std": round(std(d["j"]), 4),
            "p50_latency": round(percentile(sorted(d["latencies"]), 50), 4),
            "p95_latency": round(percentile(sorted(d["latencies"]), 95), 4),
        }

    eval_output = {
        "metadata": {
            **metadata,
            "evaluation_timestamp": datetime.now().isoformat(),
            "judge_model": "gemini-2.5-flash"
        },
        "summary": {
            "total_questions": len(evaluated),
            "overall": {
                "f1_mean": round(overall_f1_m, 4),
                "f1_std": round(overall_f1_s, 4),
                "bleu1_mean": round(overall_b1_m, 4),
                "bleu1_std": round(overall_b1_s, 4),
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
            },
            "compression": {
                "input_tokens": add_metrics.get("input_tokens"),
                "stored_tokens": add_metrics.get("stored_tokens"),
                "compression_ratio": add_metrics.get("compression_ratio"),
                "reduction_percent": add_metrics.get("reduction_percent")
            } if add_metrics else None
        },
        "results": evaluated
    }

    output_file = os.path.join(results_dir, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump(eval_output, f, indent=2)

    print(f"\n  Saved → {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
