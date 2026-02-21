import os
import glob
import json
from typing import List, Dict

# Setup paths
longmem_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(longmem_dir, "results")

def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (p / 100) * (len(sorted_vals) - 1)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])

def fmt(m: float, s: float, scale: float = 100) -> str:
    return f"{m*scale:.2f} ± {s*scale:.2f}"

def main():
    print("=" * 80)
    print("LONGMEMEVAL BENCHMARK - MERGE RESULTS")
    print("=" * 80)
    
    # 1. Find all `eval_results_longmemeval_oracle_*.json` files
    # (Excluding the final merged file if it exists)
    pattern = os.path.join(results_dir, "eval_results_longmemeval_oracle_*.json")
    files = glob.glob(pattern)
    files = [f for f in files if "MERGED" not in f]
    
    if not files:
        print("[ERROR] No eval_results files found to merge.")
        return
        
    print(f"Found {len(files)} result batches:")
    for f in sorted(files):
        print(f"  - {os.path.basename(f)}")
        
    all_results = []
    
    for file_path in files:
        with open(file_path, "r") as f:
            data = json.load(f)
            all_results.extend(data.get("results", []))
            
    # Remove duplicates if any (just in case batches overlapped)
    unique_results = {}
    for r in all_results:
        # Use question ID as unique key
        unique_results[r["question_id"]] = r
        
    final_results = list(unique_results.values())
    # Sort them by their original index
    final_results.sort(key=lambda x: x.get("original_index", 0))
    
    print(f"\nTotal unique questions evaluated: {len(final_results)}")
    
    # Calculate merged metrics
    cat_data = {}
    for r in final_results:
        cat = r["question_type"]
        if cat not in cat_data:
            cat_data[cat] = {"j": [], "latencies": []}
        cat_data[cat]["j"].append(r["llm_judge_score"])
        cat_data[cat]["latencies"].append(r["latency_seconds"])
        
    all_j = [r["llm_judge_score"] for r in final_results]
    all_lat = [r["latency_seconds"] for r in final_results]
    
    overall_j_m, overall_j_s = mean(all_j), std(all_j)
    overall_p50 = percentile(sorted(all_lat), 50)
    overall_p95 = percentile(sorted(all_lat), 95)
    
    print("\n" + "=" * 90)
    print("FINAL COMBINED BENCHMARK REPORT")
    print("=" * 90)

    # Table header
    print(f"\n{'Category':<28} {'N':>4} | {'J ↑':>10} | {'P50':>7} {'P95':>7}")
    print("-" * 75)

    per_cat_summary = {}
    for cat in sorted(cat_data.keys()):
        d = cat_data[cat]
        n = len(d["j"])
        j_m, j_s = mean(d["j"]), std(d["j"])
        p50 = percentile(sorted(d["latencies"]), 50)
        p95 = percentile(sorted(d["latencies"]), 95)
        print(f"{cat:<28} {n:>4} | {fmt(j_m,0):>10} | {p50:>6.2f}s {p95:>6.2f}s")
        
        per_cat_summary[cat] = {
            "name": cat,
            "count": n,
            "j_mean": round(j_m, 4),
            "j_std": round(j_s, 4),
            "p50_latency": round(p50, 4),
            "p95_latency": round(p95, 4),
        }

    print("-" * 75)
    print(f"{'OVERALL COMBINED':<28} {len(final_results):>4} | {fmt(overall_j_m,0):>10} | {overall_p50:>6.2f}s {overall_p95:>6.2f}s")
    
    # Save the huge combined file
    merged_output = {
        "dataset": "longmemeval_oracle.json",
        "total_questions_combined": len(final_results),
        "summary": {
            "overall": {
                "j_mean": round(overall_j_m, 4),
                "j_std": round(overall_j_s, 4),
                "p50_latency": round(overall_p50, 4),
                "p95_latency": round(overall_p95, 4),
                "correct_count": sum(1 for r in final_results if r.get("llm_judge_score") == 1.0),
                "accuracy_pct": round(mean(all_j) * 100, 2)
            },
            "by_category": per_cat_summary
        },
        "results": final_results
    }
    
    out_file = os.path.join(results_dir, "eval_results_longmemeval_oracle_MERGED_ALL.json")
    with open(out_file, "w") as f:
        json.dump(merged_output, f, indent=2)
        
    print(f"\nSaved massive combined report -> {out_file}")

if __name__ == "__main__":
    main()
