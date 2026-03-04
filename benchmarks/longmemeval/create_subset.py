"""
Create a balanced subset of LongMemEval for benchmarking.

Selects N questions from each of the 6 categories (default: 10 each = 60 total)
and writes them as a new dataset file that the existing add.py / search.py /
evaluate.py scripts can consume directly.

Usage:
    python benchmarks/longmemeval/create_subset.py --per-category 10
    python benchmarks/longmemeval/create_subset.py --per-category 10 --seed 42
"""

import os
import sys
import json
import random
import argparse
from collections import Counter, defaultdict

# Setup paths
longmem_dir = os.path.dirname(os.path.abspath(__file__))
benchmarks_dir = os.path.dirname(longmem_dir)
xmem_root = os.path.dirname(benchmarks_dir)
sys.path.insert(0, xmem_root)


def main():
    parser = argparse.ArgumentParser(
        description="Create a balanced LongMemEval subset with N questions per category"
    )
    parser.add_argument(
        "--source", type=str, default="longmemeval_oracle.json",
        help="Source dataset file in LongMemEval/data/ (default: longmemeval_oracle.json)",
    )
    parser.add_argument(
        "--per-category", type=int, default=10,
        help="Number of questions to select per category (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output filename (default: longmemeval_subset_<N>pc.json)",
    )
    args = parser.parse_args()

    # ── Load source dataset ────────────────────────────────────────────
    # Try same path as add.py first: ../LongMemEval/data/ (sibling of xmem_root)
    data_path = os.path.join(os.path.dirname(xmem_root), "LongMemEval", "data", args.source)
    if not os.path.exists(data_path):
        # Fallback: inside xmem_root
        data_path = os.path.join(xmem_root, "LongMemEval", "data", args.source)
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at either location")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} questions from {args.source}")

    # ── Group by category ──────────────────────────────────────────────
    by_category: dict[str, list] = defaultdict(list)
    for entry in data:
        by_category[entry["question_type"]].append(entry)

    print(f"\nCategories ({len(by_category)}):")
    for cat in sorted(by_category.keys()):
        print(f"  {cat}: {len(by_category[cat])} questions")

    # ── Sample N per category ──────────────────────────────────────────
    random.seed(args.seed)
    n = args.per_category

    subset = []
    print(f"\nSampling {n} questions per category (seed={args.seed}):\n")

    for cat in sorted(by_category.keys()):
        pool = by_category[cat]
        k = min(n, len(pool))
        if k < n:
            print(f"  [WARNING] {cat} has only {len(pool)} questions, taking all {k}")
        selected = random.sample(pool, k)
        subset.extend(selected)
        print(f"  {cat}: selected {k} questions")

    total = len(subset)
    print(f"\n  Total subset: {total} questions")

    # ── Re-index for the pipeline  ─────────────────────────────────────
    # The add/search/evaluate scripts use 1-based indexing and process
    # sequentially, so we just write the subset as a flat list.
    # Shuffle to interleave categories (more realistic benchmark).
    random.shuffle(subset)

    # ── Verify distribution ────────────────────────────────────────────
    final_dist = Counter(q["question_type"] for q in subset)
    print(f"\nFinal distribution:")
    for cat, cnt in sorted(final_dist.items()):
        print(f"  {cat}: {cnt}")

    # ── Write output ───────────────────────────────────────────────────
    if args.output:
        out_name = args.output
    else:
        out_name = f"longmemeval_subset_{n}pc.json"

    # Write to where add.py/search.py expect it: ../LongMemEval/data/ (sibling of xmem_root)
    out_dir = os.path.join(os.path.dirname(xmem_root), "LongMemEval", "data")
    if not os.path.exists(out_dir):
        # Fallback: try inside xmem_root
        out_dir = os.path.join(xmem_root, "LongMemEval", "data")
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)

    print(f"\nSubset saved → {out_path}")
    print(f"  File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

    # ── Print next steps ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"NEXT STEPS:")
    print(f"{'='*60}")
    print(f"\n  1. ADD (ingest sessions into XMEM memory):")
    print(f"     python benchmarks/longmemeval/add.py --data {out_name} --evalset {total}")
    print(f"\n  2. SEARCH (answer questions using stored memory):")
    print(f"     python benchmarks/longmemeval/search.py --data {out_name} --evalset {total}")
    print(f"\n  3. EVALUATE (score answers with LLM judge):")
    print(f"     python benchmarks/longmemeval/evaluate.py --data {out_name} --evalset {total}")


if __name__ == "__main__":
    main()
