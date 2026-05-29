"""Shared lightweight answer metrics for benchmark smoke summaries."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, reference: str) -> dict[str, float | bool]:
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    return {
        "exact_match": normalized_prediction == normalized_reference,
        "contains": bool(
            normalized_reference
            and normalized_reference in normalized_prediction
        ),
        "token_f1": round(token_f1(prediction, reference), 4),
    }


def summarize_results(
    results: list[dict[str, Any]],
    *,
    group_field: str = "question_type",
) -> dict[str, Any]:
    if not results:
        return {"count": 0, "overall": {}, f"by_{group_field}": {}}

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        buckets[str(result.get(group_field) or "unknown")].append(result)
    return {
        "count": len(results),
        "overall": _summarize_bucket(results),
        f"by_{group_field}": {
            label: _summarize_bucket(bucket)
            for label, bucket in sorted(buckets.items())
        },
    }


def _summarize_bucket(results: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(results)
    exact = sum(1 for result in results if result.get("metrics", {}).get("exact_match"))
    contains = sum(1 for result in results if result.get("metrics", {}).get("contains"))
    f1_scores = [
        float(result.get("metrics", {}).get("token_f1") or 0.0)
        for result in results
    ]
    avg_retrieve_ms = (
        sum(float(result.get("retrieve_elapsed_ms") or 0.0) for result in results)
        / count
    )
    return {
        "count": count,
        "exact_match": round(exact / count, 4),
        "contains": round(contains / count, 4),
        "token_f1": round(sum(f1_scores) / count, 4),
        "avg_retrieve_ms": round(avg_retrieve_ms, 2),
    }
