"""Lightweight answer metrics and aggregation for LongMemEval outputs."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
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
    common = set(pred_tokens) & set(ref_tokens)
    overlap = sum(
        min(pred_tokens.count(token), ref_tokens.count(token))
        for token in common
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, reference: str) -> dict[str, float | bool]:
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)
    exact_match = normalized_prediction == normalized_reference
    contains = bool(
        normalized_reference
        and normalized_reference in normalized_prediction
    )
    return {
        "exact_match": exact_match,
        "contains": contains,
        "token_f1": round(token_f1(prediction, reference), 4),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"count": 0, "overall": {}, "by_question_type": {}}

    overall = _summarize_bucket(results)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        buckets[str(result.get("question_type") or "unknown")].append(result)
    return {
        "count": len(results),
        "overall": overall,
        "by_question_type": {
            question_type: _summarize_bucket(bucket)
            for question_type, bucket in sorted(buckets.items())
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _summarize_bucket(results: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(results)
    exact = sum(1 for result in results if result.get("metrics", {}).get("exact_match"))
    contains = sum(1 for result in results if result.get("metrics", {}).get("contains"))
    f1_scores = [
        float(result.get("metrics", {}).get("token_f1") or 0.0)
        for result in results
    ]
    avg_f1 = sum(f1_scores) / count
    avg_retrieve_ms = (
        sum(float(result.get("retrieve_elapsed_ms") or 0.0) for result in results)
        / count
    )
    return {
        "count": count,
        "exact_match": round(exact / count, 4),
        "contains": round(contains / count, 4),
        "token_f1": round(avg_f1, 4),
        "avg_retrieve_ms": round(avg_retrieve_ms, 2),
    }
