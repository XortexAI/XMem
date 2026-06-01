"""LLM-as-judge evaluation for BEAM benchmark outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from benchmarks.common.io import append_jsonl, read_jsonl, write_json


DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_JUDGE_MODEL = "gpt-4o"
PASS_THRESHOLD = 0.5

JUDGE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response
demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): <question>
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior
that the LLM response should demonstrate.

IMPORTANT: Pay careful attention to whether the rubric specifies positive
requirements or negative constraints.

## RESPONSIVENESS REQUIREMENT
A compliant response must be on-topic with respect to the QUESTION and attempt
to answer it. If the response does not address the QUESTION, score 0.0.

## SEMANTIC TOLERANCE RULES
Judge by meaning, not exact wording. Accept paraphrases and synonyms that
preserve intent. Ignore case, punctuation, and whitespace differences.
Numbers, currencies, dates, and durations may appear in equivalent forms.

## STYLE NEUTRALITY
Ignore tone, politeness, length, and flourish unless the rubric explicitly
requires a format or structure.

## SCORING SCALE
- 1.0: complete compliance.
- 0.5: partial compliance.
- 0.0: no compliance.

## OUTPUT FORMAT
Return only a JSON object:

{
  "score": 1.0,
  "reason": "why the rubric criterion was or was not satisfied"
}
""".strip()


def main() -> None:
    try:
        args = parse_args()
        evaluator = BeamEvaluator(
            results_path=args.results_path,
            output_dir=args.output_dir or args.results_path.parent,
            model=args.judge_model,
            api_key_env=args.openai_api_key_env,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
        )
        print(json.dumps(evaluator.run(), indent=2))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


class BeamEvaluator:
    def __init__(
        self,
        *,
        results_path: Path,
        output_dir: Path,
        model: str,
        api_key_env: str,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> None:
        self.results_path = results_path
        self.output_dir = output_dir
        self.model = model
        self.api_key_env = api_key_env
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.evaluations_path = output_dir / "evaluations.jsonl"
        self.summary_path = output_dir / "evaluation_summary.json"

    def run(self) -> dict[str, Any]:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key. Set {self.api_key_env}.")

        client = OpenAI(api_key=api_key)
        completed = {
            str(row.get("question_id"))
            for row in read_jsonl(self.evaluations_path)
        }
        started = time.time()
        for index, result in enumerate(read_jsonl(self.results_path), start=1):
            question_id = str(result.get("question_id") or "")
            if question_id in completed:
                continue
            evaluation = self._evaluate_result(client, result)
            append_jsonl(self.evaluations_path, evaluation)
            print(
                f"[EVAL {index}] {question_id}: "
                f"score={evaluation['judge_score']} "
                f"passed={evaluation['passed']}",
                flush=True,
            )

        evaluations = read_jsonl(self.evaluations_path)
        summary = summarize_evaluations(evaluations)
        summary["results_path"] = str(self.results_path)
        summary["judge_model"] = self.model
        summary["pass_threshold"] = PASS_THRESHOLD
        summary["duration_seconds"] = round(time.time() - started, 2)
        write_json(self.summary_path, summary)
        return summary

    def _evaluate_result(
        self,
        client: OpenAI,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        rubric = [str(item) for item in result.get("rubric") or []]
        if not rubric:
            reference = str(result.get("reference_answer") or "")
            rubric = [reference] if reference else ["The response answers correctly."]

        judge_items = [
            self._judge_rubric_item(
                client,
                question=str(result.get("question") or ""),
                prediction=str(result.get("prediction") or ""),
                rubric_item=item,
            )
            for item in rubric
        ]
        score = sum(float(item["score"]) for item in judge_items) / len(judge_items)
        return {
            "question_id": result.get("question_id"),
            "conversation_id": result.get("conversation_id"),
            "question_type": result.get("question_type") or "unknown",
            "question": result.get("question"),
            "reference_answer": result.get("reference_answer"),
            "prediction": result.get("prediction"),
            "judge_score": round(score, 4),
            "passed": score >= PASS_THRESHOLD,
            "judge_items": judge_items,
        }

    def _judge_rubric_item(
        self,
        client: OpenAI,
        *,
        question: str,
        prediction: str,
        rubric_item: str,
    ) -> dict[str, Any]:
        prompt = (
            JUDGE_PROMPT.replace("<question>", question)
            .replace("<rubric_item>", rubric_item)
            .replace("<llm_response>", prediction)
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                return normalize_judge_response(content)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
        raise RuntimeError(f"Judge call failed: {last_error}") from last_error


def normalize_judge_response(content: str) -> dict[str, Any]:
    payload = parse_json_response(content)
    raw_score = payload.get("score", 0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    score = min(1.0, max(0.0, score))
    return {
        "score": score,
        "reason": str(payload.get("reason") or ""),
    }


def parse_json_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object.")
    return payload


def summarize_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    if not evaluations:
        return {"count": 0, "overall": {}, "by_question_type": {}}

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for evaluation in evaluations:
        buckets[str(evaluation.get("question_type") or "unknown")].append(evaluation)

    return {
        "count": len(evaluations),
        "overall": summarize_bucket(evaluations),
        "by_question_type": {
            label: summarize_bucket(bucket)
            for label, bucket in sorted(buckets.items())
        },
    }


def summarize_bucket(evaluations: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(evaluations)
    passed = sum(1 for item in evaluations if item.get("passed"))
    score = sum(float(item.get("judge_score") or 0.0) for item in evaluations)
    return {
        "count": count,
        "passed": passed,
        "pass_rate": round(passed / count, 4),
        "avg_judge_score": round(score / count, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BEAM XMem results.")
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to BEAM results.jsonl generated by benchmarks.beam.run.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--openai-api-key-env", default=DEFAULT_OPENAI_API_KEY_ENV)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
