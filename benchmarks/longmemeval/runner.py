"""Benchmark orchestration for LongMemEval against the Python XMem API."""

from __future__ import annotations

import time
from typing import Any

from .client import XMemApiClient
from .config import BenchmarkConfig
from .dataset import (
    LongMemEvalExample,
    build_ingest_items,
    load_examples,
    select_examples,
)
from .metrics import (
    append_jsonl,
    read_jsonl,
    score_answer,
    summarize_results,
    write_json,
)


class LongMemEvalRunner:
    """Runs LongMemEval examples against the XMem Python HTTP service."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.results_path = config.output_dir / "results.jsonl"
        self.predictions_path = config.output_dir / "predictions.jsonl"
        self.summary_path = config.output_dir / "summary.json"

    async def run(self) -> dict[str, Any]:
        examples = select_examples(
            load_examples(self.config.dataset_path),
            offset=self.config.offset,
            limit=self.config.limit,
            question_type=self.config.question_type,
        )
        if self.config.dry_run:
            return self._dry_run_summary(examples)

        completed_ids = self._completed_question_ids() if self.config.resume else set()
        api_key = self.config.require_api_key()
        run_started = time.time()

        async with XMemApiClient(
            base_url=self.config.api_base_url,
            api_key=api_key,
            timeout_seconds=self.config.api_timeout_seconds,
            max_retries=self.config.max_retries,
            retry_backoff_seconds=self.config.retry_backoff_seconds,
        ) as client:
            for index, example in enumerate(examples, start=1):
                if example.question_id in completed_ids:
                    continue
                result = await self._run_example(
                    client,
                    example,
                    index=index,
                    total=len(examples),
                )
                append_jsonl(self.results_path, result)
                append_jsonl(
                    self.predictions_path,
                    {
                        "question_id": result["question_id"],
                        "hypothesis": result["prediction"],
                    },
                )

        all_results = read_jsonl(self.results_path)
        summary = summarize_results(all_results)
        summary["dataset_path"] = str(self.config.dataset_path)
        summary["api_base_url"] = self.config.api_base_url
        summary["duration_seconds"] = round(time.time() - run_started, 2)
        write_json(self.summary_path, summary)
        return summary

    async def _run_example(
        self,
        client: XMemApiClient,
        example: LongMemEvalExample,
        *,
        index: int,
        total: int,
    ) -> dict[str, Any]:
        user_id = f"{self.config.user_prefix}-{example.user_id_suffix}"
        ingest_count = 0
        ingest_elapsed_ms = 0.0

        if not self.config.skip_ingest:
            items = build_ingest_items(
                example,
                user_id=user_id,
                effort_level=self.config.effort_level,
            )
            ingest_count = len(items)
            ingest_elapsed_ms = await self._ingest_items(client, items)

        retrieve = await client.retrieve(
            {
                "query": example.question,
                "user_id": user_id,
                "top_k": self.config.top_k,
            }
        )
        prediction = str(retrieve.data.get("answer") or "")
        result = {
            "question_id": example.question_id,
            "question_type": example.question_type or "unknown",
            "question": example.question,
            "reference_answer": example.answer,
            "prediction": prediction,
            "metrics": score_answer(prediction, example.answer),
            "source_count": len(retrieve.data.get("sources") or []),
            "confidence": retrieve.data.get("confidence"),
            "user_id": user_id,
            "ingest_count": ingest_count,
            "ingest_elapsed_ms": round(ingest_elapsed_ms, 2),
            "retrieve_elapsed_ms": retrieve.elapsed_ms,
            "index": index,
            "total": total,
        }
        print(
            f"[{index}/{total}] {example.question_id}: "
            f"f1={result['metrics']['token_f1']} retrieve_ms={retrieve.elapsed_ms}"
        )
        return result

    async def _ingest_items(self, client: XMemApiClient, items: list[Any]) -> float:
        if not items:
            return 0.0

        elapsed_ms = 0.0
        processed = 0
        for start in range(0, len(items), self.config.batch_size):
            chunk = items[start : start + self.config.batch_size]
            payload = [item.__dict__ for item in chunk]
            if self.config.ingest_api_version == "v1":
                result = await client.batch_ingest_v1(payload)
                elapsed_ms += result.elapsed_ms
                processed += len(chunk)
                print(
                    f"[INGEST] processed={processed}/{len(items)}",
                    flush=True,
                )
                continue

            accepted = await client.batch_ingest_v2(payload)
            elapsed_ms += accepted.elapsed_ms
            status_url = str(accepted.data.get("status_url") or "")
            if not status_url:
                raise RuntimeError(
                    "XMem v2 batch ingest response did not include status_url"
                )
            status = await client.poll_job(
                status_url,
                interval_seconds=self.config.poll_interval_seconds,
                timeout_seconds=self.config.poll_timeout_seconds,
            )
            elapsed_ms += status.elapsed_ms
            if str(status.data.get("status") or "").lower() != "succeeded":
                error = status.data.get("error") or status.data
                raise RuntimeError(
                    f"XMem batch ingest job failed: {error}"
                )
            processed += len(chunk)
            print(
                f"[INGEST] processed={processed}/{len(items)}",
                flush=True,
            )
        return elapsed_ms

    def _completed_question_ids(self) -> set[str]:
        return {str(row.get("question_id")) for row in read_jsonl(self.results_path)}

    def _dry_run_summary(self, examples: list[LongMemEvalExample]) -> dict[str, Any]:
        ingest_counts = [
            len(
                build_ingest_items(
                    example,
                    user_id="dry-run",
                    effort_level=self.config.effort_level,
                )
            )
            for example in examples
        ]
        summary = {
            "dry_run": True,
            "dataset_path": str(self.config.dataset_path),
            "selected_examples": len(examples),
            "total_ingest_items": sum(ingest_counts),
            "min_ingest_items": min(ingest_counts) if ingest_counts else 0,
            "max_ingest_items": max(ingest_counts) if ingest_counts else 0,
            "question_types": sorted(
                {example.question_type or "unknown" for example in examples}
            ),
        }
        write_json(self.summary_path, summary)
        return summary
