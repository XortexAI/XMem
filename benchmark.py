#!/usr/bin/env python3
"""
XMem Python Benchmark — real LLM calls, timed per-agent.

Wraps the real LangChain model with a timing proxy so we can separate:
  - Total time (end-to-end per agent)
  - LLM time   (actual network round-trip to the provider)
  - Overhead   (prompt building, response parsing, orchestration)

Run:
    cd XMem
    python benchmark.py

Compare the output with:  cd xmem-go && go run ./cmd/benchmark/
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import patch

logging.basicConfig(level=logging.WARNING)

from langchain_core.language_models import BaseChatModel

from src.models import get_model
from src.pipelines.ingest import IngestPipeline
from src.pipelines.retrieval import RetrievalPipeline

TEST_QUERY = (
    "My name is Bob, and I started a new job at Vercel as a frontend developer today!"
)
TEST_RESPONSE = "Congratulations on your new role Bob! That's wonderful news."
SESSION_DT = "4:00 pm on 20 May, 2026"
BENCH_USER_ID = "bench-user"


def indent_lines(text: str, spaces: int = 4) -> str:
    indent = " " * spaces
    return "\n".join(indent + line for line in str(text).splitlines())


def estimate_tokens(text: str) -> int:
    text = str(text or "").strip()
    if not text:
        return 0
    return (len(text) + 3) // 4


def message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "user"))
    role = getattr(message, "type", None) or getattr(message, "role", None)
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    return str(role or "user")


def message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    parts.append("<image_url>")
                else:
                    parts.append(json.dumps(item, default=str))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def response_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def message_tokens(messages: Any) -> int:
    if isinstance(messages, (str, bytes)):
        return estimate_tokens(str(messages))
    if not isinstance(messages, list):
        return estimate_tokens(message_content(messages))
    return sum(estimate_tokens(message_role(m)) + estimate_tokens(message_content(m)) for m in messages)


def response_tokens(response: Any) -> tuple[int, int, bool]:
    usage = getattr(response, "usage_metadata", None) or {}
    if isinstance(usage, dict):
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        if input_tokens or output_tokens:
            return input_tokens, output_tokens, False

    metadata = getattr(response, "response_metadata", None) or {}
    if isinstance(metadata, dict):
        token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
        if isinstance(token_usage, dict):
            input_tokens = int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0)
            output_tokens = int(token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0)
            if input_tokens or output_tokens:
                return input_tokens, output_tokens, False
    return 0, estimate_tokens(response_content(response)), True


def infer_agent(messages: Any, default_agent: str) -> str:
    texts = []
    if isinstance(messages, list):
        texts = [message_content(m) for m in messages]
    else:
        texts = [str(messages)]
    combined = "\n".join(texts).lower()
    if "analyze this user input" in combined or "intent router" in combined:
        return "classifier"
    if "extract all temporal events" in combined or "event extraction assistant" in combined:
        return "temporal"
    if "summarize this conversation" in combined:
        return "summarizer"
    if "profile facts" in combined or "extracts structured user facts" in combined or "extract important user profiles" in combined or "build a complete picture of the user" in combined:
        return "profiler"
    if "## domain:" in combined or "judge agent" in combined:
        return "judge"
    if "image analysis" in combined or "analyse this image" in combined:
        return "image"
    if "extract code snippets" in combined:
        return "snippet"
    if "extract code annotations" in combined:
        return "code"
    return default_agent


class TimedModel:
    """Proxy that wraps a real LangChain model and traces/times LLM calls."""

    def __init__(
        self,
        inner: BaseChatModel,
        tracker: Optional[dict[str, Any]] = None,
        agent: str = "model",
        trace: bool = True,
    ):
        self._inner = inner
        self._tracker = tracker or {
            "llm_time_ns": 0,
            "call_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "agents": {},
        }
        self._agent = agent
        self._trace = trace

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _count_tokens(self, messages: Any, result: Any) -> tuple[int, int]:
        input_tokens, output_tokens, _ = response_tokens(result)
        if input_tokens == 0:
            input_tokens = message_tokens(messages)
        return input_tokens, output_tokens

    def _record_metrics(self, agent: str, elapsed_ns: int, input_tokens: int, output_tokens: int) -> None:
        self._tracker["llm_time_ns"] += elapsed_ns
        self._tracker["call_count"] += 1
        self._tracker["input_tokens"] += input_tokens
        self._tracker["output_tokens"] += output_tokens

        agents = self._tracker.setdefault("agents", {})
        metrics = agents.setdefault(
            agent,
            {"llm_time_ns": 0, "call_count": 0, "input_tokens": 0, "output_tokens": 0},
        )
        metrics["llm_time_ns"] += elapsed_ns
        metrics["call_count"] += 1
        metrics["input_tokens"] += input_tokens
        metrics["output_tokens"] += output_tokens

    async def ainvoke(self, messages: Any, **kwargs) -> Any:
        start = time.perf_counter_ns()
        result = await self._inner.ainvoke(messages, **kwargs)
        elapsed_ns = time.perf_counter_ns() - start
        input_tokens, output_tokens = self._count_tokens(messages, result)
        agent = infer_agent(messages, self._agent)
        self._record_metrics(agent, elapsed_ns, input_tokens, output_tokens)
        if self._trace:
            self._print_trace("ainvoke", messages, result, elapsed_ns)
        return result

    def invoke(self, messages: Any, **kwargs) -> Any:
        start = time.perf_counter_ns()
        result = self._inner.invoke(messages, **kwargs)
        elapsed_ns = time.perf_counter_ns() - start
        input_tokens, output_tokens = self._count_tokens(messages, result)
        agent = infer_agent(messages, self._agent)
        self._record_metrics(agent, elapsed_ns, input_tokens, output_tokens)
        if self._trace:
            self._print_trace("invoke", messages, result, elapsed_ns)
        return result

    def bind_tools(self, tools, **kwargs):
        return TimedModel(
            self._inner.bind_tools(tools, **kwargs),
            tracker=self._tracker,
            agent=self._agent,
            trace=self._trace,
        )

    def for_agent(self, agent: str) -> "TimedModel":
        return TimedModel(self._inner, tracker=self._tracker, agent=agent, trace=self._trace)

    def _print_trace(self, call_type: str, messages: Any, result: Any, elapsed_ns: int) -> None:
        agent = infer_agent(messages, self._agent)

        print()
        print(f"  \033[35m┌─── [LLM Call: {agent} / {call_type}] ───────────────────────────────────┐\033[0m")
        if isinstance(messages, list):
            for message in messages:
                role = message_role(message)
                if role == "system":
                    continue
                print(f"  \033[35m│\033[0m \033[1;33m{role.upper()}:\033[0m")
                print(indent_lines(message_content(message)))
        else:
            print("  \033[35m│\033[0m \033[1;33mPROMPT:\033[0m")
            print(indent_lines(str(messages)))
        print("  \033[35m├─────────────────────────────────────────────────────────────────────────────────┤\033[0m")
        print("  \033[35m│\033[0m \033[1;32mResponse:\033[0m")
        print(indent_lines(response_content(result)))
        print("  \033[35m└─────────────────────────────────────────────────────────────────────────────────┘\033[0m")

    @property
    def llm_duration_ms(self) -> float:
        return self._tracker["llm_time_ns"] / 1_000_000

    @property
    def call_count(self) -> int:
        return self._tracker["call_count"]

    @property
    def input_tokens(self) -> int:
        return self._tracker["input_tokens"]

    @property
    def output_tokens(self) -> int:
        return self._tracker["output_tokens"]

    @property
    def agent_metrics(self) -> dict[str, dict[str, int]]:
        return self._tracker.setdefault("agents", {})


@dataclass
class Timing:
    name: str
    total_ms: float
    llm_ms: float
    calls: int
    concurrent: bool = False
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def overhead_ms(self) -> float:
        return self.total_ms - self.llm_ms


def _truncate(text: str, max_len: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[:max_len] + "..."


def _ingest_stats(state: dict[str, Any]) -> str:
    lines = []
    cls = state.get("classification_result")
    if cls and cls.classifications:
        lines.append(f"classifications={len(cls.classifications)}")
        for c in cls.classifications:
            lines.append(f"  {c['source']}: {c['query']}")
    for domain in ("profile", "temporal", "summary"):
        judge = state.get(f"{domain}_judge")
        if judge and judge.operations:
            lines.append(f"Judge ({domain}): {len(judge.operations)} op(s)")
            for op in judge.operations:
                preview = op.content[:70] + "..." if len(op.content) > 70 else op.content
                lines.append(f"  {op.type.value}: {preview}")
    return "\n  ".join(lines)


async def bench_full_ingest(real_model: BaseChatModel, tracker: dict[str, Any]) -> tuple[Timing, IngestPipeline]:
    tm = TimedModel(real_model, tracker=tracker)

    def _timed_get_model(*_args, **_kwargs):
        return tm

    with patch("src.pipelines.ingest.get_model", _timed_get_model), \
         patch("src.pipelines.ingest.get_vision_model", _timed_get_model):
        pipeline = IngestPipeline()

    start_calls = tm.call_count
    start_input_tokens = tm.input_tokens
    start_output_tokens = tm.output_tokens
    start_llm_ms = tm.llm_duration_ms
    start = time.perf_counter_ns()
    state = await pipeline.run(
        user_query=TEST_QUERY,
        agent_response=TEST_RESPONSE,
        user_id=BENCH_USER_ID,
        session_datetime=SESSION_DT,
    )
    total_ms = (time.perf_counter_ns() - start) / 1_000_000
    calls = tm.call_count - start_calls
    input_tokens = tm.input_tokens - start_input_tokens
    output_tokens = tm.output_tokens - start_output_tokens
    llm_ms = tm.llm_duration_ms - start_llm_ms
    stats = _ingest_stats(state)
    print(f"  Full Ingest Pipeline          calls={calls}  (parallel — LLM sum > wall clock)")
    if stats:
        print(f"  {indent_lines(stats, 2).lstrip()}")
    timing = Timing("Full Ingest Pipeline", total_ms, llm_ms, calls,
                    concurrent=True, input_tokens=input_tokens, output_tokens=output_tokens)
    return timing, pipeline


async def bench_full_retrieval(real_model: BaseChatModel, vector_store: Any, neo4j: Any, tracker: dict[str, Any]) -> Timing:
    tm = TimedModel(real_model, tracker=tracker, agent="retrieval")
    pipeline = RetrievalPipeline(
        model=tm,
        vector_store=vector_store,
        neo4j_client=neo4j,
    )

    start_calls = tm.call_count
    start_input_tokens = tm.input_tokens
    start_output_tokens = tm.output_tokens
    start_llm_ms = tm.llm_duration_ms
    start = time.perf_counter_ns()
    result = await pipeline.run(
        query="What is my name and where do I work?",
        user_id=BENCH_USER_ID,
    )
    total_ms = (time.perf_counter_ns() - start) / 1_000_000
    calls = tm.call_count - start_calls
    input_tokens = tm.input_tokens - start_input_tokens
    output_tokens = tm.output_tokens - start_output_tokens
    llm_ms = tm.llm_duration_ms - start_llm_ms
    print(
        f"  Full Retrieval Pipeline       calls={calls}  "
        f"answer={result.answer!r}  sources={result.source_count}  confidence={result.confidence:.2f}"
    )
    return Timing("Full Retrieval Pipeline", total_ms, llm_ms, calls,
                  input_tokens=input_tokens, output_tokens=output_tokens)


def _print_summary(tracker: dict[str, Any], model_name: str, pipeline_timings: list[Timing] | None = None) -> None:
    metrics_by_agent = tracker.get("agents", {})

    print()
    print("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                          XMem-Python Pipeline Metrics Summary                                   ║")
    print(f"║  Model: {str(model_name):<85}║")
    print("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  {'Agent':<20} {'Calls':>6} {'LLM Time':>12} {'Overhead':>12} {'In Tokens':>12} {'Out Tokens':>12} ║")
    print("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣")
    for agent in sorted(metrics_by_agent):
        metrics = metrics_by_agent[agent]
        llm_ms = metrics.get("llm_time_ns", 0) / 1_000_000
        print(
            f"║  {agent:<20} {metrics.get('call_count', 0):>6} "
            f"{llm_ms:>11.0f}ms {'—':>12} {metrics.get('input_tokens', 0):>12} "
            f"{metrics.get('output_tokens', 0):>12} ║"
        )
    if pipeline_timings:
        print("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣")
        print(f"║  {'Pipeline':<20} {'Calls':>6} {'LLM Time':>12} {'Overhead':>12} {'Wall Clock':>12} {'':>12} ║")
        print("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣")
        for pt in pipeline_timings:
            overhead = max(0, pt.total_ms - pt.llm_ms)
            print(
                f"║  {pt.name:<20} {pt.calls:>6} "
                f"{pt.llm_ms:>11.0f}ms {overhead:>11.0f}ms {pt.total_ms:>11.0f}ms {'':>12} ║"
            )
    print("╚══════════════════════════════════════════════════════════════════════════════════════════════════╝")

    if pipeline_timings:
        total_in = tracker.get("input_tokens", 0)
        total_out = tracker.get("output_tokens", 0)
        query_tokens = estimate_tokens(TEST_QUERY)

        input_price = 0.15 / 1_000_000
        output_price = 0.60 / 1_000_000

        ingest_pt = next((pt for pt in pipeline_timings if "Ingest" in pt.name), None)
        retrieve_pt = next((pt for pt in pipeline_timings if "Retrieval" in pt.name), None)

        print()
        print(f"  User Query Tokens:    ~{query_tokens}")
        print(f"  Total Input Tokens:    {total_in}")
        print(f"  Total Output Tokens:   {total_out}")
        if ingest_pt:
            cost = ingest_pt.input_tokens * input_price + ingest_pt.output_tokens * output_price
            print(f"  Cost to Ingest:        ${cost:.6f}  ({ingest_pt.input_tokens} in / {ingest_pt.output_tokens} out)")
        if retrieve_pt:
            cost = retrieve_pt.input_tokens * input_price + retrieve_pt.output_tokens * output_price
            print(f"  Cost to Retrieve:      ${cost:.6f}  ({retrieve_pt.input_tokens} in / {retrieve_pt.output_tokens} out)")
        print()
        print("  Cost estimate based on gpt-4o-mini pricing ($0.15/1M input, $0.60/1M output).")
        print("  Actual cost varies by provider/model. Tokens may be estimated if provider didn't return usage.")


async def main():
    real_model = get_model()
    model_name = getattr(real_model, "model_name", getattr(real_model, "model", "unknown"))
    print(f"Model: {model_name}\n")

    tracker: dict[str, Any] = {
        "llm_time_ns": 0,
        "call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "agents": {},
    }

    pipeline_timings: list[Timing] = []

    print("Running full ingest pipeline...")
    ingest_timing, ingest_pipeline = await bench_full_ingest(real_model, tracker)
    pipeline_timings.append(ingest_timing)

    print("\nRunning full retrieval pipeline (after ingest)...")
    retrieval_timing = await bench_full_retrieval(real_model, ingest_pipeline.vector_store, ingest_pipeline.neo4j, tracker)
    pipeline_timings.append(retrieval_timing)

    _print_summary(tracker, str(model_name), pipeline_timings)


if __name__ == "__main__":
    asyncio.run(main())
