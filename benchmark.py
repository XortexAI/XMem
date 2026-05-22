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
import hashlib
import logging
import math
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import patch

logging.basicConfig(level=logging.WARNING)

from langchain_core.language_models import BaseChatModel

from src.config import settings
from src.models import get_model
from src.agents.classifier import ClassifierAgent
from src.agents.profiler import ProfilerAgent
from src.agents.temporal import TemporalAgent
from src.agents.summarizer import SummarizerAgent
from src.agents.judge import JudgeAgent
from src.pipelines.ingest import IngestPipeline
from src.pipelines.retrieval import RetrievalPipeline
from src.storage.base import SearchResult

TEST_QUERY = (
    "My name is Alice and I work at Google as a senior software engineer. "
    "My birthday is April 5th. I love sushi and hiking on weekends."
)
TEST_RESPONSE = "Nice to meet you Alice! That sounds like a great lifestyle."
SESSION_DT = "4:04 pm on 20 January, 2025"
BENCH_USER_ID = "bench-user"


# ── In-memory stores (mirrors Go benchmark — no Pinecone/Neo4j required) ──


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.next_id = 1

    def add(self, texts, embeddings, ids=None, metadata=None):
        created = []
        for idx, text in enumerate(texts):
            record_id = ids[idx] if ids else f"vec-{self.next_id}"
            self.next_id += 1
            self.records[record_id] = {
                "content": text,
                "embedding": embeddings[idx],
                "metadata": (metadata or [{}])[idx],
                "score": 1.0,
            }
            created.append(record_id)
        return created

    def update(self, id, text=None, embedding=None, metadata=None):
        if id not in self.records:
            return False
        current = self.records[id]
        if text is not None:
            current["content"] = text
        if embedding is not None:
            current["embedding"] = embedding
        if metadata is not None:
            current["metadata"] = metadata
        return True

    def delete(self, ids):
        for record_id in ids:
            self.records.pop(record_id, None)
        return True

    def get(self, ids):
        return [
            {"id": record_id, **self.records[record_id]}
            for record_id in ids
            if record_id in self.records
        ]

    def search_by_metadata(self, filters, top_k=10):
        matches = []
        for record_id, record in self.records.items():
            metadata = record["metadata"]
            if all(metadata.get(key) == value for key, value in filters.items()):
                matches.append(
                    SearchResult(
                        id=record_id,
                        content=record["content"],
                        score=record.get("score", 1.0),
                        metadata=metadata,
                    )
                )
        return matches[:top_k]

    async def search_by_text(self, query_text, top_k=10, filters=None):
        filters = filters or {}
        query = query_text.lower()
        matches = []
        for record_id, record in self.records.items():
            metadata = record["metadata"]
            if filters and not all(metadata.get(k) == v for k, v in filters.items()):
                continue
            content = record["content"].lower()
            score = 1.0 if query and query in content else record.get("score", 0.5)
            if filters or score > 0:
                matches.append(
                    SearchResult(
                        id=record_id,
                        content=record["content"],
                        score=score,
                        metadata=metadata,
                    )
                )
        matches.sort(key=lambda r: r.score, reverse=True)
        return matches[:top_k]

    def search(self, query_embedding, top_k=5, filters=None):
        return self.search_by_metadata(filters or {}, top_k=top_k)

    def health_check(self):
        return True


class FakeNeo4jClient:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.connected = False

    def connect(self):
        self.connected = True

    def close(self):
        pass

    def search_events_by_name(self, event_name: str, user_id: str, top_k: int = 1):
        return [
            event for event in self.events
            if event.get("user_id", user_id) == user_id
            and event_name.lower() in event.get("event_name", "").lower()
        ][:top_k]

    def search_events_by_embedding(self, user_id: str, query_text: str, top_k: int = 3, similarity_threshold: float = 0.0):
        query = query_text.lower()
        matches = []
        for event in self.events:
            if event.get("user_id", user_id) != user_id:
                continue
            text = " ".join(
                str(event.get(k, "")) for k in ("event_name", "desc", "date_expression")
            ).lower()
            score = 1.0 if query and any(w in text for w in query.split()) else 0.1
            matches.append({**event, "similarity_score": score})
        matches.sort(key=lambda e: e.get("similarity_score", 0), reverse=True)
        return matches[:top_k]

    def create_event(self, user_id: str, date_str: str, event_data: dict[str, Any]):
        self.events.append({"user_id": user_id, "date": date_str, **event_data})

    def update_event(self, user_id: str, date_str: str, event_data: dict[str, Any]):
        for event in self.events:
            if event.get("user_id") == user_id and event.get("date") == date_str:
                event.update(event_data)
                return True
        self.create_event(user_id, date_str, event_data)
        return True

    def delete_event(self, user_id: str, date_str: str, event_name: str | None = None):
        self.events = [
            event for event in self.events
            if not (
                event.get("user_id") == user_id
                and event.get("date") == date_str
                and (event_name is None or event.get("event_name") == event_name)
            )
        ]
        return True


class FakeCodeGraphClient:
    def connect(self):
        pass

    def close(self):
        pass

    def setup(self):
        pass

    def create_annotation(self, **kwargs):
        return "ann-1"


def hash_embed(text: str) -> list[float]:
    """Local hash embedder — matches Go benchmark (no embedding API calls)."""
    dim = int(settings.pinecone_dimension or 384)
    vec = [0.0] * dim
    words = text.lower().split() or [text.lower()]
    for word in words:
        digest = hashlib.sha256(word.encode()).digest()
        idx = struct.unpack(">Q", digest[:8])[0] % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm:
        vec = [v / norm for v in vec]
    return vec


class TimedModel:
    """Proxy that wraps a real LangChain model and tracks LLM call time."""

    def __init__(self, inner: BaseChatModel, tracker: Optional[dict[str, int]] = None):
        self._inner = inner
        self._tracker = tracker or {"llm_time_ns": 0, "call_count": 0}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def ainvoke(self, messages: Any, **kwargs) -> Any:
        start = time.perf_counter_ns()
        result = await self._inner.ainvoke(messages, **kwargs)
        self._tracker["llm_time_ns"] += time.perf_counter_ns() - start
        self._tracker["call_count"] += 1
        return result

    def invoke(self, messages: Any, **kwargs) -> Any:
        start = time.perf_counter_ns()
        result = self._inner.invoke(messages, **kwargs)
        self._tracker["llm_time_ns"] += time.perf_counter_ns() - start
        self._tracker["call_count"] += 1
        return result

    def bind_tools(self, tools, **kwargs):
        return TimedModel(self._inner.bind_tools(tools, **kwargs), tracker=self._tracker)

    @property
    def llm_duration_ms(self) -> float:
        return self._tracker["llm_time_ns"] / 1_000_000

    @property
    def call_count(self) -> int:
        return self._tracker["call_count"]


@dataclass
class Timing:
    name: str
    total_ms: float
    llm_ms: float
    calls: int
    concurrent: bool = False

    @property
    def overhead_ms(self) -> float:
        return self.total_ms - self.llm_ms


def _truncate(text: str, max_len: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[:max_len] + "..."


def _ingest_stats(state: dict[str, Any]) -> str:
    parts = []
    cls = state.get("classification_result")
    if cls and cls.classifications:
        parts.append(f"classifications={len(cls.classifications)}")
    for domain in ("profile", "temporal", "summary"):
        judge = state.get(f"{domain}_judge")
        if judge and judge.operations:
            parts.append(f"{domain}_ops={len(judge.operations)}")
    return "  ".join(parts)


async def bench_classifier(real_model: BaseChatModel) -> Timing:
    tm = TimedModel(real_model)
    agent = ClassifierAgent(model=tm)
    start = time.perf_counter_ns()
    result = await agent.arun({"user_query": TEST_QUERY})
    total_ms = (time.perf_counter_ns() - start) / 1_000_000
    n = len(result.classifications) if result.classifications else 0
    print(f"  Classifier Agent              results={n}")
    return Timing("Classifier Agent", total_ms, tm.llm_duration_ms, tm.call_count)


async def bench_profiler(real_model: BaseChatModel) -> Timing:
    tm = TimedModel(real_model)
    agent = ProfilerAgent(model=tm)
    start = time.perf_counter_ns()
    result = await agent.arun({"classifier_output": TEST_QUERY})
    total_ms = (time.perf_counter_ns() - start) / 1_000_000
    n = len(result.facts) if result.facts else 0
    print(f"  Profiler Agent                facts={n}")
    return Timing("Profiler Agent", total_ms, tm.llm_duration_ms, tm.call_count)


async def bench_temporal(real_model: BaseChatModel) -> Timing:
    tm = TimedModel(real_model)
    agent = TemporalAgent(model=tm)
    start = time.perf_counter_ns()
    result = await agent.arun({
        "classifier_output": TEST_QUERY,
        "session_datetime": SESSION_DT,
    })
    total_ms = (time.perf_counter_ns() - start) / 1_000_000
    n = len(result.events) if result.events else 0
    print(f"  Temporal Agent                events={n}")
    return Timing("Temporal Agent", total_ms, tm.llm_duration_ms, tm.call_count)


async def bench_summarizer(real_model: BaseChatModel) -> Timing:
    tm = TimedModel(real_model)
    agent = SummarizerAgent(model=tm)
    start = time.perf_counter_ns()
    result = await agent.arun({
        "user_query": TEST_QUERY,
        "agent_response": TEST_RESPONSE,
    })
    total_ms = (time.perf_counter_ns() - start) / 1_000_000
    summary = result.summary if result.summary else ""
    n = len([line for line in summary.splitlines() if line.strip()])
    print(f"  Summarizer Agent              bullets={n}")
    return Timing("Summarizer Agent", total_ms, tm.llm_duration_ms, tm.call_count)


async def bench_judge_deterministic(real_model: BaseChatModel) -> Timing:
    tm = TimedModel(real_model)
    agent = JudgeAgent(model=tm, vector_store=None, graph_event_search=None, top_k=3)
    items = [
        {"topic": "basic_info", "sub_topic": "name", "memo": "Alice"},
        {"topic": "work", "sub_topic": "company", "memo": "Google"},
        {"topic": "work", "sub_topic": "title", "memo": "Senior Software Engineer"},
    ]
    start = time.perf_counter_ns()
    result = await agent.arun_deterministic({
        "domain": "profile",
        "new_items": items,
        "user_id": BENCH_USER_ID,
    })
    total_ms = (time.perf_counter_ns() - start) / 1_000_000
    n = len(result.operations) if result.operations else 0
    print(f"  Judge (deterministic)         ops={n}")
    return Timing("Judge (deterministic)", total_ms, tm.llm_duration_ms, tm.call_count)


async def bench_judge_llm(real_model: BaseChatModel) -> Timing:
    tm = TimedModel(real_model)
    agent = JudgeAgent(model=tm, vector_store=None, graph_event_search=None, top_k=3)
    items = [
        "User's name is Alice and works at Google as a senior software engineer",
        "User's birthday is April 5th",
        "User loves sushi and hiking on weekends",
    ]
    start = time.perf_counter_ns()
    result = await agent.arun({
        "domain": "summary",
        "new_items": items,
        "user_id": BENCH_USER_ID,
    })
    total_ms = (time.perf_counter_ns() - start) / 1_000_000
    n = len(result.operations) if result.operations else 0
    print(f"  Judge (LLM)                   ops={n}")
    return Timing("Judge (LLM)", total_ms, tm.llm_duration_ms, tm.call_count)


async def bench_full_ingest(real_model: BaseChatModel, vector_store: InMemoryVectorStore, neo4j: FakeNeo4jClient) -> Timing:
    tm = TimedModel(real_model)

    def _timed_get_model(*_args, **_kwargs):
        return tm

    with patch("src.pipelines.ingest.get_model", _timed_get_model), \
         patch("src.pipelines.ingest.get_vision_model", _timed_get_model), \
         patch("src.storage.factory.get_vector_store", lambda *args, **kwargs: vector_store):
        pipeline = IngestPipeline(
            vector_store=vector_store,
            neo4j_client=neo4j,
            code_graph_client=FakeCodeGraphClient(),
            embed_fn=hash_embed,
        )

    start = time.perf_counter_ns()
    state = await pipeline.run(
        user_query=TEST_QUERY,
        agent_response=TEST_RESPONSE,
        user_id=BENCH_USER_ID,
        session_datetime=SESSION_DT,
    )
    total_ms = (time.perf_counter_ns() - start) / 1_000_000
    print(f"  Full Ingest Pipeline          calls={tm.call_count}  {_ingest_stats(state)} (parallel — LLM sum > wall clock)")
    return Timing("Full Ingest Pipeline", total_ms, tm.llm_duration_ms, tm.call_count, concurrent=True)


async def bench_full_retrieval(real_model: BaseChatModel, vector_store: InMemoryVectorStore, neo4j: FakeNeo4jClient) -> Timing:
    tm = TimedModel(real_model)
    pipeline = RetrievalPipeline(
        model=tm,
        vector_store=vector_store,
        neo4j_client=neo4j,
    )

    start = time.perf_counter_ns()
    result = await pipeline.run(
        query="What is my name and where do I work?",
        user_id=BENCH_USER_ID,
    )
    total_ms = (time.perf_counter_ns() - start) / 1_000_000
    print(
        f"  Full Retrieval Pipeline       calls={tm.call_count}  "
        f"answer={result.answer!r}  sources={result.source_count}  confidence={result.confidence:.2f}"
    )
    return Timing("Full Retrieval Pipeline", total_ms, tm.llm_duration_ms, tm.call_count)


async def bench_concurrent_agents(real_model: BaseChatModel) -> Timing:
    tm = TimedModel(real_model)
    start = time.perf_counter_ns()

    classifier = ClassifierAgent(model=tm)
    await classifier.arun({"user_query": TEST_QUERY})

    profiler = ProfilerAgent(model=tm)
    temporal = TemporalAgent(model=tm)
    summarizer = SummarizerAgent(model=tm)

    await asyncio.gather(
        profiler.arun({"classifier_output": TEST_QUERY}),
        temporal.arun({"classifier_output": TEST_QUERY, "session_datetime": SESSION_DT}),
        summarizer.arun({"user_query": TEST_QUERY, "agent_response": TEST_RESPONSE}),
    )

    total_ms = (time.perf_counter_ns() - start) / 1_000_000
    print(f"  Concurrent Pipeline Sim       calls={tm.call_count} (parallel — LLM sum > wall clock)")
    return Timing("Concurrent Pipeline Sim", total_ms, tm.llm_duration_ms, tm.call_count, concurrent=True)


def _print_summary(timings: List[Timing], model_name: str) -> None:
    print()
    print("╔════════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                        XMem-Python Benchmark Summary                                    ║")
    print(f"║  Model: {str(model_name):<77}║")
    print("╠════════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  {'Component':<30} {'Total':>10} {'LLM Time':>12} {'Overhead':>10} {'Calls':>6} {'':>8} ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════════╣")
    for t in timings:
        if t.concurrent:
            saved = max(0.0, t.llm_ms - t.total_ms)
            print(
                f"║  {t.name:<30} {t.total_ms:>9.0f}ms {t.llm_ms:>10.0f}ms† {saved:>9.0f}ms {t.calls:>6} {'parallel':>8} ║"
            )
        else:
            print(
                f"║  {t.name:<30} {t.total_ms:>9.0f}ms {t.llm_ms:>11.0f}ms {t.overhead_ms:>9.1f}ms {t.calls:>6} {'':>8} ║"
            )
    print("╚════════════════════════════════════════════════════════════════════════════════════════╝")
    print()
    print("Sequential agents:  Overhead = Total - LLM Time (prompt building, parsing, etc.)")
    print("Parallel agents:    LLM Time† = cumulative across asyncio tasks; Overhead = time saved by concurrency")
    print("Compare sequential 'Overhead' with Go benchmark output.")


async def main():
    real_model = get_model()
    model_name = getattr(real_model, "model_name", getattr(real_model, "model", "unknown"))
    print(f"Model: {model_name}\n")

    timings: List[Timing] = []

    print("Running individual agent benchmarks (real LLM calls)...")
    print("─" * 70)
    timings.append(await bench_classifier(real_model))
    timings.append(await bench_profiler(real_model))
    timings.append(await bench_temporal(real_model))
    timings.append(await bench_summarizer(real_model))
    timings.append(await bench_judge_deterministic(real_model))
    timings.append(await bench_judge_llm(real_model))
    print("─" * 70)

    # Shared stores: ingest writes here, retrieval reads the same data.
    vector_store = InMemoryVectorStore()
    neo4j = FakeNeo4jClient()
    neo4j.connect()

    print("\nRunning full ingest pipeline...")
    timings.append(await bench_full_ingest(real_model, vector_store, neo4j))

    print("\nRunning full retrieval pipeline (after ingest)...")
    timings.append(await bench_full_retrieval(real_model, vector_store, neo4j))

    print("\nRunning concurrent agent benchmark (classifier → profiler+temporal+summarizer in parallel)...")
    timings.append(await bench_concurrent_agents(real_model))

    _print_summary(timings, str(model_name))


if __name__ == "__main__":
    asyncio.run(main())
