from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


os.environ["API_KEYS"] = '["test-static-key"]'
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-xmem")
os.environ.setdefault("NEO4J_PASSWORD", "test-neo4j-password")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("MONGODB_URI", "mongodb://127.0.0.1:1")
os.environ.setdefault("ENABLE_ANALYTICS", "false")
os.environ.setdefault("ENABLE_PROMETHEUS", "false")


@dataclass
class FakeLLMResponse:
    content: Any
    tool_calls: list[dict[str, Any]] | None = None
    usage_metadata: dict[str, int] | None = None
    response_metadata: dict[str, Any] | None = None


class FakeChatModel:
    """Deterministic model double for LangChain-style chat models."""

    model = "fake-chat-model"

    def __init__(
        self,
        responses: list[Any] | None = None,
        tool_responses: list[Any] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.tool_responses = list(tool_responses or [])
        self.calls: list[Any] = []
        self.bound_tools: list[Any] | None = None

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return _FakeBoundToolModel(self)

    async def ainvoke(self, messages):
        self.calls.append(messages)
        response = self.responses.pop(0) if self.responses else ""
        return response if isinstance(response, FakeLLMResponse) else FakeLLMResponse(response)

    async def astream(self, messages):
        response = await self.ainvoke(messages)
        yield response


class _FakeBoundToolModel:
    model = "fake-chat-model-with-tools"

    def __init__(self, model: FakeChatModel) -> None:
        self._model = model
        self.calls: list[Any] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        response = self._model.tool_responses.pop(0) if self._model.tool_responses else FakeLLMResponse("")
        return response if isinstance(response, FakeLLMResponse) else FakeLLMResponse(response)

    async def astream(self, messages):
        response = await self.ainvoke(messages)
        yield response


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.add_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.delete_calls: list[list[str]] = []
        self.next_id = 1

    def seed(self, record_id: str, content: str, metadata: dict[str, Any], score: float = 0.9):
        self.records[record_id] = {
            "content": content,
            "metadata": metadata,
            "embedding": [0.0, 0.0, 0.0],
            "score": score,
        }

    def add(self, texts, embeddings, ids=None, metadata=None):
        self.add_calls.append({"texts": texts, "embeddings": embeddings, "ids": ids, "metadata": metadata})
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
        self.update_calls.append({"id": id, "text": text, "embedding": embedding, "metadata": metadata})
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
        self.delete_calls.append(list(ids))
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
        from src.storage.base import SearchResult

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
        return self.search_by_metadata(filters, top_k=top_k) if filters else self.search_by_metadata({}, top_k=top_k)

    def search(self, query_embedding, top_k=5, filters=None):
        return self.search_by_metadata(filters or {}, top_k=top_k)

    def health_check(self):
        return True

    def get_stats(self):
        return SimpleNamespace(total_vector_count=len(self.records), dimension=3, namespaces={})


class FakeNeo4jClient:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.connected = False
        self.closed = False

    def connect(self):
        self.connected = True

    def close(self):
        self.closed = True

    def seed_event(self, **event):
        self.events.append(event)

    def search_events_by_name(self, event_name: str, user_id: str, top_k: int = 1):
        return [
            event for event in self.events
            if event.get("user_id", user_id) == user_id
            and event.get("event_name", "").lower() == event_name.lower()
        ][:top_k]

    def search_events_by_embedding(self, user_id: str, query_text: str, top_k: int = 3, similarity_threshold: float = 0.0):
        return [
            event for event in self.events
            if event.get("user_id", user_id) == user_id
        ][:top_k]

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


@pytest.fixture
def fake_model():
    return FakeChatModel()


@pytest.fixture
def vector_store():
    return InMemoryVectorStore()


@pytest.fixture
def neo4j_client():
    return FakeNeo4jClient()


@pytest.fixture
def fast_embed():
    return lambda text: [float(len(text)), 0.0, 1.0]


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
