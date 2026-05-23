import os
import sys
import types
import importlib.util
from pathlib import Path

import pytest

os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("NEO4J_PASSWORD", "test-neo4j-password")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

langchain_core = types.ModuleType("langchain_core")
language_models = types.ModuleType("langchain_core.language_models")
messages = types.ModuleType("langchain_core.messages")
language_models.BaseChatModel = object
messages.HumanMessage = object
messages.SystemMessage = object
langchain_core.language_models = language_models
langchain_core.messages = messages
sys.modules.setdefault("langchain_core", langchain_core)
sys.modules.setdefault("langchain_core.language_models", language_models)
sys.modules.setdefault("langchain_core.messages", messages)

from src.agents.judge import JudgeAgent
from src.schemas.code import (
    code_annotation_fields_from_storage_content,
    code_annotation_pinecone_metadata,
    snippet_fields_from_storage_content,
    snippet_pinecone_metadata,
)
from src.schemas.judge import JudgeDomain, OperationType
from src.schemas.weaver import OpStatus
from src.storage.base import SearchResult

WEAVER_PATH = Path(__file__).resolve().parents[1] / "src" / "pipelines" / "weaver.py"
weaver_spec = importlib.util.spec_from_file_location("weaver_under_test", WEAVER_PATH)
weaver_module = importlib.util.module_from_spec(weaver_spec)
weaver_spec.loader.exec_module(weaver_module)
Weaver = weaver_module.Weaver


class ModelMustNotBeCalled:
    async def ainvoke(self, messages):
        raise AssertionError("deterministic judge should not call the LLM")


class FakeVectorStore:
    def __init__(self):
        self.records = {}
        self.next_id = 1
        self.add_calls = []
        self.update_calls = []
        self.delete_calls = []

    def seed(self, record_id, text, metadata):
        self.records[record_id] = {
            "text": text,
            "metadata": metadata,
            "embedding": [0.0, 0.0, 0.0],
        }

    def search_by_metadata(self, filters, top_k=1):
        matches = []
        for record_id, record in self.records.items():
            metadata = record["metadata"]
            if all(metadata.get(key) == value for key, value in filters.items()):
                matches.append(SearchResult(
                    id=record_id,
                    content=record["text"],
                    score=1.0,
                    metadata=metadata,
                ))
        return matches[:top_k]

    def add(self, texts, embeddings, metadata):
        self.add_calls.append((texts, embeddings, metadata))
        ids = []
        for text, embedding, meta in zip(texts, embeddings, metadata):
            record_id = f"vec-{self.next_id}"
            self.next_id += 1
            self.records[record_id] = {
                "text": text,
                "metadata": meta,
                "embedding": embedding,
            }
            ids.append(record_id)
        return ids

    def update(self, id, text, embedding, metadata):
        self.update_calls.append((id, text, embedding, metadata))
        if id not in self.records:
            return False
        self.records[id] = {
            "text": text,
            "metadata": metadata,
            "embedding": embedding,
        }
        return True

    def delete(self, ids):
        self.delete_calls.append(ids)
        for record_id in ids:
            self.records.pop(record_id, None)
        return True


class FakeTemporalGraph:
    def __init__(self):
        self.events = {}
        self.created = []
        self.updated = []
        self.deleted = []

    def seed(self, date, event_name, **event_data):
        self.events[(date, event_name.lower())] = {
            "date": date,
            "event_name": event_name,
            "desc": event_data.get("desc", ""),
            "year": event_data.get("year", ""),
            "time": event_data.get("time", ""),
            "date_expression": event_data.get("date_expression", ""),
        }

    async def search(self, event_name, user_id, top_k=1):
        matches = [
            event
            for (date, name), event in self.events.items()
            if name == event_name.lower()
        ]
        return [
            SearchResult(
                id=f"{event['date']}_{event['event_name']}",
                content=(
                    f"{event['date']} | {event['event_name']} | "
                    f"{event['desc']} | {event['year']} | {event['time']} | "
                    f"{event['date_expression']}"
                ),
                score=1.0,
                metadata=event,
            )
            for event in matches[:top_k]
        ]

    async def create(self, user_id, date_str, event_data):
        event_name = event_data.get("event_name", "")
        self.created.append((user_id, date_str, event_data))
        self.events[(date_str, event_name.lower())] = {
            "date": date_str,
            **event_data,
        }

    async def update(self, user_id, date_str, event_data):
        event_name = event_data.get("event_name", "")
        self.updated.append((user_id, date_str, event_data))
        self.events[(date_str, event_name.lower())] = {
            "date": date_str,
            **event_data,
        }

    async def delete(self, user_id, embedding_id):
        date_str, event_name = embedding_id.split("_", 1)
        self.deleted.append((user_id, embedding_id))
        self.events.pop((date_str, event_name.lower()), None)


def fake_embed(text):
    return [float(len(text)), 0.0, 1.0]


@pytest.mark.asyncio
async def test_profile_deterministic_judge_noops_existing_memory_without_llm():
    store = FakeVectorStore()
    store.seed(
        "profile-1",
        "work / company = Google",
        {
            "user_id": "user-1",
            "domain": "profile",
            "main_content": "work_company",
            "subcontent": "Google",
        },
    )
    judge = JudgeAgent(model=ModelMustNotBeCalled(), vector_store=store)

    result = await judge.arun_deterministic({
        "domain": "profile",
        "new_items": [{
            "topic": "work",
            "sub_topic": "company",
            "memo": "Google",
        }],
        "user_id": "user-1",
    })

    assert result.confidence == 1.0
    assert len(result.operations) == 1
    assert result.operations[0].type == OperationType.NOOP
    assert result.operations[0].embedding_id == "profile-1"

    weaver = Weaver(vector_store=store, embed_fn=fake_embed)
    weaver_result = await weaver.execute(result, JudgeDomain.PROFILE, "user-1")

    assert weaver_result.total == 0
    assert not store.add_calls
    assert not store.update_calls


@pytest.mark.asyncio
async def test_profile_memory_layer_updates_changed_profile_fact():
    store = FakeVectorStore()
    store.seed(
        "profile-1",
        "work / company = Google",
        {
            "user_id": "user-1",
            "domain": "profile",
            "main_content": "work_company",
            "subcontent": "Google",
        },
    )
    judge = JudgeAgent(model=ModelMustNotBeCalled(), vector_store=store)

    judge_result = await judge.arun_deterministic({
        "domain": "profile",
        "new_items": [{
            "topic": "work",
            "sub_topic": "company",
            "memo": "OpenAI",
        }],
        "user_id": "user-1",
    })

    assert judge_result.operations[0].type == OperationType.UPDATE

    weaver = Weaver(vector_store=store, embed_fn=fake_embed)
    weaver_result = await weaver.execute(
        judge_result,
        JudgeDomain.PROFILE,
        "user-1",
    )

    assert weaver_result.total == 1
    assert weaver_result.executed[0].status == OpStatus.SUCCESS
    assert store.records["profile-1"]["text"] == "work / company = OpenAI"
    assert store.records["profile-1"]["metadata"]["main_content"] == "work_company"
    assert store.records["profile-1"]["metadata"]["subcontent"] == "OpenAI"


@pytest.mark.asyncio
async def test_temporal_memory_layer_adds_new_event():
    graph = FakeTemporalGraph()
    judge = JudgeAgent(
        model=ModelMustNotBeCalled(),
        graph_event_search=graph.search,
    )

    judge_result = await judge.arun_deterministic({
        "domain": "temporal",
        "new_items": [{
            "date": "04-24",
            "event_name": "Demo",
            "desc": "Product demo",
            "year": "2026",
            "time": "10:00",
            "date_expression": "today",
        }],
        "user_id": "user-1",
    })

    assert judge_result.operations[0].type == OperationType.ADD

    weaver = Weaver(
        graph_create_event=graph.create,
        graph_update_event=graph.update,
        graph_delete_event=graph.delete,
    )
    weaver_result = await weaver.execute(
        judge_result,
        JudgeDomain.TEMPORAL,
        "user-1",
    )

    assert weaver_result.total == 1
    assert weaver_result.executed[0].status == OpStatus.SUCCESS
    assert graph.created[0][1] == "04-24"
    assert graph.events[("04-24", "demo")]["desc"] == "Product demo"


@pytest.mark.asyncio
async def test_temporal_memory_layer_recreates_event_when_date_changes():
    graph = FakeTemporalGraph()
    graph.seed(
        "04-24",
        "Demo",
        desc="Product demo",
        year="2026",
        time="10:00",
        date_expression="today",
    )
    judge = JudgeAgent(
        model=ModelMustNotBeCalled(),
        graph_event_search=graph.search,
    )

    judge_result = await judge.arun_deterministic({
        "domain": "temporal",
        "new_items": [{
            "date": "04-25",
            "event_name": "Demo",
            "desc": "Product demo",
            "year": "2026",
            "time": "10:00",
            "date_expression": "tomorrow",
        }],
        "user_id": "user-1",
    })

    assert [op.type for op in judge_result.operations] == [
        OperationType.DELETE,
        OperationType.ADD,
    ]

    weaver = Weaver(
        graph_create_event=graph.create,
        graph_update_event=graph.update,
        graph_delete_event=graph.delete,
    )
    weaver_result = await weaver.execute(
        judge_result,
        JudgeDomain.TEMPORAL,
        "user-1",
    )

    assert weaver_result.total == 2
    assert weaver_result.succeeded == 2
    assert graph.deleted == [("user-1", "04-24_Demo")]
    assert ("04-24", "demo") not in graph.events
    assert graph.events[("04-25", "demo")]["date_expression"] == "tomorrow"


@pytest.mark.asyncio
async def test_temporal_memory_layer_updates_same_date_changed_details():
    graph = FakeTemporalGraph()
    graph.seed(
        "04-24",
        "Demo",
        desc="Product demo",
        year="2026",
        time="10:00",
        date_expression="today",
    )
    judge = JudgeAgent(
        model=ModelMustNotBeCalled(),
        graph_event_search=graph.search,
    )

    judge_result = await judge.arun_deterministic({
        "domain": "temporal",
        "new_items": [{
            "date": "04-24",
            "event_name": "Demo",
            "desc": "Updated product demo",
            "year": "2026",
            "time": "11:00",
            "date_expression": "today",
        }],
        "user_id": "user-1",
    })

    assert judge_result.operations[0].type == OperationType.UPDATE

    weaver = Weaver(
        graph_create_event=graph.create,
        graph_update_event=graph.update,
        graph_delete_event=graph.delete,
    )
    weaver_result = await weaver.execute(
        judge_result,
        JudgeDomain.TEMPORAL,
        "user-1",
    )

    assert weaver_result.total == 1
    assert weaver_result.executed[0].status == OpStatus.SUCCESS
    assert graph.updated[0][1] == "04-24"
    assert graph.events[("04-24", "demo")]["desc"] == "Updated product demo"


@pytest.mark.asyncio
async def test_snippet_deterministic_judge_noops_same_snippet_across_sessions():
    store = FakeVectorStore()
    content = "Binary search helper | def bs():\\n    return 1 | python | utility | search"
    fields = snippet_fields_from_storage_content(content)
    store.seed(
        "snippet-1",
        "Binary search helper",
        snippet_pinecone_metadata("user-1", fields),
    )
    judge = JudgeAgent(model=ModelMustNotBeCalled(), vector_store=store)

    judge_result = await judge.arun_deterministic({
        "domain": "snippet",
        "new_items": [
            "Same helper pasted later | def bs():\n    return 1 | Python | utility | dsa"
        ],
        "user_id": "user-1",
    })

    assert judge_result.confidence == 1.0
    assert judge_result.operations[0].type == OperationType.NOOP
    assert judge_result.operations[0].embedding_id == "snippet-1"

    weaver = Weaver(vector_store=store, embed_fn=fake_embed)
    weaver_result = await weaver.execute(
        judge_result,
        JudgeDomain.SNIPPET,
        "user-1",
    )

    assert weaver_result.total == 0
    assert not store.add_calls


@pytest.mark.asyncio
async def test_snippet_deterministic_judge_deduplicates_incoming_batch():
    store = FakeVectorStore()
    judge = JudgeAgent(model=ModelMustNotBeCalled(), vector_store=store)

    judge_result = await judge.arun_deterministic({
        "domain": "snippet",
        "new_items": [
            "Binary search helper | def bs():\n    return 1 | python | utility | search",
            "Binary search helper again | def bs():\n    return 1 | Python | utility | dsa",
        ],
        "user_id": "user-1",
    })

    assert judge_result.confidence == 1.0
    assert len(judge_result.operations) == 1
    assert judge_result.operations[0].type == OperationType.ADD


@pytest.mark.asyncio
async def test_snippet_weaver_stores_identity_hash_and_search_text():
    store = FakeVectorStore()
    judge_result = await JudgeAgent(
        model=ModelMustNotBeCalled(),
        vector_store=store,
    ).arun_deterministic({
        "domain": "snippet",
        "new_items": [
            "Binary search helper | def bs():\\n    return 1 | python | utility | search,array"
        ],
        "user_id": "user-1",
    })

    weaver = Weaver(vector_store=store, embed_fn=fake_embed)
    weaver_result = await weaver.execute(
        judge_result,
        JudgeDomain.SNIPPET,
        "user-1",
    )

    assert weaver_result.succeeded == 1
    record = store.records[weaver_result.executed[0].new_id]
    assert record["text"] == "Binary search helper\nlanguage: python\ntags: search,array"
    assert record["metadata"]["domain"] == "snippet"
    assert len(record["metadata"]["snippet_hash"]) == 64
    assert record["metadata"]["code_snippet"] == "def bs():\n    return 1"


@pytest.mark.asyncio
async def test_code_deterministic_judge_updates_same_target_annotation():
    store = FakeVectorStore()
    existing = (
        "bug_report | Auth.login | src/auth.py | api | high | "
        "Token refresh can fail"
    )
    store.seed(
        "code-1",
        existing,
        code_annotation_pinecone_metadata(
            "user-1",
            code_annotation_fields_from_storage_content(existing),
        ),
    )
    judge = JudgeAgent(model=ModelMustNotBeCalled(), vector_store=store)

    changed = (
        "bug_report | Auth.login | src/auth.py | api | high | "
        "Token refresh can fail after session rotation"
    )
    judge_result = await judge.arun_deterministic({
        "domain": "code",
        "new_items": [changed],
        "user_id": "user-1",
    })

    assert judge_result.operations[0].type == OperationType.UPDATE
    assert judge_result.operations[0].embedding_id == "code-1"

    weaver = Weaver(vector_store=store, embed_fn=fake_embed)
    weaver_result = await weaver.execute(judge_result, JudgeDomain.CODE, "user-1")

    assert weaver_result.succeeded == 1
    assert store.records["code-1"]["metadata"]["annotation_key"] == (
        "api|src/auth.py|auth.login|bug_report"
    )
    assert len(store.records["code-1"]["metadata"]["annotation_hash"]) == 64


@pytest.mark.asyncio
async def test_code_deterministic_judge_deduplicates_incoming_batch():
    store = FakeVectorStore()
    judge = JudgeAgent(model=ModelMustNotBeCalled(), vector_store=store)

    first = (
        "bug_report | Auth.login | src/auth.py | api | high | "
        "Token refresh can fail"
    )
    changed = (
        "bug_report | Auth.login | src/auth.py | api | high | "
        "Token refresh can fail after session rotation"
    )
    judge_result = await judge.arun_deterministic({
        "domain": "code",
        "new_items": [first, changed],
        "user_id": "user-1",
    })

    assert judge_result.confidence == 1.0
    assert len(judge_result.operations) == 1
    assert judge_result.operations[0].type == OperationType.ADD
    assert judge_result.operations[0].content == changed
