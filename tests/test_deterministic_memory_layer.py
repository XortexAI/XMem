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
from src.schemas.judge import JudgeDomain, JudgeResult, Operation, OperationType
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

    def add(self, texts, embeddings, ids=None, metadata=None):
        self.add_calls.append((texts, embeddings, metadata))
        created_ids = []
        for idx, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
            record_id = ids[idx] if ids else f"vec-{self.next_id}"
            if not ids:
                self.next_id += 1
            self.records[record_id] = {
                "text": text,
                "metadata": meta,
                "embedding": embedding,
            }
            created_ids.append(record_id)
        return created_ids

    def update(self, id, text=None, embedding=None, metadata=None):
        self.update_calls.append((id, text, embedding, metadata))
        if id not in self.records:
            return False
        current = self.records[id]
        merged_metadata = dict(current["metadata"])
        merged_metadata.update(metadata or {})
        self.records[id] = {
            "text": text if text is not None else current["text"],
            "metadata": merged_metadata,
            "embedding": embedding if embedding is not None else current["embedding"],
        }
        return True

    def delete(self, ids):
        self.delete_calls.append(ids)
        for record_id in ids:
            self.records.pop(record_id, None)
        return True

    def get(self, ids):
        return [
            {
                "id": record_id,
                "content": self.records[record_id]["text"],
                "metadata": self.records[record_id]["metadata"],
                "embedding": self.records[record_id]["embedding"],
            }
            for record_id in ids
            if record_id in self.records
        ]


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
    new_id = weaver_result.executed[0].new_id
    assert new_id
    assert store.records["profile-1"]["text"] == "work / company = Google"
    assert store.records["profile-1"]["metadata"]["is_current"] is False
    assert store.records["profile-1"]["metadata"]["superseded_by"] == new_id
    assert store.records[new_id]["text"] == "work / company = OpenAI"
    assert store.records[new_id]["metadata"]["main_content"] == "work_company"
    assert store.records[new_id]["metadata"]["subcontent"] == "OpenAI"
    assert store.records[new_id]["metadata"]["parent_memory_id"] == "profile-1"
    assert store.records[new_id]["metadata"]["version"] == 2


@pytest.mark.asyncio
async def test_weaver_skips_duplicate_memory_by_content_hash():
    store = FakeVectorStore()
    weaver = Weaver(vector_store=store, embed_fn=fake_embed)

    first = await weaver.execute(
        JudgeResult(operations=[
            Operation(type=OperationType.ADD, content="work / title = Engineer"),
        ]),
        JudgeDomain.PROFILE,
        "user-1",
    )
    second = await weaver.execute(
        JudgeResult(operations=[
            Operation(type=OperationType.ADD, content="work / title = Engineer"),
        ]),
        JudgeDomain.PROFILE,
        "user-1",
    )

    assert first.executed[0].status == OpStatus.SUCCESS
    assert second.executed[0].status == OpStatus.SKIPPED
    assert second.executed[0].embedding_id == first.executed[0].new_id
    assert len(store.add_calls) == 1
    assert not store.update_calls


@pytest.mark.asyncio
async def test_forget_deletes_all_versions_in_lineage():
    store = FakeVectorStore()
    weaver = Weaver(vector_store=store, embed_fn=fake_embed)

    first = await weaver.execute(
        JudgeResult(operations=[
            Operation(type=OperationType.ADD, content="work / company = Google"),
        ]),
        JudgeDomain.PROFILE,
        "user-1",
    )
    updated = await weaver.execute(
        JudgeResult(operations=[
            Operation(
                type=OperationType.UPDATE,
                embedding_id=first.executed[0].new_id,
                content="work / company = OpenAI",
            ),
        ]),
        JudgeDomain.PROFILE,
        "user-1",
    )

    deleted = await weaver.execute(
        JudgeResult(operations=[
            Operation(type=OperationType.DELETE, embedding_id=updated.executed[0].new_id),
        ]),
        JudgeDomain.PROFILE,
        "user-1",
    )

    assert deleted.executed[0].status == OpStatus.SUCCESS
    assert store.records == {}


@pytest.mark.asyncio
async def test_stale_update_supersedes_current_lineage_version():
    store = FakeVectorStore()
    weaver = Weaver(vector_store=store, embed_fn=fake_embed)

    first = await weaver.execute(
        JudgeResult(operations=[
            Operation(type=OperationType.ADD, content="work / company = Google"),
        ]),
        JudgeDomain.PROFILE,
        "user-1",
    )
    first_id = first.executed[0].new_id
    second = await weaver.execute(
        JudgeResult(operations=[
            Operation(
                type=OperationType.UPDATE,
                embedding_id=first_id,
                content="work / company = OpenAI",
            ),
        ]),
        JudgeDomain.PROFILE,
        "user-1",
    )

    stale = await weaver.execute(
        JudgeResult(operations=[
            Operation(
                type=OperationType.UPDATE,
                embedding_id=first_id,
                content="work / company = Anthropic",
            ),
        ]),
        JudgeDomain.PROFILE,
        "user-1",
    )

    second_id = second.executed[0].new_id
    third_id = stale.executed[0].new_id

    assert stale.executed[0].embedding_id == second_id
    assert store.records[first_id]["metadata"]["is_current"] is False
    assert store.records[second_id]["metadata"]["is_current"] is False
    assert store.records[third_id]["metadata"]["is_current"] is True
    assert store.records[third_id]["metadata"]["version"] == 3


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
