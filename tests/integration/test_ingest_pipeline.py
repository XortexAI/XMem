from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.pipelines.ingest import IngestPipeline
from src.schemas.classification import ClassificationResult
from src.schemas.events import EventData, EventResult
from src.schemas.judge import JudgeResult, Operation, OperationType
from src.schemas.profile import ProfileFact, ProfileResult
from src.schemas.summary import SummaryResult
from src.schemas.weaver import ExecutedOp, OpStatus, WeaverResult


class StaticAgent:
    def __init__(self, result):
        self.result = result
        self.states = []

    async def arun(self, state):
        self.states.append(state)
        return self.result


class StaticJudge:
    def __init__(self):
        self.states = []

    async def arun(self, state):
        self.states.append(("llm", state))
        return JudgeResult(operations=[Operation(type=OperationType.ADD, content=state["new_items"][0])], confidence=0.8)

    async def arun_deterministic(self, state):
        self.states.append(("deterministic", state))
        item = state["new_items"][0]
        content = item if isinstance(item, str) else " | ".join(str(v) for v in item.values())
        return JudgeResult(operations=[Operation(type=OperationType.ADD, content=content)], confidence=1.0)


class RecordingWeaver:
    def __init__(self):
        self.calls = []
        self.snippet_vector_store = None

    async def execute(self, judge_result, domain, user_id):
        self.calls.append((judge_result, domain, user_id))
        return WeaverResult(
            executed=[
                ExecutedOp(type=op.type, status=OpStatus.SUCCESS, content=op.content)
                for op in judge_result.operations
                if op.type != OperationType.NOOP
            ]
        )


def _pipeline(org_id="default"):
    pipeline = IngestPipeline.__new__(IngestPipeline)
    pipeline.org_id = org_id
    pipeline.judge = StaticJudge()
    pipeline.weaver = RecordingWeaver()
    pipeline._snippet_stores = {}
    pipeline._get_snippet_store = lambda user_id: f"snippet-store:{user_id}"
    return pipeline


def test_route_after_classify_fans_out_to_expected_nodes():
    pipeline = _pipeline(org_id="acme")
    routes = pipeline._route_after_classify({
        "user_query": "I joined XMem and found a code issue tomorrow",
        "user_id": "user-1",
        "classification_result": ClassificationResult(classifications=[
            {"source": "profile", "query": "I joined XMem"},
            {"source": "event", "query": "tomorrow"},
            {"source": "code", "query": "retry can fail"},
        ]),
    })

    assert [route.node for route in routes] == [
        "extract_summary",
        "extract_profile",
        "extract_temporal",
        "extract_code",
    ]


@pytest.mark.asyncio
async def test_profile_and_temporal_nodes_use_deterministic_judge():
    pipeline = _pipeline()
    pipeline.profiler = StaticAgent(ProfileResult(facts=[
        ProfileFact(topic="work", sub_topic="company", memo="XMem")
    ]))
    pipeline.temporal = StaticAgent(EventResult(events=[
        EventData(date="05-11", event_name="Launch", year=2026, desc="Ship tests")
    ]))

    profile = await pipeline._node_extract_profile({"profile_queries": ["I work at XMem"], "user_id": "user-1"})
    temporal = await pipeline._node_extract_temporal({"temporal_queries": ["launch today"], "user_id": "user-1"})

    assert profile["profile_weaver"].succeeded == 1
    assert temporal["temporal_weaver"].succeeded == 1
    assert [kind for kind, _state in pipeline.judge.states] == ["deterministic", "deterministic"]


@pytest.mark.asyncio
async def test_summary_node_splits_bullets_and_runs_judge_and_weaver():
    pipeline = _pipeline()
    pipeline.summarizer = StaticAgent(SummaryResult(summary="- Likes deterministic tests\n- Uses XMem"))

    result = await pipeline._node_extract_summary({
        "user_query": "remember this",
        "agent_response": "ok",
        "user_id": "user-1",
    })

    assert result["summary_weaver"].succeeded == 1
    kind, state = pipeline.judge.states[0]
    assert kind == "llm"
    assert state["new_items"] == ["Likes deterministic tests", "Uses XMem"]
