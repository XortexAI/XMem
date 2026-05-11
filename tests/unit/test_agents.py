from __future__ import annotations

import pytest

from src.agents.base import BaseAgent
from src.agents.classifier import ClassifierAgent
from src.agents.code import CodeAgent
from src.agents.judge import (
    _build_profile_metadata_key,
    _dedupe_profile_items,
    _same_temporal_event,
)
from src.agents.profiler import ProfilerAgent
from src.agents.snippet import SnippetAgent
from src.agents.summarizer import SummarizerAgent
from src.agents.temporal import TemporalAgent
from src.schemas.code import AnnotationSeverity, AnnotationType, SnippetType
from tests.conftest import FakeChatModel, FakeLLMResponse


class ConcreteAgent(BaseAgent):
    async def arun(self, state):
        return state


@pytest.mark.asyncio
async def test_base_agent_builds_messages_and_normalizes_list_content():
    model = FakeChatModel(responses=[FakeLLMResponse([{"text": "hello"}, "world"])])
    agent = ConcreteAgent(model=model, name="test", system_prompt="system")

    assert agent._build_messages("user") == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert await agent._call_model([{"role": "user", "content": "hi"}]) == "hello\nworld"
    assert agent._detect_provider() == "unknown"


@pytest.mark.asyncio
async def test_classifier_and_profiler_parse_model_output():
    classifier = ClassifierAgent(FakeChatModel(responses=["profile::I work at XMem\ncode::Explain retry"]))
    classified = await classifier.arun({"user_query": "I work at XMem and need retry code"})
    assert classified.classifications == [
        {"source": "profile", "query": "I work at XMem"},
        {"source": "code", "query": "Explain retry"},
    ]

    profiler = ProfilerAgent(FakeChatModel(responses=["---\nwork::company::XMem"]))
    profile = await profiler.arun({"classifier_output": "I work at XMem"})
    assert profile.facts[0].topic == "work"
    assert profile.facts[0].sub_topic == "company"
    assert profile.facts[0].memo == "XMem"


@pytest.mark.asyncio
async def test_summarizer_and_temporal_empty_paths_do_not_call_model():
    model = FakeChatModel()
    assert (await SummarizerAgent(model).arun({})).is_empty
    assert (await TemporalAgent(model).arun({})).is_empty
    assert model.calls == []
    assert TemporalAgent._validate_date_format("02-29")
    assert not TemporalAgent._validate_date_format("02-30")


def test_code_and_snippet_agents_parse_json_fences_and_defaults():
    code = CodeAgent(FakeChatModel())._parse_response(
        """
        ```json
        {"annotations": [{
          "target_symbol": "PaymentProcessor.process",
          "target_file": "src/payments.py",
          "content": "Retry can duplicate charges",
          "annotation_type": "bug_report",
          "severity": "high",
          "repo": "payments"
        }]}
        ```
        """
    )
    assert code.annotations[0].annotation_type == AnnotationType.BUG_REPORT
    assert code.annotations[0].severity == AnnotationSeverity.HIGH

    snippets = SnippetAgent(FakeChatModel())._parse_response(
        '{"snippets": [{"content": "Binary search", "language": "python", "snippet_type": "pattern", "tags": "search,array"}]}'
    )
    assert snippets.snippets[0].snippet_type == SnippetType.PATTERN
    assert snippets.snippets[0].tags == ["search", "array"]


def test_judge_deterministic_helper_functions():
    assert _build_profile_metadata_key({"topic": "Work", "sub_topic": "Company"}) == "work_company"
    assert _dedupe_profile_items([
        {"topic": "work", "sub_topic": "company", "memo": "Old"},
        {"topic": "work", "sub_topic": "company", "memo": "New"},
    ]) == [{"topic": "work", "sub_topic": "company", "memo": "New"}]
    assert _same_temporal_event(
        {"date": "05-11", "event_name": "Demo", "desc": "Launch", "year": "2026", "time": "", "date_expression": "today"},
        {"date": "05-11", "event_name": "demo", "desc": "Launch", "year": 2026, "time": None, "date_expression": "today"},
    )
