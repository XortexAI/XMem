from __future__ import annotations

import pytest

from src.utils.exceptions import ValidationError, XMemError
from src.utils.retry import RetryConfig, with_async_retry, with_retry
from src.utils.text import (
    attribute_unify,
    pack_classifications_into_string,
    pack_profiles_into_string,
    parse_raw_response_to_classifications,
    parse_raw_response_to_event,
    parse_raw_response_to_events,
    parse_raw_response_to_image,
    parse_raw_response_to_profiles,
)


def test_classification_text_helpers_round_trip_valid_sources():
    packed = pack_classifications_into_string([
        {"source": "profile", "query": "I work at XMem"},
        {"source": "code", "query": "Explain src/api/app.py"},
    ])

    assert "profile::I work at XMem" in packed
    assert parse_raw_response_to_classifications(
        """
        PROFILE::I work at XMem
        not parseable
        unknown::ignored
        code::Explain src/api/app.py
        """
    ) == [
        {"source": "profile", "query": "I work at XMem"},
        {"source": "code", "query": "Explain src/api/app.py"},
    ]


def test_profile_helpers_preserve_memo_separator_text():
    facts = [{"topic": "Basic Info", "sub_topic": "Favorite Food", "memo": "salt::pepper"}]

    assert attribute_unify("Basic Info") == "basic_info"
    assert pack_profiles_into_string(facts) == "basic_info::favorite_food::salt::pepper"
    assert parse_raw_response_to_profiles("thinking\n---\nwork::company::OpenAI::Research") == [
        {"topic": "work", "sub_topic": "company", "memo": "OpenAI::Research"}
    ]


def test_temporal_and_image_parsers_handle_empty_and_structured_outputs():
    assert parse_raw_response_to_events("NO_EVENT") == []
    assert parse_raw_response_to_event("DATE: 05-11\nEVENT_NAME: Demo\nYEAR: 2026") == {
        "date": "05-11",
        "event_name": "Demo",
        "year": 2026,
        "desc": None,
        "time": None,
        "date_expression": None,
    }

    image = parse_raw_response_to_image(
        """
        DESCRIPTION: Whiteboard architecture sketch
        OBSERVATIONS:
        - [document] API gateway diagram (confidence: high)
        - [text] TODO near auth service
        """
    )

    assert image["description"] == "Whiteboard architecture sketch"
    assert image["observations"][0]["category"] == "document"
    assert image["observations"][0]["confidence"] == "high"
    assert image["observations"][1]["description"] == "TODO near auth service"


def test_retry_retries_transient_failures_and_skips_validation_errors():
    attempts = {"count": 0}

    @with_retry(config=RetryConfig(max_retries=2, delay=0, retryable_exceptions=(RuntimeError,)))
    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary")
        return "ok"

    assert flaky() == "ok"
    assert attempts["count"] == 3

    @with_retry(max_retries=3, delay=0)
    def invalid():
        raise ValidationError("bad input")

    with pytest.raises(ValidationError):
        invalid()


@pytest.mark.asyncio
async def test_async_retry_retries_async_transient_failures(monkeypatch):
    attempts = {"count": 0}

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("asyncio.sleep", no_sleep)

    @with_async_retry(config=RetryConfig(max_retries=1, delay=0, retryable_exceptions=(RuntimeError,)))
    async def flaky():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary")
        return "ok"

    assert await flaky() == "ok"
    assert attempts["count"] == 2


def test_xmem_error_serializes_context():
    error = XMemError("failed", operation="write", details={"id": "1"})

    assert str(error) == "[write] failed"
    assert error.to_dict() == {
        "error": "XMemError",
        "message": "failed",
        "operation": "write",
        "details": {"id": "1"},
    }
