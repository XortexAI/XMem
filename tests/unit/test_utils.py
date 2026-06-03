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


# ============================================================================
# VALIDATION TESTS - Empty String, Null, and Length Constraint Checks
# ============================================================================

class TestTextValidationEmptyAndNull:
    """Test that text parsing functions reject empty and null inputs."""

    def test_parse_classifications_rejects_empty_string(self):
        """Empty strings should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            parse_raw_response_to_classifications("")

    def test_parse_classifications_rejects_whitespace_only(self):
        """Whitespace-only strings should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            parse_raw_response_to_classifications("   \n\t  ")

    def test_parse_classifications_rejects_none(self):
        """None input should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be None"):
            parse_raw_response_to_classifications(None)

    def test_parse_profiles_rejects_empty_string(self):
        """Empty strings should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            parse_raw_response_to_profiles("")

    def test_parse_profiles_rejects_whitespace_only(self):
        """Whitespace-only strings should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            parse_raw_response_to_profiles("   \n\t  ")

    def test_parse_profiles_rejects_none(self):
        """None input should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be None"):
            parse_raw_response_to_profiles(None)

    def test_parse_events_rejects_empty_string(self):
        """Empty strings should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            parse_raw_response_to_events("")

    def test_parse_events_rejects_whitespace_only(self):
        """Whitespace-only strings should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            parse_raw_response_to_events("   \n\t  ")

    def test_parse_events_rejects_none(self):
        """None input should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be None"):
            parse_raw_response_to_events(None)

    def test_parse_event_rejects_empty_string(self):
        """Empty strings should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            parse_raw_response_to_event("")

    def test_parse_event_rejects_whitespace_only(self):
        """Whitespace-only strings should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            parse_raw_response_to_event("   \n\t  ")

    def test_parse_event_rejects_none(self):
        """None input should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be None"):
            parse_raw_response_to_event(None)

    def test_parse_image_rejects_empty_string(self):
        """Empty strings should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            parse_raw_response_to_image("")

    def test_parse_image_rejects_whitespace_only(self):
        """Whitespace-only strings should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            parse_raw_response_to_image("   \n\t  ")

    def test_parse_image_rejects_none(self):
        """None input should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be None"):
            parse_raw_response_to_image(None)


class TestTextValidationTypeChecks:
    """Test that text parsing functions enforce type checks."""

    def test_parse_classifications_rejects_non_string_types(self):
        """Non-string types should raise TypeError."""
        with pytest.raises(TypeError, match="Input must be a string"):
            parse_raw_response_to_classifications(123)

        with pytest.raises(TypeError, match="Input must be a string"):
            parse_raw_response_to_classifications(["profile::test"])

        with pytest.raises(TypeError, match="Input must be a string"):
            parse_raw_response_to_classifications({"text": "test"})

    def test_parse_profiles_rejects_non_string_types(self):
        """Non-string types should raise TypeError."""
        with pytest.raises(TypeError, match="Input must be a string"):
            parse_raw_response_to_profiles(456)

    def test_parse_events_rejects_non_string_types(self):
        """Non-string types should raise TypeError."""
        with pytest.raises(TypeError, match="Input must be a string"):
            parse_raw_response_to_events(789)

    def test_parse_image_rejects_non_string_types(self):
        """Non-string types should raise TypeError."""
        with pytest.raises(TypeError, match="Input must be a string"):
            parse_raw_response_to_image(999)


class TestTextValidationLengthConstraints:
    """Test that text parsing functions enforce maximum length constraints."""

    def test_parse_classifications_rejects_oversized_payload(self):
        """Oversized payloads should raise ValidationError."""
        from src.utils.text import MAX_STRING_LENGTH
        
        oversized = "x" * (MAX_STRING_LENGTH + 1)
        with pytest.raises(ValidationError, match="exceeds maximum allowed length"):
            parse_raw_response_to_classifications(oversized)

    def test_parse_profiles_rejects_oversized_payload(self):
        """Oversized payloads should raise ValidationError."""
        from src.utils.text import MAX_STRING_LENGTH
        
        oversized = "x" * (MAX_STRING_LENGTH + 1)
        with pytest.raises(ValidationError, match="exceeds maximum allowed length"):
            parse_raw_response_to_profiles(oversized)

    def test_parse_events_rejects_oversized_payload(self):
        """Oversized payloads should raise ValidationError."""
        from src.utils.text import MAX_STRING_LENGTH
        
        oversized = "x" * (MAX_STRING_LENGTH + 1)
        with pytest.raises(ValidationError, match="exceeds maximum allowed length"):
            parse_raw_response_to_events(oversized)

    def test_parse_image_rejects_oversized_payload(self):
        """Oversized payloads should raise ValidationError."""
        from src.utils.text import MAX_STRING_LENGTH
        
        oversized = "x" * (MAX_STRING_LENGTH + 1)
        with pytest.raises(ValidationError, match="exceeds maximum allowed length"):
            parse_raw_response_to_image(oversized)


class TestTextValidationRegressions:
    """Test that valid inputs still work correctly after validation additions."""

    def test_parse_classifications_valid_inputs_still_work(self):
        """Valid classification inputs should still parse correctly."""
        result = parse_raw_response_to_classifications(
            "profile::I work at XMem\ncode::Explain the API"
        )
        assert len(result) == 2
        assert result[0]["source"] == "profile"
        assert result[0]["query"] == "I work at XMem"

    def test_parse_profiles_valid_inputs_still_work(self):
        """Valid profile inputs should still parse correctly."""
        result = parse_raw_response_to_profiles(
            "thinking\n---\nwork::company::Google\nbasic_info::name::Alice"
        )
        assert len(result) == 2
        assert result[0]["topic"] == "work"

    def test_parse_events_valid_inputs_still_work(self):
        """Valid event inputs should still parse correctly."""
        result = parse_raw_response_to_events(
            "DATE: 05-11\nEVENT_NAME: Demo\nYEAR: 2026"
        )
        assert len(result) == 1
        assert result[0]["date"] == "05-11"

    def test_parse_event_valid_inputs_still_work(self):
        """Valid single event inputs should still parse correctly."""
        result = parse_raw_response_to_event(
            "DATE: 05-11\nEVENT_NAME: Demo\nYEAR: 2026"
        )
        assert result["date"] == "05-11"
        assert result["event_name"] == "Demo"

    def test_parse_image_valid_inputs_still_work(self):
        """Valid image inputs should still parse correctly."""
        result = parse_raw_response_to_image(
            "DESCRIPTION: Test image\nOBSERVATIONS:\n- [document] test"
        )
        assert result["description"] == "Test image"
        assert len(result["observations"]) == 1

    def test_parse_events_no_event_keyword_still_works(self):
        """NO_EVENT keyword should still return empty list."""
        result = parse_raw_response_to_events("NO_EVENT")
        assert result == []

    def test_parse_event_with_no_event_keyword_returns_none(self):
        """NO_EVENT keyword should still return None for single event parser."""
        result = parse_raw_response_to_event("NO_EVENT")
        assert result is None

