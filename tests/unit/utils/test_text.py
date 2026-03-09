import pytest
from src.utils.text import (
    attribute_unify,
    pack_classifications_into_string,
    parse_raw_response_to_classifications,
)

def test_attribute_unify():
    assert attribute_unify("Favorite Food") == "favorite_food"
    assert attribute_unify("dark_mode") == "dark_mode"
    assert attribute_unify("") == ""


def test_pack_multiple_classifications():
    result = pack_classifications_into_string([
        {"source": "profile", "query": "My name is John"},
        {"source": "event", "query": "my birthday is April 5th"},
    ])
    lines = result.split("\n")
    assert len(lines) == 2
    assert lines[0] == "profile::My name is John"
    assert lines[1] == "event::my birthday is April 5th"


def test_pack_empty_list():
    assert pack_classifications_into_string([]) == ""



class TestParseClassifications:
    def test_parse_multiple_lines(self):
        raw = "event::I graduated in 2020\nprofile::I work as a developer"
        result = parse_raw_response_to_classifications(raw)
        assert len(result) == 2
        assert result[0]["source"] == "event"
        assert result[1]["source"] == "profile"

    def test_ignores_preamble_and_invalid_lines(self):
        # Combined edge case test: preamble + invalid source + valid source
        raw = (
            "Analysis:\n"
            "invalid::junk\n"
            "profile::Valid query"
        )
        result = parse_raw_response_to_classifications(raw)
        assert len(result) == 1
        assert result[0] == {"source": "profile", "query": "Valid query"}

    def test_query_with_separator_in_text(self):
        raw = "code::Fix error: TypeError:: 'int' not iterable"
        result = parse_raw_response_to_classifications(raw)
        assert len(result) == 1
        assert result[0]["query"] == "Fix error: TypeError:: 'int' not iterable"

    def test_empty_and_trivial_input(self):
        assert parse_raw_response_to_classifications("") == []
        assert parse_raw_response_to_classifications("(empty)") == []


    def test_short_parts_edge_case(self):
        """
        Tests the edge case where parts < 2.
        This branch is normally unreachable with standard strings because we previously
        ensure that LLM_TAB_SEPARATOR is in the line.
        We test it using a mock object that overrides __contains__ and split.
        """
        class MockStr(str):
            def __contains__(self, item):
                return True
            def split(self, *args, **kwargs):
                return ["single_part"]

        class MockContent:
            def strip(self):
                return self
            def splitlines(self):
                return [MockStr("dummy")]

        result = parse_raw_response_to_classifications(MockContent())
        assert result == []
