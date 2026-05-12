import pytest

from src.utils.text import (
    attribute_unify,
    pack_classifications_into_string,
    parse_raw_response_to_classifications,
    pack_profiles_into_string,
    parse_raw_response_to_profiles,
    parse_raw_response_to_events,
    parse_raw_response_to_event,
    parse_raw_response_to_image,
)


# ---------------------------------------------------------------------------
# attribute_unify
# ---------------------------------------------------------------------------

class TestAttributeUnify:

    def test_lowercases_value(self):
        assert attribute_unify("Hello") == "hello"

    def test_replaces_spaces_with_underscores(self):
        assert attribute_unify("my topic") == "my_topic"

    def test_handles_uppercase_and_spaces_together(self):
        assert attribute_unify("Work Experience") == "work_experience"

    def test_already_lowercase_unchanged(self):
        assert attribute_unify("work") == "work"

    def test_empty_string_returns_empty(self):
        assert attribute_unify("") == ""


# ---------------------------------------------------------------------------
# pack_classifications_into_string
# ---------------------------------------------------------------------------

class TestPackClassificationsIntoString:

    def test_single_classification_formatted_correctly(self):
        result = pack_classifications_into_string([{"source": "profile", "query": "my name"}])
        assert result == "profile::my name"

    def test_multiple_classifications_joined_with_newline(self):
        result = pack_classifications_into_string([
            {"source": "profile", "query": "my name"},
            {"source": "code", "query": "write hello world"},
        ])
        assert result == "profile::my name\ncode::write hello world"

    def test_empty_list_returns_empty_string(self):
        assert pack_classifications_into_string([]) == ""


# ---------------------------------------------------------------------------
# parse_raw_response_to_classifications
# ---------------------------------------------------------------------------

class TestParseRawResponseToClassifications:

    def test_valid_line_returns_classification(self):
        result = parse_raw_response_to_classifications("profile::my name is Alice")
        assert len(result) == 1
        assert result[0]["source"] == "profile"
        assert result[0]["query"] == "my name is Alice"

    def test_all_four_valid_sources_accepted(self):
        content = "profile::q1\ncode::q2\nevent::q3\nimage::q4"
        result = parse_raw_response_to_classifications(content)
        sources = [r["source"] for r in result]
        assert sources == ["profile", "code", "event", "image"]

    def test_unknown_source_is_skipped(self):
        result = parse_raw_response_to_classifications("memory::something")
        assert result == []

    def test_line_without_separator_is_skipped(self):
        result = parse_raw_response_to_classifications("profile something without separator")
        assert result == []

    def test_empty_query_after_separator_is_skipped(self):
        result = parse_raw_response_to_classifications("profile::")
        assert result == []

    def test_uppercase_source_is_normalized(self):
        result = parse_raw_response_to_classifications("PROFILE::my name")
        assert len(result) == 1
        assert result[0]["source"] == "profile"

    def test_whitespace_around_values_is_stripped(self):
        result = parse_raw_response_to_classifications("  profile  ::  my name  ")
        assert result[0]["source"] == "profile"
        assert result[0]["query"] == "my name"

    def test_multiple_valid_lines_all_returned(self):
        content = "profile::name\ncode::hello"
        result = parse_raw_response_to_classifications(content)
        assert len(result) == 2

    def test_mixed_valid_and_invalid_lines(self):
        content = "profile::name\nbadline\nunknown::query"
        result = parse_raw_response_to_classifications(content)
        assert len(result) == 1
        assert result[0]["source"] == "profile"

    def test_empty_string_returns_empty_list(self):
        assert parse_raw_response_to_classifications("") == []


# ---------------------------------------------------------------------------
# pack_profiles_into_string
# ---------------------------------------------------------------------------

class TestPackProfilesIntoString:

    def test_single_dict_fact_formatted_correctly(self):
        facts = [{"topic": "work", "sub_topic": "company", "memo": "Google"}]
        result = pack_profiles_into_string(facts)
        assert result == "work::company::Google"

    def test_topic_with_spaces_is_underscored(self):
        facts = [{"topic": "basic info", "sub_topic": "full name", "memo": "Alice"}]
        result = pack_profiles_into_string(facts)
        assert result == "basic_info::full_name::Alice"

    def test_multiple_facts_joined_with_newline(self):
        facts = [
            {"topic": "work", "sub_topic": "company", "memo": "Google"},
            {"topic": "basic_info", "sub_topic": "name", "memo": "Alice"},
        ]
        result = pack_profiles_into_string(facts)
        assert "work::company::Google" in result
        assert "basic_info::name::Alice" in result
        assert result.count("\n") == 1

    def test_empty_list_returns_none_string(self):
        assert pack_profiles_into_string([]) == "NONE"

    def test_accepts_object_with_attributes(self):
        class Fact:
            topic = "work"
            sub_topic = "company"
            memo = "Google"

        result = pack_profiles_into_string([Fact()])
        assert result == "work::company::Google"

    def test_memo_whitespace_is_stripped(self):
        facts = [{"topic": "work", "sub_topic": "company", "memo": "  Google  "}]
        result = pack_profiles_into_string(facts)
        assert result == "work::company::Google"


# ---------------------------------------------------------------------------
# parse_raw_response_to_profiles
# ---------------------------------------------------------------------------

class TestParseRawResponseToProfiles:

    def test_valid_line_returns_fact_dict(self):
        result = parse_raw_response_to_profiles("work::company::Google")
        assert len(result) == 1
        assert result[0] == {"topic": "work", "sub_topic": "company", "memo": "Google"}

    def test_ignores_text_before_triple_dash(self):
        content = "I am thinking about this...\n---\nwork::company::Google"
        result = parse_raw_response_to_profiles(content)
        assert len(result) == 1
        assert result[0]["topic"] == "work"

    def test_no_triple_dash_parses_everything(self):
        result = parse_raw_response_to_profiles("work::company::Google")
        assert len(result) == 1

    def test_memo_containing_separator_is_rejoined(self):
        result = parse_raw_response_to_profiles("work::company::Google::Mountain View")
        assert result[0]["memo"] == "Google::Mountain View"

    def test_line_with_fewer_than_three_parts_skipped(self):
        result = parse_raw_response_to_profiles("work::company")
        assert result == []

    def test_line_without_separator_skipped(self):
        result = parse_raw_response_to_profiles("just a plain line")
        assert result == []

    def test_empty_string_returns_empty_list(self):
        assert parse_raw_response_to_profiles("") == []

    def test_topic_and_sub_topic_lowercased(self):
        result = parse_raw_response_to_profiles("WORK::COMPANY::Google")
        assert result[0]["topic"] == "work"
        assert result[0]["sub_topic"] == "company"


# ---------------------------------------------------------------------------
# parse_raw_response_to_events
# ---------------------------------------------------------------------------

class TestParseRawResponseToEvents:

    def test_no_event_marker_returns_empty_list(self):
        assert parse_raw_response_to_events("NO_EVENT") == []

    def test_no_event_lowercase_returns_empty_list(self):
        assert parse_raw_response_to_events("no_event") == []

    def test_single_event_block_parsed(self):
        content = "DATE: 03-15\nEVENT_NAME: Birthday\nYEAR: 1995\nDESC: Alice's birthday"
        result = parse_raw_response_to_events(content)
        assert len(result) == 1
        assert result[0]["date"] == "03-15"
        assert result[0]["event_name"] == "Birthday"

    def test_multiple_events_separated_by_dashes(self):
        content = "DATE: 03-15\nEVENT_NAME: Birthday\n---\nDATE: 12-25\nEVENT_NAME: Christmas"
        result = parse_raw_response_to_events(content)
        assert len(result) == 2

    def test_event_without_date_is_excluded(self):
        content = "EVENT_NAME: Birthday\nDESC: Something"
        result = parse_raw_response_to_events(content)
        assert result == []

    def test_year_stored_as_integer_when_valid(self):
        content = "DATE: 03-15\nYEAR: 1995"
        result = parse_raw_response_to_events(content)
        assert result[0]["year"] == 1995
        assert isinstance(result[0]["year"], int)

    def test_year_stored_as_string_when_not_a_number(self):
        content = "DATE: 03-15\nYEAR: unknown"
        result = parse_raw_response_to_events(content)
        assert result[0]["year"] == "unknown"

    def test_all_fields_parsed_correctly(self):
        content = (
            "DATE: 06-10\n"
            "EVENT_NAME: Graduation\n"
            "YEAR: 2023\n"
            "DESC: University graduation ceremony\n"
            "TIME: 2:00 PM\n"
            "DATE_EXPRESSION: next June"
        )
        result = parse_raw_response_to_events(content)
        event = result[0]
        assert event["date"] == "06-10"
        assert event["event_name"] == "Graduation"
        assert event["year"] == 2023
        assert event["desc"] == "University graduation ceremony"
        assert event["time"] == "2:00 PM"
        assert event["date_expression"] == "next June"

    def test_empty_string_returns_empty_list(self):
        assert parse_raw_response_to_events("") == []


# ---------------------------------------------------------------------------
# parse_raw_response_to_event (single-event wrapper)
# ---------------------------------------------------------------------------

class TestParseRawResponseToEvent:

    def test_returns_first_event(self):
        content = "DATE: 03-15\nEVENT_NAME: Birthday\n---\nDATE: 12-25\nEVENT_NAME: Christmas"
        result = parse_raw_response_to_event(content)
        assert result["date"] == "03-15"

    def test_returns_none_when_no_event(self):
        assert parse_raw_response_to_event("NO_EVENT") is None

    def test_returns_none_for_empty_string(self):
        assert parse_raw_response_to_event("") is None


# ---------------------------------------------------------------------------
# parse_raw_response_to_image
# ---------------------------------------------------------------------------

class TestParseRawResponseToImage:

    def test_full_format_parsed_correctly(self):
        content = (
            "DESCRIPTION: A dog sitting in a park\n\n"
            "OBSERVATIONS:\n"
            "- [animal] brown labrador (confidence: high)\n"
            "- [setting] outdoor park bench (confidence: medium)\n"
        )
        result = parse_raw_response_to_image(content)
        assert result["description"] == "A dog sitting in a park"
        assert len(result["observations"]) == 2

    def test_missing_description_returns_empty_string(self):
        content = "OBSERVATIONS:\n- [animal] a cat\n"
        result = parse_raw_response_to_image(content)
        assert result["description"] == ""

    def test_missing_observations_returns_empty_list(self):
        content = "DESCRIPTION: A dog in a park"
        result = parse_raw_response_to_image(content)
        assert result["observations"] == []

    def test_observation_category_extracted(self):
        content = "OBSERVATIONS:\n- [animal] brown dog\n"
        result = parse_raw_response_to_image(content)
        assert result["observations"][0]["category"] == "animal"

    def test_observation_without_category_defaults_to_other(self):
        content = "OBSERVATIONS:\n- just a plain observation\n"
        result = parse_raw_response_to_image(content)
        assert result["observations"][0]["category"] == "other"

    def test_observation_confidence_extracted(self):
        content = "OBSERVATIONS:\n- [animal] brown dog (confidence: high)\n"
        result = parse_raw_response_to_image(content)
        assert result["observations"][0]["confidence"] == "high"

    def test_observation_without_confidence_is_none(self):
        content = "OBSERVATIONS:\n- [animal] brown dog\n"
        result = parse_raw_response_to_image(content)
        assert result["observations"][0]["confidence"] is None

    def test_observation_description_extracted(self):
        content = "OBSERVATIONS:\n- [animal] brown dog (confidence: high)\n"
        result = parse_raw_response_to_image(content)
        assert result["observations"][0]["description"] == "brown dog"

    def test_empty_string_returns_defaults(self):
        result = parse_raw_response_to_image("")
        assert result["description"] == ""
        assert result["observations"] == []
