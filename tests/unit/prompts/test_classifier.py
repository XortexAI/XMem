import sys
from unittest.mock import MagicMock, patch

def get_mocks():
    """Return a dictionary of mocks for missing dependencies."""
    mocks = {
        "pydantic": MagicMock(),
        "pydantic_settings": MagicMock(),
        "langchain_core": MagicMock(),
        "langchain_core.language_models": MagicMock(),
        "typing_extensions": MagicMock(),
    }

    # Provide a real-ish Classification for typing
    class MockClassification(dict):
        pass

    schema_mock = MagicMock()
    schema_mock.Classification = MockClassification
    mocks["src.schemas.classification"] = schema_mock

    return mocks

def test_build_system_prompt_structure():
    """Verify that the system prompt contains all expected sections."""
    with patch.dict("sys.modules", get_mocks()):
        from src.prompts.classifier import build_system_prompt
        prompt = build_system_prompt()

        assert "You are an intelligent intent router" in prompt
        assert "## Available Agents" in prompt
        assert "### 1. `code`" in prompt
        assert "### 2. `profile`" in prompt
        assert "### 3. `event`" in prompt
        assert "## Logic & Strategy" in prompt
        assert "## Output Format (Strict)" in prompt
        assert "## Examples" in prompt

def test_build_system_prompt_contains_keywords():
    """Verify that keywords from all agents are present in the system prompt."""
    with patch.dict("sys.modules", get_mocks()):
        from src.prompts.classifier import build_system_prompt
        from src.prompts.classifier_keywords import (
            CODE_AGENT_KEYWORDS,
            EVENT_AGENT_KEYWORDS,
            PROFILE_AGENT_KEYWORDS,
        )
        prompt = build_system_prompt()

        # Check a few representative keywords from each agent
        assert any(k in prompt for k in CODE_AGENT_KEYWORDS)
        assert any(k in prompt for k in EVENT_AGENT_KEYWORDS)
        assert any(k in prompt for k in PROFILE_AGENT_KEYWORDS)

        # Verify keyword blocks are injected
        assert "**Keywords**:" in prompt

def test_build_system_prompt_contains_examples():
    """Verify that examples are correctly injected into the system prompt."""
    with patch.dict("sys.modules", get_mocks()):
        from src.prompts.classifier import build_system_prompt
        from src.prompts.examples.classification import CLASSIFICATION_EXAMPLES
        prompt = build_system_prompt()

        # Check that <example> tags are present
        assert "<example>" in prompt
        assert "</example>" in prompt
        assert "<input>" in prompt
        assert "<output>" in prompt

        # Verify at least one specific example from CLASSIFICATION_EXAMPLES is present
        if CLASSIFICATION_EXAMPLES:
            first_input = CLASSIFICATION_EXAMPLES[0][0]
            assert first_input in prompt

def test_pack_classification_query():
    """Verify that the user query is correctly wrapped."""
    with patch.dict("sys.modules", get_mocks()):
        from src.prompts.classifier import pack_classification_query
        user_input = "Tell me a joke about Python"
        packed = pack_classification_query(user_input)

        assert "Analyze this user input:" in packed
        assert f"User Input: {user_input}" in packed
