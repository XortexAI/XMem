import sys
from unittest.mock import MagicMock, patch

# Mock dependencies
pydantic_mock = MagicMock()
pydantic_mock.BaseModel = dict
sys.modules["pydantic"] = pydantic_mock

pydantic_settings_mock = MagicMock()
pydantic_settings_mock.BaseSettings = dict
sys.modules["pydantic_settings"] = pydantic_settings_mock

classification_mock = MagicMock()
classification_mock.Classification = dict
sys.modules["src.schemas.classification"] = classification_mock

# Now import the target
from src.prompts.classifier import build_system_prompt

def test_build_system_prompt_returns_string():
    prompt = build_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "You are an intelligent intent router" in prompt
    assert "<example>" in prompt
    assert "</example>" in prompt

def test_build_system_prompt_caching():
    # If lru_cache is working, subsequent calls should return the same object
    # (strings are immutable but identity might differ if reconstructed,
    # though CPython often interns string literals. However, format() creates new strings).
    prompt1 = build_system_prompt()
    prompt2 = build_system_prompt()
    assert prompt1 is prompt2  # This confirms caching because without it, format() would create a new string object each time
