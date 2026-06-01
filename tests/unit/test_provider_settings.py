import pytest

from src.config.settings import Settings
from src.models import deepseek as deepseek_module
from src.models import mimo as mimo_module


def test_settings_accept_deepseek_fallback_provider():
    settings = Settings(
        neo4j_password="test-password",
        fallback_order=["deepseek"],
        deepseek_api_key="test-key",
    )

    assert settings.fallback_order == ["deepseek"]


def test_settings_accept_mimo_fallback_provider():
    settings = Settings(
        neo4j_password="test-password",
        fallback_order=["mimo"],
        mimo_api_key="test-key",
    )

    assert settings.fallback_order == ["mimo"]


def test_build_deepseek_model_uses_official_endpoint(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(deepseek_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(deepseek_module.settings, "deepseek_api_key", "deepseek-key", raising=False)
    monkeypatch.setattr(deepseek_module.settings, "deepseek_base_url", "https://api.deepseek.com", raising=False)
    monkeypatch.setattr(deepseek_module.settings, "deepseek_model", "deepseek-v4-flash", raising=False)
    monkeypatch.setattr(deepseek_module.settings, "temperature", 0.3, raising=False)
    monkeypatch.setattr(deepseek_module.settings, "llm_timeout_seconds", 15.0, raising=False)

    deepseek_module.build_deepseek_model()

    assert captured["api_key"] == "deepseek-key"
    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["timeout"] == 15.0


def test_build_mimo_model_uses_official_endpoint(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(mimo_module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(mimo_module.settings, "mimo_api_key", "mimo-key", raising=False)
    monkeypatch.setattr(mimo_module.settings, "mimo_base_url", "https://api.xiaomimimo.com/v1", raising=False)
    monkeypatch.setattr(mimo_module.settings, "mimo_model", "mimo-v2.5-pro", raising=False)
    monkeypatch.setattr(mimo_module.settings, "temperature", 0.3, raising=False)
    monkeypatch.setattr(mimo_module.settings, "llm_timeout_seconds", 15.0, raising=False)

    mimo_module.build_mimo_model()

    assert captured["api_key"] == "mimo-key"
    assert captured["base_url"] == "https://api.xiaomimimo.com/v1"
    assert captured["model"] == "mimo-v2.5-pro"
    assert captured["timeout"] == 15.0


def test_build_deepseek_model_requires_api_key(monkeypatch):
    monkeypatch.setattr(deepseek_module.settings, "deepseek_api_key", None, raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is not set"):
        deepseek_module.build_deepseek_model()


def test_build_mimo_model_requires_api_key(monkeypatch):
    monkeypatch.setattr(mimo_module.settings, "mimo_api_key", None, raising=False)

    with pytest.raises(ValueError, match="MIMO_API_KEY is not set"):
        mimo_module.build_mimo_model()
