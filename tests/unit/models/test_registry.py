import pytest
from unittest.mock import MagicMock, patch
import sys
import importlib


@pytest.fixture
def mock_modules():
    with patch.dict(
        sys.modules,
        {
            "src.models.gemini": MagicMock(),
            "src.models.claude": MagicMock(),
            "src.models.openai": MagicMock(),
        },
    ):
        yield


def test_build_gemini(mock_modules):
    from src.models import registry

    importlib.reload(registry)  # Reload to ensure _BUILDERS is fresh if needed

    mock_builder = MagicMock()
    sys.modules["src.models.gemini"].build_gemini_model = mock_builder

    # We are testing the _BUILDERS map effectively
    registry._BUILDERS["gemini"](temperature=0.5)

    mock_builder.assert_called_once_with(temperature=0.5)


def test_build_claude(mock_modules):
    from src.models import registry

    importlib.reload(registry)

    mock_builder = MagicMock()
    sys.modules["src.models.claude"].build_claude_model = mock_builder

    registry._BUILDERS["claude"](model_name="claude-3")

    mock_builder.assert_called_once_with(model_name="claude-3")


def test_build_openai(mock_modules):
    from src.models import registry

    importlib.reload(registry)

    mock_builder = MagicMock()
    sys.modules["src.models.openai"].build_openai_model = mock_builder

    registry._BUILDERS["openai"]()

    mock_builder.assert_called_once()


def test_get_model_specific_provider(mock_modules):
    from src.models import registry

    importlib.reload(registry)

    mock_builder = MagicMock()
    sys.modules["src.models.gemini"].build_gemini_model = mock_builder

    registry.get_model("gemini", temperature=0.7)

    mock_builder.assert_called_once_with(temperature=0.7)


def test_get_model_fallback(mock_modules):
    from src.models import registry

    importlib.reload(registry)

    # Mock settings.fallback_order and API keys
    # Note: registry imports settings, so we need to patch it in registry
    with patch("src.models.registry.settings") as mock_settings:
        mock_settings.fallback_order = ["openai", "gemini"]
        mock_settings.openai_api_key = "sk-test"
        mock_settings.gemini_api_key = None

        mock_openai_builder = MagicMock()
        sys.modules["src.models.openai"].build_openai_model = mock_openai_builder

        registry.get_model()

        mock_openai_builder.assert_called_once()

def test_get_model_caching(mock_modules):
    from src.models import registry

    importlib.reload(registry)

    # Configure mock builder to return a NEW mock instance each time called
    mock_builder = MagicMock()
    mock_builder.side_effect = lambda **kwargs: MagicMock()

    sys.modules["src.models.openai"].build_openai_model = mock_builder

    # First call
    model1 = registry.get_model("openai", temperature=0.7)

    # Second call (same args) -> Should be cached
    model2 = registry.get_model("openai", temperature=0.7)

    # Third call (different args) -> Should be new
    model3 = registry.get_model("openai", temperature=0.8)

    assert model1 is model2
    assert model1 is not model3

    # Builder called for 0.7 and 0.8
    assert mock_builder.call_count == 2
