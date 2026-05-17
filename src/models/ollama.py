"""
Ollama model factory for local chat models.
"""

from langchain_core.language_models import BaseChatModel

from src.config import settings


def build_ollama_model(
    model_name: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "Ollama chat support requires langchain-ollama. "
            "Install it with: pip install -e \".[local]\""
        ) from exc

    return ChatOllama(
        model=model_name or settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature if temperature is not None else settings.temperature,
    )
