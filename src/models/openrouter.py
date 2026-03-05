"""
OpenRouter model factory.

Uses ChatOpenAI with a custom base_url pointing to OpenRouter's API,
letting you access 200+ models (Llama, Mistral, Gemini, Claude, etc.)
through a single API key.
"""

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from src.config import settings


def build_openrouter_model(
    model_name: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    api_key = settings.openrouter_api_key
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    return ChatOpenAI(
        model=model_name or settings.openrouter_model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature if temperature is not None else settings.temperature,
    )
