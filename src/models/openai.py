"""
OpenAI model factory.
"""

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from src.config import settings


def build_openai_model(
    model_name: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    api_key = settings.openai_api_key
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    return ChatOpenAI(
        model=model_name or settings.openai_model,
        api_key=api_key,
        temperature=temperature if temperature is not None else settings.temperature,
    )
