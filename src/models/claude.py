"""
Claude model factory.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from src.config import settings


def build_claude_model(
    model_name: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    api_key = settings.claude_api_key
    if not api_key:
        raise ValueError("CLAUDE_API_KEY is not set")

    return ChatAnthropic(
        model=model_name or settings.claude_model,
        api_key=api_key,
        temperature=temperature if temperature is not None else settings.temperature,
    )
