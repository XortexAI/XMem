"""
DeepSeek model factory.
"""

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from src.config import settings


def build_deepseek_model(
    model_name: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    api_key = settings.deepseek_api_key
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    return ChatOpenAI(
        model=model_name or settings.deepseek_model,
        api_key=api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature if temperature is not None else settings.temperature,
        timeout=settings.llm_timeout_seconds,
    )
