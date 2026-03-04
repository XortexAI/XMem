"""
Gemini model factory.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from src.config import settings


def build_gemini_model(
    model_name: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    api_key = settings.gemini_api_key
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    resolved_model = model_name or settings.gemini_model

    kwargs = dict(
        model=resolved_model,
        google_api_key=api_key,
        temperature=temperature if temperature is not None else settings.temperature,
    )

    # For Gemini 3+ / 2.5 thinking models, set thinking_level to reduce latency
    # Env var GEMINI_THINKING_LEVEL: 'low', 'medium', 'high' (default: 'low')
    if "gemini-3" in resolved_model or "gemini-2.5" in resolved_model:
        thinking_level = os.environ.get("GEMINI_THINKING_LEVEL", "low")
        if thinking_level and thinking_level.lower() in ("low", "medium", "high"):
            kwargs["thinking_level"] = thinking_level.lower()

    return ChatGoogleGenerativeAI(**kwargs)
