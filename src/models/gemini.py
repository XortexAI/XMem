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
        max_output_tokens=8192,
    )

    # For Gemini thinking models, optionally set thinking_level.
    # Only applied if GEMINI_THINKING_LEVEL is explicitly set in env.
    # Values: 'low', 'medium', 'high'. If unset, no thinking config is applied.
    thinking_level = os.environ.get("GEMINI_THINKING_LEVEL", "")
    if thinking_level and thinking_level.lower() in ("low", "medium", "high"):
        kwargs["thinking_level"] = thinking_level.lower()

    return ChatGoogleGenerativeAI(**kwargs)
