"""
Model registry — single entry-point for getting an LLM instance.

Usage:
    from src.models import get_model
    model = get_model()                   # uses first available provider from fallback_order
    model = get_model("gemini")           # force a specific provider
    model = get_model(temperature=0.0)    # override temperature
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel

from src.config import settings
from src.models.base import Provider

logger = logging.getLogger("xmem.models")

_BUILDERS = {
    "gemini": lambda **kw: _build_gemini(**kw),
    "claude": lambda **kw: _build_claude(**kw),
    "openai": lambda **kw: _build_openai(**kw),
}

_KEY_MAP = {
    "gemini": lambda: settings.gemini_api_key,
    "claude": lambda: settings.claude_api_key,
    "openai": lambda: settings.openai_api_key,
}


def _build_gemini(**kw) -> BaseChatModel:
    from src.models.gemini import build_gemini_model
    return build_gemini_model(**kw)


def _build_claude(**kw) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic
    api_key = settings.claude_api_key
    if not api_key:
        raise ValueError("CLAUDE_API_KEY is not set")
    return ChatAnthropic(
        model=kw.get("model_name") or settings.claude_model,
        api_key=api_key,
        temperature=kw.get("temperature") if kw.get("temperature") is not None else settings.temperature,
    )


def _build_openai(**kw) -> BaseChatModel:
    from src.models.openai import build_openai_model
    return build_openai_model(**kw)


def get_model(
    provider: Optional[Provider] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """Build and return a chat model.

    If *provider* is None the first provider from ``settings.fallback_order``
    whose API key is configured will be used.  Raises ``RuntimeError`` if no
    provider can be initialised.
    """
    kw: dict = {}
    if model_name is not None:
        kw["model_name"] = model_name
    if temperature is not None:
        kw["temperature"] = temperature

    if provider:
        return _BUILDERS[provider](**kw)

    # Auto-select from fallback order
    errors: list[str] = []
    for p in settings.fallback_order:
        key_fn = _KEY_MAP.get(p)
        if key_fn and key_fn():
            try:
                model = _BUILDERS[p](**kw)
                logger.info("Using provider: %s", p)
                return model
            except Exception as exc:
                errors.append(f"{p}: {exc}")
                logger.warning("Provider %s failed: %s", p, exc)

    raise RuntimeError(
        f"No LLM provider could be initialised. Tried: {settings.fallback_order}. "
        f"Errors: {errors}"
    )
