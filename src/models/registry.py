"""
Model registry — single entry-point for getting an LLM instance.

Usage:
    from src.models import get_model
    model = get_model()                   # uses first available provider from fallback_order
    model = get_model("gemini")           # force a specific provider
    model = get_model(temperature=0.0)    # override temperature
"""

from __future__ import annotations

import importlib
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel

from src.config import settings
from src.models.base import Provider

logger = logging.getLogger("xmem.models")


def _build_from_module(module_name: str, func_name: str, **kwargs) -> BaseChatModel:
    module = importlib.import_module(f"src.models.{module_name}")
    factory_fn = getattr(module, func_name)
    return factory_fn(**kwargs)


_BUILDERS = {
    "gemini": lambda **kw: _build_from_module("gemini", "build_gemini_model", **kw),
    "claude": lambda **kw: _build_from_module("claude", "build_claude_model", **kw),
    "openai": lambda **kw: _build_from_module("openai", "build_openai_model", **kw),
}


_KEY_MAP = {
    "gemini": lambda: settings.gemini_api_key,
    "claude": lambda: settings.claude_api_key,
    "openai": lambda: settings.openai_api_key,
}


def get_model(
    provider: Optional[Provider] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """Build and return a chat model.

    If *provider* is None the first provider from settings.fallback_order
    whose API key is configured will be used.  Raises RuntimeError if no
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
