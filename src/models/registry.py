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
from functools import lru_cache
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
    "openrouter": lambda **kw: _build_from_module("openrouter", "build_openrouter_model", **kw),
}


_KEY_MAP = {
    "gemini": lambda: settings.gemini_api_key,
    "claude": lambda: settings.claude_api_key,
    "openai": lambda: settings.openai_api_key,
    "openrouter": lambda: settings.openrouter_api_key,
}


@lru_cache(maxsize=16)
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


# ---------------------------------------------------------------------------
# Vision model (for image analysis)
# ---------------------------------------------------------------------------

_VISION_MODEL_MAP = {
    "gemini": lambda: settings.gemini_vision_model,
    "claude": lambda: settings.claude_vision_model,
    "openai": lambda: settings.openai_vision_model,
    "openrouter": lambda: settings.openrouter_vision_model,
}


@lru_cache(maxsize=16)
def get_vision_model(
    provider: Optional[Provider] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """Build and return a vision-capable chat model.

    Vision models accept multimodal input (text + images).  The model name
    is resolved from the ``*_vision_model`` settings field for the chosen
    provider, falling back through ``settings.fallback_order`` when
    *provider* is ``None``.

    The returned object is a normal LangChain ``BaseChatModel``; the only
    difference from ``get_model`` is the model name pointing at a
    vision-capable variant.
    """
    if provider:
        vision_name = _VISION_MODEL_MAP[provider]()
        return get_model(provider=provider, model_name=vision_name, temperature=temperature)

    # Auto-select from fallback order
    errors: list[str] = []
    for p in settings.fallback_order:
        key_fn = _KEY_MAP.get(p)
        if key_fn and key_fn():
            try:
                vision_name = _VISION_MODEL_MAP[p]()
                model = _BUILDERS[p](model_name=vision_name, **({"temperature": temperature} if temperature is not None else {}))
                logger.info("Using vision provider: %s (model: %s)", p, vision_name)
                return model
            except Exception as exc:
                errors.append(f"{p}: {exc}")
                logger.warning("Vision provider %s failed: %s", p, exc)

    raise RuntimeError(
        f"No vision-capable LLM provider could be initialised. "
        f"Tried: {settings.fallback_order}. Errors: {errors}"
    )
