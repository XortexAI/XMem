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


# ─────────────────────────────────────────────────────────────────────────────
# Context Window Mappings (in tokens) for each model
# ─────────────────────────────────────────────────────────────────────────────

_CONTEXT_WINDOWS = {
    # Claude models
    "claude": {
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-sonnet": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "claude-opus": 200000,
        "claude-sonnet": 200000,
        "claude-haiku": 200000,
        "default": 200000,
    },
    # OpenAI models
    "openai": {
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
        "default": 128000,
    },
    # Gemini models
    "gemini": {
        "gemini-2.0-flash": 1000000,
        "gemini-2.0-pro": 1000000,
        "gemini-1.5-pro": 1000000,
        "gemini-1.5-flash": 1000000,
        "gemini-pro": 32768,
        "default": 1000000,
    },
    # DeepSeek models
    "deepseek": {
        "deepseek-chat": 128000,
        "deepseek-coder": 128000,
        "default": 128000,
    },
    # Groq models
    "groq": {
        "mixtral-8x7b-32768": 32768,
        "llama2-70b-4096": 4096,
        "default": 32768,
    },
    # OpenRouter (varies by model, use conservative default)
    "openrouter": {
        "default": 128000,
    },
    # Ollama (local, typically depends on model)
    "ollama": {
        "default": 8000,
    },
    # Bedrock (varies by model)
    "bedrock": {
        "default": 100000,
    },
    # Mimo
    "mimo": {
        "default": 32768,
    },
}


def _build_from_module(module_name: str, func_name: str, **kwargs) -> BaseChatModel:
    module = importlib.import_module(f"src.models.{module_name}")
    factory_fn = getattr(module, func_name)
    return factory_fn(**kwargs)


_BUILDERS = {
    "gemini": lambda **kw: _build_from_module("gemini", "build_gemini_model", **kw),
    "claude": lambda **kw: _build_from_module("claude", "build_claude_model", **kw),
    "openai": lambda **kw: _build_from_module("openai", "build_openai_model", **kw),
    "deepseek": lambda **kw: _build_from_module(
        "deepseek", "build_deepseek_model", **kw
    ),
    "groq": lambda **kw: _build_from_module("groq", "build_groq_model", **kw),
    "mimo": lambda **kw: _build_from_module("mimo", "build_mimo_model", **kw),
    "openrouter": lambda **kw: _build_from_module(
        "openrouter", "build_openrouter_model", **kw
    ),
    "bedrock": lambda **kw: _build_from_module("bedrock", "build_bedrock_model", **kw),
    "ollama": lambda **kw: _build_from_module("ollama", "build_ollama_model", **kw),
}


_KEY_MAP = {
    "gemini": lambda: settings.gemini_api_key,
    "claude": lambda: settings.claude_api_key,
    "openai": lambda: settings.openai_api_key,
    "deepseek": lambda: settings.deepseek_api_key,
    "groq": lambda: settings.groq_api_key,
    "mimo": lambda: settings.mimo_api_key,
    "openrouter": lambda: settings.openrouter_api_key,
    "bedrock": lambda: settings.aws_access_key_id,
    "ollama": lambda: True,
}


def get_model_context_window(
    provider: Provider, model_name: Optional[str] = None
) -> int:
    """
    Retrieve the context window (max tokens) for a given provider and model.

    Args:
        provider: The provider name (e.g., 'claude', 'openai', 'gemini')
        model_name: Specific model name. If None, uses the provider default.

    Returns:
        Context window size in tokens.
    """
    if provider not in _CONTEXT_WINDOWS:
        logger.warning(
            f"Provider '{provider}' not found in context window mapping. Using 8192 default."
        )
        return 8192

    provider_windows = _CONTEXT_WINDOWS[provider]

    if model_name:
        # Try exact match first
        if model_name in provider_windows:
            return provider_windows[model_name]
        # Try partial match (e.g., "gpt-4o" matches "gpt-4o-mini")
        # Sort by key length descending so more-specific keys win over shorter prefixes
        for key, window in sorted(
            ((k, v) for k, v in provider_windows.items() if k != "default"),
            key=lambda kv: len(kv[0]),
            reverse=True,
        ):
            if key in model_name:
                logger.debug(
                    f"Matched model '{model_name}' to key '{key}' with context window {window}"
                )
                return window

    # Fall back to provider default
    return provider_windows.get("default", 8192)


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
    "deepseek": lambda: settings.deepseek_vision_model,
    "groq": lambda: settings.groq_vision_model,
    "mimo": lambda: settings.mimo_vision_model,
    "openrouter": lambda: settings.openrouter_vision_model,
    "bedrock": lambda: settings.bedrock_vision_model,
    "ollama": lambda: settings.ollama_vision_model,
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
        return get_model(
            provider=provider, model_name=vision_name, temperature=temperature
        )

    # Auto-select from fallback order
    errors: list[str] = []
    for p in settings.fallback_order:
        key_fn = _KEY_MAP.get(p)
        if key_fn and key_fn():
            try:
                vision_name = _VISION_MODEL_MAP[p]()
                model = _BUILDERS[p](
                    model_name=vision_name,
                    **({"temperature": temperature} if temperature is not None else {}),
                )
                logger.info("Using vision provider: %s (model: %s)", p, vision_name)
                return model
            except Exception as exc:
                errors.append(f"{p}: {exc}")
                logger.warning("Vision provider %s failed: %s", p, exc)

    raise RuntimeError(
        f"No vision-capable LLM provider could be initialised. "
        f"Tried: {settings.fallback_order}. Errors: {errors}"
    )
