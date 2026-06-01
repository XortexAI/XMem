"""
Groq model factory.

Uses ChatGroq with Groq's LPU (Language Processing Unit) for ultra-low-latency
inference — typically 10–20x faster than GPU-based providers. Ideal for XMem's
memory pipeline where ingest latency is critical.
"""

from langchain_core.language_models import BaseChatModel

from src.config import settings


def build_groq_model(
    model_name: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise ImportError(
            "Groq support requires langchain-groq. "
            "Install it with: pip install -e \".[local]\""
        ) from exc

    api_key = settings.groq_api_key
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")

    return ChatGroq(
        api_key=api_key,
        model=model_name or settings.groq_model,
        temperature=temperature if temperature is not None else settings.temperature,
    )
