"""
Gemini model factory.
"""

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

    return ChatGoogleGenerativeAI(
        model=model_name or settings.gemini_model,
        google_api_key=api_key,
        temperature=temperature if temperature is not None else settings.temperature,
    )
