import logging
from typing import List, Optional

from google import genai
from google.genai import types

from src.config import settings

logger = logging.getLogger("xmem.utils.embeddings")

_embedding_client: Optional[genai.Client] = None


def get_embedding_client() -> genai.Client:
    global _embedding_client
    if _embedding_client is None:
        api_key_to_use = settings.gemini_api_key or None
        _embedding_client = genai.Client(api_key=api_key_to_use) if api_key_to_use else genai.Client()
        logger.info("Loaded embedding client for model: %s", settings.embedding_model)
    return _embedding_client


def embed_text(text: str) -> List[float]:
    """Embed a single text string → list of floats."""
    client = get_embedding_client()
    result = client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=settings.pinecone_dimension)
    )
    [embedding_obj] = result.embeddings
    return embedding_obj.values
