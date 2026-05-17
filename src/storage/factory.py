"""Factory for vector store backends.

Cloud remains the default. Local providers are selected with
VECTOR_STORE_PROVIDER without changing pipeline code.
"""

from __future__ import annotations

from typing import Optional

from src.config import settings
from src.storage.base import BaseVectorStore
from src.storage.local import ChromaVectorStore, PGVectorStore, SQLiteVectorStore
from src.storage.pinecone import PineconeVectorStore


def get_vector_store(
    *,
    namespace: Optional[str] = None,
    dimension: Optional[int] = None,
    create_if_not_exists: bool = True,
) -> BaseVectorStore:
    provider = (settings.vector_store_provider or "pinecone").strip().lower()
    dimension = dimension or settings.pinecone_dimension
    namespace = namespace or settings.pinecone_namespace

    if provider == "pinecone":
        return PineconeVectorStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            dimension=dimension,
            metric=settings.pinecone_metric,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
            namespace=namespace,
            create_if_not_exists=create_if_not_exists,
        )

    if provider == "pgvector":
        return PGVectorStore(
            url=settings.pgvector_url,
            table=settings.pgvector_table,
            namespace=namespace,
            dimension=dimension,
            create_if_not_exists=create_if_not_exists,
        )

    if provider == "chroma":
        return ChromaVectorStore(
            persist_dir=settings.chroma_persist_dir,
            namespace=namespace,
            dimension=dimension,
            create_if_not_exists=create_if_not_exists,
        )

    if provider in {"sqlite", "sqlite_vec", "sqlite-vec"}:
        return SQLiteVectorStore(
            path=settings.sqlite_vector_path,
            namespace=namespace,
            dimension=dimension,
            create_if_not_exists=create_if_not_exists,
        )

    raise ValueError(
        f"Unsupported VECTOR_STORE_PROVIDER={provider!r}. "
        "Use pinecone, pgvector, chroma, or sqlite."
    )
