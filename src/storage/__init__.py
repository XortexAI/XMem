"""
================================================================================
STORAGE PACKAGE - Vector Store Implementations for XMem
================================================================================

This package provides vector store functionality using the Strategy Pattern.
The base class defines the interface; concrete classes implement specific backends.

ARCHITECTURE:
-------------
    BaseVectorStore (Abstract Interface)
           │
           ├── PineconeVectorStore (Managed cloud service)
           ├── QdrantVectorStore   (Future: open-source)
           ├── ChromaVectorStore   (Future: embedded)
           └── MockVectorStore     (Future: for testing)

USAGE:
------
    # Import the interface and a specific implementation
    from src.storage import BaseVectorStore, PineconeVectorStore
    
    # Create a store instance (uses settings by default)
    store = PineconeVectorStore()
    
    # Or with custom configuration
    store = PineconeVectorStore(
        index_name="my-index",
        namespace="production"
    )
    
    # Use the store
    ids = store.add(texts=["Hello"], embeddings=[[0.1, 0.2, ...]])
    results = store.search(query_embedding=[0.1, 0.2, ...], top_k=5)

DEPENDENCY INJECTION:
---------------------
    # Your business logic depends on the INTERFACE, not concrete implementation
    def process_memories(store: BaseVectorStore):
        # This works with ANY vector store implementation
        results = store.search(query_embedding, top_k=10)
        return results
    
    # At app startup, inject the concrete implementation
    store = PineconeVectorStore()  # or QdrantVectorStore() in the future
    process_memories(store)

================================================================================
"""

# ============================================================================
# IMPORTS FROM SUBMODULES
# ============================================================================

# Import base class and data structures
from .base import (
    # Abstract base class (interface)
    BaseVectorStore,
    
    # Data classes for structured data
    SearchResult,
    VectorDocument,
    IndexStats,
    
    # Enums
    DistanceMetric,
)

# Import concrete implementations
from .pinecone import PineconeVectorStore

# Re-export exceptions for convenience
# (They're also available from src.utils.exceptions)
from ..utils.exceptions import (
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreValidationError,
    VectorNotFoundError,
)

# ============================================================================
# __all__ - Defines what's exported with: from src.storage import *
# ============================================================================

__all__ = [
    # Base class (interface)
    "BaseVectorStore",
    
    # Data classes
    "SearchResult",
    "VectorDocument", 
    "IndexStats",
    
    # Enums
    "DistanceMetric",
    
    # Concrete implementations
    "PineconeVectorStore",
    
    # Exceptions (for convenience)
    "VectorStoreError",
    "VectorStoreConnectionError",
    "VectorStoreValidationError",
    "VectorNotFoundError",
]
