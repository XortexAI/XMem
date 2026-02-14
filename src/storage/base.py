"""
================================================================================
BASE VECTOR STORE - ABSTRACT BASE CLASS (Interface Definition)
================================================================================

WHY THIS FILE EXISTS (Design Pattern Explanation):
--------------------------------------------------
This file defines an INTERFACE (contract) that all vector store implementations
must follow. This is called the "Strategy Pattern" or "Dependency Inversion Principle".

WHAT IS AN ABSTRACT BASE CLASS (ABC)?
-------------------------------------
- An ABC is a class that CANNOT be instantiated directly
- It defines METHOD SIGNATURES that child classes MUST implement
- If a child class doesn't implement all @abstractmethod methods, Python raises TypeError

WHY DEFINE add() HERE AND ALSO IN pinecone.py?
----------------------------------------------
- base.py: Defines WHAT methods exist and their signatures (the contract/interface)
- pinecone.py: Defines HOW those methods actually work with Pinecone specifically

BENEFITS OF THIS PATTERN:
-------------------------
1. SWAPPABILITY: Switch from Pinecone to Qdrant/Chroma without changing app code
2. TESTABILITY: Mock the interface in unit tests without real database
3. TYPE SAFETY: IDE autocomplete works; type checkers catch errors
4. DOCUMENTATION: Interface documents expected behavior for all implementations
5. LOOSE COUPLING: App code depends on abstract interface, not concrete implementation

EXAMPLE USAGE:
--------------
    # In your app code, depend on the INTERFACE, not the concrete class:
    def process_memories(store: BaseVectorStore):  # <- Takes ANY vector store
        store.add(texts, embeddings)  # Works with Pinecone, Qdrant, Chroma, etc.
    
    # At startup, inject the concrete implementation:
    store = PineconeVectorStore()  # or QdrantVectorStore() or ChromaVectorStore()
    process_memories(store)

================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

# abc module: Provides Abstract Base Class functionality
# ABC: Base class that makes this class abstract (cannot be instantiated)
# abstractmethod: Decorator that marks methods that MUST be implemented by children
from abc import ABC, abstractmethod

# typing module: Provides type hints for better code documentation and IDE support
# List: Type hint for list, e.g., List[str] means "list of strings"
# Dict: Type hint for dictionary, e.g., Dict[str, Any] means "dict with string keys"
# Any: Type hint meaning "any type is allowed"
# Optional: Type hint meaning "this value can be None", e.g., Optional[str] = str | None
# Tuple: Type hint for tuple, e.g., Tuple[str, int] means "(string, integer)"
from typing import List, Dict, Any, Optional, Tuple

# dataclasses module: Provides @dataclass decorator for automatic __init__, __repr__, etc.
# dataclass: Decorator that auto-generates boilerplate code for data-holding classes
# field: Function to customize individual fields in a dataclass
from dataclasses import dataclass, field

# enum module: Provides Enum class for creating enumerated constants
# Enum: Base class for creating enumeration types (fixed set of named values)
from enum import Enum
from ..config import get_logger
from ..utils.exceptions import (
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreValidationError,
    VectorNotFoundError,
)

logger = get_logger(__name__)
class DistanceMetric(str, Enum):
    COSINE = "cosine"           # Cosine similarity: measures angle between vectors (most common)
    EUCLIDEAN = "euclidean"     # Euclidean distance: straight-line distance in vector space
    DOT_PRODUCT = "dotproduct"  # Dot product: sum of element-wise products (faster but requires normalized vectors)


@dataclass
class SearchResult:
    """
    Represents a single search result from the vector store.
    
    WHAT IS @dataclass?
    -------------------
    - A decorator that auto-generates __init__, __repr__, __eq__ methods
    - You just define attributes with type hints; Python creates the boilerplate
    
    WITHOUT @dataclass (you'd have to write):
    ------------------------------------------
        class SearchResult:
            def __init__(self, id, content, score, metadata):
                self.id = id
                self.content = content
                self.score = score
                self.metadata = metadata
            def __repr__(self):
                return f"SearchResult(id={self.id}, ...)"
            def __eq__(self, other):
                return self.id == other.id and ...
    
    WITH @dataclass (all auto-generated):
    -------------------------------------
        @dataclass
        class SearchResult:
            id: str
            content: str
            score: float
            metadata: Dict[str, Any]
    
    USAGE:
    ------
        result = SearchResult(id="123", content="hello", score=0.95, metadata={})
        print(result)  # SearchResult(id='123', content='hello', score=0.95, metadata={})
        print(result.score)  # 0.95
    """
    
    # Unique identifier of the document/vector
    id: str
    
    # The actual text content that was embedded
    content: str
    
    # Similarity score (0.0 to 1.0 for cosine, higher = more similar)
    score: float
    
    # Additional metadata stored with the vector (user_id, timestamp, tags, etc.)
    # field(default_factory=dict) means: if not provided, create a new empty dict
    # WHY default_factory? Because mutable defaults (like {}) are shared across instances!
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Called automatically after __init__ completes.
        
        Use for validation or transformation of input values.
        """
        # Validate score is in reasonable range
        if not 0.0 <= self.score <= 1.0:
            logger.warning(
                f"SearchResult score {self.score} is outside normal range [0, 1]. "
                "This may indicate a different similarity metric is being used."
            )


@dataclass
class VectorDocument:
    """
    Represents a document with its embedding, used for batch operations.
    
    This provides a clean structure for adding multiple documents at once,
    rather than passing parallel lists (texts, embeddings, ids, metadata).
    """
    
    # The text content to store
    text: str
    
    # The embedding vector (list of floats representing the text in vector space)
    embedding: List[float]
    
    # Optional: provide your own ID, or let the store generate one
    id: Optional[str] = None
    
    # Optional: additional metadata to store with the vector
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class IndexStats:
    """
    Statistics about a vector store index.
    
    Useful for monitoring, debugging, and capacity planning.
    """
    
    # Total number of vectors in the index
    total_vector_count: int
    
    # Dimension of vectors in this index
    dimension: int
    
    # Count of vectors per namespace (for stores that support namespaces)
    namespaces: Dict[str, int] = field(default_factory=dict)
    
    # Index fullness percentage (for stores with capacity limits)
    fullness_percentage: Optional[float] = None

class BaseVectorStore(ABC):
    """
    Abstract base class defining the interface for all vector store implementations.
    
    ================================================================================
    DESIGN PATTERN: Strategy Pattern / Dependency Inversion Principle
    ================================================================================
    
    This class defines WHAT operations a vector store must support, not HOW they work.
    Concrete implementations (PineconeVectorStore, QdrantVectorStore, etc.) provide the HOW.
    
    WHY IS THIS GOOD PRACTICE?
    --------------------------
    
    1. LOOSE COUPLING:
       Your application code depends on BaseVectorStore (abstract), not PineconeVectorStore.
       If you switch from Pinecone to Qdrant, only the instantiation changes.
       
       # BAD (tight coupling):
       def save_memory(store: PineconeVectorStore):  # Depends on CONCRETE class
           store.add(...)  # Can ONLY work with Pinecone
       
       # GOOD (loose coupling):
       def save_memory(store: BaseVectorStore):  # Depends on ABSTRACT interface
           store.add(...)  # Works with ANY implementation
    
    2. TESTABILITY:
       You can create a MockVectorStore for unit tests without needing real Pinecone.
       
       class MockVectorStore(BaseVectorStore):
           def add(self, ...): return ["mock-id"]
           def search(self, ...): return [SearchResult(...)]
       
       # In tests:
       store = MockVectorStore()
       result = my_function(store)  # Tests without network calls!
    
    3. DOCUMENTATION:
       This class documents the expected behavior of ALL vector stores.
       Anyone implementing a new backend knows exactly what methods to implement.
    
    4. TYPE CHECKING:
       IDEs and type checkers (mypy, pyright) can verify you're using valid methods.
       If you call store.invalid_method(), the type checker will catch it.
    
    SUPPORTED IMPLEMENTATIONS:
    --------------------------
    - PineconeVectorStore: Managed vector database, serverless, scalable
    - QdrantVectorStore: Open-source, self-hosted or cloud
    - ChromaVectorStore: Lightweight, embedded, great for development
    - PGVectorStore: PostgreSQL extension, good if you already use Postgres
    - WeaviateVectorStore: Open-source, GraphQL API
    
    ================================================================================
    """
    
    @abstractmethod
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add texts with their corresponding embeddings to the vector store.
        
        WHAT IS @abstractmethod?
        ------------------------
        - Decorator from abc module that marks this method as REQUIRED
        - Any class inheriting from BaseVectorStore MUST implement this method
        - If not implemented, Python raises TypeError when you try to instantiate
        
        WHY pass AT THE END?
        --------------------
        - Abstract methods have no implementation (the base class doesn't know HOW)
        - 'pass' is a no-op statement that satisfies Python's requirement for a body
        - The docstring serves as documentation for implementers
        
        Args:
            texts: List of text strings to store
                   Example: ["Hello world", "How are you?"]
            
            embeddings: List of embedding vectors, one per text
                        Each embedding is a list of floats (typically 384-1536 dimensions)
                        Example: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
                        
            ids: Optional list of unique IDs for each text
                 If not provided, the implementation should generate UUIDs
                 Example: ["doc-001", "doc-002"]
                 
            metadata: Optional list of metadata dicts, one per text
                      Example: [{"user_id": "u1", "type": "note"}, {"user_id": "u1"}]
        
        Returns:
            List of IDs for the added documents (generated if not provided)
        
        Raises:
            VectorStoreValidationError: If inputs are invalid (dimension mismatch, empty lists)
            VectorStoreConnectionError: If connection to store fails
            VectorStoreError: For other storage-related errors
        
        Example:
            ids = store.add(
                texts=["Memory 1", "Memory 2"],
                embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
                metadata=[{"user": "alice"}, {"user": "alice"}]
            )
            print(ids)  # ["abc-123", "def-456"]
        """
        pass  # No implementation - concrete classes must provide this
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents using vector similarity.
        
        This is the core retrieval operation - given a query vector,
        find the most similar stored vectors.
        
        Args:
            query_embedding: The embedding vector of the search query
                            Must have same dimension as stored embeddings
                            Example: [0.1, 0.2, 0.3, ...]
            
            top_k: Number of results to return (default: 5)
                   Higher values = more results but slower
                   Example: top_k=10 returns 10 most similar documents
            
            filters: Optional metadata filters to narrow search
                     Format varies by implementation, typically key-value pairs
                     Example: {"user_id": "alice", "type": "note"}
        
        Returns:
            List of SearchResult objects, sorted by similarity (highest first)
            Each result contains: id, content, score, metadata
        
        Raises:
            VectorStoreValidationError: If query embedding dimension is wrong
            VectorStoreConnectionError: If connection to store fails
            VectorStoreError: For other search-related errors
        
        Example:
            results = store.search(
                query_embedding=[0.1, 0.2, ...],
                top_k=3,
                filters={"user_id": "alice"}
            )
            for r in results:
                print(f"{r.content} (score: {r.score})")
        """
        pass
    
    @abstractmethod
    def update(
        self,
        id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing document in the vector store.
        
        Allows partial updates - only provided fields are updated.
        
        Args:
            id: The unique ID of the document to update
            
            text: New text content (optional)
                  If provided, should also provide new embedding
            
            embedding: New embedding vector (optional)
                       Required if text is changed
            
            metadata: New/updated metadata fields (optional)
                      Merged with existing metadata
        
        Returns:
            True if update was successful, False if document not found
        
        Raises:
            VectorNotFoundError: If the document ID doesn't exist (alternative to returning False)
            VectorStoreValidationError: If embedding dimension is wrong
            VectorStoreError: For other update-related errors
        
        Example:
            success = store.update(
                id="doc-123",
                text="Updated content",
                embedding=[0.5, 0.6, ...],
                metadata={"edited": True, "edit_time": "2024-01-15"}
            )
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents from the vector store by their IDs.
        
        Args:
            ids: List of document IDs to delete
                 Example: ["doc-001", "doc-002"]
        
        Returns:
            True if deletion was successful
            (Note: returns True even if some IDs didn't exist - idempotent)
        
        Raises:
            VectorStoreConnectionError: If connection to store fails
            VectorStoreError: For other deletion-related errors
        
        Example:
            store.delete(ids=["doc-001", "doc-002"])
        """
        pass
    
    @abstractmethod
    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents by their IDs (exact lookup, not similarity search).
        
        Args:
            ids: List of document IDs to retrieve
        
        Returns:
            List of document dictionaries, each containing:
            - id: The document ID
            - content: The text content
            - metadata: The metadata dict
            - embedding: The embedding vector (optional, depends on implementation)
        
        Raises:
            VectorStoreConnectionError: If connection to store fails
            VectorStoreError: For other retrieval-related errors
        
        Example:
            docs = store.get(ids=["doc-001", "doc-002"])
            for doc in docs:
                print(f"{doc['id']}: {doc['content']}")
        """
        pass
    
    # ========================================================================
    # OPTIONAL ABSTRACT METHODS - Override if your backend supports them
    # ========================================================================
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the vector store connection is healthy.
        
        Use for monitoring, readiness probes, and connection validation.
        
        Returns:
            True if the store is accessible and working
            False if there are connection issues
        
        Example:
            if not store.health_check():
                logger.error("Vector store is unhealthy!")
                notify_ops_team()
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> IndexStats:
        """
        Get statistics about the vector store index.
        
        Returns:
            IndexStats object with vector count, dimension, etc.
        
        Example:
            stats = store.get_stats()
            print(f"Index has {stats.total_vector_count} vectors")
        """
        pass
    
    # ========================================================================
    # CONCRETE HELPER METHODS - Shared implementation for all stores
    # ========================================================================
    
    def validate_embeddings(
        self,
        embeddings: List[List[float]],
        expected_dimension: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate embedding format and dimensions.
        
        This is a CONCRETE method (not abstract) - it has a real implementation
        that all child classes can use without overriding.
        
        Args:
            embeddings: List of embedding vectors to validate
            expected_dimension: Expected vector dimension (if known)
        
        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        
        Example:
            is_valid, error = store.validate_embeddings(embeddings, expected_dimension=384)
            if not is_valid:
                raise VectorStoreValidationError(error)
        """
        # Check if embeddings list is empty
        if not embeddings:
            return False, "Embeddings list cannot be empty"
        
        # Get dimension from first embedding
        first_dim = len(embeddings[0])
        
        # Check all embeddings have same dimension
        for i, emb in enumerate(embeddings):
            if len(emb) != first_dim:
                return False, f"Embedding {i} has dimension {len(emb)}, expected {first_dim}"
        
        # Check against expected dimension if provided
        if expected_dimension is not None and first_dim != expected_dimension:
            return False, f"Embedding dimension {first_dim} doesn't match expected {expected_dimension}"
        
        return True, None
    
    def validate_inputs(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Validate inputs for add operation.
        
        Raises VectorStoreValidationError if validation fails.
        
        Args:
            texts: List of text strings
            embeddings: List of embedding vectors
            ids: Optional list of IDs
            metadata: Optional list of metadata dicts
        
        Raises:
            VectorStoreValidationError: If any validation check fails
        """
        # Check texts is not empty
        if not texts:
            raise VectorStoreValidationError(
                "Texts list cannot be empty",
                operation="add"
            )
        
        # Check embeddings is not empty
        if not embeddings:
            raise VectorStoreValidationError(
                "Embeddings list cannot be empty",
                operation="add"
            )
        
        # Check lengths match
        if len(texts) != len(embeddings):
            raise VectorStoreValidationError(
                f"Length mismatch: {len(texts)} texts vs {len(embeddings)} embeddings",
                operation="add",
                details={"texts_count": len(texts), "embeddings_count": len(embeddings)}
            )
        
        # Check IDs length if provided
        if ids is not None and len(ids) != len(texts):
            raise VectorStoreValidationError(
                f"Length mismatch: {len(texts)} texts vs {len(ids)} ids",
                operation="add"
            )
        
        # Check metadata length if provided
        if metadata is not None and len(metadata) != len(texts):
            raise VectorStoreValidationError(
                f"Length mismatch: {len(texts)} texts vs {len(metadata)} metadata entries",
                operation="add"
            )
        
        # Validate embedding dimensions are consistent
        is_valid, error = self.validate_embeddings(embeddings)
        if not is_valid:
            raise VectorStoreValidationError(error, operation="add")
