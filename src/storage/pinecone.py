"""
================================================================================
PINECONE VECTOR STORE - CONCRETE IMPLEMENTATION
================================================================================

This file provides the CONCRETE implementation of BaseVectorStore for Pinecone.
While base.py defines WHAT methods exist, this file defines HOW they work with Pinecone.


RELATIONSHIP TO base.py:
------------------------
- base.py: Defines the interface (abstract methods)
- pinecone.py: Implements the interface for Pinecone specifically

    BaseVectorStore (Abstract)
           │
           ├── PineconeVectorStore (this file)
           ├── QdrantVectorStore (future)
           ├── ChromaVectorStore (future)
           └── PGVectorStore (future)

================================================================================
"""
from typing import List, Dict, Any, Optional, Final
import uuid

# ----------------------------------------------------------------------------
# THIRD-PARTY IMPORTS (with graceful degradation)
# ----------------------------------------------------------------------------

# Try to import Pinecone SDK
# If not installed, set flag to False so we can raise helpful error later
try:
    # Pinecone: Main client class for interacting with Pinecone service
    # ServerlessSpec: Configuration for serverless index deployment
    from pinecone import Pinecone, ServerlessSpec
    
    # Set flag indicating Pinecone is available
    PINECONE_AVAILABLE: Final[bool] = True
    
except ImportError:
    # ImportError: Raised when an import statement fails to find the module
    # This happens if pinecone-client is not installed
    PINECONE_AVAILABLE: Final[bool] = False

from .base import (
    BaseVectorStore,
    SearchResult,
    IndexStats,
)
from ..config import settings, get_logger
from ..utils.exceptions import (
    VectorStoreConnectionError,
    VectorStoreValidationError,
)
from ..utils.retry import with_retry, RetryConfig
logger = get_logger(__name__)

# Final[int] = type hint indicating this should never be reassigned
PINECONE_BATCH_SIZE: Final[int] = 100  # Recommended by Pinecone for upsert operations
PINECONE_RETRY_CONFIG: Final[RetryConfig] = RetryConfig(
    max_retries=3,           # Retry up to 3 times
    delay=1.0,               # Start with 1 second delay
    backoff_multiplier=2.0,  # Double the delay each retry: 1s, 2s, 4s
    max_delay=30.0,          # Cap at 30 seconds
)

class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone implementation of the BaseVectorStore interface.
    
    This class provides all the operations defined in BaseVectorStore,
    implemented using the Pinecone Python SDK.
    
    CLASS STRUCTURE:
    ----------------
    class PineconeVectorStore(BaseVectorStore):
        │
        ├── __init__()          # Constructor - sets up Pinecone connection
        │
        ├── CRUD Operations (from BaseVectorStore):
        │   ├── add()                # Add vectors to the index
        │   ├── search()             # Find similar vectors
        │   ├── update()             # Update existing vector
        │   ├── delete()             # Remove vectors
        │   ├── get()                # Retrieve vectors by ID
        │   ├── search_by_metadata() # Retrieve by metadata only
        │   └── search_by_text()     # Embed text and search
        │
        ├── Management Operations (from BaseVectorStore):
        │   ├── health_check()  # Check connection status
        │   └── get_stats()     # Get index statistics
        │
        └── Pinecone-Specific Operations:
            ├── count()         # Count vectors in namespace
            ├── clear()         # Clear all vectors in namespace
            ├── delete_index()  # Delete entire index
            └── _build_filter() # Helper for filter construction
    
    USAGE:
    ------
        # Option 1: Use settings defaults
        store = PineconeVectorStore()
        
        # Option 2: Override specific settings
        store = PineconeVectorStore(
            index_name="my-custom-index",
            namespace="production"
        )
        
        # Add vectors
        ids = store.add(
            texts=["Hello world", "How are you?"],
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]]
        )
        
        # Search
        results = store.search(
            query_embedding=[0.15, 0.25, ...],
            top_k=5,
            filters={"user_id": "alice"}
        )
    """
    
    # ========================================================================
    # CONSTRUCTOR
    # ========================================================================
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        metric: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        namespace: Optional[str] = None,
        create_if_not_exists: bool = True
    ) -> None:
        """
        Initialize the Pinecone Vector Store.
        
        WHAT IS __init__?
        -----------------
        - The constructor method, called when you create a new instance
        - PineconeVectorStore() calls this method automatically
        - self: Reference to the instance being created
        
        PARAMETER PATTERN: Optional[T] = None WITH SETTINGS FALLBACK
        -------------------------------------------------------------
        Each parameter can be:
        - Explicitly provided: PineconeVectorStore(api_key="xxx")
        - Omitted (defaults to None): Falls back to settings
        
        This allows:
        - Easy default usage: PineconeVectorStore()
        - Override for testing: PineconeVectorStore(api_key="test-key")
        - Partial override: PineconeVectorStore(namespace="test")
        
        Args:
            api_key: Pinecone API key (falls back to settings.pinecone_api_key)
            index_name: Name of the index (falls back to settings.pinecone_index_name)
            dimension: Vector dimension (falls back to settings.pinecone_dimension)
            metric: Distance metric (falls back to settings.pinecone_metric)
            cloud: Cloud provider (falls back to settings.pinecone_cloud)
            region: Cloud region (falls back to settings.pinecone_region)
            namespace: Namespace for vector isolation (falls back to settings.pinecone_namespace)
            create_if_not_exists: If True, create index if it doesn't exist
        
        Raises:
            ImportError: If pinecone-client is not installed
            VectorStoreConnectionError: If connection to Pinecone fails
        
        Example:
            # Use all defaults from settings
            store = PineconeVectorStore()
            
            # Override specific settings
            store = PineconeVectorStore(
                namespace="testing",
                create_if_not_exists=False
            )
        """
        
        # --------------------------------------------------------------------
        # STEP 1: Check if Pinecone SDK is available
        # --------------------------------------------------------------------
        
        # If the import at the top failed, PINECONE_AVAILABLE will be False
        if not PINECONE_AVAILABLE:
            # Raise ImportError with helpful installation instructions
            raise ImportError(
                "Pinecone SDK is not installed. "
                "Install it with: pip install pinecone-client"
            )
        
        # --------------------------------------------------------------------
        # STEP 2: Resolve configuration (explicit params > settings > defaults)
        # --------------------------------------------------------------------
        
        # Pattern: value = explicit_param if explicit_param is not None else settings.value
        # This allows overriding settings for testing or special cases
        
        # 'or' operator: Uses right side if left side is falsy (None, "", 0, False)
        # self._api_key: Leading underscore indicates "internal" attribute
        self._api_key: str = api_key or settings.pinecone_api_key
        self._index_name: str = index_name or settings.pinecone_index_name
        self._dimension: int = dimension or settings.pinecone_dimension
        self._metric: str = metric or settings.pinecone_metric
        self._cloud: str = cloud or settings.pinecone_cloud
        self._region: str = region or settings.pinecone_region
        self._namespace: str = namespace or settings.pinecone_namespace
        
        if not self._api_key:
            raise VectorStoreValidationError(
                "Pinecone API key is required. "
                "Set PINECONE_API_KEY environment variable or pass api_key parameter.",
                operation="init"
            )
        
        logger.info(
            f"Initializing PineconeVectorStore: "
            f"index={self._index_name}, namespace={self._namespace}, dimension={self._dimension}"
        )
        
        try:
            # Create Pinecone client instance
            # self._pc: The Pinecone client object for API calls
            self._pc: Pinecone = Pinecone(api_key=self._api_key)
            
        except Exception as e:
            # Wrap any connection errors in our custom exception type
            raise VectorStoreConnectionError(
                f"Failed to initialize Pinecone client: {e}",
                operation="init",
                details={"error_type": type(e).__name__}
            )
        
        if create_if_not_exists:
            # Get list of existing index names using list comprehension
            # [idx.name for idx in ...] creates a list of just the names
            existing_indexes: List[str] = [
                idx.name for idx in self._pc.list_indexes()
            ]
            
            # Check if our index exists
            if self._index_name not in existing_indexes:
                # Log index creation
                logger.info(
                    f"Creating new Pinecone index: {self._index_name} "
                    f"(dimension={self._dimension}, metric={self._metric})"
                )
                
                try:
                    # Create the index with serverless spec
                    self._pc.create_index(
                        name=self._index_name,
                        dimension=self._dimension,
                        metric=self._metric,
                        spec=ServerlessSpec(
                            cloud=self._cloud,
                            region=self._region
                        )
                    )
                    
                    logger.info(f"Successfully created index: {self._index_name}")
                    
                except Exception as e:
                    raise VectorStoreConnectionError(
                        f"Failed to create Pinecone index: {e}",
                        operation="init",
                        details={
                            "index_name": self._index_name,
                            "dimension": self._dimension
                        }
                    )
        
        try:
            # Get an Index object for operations (upsert, query, etc.)
            # self._index: The Index object used for all vector operations
            self._index = self._pc.Index(self._index_name)
            
            logger.info(f"Connected to Pinecone index: {self._index_name}")
            
        except Exception as e:
            raise VectorStoreConnectionError(
                f"Failed to connect to Pinecone index: {e}",
                operation="init",
                details={"index_name": self._index_name}
            )
    
    # ========================================================================
    # PROPERTIES - Controlled access to internal state
    # ========================================================================
    
    @property
    def index_name(self) -> str:
        """
        Get the name of the Pinecone index.
        
        WHAT IS @property?
        ------------------
        - A decorator that makes a method accessible like an attribute
        - store.index_name instead of store.index_name()
        - Allows adding logic to attribute access (validation, logging, etc.)
        
        Returns:
            The index name as a string
        """
        return self._index_name
    
    @property
    def namespace(self) -> str:
        """Get the current namespace."""
        return self._namespace
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension for this index."""
        return self._dimension
    
    # ========================================================================
    # CRUD OPERATIONS - Core data manipulation methods
    # ========================================================================
    
    @with_retry(config=PINECONE_RETRY_CONFIG)
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add texts with their embeddings to the Pinecone index.
        
        This implements the abstract add() method from BaseVectorStore.
        
        PINECONE SPECIFICS:
        -------------------
        - Pinecone stores vectors with metadata
        - We store the text content in metadata under "content" key
        - Vectors are batched in groups of 100 (Pinecone recommendation)
        - upsert() will update if ID exists, insert if new
        
        Args:
            texts: List of text strings to store
            embeddings: Corresponding embedding vectors
            ids: Optional custom IDs (generated if not provided)
            metadata: Optional metadata for each text
        
        Returns:
            List of IDs for the added/updated vectors
        
        Raises:
            VectorStoreValidationError: If input validation fails
            VectorStoreConnectionError: If Pinecone API call fails
        
        Example:
            ids = store.add(
                texts=["Memory one", "Memory two"],
                embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
                metadata=[{"user": "alice"}, {"user": "alice"}]
            )
        """
        
        # --------------------------------------------------------------------
        # STEP 1: Validate inputs using inherited method from BaseVectorStore
        # --------------------------------------------------------------------
        
        # This method is defined in base.py and checks:
        # - Non-empty inputs
        # - Matching lengths
        # - Consistent embedding dimensions
        self.validate_inputs(texts, embeddings, ids, metadata)
        
        # Additional validation: check embedding dimension matches index
        is_valid, error = self.validate_embeddings(
            embeddings, 
            expected_dimension=self._dimension
        )
        if not is_valid:
            raise VectorStoreValidationError(error, operation="add")
        
        # --------------------------------------------------------------------
        # STEP 2: Generate IDs if not provided
        # --------------------------------------------------------------------
        
        # List comprehension: [expression for variable in iterable]
        # str(uuid.uuid4()) generates a random UUID string
        # _ is used when we don't need the loop variable
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # --------------------------------------------------------------------
        # STEP 3: Initialize metadata if not provided
        # --------------------------------------------------------------------
        
        if metadata is None:
            # Create list of empty dicts, one per text
            metadata = [{} for _ in texts]
        
        # --------------------------------------------------------------------
        # STEP 4: Prepare vectors for Pinecone upsert
        # --------------------------------------------------------------------
        
        # Build list of vector objects for Pinecone
        vectors: List[Dict[str, Any]] = []
        
        # zip(): Combines multiple iterables element-by-element
        # zip([1,2], [a,b], [x,y]) -> [(1,a,x), (2,b,y)]
        # enumerate(): Adds index to each element
        # enumerate(zip(...)) -> [(0, (1,a,x)), (1, (2,b,y))]
        for i, (text, embedding, vec_id, meta) in enumerate(
            zip(texts, embeddings, ids, metadata)
        ):
            # Create a copy of metadata and add the text content
            # {**meta, "content": text} = dict unpacking + new key
            # This creates a new dict with all of meta's keys plus "content"
            meta_with_content: Dict[str, Any] = {
                **meta,           # Unpack all existing metadata
                "content": text   # Add text as "content" field
            }
            
            # Build the vector object in Pinecone's expected format
            vectors.append({
                "id": vec_id,              # Unique identifier
                "values": embedding,       # The embedding vector
                "metadata": meta_with_content  # Metadata including content
            })
        
        # --------------------------------------------------------------------
        # STEP 5: Upsert vectors in batches
        # --------------------------------------------------------------------
        
        # Log the operation
        logger.info(f"Adding {len(vectors)} vectors to namespace '{self._namespace}'")
        
        # Batch upsert: Pinecone recommends batches of 100 for performance
        # range(start, stop, step): 0, 100, 200, ...
        for i in range(0, len(vectors), PINECONE_BATCH_SIZE):
            # Slice the vectors list: [i:i+100]
            batch: List[Dict[str, Any]] = vectors[i:i + PINECONE_BATCH_SIZE]
            
            # Calculate batch number for logging
            batch_num = (i // PINECONE_BATCH_SIZE) + 1
            total_batches = (len(vectors) + PINECONE_BATCH_SIZE - 1) // PINECONE_BATCH_SIZE
            
            logger.debug(f"Upserting batch {batch_num}/{total_batches} ({len(batch)} vectors)")
            
            # Call Pinecone upsert API
            # upsert = update if exists, insert if new
            self._index.upsert(
                vectors=batch,
                namespace=self._namespace
            )
        
        logger.info(f"Successfully added {len(vectors)} vectors")
        
        # Return the list of IDs
        return ids
    
    @with_retry(config=PINECONE_RETRY_CONFIG)
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors using cosine similarity.
        
        This implements the abstract search() method from BaseVectorStore.
        
        PINECONE SPECIFICS:
        -------------------
        - Pinecone uses a specific filter syntax with operators like $eq, $and
        - We convert simple dict filters to Pinecone format
        - Results include similarity scores (0-1 for cosine)
        
        Args:
            query_embedding: The embedding vector to search for
            top_k: Number of results to return
            filters: Optional metadata filters
                     Example: {"user_id": "alice", "type": "note"}
        
        Returns:
            List of SearchResult objects sorted by similarity (highest first)
        
        Raises:
            VectorStoreValidationError: If query embedding dimension is wrong
            VectorStoreConnectionError: If Pinecone API call fails
        
        Example:
            results = store.search(
                query_embedding=[0.1, 0.2, ...],
                top_k=5,
                filters={"user_id": "alice"}
            )
            for r in results:
                print(f"Score: {r.score}, Content: {r.content}")
        """
        
        # --------------------------------------------------------------------
        # STEP 1: Validate query embedding dimension
        # --------------------------------------------------------------------
        
        if len(query_embedding) != self._dimension:
            raise VectorStoreValidationError(
                f"Query embedding dimension {len(query_embedding)} "
                f"doesn't match index dimension {self._dimension}",
                operation="search",
                details={
                    "query_dimension": len(query_embedding),
                    "index_dimension": self._dimension
                }
            )
        
        # --------------------------------------------------------------------
        # STEP 2: Build Pinecone filter from dict
        # --------------------------------------------------------------------
        
        # Convert simple filters to Pinecone format
        pinecone_filter: Optional[Dict[str, Any]] = self._build_filter(filters)
        
        # --------------------------------------------------------------------
        # STEP 3: Query Pinecone
        # --------------------------------------------------------------------
        
        logger.debug(
            f"Searching namespace '{self._namespace}' "
            f"(top_k={top_k}, filter={pinecone_filter})"
        )
        
        # Call Pinecone query API
        results = self._index.query(
            vector=query_embedding,      # The query vector
            top_k=top_k,                 # Number of results
            include_metadata=True,       # Include metadata in results
            namespace=self._namespace,   # Search within this namespace
            filter=pinecone_filter       # Optional filters
        )
        
        # --------------------------------------------------------------------
        # STEP 4: Convert Pinecone results to SearchResult objects
        # --------------------------------------------------------------------
        
        search_results: List[SearchResult] = []
        
        # results.matches: List of match objects from Pinecone
        for match in results.matches:
            # Get metadata (or empty dict if None)
            # 'or {}' provides default if match.metadata is None
            metadata: Dict[str, Any] = match.metadata or {}
            
            # Extract and remove content from metadata
            # pop(key, default): Removes key from dict and returns value
            # If key doesn't exist, returns default instead of raising error
            content: str = metadata.pop("content", "")
            
            # Create SearchResult object (defined in base.py)
            search_results.append(SearchResult(
                id=match.id,           # Vector ID
                content=content,       # The original text
                score=match.score,     # Similarity score (0-1 for cosine)
                metadata=metadata      # Remaining metadata
            ))
        
        logger.debug(f"Found {len(search_results)} results")
        
        return search_results
    
    @with_retry(config=PINECONE_RETRY_CONFIG)
    def update(
        self,
        id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing vector in the Pinecone index.
        
        This implements the abstract update() method from BaseVectorStore.
        
        PINECONE SPECIFICS:
        -------------------
        - Pinecone doesn't have a native "update" operation
        - We implement update by: fetch existing -> merge changes -> upsert
        - This is a common pattern called "read-modify-write"
        
        Args:
            id: The ID of the vector to update
            text: New text content (optional)
            embedding: New embedding vector (optional)
            metadata: Metadata to merge with existing (optional)
        
        Returns:
            True if update succeeded, False if vector not found
        
        Raises:
            VectorStoreValidationError: If new embedding dimension is wrong
            VectorStoreConnectionError: If Pinecone API call fails
        
        Example:
            success = store.update(
                id="vec-123",
                text="Updated content",
                embedding=[0.5, 0.6, ...],
                metadata={"edited": True}
            )
        """
        
        # --------------------------------------------------------------------
        # STEP 1: Validate new embedding dimension if provided
        # --------------------------------------------------------------------
        
        if embedding is not None and len(embedding) != self._dimension:
            raise VectorStoreValidationError(
                f"Embedding dimension {len(embedding)} "
                f"doesn't match index dimension {self._dimension}",
                operation="update"
            )
        
        # --------------------------------------------------------------------
        # STEP 2: Fetch existing vector
        # --------------------------------------------------------------------
        
        logger.debug(f"Updating vector {id} in namespace '{self._namespace}'")
        
        # fetch(): Retrieve vectors by ID
        fetch_result = self._index.fetch(
            ids=[id],                    # List of IDs to fetch
            namespace=self._namespace    # Namespace to search in
        )
        
        # Check if vector exists
        # fetch_result.vectors: Dict[str, VectorData]
        if id not in fetch_result.vectors:
            logger.warning(f"Vector {id} not found for update")
            return False
        
        # Get the existing vector data
        existing = fetch_result.vectors[id]
        
        # --------------------------------------------------------------------
        # STEP 3: Prepare updated values
        # --------------------------------------------------------------------
        
        # Use new embedding if provided, otherwise keep existing
        # Conditional expression: value_if_true if condition else value_if_false
        new_values: List[float] = (
            embedding if embedding is not None 
            else existing.values
        )
        
        # Start with copy of existing metadata
        # .copy() creates shallow copy to avoid mutating original
        new_metadata: Dict[str, Any] = (
            existing.metadata.copy() if existing.metadata 
            else {}
        )
        
        # Update content if new text provided
        if text is not None:
            new_metadata["content"] = text
        
        # Merge new metadata if provided
        # .update() adds/updates keys from the given dict
        if metadata is not None:
            new_metadata.update(metadata)
        
        # --------------------------------------------------------------------
        # STEP 4: Upsert the updated vector
        # --------------------------------------------------------------------
        
        self._index.upsert(
            vectors=[{
                "id": id,
                "values": new_values,
                "metadata": new_metadata
            }],
            namespace=self._namespace
        )
        
        logger.info(f"Successfully updated vector {id}")
        
        return True
    
    @with_retry(config=PINECONE_RETRY_CONFIG)
    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors from the Pinecone index by their IDs.
        
        This implements the abstract delete() method from BaseVectorStore.
        
        Args:
            ids: List of vector IDs to delete
        
        Returns:
            True if deletion succeeded (even if some IDs didn't exist)
        
        Raises:
            VectorStoreConnectionError: If Pinecone API call fails
        
        Example:
            store.delete(ids=["vec-001", "vec-002"])
        """
        
        # Validate input
        if not ids:
            logger.warning("delete() called with empty IDs list")
            return True  # Nothing to delete is technically a success
        
        logger.info(f"Deleting {len(ids)} vectors from namespace '{self._namespace}'")
        
        # Call Pinecone delete API
        self._index.delete(
            ids=ids,
            namespace=self._namespace
        )
        
        logger.info(f"Successfully deleted {len(ids)} vectors")
        
        return True
    
    @with_retry(config=PINECONE_RETRY_CONFIG)
    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve vectors from the Pinecone index by their IDs.
        
        This implements the abstract get() method from BaseVectorStore.
        
        Args:
            ids: List of vector IDs to retrieve
        
        Returns:
            List of document dicts, each containing:
            - id: The vector ID
            - content: The text content
            - metadata: Additional metadata
            - embedding: The embedding vector
        
        Raises:
            VectorStoreConnectionError: If Pinecone API call fails
        
        Example:
            docs = store.get(ids=["vec-001", "vec-002"])
            for doc in docs:
                print(f"{doc['id']}: {doc['content']}")
        """
        
        # Validate input
        if not ids:
            logger.warning("get() called with empty IDs list")
            return []
        
        logger.debug(f"Fetching {len(ids)} vectors from namespace '{self._namespace}'")
        
        # Call Pinecone fetch API
        fetch_result = self._index.fetch(
            ids=ids,
            namespace=self._namespace
        )
        
        # Convert Pinecone results to our format
        documents: List[Dict[str, Any]] = []
        
        # .items() returns (key, value) pairs from dict
        for vec_id, vector_data in fetch_result.vectors.items():
            # Get metadata or empty dict
            metadata: Dict[str, Any] = vector_data.metadata or {}
            
            # Extract content from metadata
            content: str = metadata.pop("content", "")
            
            # Build document dict
            documents.append({
                "id": vec_id,
                "content": content,
                "metadata": metadata,
                "embedding": vector_data.values
            })
        
        logger.debug(f"Retrieved {len(documents)} documents")
        
        return documents
    
    # ========================================================================
    # MANAGEMENT OPERATIONS
    # ========================================================================
    
    @with_retry(config=PINECONE_RETRY_CONFIG)
    def health_check(self) -> bool:
        """
        Check if the Pinecone connection is healthy.
        
        This implements the abstract health_check() method from BaseVectorStore.
        
        Use for:
        - Kubernetes readiness probes
        - Health monitoring dashboards
        - Connection validation at startup
        
        Returns:
            True if connection is healthy, False otherwise
        
        Example:
            if not store.health_check():
                send_alert("Vector store is down!")
        """
        
        try:
            # Call describe_index_stats as a lightweight health check
            # This verifies both client connection and index accessibility
            stats = self._index.describe_index_stats()
            
            # If we got stats without error, connection is healthy
            logger.debug(f"Health check passed: {stats.total_vector_count} vectors")
            return True
            
        except Exception as e:
            # Log the failure
            logger.warning(f"Health check failed: {e}")
            return False
    
    @with_retry(config=PINECONE_RETRY_CONFIG)
    def get_stats(self) -> IndexStats:
        """
        Get statistics about the Pinecone index.
        
        This implements the abstract get_stats() method from BaseVectorStore.
        
        Returns:
            IndexStats object with vector count, dimension, etc.
        
        Example:
            stats = store.get_stats()
            print(f"Total vectors: {stats.total_vector_count}")
            print(f"Vectors in default namespace: {stats.namespaces.get('default', 0)}")
        """
        
        # Get index statistics from Pinecone
        stats = self._index.describe_index_stats()
        
        # Build namespace counts dict
        # stats.namespaces is Dict[str, NamespaceStats]
        namespace_counts: Dict[str, int] = {}
        
        if stats.namespaces:
            for ns_name, ns_stats in stats.namespaces.items():
                # ns_stats has vector_count attribute
                namespace_counts[ns_name] = ns_stats.get("vector_count", 0)
        
        # Return our IndexStats dataclass
        return IndexStats(
            total_vector_count=stats.total_vector_count,
            dimension=self._dimension,
            namespaces=namespace_counts,
            fullness_percentage=stats.index_fullness if hasattr(stats, 'index_fullness') else None
        )
    
    # ========================================================================
    # PINECONE-SPECIFIC METHODS (Internal helpers)
    # ========================================================================
    
    @with_retry(config=PINECONE_RETRY_CONFIG)
    def count(self) -> int:
        """
        Get the number of vectors in the current namespace.
        
        Note: This is Pinecone-specific (not in BaseVectorStore interface).
        Other backends may have different counting mechanisms.
        
        Returns:
            Approximate count of vectors in the namespace
        
        Example:
            print(f"Namespace has {store.count()} vectors")
        """
        
        # Get index statistics
        stats = self._index.describe_index_stats()
        
        # If using a namespace, get count for that namespace
        if self._namespace:
            # .get() with default handles missing namespace
            namespace_stats = stats.namespaces.get(self._namespace, {})
            return namespace_stats.get("vector_count", 0)
        
        # Otherwise return total count
        return stats.total_vector_count
    
    @with_retry(config=PINECONE_RETRY_CONFIG)
    def clear(self) -> bool:
        """
        Clear all vectors from the current namespace.
        
        WARNING: This is destructive and cannot be undone!
        
        Returns:
            True if clear succeeded
        
        Example:
            if confirm_deletion():
                store.clear()
        """
        
        logger.warning(f"Clearing all vectors from namespace '{self._namespace}'")
        
        # Pinecone requires delete_all=True for clearing namespace
        self._index.delete(
            delete_all=True,
            namespace=self._namespace
        )
        
        logger.info(f"Successfully cleared namespace '{self._namespace}'")
        
        return True
    
    def delete_index(self) -> None:
        """
        Delete the entire Pinecone index.
        
        WARNING: This is extremely destructive!
        - Deletes ALL data in ALL namespaces
        - Cannot be undone
        - Requires re-creating index to use again
        
        Use only for cleanup/teardown scenarios.
        
        Example:
            # In test cleanup:
            store.delete_index()
        """
        
        logger.warning(f"DELETING ENTIRE INDEX: {self._index_name}")
        
        # Delete the index via the Pinecone client
        self._pc.delete_index(self._index_name)
        
        logger.info(f"Successfully deleted index: {self._index_name}")
    
    # ========================================================================
    # METADATA-ONLY SEARCH (BaseVectorStore interface)
    # ========================================================================

    @with_retry(config=PINECONE_RETRY_CONFIG)
    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Query Pinecone using only metadata filters.

        This implements the abstract search_by_metadata() method from BaseVectorStore.

        Pinecone always requires a vector in the query call, so we supply a
        zero-vector and rely entirely on the metadata filter for selection.
        The similarity scores in the results are NOT meaningful — treat every
        returned record as a 1.0 match.

        Args:
            filters: Metadata key-value pairs, e.g.
                     {"main_content": "work_company", "user_id": "u1"}
            top_k: Max results (default 10).

        Returns:
            List[SearchResult] — matched records.
        """
        if not filters:
            logger.warning("search_by_metadata called with empty filters")
            return []

        pinecone_filter = self._build_filter(filters)

        # Zero-vector: Pinecone requires a vector but we only care about filter
        dummy_vector = [0.0] * self._dimension

        logger.debug(
            "Metadata search namespace='%s' filter=%s top_k=%d",
            self._namespace, pinecone_filter, top_k,
        )

        results = self._index.query(
            vector=dummy_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self._namespace,
            filter=pinecone_filter,
        )

        search_results: List[SearchResult] = []
        for match in results.matches:
            metadata: Dict[str, Any] = match.metadata or {}
            content: str = metadata.pop("content", "")
            search_results.append(SearchResult(
                id=match.id,
                content=content,
                score=1.0,      # score is meaningless for metadata-only queries
                metadata=metadata,
            ))

        logger.debug("Metadata search returned %d results", len(search_results))
        return search_results

    # ========================================================================
    # TEXT SEARCH (BaseVectorStore interface)
    # ========================================================================

    async def search_by_text(
        self,
        query_text: str,
        top_k: int = 2,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Embed a query string and search for similar vectors.

        This implements the abstract search_by_text() method from BaseVectorStore.

        This is a convenience wrapper that:
        1. Generates an embedding for `query_text` using SentenceTransformer.
        2. Delegates to the existing `search()` method.

        The Judge agent uses this for the *summary* domain to find existing
        similar records before deciding ADD / UPDATE / NOOP.

        Args:
            query_text: The text to embed and search for.
            top_k: Number of results to return.
            filters: Optional metadata filters (e.g. user_id, domain).

        Returns:
            List[SearchResult] — matched records sorted by similarity.
        """
        from src.pipelines.ingest import embed_text

        query_embedding = embed_text(query_text)
        return self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _build_filter(
        self, 
        filters: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Convert simple filter dict to Pinecone filter format.
        
        WHY IS THIS METHOD NAMED WITH UNDERSCORE?
        ------------------------------------------
        Leading underscore (_build_filter) is a Python convention meaning:
        "This is an internal/private method, don't call it from outside the class"
        
        It's not enforced by Python (you CAN still call it), but it signals intent.
        
        PINECONE FILTER FORMAT:
        -----------------------
        Simple filter: {"field": {"$eq": "value"}}
        Multiple filters: {"$and": [{"f1": {"$eq": "v1"}}, {"f2": {"$eq": "v2"}}]}
        
        This method converts simple dict {"key": "value"} to Pinecone format.
        
        Args:
            filters: Simple dict like {"user_id": "alice", "type": "note"}
        
        Returns:
            Pinecone-formatted filter dict, or None if no filters
        
        Example:
            self._build_filter({"user_id": "alice"})
            # Returns: {"user_id": {"$eq": "alice"}}
            
            self._build_filter({"user_id": "alice", "type": "note"})
            # Returns: {"$and": [
            #     {"user_id": {"$eq": "alice"}},
            #     {"type": {"$eq": "note"}}
            # ]}
        """
        
        # Return None if no filters provided
        if not filters:
            return None
        
        # Single filter: convert directly
        if len(filters) == 1:
            # Get the single key-value pair
            # Use next(iter(...)) to avoid creating a full list of items
            key, value = next(iter(filters.items()))
            
            # Return Pinecone format
            return {key: {"$eq": value}}
        
        # Multiple filters: combine with $and
        # List comprehension builds list of filter conditions
        filter_conditions: List[Dict[str, Any]] = [
            {k: {"$eq": v}} for k, v in filters.items()
        ]
        
        return {"$and": filter_conditions}
    
    # ========================================================================
    # CONTEXT MANAGER SUPPORT (Optional)
    # ========================================================================
    
    def __enter__(self) -> "PineconeVectorStore":
        """
        Support for 'with' statement (context manager).
        
        WHAT IS A CONTEXT MANAGER?
        --------------------------
        Allows using the class with 'with' statement for automatic cleanup:
        
            with PineconeVectorStore() as store:
                store.add(...)
            # Automatic cleanup when exiting 'with' block
        
        __enter__ is called when entering the 'with' block.
        
        Returns:
            self (the store instance)
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context manager.
        
        Called when exiting the 'with' block (even if exception occurred).
        
        Args:
            exc_type: Exception type if one occurred, else None
            exc_val: Exception value if one occurred, else None
            exc_tb: Exception traceback if one occurred, else None
        
        Note:
            Pinecone client doesn't require explicit cleanup,
            but this method is here for consistency and future use.
        """
        # Log if there was an exception
        if exc_type is not None:
            logger.error(f"Error in context manager: {exc_type.__name__}: {exc_val}")
        
        # Pinecone doesn't need explicit cleanup
        # If it did, we'd call cleanup methods here
        pass
    
    # ========================================================================
    # STRING REPRESENTATION
    # ========================================================================
    
    def __repr__(self) -> str:
        """
        Return string representation for debugging.
        
        WHAT IS __repr__?
        -----------------
        Called when you do repr(object) or in debugger.
        Should return a string that could recreate the object.
        
        Returns:
            String like "PineconeVectorStore(index='my-index', namespace='default')"
        """
        return (
            f"PineconeVectorStore("
            f"index='{self._index_name}', "
            f"namespace='{self._namespace}', "
            f"dimension={self._dimension})"
        )
    
    def __str__(self) -> str:
        """
        Return user-friendly string representation.
        
        WHAT IS __str__?
        ----------------
        Called when you do str(object) or print(object).
        Should return human-readable description.
        
        Returns:
            Friendly string like "Pinecone: my-index (default)"
        """
        return f"Pinecone: {self._index_name} ({self._namespace})"
