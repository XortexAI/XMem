"""
================================================================================
CUSTOM EXCEPTIONS - Centralized Error Handling for XMem
================================================================================

EXCEPTION HIERARCHY:
--------------------
    XMemError (base for all XMem errors)
    │
    ├── ConfigurationError
    │   └── Missing API keys, invalid settings
    │
    ├── ValidationError
    │   └── Invalid input data, schema violations
    │
    ├── VectorStoreError
    │   ├── VectorStoreConnectionError
    │   ├── VectorStoreValidationError
    │   └── VectorNotFoundError
    │
    ├── DatabaseError
    │   └── DatabaseConnectionError
    │
    ├── LLMError
    │   ├── LLMRateLimitError
    │   └── LLMContextLengthError
    │
    └── EmbeddingError

"""

from typing import Dict, Any, Optional


# ============================================================================
# BASE EXCEPTION
# ============================================================================

class XMemError(Exception):
    """
    Base exception class for all XMem errors.
    
    All custom exceptions in XMem should inherit from this class.
    This allows catching all XMem-related errors with a single except clause.
    
    ATTRIBUTES:
    -----------
    message : str
        Human-readable error description
    operation : Optional[str]
        Name of the operation that failed (e.g., "add", "search", "connect")
    details : Dict[str, Any]
        Additional context for debugging (IDs, counts, parameters)
    
    USAGE:
    ------
        try:
            do_something()
        except XMemError as e:
            # Catches ALL XMem errors
            logger.error(f"XMem error in {e.operation}: {e}")
            logger.debug(f"Details: {e.details}")
    
    Example:
        raise XMemError(
            "Failed to process memory",
            operation="process_memory",
            details={"user_id": "123", "memory_count": 5}
        )
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the exception with context.
        
        Args:
            message: Human-readable description of what went wrong
            operation: Name of the operation that failed (for logging/debugging)
            details: Dictionary of additional context (IDs, counts, etc.)
        """
        # Call parent Exception.__init__ with the message
        # super().__init__(message) sets self.args = (message,)
        super().__init__(message)
        
        # Store the original message for access
        self.message: str = message
        
        # Store operation name (e.g., "add", "search", "delete")
        self.operation: Optional[str] = operation
        
        # Store additional details (e.g., {"ids": ["id1", "id2"]})
        # 'or {}' ensures details is never None, always a dict
        self.details: Dict[str, Any] = details or {}
    
    def __str__(self) -> str:
        """
        Return string representation of the exception.
        
        Called when you do str(exception) or print(exception).
        Includes operation name if available for better debugging.
        
        Returns:
            Formatted error message string
        """
        # Include operation context if available
        if self.operation:
            return f"[{self.operation}] {self.message}"
        return self.message
    
    def __repr__(self) -> str:
        """
        Return detailed representation for debugging.
        
        Called by repr(exception) and in debugger.
        Shows all attributes for complete debugging info.
        
        Returns:
            Detailed string representation
        """
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"operation={self.operation!r}, "
            f"details={self.details!r})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for JSON serialization.
        
        Useful for API error responses.
        
        Returns:
            Dictionary representation of the error
        
        Example:
            except XMemError as e:
                return jsonify(e.to_dict()), 500
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "operation": self.operation,
            "details": self.details,
        }


# ============================================================================
# CONFIGURATION ERRORS
# ============================================================================

class ConfigurationError(XMemError):
    """
    Raised when there's a configuration or settings error.
    
    Examples:
    - Missing required API key
    - Invalid configuration value
    - Incompatible settings combination
    
    Usage:
        if not settings.pinecone_api_key:
            raise ConfigurationError(
                "Pinecone API key is required",
                operation="init",
                details={"setting": "PINECONE_API_KEY"}
            )
    """
    pass  # Inherits everything from XMemError


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class ValidationError(XMemError):
    """
    Raised when input validation fails.
    
    Examples:
    - Invalid data format
    - Schema validation failure
    - Constraint violation
    
    Usage:
        if len(texts) != len(embeddings):
            raise ValidationError(
                f"Length mismatch: {len(texts)} texts vs {len(embeddings)} embeddings",
                operation="add",
                details={"texts_count": len(texts), "embeddings_count": len(embeddings)}
            )
    """
    pass


# ============================================================================
# VECTOR STORE ERRORS
# ============================================================================

class VectorStoreError(XMemError):
    """
    Base exception for all vector store related errors.
    
    Inherit from this for specific vector store errors.
    Allows catching all storage errors with one except clause.
    
    Usage:
        try:
            store.add(texts, embeddings)
        except VectorStoreError as e:
            # Handles connection, validation, and not-found errors
            logger.error(f"Storage operation failed: {e}")
    """
    pass


class VectorStoreConnectionError(VectorStoreError):
    """
    Raised when connection to the vector store fails.
    
    Examples:
    - Network timeout
    - Invalid credentials
    - Service unavailable
    - Rate limiting
    
    Usage:
        try:
            client = Pinecone(api_key=api_key)
        except Exception as e:
            raise VectorStoreConnectionError(
                f"Failed to connect to Pinecone: {e}",
                operation="init",
                details={"original_error": str(e)}
            )
    """
    pass


class VectorStoreValidationError(VectorStoreError):
    """
    Raised when vector store input validation fails.
    
    Examples:
    - Embedding dimension mismatch
    - Empty input lists
    - Invalid metadata format
    
    Usage:
        if len(embedding) != expected_dim:
            raise VectorStoreValidationError(
                f"Dimension mismatch: got {len(embedding)}, expected {expected_dim}",
                operation="search"
            )
    """
    pass


class VectorNotFoundError(VectorStoreError):
    """
    Raised when a requested vector ID doesn't exist.
    
    Examples:
    - Trying to update non-existent vector
    - Trying to get vector by invalid ID
    
    Usage:
        if id not in fetch_result.vectors:
            raise VectorNotFoundError(
                f"Vector with ID '{id}' not found",
                operation="update",
                details={"id": id}
            )
    """
    pass


# ============================================================================
# DATABASE ERRORS
# ============================================================================

class DatabaseError(XMemError):
    """
    Base exception for all database related errors.
    
    Use for MongoDB, Neo4j, or any other database operations.
    
    Usage:
        try:
            collection.insert_one(document)
        except PyMongoError as e:
            raise DatabaseError(
                f"Failed to insert document: {e}",
                operation="insert"
            )
    """
    pass


class DatabaseConnectionError(DatabaseError):
    """
    Raised when database connection fails.
    
    Examples:
    - Cannot connect to MongoDB
    - Neo4j authentication failed
    - Connection pool exhausted
    
    Usage:
        try:
            client = MongoClient(uri)
        except ConnectionFailure as e:
            raise DatabaseConnectionError(
                f"Failed to connect to MongoDB: {e}",
                operation="connect",
                details={"uri": uri}
            )
    """
    pass


# ============================================================================
# LLM ERRORS
# ============================================================================

class LLMError(XMemError):
    """
    Base exception for all LLM (Language Model) related errors.
    
    Use for OpenAI, Anthropic, Google, or any LLM API operations.
    
    Usage:
        try:
            response = client.chat.completions.create(...)
        except OpenAIError as e:
            raise LLMError(
                f"LLM call failed: {e}",
                operation="generate"
            )
    """
    pass


class LLMRateLimitError(LLMError):
    """
    Raised when LLM API rate limit is exceeded.
    
    This is often retryable after waiting.
    
    Usage:
        except RateLimitError as e:
            raise LLMRateLimitError(
                "Rate limit exceeded, please retry later",
                operation="generate",
                details={"retry_after": e.retry_after}
            )
    """
    pass


class LLMContextLengthError(LLMError):
    """
    Raised when input exceeds LLM's context length limit.
    
    Examples:
    - Prompt too long for model
    - Combined input + output exceeds limit
    
    Usage:
        if token_count > model_limit:
            raise LLMContextLengthError(
                f"Input exceeds context limit: {token_count} > {model_limit}",
                operation="generate",
                details={"token_count": token_count, "limit": model_limit}
            )
    """
    pass


# ============================================================================
# EMBEDDING ERRORS
# ============================================================================

class EmbeddingError(XMemError):
    """
    Raised when embedding generation fails.
    
    Examples:
    - Model loading failed
    - Text too long for embedding model
    - Encoding error
    
    Usage:
        try:
            embedding = model.encode(text)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embedding: {e}",
                operation="embed",
                details={"text_length": len(text)}
            )
    """
    pass
