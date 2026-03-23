"""
================================================================================
RETRY UTILITIES - Exponential Backoff for Transient Failures
================================================================================

WHY RETRY LOGIC?
----------------
Network calls can fail temporarily due to:
- Network timeouts
- Rate limiting (429 errors)
- Service temporarily unavailable (503 errors)
- Connection resets

Instead of immediately failing, we retry with exponential backoff:
- 1st retry: wait 1 second
- 2nd retry: wait 2 seconds
- 3rd retry: wait 4 seconds

This gives the service time to recover without overwhelming it.

USAGE:
------
    from src.utils.retry import with_retry, RetryConfig
    
    # Basic usage with defaults
    @with_retry()
    def call_external_api():
        return requests.get("https://api.example.com")
    
    # Custom configuration
    @with_retry(max_retries=5, delay=0.5, backoff=2.0)
    def call_api_with_custom_retry():
        return api.call()
    
    # Using RetryConfig for reusable settings
    api_retry_config = RetryConfig(max_retries=3, delay=1.0)
    
    @with_retry(config=api_retry_config)
    def another_api_call():
        return api.call()

================================================================================
"""

from typing import TypeVar, Callable, Optional, Type, Tuple, Any

# functools: Higher-order functions
# wraps: Preserves function metadata when creating decorators
from functools import wraps
import time
import logging
from dataclasses import dataclass, field
from .exceptions import ValidationError

logger = logging.getLogger(__name__)
T = TypeVar("T")

@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.
    
    Using a dataclass allows:
    - Reusable configurations across multiple functions
    - Clear documentation of all options
    - Easy modification and testing
    
    ATTRIBUTES:
    -----------
    max_retries : int
        Maximum number of retry attempts (default: 3)
        Total attempts = max_retries + 1 (initial + retries)
    
    delay : float
        Initial delay between retries in seconds (default: 1.0)
        Actual delay = delay * (backoff_multiplier ^ attempt)
    
    backoff_multiplier : float
        Multiplier for exponential backoff (default: 2.0)
        delay=1, backoff=2: waits 1s, 2s, 4s, 8s...
    
    max_delay : float
        Maximum delay cap in seconds (default: 60.0)
        Prevents extremely long waits
    
    retryable_exceptions : Tuple[Type[Exception], ...]
        Exception types that should trigger retry
        Default: (Exception,) - retry all exceptions
        Example: (ConnectionError, TimeoutError)
    
    non_retryable_exceptions : Tuple[Type[Exception], ...]
        Exception types that should NOT be retried
        Default: (ValidationError,) - don't retry validation errors
    
    USAGE:
    ------
        # Create reusable config
        api_config = RetryConfig(
            max_retries=5,
            delay=0.5,
            retryable_exceptions=(ConnectionError, TimeoutError)
        )
        
        @with_retry(config=api_config)
        def call_api():
            ...
    """
    
    max_retries: int = 3
    delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 60.0
    retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )
    non_retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (ValidationError,)
    )

# Default configuration instance
DEFAULT_RETRY_CONFIG = RetryConfig()

def with_retry(
    max_retries: Optional[int] = None,
    delay: Optional[float] = None,
    backoff_multiplier: Optional[float] = None,
    max_delay: Optional[float] = None,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    non_retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    config: Optional[RetryConfig] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that adds retry logic with exponential backoff.
    
    WHAT IS A DECORATOR?
    --------------------
    A decorator wraps a function to add behavior. The @decorator syntax
    is equivalent to: function = decorator(function)
    
    WHAT IS EXPONENTIAL BACKOFF?
    ----------------------------
    Instead of retrying immediately, wait longer after each failure:
    - 1st retry: wait 1s (delay * 2^0)
    - 2nd retry: wait 2s (delay * 2^1)
    - 3rd retry: wait 4s (delay * 2^2)
    
    This prevents overwhelming a struggling service.
    
    DECORATOR FACTORY PATTERN:
    --------------------------
    This is a "decorator factory" - it returns a decorator.
    
        @with_retry(max_retries=3)  # Factory call returns decorator
        def my_func():              # Decorator wraps this function
            pass
    
    Args:
        max_retries: Maximum retry attempts (default: 3)
        delay: Initial delay in seconds (default: 1.0)
        backoff_multiplier: Backoff multiplier (default: 2.0)
        max_delay: Maximum delay cap (default: 60.0)
        retryable_exceptions: Exception types to retry
        non_retryable_exceptions: Exception types to NOT retry
        config: RetryConfig instance (overrides individual params)
    
    Returns:
        Decorator function that wraps the target function
    
    USAGE:
    ------
        # Basic usage
        @with_retry()
        def call_api():
            return requests.get(url)
        
        # Custom parameters
        @with_retry(max_retries=5, delay=0.5)
        def call_api_custom():
            return api.call()
        
        # With config object
        config = RetryConfig(max_retries=3)
        @with_retry(config=config)
        def call_api_with_config():
            return api.call()
    
    Example with full flow:
        @with_retry(max_retries=3, delay=1.0)
        def fetch_data():
            return external_api.get()
        
        # If fetch_data() fails:
        # - Attempt 1: fails → wait 1s
        # - Attempt 2: fails → wait 2s
        # - Attempt 3: fails → wait 4s
        # - Attempt 4: fails → raise exception
    """
    
    # ========================================================================
    # STEP 1: Resolve configuration
    # ========================================================================
    
    # Use provided config or build from individual params
    if config is not None:
        # Use the provided config object
        effective_config = config
    else:
        # Build config from individual params, falling back to defaults
        effective_config = RetryConfig(
            max_retries=max_retries if max_retries is not None else DEFAULT_RETRY_CONFIG.max_retries,
            delay=delay if delay is not None else DEFAULT_RETRY_CONFIG.delay,
            backoff_multiplier=backoff_multiplier if backoff_multiplier is not None else DEFAULT_RETRY_CONFIG.backoff_multiplier,
            max_delay=max_delay if max_delay is not None else DEFAULT_RETRY_CONFIG.max_delay,
            retryable_exceptions=retryable_exceptions if retryable_exceptions is not None else DEFAULT_RETRY_CONFIG.retryable_exceptions,
            non_retryable_exceptions=non_retryable_exceptions if non_retryable_exceptions is not None else DEFAULT_RETRY_CONFIG.non_retryable_exceptions,
        )
    
    # ========================================================================
    # STEP 2: Create the actual decorator
    # ========================================================================
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """
        The actual decorator that wraps the function.
        
        Args:
            func: The function to wrap with retry logic
        
        Returns:
            Wrapped function with retry behavior
        """
        
        # @wraps(func) preserves the original function's metadata
        # Without this, __name__, __doc__, etc. would be lost
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """
            Wrapper function that implements retry logic.
            
            *args: Positional arguments passed to the wrapped function
            **kwargs: Keyword arguments passed to the wrapped function
            
            Returns:
                Result from the wrapped function
            
            Raises:
                The last exception if all retries are exhausted
            """
            
            # Track the last exception for re-raising if all retries fail
            last_exception: Optional[Exception] = None
            
            # Total attempts = initial attempt + retries
            total_attempts = effective_config.max_retries + 1
            
            # ================================================================
            # RETRY LOOP
            # ================================================================
            
            for attempt in range(total_attempts):
                try:
                    # Call the original function with all its arguments
                    # If successful, return immediately
                    return func(*args, **kwargs)
                    
                except effective_config.non_retryable_exceptions as e:
                    # These exceptions should not be retried
                    # (e.g., validation errors won't succeed on retry)
                    logger.debug(
                        f"Non-retryable exception in {func.__name__}: "
                        f"{type(e).__name__}: {e}"
                    )
                    raise
                    
                except effective_config.retryable_exceptions as e:
                    # Store the exception in case we need to re-raise it
                    last_exception = e
                    
                    # Check if we have retries left
                    if attempt < effective_config.max_retries:
                        # Calculate delay with exponential backoff
                        # delay * (backoff ^ attempt) = 1, 2, 4, 8, ...
                        current_delay = effective_config.delay * (
                            effective_config.backoff_multiplier ** attempt
                        )
                        
                        # Cap the delay at max_delay
                        current_delay = min(current_delay, effective_config.max_delay)
                        
                        # Log the retry attempt
                        logger.warning(
                            f"Attempt {attempt + 1}/{total_attempts} failed for "
                            f"{func.__name__}: {type(e).__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        
                        # Wait before retrying
                        time.sleep(current_delay)
                        
                    else:
                        # No more retries - log the final failure
                        logger.error(
                            f"All {total_attempts} attempts failed for "
                            f"{func.__name__}: {type(e).__name__}: {e}"
                        )
            
            # ================================================================
            # ALL RETRIES EXHAUSTED
            # ================================================================
            
            # If we get here, all retries failed
            # Re-raise the last exception
            if last_exception is not None:
                raise last_exception
            
            # This should never happen, but satisfy type checker
            raise RuntimeError(f"Unexpected state in retry logic for {func.__name__}")
        
        # Return the wrapper function
        return wrapper
    
    # Return the decorator
    return decorator

def with_async_retry(
    max_retries: Optional[int] = None,
    delay: Optional[float] = None,
    backoff_multiplier: Optional[float] = None,
    max_delay: Optional[float] = None,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    non_retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    config: Optional[RetryConfig] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Async version of with_retry for async/await functions.
    
    Uses asyncio.sleep instead of time.sleep to avoid blocking.
    
    USAGE:
    ------
        @with_async_retry(max_retries=3)
        async def fetch_data():
            async with aiohttp.ClientSession() as session:
                return await session.get(url)
    """
    
    # Import asyncio here to avoid import if not using async
    import asyncio
    
    # Resolve configuration (same as sync version)
    if config is not None:
        effective_config = config
    else:
        effective_config = RetryConfig(
            max_retries=max_retries if max_retries is not None else DEFAULT_RETRY_CONFIG.max_retries,
            delay=delay if delay is not None else DEFAULT_RETRY_CONFIG.delay,
            backoff_multiplier=backoff_multiplier if backoff_multiplier is not None else DEFAULT_RETRY_CONFIG.backoff_multiplier,
            max_delay=max_delay if max_delay is not None else DEFAULT_RETRY_CONFIG.max_delay,
            retryable_exceptions=retryable_exceptions if retryable_exceptions is not None else DEFAULT_RETRY_CONFIG.retryable_exceptions,
            non_retryable_exceptions=non_retryable_exceptions if non_retryable_exceptions is not None else DEFAULT_RETRY_CONFIG.non_retryable_exceptions,
        )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """Async decorator wrapper."""
        
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            """Async wrapper with retry logic."""
            
            last_exception: Optional[Exception] = None
            total_attempts = effective_config.max_retries + 1
            
            for attempt in range(total_attempts):
                try:
                    # Await the async function
                    return await func(*args, **kwargs)
                    
                except effective_config.non_retryable_exceptions as e:
                    logger.debug(
                        f"Non-retryable exception in {func.__name__}: "
                        f"{type(e).__name__}: {e}"
                    )
                    raise
                    
                except effective_config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < effective_config.max_retries:
                        current_delay = effective_config.delay * (
                            effective_config.backoff_multiplier ** attempt
                        )
                        current_delay = min(current_delay, effective_config.max_delay)
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{total_attempts} failed for "
                            f"{func.__name__}: {type(e).__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        
                        # Use asyncio.sleep for non-blocking wait
                        await asyncio.sleep(current_delay)
                        
                    else:
                        logger.error(
                            f"All {total_attempts} attempts failed for "
                            f"{func.__name__}: {type(e).__name__}: {e}"
                        )
            
            if last_exception is not None:
                raise last_exception
            
            raise RuntimeError(f"Unexpected state in async retry logic for {func.__name__}")
        
        return wrapper
    
    return decorator
