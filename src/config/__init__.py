"""
================================================================================
CONFIG PACKAGE - Centralized Configuration
================================================================================

This package provides all configuration-related functionality:
- Settings: Environment variables and configuration values
- Logging: Logging setup and utilities

USAGE:
------
    # Import settings singleton
    from src.config import settings
    
    # Access configuration values
    api_key = settings.pinecone_api_key
    model = settings.embedding_model
    
    # Import logging utilities
    from src.config import get_logger, setup_logging
    
    # Setup logging at app startup
    setup_logging()
    
    # Get logger in any module
    logger = get_logger(__name__)
    logger.info("Hello from my module")

================================================================================
"""

# Import Settings class and create singleton instance
from .settings import Settings

# Create the settings singleton
# This is instantiated once when the config package is first imported
# All subsequent imports get the same instance
settings = Settings()

# Import logging utilities
from .logging import (
    # Setup function (call once at app startup)
    setup_logging,
    
    # Get logger (call in each module)
    get_logger,
    
    # Runtime log level control
    set_log_level,
    disable_logging,
    enable_logging,
    
    # Configuration classes
    LogConfig,
    LogLevel,
)

__all__ = [
    # Settings
    "Settings",
    "settings",
    
    # Logging
    "setup_logging",
    "get_logger",
    "set_log_level",
    "disable_logging",
    "enable_logging",
    "LogConfig",
    "LogLevel",
]
