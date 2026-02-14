"""
================================================================================
LOGGING CONFIGURATION - Centralized Logging Setup
================================================================================

LOG LEVELS (from least to most severe):
---------------------------------------
    DEBUG    - Detailed information for debugging
    INFO     - General operational messages
    WARNING  - Something unexpected but not critical
    ERROR    - Something failed but app continues
    CRITICAL - App may not be able to continue

USAGE:
------
    # In any module:
    from src.config import get_logger
    
    logger = get_logger(__name__)
    
    logger.debug("Detailed debug info")
    logger.info("Operation completed successfully")
    logger.warning("Something unusual happened")
    logger.error("Operation failed", exc_info=True)
    logger.critical("Application cannot continue")

STRUCTURED LOGGING:
-------------------
    # Include context in log messages
    logger.info(
        "Processing request",
        extra={
            "user_id": user_id,
            "request_id": request_id,
            "operation": "add_memory"
        }
    )

================================================================================
"""

import logging

from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import sys
import os
from pathlib import Path
from typing import Optional
from enum import Enum
from dataclasses import dataclass, field


class LogLevel(str, Enum):
    """
    Log level enumeration for type-safe log level configuration.
    
    Inheriting from str makes these usable as strings:
        LogLevel.INFO == "INFO"  # True
    """
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogConfig:
    """
    Configuration for logging behavior.
    
    ATTRIBUTES:
    -----------
    level : LogLevel
        Minimum log level to output (default: INFO)
    
    format : str
        Log message format string
        Available fields: %(asctime)s, %(name)s, %(levelname)s, %(message)s, etc.
    
    date_format : str
        Date/time format for %(asctime)s
    
    enable_console : bool
        Output logs to console/stdout (default: True)
    
    enable_file : bool
        Output logs to file (default: False)
    
    log_file : Optional[str]
        Path to log file (default: logs/xmem.log)
    
    max_file_size : int
        Maximum log file size in bytes before rotation (default: 10MB)
    
    backup_count : int
        Number of backup files to keep (default: 5)
    
    enable_json : bool
        Use JSON format for structured logging (default: False)
    """
    
    level: LogLevel = LogLevel.INFO
    
    # Standard format: timestamp - logger name - level - message
    format: str = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
    
    # ISO 8601 date format
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Console logging (stdout)
    enable_console: bool = True
    
    # File logging
    enable_file: bool = False
    log_file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    
    # JSON structured logging (useful for log aggregation)
    enable_json: bool = False


# Default configuration
DEFAULT_LOG_CONFIG = LogConfig()
class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console output.
    
    Colors make it easier to visually scan logs:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red background
    
    Note: Colors only work in terminals that support ANSI codes.
    """
    
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"  # Reset to default
    
    def __init__(self, fmt: str, datefmt: str):
        """Initialize with format strings."""
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors.
        
        Args:
            record: The log record to format
        
        Returns:
            Formatted string with ANSI color codes
        """
        # Get the base formatted message
        message = super().format(record)
        
        # Add color based on log level
        color = self.COLORS.get(record.levelname, "")
        if color:
            # Wrap the entire message in color codes
            return f"{color}{message}{self.RESET}"
        
        return message


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs logs as JSON objects.
    
    Useful for:
    - Log aggregation systems (ELK, Splunk, CloudWatch)
    - Structured log analysis
    - Machine parsing
    
    Output format:
    {
        "timestamp": "2024-01-15T10:30:00",
        "level": "INFO",
        "logger": "src.storage.pinecone",
        "message": "Added 100 vectors",
        "extra": { ... }
    }
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.
        
        Args:
            record: The log record to format
        
        Returns:
            JSON string representation
        """
        import json
        from datetime import datetime
        
        # Build log entry dict
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add any extra fields
        # Extra fields are added via: logger.info("msg", extra={"key": "value"})
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message"
            ):
                log_entry[key] = value
        
        return json.dumps(log_entry)

def setup_logging(
    config: Optional[LogConfig] = None,
    level: Optional[LogLevel] = None,
    enable_console: Optional[bool] = None,
    enable_file: Optional[bool] = None,
    log_file: Optional[str] = None,
    enable_json: Optional[bool] = None,
) -> logging.Logger:
    """
    Set up logging for the entire application.
    
    Call this once at application startup to configure all loggers.
    
    Args:
        config: LogConfig instance (overrides individual params)
        level: Log level override
        enable_console: Console output override
        enable_file: File output override
        log_file: Log file path override
        enable_json: JSON format override
    
    Returns:
        Root logger for the application
    
    USAGE:
    ------
        # At application startup (e.g., main.py):
        from src.config.logging import setup_logging, LogLevel
        
        # Basic setup
        setup_logging()
        
        # Custom setup
        setup_logging(
            level=LogLevel.DEBUG,
            enable_file=True,
            log_file="logs/xmem.log"
        )
        
        # Using config object
        from src.config.logging import LogConfig
        config = LogConfig(level=LogLevel.DEBUG, enable_file=True)
        setup_logging(config=config)
    """
    
    # Resolve configuration
    if config is not None:
        effective_config = config
    else:
        # Build config from params with defaults
        effective_config = LogConfig(
            level=level if level is not None else DEFAULT_LOG_CONFIG.level,
            enable_console=enable_console if enable_console is not None else DEFAULT_LOG_CONFIG.enable_console,
            enable_file=enable_file if enable_file is not None else DEFAULT_LOG_CONFIG.enable_file,
            log_file=log_file if log_file is not None else DEFAULT_LOG_CONFIG.log_file,
            enable_json=enable_json if enable_json is not None else DEFAULT_LOG_CONFIG.enable_json,
        )
    
    # Get the root logger for our application
    # Using "src" as the root means all "src.*" loggers inherit this config
    root_logger = logging.getLogger("src")
    
    # Set the log level
    root_logger.setLevel(effective_config.level.value)
    
    # Remove existing handlers (prevent duplicate logs on re-configuration)
    root_logger.handlers.clear()
    
    if effective_config.enable_console:
        # StreamHandler outputs to stdout (console)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(effective_config.level.value)
        
        # Use appropriate formatter
        if effective_config.enable_json:
            formatter = JSONFormatter()
        else:
            # Use colored formatter for console
            formatter = ColoredFormatter(
                fmt=effective_config.format,
                datefmt=effective_config.date_format
            )
        
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    if effective_config.enable_file:
        # Determine log file path
        log_file_path = effective_config.log_file or "logs/xmem.log"
        
        # Create log directory if it doesn't exist
        log_dir = Path(log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # RotatingFileHandler: Rotates log files when they reach max size
        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=effective_config.max_file_size,
            backupCount=effective_config.backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(effective_config.level.value)
        
        # Use appropriate formatter (JSON for files is common for log aggregation)
        if effective_config.enable_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                fmt=effective_config.format,
                datefmt=effective_config.date_format
            )
        
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log that logging is configured
    root_logger.debug(
        f"Logging configured: level={effective_config.level.value}, "
        f"console={effective_config.enable_console}, "
        f"file={effective_config.enable_file}"
    )
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.
    
    This is the primary way to get a logger in your modules.
    
    Args:
        name: Logger name (typically __name__ for module name)
    
    Returns:
        Logger instance
    
    USAGE:
    ------
        # At the top of each module:
        from src.config import get_logger
        
        logger = get_logger(__name__)
        
        # Then use throughout the module:
        logger.info("Operation completed")
        logger.error("Something failed", exc_info=True)
    """
    return logging.getLogger(name)

def set_log_level(level: LogLevel) -> None:
    """
    Change the log level at runtime.
    
    Useful for debugging in production without restart.
    
    Args:
        level: New log level
    
    Usage:
        from src.config.logging import set_log_level, LogLevel
        
        # Enable debug logging temporarily
        set_log_level(LogLevel.DEBUG)
        
        # Back to normal
        set_log_level(LogLevel.INFO)
    """
    root_logger = logging.getLogger("src")
    root_logger.setLevel(level.value)
    
    # Update all handlers
    for handler in root_logger.handlers:
        handler.setLevel(level.value)
    
    root_logger.info(f"Log level changed to {level.value}")


def disable_logging() -> None:
    """
    Disable all logging (useful for tests).
    
    Usage:
        from src.config.logging import disable_logging
        
        disable_logging()  # No more log output
    """
    logging.getLogger("src").disabled = True


def enable_logging() -> None:
    """
    Re-enable logging after disabling.
    
    Usage:
        from src.config.logging import enable_logging
        
        enable_logging()  # Logs work again
    """
    logging.getLogger("src").disabled = False
