"""
Logging configuration for the Bitcoin Volatility Trading System.

Provides a centralized logging setup with both console and file handlers.
Supports rotating file logs to prevent disk space issues.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Store configured loggers to avoid duplicate handlers
_loggers: dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "btc_volatility",
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up and configure a logger with console and optional file output.

    Args:
        name: Logger name (typically module name).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file. If None, only console logging is enabled.
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of backup log files to keep.
        console_output: Whether to output logs to console.

    Returns:
        Configured logging.Logger instance.

    Example:
        >>> logger = setup_logger("my_module", level="DEBUG", log_file="logs/app.log")
        >>> logger.info("Application started")
    """
    # Return existing logger if already configured
    if name in _loggers:
        return _loggers[name]

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    # Clear any existing handlers (in case of re-initialization)
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)

        # Create log directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Cache the logger
    _loggers[name] = logger

    return logger


def get_logger(name: str = "btc_volatility") -> logging.Logger:
    """
    Get an existing logger or create a basic one.

    Args:
        name: Logger name to retrieve.

    Returns:
        Logger instance. If not previously configured, returns a basic logger.

    Example:
        >>> logger = get_logger("my_module")
        >>> logger.info("Processing data...")
    """
    if name in _loggers:
        return _loggers[name]

    # Create a basic logger if not configured
    return setup_logger(name)


class LoggerMixin:
    """
    Mixin class that provides a logger property for classes.

    Usage:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("Doing something")
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        # Use class name as logger name
        logger_name = f"btc_volatility.{self.__class__.__name__}"

        if logger_name not in _loggers:
            # Create a child logger
            return logging.getLogger(logger_name)

        return _loggers[logger_name]


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance. If None, uses the default logger.

    Example:
        >>> @log_execution_time()
        ... def slow_function():
        ...     time.sleep(1)
    """
    import functools
    import time

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger()

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time

            logger.debug(f"{func.__name__} executed in {elapsed:.4f}s")
            return result

        return wrapper

    return decorator
