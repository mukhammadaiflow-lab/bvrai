"""
Standardized Logging Configuration

This module provides a consistent, structured logging setup for the entire platform.
Supports JSON logging for production and human-readable format for development.
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union
import uuid


# =============================================================================
# Configuration
# =============================================================================


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output formats."""

    JSON = "json"
    PRETTY = "pretty"
    SIMPLE = "simple"


# Default configuration
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_LOG_FORMAT = os.getenv("LOG_FORMAT", "json" if os.getenv("ENVIRONMENT") == "production" else "pretty")
SERVICE_NAME = os.getenv("SERVICE_NAME", "bvrai-platform")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


# =============================================================================
# Structured Log Record
# =============================================================================


class StructuredLogRecord:
    """
    Structured log record for consistent log output.

    Includes standard fields for correlation, tracing, and context.
    """

    def __init__(
        self,
        level: str,
        message: str,
        logger_name: str,
        **extra: Any,
    ):
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.level = level
        self.message = message
        self.logger = logger_name
        self.service = SERVICE_NAME
        self.environment = ENVIRONMENT

        # Standard context fields
        self.request_id: Optional[str] = extra.pop("request_id", None)
        self.correlation_id: Optional[str] = extra.pop("correlation_id", None)
        self.user_id: Optional[str] = extra.pop("user_id", None)
        self.organization_id: Optional[str] = extra.pop("organization_id", None)

        # Trace context
        self.trace_id: Optional[str] = extra.pop("trace_id", None)
        self.span_id: Optional[str] = extra.pop("span_id", None)

        # Error context
        self.error: Optional[Dict[str, Any]] = extra.pop("error", None)
        self.stack_trace: Optional[str] = extra.pop("stack_trace", None)

        # Additional context
        self.extra = extra

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "logger": self.logger,
            "service": self.service,
            "environment": self.environment,
        }

        # Add optional fields if present
        if self.request_id:
            result["request_id"] = self.request_id
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.user_id:
            result["user_id"] = self.user_id
        if self.organization_id:
            result["organization_id"] = self.organization_id
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.error:
            result["error"] = self.error
        if self.stack_trace:
            result["stack_trace"] = self.stack_trace
        if self.extra:
            result["context"] = self.extra

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


# =============================================================================
# Custom Formatters
# =============================================================================


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract extra fields from record
        extra = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message", "exc_info", "exc_text",
                "stack_info",
            }:
                extra[key] = value

        # Build structured record
        structured = StructuredLogRecord(
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
            **extra,
        )

        # Add exception info if present
        if record.exc_info:
            structured.stack_trace = "".join(traceback.format_exception(*record.exc_info))
            structured.error = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
            }

        return structured.to_json()


class PrettyFormatter(logging.Formatter):
    """Human-readable log formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",   # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, "")

        # Build the message
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = f"{color}{record.levelname:8}{self.RESET}"
        logger = f"\033[90m{record.name}\033[0m"

        message = f"{timestamp} | {level} | {logger} | {record.getMessage()}"

        # Add extra context
        extra_fields = []
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message", "exc_info", "exc_text",
                "stack_info",
            }:
                continue
            extra_fields.append(f"{key}={value}")

        if extra_fields:
            message += f" | {' '.join(extra_fields)}"

        # Add exception info
        if record.exc_info:
            message += f"\n{''.join(traceback.format_exception(*record.exc_info))}"

        return message


class SimpleFormatter(logging.Formatter):
    """Simple log formatter without colors."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record simply."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return f"{timestamp} [{record.levelname}] {record.name}: {record.getMessage()}"


# =============================================================================
# Logger Setup
# =============================================================================


def setup_logging(
    level: str = DEFAULT_LOG_LEVEL,
    format: str = DEFAULT_LOG_FORMAT,
    service_name: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format (json, pretty, simple)
        service_name: Service name for log entries
    """
    global SERVICE_NAME

    if service_name:
        SERVICE_NAME = service_name

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Select formatter
    if format == LogFormat.JSON or format == "json":
        formatter = JSONFormatter()
    elif format == LogFormat.PRETTY or format == "pretty":
        formatter = PrettyFormatter()
    else:
        formatter = SimpleFormatter()

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)

    # Log startup message
    root_logger.info(
        f"Logging configured",
        extra={
            "level": level,
            "format": format,
            "service": SERVICE_NAME,
        }
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# =============================================================================
# Context Logging
# =============================================================================


class LogContext:
    """
    Thread-local context for adding request-scoped data to logs.

    Usage:
        with LogContext(request_id="abc123", user_id="user1"):
            logger.info("Processing request")
    """

    _context: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any):
        self._kwargs = kwargs
        self._previous: Dict[str, Any] = {}

    def __enter__(self) -> "LogContext":
        # Store previous values and set new ones
        for key, value in self._kwargs.items():
            self._previous[key] = LogContext._context.get(key)
            LogContext._context[key] = value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Restore previous values
        for key, value in self._previous.items():
            if value is None:
                LogContext._context.pop(key, None)
            else:
                LogContext._context[key] = value

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a value from the current context."""
        return cls._context.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a value in the current context."""
        cls._context[key] = value

    @classmethod
    def clear(cls) -> None:
        """Clear the current context."""
        cls._context.clear()

    @classmethod
    def all(cls) -> Dict[str, Any]:
        """Get all context values."""
        return cls._context.copy()


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes context in log records.

    Usage:
        logger = ContextLogger(logging.getLogger(__name__))
        logger.info("Message")  # Automatically includes context
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message, adding context."""
        extra = kwargs.get("extra", {})

        # Add context values
        extra.update(LogContext.all())

        kwargs["extra"] = extra
        return msg, kwargs


def get_context_logger(name: str) -> ContextLogger:
    """
    Get a context-aware logger.

    Args:
        name: Logger name

    Returns:
        ContextLogger instance
    """
    return ContextLogger(logging.getLogger(name), {})


# =============================================================================
# Decorators
# =============================================================================


def log_function_call(
    level: str = "DEBUG",
    include_args: bool = True,
    include_result: bool = False,
):
    """
    Decorator to log function calls.

    Args:
        level: Log level for the messages
        include_args: Include function arguments in log
        include_result: Include function result in log
    """
    def decorator(func: Callable) -> Callable:
        logger = get_logger(func.__module__)
        log_func = getattr(logger, level.lower())

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            call_id = str(uuid.uuid4())[:8]

            # Log entry
            entry_msg = f"Calling {func.__name__}"
            entry_extra = {"call_id": call_id}
            if include_args:
                entry_extra["args"] = str(args)[:200]
                entry_extra["kwargs"] = str(kwargs)[:200]

            log_func(entry_msg, extra=entry_extra)

            try:
                result = await func(*args, **kwargs)

                # Log exit
                exit_msg = f"Completed {func.__name__}"
                exit_extra = {"call_id": call_id}
                if include_result:
                    exit_extra["result"] = str(result)[:200]

                log_func(exit_msg, extra=exit_extra)
                return result

            except Exception as e:
                logger.exception(
                    f"Failed {func.__name__}: {str(e)}",
                    extra={"call_id": call_id},
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            call_id = str(uuid.uuid4())[:8]

            # Log entry
            entry_msg = f"Calling {func.__name__}"
            entry_extra = {"call_id": call_id}
            if include_args:
                entry_extra["args"] = str(args)[:200]
                entry_extra["kwargs"] = str(kwargs)[:200]

            log_func(entry_msg, extra=entry_extra)

            try:
                result = func(*args, **kwargs)

                # Log exit
                exit_msg = f"Completed {func.__name__}"
                exit_extra = {"call_id": call_id}
                if include_result:
                    exit_extra["result"] = str(result)[:200]

                log_func(exit_msg, extra=exit_extra)
                return result

            except Exception as e:
                logger.exception(
                    f"Failed {func.__name__}: {str(e)}",
                    extra={"call_id": call_id},
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "LogLevel",
    "LogFormat",
    "StructuredLogRecord",
    "JSONFormatter",
    "PrettyFormatter",
    "SimpleFormatter",
    "setup_logging",
    "get_logger",
    "LogContext",
    "ContextLogger",
    "get_context_logger",
    "log_function_call",
]
