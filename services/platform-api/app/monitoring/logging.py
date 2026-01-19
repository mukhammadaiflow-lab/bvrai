"""Structured logging for observability."""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from contextvars import ContextVar
import asyncio
import logging
import json
import sys
import traceback
import uuid

# Context variable for request correlation
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
_request_context: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogRecord:
    """Structured log record."""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    correlation_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "logger": self.logger_name,
        }
        if self.correlation_id:
            data["correlation_id"] = self.correlation_id
        if self.context:
            data["context"] = self.context
        if self.exception:
            data["exception"] = self.exception
        if self.stack_trace:
            data["stack_trace"] = self.stack_trace
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Outputs logs as JSON for easy parsing by log aggregators.
    """

    def __init__(
        self,
        include_stack_trace: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.include_stack_trace = include_stack_trace
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Build base log entry
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID if present
        correlation_id = _correlation_id.get()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        # Add request context if present
        request_ctx = _request_context.get()
        if request_ctx:
            log_entry["request"] = request_ctx

        # Add extra fields from record
        if hasattr(record, "extra"):
            log_entry["context"] = record.extra

        # Add any extra fields from formatter config
        log_entry.update(self.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
            }
            if self.include_stack_trace:
                log_entry["stack_trace"] = self.formatException(record.exc_info)

        # Add location info
        log_entry["location"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Console formatter with colors and structured context.

    Provides human-readable output for development.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console."""
        # Build timestamp
        timestamp = datetime.utcfromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Get level with optional color
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            level = f"{color}{level:8}{self.RESET}"
        else:
            level = f"{level:8}"

        # Build base message
        message = f"{timestamp} | {level} | {record.name} | {record.getMessage()}"

        # Add correlation ID if present
        correlation_id = _correlation_id.get()
        if correlation_id:
            message = f"{message} | cid={correlation_id[:8]}"

        # Add extra context if present
        if hasattr(record, "extra") and record.extra:
            ctx_str = " ".join(f"{k}={v}" for k, v in record.extra.items())
            message = f"{message} | {ctx_str}"

        # Add exception if present
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        return message


class ContextLogger:
    """
    Logger with automatic context injection.

    Usage:
        logger = ContextLogger(__name__)
        logger.info("Processing request", user_id="123", action="create")
    """

    def __init__(
        self,
        name: str,
        default_context: Optional[Dict[str, Any]] = None,
    ):
        self._logger = logging.getLogger(name)
        self._default_context = default_context or {}

    def _log(
        self,
        level: int,
        message: str,
        exc_info: bool = False,
        **context,
    ) -> None:
        """Internal logging method."""
        # Merge contexts
        merged_context = {
            **self._default_context,
            **context,
        }

        # Create log record with extra context
        extra = {"extra": merged_context}
        self._logger.log(level, message, exc_info=exc_info, extra=extra)

    def debug(self, message: str, **context) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **context)

    def info(self, message: str, **context) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **context)

    def warning(self, message: str, **context) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **context)

    def error(self, message: str, exc_info: bool = False, **context) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, exc_info=exc_info, **context)

    def critical(self, message: str, exc_info: bool = False, **context) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, exc_info=exc_info, **context)

    def exception(self, message: str, **context) -> None:
        """Log exception with traceback."""
        self._log(logging.ERROR, message, exc_info=True, **context)

    def bind(self, **context) -> "ContextLogger":
        """Create a new logger with additional context."""
        new_context = {**self._default_context, **context}
        return ContextLogger(self._logger.name, new_context)


def get_logger(name: str) -> ContextLogger:
    """Get a context logger."""
    return ContextLogger(name)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set the correlation ID for the current context."""
    cid = correlation_id or str(uuid.uuid4())
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _correlation_id.get()


def set_request_context(context: Dict[str, Any]) -> None:
    """Set request context for logging."""
    _request_context.set(context)


def clear_request_context() -> None:
    """Clear request context."""
    _request_context.set({})


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (for production)
        extra_fields: Additional fields to include in all logs
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Set formatter
    if json_format:
        formatter = JSONFormatter(extra_fields=extra_fields)
    else:
        formatter = ConsoleFormatter()

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


class LogBuffer:
    """
    Buffer for collecting logs.

    Useful for capturing logs during tests or for batch processing.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._records: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def add(self, record: Dict[str, Any]) -> None:
        """Add a log record."""
        async with self._lock:
            self._records.append(record)
            if len(self._records) > self.max_size:
                self._records = self._records[-self.max_size:]

    def get_records(
        self,
        level: Optional[str] = None,
        logger: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get buffered records with optional filtering."""
        records = self._records
        if level:
            records = [r for r in records if r.get("level") == level.lower()]
        if logger:
            records = [r for r in records if r.get("logger") == logger]
        return records

    def clear(self) -> None:
        """Clear buffered records."""
        self._records = []


class BufferHandler(logging.Handler):
    """Handler that writes to a LogBuffer."""

    def __init__(self, buffer: LogBuffer):
        super().__init__()
        self.buffer = buffer
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record to the buffer."""
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "extra"):
            log_entry["context"] = record.extra

        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.format(record) if record.exc_info else None,
            }

        # Add to buffer (async-safe)
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self.buffer.add(log_entry))
        except RuntimeError:
            # No event loop running, use synchronous approach
            self.buffer._records.append(log_entry)


class AuditLogger:
    """
    Audit logger for security and compliance logging.

    Logs actions with actor, action, resource, and result information.
    """

    def __init__(self, logger_name: str = "audit"):
        self._logger = get_logger(logger_name)

    def log(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        actor_id: Optional[str] = None,
        actor_type: str = "user",
        result: str = "success",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an audit event."""
        self._logger.info(
            f"Audit: {action} {resource_type}",
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            actor_id=actor_id,
            actor_type=actor_type,
            result=result,
            **(details or {}),
        )

    def user_action(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        result: str = "success",
        **details,
    ) -> None:
        """Log a user action."""
        self.log(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            actor_id=user_id,
            actor_type="user",
            result=result,
            details=details,
        )

    def system_action(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        result: str = "success",
        **details,
    ) -> None:
        """Log a system action."""
        self.log(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            actor_id="system",
            actor_type="system",
            result=result,
            details=details,
        )


class RequestLogger:
    """
    Logger for HTTP request/response logging.

    Automatically logs request start, completion, and errors.
    """

    def __init__(self, logger_name: str = "http"):
        self._logger = get_logger(logger_name)

    def log_request_start(
        self,
        method: str,
        path: str,
        correlation_id: str,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        """Log request start."""
        self._logger.info(
            f"Request started: {method} {path}",
            method=method,
            path=path,
            correlation_id=correlation_id,
            client_ip=client_ip,
            user_agent=user_agent,
        )

    def log_request_complete(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        correlation_id: str,
    ) -> None:
        """Log request completion."""
        level_method = self._logger.info
        if status_code >= 500:
            level_method = self._logger.error
        elif status_code >= 400:
            level_method = self._logger.warning

        level_method(
            f"Request completed: {method} {path} -> {status_code}",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=round(duration_ms, 2),
            correlation_id=correlation_id,
        )

    def log_request_error(
        self,
        method: str,
        path: str,
        error: Exception,
        correlation_id: str,
    ) -> None:
        """Log request error."""
        self._logger.error(
            f"Request error: {method} {path} - {str(error)}",
            method=method,
            path=path,
            error_type=type(error).__name__,
            error_message=str(error),
            correlation_id=correlation_id,
            exc_info=True,
        )


class CallLogger:
    """
    Logger for voice call events.

    Provides structured logging for call lifecycle events.
    """

    def __init__(self, logger_name: str = "calls"):
        self._logger = get_logger(logger_name)

    def log_call_started(
        self,
        call_id: str,
        agent_id: str,
        direction: str,
        phone_number: Optional[str] = None,
    ) -> None:
        """Log call started."""
        self._logger.info(
            f"Call started: {call_id}",
            call_id=call_id,
            agent_id=agent_id,
            direction=direction,
            phone_number=phone_number,
        )

    def log_call_ended(
        self,
        call_id: str,
        duration_seconds: float,
        end_reason: str,
        status: str,
    ) -> None:
        """Log call ended."""
        self._logger.info(
            f"Call ended: {call_id}",
            call_id=call_id,
            duration_seconds=round(duration_seconds, 2),
            end_reason=end_reason,
            status=status,
        )

    def log_speech_event(
        self,
        call_id: str,
        speaker: str,
        text: str,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Log speech event."""
        self._logger.debug(
            f"Speech: [{speaker}] {text[:100]}...",
            call_id=call_id,
            speaker=speaker,
            text=text,
            duration_ms=duration_ms,
        )

    def log_function_call(
        self,
        call_id: str,
        function_name: str,
        arguments: Dict[str, Any],
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log function call."""
        if error:
            self._logger.warning(
                f"Function call failed: {function_name}",
                call_id=call_id,
                function_name=function_name,
                arguments=arguments,
                error=error,
            )
        else:
            self._logger.debug(
                f"Function call: {function_name}",
                call_id=call_id,
                function_name=function_name,
                arguments=arguments,
                result=result,
            )


# Global loggers
_audit_logger: Optional[AuditLogger] = None
_request_logger: Optional[RequestLogger] = None
_call_logger: Optional[CallLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def get_request_logger() -> RequestLogger:
    """Get the request logger."""
    global _request_logger
    if _request_logger is None:
        _request_logger = RequestLogger()
    return _request_logger


def get_call_logger() -> CallLogger:
    """Get the call logger."""
    global _call_logger
    if _call_logger is None:
        _call_logger = CallLogger()
    return _call_logger
