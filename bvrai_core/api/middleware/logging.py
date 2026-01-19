"""
Request Logging Middleware

This module provides comprehensive request/response logging
for API monitoring and debugging.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
)

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class LogFormat(str, Enum):
    """Log output formats."""

    JSON = "json"
    TEXT = "text"
    COMBINED = "combined"  # Apache combined log format


@dataclass
class AccessLogEntry:
    """Access log entry."""

    # Request info
    request_id: str
    timestamp: datetime
    method: str
    path: str
    query_string: Optional[str] = None

    # Response info
    status_code: int = 0
    response_size: int = 0
    duration_ms: float = 0.0

    # Client info
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None

    # Auth info
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    api_key_id: Optional[str] = None

    # Additional context
    error: Optional[str] = None
    error_code: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "method": self.method,
            "path": self.path,
            "query_string": self.query_string,
            "status_code": self.status_code,
            "response_size": self.response_size,
            "duration_ms": round(self.duration_ms, 3),
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "api_key_id": self.api_key_id,
        }

        if self.error:
            data["error"] = self.error
            data["error_code"] = self.error_code

        if self.extra:
            data["extra"] = self.extra

        return json.dumps(data)

    def to_text(self) -> str:
        """Convert to text format."""
        parts = [
            f"[{self.timestamp.isoformat()}]",
            f"{self.method}",
            f"{self.path}",
            f"status={self.status_code}",
            f"duration={self.duration_ms:.1f}ms",
            f"size={self.response_size}",
        ]

        if self.client_ip:
            parts.append(f"ip={self.client_ip}")

        if self.user_id:
            parts.append(f"user={self.user_id}")

        if self.error:
            parts.append(f"error={self.error}")

        return " ".join(parts)

    def to_combined(self) -> str:
        """Convert to Apache combined log format."""
        # Format: %h %l %u %t "%r" %>s %b "%{Referer}i" "%{User-Agent}i"
        remote_host = self.client_ip or "-"
        remote_logname = "-"
        remote_user = self.user_id or "-"
        time_str = self.timestamp.strftime("[%d/%b/%Y:%H:%M:%S %z]")
        request_line = f"{self.method} {self.path} HTTP/1.1"
        status = self.status_code
        size = self.response_size or "-"
        referer = "-"
        user_agent = self.user_agent or "-"

        return (
            f'{remote_host} {remote_logname} {remote_user} {time_str} '
            f'"{request_line}" {status} {size} "{referer}" "{user_agent}"'
        )


class LogConfig(BaseModel):
    """Logging configuration."""

    enabled: bool = Field(default=True, description="Enable request logging")
    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    format: LogFormat = Field(default=LogFormat.JSON, description="Log format")

    # What to log
    log_request_headers: bool = Field(
        default=False,
        description="Log request headers",
    )
    log_request_body: bool = Field(
        default=False,
        description="Log request body",
    )
    log_response_body: bool = Field(
        default=False,
        description="Log response body",
    )

    # Size limits for body logging
    max_request_body_log_size: int = Field(
        default=10000,
        description="Max request body size to log",
    )
    max_response_body_log_size: int = Field(
        default=10000,
        description="Max response body size to log",
    )

    # Sensitive data handling
    mask_headers: List[str] = Field(
        default_factory=lambda: [
            "authorization",
            "x-api-key",
            "cookie",
            "set-cookie",
        ],
        description="Headers to mask",
    )
    mask_fields: List[str] = Field(
        default_factory=lambda: [
            "password",
            "secret",
            "token",
            "api_key",
            "credit_card",
            "ssn",
        ],
        description="JSON fields to mask",
    )

    # Paths to exclude
    exclude_paths: List[str] = Field(
        default_factory=lambda: [
            "/health",
            "/ready",
            "/metrics",
        ],
        description="Paths to exclude from logging",
    )

    # Slow request threshold
    slow_request_threshold_ms: float = Field(
        default=1000.0,
        description="Threshold for slow request warning",
    )


class RequestLogger:
    """
    Comprehensive request logger.

    Features:
    - Structured JSON logging
    - Request/response body logging (with size limits)
    - Sensitive data masking
    - Slow request detection
    - Custom log handlers
    """

    def __init__(
        self,
        config: Optional[LogConfig] = None,
        custom_handlers: Optional[List[Callable[[AccessLogEntry], None]]] = None,
    ):
        """
        Initialize logger.

        Args:
            config: Logging configuration
            custom_handlers: Additional log handlers
        """
        self.config = config or LogConfig()
        self.custom_handlers = custom_handlers or []

        # Set up Python logger
        self._logger = logging.getLogger("api.access")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def should_log(self, path: str) -> bool:
        """Check if request should be logged."""
        if not self.config.enabled:
            return False

        for exclude_path in self.config.exclude_paths:
            if path.startswith(exclude_path):
                return False

        return True

    def log_request(self, entry: AccessLogEntry) -> None:
        """Log a request."""
        if not self.config.enabled:
            return

        # Format log message
        if self.config.format == LogFormat.JSON:
            message = entry.to_json()
        elif self.config.format == LogFormat.COMBINED:
            message = entry.to_combined()
        else:
            message = entry.to_text()

        # Determine log level
        if entry.status_code >= 500:
            self._logger.error(message)
        elif entry.status_code >= 400:
            self._logger.warning(message)
        elif entry.duration_ms > self.config.slow_request_threshold_ms:
            self._logger.warning(f"SLOW REQUEST: {message}")
        else:
            self._logger.info(message)

        # Call custom handlers
        for handler in self.custom_handlers:
            try:
                handler(entry)
            except Exception as e:
                logger.error(f"Custom log handler error: {e}")

    def mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive fields in data."""
        if not data:
            return data

        masked = {}
        for key, value in data.items():
            key_lower = key.lower()

            # Check if field should be masked
            should_mask = any(
                mask_field in key_lower
                for mask_field in self.config.mask_fields
            )

            if should_mask:
                if isinstance(value, str):
                    masked[key] = "***MASKED***"
                else:
                    masked[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked[key] = self.mask_sensitive_data(value)
            elif isinstance(value, list):
                masked[key] = [
                    self.mask_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                masked[key] = value

        return masked

    def mask_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Mask sensitive headers."""
        masked = {}
        for key, value in headers.items():
            if key.lower() in self.config.mask_headers:
                # Show partial value for debugging
                if len(value) > 10:
                    masked[key] = f"{value[:4]}...{value[-4:]}"
                else:
                    masked[key] = "***"
            else:
                masked[key] = value
        return masked

    def add_handler(self, handler: Callable[[AccessLogEntry], None]) -> None:
        """Add a custom log handler."""
        self.custom_handlers.append(handler)


class RequestLogMiddleware:
    """
    FastAPI middleware for request logging.

    Usage:
        app = FastAPI()
        request_logger = RequestLogger(config)
        app.add_middleware(RequestLogMiddleware, logger=request_logger)
    """

    def __init__(
        self,
        app,
        logger: RequestLogger,
        get_auth_context: Optional[Callable] = None,
    ):
        """
        Initialize middleware.

        Args:
            app: FastAPI application
            logger: Request logger instance
            get_auth_context: Function to get auth context from request
        """
        self.app = app
        self.logger = logger
        self.get_auth_context = get_auth_context

    async def __call__(self, scope, receive, send):
        """Process request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract request info
        from starlette.requests import Request
        request = Request(scope, receive)

        path = request.url.path
        if not self.logger.should_log(path):
            await self.app(scope, receive, send)
            return

        # Start timing
        start_time = time.time()
        request_id = request.headers.get(
            "X-Request-ID",
            str(uuid.uuid4()),
        )

        # Create log entry
        entry = AccessLogEntry(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            method=request.method,
            path=path,
            query_string=str(request.query_params) if request.query_params else None,
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
        )

        # Get auth context if available
        if self.get_auth_context:
            try:
                auth = await self.get_auth_context(request)
                if auth:
                    entry.user_id = getattr(auth, "user_id", None)
                    entry.organization_id = getattr(auth, "organization_id", None)
                    entry.api_key_id = getattr(auth, "api_key_id", None)
            except Exception:
                pass

        # Capture response info
        response_status = 0
        response_size = 0
        response_body_chunks = []

        async def send_wrapper(message):
            nonlocal response_status, response_size

            if message["type"] == "http.response.start":
                response_status = message["status"]
                # Add request ID to response headers
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers

            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                response_size += len(body)

                if self.logger.config.log_response_body:
                    response_body_chunks.append(body)

            await send(message)

        # Process request
        error_message = None
        error_code = None

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            error_message = str(e)
            error_code = getattr(e, "code", None)
            if hasattr(error_code, "value"):
                error_code = error_code.value
            raise

        finally:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Update entry
            entry.status_code = response_status
            entry.response_size = response_size
            entry.duration_ms = duration_ms
            entry.error = error_message
            entry.error_code = error_code

            # Log response body if enabled
            if self.logger.config.log_response_body and response_body_chunks:
                full_body = b"".join(response_body_chunks)
                if len(full_body) <= self.logger.config.max_response_body_log_size:
                    try:
                        body_data = json.loads(full_body.decode())
                        entry.extra["response_body"] = self.logger.mask_sensitive_data(body_data)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        entry.extra["response_body_size"] = len(full_body)

            # Log the request
            self.logger.log_request(entry)


class AsyncLogBuffer:
    """
    Async buffer for batch log writing.

    Collects log entries and writes them in batches
    for better performance.
    """

    def __init__(
        self,
        writer: Callable[[List[AccessLogEntry]], None],
        batch_size: int = 100,
        flush_interval_seconds: float = 5.0,
    ):
        """
        Initialize buffer.

        Args:
            writer: Function to write batch of entries
            batch_size: Max entries before flush
            flush_interval_seconds: Time before auto-flush
        """
        self.writer = writer
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds

        self._buffer: List[AccessLogEntry] = []
        self._lock = None  # Initialized lazily
        self._flush_task: Optional[Any] = None

    async def _get_lock(self):
        """Get or create async lock."""
        if self._lock is None:
            import asyncio
            self._lock = asyncio.Lock()
        return self._lock

    async def add(self, entry: AccessLogEntry) -> None:
        """Add entry to buffer."""
        lock = await self._get_lock()
        async with lock:
            self._buffer.append(entry)

            if len(self._buffer) >= self.batch_size:
                await self._flush()

    async def _flush(self) -> None:
        """Flush buffer to writer."""
        if not self._buffer:
            return

        entries = self._buffer.copy()
        self._buffer.clear()

        try:
            self.writer(entries)
        except Exception as e:
            logger.error(f"Failed to write log batch: {e}")

    async def start_auto_flush(self) -> None:
        """Start automatic flush task."""
        import asyncio

        async def flush_loop():
            while True:
                await asyncio.sleep(self.flush_interval)
                lock = await self._get_lock()
                async with lock:
                    await self._flush()

        self._flush_task = asyncio.create_task(flush_loop())

    async def stop(self) -> None:
        """Stop auto-flush and flush remaining."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except Exception:
                pass

        lock = await self._get_lock()
        async with lock:
            await self._flush()


def create_request_logger(
    config: Optional[LogConfig] = None,
) -> RequestLogger:
    """Create a request logger instance."""
    return RequestLogger(config)


__all__ = [
    "LogLevel",
    "LogFormat",
    "AccessLogEntry",
    "LogConfig",
    "RequestLogger",
    "RequestLogMiddleware",
    "AsyncLogBuffer",
    "create_request_logger",
]
