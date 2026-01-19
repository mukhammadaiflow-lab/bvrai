"""
Request Tracking Middleware

This module provides request context tracking, correlation IDs,
and distributed tracing support.
"""

import asyncio
import contextvars
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
)

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# Context variable for current request
_current_request_context: contextvars.ContextVar[Optional["RequestContext"]] = (
    contextvars.ContextVar("request_context", default=None)
)


@dataclass
class SpanContext:
    """Distributed tracing span context."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None

    # Sampling
    sampled: bool = True

    # Baggage items (propagated across services)
    baggage: Dict[str, str] = field(default_factory=dict)

    def to_headers(self) -> Dict[str, str]:
        """Convert to trace headers (W3C Trace Context format)."""
        # traceparent header
        flags = "01" if self.sampled else "00"
        traceparent = f"00-{self.trace_id}-{self.span_id}-{flags}"

        headers = {"traceparent": traceparent}

        # tracestate header for baggage
        if self.baggage:
            tracestate = ",".join(f"{k}={v}" for k, v in self.baggage.items())
            headers["tracestate"] = tracestate

        return headers

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional["SpanContext"]:
        """Parse from trace headers."""
        traceparent = headers.get("traceparent")
        if not traceparent:
            return None

        try:
            parts = traceparent.split("-")
            if len(parts) != 4 or parts[0] != "00":
                return None

            trace_id = parts[1]
            span_id = parts[2]
            flags = parts[3]
            sampled = flags == "01"

            # Parse tracestate
            baggage = {}
            tracestate = headers.get("tracestate", "")
            if tracestate:
                for item in tracestate.split(","):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        baggage[k.strip()] = v.strip()

            return cls(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=None,
                sampled=sampled,
                baggage=baggage,
            )

        except Exception as e:
            logger.debug(f"Failed to parse traceparent: {e}")
            return None

    @classmethod
    def new(cls, parent: Optional["SpanContext"] = None) -> "SpanContext":
        """Create a new span context."""
        if parent:
            return cls(
                trace_id=parent.trace_id,
                span_id=secrets.token_hex(8),
                parent_span_id=parent.span_id,
                sampled=parent.sampled,
                baggage=dict(parent.baggage),
            )

        return cls(
            trace_id=secrets.token_hex(16),
            span_id=secrets.token_hex(8),
            parent_span_id=None,
            sampled=True,
        )


@dataclass
class RequestContext:
    """
    Context for a single request.

    Provides:
    - Request identification
    - Distributed tracing
    - Custom attributes
    - Timing information
    """

    # Identifiers
    request_id: str
    correlation_id: str

    # Timing
    start_time: float
    start_timestamp: datetime

    # Request info
    method: str = ""
    path: str = ""
    client_ip: Optional[str] = None

    # Auth info
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    api_key_id: Optional[str] = None

    # Tracing
    span_context: Optional[SpanContext] = None

    # Custom attributes
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Child spans
    spans: List["Span"] = field(default_factory=list)

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000

    @property
    def trace_id(self) -> Optional[str]:
        """Get trace ID if tracing is enabled."""
        return self.span_context.trace_id if self.span_context else None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a custom attribute."""
        self.attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get a custom attribute."""
        return self.attributes.get(key, default)

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "Span":
        """Start a child span."""
        span = Span(
            name=name,
            context=self,
            parent_span_id=self.span_context.span_id if self.span_context else None,
            attributes=attributes or {},
        )
        self.spans.append(span)
        return span

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "method": self.method,
            "path": self.path,
            "elapsed_ms": self.elapsed_ms,
            "attributes": self.attributes,
        }

        if self.trace_id:
            data["trace_id"] = self.trace_id

        if self.user_id:
            data["user_id"] = self.user_id

        if self.organization_id:
            data["organization_id"] = self.organization_id

        return data


@dataclass
class Span:
    """A span within a request for timing specific operations."""

    name: str
    context: RequestContext
    parent_span_id: Optional[str] = None

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Span ID
    span_id: str = field(default_factory=lambda: secrets.token_hex(8))

    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "OK"
    error: Optional[str] = None

    def __enter__(self) -> "Span":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if exc_val:
            self.status = "ERROR"
            self.error = str(exc_val)
        self.end()

    async def __aenter__(self) -> "Span":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if exc_val:
            self.status = "ERROR"
            self.error = str(exc_val)
        self.end()

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value

    def set_status(self, status: str, error: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        self.error = error

    def end(self) -> None:
        """End the span."""
        if self.end_time is None:
            self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


class RequestTracker:
    """
    Tracks request context throughout the request lifecycle.

    Features:
    - Request ID generation and propagation
    - Correlation ID tracking
    - Distributed tracing support
    - Custom attribute storage
    - Span timing
    """

    def __init__(
        self,
        enable_tracing: bool = True,
        sampling_rate: float = 1.0,
    ):
        """
        Initialize tracker.

        Args:
            enable_tracing: Enable distributed tracing
            sampling_rate: Sampling rate for traces (0.0 to 1.0)
        """
        self.enable_tracing = enable_tracing
        self.sampling_rate = sampling_rate

        # Callbacks
        self._on_request_start: List[Callable[[RequestContext], None]] = []
        self._on_request_end: List[Callable[[RequestContext], None]] = []
        self._on_span_end: List[Callable[[Span], None]] = []

    def create_context(
        self,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        incoming_headers: Optional[Dict[str, str]] = None,
        method: str = "",
        path: str = "",
        client_ip: Optional[str] = None,
    ) -> RequestContext:
        """
        Create a new request context.

        Args:
            request_id: Request ID (generated if not provided)
            correlation_id: Correlation ID (uses request_id if not provided)
            incoming_headers: Incoming request headers for trace propagation
            method: HTTP method
            path: Request path
            client_ip: Client IP address

        Returns:
            New request context
        """
        # Generate IDs
        request_id = request_id or str(uuid.uuid4())
        correlation_id = correlation_id or request_id

        # Handle tracing
        span_context = None
        if self.enable_tracing:
            # Try to extract parent context from headers
            parent_context = None
            if incoming_headers:
                parent_context = SpanContext.from_headers(incoming_headers)

            # Create span context
            if parent_context:
                span_context = SpanContext.new(parent_context)
            else:
                span_context = SpanContext.new()

            # Apply sampling
            if secrets.randbelow(100) / 100 > self.sampling_rate:
                span_context.sampled = False

        # Create context
        context = RequestContext(
            request_id=request_id,
            correlation_id=correlation_id,
            start_time=time.time(),
            start_timestamp=datetime.utcnow(),
            method=method,
            path=path,
            client_ip=client_ip,
            span_context=span_context,
        )

        # Set as current context
        _current_request_context.set(context)

        # Call callbacks
        for callback in self._on_request_start:
            try:
                callback(context)
            except Exception as e:
                logger.error(f"Request start callback error: {e}")

        return context

    def end_context(self, context: RequestContext) -> None:
        """
        End a request context.

        Args:
            context: Request context to end
        """
        # End any open spans
        for span in context.spans:
            if span.end_time is None:
                span.end()
            # Call span callbacks
            for callback in self._on_span_end:
                try:
                    callback(span)
                except Exception as e:
                    logger.error(f"Span end callback error: {e}")

        # Call callbacks
        for callback in self._on_request_end:
            try:
                callback(context)
            except Exception as e:
                logger.error(f"Request end callback error: {e}")

        # Clear current context
        _current_request_context.set(None)

    def on_request_start(self, callback: Callable[[RequestContext], None]) -> None:
        """Register request start callback."""
        self._on_request_start.append(callback)

    def on_request_end(self, callback: Callable[[RequestContext], None]) -> None:
        """Register request end callback."""
        self._on_request_end.append(callback)

    def on_span_end(self, callback: Callable[[Span], None]) -> None:
        """Register span end callback."""
        self._on_span_end.append(callback)


def get_current_context() -> Optional[RequestContext]:
    """Get the current request context."""
    return _current_request_context.get()


def get_request_id() -> Optional[str]:
    """Get the current request ID."""
    context = get_current_context()
    return context.request_id if context else None


def get_trace_id() -> Optional[str]:
    """Get the current trace ID."""
    context = get_current_context()
    return context.trace_id if context else None


def set_attribute(key: str, value: Any) -> None:
    """Set attribute on current context."""
    context = get_current_context()
    if context:
        context.set_attribute(key, value)


def start_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Optional[Span]:
    """Start a span in the current context."""
    context = get_current_context()
    if context:
        return context.start_span(name, attributes)
    return None


class RequestTrackingMiddleware:
    """
    FastAPI middleware for request tracking.

    Usage:
        app = FastAPI()
        tracker = RequestTracker()
        app.add_middleware(RequestTrackingMiddleware, tracker=tracker)
    """

    def __init__(
        self,
        app,
        tracker: RequestTracker,
        header_name: str = "X-Request-ID",
        correlation_header: str = "X-Correlation-ID",
    ):
        """
        Initialize middleware.

        Args:
            app: FastAPI application
            tracker: Request tracker instance
            header_name: Request ID header name
            correlation_header: Correlation ID header name
        """
        self.app = app
        self.tracker = tracker
        self.header_name = header_name
        self.correlation_header = correlation_header

    async def __call__(self, scope, receive, send):
        """Process request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract request info
        from starlette.requests import Request
        request = Request(scope, receive)

        # Get headers
        headers = dict(request.headers)
        request_id = headers.get(self.header_name.lower())
        correlation_id = headers.get(self.correlation_header.lower())

        # Create context
        context = self.tracker.create_context(
            request_id=request_id,
            correlation_id=correlation_id,
            incoming_headers=headers,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None,
        )

        # Add response headers
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))

                # Add request ID
                headers.append((
                    self.header_name.lower().encode(),
                    context.request_id.encode(),
                ))

                # Add correlation ID
                headers.append((
                    self.correlation_header.lower().encode(),
                    context.correlation_id.encode(),
                ))

                # Add trace headers if tracing enabled
                if context.span_context:
                    for key, value in context.span_context.to_headers().items():
                        headers.append((key.encode(), value.encode()))

                message["headers"] = headers

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self.tracker.end_context(context)


def create_request_tracker(
    enable_tracing: bool = True,
    sampling_rate: float = 1.0,
) -> RequestTracker:
    """Create a request tracker instance."""
    return RequestTracker(
        enable_tracing=enable_tracing,
        sampling_rate=sampling_rate,
    )


__all__ = [
    "SpanContext",
    "RequestContext",
    "Span",
    "RequestTracker",
    "RequestTrackingMiddleware",
    "get_current_context",
    "get_request_id",
    "get_trace_id",
    "set_attribute",
    "start_span",
    "create_request_tracker",
]
