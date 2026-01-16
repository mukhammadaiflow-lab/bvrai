"""
Distributed Tracing
===================

Comprehensive distributed tracing with spans, context propagation,
and integration with Jaeger and Zipkin.

Author: Platform Observability Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import secrets
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class SpanKind(str, Enum):
    """Kind of span"""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Span status"""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """
    Span context for propagation across process boundaries.

    Contains trace ID, span ID, and trace flags for distributed tracing.
    """

    trace_id: str
    span_id: str
    trace_flags: int = 1  # Sampled flag
    trace_state: Dict[str, str] = field(default_factory=dict)
    is_remote: bool = False

    @classmethod
    def generate(cls) -> "SpanContext":
        """Generate a new span context"""
        return cls(
            trace_id=secrets.token_hex(16),
            span_id=secrets.token_hex(8),
        )

    @classmethod
    def from_parent(cls, parent: "SpanContext") -> "SpanContext":
        """Create child context from parent"""
        return cls(
            trace_id=parent.trace_id,
            span_id=secrets.token_hex(8),
            trace_flags=parent.trace_flags,
            trace_state=dict(parent.trace_state),
        )

    @property
    def is_valid(self) -> bool:
        """Check if context is valid"""
        return bool(self.trace_id and self.span_id)

    @property
    def is_sampled(self) -> bool:
        """Check if trace is sampled"""
        return bool(self.trace_flags & 0x01)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "trace_flags": self.trace_flags,
            "trace_state": self.trace_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanContext":
        """Create from dictionary"""
        return cls(
            trace_id=data.get("trace_id", ""),
            span_id=data.get("span_id", ""),
            trace_flags=data.get("trace_flags", 1),
            trace_state=data.get("trace_state", {}),
            is_remote=True,
        )


@dataclass
class SpanEvent:
    """An event within a span"""

    name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """A link to another span"""

    context: SpanContext
    attributes: Dict[str, Any] = field(default_factory=dict)


class Span:
    """
    Represents a single operation within a trace.

    Usage:
        with tracer.start_span("process_request") as span:
            span.set_attribute("user_id", user_id)
            # Do work
            span.add_event("request_processed")
    """

    def __init__(
        self,
        name: str,
        context: SpanContext,
        parent: Optional[SpanContext] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[SpanLink]] = None,
    ):
        self.name = name
        self.context = context
        self.parent_context = parent
        self.kind = kind
        self._attributes: Dict[str, Any] = attributes or {}
        self._events: List[SpanEvent] = []
        self._links: List[SpanLink] = links or []
        self._status = SpanStatus.UNSET
        self._status_message: str = ""
        self._start_time = time.time()
        self._end_time: Optional[float] = None
        self._ended = False
        self._lock = threading.Lock()

    @property
    def trace_id(self) -> str:
        """Get trace ID"""
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        """Get span ID"""
        return self.context.span_id

    @property
    def parent_span_id(self) -> Optional[str]:
        """Get parent span ID"""
        return self.parent_context.span_id if self.parent_context else None

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds"""
        end = self._end_time or time.time()
        return (end - self._start_time) * 1000

    @property
    def is_recording(self) -> bool:
        """Check if span is recording"""
        return not self._ended and self.context.is_sampled

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set a single attribute"""
        if self.is_recording:
            with self._lock:
                self._attributes[key] = value
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "Span":
        """Set multiple attributes"""
        if self.is_recording:
            with self._lock:
                self._attributes.update(attributes)
        return self

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "Span":
        """Add an event to the span"""
        if self.is_recording:
            event = SpanEvent(
                name=name,
                timestamp=timestamp or datetime.utcnow(),
                attributes=attributes or {},
            )
            with self._lock:
                self._events.append(event)
        return self

    def add_link(
        self,
        context: SpanContext,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Span":
        """Add a link to another span"""
        if self.is_recording:
            link = SpanLink(
                context=context,
                attributes=attributes or {},
            )
            with self._lock:
                self._links.append(link)
        return self

    def set_status(
        self,
        status: SpanStatus,
        message: str = "",
    ) -> "Span":
        """Set span status"""
        if self.is_recording:
            with self._lock:
                self._status = status
                self._status_message = message
        return self

    def record_exception(
        self,
        exception: Exception,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Span":
        """Record an exception"""
        if self.is_recording:
            exc_attrs = {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                **(attributes or {}),
            }
            self.add_event("exception", exc_attrs)
            self.set_status(SpanStatus.ERROR, str(exception))
        return self

    def end(self, end_time: Optional[float] = None) -> None:
        """End the span"""
        with self._lock:
            if self._ended:
                return
            self._end_time = end_time or time.time()
            self._ended = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "duration_ms": self.duration_ms,
            "status": self._status.value,
            "status_message": self._status_message,
            "attributes": self._attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self._events
            ],
            "links": [
                {
                    "trace_id": l.context.trace_id,
                    "span_id": l.context.span_id,
                    "attributes": l.attributes,
                }
                for l in self._links
            ],
        }

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_val:
            self.record_exception(exc_val)
        self.end()


# Context variable for current span
_current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "current_span", default=None
)


def get_current_span() -> Optional[Span]:
    """Get the current active span"""
    return _current_span.get()


class TraceExporter(ABC):
    """Base class for trace exporters"""

    @abstractmethod
    async def export(self, spans: List[Span]) -> bool:
        """Export spans"""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the exporter"""
        pass


class JaegerExporter(TraceExporter):
    """
    Exports traces to Jaeger.
    """

    def __init__(
        self,
        agent_host: str = "localhost",
        agent_port: int = 6831,
        collector_endpoint: Optional[str] = None,
        service_name: str = "voice_ai",
    ):
        self._agent_host = agent_host
        self._agent_port = agent_port
        self._collector_endpoint = collector_endpoint
        self._service_name = service_name
        self._logger = structlog.get_logger("jaeger_exporter")

    async def export(self, spans: List[Span]) -> bool:
        """Export spans to Jaeger"""
        if not spans:
            return True

        try:
            jaeger_spans = [self._convert_span(span) for span in spans]

            if self._collector_endpoint:
                await self._send_http(jaeger_spans)
            else:
                await self._send_udp(jaeger_spans)

            return True
        except Exception as e:
            self._logger.error("jaeger_export_error", error=str(e))
            return False

    async def _send_http(self, spans: List[Dict[str, Any]]) -> None:
        """Send spans via HTTP"""
        # In a real implementation, this would use aiohttp
        self._logger.debug(
            "jaeger_http_send",
            endpoint=self._collector_endpoint,
            span_count=len(spans),
        )

    async def _send_udp(self, spans: List[Dict[str, Any]]) -> None:
        """Send spans via UDP (Thrift compact protocol)"""
        # In a real implementation, this would use Thrift
        self._logger.debug(
            "jaeger_udp_send",
            host=self._agent_host,
            port=self._agent_port,
            span_count=len(spans),
        )

    def _convert_span(self, span: Span) -> Dict[str, Any]:
        """Convert span to Jaeger format"""
        tags = [
            {"key": k, "type": "string", "value": str(v)}
            for k, v in span._attributes.items()
        ]

        logs = [
            {
                "timestamp": int(e.timestamp.timestamp() * 1000000),
                "fields": [
                    {"key": "event", "type": "string", "value": e.name},
                    *[
                        {"key": k, "type": "string", "value": str(v)}
                        for k, v in e.attributes.items()
                    ],
                ],
            }
            for e in span._events
        ]

        references = []
        if span.parent_context:
            references.append({
                "refType": "CHILD_OF",
                "traceIdHigh": span.parent_context.trace_id[:16],
                "traceIdLow": span.parent_context.trace_id[16:],
                "spanId": span.parent_context.span_id,
            })

        return {
            "traceIdHigh": span.trace_id[:16],
            "traceIdLow": span.trace_id[16:],
            "spanId": span.span_id,
            "operationName": span.name,
            "references": references,
            "flags": span.context.trace_flags,
            "startTime": int(span._start_time * 1000000),
            "duration": int(span.duration_ms * 1000),
            "tags": tags,
            "logs": logs,
        }

    async def shutdown(self) -> None:
        """Shutdown the exporter"""
        pass


class ZipkinExporter(TraceExporter):
    """
    Exports traces to Zipkin.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:9411/api/v2/spans",
        service_name: str = "voice_ai",
        local_endpoint: Optional[Dict[str, Any]] = None,
    ):
        self._endpoint = endpoint
        self._service_name = service_name
        self._local_endpoint = local_endpoint or {"serviceName": service_name}
        self._logger = structlog.get_logger("zipkin_exporter")

    async def export(self, spans: List[Span]) -> bool:
        """Export spans to Zipkin"""
        if not spans:
            return True

        try:
            zipkin_spans = [self._convert_span(span) for span in spans]

            # In a real implementation, this would use aiohttp
            self._logger.debug(
                "zipkin_send",
                endpoint=self._endpoint,
                span_count=len(zipkin_spans),
            )

            return True
        except Exception as e:
            self._logger.error("zipkin_export_error", error=str(e))
            return False

    def _convert_span(self, span: Span) -> Dict[str, Any]:
        """Convert span to Zipkin format"""
        zipkin_span: Dict[str, Any] = {
            "traceId": span.trace_id,
            "id": span.span_id,
            "name": span.name,
            "timestamp": int(span._start_time * 1000000),
            "duration": int(span.duration_ms * 1000),
            "localEndpoint": self._local_endpoint,
            "tags": {k: str(v) for k, v in span._attributes.items()},
        }

        if span.parent_context:
            zipkin_span["parentId"] = span.parent_context.span_id

        # Map span kind
        kind_map = {
            SpanKind.SERVER: "SERVER",
            SpanKind.CLIENT: "CLIENT",
            SpanKind.PRODUCER: "PRODUCER",
            SpanKind.CONSUMER: "CONSUMER",
        }
        if span.kind in kind_map:
            zipkin_span["kind"] = kind_map[span.kind]

        # Add annotations for events
        if span._events:
            zipkin_span["annotations"] = [
                {
                    "timestamp": int(e.timestamp.timestamp() * 1000000),
                    "value": e.name,
                }
                for e in span._events
            ]

        return zipkin_span

    async def shutdown(self) -> None:
        """Shutdown the exporter"""
        pass


class Tracer:
    """
    Creates and manages spans for distributed tracing.

    Usage:
        tracer = Tracer("my_service")

        with tracer.start_span("process_request") as span:
            span.set_attribute("user_id", user_id)
            result = process_request()
    """

    def __init__(
        self,
        name: str,
        exporter: Optional[TraceExporter] = None,
        sampler: Optional[Callable[[str], bool]] = None,
    ):
        self.name = name
        self._exporter = exporter
        self._sampler = sampler or (lambda _: True)
        self._logger = structlog.get_logger(f"tracer.{name}")
        self._pending_spans: List[Span] = []
        self._lock = threading.Lock()

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[SpanLink]] = None,
        parent: Optional[Union[Span, SpanContext]] = None,
    ) -> Iterator[Span]:
        """Start a new span as context manager"""
        span = self._create_span(name, kind, attributes, links, parent)
        token = _current_span.set(span)

        try:
            yield span
        finally:
            span.end()
            _current_span.reset(token)
            self._on_span_end(span)

    def start_as_current_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to wrap a function in a span"""

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                with self.start_span(name, kind, attributes):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def _create_span(
        self,
        name: str,
        kind: SpanKind,
        attributes: Optional[Dict[str, Any]],
        links: Optional[List[SpanLink]],
        parent: Optional[Union[Span, SpanContext]],
    ) -> Span:
        """Create a new span"""
        parent_context: Optional[SpanContext] = None

        if parent:
            if isinstance(parent, Span):
                parent_context = parent.context
            else:
                parent_context = parent
        else:
            # Check for current span
            current = get_current_span()
            if current:
                parent_context = current.context

        # Generate context
        if parent_context:
            context = SpanContext.from_parent(parent_context)
        else:
            context = SpanContext.generate()

        return Span(
            name=name,
            context=context,
            parent=parent_context,
            kind=kind,
            attributes=attributes,
            links=links,
        )

    def _on_span_end(self, span: Span) -> None:
        """Called when a span ends"""
        if not span.context.is_sampled:
            return

        with self._lock:
            self._pending_spans.append(span)

    async def flush(self) -> None:
        """Flush pending spans to exporter"""
        if not self._exporter:
            return

        with self._lock:
            spans = self._pending_spans
            self._pending_spans = []

        if spans:
            await self._exporter.export(spans)


class TracerProvider:
    """
    Provider for creating tracers.

    Usage:
        provider = TracerProvider()
        provider.set_exporter(JaegerExporter())
        tracer = provider.get_tracer("my_service")
    """

    def __init__(self):
        self._tracers: Dict[str, Tracer] = {}
        self._exporter: Optional[TraceExporter] = None
        self._sampler: Optional[Callable[[str], bool]] = None
        self._lock = threading.Lock()
        self._logger = structlog.get_logger("tracer_provider")

    def set_exporter(self, exporter: TraceExporter) -> None:
        """Set the trace exporter"""
        self._exporter = exporter

    def set_sampler(self, sampler: Callable[[str], bool]) -> None:
        """Set the sampler function"""
        self._sampler = sampler

    def get_tracer(self, name: str) -> Tracer:
        """Get or create a tracer"""
        with self._lock:
            if name not in self._tracers:
                self._tracers[name] = Tracer(
                    name=name,
                    exporter=self._exporter,
                    sampler=self._sampler,
                )
            return self._tracers[name]

    async def shutdown(self) -> None:
        """Shutdown all tracers"""
        for tracer in self._tracers.values():
            await tracer.flush()

        if self._exporter:
            await self._exporter.shutdown()


# =============================================================================
# CONTEXT PROPAGATION
# =============================================================================


def inject_context(
    context: SpanContext,
    carrier: Dict[str, str],
    format: str = "w3c",
) -> None:
    """
    Inject span context into a carrier for propagation.

    Supports W3C Trace Context and B3 formats.
    """
    if format == "w3c":
        # W3C Trace Context format
        carrier["traceparent"] = (
            f"00-{context.trace_id}-{context.span_id}-"
            f"{context.trace_flags:02x}"
        )
        if context.trace_state:
            state_str = ",".join(f"{k}={v}" for k, v in context.trace_state.items())
            carrier["tracestate"] = state_str

    elif format == "b3":
        # B3 single header format
        carrier["b3"] = (
            f"{context.trace_id}-{context.span_id}-"
            f"{'1' if context.is_sampled else '0'}"
        )

    elif format == "b3-multi":
        # B3 multi-header format
        carrier["X-B3-TraceId"] = context.trace_id
        carrier["X-B3-SpanId"] = context.span_id
        carrier["X-B3-Sampled"] = "1" if context.is_sampled else "0"


def extract_context(
    carrier: Dict[str, str],
    format: str = "w3c",
) -> Optional[SpanContext]:
    """
    Extract span context from a carrier.

    Supports W3C Trace Context and B3 formats.
    """
    if format == "w3c":
        traceparent = carrier.get("traceparent")
        if not traceparent:
            return None

        parts = traceparent.split("-")
        if len(parts) < 4:
            return None

        context = SpanContext(
            trace_id=parts[1],
            span_id=parts[2],
            trace_flags=int(parts[3], 16),
            is_remote=True,
        )

        # Parse tracestate
        tracestate = carrier.get("tracestate")
        if tracestate:
            for pair in tracestate.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    context.trace_state[k.strip()] = v.strip()

        return context

    elif format == "b3":
        b3_header = carrier.get("b3")
        if not b3_header:
            return None

        parts = b3_header.split("-")
        if len(parts) < 2:
            return None

        sampled = len(parts) > 2 and parts[2] == "1"

        return SpanContext(
            trace_id=parts[0],
            span_id=parts[1],
            trace_flags=1 if sampled else 0,
            is_remote=True,
        )

    elif format == "b3-multi":
        trace_id = carrier.get("X-B3-TraceId")
        span_id = carrier.get("X-B3-SpanId")

        if not trace_id or not span_id:
            return None

        sampled = carrier.get("X-B3-Sampled", "0") == "1"

        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=1 if sampled else 0,
            is_remote=True,
        )

    return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_default_provider: Optional[TracerProvider] = None
_provider_lock = threading.Lock()


def _get_provider() -> TracerProvider:
    """Get or create the default tracer provider"""
    global _default_provider

    with _provider_lock:
        if _default_provider is None:
            _default_provider = TracerProvider()
        return _default_provider


def trace(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to trace a function.

    Usage:
        @trace("process_request")
        def process_request(request):
            # ...
    """
    tracer = _get_provider().get_tracer("default")
    return tracer.start_as_current_span(name, kind, attributes)
