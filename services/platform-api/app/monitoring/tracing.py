"""Distributed tracing for observability."""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from contextvars import ContextVar
import asyncio
import uuid
import time
import logging
import functools

logger = logging.getLogger(__name__)


class SpanKind(str, Enum):
    """Type of span."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Span execution status."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Context for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    sampled: bool = True

    def to_headers(self) -> Dict[str, str]:
        """Convert to propagation headers."""
        return {
            "X-Trace-ID": self.trace_id,
            "X-Span-ID": self.span_id,
            "X-Parent-Span-ID": self.parent_span_id or "",
            "X-Sampled": "1" if self.sampled else "0",
        }

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional["SpanContext"]:
        """Create from propagation headers."""
        trace_id = headers.get("X-Trace-ID") or headers.get("x-trace-id")
        span_id = headers.get("X-Span-ID") or headers.get("x-span-id")
        parent_span_id = headers.get("X-Parent-Span-ID") or headers.get("x-parent-span-id")
        sampled = headers.get("X-Sampled", "1") != "0"

        if trace_id:
            return cls(
                trace_id=trace_id,
                span_id=span_id or cls.generate_id(),
                parent_span_id=parent_span_id or None,
                sampled=sampled,
            )
        return None

    @staticmethod
    def generate_id() -> str:
        """Generate a new ID."""
        return uuid.uuid4().hex[:16]


@dataclass
class SpanEvent:
    """Event within a span."""
    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A single span in a trace."""
    context: SpanContext
    name: str
    kind: SpanKind
    start_time: datetime
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        self.events.append(SpanEvent(
            name=name,
            timestamp=datetime.utcnow(),
            attributes=attributes or {},
        ))

    def set_status(self, status: SpanStatus, error: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        if error:
            self.error = error

    def end(self) -> None:
        """End the span."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
            "error": self.error,
        }


# Context variable for current span
_current_span: ContextVar[Optional[Span]] = ContextVar("current_span", default=None)


class Tracer:
    """
    Tracer for distributed tracing.

    Usage:
        tracer = Tracer("api-service")

        # Create spans manually
        with tracer.start_span("process_request") as span:
            span.set_attribute("user_id", "123")
            do_work()

        # Use decorator
        @tracer.trace()
        async def my_function():
            ...
    """

    def __init__(
        self,
        service_name: str,
        sample_rate: float = 1.0,
        exporter: Optional["SpanExporter"] = None,
    ):
        self.service_name = service_name
        self.sample_rate = sample_rate
        self.exporter = exporter or ConsoleExporter()
        self._spans: List[Span] = []

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "SpanContextManager":
        """Start a new span."""
        # Get parent context
        if parent is None:
            current = _current_span.get()
            if current:
                parent = SpanContext(
                    trace_id=current.context.trace_id,
                    span_id=SpanContext.generate_id(),
                    parent_span_id=current.context.span_id,
                    sampled=current.context.sampled,
                )

        # Create new context if no parent
        if parent is None:
            import random
            sampled = random.random() < self.sample_rate
            parent = SpanContext(
                trace_id=SpanContext.generate_id() + SpanContext.generate_id(),
                span_id=SpanContext.generate_id(),
                sampled=sampled,
            )
        else:
            # Create child span
            parent = SpanContext(
                trace_id=parent.trace_id,
                span_id=SpanContext.generate_id(),
                parent_span_id=parent.span_id,
                sampled=parent.sampled,
            )

        # Create span
        span = Span(
            context=parent,
            name=name,
            kind=kind,
            start_time=datetime.utcnow(),
            attributes={
                "service.name": self.service_name,
                **(attributes or {}),
            },
        )

        return SpanContextManager(self, span)

    def _end_span(self, span: Span) -> None:
        """End a span and export it."""
        span.end()
        if span.context.sampled:
            self._spans.append(span)
            self.exporter.export(span)

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return _current_span.get()

    def trace(
        self,
        name: Optional[str] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Decorator for tracing functions.

        Usage:
            @tracer.trace()
            async def my_function():
                ...

            @tracer.trace("custom_name", kind=SpanKind.CLIENT)
            def sync_function():
                ...
        """
        def decorator(func: Callable) -> Callable:
            span_name = name or func.__name__

            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with self.start_span(span_name, kind=kind, attributes=attributes) as span:
                        span.set_attribute("function.name", func.__name__)
                        try:
                            result = await func(*args, **kwargs)
                            span.set_status(SpanStatus.OK)
                            return result
                        except Exception as e:
                            span.set_status(SpanStatus.ERROR, str(e))
                            raise
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.start_span(span_name, kind=kind, attributes=attributes) as span:
                        span.set_attribute("function.name", func.__name__)
                        try:
                            result = func(*args, **kwargs)
                            span.set_status(SpanStatus.OK)
                            return result
                        except Exception as e:
                            span.set_status(SpanStatus.ERROR, str(e))
                            raise
                return sync_wrapper

        return decorator

    def extract_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        """Extract trace context from headers."""
        return SpanContext.from_headers(headers)

    def inject_context(self, span: Optional[Span] = None) -> Dict[str, str]:
        """Inject trace context into headers."""
        span = span or self.get_current_span()
        if span:
            return span.context.to_headers()
        return {}


class SpanContextManager:
    """Context manager for spans."""

    def __init__(self, tracer: Tracer, span: Span):
        self._tracer = tracer
        self._span = span
        self._token = None

    def __enter__(self) -> Span:
        self._token = _current_span.set(self._span)
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self._span.set_status(SpanStatus.ERROR, str(exc_val))
        elif self._span.status == SpanStatus.UNSET:
            self._span.set_status(SpanStatus.OK)

        self._tracer._end_span(self._span)
        _current_span.reset(self._token)
        return False

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


class SpanExporter:
    """Base class for span exporters."""

    def export(self, span: Span) -> None:
        """Export a span."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleExporter(SpanExporter):
    """Export spans to console."""

    def export(self, span: Span) -> None:
        """Export span to console."""
        logger.info(
            f"Span: {span.name} "
            f"[{span.context.trace_id[:8]}] "
            f"duration={span.duration_ms:.2f}ms "
            f"status={span.status.value}"
        )


class BatchExporter(SpanExporter):
    """
    Batch exporter that collects spans and exports in batches.

    Useful for sending to external tracing systems.
    """

    def __init__(
        self,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        export_func: Optional[Callable[[List[Span]], None]] = None,
    ):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.export_func = export_func or self._default_export
        self._batch: List[Span] = []
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the batch exporter."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Stop the batch exporter."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._flush()

    def export(self, span: Span) -> None:
        """Add span to batch."""
        self._batch.append(span)
        if len(self._batch) >= self.batch_size:
            asyncio.create_task(self._flush())

    async def _flush_loop(self) -> None:
        """Periodic flush loop."""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self._flush()

    async def _flush(self) -> None:
        """Flush current batch."""
        if not self._batch:
            return

        async with self._lock:
            batch = self._batch
            self._batch = []

        try:
            if asyncio.iscoroutinefunction(self.export_func):
                await self.export_func(batch)
            else:
                self.export_func(batch)
        except Exception as e:
            logger.error(f"Failed to export spans: {e}")

    def _default_export(self, spans: List[Span]) -> None:
        """Default export function."""
        for span in spans:
            logger.info(
                f"Span: {span.name} "
                f"[{span.context.trace_id[:8]}] "
                f"duration={span.duration_ms:.2f}ms"
            )


class InMemoryExporter(SpanExporter):
    """
    In-memory exporter for testing.

    Stores spans in memory for inspection.
    """

    def __init__(self, max_spans: int = 1000):
        self.max_spans = max_spans
        self._spans: List[Span] = []

    def export(self, span: Span) -> None:
        """Store span in memory."""
        self._spans.append(span)
        if len(self._spans) > self.max_spans:
            self._spans = self._spans[-self.max_spans:]

    def get_spans(self) -> List[Span]:
        """Get all stored spans."""
        return self._spans.copy()

    def clear(self) -> None:
        """Clear stored spans."""
        self._spans.clear()

    def find_span(self, name: str) -> Optional[Span]:
        """Find a span by name."""
        for span in reversed(self._spans):
            if span.name == name:
                return span
        return None


# Global tracer
_tracer: Optional[Tracer] = None


def get_tracer(service_name: str = "bvrai") -> Tracer:
    """Get or create the global tracer."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(service_name)
    return _tracer


def trace(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Callable:
    """Convenience decorator using global tracer."""
    return get_tracer().trace(name, kind)
