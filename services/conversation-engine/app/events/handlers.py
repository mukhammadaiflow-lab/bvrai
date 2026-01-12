"""Event handlers for common operations."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any, Callable, Awaitable, Type
from dataclasses import dataclass
from functools import wraps

from app.events.bus import Event, EventType, EventHandler


logger = structlog.get_logger()


@dataclass
class HandlerRegistration:
    """Registration info for a handler."""
    event_type: EventType
    handler: EventHandler
    priority: int
    filter_fn: Optional[Callable[[Event], bool]]
    name: str


class EventHandlerRegistry:
    """
    Registry for event handlers.

    Provides:
    - Handler registration with priorities
    - Event filtering
    - Handler grouping
    - Automatic retry
    """

    def __init__(self):
        self._handlers: Dict[str, List[HandlerRegistration]] = {}
        self._handler_groups: Dict[str, List[str]] = {}

    def register(
        self,
        event_type: EventType,
        handler: EventHandler,
        priority: int = 0,
        filter_fn: Optional[Callable[[Event], bool]] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
    ) -> str:
        """
        Register an event handler.

        Args:
            event_type: Event type to handle
            handler: Handler function
            priority: Higher priority runs first
            filter_fn: Optional filter function
            name: Handler name for debugging
            group: Handler group for batch operations

        Returns:
            Handler registration name
        """
        handler_name = name or f"{event_type.value}_{id(handler)}"

        registration = HandlerRegistration(
            event_type=event_type,
            handler=handler,
            priority=priority,
            filter_fn=filter_fn,
            name=handler_name,
        )

        key = event_type.value
        if key not in self._handlers:
            self._handlers[key] = []

        self._handlers[key].append(registration)

        # Sort by priority (descending)
        self._handlers[key].sort(key=lambda r: r.priority, reverse=True)

        # Track in group
        if group:
            if group not in self._handler_groups:
                self._handler_groups[group] = []
            self._handler_groups[group].append(handler_name)

        logger.debug(
            "handler_registered",
            name=handler_name,
            event_type=event_type.value,
            priority=priority,
        )

        return handler_name

    def unregister(self, name: str) -> bool:
        """Unregister a handler by name."""
        for key, handlers in self._handlers.items():
            for i, reg in enumerate(handlers):
                if reg.name == name:
                    del handlers[i]
                    return True
        return False

    def get_handlers(
        self,
        event_type: EventType,
        event: Optional[Event] = None,
    ) -> List[EventHandler]:
        """Get handlers for an event type, optionally filtered."""
        key = event_type.value
        if key not in self._handlers:
            return []

        handlers = []
        for reg in self._handlers[key]:
            # Apply filter if present
            if reg.filter_fn and event:
                if not reg.filter_fn(event):
                    continue
            handlers.append(reg.handler)

        return handlers

    def disable_group(self, group: str) -> int:
        """Disable all handlers in a group."""
        if group not in self._handler_groups:
            return 0

        count = 0
        for name in self._handler_groups[group]:
            if self.unregister(name):
                count += 1

        return count


def on_event(
    event_type: EventType,
    priority: int = 0,
    filter_fn: Optional[Callable[[Event], bool]] = None,
):
    """Decorator for event handlers."""
    def decorator(func: EventHandler) -> EventHandler:
        func._event_type = event_type
        func._priority = priority
        func._filter_fn = filter_fn
        return func
    return decorator


def retry_on_error(
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    exceptions: tuple = (Exception,),
):
    """Decorator to add retry logic to handlers."""
    def decorator(handler: EventHandler) -> EventHandler:
        @wraps(handler)
        async def wrapper(event: Event) -> None:
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    await handler(event)
                    return
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries:
                        await asyncio.sleep(delay_seconds * (attempt + 1))
                        logger.warning(
                            "handler_retry",
                            handler=handler.__name__,
                            attempt=attempt + 1,
                            error=str(e),
                        )

            # All retries failed
            logger.error(
                "handler_failed_after_retries",
                handler=handler.__name__,
                retries=max_retries,
                error=str(last_error),
            )
            raise last_error

        return wrapper
    return decorator


def filter_session(session_id: str) -> Callable[[Event], bool]:
    """Create filter for specific session."""
    def filter_fn(event: Event) -> bool:
        return event.session_id == session_id
    return filter_fn


def filter_sessions(session_ids: List[str]) -> Callable[[Event], bool]:
    """Create filter for multiple sessions."""
    session_set = set(session_ids)
    def filter_fn(event: Event) -> bool:
        return event.session_id in session_set
    return filter_fn


def filter_data(key: str, value: Any) -> Callable[[Event], bool]:
    """Create filter based on event data."""
    def filter_fn(event: Event) -> bool:
        return event.data.get(key) == value
    return filter_fn


# Common handler implementations
class AnalyticsHandler:
    """Handler for recording analytics events."""

    def __init__(self, metrics_client=None):
        self._metrics = metrics_client
        self._event_counts: Dict[str, int] = {}

    async def handle_event(self, event: Event) -> None:
        """Record event for analytics."""
        event_type = event.type.value
        self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1

        if self._metrics:
            await self._metrics.increment(f"events.{event_type}")

            # Record latency for specific events
            if event.type == EventType.LLM_RESPONSE:
                latency = event.data.get("latency_ms", 0)
                await self._metrics.histogram("llm.latency_ms", latency)

            elif event.type == EventType.TRANSCRIPT_FINAL:
                duration = event.data.get("duration_ms", 0)
                await self._metrics.histogram("asr.duration_ms", duration)

    def get_counts(self) -> Dict[str, int]:
        """Get event counts."""
        return dict(self._event_counts)


class LoggingHandler:
    """Handler for logging events."""

    def __init__(self, log_level: str = "info"):
        self._log_level = log_level

    async def handle_event(self, event: Event) -> None:
        """Log event."""
        log_data = {
            "event_type": event.type.value,
            "session_id": event.session_id,
            "event_id": event.event_id,
        }

        # Add relevant data fields
        if event.type == EventType.ERROR_OCCURRED:
            log_data["error"] = event.data.get("error")
            log_data["error_type"] = event.data.get("error_type")

        if self._log_level == "debug":
            log_data["data"] = event.data

        getattr(logger, self._log_level)("event", **log_data)


class WebhookHandler:
    """Handler for sending events to webhooks."""

    def __init__(
        self,
        webhook_url: str,
        event_types: Optional[List[EventType]] = None,
        http_client=None,
    ):
        self.webhook_url = webhook_url
        self.event_types = set(event_types) if event_types else None
        self._http_client = http_client

    async def handle_event(self, event: Event) -> None:
        """Send event to webhook."""
        if self.event_types and event.type not in self.event_types:
            return

        if not self._http_client:
            return

        try:
            await self._http_client.post(
                self.webhook_url,
                json=event.to_dict(),
                timeout=5.0,
            )
        except Exception as e:
            logger.error(
                "webhook_error",
                url=self.webhook_url,
                error=str(e),
            )
