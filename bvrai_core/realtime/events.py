"""
Event System Module

This module provides an event bus for publishing and subscribing
to events across the real-time system.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
)

from .base import Event, EventType, EventPriority


logger = logging.getLogger(__name__)


@dataclass
class EventSubscription:
    """Subscription to events."""

    id: str = ""
    event_types: Set[EventType] = field(default_factory=set)
    handler: Optional[Callable[[Event], None]] = None
    filter_func: Optional[Callable[[Event], bool]] = None

    # Filtering
    session_filter: Optional[str] = None
    source_filter: Optional[str] = None

    # Options
    priority: int = 0
    once: bool = False

    # State
    active: bool = True
    invocation_count: int = 0

    def matches(self, event: Event) -> bool:
        """Check if event matches subscription."""
        if not self.active:
            return False

        # Check event type
        if self.event_types and event.type not in self.event_types:
            return False

        # Check session filter
        if self.session_filter and event.session_id != self.session_filter:
            return False

        # Check source filter
        if self.source_filter and event.source != self.source_filter:
            return False

        # Check custom filter
        if self.filter_func and not self.filter_func(event):
            return False

        return True


class EventHandler:
    """Wrapper for event handler functions."""

    def __init__(
        self,
        handler: Callable[[Event], None],
        event_types: Optional[Set[EventType]] = None,
    ):
        self.handler = handler
        self.event_types = event_types or set()
        self.invocation_count = 0

    async def handle(self, event: Event) -> None:
        """Handle an event."""
        self.invocation_count += 1

        if asyncio.iscoroutinefunction(self.handler):
            await self.handler(event)
        else:
            await asyncio.to_thread(self.handler, event)


class EventBus:
    """
    Central event bus for the real-time system.

    Provides pub/sub functionality for events with filtering,
    prioritization, and async handling.
    """

    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize event bus.

        Args:
            max_queue_size: Maximum events in queue
        """
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._type_subscriptions: Dict[EventType, Set[str]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._process_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_dropped": 0,
            "handlers_invoked": 0,
        }

    async def start(self) -> None:
        """Start the event bus."""
        self._running = True
        self._process_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")

    async def stop(self) -> None:
        """Stop the event bus."""
        self._running = False

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        logger.info("Event bus stopped")

    def subscribe(
        self,
        event_types: Union[EventType, List[EventType], Set[EventType]],
        handler: Callable[[Event], None],
        session_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        filter_func: Optional[Callable[[Event], bool]] = None,
        priority: int = 0,
        once: bool = False,
    ) -> str:
        """
        Subscribe to events.

        Args:
            event_types: Event type(s) to subscribe to
            handler: Handler function
            session_filter: Filter by session ID
            source_filter: Filter by source
            filter_func: Custom filter function
            priority: Handler priority
            once: Unsubscribe after first invocation

        Returns:
            Subscription ID
        """
        import uuid

        # Normalize event types
        if isinstance(event_types, EventType):
            types_set = {event_types}
        elif isinstance(event_types, list):
            types_set = set(event_types)
        else:
            types_set = event_types

        # Create subscription
        subscription_id = str(uuid.uuid4())
        subscription = EventSubscription(
            id=subscription_id,
            event_types=types_set,
            handler=handler,
            session_filter=session_filter,
            source_filter=source_filter,
            filter_func=filter_func,
            priority=priority,
            once=once,
        )

        # Register subscription
        self._subscriptions[subscription_id] = subscription

        # Index by event type
        for event_type in types_set:
            if event_type not in self._type_subscriptions:
                self._type_subscriptions[event_type] = set()
            self._type_subscriptions[event_type].add(subscription_id)

        logger.debug(f"Subscription created: {subscription_id} for {types_set}")

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: Subscription ID

        Returns:
            True if unsubscribed
        """
        subscription = self._subscriptions.get(subscription_id)
        if not subscription:
            return False

        # Remove from type index
        for event_type in subscription.event_types:
            if event_type in self._type_subscriptions:
                self._type_subscriptions[event_type].discard(subscription_id)

        # Remove subscription
        del self._subscriptions[subscription_id]

        logger.debug(f"Subscription removed: {subscription_id}")

        return True

    async def publish(
        self,
        event: Event,
        blocking: bool = False,
    ) -> bool:
        """
        Publish an event.

        Args:
            event: Event to publish
            blocking: Wait for processing

        Returns:
            True if queued
        """
        try:
            if blocking:
                await self._event_queue.put(event)
            else:
                self._event_queue.put_nowait(event)

            self._stats["events_published"] += 1

            return True

        except asyncio.QueueFull:
            self._stats["events_dropped"] += 1
            logger.warning(f"Event queue full, dropping event: {event.type}")
            return False

    async def emit(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        source: str = "",
        session_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        call_id: Optional[str] = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> Event:
        """
        Emit an event with the given parameters.

        Args:
            event_type: Type of event
            data: Event data
            source: Event source
            session_id: Session ID
            connection_id: Connection ID
            call_id: Call ID
            priority: Event priority

        Returns:
            Created event
        """
        event = Event(
            type=event_type,
            priority=priority,
            source=source,
            session_id=session_id,
            connection_id=connection_id,
            call_id=call_id,
            data=data or {},
        )

        await self.publish(event)

        return event

    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )

                await self._dispatch_event(event)

                self._stats["events_processed"] += 1

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    async def _dispatch_event(self, event: Event) -> None:
        """Dispatch an event to matching subscribers."""
        # Get potential subscriptions
        subscription_ids = set()

        # Add type-specific subscriptions
        if event.type in self._type_subscriptions:
            subscription_ids.update(self._type_subscriptions[event.type])

        # Add wildcard subscriptions (empty event_types means all)
        for sub_id, sub in self._subscriptions.items():
            if not sub.event_types:
                subscription_ids.add(sub_id)

        # Filter and sort by priority
        matching_subs = [
            self._subscriptions[sub_id]
            for sub_id in subscription_ids
            if sub_id in self._subscriptions
            and self._subscriptions[sub_id].matches(event)
        ]

        matching_subs.sort(key=lambda s: s.priority, reverse=True)

        # Dispatch to handlers
        for subscription in matching_subs:
            if event.cancelled:
                break

            try:
                if subscription.handler:
                    if asyncio.iscoroutinefunction(subscription.handler):
                        await subscription.handler(event)
                    else:
                        await asyncio.to_thread(subscription.handler, event)

                    subscription.invocation_count += 1
                    self._stats["handlers_invoked"] += 1

                    # Handle once subscriptions
                    if subscription.once:
                        self.unsubscribe(subscription.id)

            except Exception as e:
                logger.error(f"Error in event handler: {e}")

        event.processed = True

    def on(
        self,
        event_types: Union[EventType, List[EventType]],
        **kwargs,
    ) -> Callable:
        """
        Decorator for subscribing to events.

        Usage:
            @event_bus.on(EventType.CALL_STARTED)
            async def handle_call(event):
                ...
        """
        def decorator(handler: Callable) -> Callable:
            self.subscribe(event_types, handler, **kwargs)
            return handler
        return decorator

    def once(
        self,
        event_types: Union[EventType, List[EventType]],
        **kwargs,
    ) -> Callable:
        """Decorator for one-time subscription."""
        def decorator(handler: Callable) -> Callable:
            self.subscribe(event_types, handler, once=True, **kwargs)
            return handler
        return decorator

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self._stats,
            "subscriptions": len(self._subscriptions),
            "queue_size": self._event_queue.qsize(),
        }

    def clear_subscriptions(self) -> None:
        """Clear all subscriptions."""
        self._subscriptions.clear()
        self._type_subscriptions.clear()


def create_event_bus(auto_start: bool = True) -> EventBus:
    """
    Create an event bus instance.

    Args:
        auto_start: Start processing immediately

    Returns:
        Event bus instance
    """
    bus = EventBus()

    if auto_start:
        asyncio.create_task(bus.start())

    return bus


__all__ = [
    "EventSubscription",
    "EventHandler",
    "EventBus",
    "create_event_bus",
]
