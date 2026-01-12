"""Event subscriber utilities."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any, Callable, Awaitable, Set
from dataclasses import dataclass
from datetime import datetime
import json

from app.events.bus import Event, EventType, event_bus


logger = structlog.get_logger()


@dataclass
class SubscriptionConfig:
    """Configuration for a subscription."""
    event_types: List[EventType]
    session_filter: Optional[str] = None
    include_data: bool = True
    buffer_size: int = 100


class EventSubscriber:
    """
    Event subscriber for receiving events.

    Provides:
    - Async iteration over events
    - Filtering by type and session
    - Buffering and backpressure
    """

    def __init__(self, config: Optional[SubscriptionConfig] = None):
        self.config = config or SubscriptionConfig(event_types=[])
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.buffer_size)
        self._subscribed = False
        self._handlers: List[Callable] = []

    async def start(self) -> None:
        """Start receiving events."""
        if self._subscribed:
            return

        handler = self._create_handler()

        for event_type in self.config.event_types:
            event_bus.subscribe(event_type, handler)
            self._handlers.append((event_type, handler))

        self._subscribed = True
        logger.debug(
            "subscriber_started",
            event_types=[et.value for et in self.config.event_types],
        )

    async def stop(self) -> None:
        """Stop receiving events."""
        for event_type, handler in self._handlers:
            event_bus.unsubscribe(event_type, handler)

        self._handlers.clear()
        self._subscribed = False

    def _create_handler(self) -> Callable[[Event], Awaitable[None]]:
        """Create handler function."""
        async def handler(event: Event) -> None:
            # Apply session filter
            if self.config.session_filter:
                if event.session_id != self.config.session_filter:
                    return

            # Add to queue (non-blocking)
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("subscriber_queue_full")
                # Drop oldest event
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(event)
                except asyncio.QueueEmpty:
                    pass

        return handler

    async def receive(self, timeout: Optional[float] = None) -> Optional[Event]:
        """
        Receive next event.

        Args:
            timeout: Timeout in seconds (None = block forever)

        Returns:
            Event or None if timeout
        """
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout,
                )
            else:
                return await self._queue.get()
        except asyncio.TimeoutError:
            return None

    async def receive_batch(
        self,
        max_events: int = 10,
        timeout: float = 1.0,
    ) -> List[Event]:
        """
        Receive a batch of events.

        Args:
            max_events: Maximum events to receive
            timeout: Timeout for batch

        Returns:
            List of events
        """
        events = []
        deadline = asyncio.get_event_loop().time() + timeout

        while len(events) < max_events:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break

            event = await self.receive(timeout=remaining)
            if event:
                events.append(event)
            else:
                break

        return events

    def pending_count(self) -> int:
        """Get number of pending events."""
        return self._queue.qsize()

    async def drain(self) -> List[Event]:
        """Drain all pending events."""
        events = []
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                events.append(event)
            except asyncio.QueueEmpty:
                break
        return events

    async def __aenter__(self) -> "EventSubscriber":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

    def __aiter__(self):
        """Async iterator."""
        return self

    async def __anext__(self) -> Event:
        """Get next event (for async iteration)."""
        event = await self.receive()
        if event is None:
            raise StopAsyncIteration
        return event


class SessionSubscriber(EventSubscriber):
    """
    Subscriber for a specific session.

    Filters all events to a single session and
    provides session-specific utilities.
    """

    def __init__(
        self,
        session_id: str,
        event_types: Optional[List[EventType]] = None,
    ):
        super().__init__(
            SubscriptionConfig(
                event_types=event_types or list(EventType),
                session_filter=session_id,
            )
        )
        self.session_id = session_id

    async def wait_for(
        self,
        event_type: EventType,
        timeout: float = 30.0,
        condition: Optional[Callable[[Event], bool]] = None,
    ) -> Optional[Event]:
        """
        Wait for a specific event type.

        Args:
            event_type: Event type to wait for
            timeout: Timeout in seconds
            condition: Optional condition to match

        Returns:
            Matching event or None
        """
        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None

            event = await self.receive(timeout=remaining)
            if not event:
                return None

            if event.type == event_type:
                if condition is None or condition(event):
                    return event

    async def wait_for_call_end(
        self,
        timeout: float = 3600.0,
    ) -> Optional[Event]:
        """Wait for call to end."""
        return await self.wait_for(EventType.CALL_ENDED, timeout=timeout)

    async def wait_for_transcript(
        self,
        timeout: float = 30.0,
        min_confidence: float = 0.0,
    ) -> Optional[str]:
        """Wait for final transcript."""
        event = await self.wait_for(
            EventType.TRANSCRIPT_FINAL,
            timeout=timeout,
            condition=lambda e: e.data.get("confidence", 0) >= min_confidence,
        )
        if event:
            return event.data.get("text")
        return None


class WebSocketBridge:
    """
    Bridge events to WebSocket connections.

    Allows clients to receive real-time events
    over WebSocket connections.
    """

    def __init__(self):
        self._connections: Dict[str, Set[Any]] = {}  # session_id -> websockets
        self._subscriber: Optional[EventSubscriber] = None
        self._running = False

    async def start(self) -> None:
        """Start the WebSocket bridge."""
        self._subscriber = EventSubscriber(
            SubscriptionConfig(
                event_types=list(EventType),
                buffer_size=1000,
            )
        )
        await self._subscriber.start()
        self._running = True

        asyncio.create_task(self._relay_events())
        logger.info("websocket_bridge_started")

    async def stop(self) -> None:
        """Stop the WebSocket bridge."""
        self._running = False
        if self._subscriber:
            await self._subscriber.stop()

    def add_connection(self, session_id: str, websocket: Any) -> None:
        """Add WebSocket connection for a session."""
        if session_id not in self._connections:
            self._connections[session_id] = set()
        self._connections[session_id].add(websocket)

    def remove_connection(self, session_id: str, websocket: Any) -> None:
        """Remove WebSocket connection."""
        if session_id in self._connections:
            self._connections[session_id].discard(websocket)
            if not self._connections[session_id]:
                del self._connections[session_id]

    async def _relay_events(self) -> None:
        """Relay events to WebSocket connections."""
        while self._running:
            try:
                event = await self._subscriber.receive(timeout=1.0)
                if not event:
                    continue

                session_id = event.session_id
                if not session_id or session_id not in self._connections:
                    continue

                # Send to all connections for this session
                message = event.to_json()
                dead_connections = []

                for ws in self._connections[session_id]:
                    try:
                        await ws.send_text(message)
                    except Exception:
                        dead_connections.append(ws)

                # Clean up dead connections
                for ws in dead_connections:
                    self.remove_connection(session_id, ws)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("websocket_relay_error", error=str(e))
