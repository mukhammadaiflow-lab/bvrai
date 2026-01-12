"""WebSocket event handling."""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of WebSocket events."""
    # Connection events
    CONNECTION_OPENED = "connection.opened"
    CONNECTION_CLOSED = "connection.closed"
    CONNECTION_ERROR = "connection.error"
    CONNECTION_AUTHENTICATED = "connection.authenticated"

    # Call events
    CALL_STARTED = "call.started"
    CALL_ENDED = "call.ended"
    CALL_UPDATED = "call.updated"
    CALL_FAILED = "call.failed"
    CALL_TRANSFERRED = "call.transferred"

    # Audio events
    AUDIO_STARTED = "audio.started"
    AUDIO_STOPPED = "audio.stopped"
    AUDIO_MUTED = "audio.muted"
    AUDIO_UNMUTED = "audio.unmuted"

    # Speech events
    SPEECH_STARTED = "speech.started"
    SPEECH_ENDED = "speech.ended"
    TRANSCRIPT_RECEIVED = "transcript.received"
    TRANSCRIPT_FINAL = "transcript.final"

    # Agent events
    AGENT_THINKING = "agent.thinking"
    AGENT_RESPONDING = "agent.responding"
    AGENT_RESPONSE_COMPLETE = "agent.response_complete"
    AGENT_FUNCTION_CALLED = "agent.function_called"
    AGENT_FUNCTION_COMPLETED = "agent.function_completed"

    # User events
    USER_INTERRUPTED = "user.interrupted"
    USER_HOLD = "user.hold"
    USER_RESUMED = "user.resumed"

    # System events
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_INFO = "system.info"

    # Custom events
    CUSTOM = "custom"


@dataclass
class WebSocketEvent:
    """WebSocket event data."""
    type: EventType
    connection_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "server"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "event_id": self.event_id,
            "connection_id": self.connection_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebSocketEvent":
        """Create from dictionary."""
        return cls(
            type=EventType(data.get("type", "custom")),
            event_id=data.get("event_id", str(uuid.uuid4())),
            connection_id=data.get("connection_id", ""),
            data=data.get("data", {}),
            source=data.get("source", "client"),
            metadata=data.get("metadata", {}),
        )


EventHandler = Callable[[WebSocketEvent], Awaitable[None]]


class EventDispatcher:
    """
    Event dispatcher for WebSocket events.

    Routes events to registered handlers.

    Usage:
        dispatcher = EventDispatcher()

        @dispatcher.on(EventType.CALL_STARTED)
        async def handle_call_started(event):
            print(f"Call started: {event.data}")

        await dispatcher.dispatch(event)
    """

    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._middleware: List[Callable[[WebSocketEvent], Awaitable[bool]]] = []

    def on(self, event_type: EventType) -> Callable:
        """
        Decorator to register event handler.

        Usage:
            @dispatcher.on(EventType.CALL_STARTED)
            async def handler(event):
                ...
        """
        def decorator(func: EventHandler) -> EventHandler:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(func)
            return func
        return decorator

    def on_all(self, handler: EventHandler) -> None:
        """Register handler for all events."""
        self._global_handlers.append(handler)

    def use(self, middleware: Callable[[WebSocketEvent], Awaitable[bool]]) -> None:
        """Add middleware (return False to stop propagation)."""
        self._middleware.append(middleware)

    def remove(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a handler."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    async def dispatch(self, event: WebSocketEvent) -> None:
        """Dispatch event to handlers."""
        # Run middleware
        for middleware in self._middleware:
            try:
                should_continue = await middleware(event)
                if not should_continue:
                    return
            except Exception as e:
                logger.error(f"Middleware error: {e}")

        # Run global handlers
        for handler in self._global_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Global handler error: {e}")

        # Run type-specific handlers
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Event handler error for {event.type}: {e}")

    async def dispatch_all(self, events: List[WebSocketEvent]) -> None:
        """Dispatch multiple events."""
        for event in events:
            await self.dispatch(event)


class EventQueue:
    """
    Async event queue with priority support.

    Usage:
        queue = EventQueue()
        await queue.put(event)
        event = await queue.get()
    """

    def __init__(self, maxsize: int = 1000):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=maxsize)
        self._counter = 0

    async def put(self, event: WebSocketEvent, priority: int = 5) -> None:
        """Add event to queue (lower priority = higher precedence)."""
        self._counter += 1
        await self._queue.put((priority, self._counter, event))

    async def get(self) -> WebSocketEvent:
        """Get next event from queue."""
        _, _, event = await self._queue.get()
        return event

    def task_done(self) -> None:
        """Mark task as done."""
        self._queue.task_done()

    @property
    def qsize(self) -> int:
        """Get queue size."""
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()


class EventProcessor:
    """
    Background event processor.

    Processes events from queue using dispatcher.

    Usage:
        processor = EventProcessor(dispatcher)
        await processor.start()

        await processor.submit(event)

        await processor.stop()
    """

    def __init__(
        self,
        dispatcher: EventDispatcher,
        num_workers: int = 4,
        queue_size: int = 1000,
    ):
        self.dispatcher = dispatcher
        self.num_workers = num_workers
        self._queue = EventQueue(maxsize=queue_size)
        self._workers: List[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """Start event processor."""
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.num_workers)
        ]
        logger.info(f"Event processor started with {self.num_workers} workers")

    async def stop(self) -> None:
        """Stop event processor."""
        self._running = False
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        logger.info("Event processor stopped")

    async def submit(self, event: WebSocketEvent, priority: int = 5) -> None:
        """Submit event for processing."""
        await self._queue.put(event, priority)

    async def _worker(self, worker_id: int) -> None:
        """Worker loop."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
                await self.dispatcher.dispatch(event)
                self._queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")


class EventEmitter:
    """
    Event emitter for creating and dispatching events.

    Usage:
        emitter = EventEmitter(dispatcher)

        await emitter.emit(EventType.CALL_STARTED, {
            "call_id": "123",
            "agent_id": "456",
        })
    """

    def __init__(
        self,
        dispatcher: Optional[EventDispatcher] = None,
        processor: Optional[EventProcessor] = None,
    ):
        self.dispatcher = dispatcher or EventDispatcher()
        self.processor = processor

    async def emit(
        self,
        event_type: EventType,
        connection_id: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> WebSocketEvent:
        """Emit an event."""
        event = WebSocketEvent(
            type=event_type,
            connection_id=connection_id,
            data=data or {},
            **kwargs,
        )

        if self.processor:
            await self.processor.submit(event)
        else:
            await self.dispatcher.dispatch(event)

        return event

    async def emit_call_started(
        self,
        connection_id: str,
        call_id: str,
        agent_id: str,
        **kwargs,
    ) -> WebSocketEvent:
        """Emit call started event."""
        return await self.emit(
            EventType.CALL_STARTED,
            connection_id,
            {
                "call_id": call_id,
                "agent_id": agent_id,
                **kwargs,
            },
        )

    async def emit_call_ended(
        self,
        connection_id: str,
        call_id: str,
        reason: str = "completed",
        duration_ms: float = 0,
        **kwargs,
    ) -> WebSocketEvent:
        """Emit call ended event."""
        return await self.emit(
            EventType.CALL_ENDED,
            connection_id,
            {
                "call_id": call_id,
                "reason": reason,
                "duration_ms": duration_ms,
                **kwargs,
            },
        )

    async def emit_transcript(
        self,
        connection_id: str,
        text: str,
        is_final: bool = False,
        confidence: float = 1.0,
        **kwargs,
    ) -> WebSocketEvent:
        """Emit transcript event."""
        event_type = EventType.TRANSCRIPT_FINAL if is_final else EventType.TRANSCRIPT_RECEIVED
        return await self.emit(
            event_type,
            connection_id,
            {
                "text": text,
                "is_final": is_final,
                "confidence": confidence,
                **kwargs,
            },
        )

    async def emit_agent_response(
        self,
        connection_id: str,
        text: str,
        call_id: Optional[str] = None,
        **kwargs,
    ) -> WebSocketEvent:
        """Emit agent response event."""
        return await self.emit(
            EventType.AGENT_RESPONDING,
            connection_id,
            {
                "text": text,
                "call_id": call_id,
                **kwargs,
            },
        )

    async def emit_function_call(
        self,
        connection_id: str,
        function_name: str,
        arguments: Dict[str, Any],
        call_id: Optional[str] = None,
        **kwargs,
    ) -> WebSocketEvent:
        """Emit function call event."""
        return await self.emit(
            EventType.AGENT_FUNCTION_CALLED,
            connection_id,
            {
                "function_name": function_name,
                "arguments": arguments,
                "call_id": call_id,
                **kwargs,
            },
        )

    async def emit_error(
        self,
        connection_id: str,
        error: str,
        code: Optional[str] = None,
        **kwargs,
    ) -> WebSocketEvent:
        """Emit error event."""
        return await self.emit(
            EventType.SYSTEM_ERROR,
            connection_id,
            {
                "error": error,
                "code": code,
                **kwargs,
            },
        )


# Global event dispatcher
_event_dispatcher: Optional[EventDispatcher] = None


def get_event_dispatcher() -> EventDispatcher:
    """Get or create the global event dispatcher."""
    global _event_dispatcher
    if _event_dispatcher is None:
        _event_dispatcher = EventDispatcher()
    return _event_dispatcher


def get_event_emitter(
    dispatcher: Optional[EventDispatcher] = None,
) -> EventEmitter:
    """Get an event emitter."""
    return EventEmitter(dispatcher or get_event_dispatcher())
