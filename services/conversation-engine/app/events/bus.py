"""Event bus for pub/sub messaging."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any, Callable, Awaitable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid


logger = structlog.get_logger()


class EventType(str, Enum):
    """Types of events in the system."""
    # Call lifecycle
    CALL_STARTED = "call.started"
    CALL_RINGING = "call.ringing"
    CALL_CONNECTED = "call.connected"
    CALL_ENDED = "call.ended"
    CALL_FAILED = "call.failed"

    # Conversation
    TURN_STARTED = "conversation.turn_started"
    TURN_ENDED = "conversation.turn_ended"
    USER_SPEAKING = "conversation.user_speaking"
    AGENT_SPEAKING = "conversation.agent_speaking"
    SILENCE_DETECTED = "conversation.silence_detected"

    # Transcription
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"

    # AI
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_STREAM_CHUNK = "llm.stream_chunk"
    FUNCTION_CALLED = "function.called"
    FUNCTION_RESULT = "function.result"

    # Audio
    AUDIO_RECEIVED = "audio.received"
    AUDIO_SENT = "audio.sent"
    TTS_STARTED = "tts.started"
    TTS_COMPLETED = "tts.completed"

    # Session
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    SESSION_ENDED = "session.ended"

    # Errors
    ERROR_OCCURRED = "error.occurred"
    RETRY_ATTEMPTED = "error.retry"

    # Analytics
    METRIC_RECORDED = "analytics.metric"
    LATENCY_RECORDED = "analytics.latency"


@dataclass
class Event:
    """An event in the system."""
    type: EventType
    data: Dict[str, Any]
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "conversation-engine"
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "type": self.type.value,
            "data": self.data,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            type=EventType(data["type"]),
            data=data.get("data", {}),
            session_id=data.get("session_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            source=data.get("source", "unknown"),
            correlation_id=data.get("correlation_id"),
        )


EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """
    Central event bus for pub/sub messaging.

    Features:
    - Async event handling
    - Multiple handlers per event type
    - Wildcard subscriptions
    - Event filtering
    - Redis pub/sub integration (optional)
    """

    def __init__(self, redis_client=None):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._wildcard_handlers: List[EventHandler] = []
        self._redis = redis_client
        self._redis_subscriber_task: Optional[asyncio.Task] = None
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None

        # Statistics
        self._events_published = 0
        self._events_processed = 0
        self._handler_errors = 0

    async def start(self) -> None:
        """Start the event bus."""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())

        if self._redis:
            self._redis_subscriber_task = asyncio.create_task(
                self._redis_subscriber()
            )

        logger.info("event_bus_started")

    async def stop(self) -> None:
        """Stop the event bus."""
        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        if self._redis_subscriber_task:
            self._redis_subscriber_task.cancel()
            try:
                await self._redis_subscriber_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "event_bus_stopped",
            events_published=self._events_published,
            events_processed=self._events_processed,
        )

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to
            handler: Async handler function
        """
        key = event_type.value
        if key not in self._handlers:
            self._handlers[key] = []
        self._handlers[key].append(handler)

        logger.debug(
            "handler_subscribed",
            event_type=event_type.value,
            handler_count=len(self._handlers[key]),
        )

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events."""
        self._wildcard_handlers.append(handler)

    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> bool:
        """Unsubscribe from an event type."""
        key = event_type.value
        if key in self._handlers and handler in self._handlers[key]:
            self._handlers[key].remove(handler)
            return True
        return False

    async def publish(self, event: Event) -> None:
        """
        Publish an event.

        Args:
            event: Event to publish
        """
        self._events_published += 1

        # Add to local queue
        await self._event_queue.put(event)

        # Publish to Redis if available
        if self._redis:
            try:
                await self._redis.publish(
                    f"events:{event.type.value}",
                    event.to_json(),
                )
            except Exception as e:
                logger.error("redis_publish_error", error=str(e))

        logger.debug(
            "event_published",
            event_type=event.type.value,
            session_id=event.session_id,
        )

    async def publish_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> Event:
        """
        Convenience method to create and publish an event.

        Returns:
            The created event
        """
        event = Event(
            type=event_type,
            data=data,
            session_id=session_id,
            correlation_id=correlation_id,
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
                self._events_processed += 1

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("event_processing_error", error=str(e))

    async def _dispatch_event(self, event: Event) -> None:
        """Dispatch event to all handlers."""
        handlers = []

        # Get type-specific handlers
        key = event.type.value
        if key in self._handlers:
            handlers.extend(self._handlers[key])

        # Add wildcard handlers
        handlers.extend(self._wildcard_handlers)

        # Execute all handlers
        tasks = []
        for handler in handlers:
            tasks.append(self._safe_handle(handler, event))

        if tasks:
            await asyncio.gather(*tasks)

    async def _safe_handle(
        self,
        handler: EventHandler,
        event: Event,
    ) -> None:
        """Safely execute a handler."""
        try:
            await asyncio.wait_for(handler(event), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                "handler_timeout",
                event_type=event.type.value,
            )
            self._handler_errors += 1
        except Exception as e:
            logger.error(
                "handler_error",
                event_type=event.type.value,
                error=str(e),
            )
            self._handler_errors += 1

    async def _redis_subscriber(self) -> None:
        """Subscribe to Redis pub/sub channels."""
        if not self._redis:
            return

        try:
            pubsub = self._redis.pubsub()
            await pubsub.psubscribe("events:*")

            async for message in pubsub.listen():
                if not self._running:
                    break

                if message["type"] == "pmessage":
                    try:
                        data = json.loads(message["data"])
                        event = Event.from_dict(data)

                        # Don't reprocess our own events
                        if event.source != "conversation-engine":
                            await self._dispatch_event(event)

                    except Exception as e:
                        logger.error("redis_message_error", error=str(e))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("redis_subscriber_error", error=str(e))

    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "events_published": self._events_published,
            "events_processed": self._events_processed,
            "handler_errors": self._handler_errors,
            "queue_size": self._event_queue.qsize(),
            "subscribed_types": list(self._handlers.keys()),
            "type_handler_counts": {
                k: len(v) for k, v in self._handlers.items()
            },
            "wildcard_handlers": len(self._wildcard_handlers),
            "redis_enabled": self._redis is not None,
        }


# Global event bus instance
event_bus = EventBus()
