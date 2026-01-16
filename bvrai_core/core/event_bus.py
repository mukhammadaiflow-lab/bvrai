"""
Distributed Event Bus System
============================

High-performance, distributed event bus for real-time event propagation
across all platform services. Supports:
- Multiple transport backends (Redis, RabbitMQ, Kafka, In-Memory)
- Event filtering and routing
- Priority queues
- Dead letter handling
- Event persistence and replay
- Async event processing

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import time
import zlib
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import partial, wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4
import re

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

T = TypeVar("T")
EventHandler = Callable[["Event"], Awaitable[None]]


# =============================================================================
# ENUMS
# =============================================================================


class EventType(str, Enum):
    """Standard event types"""

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"

    # Service events
    SERVICE_REGISTERED = "service.registered"
    SERVICE_STARTED = "service.started"
    SERVICE_STOPPED = "service.stopped"
    SERVICE_HEALTHY = "service.healthy"
    SERVICE_UNHEALTHY = "service.unhealthy"
    SERVICE_RECOVERED = "service.recovered"

    # Call events
    CALL_INITIATED = "call.initiated"
    CALL_RINGING = "call.ringing"
    CALL_ANSWERED = "call.answered"
    CALL_ENDED = "call.ended"
    CALL_FAILED = "call.failed"
    CALL_TRANSFERRED = "call.transferred"
    CALL_RECORDING_STARTED = "call.recording.started"
    CALL_RECORDING_STOPPED = "call.recording.stopped"

    # Conversation events
    CONVERSATION_STARTED = "conversation.started"
    CONVERSATION_ENDED = "conversation.ended"
    CONVERSATION_MESSAGE = "conversation.message"
    CONVERSATION_TURN_START = "conversation.turn.start"
    CONVERSATION_TURN_END = "conversation.turn.end"
    CONVERSATION_INTENT_DETECTED = "conversation.intent.detected"
    CONVERSATION_SENTIMENT_CHANGE = "conversation.sentiment.change"

    # Agent events
    AGENT_CREATED = "agent.created"
    AGENT_UPDATED = "agent.updated"
    AGENT_DELETED = "agent.deleted"
    AGENT_PUBLISHED = "agent.published"
    AGENT_DEPLOYED = "agent.deployed"

    # Voice events
    VOICE_SPEECH_START = "voice.speech.start"
    VOICE_SPEECH_END = "voice.speech.end"
    VOICE_TRANSCRIPTION = "voice.transcription"
    VOICE_SYNTHESIS_START = "voice.synthesis.start"
    VOICE_SYNTHESIS_END = "voice.synthesis.end"
    VOICE_INTERRUPTION = "voice.interruption"

    # AI events
    AI_INFERENCE_START = "ai.inference.start"
    AI_INFERENCE_END = "ai.inference.end"
    AI_TOOL_CALL = "ai.tool.call"
    AI_TOOL_RESULT = "ai.tool.result"
    AI_STREAM_TOKEN = "ai.stream.token"

    # Analytics events
    ANALYTICS_EVENT = "analytics.event"
    ANALYTICS_METRIC = "analytics.metric"

    # Billing events
    BILLING_USAGE = "billing.usage"
    BILLING_INVOICE = "billing.invoice"
    BILLING_PAYMENT = "billing.payment"

    # Custom event
    CUSTOM = "custom"


class EventPriority(int, Enum):
    """Event priority levels"""

    CRITICAL = 0      # Immediate processing, system-critical
    HIGH = 10         # High priority, process soon
    NORMAL = 50       # Normal priority
    LOW = 100         # Low priority, can be deferred
    BACKGROUND = 200  # Background processing


class EventStatus(str, Enum):
    """Event processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    DEAD_LETTER = "dead_letter"


class TransportType(str, Enum):
    """Event transport backend types"""

    MEMORY = "memory"
    REDIS = "redis"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class EventMetadata:
    """Event metadata"""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    agent_id: Optional[str] = None
    call_id: Optional[str] = None
    conversation_id: Optional[str] = None
    version: str = "1.0"
    content_type: str = "application/json"
    encoding: str = "utf-8"
    compressed: bool = False
    encrypted: bool = False
    retry_count: int = 0
    max_retries: int = 3
    ttl_seconds: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "call_id": self.call_id,
            "conversation_id": self.conversation_id,
            "version": self.version,
            "content_type": self.content_type,
            "encoding": self.encoding,
            "compressed": self.compressed,
            "encrypted": self.encrypted,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "ttl_seconds": self.ttl_seconds,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class Event:
    """
    Event object representing a platform event.

    Events are immutable once created and contain all information
    needed for processing.
    """

    type: str
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: EventMetadata = field(default_factory=EventMetadata)
    priority: EventPriority = EventPriority.NORMAL
    topic: str = "default"
    tags: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def id(self) -> str:
        return self.metadata.event_id

    @property
    def age_ms(self) -> float:
        return (datetime.utcnow() - self.timestamp).total_seconds() * 1000

    @property
    def is_expired(self) -> bool:
        if self.metadata.expires_at:
            return datetime.utcnow() > self.metadata.expires_at
        return False

    def with_correlation(self, correlation_id: str) -> "Event":
        """Create a copy with correlation ID"""
        new_metadata = EventMetadata(**vars(self.metadata))
        new_metadata.correlation_id = correlation_id
        return Event(
            type=self.type,
            source=self.source,
            data=self.data.copy(),
            metadata=new_metadata,
            priority=self.priority,
            topic=self.topic,
            tags=self.tags.copy(),
            timestamp=self.timestamp
        )

    def with_causation(self, causation_id: str) -> "Event":
        """Create a copy with causation ID"""
        new_metadata = EventMetadata(**vars(self.metadata))
        new_metadata.causation_id = causation_id
        return Event(
            type=self.type,
            source=self.source,
            data=self.data.copy(),
            metadata=new_metadata,
            priority=self.priority,
            topic=self.topic,
            tags=self.tags.copy(),
            timestamp=self.timestamp
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "source": self.source,
            "data": self.data,
            "metadata": self.metadata.to_dict(),
            "priority": self.priority.value,
            "topic": self.topic,
            "tags": list(self.tags),
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        metadata_dict = data.get("metadata", {})
        if "created_at" in metadata_dict and isinstance(metadata_dict["created_at"], str):
            metadata_dict["created_at"] = datetime.fromisoformat(metadata_dict["created_at"])
        if "expires_at" in metadata_dict and metadata_dict["expires_at"]:
            metadata_dict["expires_at"] = datetime.fromisoformat(metadata_dict["expires_at"])

        return cls(
            type=data["type"],
            source=data["source"],
            data=data.get("data", {}),
            metadata=EventMetadata(**metadata_dict),
            priority=EventPriority(data.get("priority", EventPriority.NORMAL.value)),
            topic=data.get("topic", "default"),
            tags=set(data.get("tags", [])),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else datetime.utcnow()
        )

    def serialize(self, compress: bool = False) -> bytes:
        """Serialize event to bytes"""
        data = json.dumps(self.to_dict()).encode("utf-8")
        if compress:
            data = zlib.compress(data)
        return data

    @classmethod
    def deserialize(cls, data: bytes, compressed: bool = False) -> "Event":
        """Deserialize event from bytes"""
        if compressed:
            data = zlib.decompress(data)
        return cls.from_dict(json.loads(data.decode("utf-8")))


@dataclass
class EventFilter:
    """Filter for event subscriptions"""

    event_types: Optional[Set[str]] = None
    sources: Optional[Set[str]] = None
    topics: Optional[Set[str]] = None
    tags: Optional[Set[str]] = None
    min_priority: Optional[EventPriority] = None
    max_priority: Optional[EventPriority] = None
    type_pattern: Optional[str] = None
    source_pattern: Optional[str] = None
    metadata_filters: Optional[Dict[str, Any]] = None

    _type_regex: Optional[Pattern] = field(default=None, repr=False)
    _source_regex: Optional[Pattern] = field(default=None, repr=False)

    def __post_init__(self):
        if self.type_pattern:
            self._type_regex = re.compile(self.type_pattern)
        if self.source_pattern:
            self._source_regex = re.compile(self.source_pattern)

    def matches(self, event: Event) -> bool:
        """Check if event matches this filter"""

        # Check event types
        if self.event_types and event.type not in self.event_types:
            return False

        # Check sources
        if self.sources and event.source not in self.sources:
            return False

        # Check topics
        if self.topics and event.topic not in self.topics:
            return False

        # Check tags (any match)
        if self.tags and not self.tags.intersection(event.tags):
            return False

        # Check priority range
        if self.min_priority and event.priority.value > self.min_priority.value:
            return False
        if self.max_priority and event.priority.value < self.max_priority.value:
            return False

        # Check type pattern
        if self._type_regex and not self._type_regex.match(event.type):
            return False

        # Check source pattern
        if self._source_regex and not self._source_regex.match(event.source):
            return False

        # Check metadata filters
        if self.metadata_filters:
            for key, value in self.metadata_filters.items():
                if hasattr(event.metadata, key):
                    if getattr(event.metadata, key) != value:
                        return False

        return True


@dataclass
class EventSubscriber:
    """Event subscriber configuration"""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    handler: EventHandler = field(default=None)
    filter: EventFilter = field(default_factory=EventFilter)
    priority: int = 0
    async_processing: bool = True
    batch_size: int = 1
    batch_timeout_ms: float = 100.0
    max_concurrent: int = 10
    retry_on_error: bool = True
    error_handler: Optional[Callable[[Event, Exception], Awaitable[None]]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True
    events_processed: int = 0
    events_failed: int = 0


# =============================================================================
# EVENT TRANSPORT BACKENDS
# =============================================================================


class EventTransport(ABC):
    """Abstract base class for event transport backends"""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the transport backend"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the transport backend"""
        pass

    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish an event"""
        pass

    @abstractmethod
    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], Awaitable[None]]
    ) -> str:
        """Subscribe to a topic, returns subscription ID"""
        pass

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from a topic"""
        pass

    @abstractmethod
    async def acknowledge(self, event: Event) -> None:
        """Acknowledge event processing"""
        pass

    @abstractmethod
    async def reject(self, event: Event, requeue: bool = True) -> None:
        """Reject event processing"""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected"""
        pass


class InMemoryTransport(EventTransport):
    """In-memory event transport for development and testing"""

    def __init__(
        self,
        max_queue_size: int = 10000,
        max_history_size: int = 1000
    ):
        self._queues: Dict[str, Deque[Event]] = defaultdict(lambda: deque(maxlen=max_queue_size))
        self._subscribers: Dict[str, List[Tuple[str, Callable[[Event], Awaitable[None]]]]] = defaultdict(list)
        self._subscription_counter = 0
        self._connected = False
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        self._history: Deque[Event] = deque(maxlen=max_history_size)
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger("event_bus.transport.memory")

    async def connect(self) -> None:
        self._connected = True
        self._logger.info("in_memory_transport_connected")

    async def disconnect(self) -> None:
        # Cancel all processing tasks
        for task in self._processing_tasks.values():
            task.cancel()

        self._processing_tasks.clear()
        self._connected = False
        self._logger.info("in_memory_transport_disconnected")

    async def publish(self, event: Event) -> None:
        if not self._connected:
            raise RuntimeError("Transport not connected")

        async with self._lock:
            self._queues[event.topic].append(event)
            self._history.append(event)

        # Notify subscribers
        await self._notify_subscribers(event)

    async def _notify_subscribers(self, event: Event) -> None:
        """Notify all subscribers of a new event"""
        subscribers = self._subscribers.get(event.topic, [])
        subscribers.extend(self._subscribers.get("*", []))  # Wildcard subscribers

        for sub_id, handler in subscribers:
            try:
                asyncio.create_task(handler(event))
            except Exception as e:
                self._logger.error(
                    "subscriber_notification_error",
                    subscription_id=sub_id,
                    error=str(e)
                )

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], Awaitable[None]]
    ) -> str:
        self._subscription_counter += 1
        sub_id = f"mem-sub-{self._subscription_counter}"
        self._subscribers[topic].append((sub_id, handler))
        self._logger.debug("subscription_created", subscription_id=sub_id, topic=topic)
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        for topic, subs in self._subscribers.items():
            self._subscribers[topic] = [
                (sid, h) for sid, h in subs if sid != subscription_id
            ]
        self._logger.debug("subscription_removed", subscription_id=subscription_id)

    async def acknowledge(self, event: Event) -> None:
        pass  # No-op for in-memory transport

    async def reject(self, event: Event, requeue: bool = True) -> None:
        if requeue:
            async with self._lock:
                self._queues[event.topic].appendleft(event)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """Get event history"""
        events = list(self._history)
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]


class RedisTransport(EventTransport):
    """Redis Pub/Sub event transport"""

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "builder_engine",
        stream_max_len: int = 10000,
        consumer_group: str = "default",
        consumer_name: Optional[str] = None
    ):
        self._url = url
        self._prefix = prefix
        self._stream_max_len = stream_max_len
        self._consumer_group = consumer_group
        self._consumer_name = consumer_name or f"consumer-{uuid4().hex[:8]}"
        self._redis: Optional[Any] = None
        self._pubsub: Optional[Any] = None
        self._subscriptions: Dict[str, Callable] = {}
        self._connected = False
        self._logger = structlog.get_logger("event_bus.transport.redis")
        self._listener_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis

            self._redis = await aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=False
            )
            self._pubsub = self._redis.pubsub()
            self._connected = True

            # Start listener task
            self._listener_task = asyncio.create_task(self._listen())

            self._logger.info("redis_transport_connected", url=self._url)

        except Exception as e:
            self._logger.error("redis_connection_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._pubsub:
            await self._pubsub.close()

        if self._redis:
            await self._redis.close()

        self._connected = False
        self._logger.info("redis_transport_disconnected")

    async def publish(self, event: Event) -> None:
        if not self._connected:
            raise RuntimeError("Transport not connected")

        channel = f"{self._prefix}:{event.topic}"
        data = event.serialize()

        # Publish to channel
        await self._redis.publish(channel, data)

        # Also add to stream for persistence
        stream_key = f"{self._prefix}:stream:{event.topic}"
        await self._redis.xadd(
            stream_key,
            {"event": data},
            maxlen=self._stream_max_len
        )

    async def _listen(self) -> None:
        """Listen for messages"""
        try:
            while self._connected:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )
                if message and message["type"] == "message":
                    channel = message["channel"]
                    if isinstance(channel, bytes):
                        channel = channel.decode("utf-8")

                    data = message["data"]
                    event = Event.deserialize(data)

                    # Find matching handler
                    for sub_channel, handler in self._subscriptions.items():
                        if channel.endswith(sub_channel):
                            asyncio.create_task(handler(event))

        except asyncio.CancelledError:
            pass

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], Awaitable[None]]
    ) -> str:
        channel = f"{self._prefix}:{topic}"
        await self._pubsub.subscribe(channel)
        sub_id = f"redis-{topic}-{uuid4().hex[:8]}"
        self._subscriptions[topic] = handler
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        # Extract topic from subscription ID
        parts = subscription_id.split("-")
        if len(parts) >= 2:
            topic = parts[1]
            channel = f"{self._prefix}:{topic}"
            await self._pubsub.unsubscribe(channel)
            self._subscriptions.pop(topic, None)

    async def acknowledge(self, event: Event) -> None:
        pass  # Handled by Redis Streams consumer groups

    async def reject(self, event: Event, requeue: bool = True) -> None:
        if requeue:
            await self.publish(event)

    @property
    def is_connected(self) -> bool:
        return self._connected


# =============================================================================
# DEAD LETTER QUEUE
# =============================================================================


class DeadLetterQueue:
    """Handle failed events that cannot be processed"""

    def __init__(self, max_size: int = 10000):
        self._queue: Deque[Tuple[Event, Exception, datetime]] = deque(maxlen=max_size)
        self._logger = structlog.get_logger("event_bus.dlq")

    def add(self, event: Event, error: Exception) -> None:
        """Add an event to the dead letter queue"""
        self._queue.append((event, error, datetime.utcnow()))
        self._logger.warning(
            "event_dead_lettered",
            event_id=event.id,
            event_type=event.type,
            error=str(error)
        )

    def get_all(self) -> List[Tuple[Event, Exception, datetime]]:
        """Get all dead letters"""
        return list(self._queue)

    def get_by_type(self, event_type: str) -> List[Tuple[Event, Exception, datetime]]:
        """Get dead letters by event type"""
        return [
            (e, err, ts) for e, err, ts in self._queue
            if e.type == event_type
        ]

    def replay(self, event_id: str) -> Optional[Event]:
        """Remove and return an event for replay"""
        for i, (event, _, _) in enumerate(self._queue):
            if event.id == event_id:
                self._queue.remove((event, _, _))
                return event
        return None

    def clear(self) -> int:
        """Clear all dead letters, return count"""
        count = len(self._queue)
        self._queue.clear()
        return count

    @property
    def size(self) -> int:
        return len(self._queue)


# =============================================================================
# EVENT BUS
# =============================================================================


class EventBus:
    """
    Distributed Event Bus

    High-performance event bus for real-time event propagation across
    all platform services.

    Features:
    - Multiple transport backends
    - Event filtering and routing
    - Priority queues
    - Dead letter handling
    - Event persistence and replay
    - Async event processing
    - Metrics and monitoring

    Usage:
        bus = EventBus()
        await bus.start()

        # Subscribe to events
        @bus.on("call.initiated")
        async def handle_call(event: Event):
            print(f"Call initiated: {event.data}")

        # Publish events
        await bus.publish(Event(
            type="call.initiated",
            source="telephony-gateway",
            data={"call_id": "123"}
        ))
    """

    def __init__(
        self,
        transport: Optional[EventTransport] = None,
        enable_history: bool = True,
        history_size: int = 10000,
        max_concurrent_handlers: int = 100,
        handler_timeout: float = 30.0,
        enable_metrics: bool = True
    ):
        self._transport = transport or InMemoryTransport()
        self._subscribers: Dict[str, EventSubscriber] = {}
        self._topic_subscribers: Dict[str, List[str]] = defaultdict(list)
        self._dlq = DeadLetterQueue()
        self._running = False
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger("event_bus")

        # Configuration
        self._enable_history = enable_history
        self._history_size = history_size
        self._max_concurrent_handlers = max_concurrent_handlers
        self._handler_timeout = handler_timeout
        self._enable_metrics = enable_metrics

        # Processing
        self._handler_semaphore = asyncio.Semaphore(max_concurrent_handlers)
        self._processing_tasks: Set[asyncio.Task] = set()

        # History
        self._history: Deque[Event] = deque(maxlen=history_size)

        # Metrics
        self._metrics = {
            "events_published": 0,
            "events_delivered": 0,
            "events_failed": 0,
            "events_dead_lettered": 0,
            "handlers_invoked": 0,
            "handlers_failed": 0,
            "avg_processing_time_ms": 0.0,
            "total_processing_time_ms": 0.0
        }

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the event bus"""
        if self._running:
            return

        await self._transport.connect()
        self._running = True
        self._logger.info("event_bus_started")

    async def stop(self) -> None:
        """Stop the event bus"""
        if not self._running:
            return

        # Cancel all processing tasks
        for task in self._processing_tasks:
            task.cancel()

        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)

        await self._transport.disconnect()
        self._running = False
        self._logger.info("event_bus_stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    async def publish(
        self,
        event: Event,
        wait_for_delivery: bool = False
    ) -> None:
        """
        Publish an event.

        Args:
            event: The event to publish
            wait_for_delivery: Whether to wait for all handlers to complete
        """
        if not self._running:
            raise RuntimeError("Event bus is not running")

        # Check if expired
        if event.is_expired:
            self._logger.warning("event_expired", event_id=event.id)
            return

        # Record in history
        if self._enable_history:
            self._history.append(event)

        # Update metrics
        self._metrics["events_published"] += 1

        # Publish to transport
        await self._transport.publish(event)

        # Deliver to local subscribers
        if wait_for_delivery:
            await self._deliver_event_sync(event)
        else:
            task = asyncio.create_task(self._deliver_event(event))
            self._processing_tasks.add(task)
            task.add_done_callback(self._processing_tasks.discard)

    async def publish_batch(
        self,
        events: List[Event],
        wait_for_delivery: bool = False
    ) -> None:
        """Publish multiple events"""
        tasks = [
            self.publish(event, wait_for_delivery=False)
            for event in events
        ]
        await asyncio.gather(*tasks)

        if wait_for_delivery:
            # Wait for all delivery tasks
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)

    async def emit(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Event:
        """
        Convenience method to emit an event.

        Args:
            event_type: Type of event
            source: Source of event
            data: Event data
            **kwargs: Additional event parameters

        Returns:
            The created event
        """
        event = Event(
            type=event_type,
            source=source,
            data=data,
            priority=kwargs.get("priority", EventPriority.NORMAL),
            topic=kwargs.get("topic", "default"),
            tags=kwargs.get("tags", set()),
            metadata=EventMetadata(
                correlation_id=kwargs.get("correlation_id"),
                trace_id=kwargs.get("trace_id"),
                user_id=kwargs.get("user_id"),
                organization_id=kwargs.get("organization_id"),
                session_id=kwargs.get("session_id"),
                agent_id=kwargs.get("agent_id"),
                call_id=kwargs.get("call_id"),
                conversation_id=kwargs.get("conversation_id")
            )
        )

        await self.publish(event)
        return event

    # -------------------------------------------------------------------------
    # Delivery
    # -------------------------------------------------------------------------

    async def _deliver_event(self, event: Event) -> None:
        """Deliver event to all matching subscribers (async)"""
        try:
            await self._deliver_event_sync(event)
        except Exception as e:
            self._logger.error(
                "event_delivery_error",
                event_id=event.id,
                error=str(e)
            )

    async def _deliver_event_sync(self, event: Event) -> None:
        """Deliver event to all matching subscribers (sync)"""
        # Get subscribers for this topic
        subscriber_ids = set()
        subscriber_ids.update(self._topic_subscribers.get(event.topic, []))
        subscriber_ids.update(self._topic_subscribers.get("*", []))  # Wildcard

        delivered = False
        for sub_id in subscriber_ids:
            subscriber = self._subscribers.get(sub_id)
            if not subscriber or not subscriber.active:
                continue

            if not subscriber.filter.matches(event):
                continue

            delivered = True
            await self._invoke_handler(subscriber, event)

        if delivered:
            self._metrics["events_delivered"] += 1

    async def _invoke_handler(
        self,
        subscriber: EventSubscriber,
        event: Event
    ) -> None:
        """Invoke a subscriber's handler"""
        start_time = time.time()

        try:
            async with self._handler_semaphore:
                await asyncio.wait_for(
                    subscriber.handler(event),
                    timeout=self._handler_timeout
                )

            subscriber.events_processed += 1
            self._metrics["handlers_invoked"] += 1

            # Update timing metrics
            duration_ms = (time.time() - start_time) * 1000
            self._metrics["total_processing_time_ms"] += duration_ms
            self._metrics["avg_processing_time_ms"] = (
                self._metrics["total_processing_time_ms"] /
                self._metrics["handlers_invoked"]
            )

        except asyncio.TimeoutError:
            subscriber.events_failed += 1
            self._metrics["handlers_failed"] += 1
            self._logger.error(
                "handler_timeout",
                subscriber=subscriber.name,
                event_id=event.id
            )

            if subscriber.retry_on_error and event.metadata.retry_count < event.metadata.max_retries:
                event.metadata.retry_count += 1
                await self.publish(event)
            else:
                self._dlq.add(event, TimeoutError("Handler timeout"))
                self._metrics["events_dead_lettered"] += 1

        except Exception as e:
            subscriber.events_failed += 1
            self._metrics["handlers_failed"] += 1
            self._logger.error(
                "handler_error",
                subscriber=subscriber.name,
                event_id=event.id,
                error=str(e)
            )

            # Call error handler if provided
            if subscriber.error_handler:
                try:
                    await subscriber.error_handler(event, e)
                except Exception:
                    pass

            # Retry or dead letter
            if subscriber.retry_on_error and event.metadata.retry_count < event.metadata.max_retries:
                event.metadata.retry_count += 1
                await self.publish(event)
            else:
                self._dlq.add(event, e)
                self._metrics["events_dead_lettered"] += 1

    # -------------------------------------------------------------------------
    # Subscription
    # -------------------------------------------------------------------------

    def subscribe(
        self,
        topic: str = "*",
        event_types: Optional[Set[str]] = None,
        handler: Optional[EventHandler] = None,
        filter: Optional[EventFilter] = None,
        name: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """
        Subscribe to events.

        Args:
            topic: Topic to subscribe to (* for all)
            event_types: Set of event types to receive
            handler: Handler function
            filter: Event filter
            name: Subscriber name
            priority: Handler priority (lower = higher priority)

        Returns:
            Subscription ID
        """
        subscriber = EventSubscriber(
            name=name or f"subscriber-{uuid4().hex[:8]}",
            handler=handler,
            filter=filter or EventFilter(event_types=event_types),
            priority=priority
        )

        self._subscribers[subscriber.id] = subscriber
        self._topic_subscribers[topic].append(subscriber.id)

        self._logger.info(
            "subscriber_added",
            subscriber_id=subscriber.id,
            name=subscriber.name,
            topic=topic
        )

        return subscriber.id

    def on(
        self,
        event_type: str,
        topic: str = "*"
    ) -> Callable[[EventHandler], EventHandler]:
        """
        Decorator to subscribe to an event type.

        Usage:
            @bus.on("call.initiated")
            async def handle_call(event: Event):
                print(event.data)
        """
        def decorator(handler: EventHandler) -> EventHandler:
            self.subscribe(
                topic=topic,
                event_types={event_type},
                handler=handler,
                name=handler.__name__
            )
            return handler
        return decorator

    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events"""
        if subscription_id in self._subscribers:
            del self._subscribers[subscription_id]

            # Remove from topic mapping
            for topic, subs in self._topic_subscribers.items():
                if subscription_id in subs:
                    subs.remove(subscription_id)

            self._logger.info(
                "subscriber_removed",
                subscriber_id=subscription_id
            )

    # -------------------------------------------------------------------------
    # History & Replay
    # -------------------------------------------------------------------------

    def get_history(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[Event]:
        """Get event history"""
        events = list(self._history)

        if event_type:
            events = [e for e in events if e.type == event_type]

        if source:
            events = [e for e in events if e.source == source]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    async def replay_event(self, event_id: str) -> bool:
        """Replay an event from history or DLQ"""
        # Check DLQ first
        event = self._dlq.replay(event_id)
        if event:
            await self.publish(event)
            return True

        # Check history
        for e in self._history:
            if e.id == event_id:
                new_metadata = EventMetadata(**vars(e.metadata))
                new_metadata.retry_count = 0
                replay_event = Event(
                    type=e.type,
                    source=e.source,
                    data=e.data.copy(),
                    metadata=new_metadata,
                    priority=e.priority,
                    topic=e.topic,
                    tags=e.tags.copy()
                )
                await self.publish(replay_event)
                return True

        return False

    # -------------------------------------------------------------------------
    # Dead Letter Queue
    # -------------------------------------------------------------------------

    def get_dead_letters(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Tuple[Event, Exception, datetime]]:
        """Get events from dead letter queue"""
        if event_type:
            return self._dlq.get_by_type(event_type)[:limit]
        return self._dlq.get_all()[:limit]

    async def replay_dead_letters(
        self,
        event_type: Optional[str] = None
    ) -> int:
        """Replay all dead letters of a type"""
        letters = self.get_dead_letters(event_type=event_type)
        count = 0
        for event, _, _ in letters:
            if await self.replay_event(event.id):
                count += 1
        return count

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics"""
        return {
            **self._metrics,
            "subscribers_total": len(self._subscribers),
            "subscribers_active": sum(
                1 for s in self._subscribers.values() if s.active
            ),
            "topics": list(self._topic_subscribers.keys()),
            "history_size": len(self._history),
            "dlq_size": self._dlq.size,
            "processing_tasks": len(self._processing_tasks)
        }

    def get_subscriber_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get per-subscriber statistics"""
        return {
            sub.id: {
                "name": sub.name,
                "active": sub.active,
                "events_processed": sub.events_processed,
                "events_failed": sub.events_failed,
                "success_rate": (
                    sub.events_processed / (sub.events_processed + sub.events_failed)
                    if (sub.events_processed + sub.events_failed) > 0 else 1.0
                )
            }
            for sub in self._subscribers.values()
        }


# =============================================================================
# EVENT BUILDER
# =============================================================================


class EventBuilder:
    """Fluent builder for creating events"""

    def __init__(self, event_type: str, source: str):
        self._type = event_type
        self._source = source
        self._data: Dict[str, Any] = {}
        self._metadata = EventMetadata()
        self._priority = EventPriority.NORMAL
        self._topic = "default"
        self._tags: Set[str] = set()

    def with_data(self, **kwargs) -> "EventBuilder":
        self._data.update(kwargs)
        return self

    def with_correlation(self, correlation_id: str) -> "EventBuilder":
        self._metadata.correlation_id = correlation_id
        return self

    def with_causation(self, causation_id: str) -> "EventBuilder":
        self._metadata.causation_id = causation_id
        return self

    def with_trace(self, trace_id: str, span_id: Optional[str] = None) -> "EventBuilder":
        self._metadata.trace_id = trace_id
        self._metadata.span_id = span_id
        return self

    def with_context(
        self,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        call_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> "EventBuilder":
        if user_id:
            self._metadata.user_id = user_id
        if organization_id:
            self._metadata.organization_id = organization_id
        if session_id:
            self._metadata.session_id = session_id
        if agent_id:
            self._metadata.agent_id = agent_id
        if call_id:
            self._metadata.call_id = call_id
        if conversation_id:
            self._metadata.conversation_id = conversation_id
        return self

    def with_priority(self, priority: EventPriority) -> "EventBuilder":
        self._priority = priority
        return self

    def with_topic(self, topic: str) -> "EventBuilder":
        self._topic = topic
        return self

    def with_tags(self, *tags: str) -> "EventBuilder":
        self._tags.update(tags)
        return self

    def with_ttl(self, seconds: int) -> "EventBuilder":
        self._metadata.ttl_seconds = seconds
        self._metadata.expires_at = datetime.utcnow() + timedelta(seconds=seconds)
        return self

    def build(self) -> Event:
        return Event(
            type=self._type,
            source=self._source,
            data=self._data,
            metadata=self._metadata,
            priority=self._priority,
            topic=self._topic,
            tags=self._tags
        )


def event(event_type: str, source: str) -> EventBuilder:
    """Create an event builder"""
    return EventBuilder(event_type, source)
