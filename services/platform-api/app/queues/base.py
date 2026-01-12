"""
Base Queue Implementation

Core queue abstractions:
- Message model
- Queue interface
- Basic operations
"""

from typing import Optional, Dict, Any, List, TypeVar, Generic, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MessagePriority(int, Enum):
    """Message priority levels."""
    CRITICAL = 0  # Highest
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4  # Lowest


class MessageStatus(str, Enum):
    """Message lifecycle status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD = "dead"
    DELAYED = "delayed"
    CANCELLED = "cancelled"


@dataclass
class Message:
    """Queue message."""
    id: str
    payload: Any
    queue_name: str
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Retry handling
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    next_retry_at: Optional[datetime] = None

    # Metadata
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Tenant isolation
    tenant_id: Optional[str] = None

    # Correlation
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        payload: Any,
        queue_name: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        tenant_id: Optional[str] = None,
        **kwargs,
    ) -> "Message":
        """Create a new message."""
        return cls(
            id=str(uuid.uuid4()),
            payload=payload,
            queue_name=queue_name,
            priority=priority,
            tenant_id=tenant_id,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "payload": self.payload,
            "queue_name": self.queue_name,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "last_error": self.last_error,
            "headers": self.headers,
            "metadata": self.metadata,
            "tenant_id": self.tenant_id,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        data = dict(data)
        data["priority"] = MessagePriority(data.get("priority", 2))
        data["status"] = MessageStatus(data.get("status", "pending"))
        data["created_at"] = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow()

        for field_name in ["scheduled_at", "started_at", "completed_at", "next_retry_at"]:
            if data.get(field_name):
                data[field_name] = datetime.fromisoformat(data[field_name])

        return cls(**data)

    def serialize(self) -> bytes:
        """Serialize message to bytes."""
        return json.dumps(self.to_dict()).encode()

    @classmethod
    def deserialize(cls, data: bytes) -> "Message":
        """Deserialize message from bytes."""
        return cls.from_dict(json.loads(data.decode()))

    def is_ready(self) -> bool:
        """Check if message is ready for processing."""
        if self.status != MessageStatus.PENDING and self.status != MessageStatus.RETRYING:
            return False
        if self.scheduled_at and self.scheduled_at > datetime.utcnow():
            return False
        if self.next_retry_at and self.next_retry_at > datetime.utcnow():
            return False
        return True

    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.attempts < self.max_attempts

    def mark_processing(self) -> None:
        """Mark message as processing."""
        self.status = MessageStatus.PROCESSING
        self.started_at = datetime.utcnow()
        self.attempts += 1

    def mark_completed(self) -> None:
        """Mark message as completed."""
        self.status = MessageStatus.COMPLETED
        self.completed_at = datetime.utcnow()

    def mark_failed(self, error: str) -> None:
        """Mark message as failed."""
        self.last_error = error
        if self.can_retry():
            self.status = MessageStatus.RETRYING
        else:
            self.status = MessageStatus.FAILED

    def mark_dead(self) -> None:
        """Mark message as dead (no more retries)."""
        self.status = MessageStatus.DEAD


@dataclass
class QueueConfig:
    """Queue configuration."""
    name: str
    max_size: int = 10000
    max_message_size: int = 1024 * 1024  # 1MB
    default_priority: MessagePriority = MessagePriority.NORMAL
    default_max_attempts: int = 3
    visibility_timeout_seconds: int = 30
    message_ttl_seconds: int = 86400  # 24 hours
    enable_dead_letter: bool = True
    dead_letter_queue: Optional[str] = None
    enable_metrics: bool = True
    enable_deduplication: bool = False
    deduplication_window_seconds: int = 300


@dataclass
class QueueStats:
    """Queue statistics."""
    queue_name: str
    total_messages: int = 0
    pending_messages: int = 0
    processing_messages: int = 0
    completed_messages: int = 0
    failed_messages: int = 0
    dead_messages: int = 0
    messages_per_second: float = 0.0
    avg_processing_time_ms: float = 0.0
    oldest_message_age_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "queue_name": self.queue_name,
            "total_messages": self.total_messages,
            "pending_messages": self.pending_messages,
            "processing_messages": self.processing_messages,
            "completed_messages": self.completed_messages,
            "failed_messages": self.failed_messages,
            "dead_messages": self.dead_messages,
            "messages_per_second": round(self.messages_per_second, 2),
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "oldest_message_age_seconds": round(self.oldest_message_age_seconds, 2),
        }


class Queue(ABC):
    """Abstract base queue interface."""

    def __init__(self, config: QueueConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    async def enqueue(self, message: Message) -> str:
        """Add message to queue."""
        pass

    @abstractmethod
    async def dequeue(self, count: int = 1) -> List[Message]:
        """Get messages from queue."""
        pass

    @abstractmethod
    async def ack(self, message_id: str) -> bool:
        """Acknowledge message completion."""
        pass

    @abstractmethod
    async def nack(self, message_id: str, error: Optional[str] = None) -> bool:
        """Negative acknowledge (return to queue)."""
        pass

    @abstractmethod
    async def size(self) -> int:
        """Get queue size."""
        pass

    @abstractmethod
    async def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        pass

    async def publish(
        self,
        payload: Any,
        priority: Optional[MessagePriority] = None,
        delay_seconds: int = 0,
        tenant_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Convenience method to publish message."""
        message = Message.create(
            payload=payload,
            queue_name=self.name,
            priority=priority or self.config.default_priority,
            max_attempts=self.config.default_max_attempts,
            tenant_id=tenant_id,
            **kwargs,
        )

        if delay_seconds > 0:
            message.scheduled_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
            message.status = MessageStatus.DELAYED

        return await self.enqueue(message)

    async def publish_batch(
        self,
        payloads: List[Any],
        priority: Optional[MessagePriority] = None,
        tenant_id: Optional[str] = None,
    ) -> List[str]:
        """Publish multiple messages."""
        message_ids = []
        for payload in payloads:
            msg_id = await self.publish(
                payload,
                priority=priority,
                tenant_id=tenant_id,
            )
            message_ids.append(msg_id)
        return message_ids


class InMemoryQueue(Queue):
    """
    In-memory queue implementation.

    Suitable for development and testing.
    """

    def __init__(self, config: QueueConfig):
        super().__init__(config)
        self._messages: Dict[str, Message] = {}
        self._pending: List[str] = []  # Message IDs in priority order
        self._processing: Dict[str, datetime] = {}  # message_id -> started_at
        self._stats = QueueStats(queue_name=config.name)
        self._lock = asyncio.Lock()
        self._message_times: List[float] = []  # Processing times

    async def enqueue(self, message: Message) -> str:
        """Add message to queue."""
        async with self._lock:
            # Check size limit
            if len(self._messages) >= self.config.max_size:
                raise QueueFullError(self.name, self.config.max_size)

            # Store message
            self._messages[message.id] = message

            # Add to pending in priority order
            self._insert_by_priority(message.id, message.priority)

            self._stats.total_messages += 1
            self._stats.pending_messages += 1

            logger.debug(f"Enqueued message {message.id} to {self.name}")
            return message.id

    def _insert_by_priority(self, message_id: str, priority: MessagePriority) -> None:
        """Insert message ID in priority order."""
        # Find insertion point (lower priority value = higher priority)
        insert_idx = len(self._pending)
        for i, mid in enumerate(self._pending):
            msg = self._messages.get(mid)
            if msg and msg.priority.value > priority.value:
                insert_idx = i
                break
        self._pending.insert(insert_idx, message_id)

    async def dequeue(self, count: int = 1) -> List[Message]:
        """Get messages from queue."""
        async with self._lock:
            messages = []
            now = datetime.utcnow()

            for message_id in list(self._pending):
                if len(messages) >= count:
                    break

                message = self._messages.get(message_id)
                if not message or not message.is_ready():
                    continue

                # Move to processing
                self._pending.remove(message_id)
                self._processing[message_id] = now

                message.mark_processing()
                messages.append(message)

                self._stats.pending_messages -= 1
                self._stats.processing_messages += 1

            return messages

    async def ack(self, message_id: str) -> bool:
        """Acknowledge message completion."""
        async with self._lock:
            if message_id not in self._messages:
                return False

            message = self._messages[message_id]
            message.mark_completed()

            # Record processing time
            if message_id in self._processing:
                started = self._processing.pop(message_id)
                duration = (datetime.utcnow() - started).total_seconds() * 1000
                self._message_times.append(duration)
                if len(self._message_times) > 1000:
                    self._message_times = self._message_times[-1000:]

            self._stats.processing_messages -= 1
            self._stats.completed_messages += 1

            # Remove from storage (or keep for history)
            del self._messages[message_id]

            return True

    async def nack(self, message_id: str, error: Optional[str] = None) -> bool:
        """Negative acknowledge (return to queue or dead letter)."""
        async with self._lock:
            if message_id not in self._messages:
                return False

            message = self._messages[message_id]
            message.mark_failed(error or "Unknown error")

            if message_id in self._processing:
                del self._processing[message_id]

            self._stats.processing_messages -= 1

            if message.can_retry():
                # Schedule retry
                message.next_retry_at = datetime.utcnow() + timedelta(
                    seconds=min(60 * (2 ** message.attempts), 3600)
                )
                self._insert_by_priority(message_id, message.priority)
                self._stats.pending_messages += 1
            else:
                # Move to dead letter
                message.mark_dead()
                self._stats.failed_messages += 1
                self._stats.dead_messages += 1

            return True

    async def size(self) -> int:
        """Get queue size."""
        return len(self._pending)

    async def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        self._stats.pending_messages = len(self._pending)
        self._stats.processing_messages = len(self._processing)

        if self._message_times:
            self._stats.avg_processing_time_ms = sum(self._message_times) / len(self._message_times)

        # Calculate oldest message age
        if self._pending:
            oldest_id = self._pending[0]
            oldest = self._messages.get(oldest_id)
            if oldest:
                self._stats.oldest_message_age_seconds = (
                    datetime.utcnow() - oldest.created_at
                ).total_seconds()

        return self._stats

    async def peek(self, count: int = 1) -> List[Message]:
        """Peek at messages without dequeuing."""
        messages = []
        for message_id in self._pending[:count]:
            message = self._messages.get(message_id)
            if message:
                messages.append(message)
        return messages

    async def purge(self) -> int:
        """Remove all messages from queue."""
        async with self._lock:
            count = len(self._messages)
            self._messages.clear()
            self._pending.clear()
            self._processing.clear()
            self._stats = QueueStats(queue_name=self.config.name)
            return count


class QueueFullError(Exception):
    """Raised when queue is full."""

    def __init__(self, queue_name: str, max_size: int):
        self.queue_name = queue_name
        self.max_size = max_size
        super().__init__(f"Queue {queue_name} is full (max: {max_size})")


class MessageNotFoundError(Exception):
    """Raised when message not found."""

    def __init__(self, message_id: str):
        self.message_id = message_id
        super().__init__(f"Message not found: {message_id}")


class QueueManager:
    """
    Manages multiple queues.

    Provides unified interface for queue operations.
    """

    def __init__(self):
        self._queues: Dict[str, Queue] = {}

    def create_queue(
        self,
        name: str,
        queue_class: type = InMemoryQueue,
        **kwargs,
    ) -> Queue:
        """Create and register a queue."""
        config = QueueConfig(name=name, **kwargs)
        queue = queue_class(config)
        self._queues[name] = queue
        logger.info(f"Created queue: {name}")
        return queue

    def get_queue(self, name: str) -> Optional[Queue]:
        """Get queue by name."""
        return self._queues.get(name)

    def list_queues(self) -> List[str]:
        """List all queue names."""
        return list(self._queues.keys())

    async def get_all_stats(self) -> Dict[str, QueueStats]:
        """Get stats for all queues."""
        stats = {}
        for name, queue in self._queues.items():
            stats[name] = await queue.get_stats()
        return stats

    async def publish(
        self,
        queue_name: str,
        payload: Any,
        **kwargs,
    ) -> str:
        """Publish message to queue."""
        queue = self.get_queue(queue_name)
        if not queue:
            raise ValueError(f"Queue not found: {queue_name}")
        return await queue.publish(payload, **kwargs)
