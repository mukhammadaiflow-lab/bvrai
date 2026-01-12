"""
Dead Letter Queue Implementation

Handle failed messages with:
- Failure tracking
- Retry policies
- Manual intervention
- Analysis tools
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging

from app.queues.base import (
    Message,
    MessagePriority,
    MessageStatus,
    Queue,
    QueueConfig,
    QueueStats,
)

logger = logging.getLogger(__name__)


class FailureReason(str, Enum):
    """Reasons for message failure."""
    MAX_RETRIES = "max_retries"
    TIMEOUT = "timeout"
    HANDLER_ERROR = "handler_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_ERROR = "resource_error"
    RATE_LIMITED = "rate_limited"
    POISON_MESSAGE = "poison_message"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class RetryStrategy(str, Enum):
    """Retry strategies."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"


@dataclass
class RetryPolicy:
    """Retry policy configuration."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 3600.0
    multiplier: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retryable_exceptions: List[str] = field(default_factory=lambda: [
        "ConnectionError", "TimeoutError", "ServiceUnavailableError"
    ])

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for attempt number."""
        if self.strategy == RetryStrategy.FIXED:
            delay = self.initial_delay_seconds
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.initial_delay_seconds * (self.multiplier ** attempt)
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.initial_delay_seconds * (1 + attempt)
        elif self.strategy == RetryStrategy.FIBONACCI:
            delay = self.initial_delay_seconds * self._fibonacci(attempt)
        else:
            delay = self.initial_delay_seconds

        delay = min(delay, self.max_delay_seconds)

        if self.jitter:
            import random
            jitter_range = delay * self.jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number."""
        if n <= 1:
            return 1
        a, b = 1, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b


@dataclass
class DeadMessage:
    """Dead letter message with failure context."""
    message: Message
    failure_reason: FailureReason
    failure_details: str
    original_queue: str
    failed_at: datetime = field(default_factory=datetime.utcnow)
    retry_history: List[Dict[str, Any]] = field(default_factory=list)
    can_be_reprocessed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message.to_dict(),
            "failure_reason": self.failure_reason.value,
            "failure_details": self.failure_details,
            "original_queue": self.original_queue,
            "failed_at": self.failed_at.isoformat(),
            "retry_history": self.retry_history,
            "can_be_reprocessed": self.can_be_reprocessed,
            "metadata": self.metadata,
        }


@dataclass
class DeadLetterConfig(QueueConfig):
    """Dead letter queue configuration."""
    retention_days: int = 30
    auto_cleanup: bool = True
    cleanup_interval_hours: int = 24
    max_retry_from_dlq: int = 2
    alert_threshold: int = 100
    alert_callback: Optional[Callable[[int], Awaitable[None]]] = None


class DeadLetterQueue:
    """
    Dead letter queue for failed messages.

    Stores messages that couldn't be processed after all retries.
    """

    def __init__(self, config: DeadLetterConfig):
        self.config = config
        self.name = config.name
        self._messages: Dict[str, DeadMessage] = {}
        self._by_reason: Dict[FailureReason, List[str]] = {r: [] for r in FailureReason}
        self._by_queue: Dict[str, List[str]] = {}
        self._by_tenant: Dict[str, List[str]] = {}
        self._stats = DeadLetterStats(queue_name=config.name)
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background tasks."""
        if self.config.auto_cleanup:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def add(
        self,
        message: Message,
        failure_reason: FailureReason,
        failure_details: str,
        original_queue: str,
    ) -> str:
        """Add message to dead letter queue."""
        async with self._lock:
            dead_message = DeadMessage(
                message=message,
                failure_reason=failure_reason,
                failure_details=failure_details,
                original_queue=original_queue,
                retry_history=[{
                    "attempt": message.attempts,
                    "error": message.last_error,
                    "timestamp": datetime.utcnow().isoformat(),
                }],
            )

            # Store message
            self._messages[message.id] = dead_message

            # Index by reason
            self._by_reason[failure_reason].append(message.id)

            # Index by original queue
            if original_queue not in self._by_queue:
                self._by_queue[original_queue] = []
            self._by_queue[original_queue].append(message.id)

            # Index by tenant
            if message.tenant_id:
                if message.tenant_id not in self._by_tenant:
                    self._by_tenant[message.tenant_id] = []
                self._by_tenant[message.tenant_id].append(message.id)

            # Update stats
            self._stats.total_messages += 1
            self._stats.by_reason[failure_reason.value] = (
                self._stats.by_reason.get(failure_reason.value, 0) + 1
            )

            # Check alert threshold
            if len(self._messages) >= self.config.alert_threshold:
                await self._trigger_alert()

            logger.info(f"Added message {message.id} to DLQ: {failure_reason.value}")
            return message.id

    async def get(self, message_id: str) -> Optional[DeadMessage]:
        """Get dead letter message by ID."""
        return self._messages.get(message_id)

    async def remove(self, message_id: str) -> bool:
        """Remove message from DLQ."""
        async with self._lock:
            if message_id not in self._messages:
                return False

            dead_msg = self._messages[message_id]

            # Remove from indexes
            self._by_reason[dead_msg.failure_reason].remove(message_id)

            if dead_msg.original_queue in self._by_queue:
                self._by_queue[dead_msg.original_queue].remove(message_id)

            if dead_msg.message.tenant_id in self._by_tenant:
                self._by_tenant[dead_msg.message.tenant_id].remove(message_id)

            del self._messages[message_id]
            return True

    async def list_messages(
        self,
        limit: int = 100,
        offset: int = 0,
        reason: Optional[FailureReason] = None,
        original_queue: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[DeadMessage]:
        """List dead letter messages with filtering."""
        message_ids = set(self._messages.keys())

        # Apply filters
        if reason:
            message_ids &= set(self._by_reason.get(reason, []))

        if original_queue:
            message_ids &= set(self._by_queue.get(original_queue, []))

        if tenant_id:
            message_ids &= set(self._by_tenant.get(tenant_id, []))

        # Sort by failed_at descending
        sorted_ids = sorted(
            message_ids,
            key=lambda mid: self._messages[mid].failed_at,
            reverse=True,
        )

        # Paginate
        paginated = sorted_ids[offset:offset + limit]

        return [self._messages[mid] for mid in paginated]

    async def requeue(
        self,
        message_id: str,
        target_queue: Queue,
        reset_attempts: bool = False,
    ) -> bool:
        """Requeue message to target queue."""
        dead_msg = await self.get(message_id)
        if not dead_msg:
            return False

        if not dead_msg.can_be_reprocessed:
            logger.warning(f"Message {message_id} cannot be reprocessed")
            return False

        # Check DLQ retry limit
        dlq_retries = dead_msg.metadata.get("dlq_retries", 0)
        if dlq_retries >= self.config.max_retry_from_dlq:
            dead_msg.can_be_reprocessed = False
            logger.warning(f"Message {message_id} exceeded DLQ retry limit")
            return False

        # Prepare message for requeue
        message = dead_msg.message
        if reset_attempts:
            message.attempts = 0
            message.last_error = None
        message.status = MessageStatus.PENDING
        message.next_retry_at = None

        # Add to target queue
        try:
            await target_queue.enqueue(message)

            # Update dead message metadata
            dead_msg.metadata["dlq_retries"] = dlq_retries + 1
            dead_msg.metadata["requeued_at"] = datetime.utcnow().isoformat()
            dead_msg.metadata["requeued_to"] = target_queue.name

            # Remove from DLQ
            await self.remove(message_id)

            logger.info(f"Requeued message {message_id} to {target_queue.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to requeue message {message_id}: {e}")
            return False

    async def requeue_all(
        self,
        target_queue: Queue,
        reason: Optional[FailureReason] = None,
        limit: int = 100,
    ) -> int:
        """Requeue multiple messages."""
        messages = await self.list_messages(limit=limit, reason=reason)
        requeued = 0

        for dead_msg in messages:
            if await self.requeue(dead_msg.message.id, target_queue):
                requeued += 1

        logger.info(f"Requeued {requeued} messages to {target_queue.name}")
        return requeued

    async def purge(
        self,
        reason: Optional[FailureReason] = None,
        older_than_days: Optional[int] = None,
    ) -> int:
        """Purge messages from DLQ."""
        async with self._lock:
            to_remove = []
            cutoff = None

            if older_than_days:
                cutoff = datetime.utcnow() - timedelta(days=older_than_days)

            for msg_id, dead_msg in self._messages.items():
                should_remove = False

                if reason and dead_msg.failure_reason == reason:
                    should_remove = True
                elif cutoff and dead_msg.failed_at < cutoff:
                    should_remove = True
                elif not reason and not older_than_days:
                    should_remove = True

                if should_remove:
                    to_remove.append(msg_id)

            for msg_id in to_remove:
                await self.remove(msg_id)

            logger.info(f"Purged {len(to_remove)} messages from DLQ")
            return len(to_remove)

    async def get_stats(self) -> "DeadLetterStats":
        """Get DLQ statistics."""
        self._stats.total_messages = len(self._messages)
        self._stats.by_reason = {
            r.value: len(ids) for r, ids in self._by_reason.items()
        }
        self._stats.by_queue = {
            q: len(ids) for q, ids in self._by_queue.items()
        }
        return self._stats

    async def _cleanup_loop(self) -> None:
        """Background cleanup of old messages."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_hours * 3600)
                await self.purge(older_than_days=self.config.retention_days)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DLQ cleanup error: {e}")

    async def _trigger_alert(self) -> None:
        """Trigger alert when threshold reached."""
        count = len(self._messages)
        logger.warning(f"DLQ alert threshold reached: {count} messages")

        if self.config.alert_callback:
            try:
                await self.config.alert_callback(count)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")


@dataclass
class DeadLetterStats:
    """DLQ statistics."""
    queue_name: str
    total_messages: int = 0
    by_reason: Dict[str, int] = field(default_factory=dict)
    by_queue: Dict[str, int] = field(default_factory=dict)
    oldest_message_age_hours: float = 0.0
    requeued_today: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "queue_name": self.queue_name,
            "total_messages": self.total_messages,
            "by_reason": self.by_reason,
            "by_queue": self.by_queue,
            "oldest_message_age_hours": round(self.oldest_message_age_hours, 2),
            "requeued_today": self.requeued_today,
        }


class DeadLetterManager:
    """
    Manages dead letter queues across the application.
    """

    def __init__(self):
        self._dlqs: Dict[str, DeadLetterQueue] = {}
        self._default_dlq: Optional[DeadLetterQueue] = None

    def create_dlq(
        self,
        name: str,
        config: Optional[DeadLetterConfig] = None,
    ) -> DeadLetterQueue:
        """Create a dead letter queue."""
        cfg = config or DeadLetterConfig(name=name)
        dlq = DeadLetterQueue(cfg)
        self._dlqs[name] = dlq

        if not self._default_dlq:
            self._default_dlq = dlq

        return dlq

    def get_dlq(self, name: str) -> Optional[DeadLetterQueue]:
        """Get DLQ by name."""
        return self._dlqs.get(name)

    def get_default_dlq(self) -> Optional[DeadLetterQueue]:
        """Get default DLQ."""
        return self._default_dlq

    async def handle_failure(
        self,
        message: Message,
        error: Exception,
        original_queue: str,
        dlq_name: Optional[str] = None,
    ) -> bool:
        """Handle message failure by routing to DLQ."""
        dlq = self._dlqs.get(dlq_name) if dlq_name else self._default_dlq

        if not dlq:
            logger.error("No DLQ available for failed message")
            return False

        # Determine failure reason
        reason = self._classify_error(error)

        await dlq.add(
            message=message,
            failure_reason=reason,
            failure_details=str(error),
            original_queue=original_queue,
        )

        return True

    def _classify_error(self, error: Exception) -> FailureReason:
        """Classify error into failure reason."""
        error_type = type(error).__name__

        if "Timeout" in error_type:
            return FailureReason.TIMEOUT
        elif "Validation" in error_type:
            return FailureReason.VALIDATION_ERROR
        elif "RateLimit" in error_type:
            return FailureReason.RATE_LIMITED
        elif "Connection" in error_type or "Resource" in error_type:
            return FailureReason.RESOURCE_ERROR
        else:
            return FailureReason.HANDLER_ERROR

    async def get_all_stats(self) -> Dict[str, DeadLetterStats]:
        """Get stats for all DLQs."""
        stats = {}
        for name, dlq in self._dlqs.items():
            stats[name] = await dlq.get_stats()
        return stats

    async def start_all(self) -> None:
        """Start all DLQs."""
        for dlq in self._dlqs.values():
            await dlq.start()

    async def stop_all(self) -> None:
        """Stop all DLQs."""
        for dlq in self._dlqs.values():
            await dlq.stop()
