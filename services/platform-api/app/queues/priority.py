"""
Priority Queue Implementation

Advanced priority queue with:
- Multiple priority levels
- Fair scheduling
- Priority aging
- Tenant priority isolation
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import heapq
import asyncio
import logging

from app.queues.base import (
    Message,
    MessagePriority,
    MessageStatus,
    Queue,
    QueueConfig,
    QueueStats,
    QueueFullError,
)

logger = logging.getLogger(__name__)


class PriorityLevel(str, Enum):
    """Named priority levels."""
    REALTIME = "realtime"
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BULK = "bulk"


@dataclass
class PriorityQueueConfig(QueueConfig):
    """Configuration for priority queue."""
    enable_fair_scheduling: bool = True
    fair_scheduling_window: int = 100  # Consider last N messages
    enable_priority_aging: bool = True
    aging_interval_seconds: int = 60
    aging_priority_boost: int = 1  # Boost priority by this much
    tenant_priority_enabled: bool = True
    default_tenant_priority: int = 0  # 0 = normal, negative = lower, positive = higher


@dataclass
class PriorityItem:
    """Item in priority queue with comparison support."""
    priority_score: float
    sequence: int  # Tie-breaker for FIFO within priority
    message_id: str
    message: Message

    def __lt__(self, other):
        return (self.priority_score, self.sequence) < (other.priority_score, other.sequence)


class PriorityQueue(Queue):
    """
    Advanced priority queue implementation.

    Features:
    - Multiple priority levels
    - Fair scheduling across priorities
    - Priority aging for starvation prevention
    - Tenant-based priority adjustment
    """

    def __init__(self, config: PriorityQueueConfig):
        super().__init__(config)
        self.config: PriorityQueueConfig = config
        self._heap: List[PriorityItem] = []
        self._messages: Dict[str, Message] = {}
        self._processing: Dict[str, datetime] = {}
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._stats = QueueStats(queue_name=config.name)
        self._tenant_priorities: Dict[str, int] = {}
        self._priority_counts: Dict[MessagePriority, int] = {p: 0 for p in MessagePriority}
        self._processing_times: List[float] = []
        self._aging_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background tasks."""
        if self.config.enable_priority_aging:
            self._aging_task = asyncio.create_task(self._age_priorities())

    async def stop(self) -> None:
        """Stop background tasks."""
        if self._aging_task:
            self._aging_task.cancel()
            try:
                await self._aging_task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, message: Message) -> str:
        """Add message to priority queue."""
        async with self._lock:
            if len(self._messages) >= self.config.max_size:
                raise QueueFullError(self.name, self.config.max_size)

            # Calculate priority score
            priority_score = self._calculate_priority_score(message)

            # Create priority item
            self._sequence += 1
            item = PriorityItem(
                priority_score=priority_score,
                sequence=self._sequence,
                message_id=message.id,
                message=message,
            )

            # Add to heap and storage
            heapq.heappush(self._heap, item)
            self._messages[message.id] = message

            # Update stats
            self._stats.total_messages += 1
            self._stats.pending_messages += 1
            self._priority_counts[message.priority] += 1

            logger.debug(f"Enqueued {message.id} with priority score {priority_score}")
            return message.id

    def _calculate_priority_score(self, message: Message) -> float:
        """
        Calculate priority score for message.

        Lower score = higher priority.
        """
        # Base priority from message
        base_priority = message.priority.value

        # Tenant priority adjustment
        tenant_adjustment = 0
        if self.config.tenant_priority_enabled and message.tenant_id:
            tenant_adjustment = self._tenant_priorities.get(message.tenant_id, 0)

        # Scheduled time consideration
        time_factor = 0
        if message.scheduled_at:
            # Future messages get lower priority
            delay = (message.scheduled_at - datetime.utcnow()).total_seconds()
            if delay > 0:
                time_factor = min(delay / 3600, 10)  # Cap at 10 hours worth

        # Fair scheduling adjustment
        fair_adjustment = 0
        if self.config.enable_fair_scheduling:
            # Boost priority if this priority level is underrepresented
            total = sum(self._priority_counts.values()) or 1
            level_ratio = self._priority_counts[message.priority] / total
            if level_ratio < 0.1:  # Less than 10% of messages
                fair_adjustment = -0.5

        final_score = base_priority - tenant_adjustment + time_factor + fair_adjustment
        return final_score

    async def dequeue(self, count: int = 1) -> List[Message]:
        """Get highest priority messages."""
        async with self._lock:
            messages = []
            now = datetime.utcnow()
            skipped = []

            while len(messages) < count and self._heap:
                item = heapq.heappop(self._heap)
                message = item.message

                # Check if message is ready
                if not message.is_ready():
                    skipped.append(item)
                    continue

                # Check if still exists (not cancelled)
                if message.id not in self._messages:
                    continue

                # Move to processing
                self._processing[message.id] = now
                message.mark_processing()
                messages.append(message)

                self._stats.pending_messages -= 1
                self._stats.processing_messages += 1
                self._priority_counts[message.priority] -= 1

            # Re-add skipped items
            for item in skipped:
                heapq.heappush(self._heap, item)

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
                self._processing_times.append(duration)
                if len(self._processing_times) > 1000:
                    self._processing_times = self._processing_times[-1000:]

            self._stats.processing_messages -= 1
            self._stats.completed_messages += 1

            del self._messages[message_id]
            return True

    async def nack(self, message_id: str, error: Optional[str] = None) -> bool:
        """Return message to queue with retry logic."""
        async with self._lock:
            if message_id not in self._messages:
                return False

            message = self._messages[message_id]
            message.mark_failed(error or "Unknown error")

            if message_id in self._processing:
                del self._processing[message_id]

            self._stats.processing_messages -= 1

            if message.can_retry():
                # Schedule retry with exponential backoff
                backoff = min(60 * (2 ** message.attempts), 3600)
                message.next_retry_at = datetime.utcnow() + timedelta(seconds=backoff)
                message.status = MessageStatus.RETRYING

                # Re-add to queue with boosted priority (higher than before)
                boosted_priority = max(0, message.priority.value - 1)
                message.priority = MessagePriority(boosted_priority)

                self._sequence += 1
                item = PriorityItem(
                    priority_score=self._calculate_priority_score(message),
                    sequence=self._sequence,
                    message_id=message.id,
                    message=message,
                )
                heapq.heappush(self._heap, item)
                self._stats.pending_messages += 1
                self._priority_counts[message.priority] += 1
            else:
                message.mark_dead()
                self._stats.failed_messages += 1
                self._stats.dead_messages += 1

            return True

    async def size(self) -> int:
        """Get queue size."""
        return len(self._heap)

    async def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        self._stats.pending_messages = len(self._heap)
        self._stats.processing_messages = len(self._processing)

        if self._processing_times:
            self._stats.avg_processing_time_ms = (
                sum(self._processing_times) / len(self._processing_times)
            )

        return self._stats

    async def set_tenant_priority(
        self,
        tenant_id: str,
        priority_adjustment: int,
    ) -> None:
        """Set priority adjustment for tenant."""
        self._tenant_priorities[tenant_id] = priority_adjustment
        logger.info(f"Set tenant {tenant_id} priority adjustment to {priority_adjustment}")

    async def _age_priorities(self) -> None:
        """Background task to age message priorities."""
        while True:
            try:
                await asyncio.sleep(self.config.aging_interval_seconds)

                async with self._lock:
                    now = datetime.utcnow()
                    aged_items = []

                    # Check each message for aging
                    new_heap = []
                    while self._heap:
                        item = heapq.heappop(self._heap)
                        message = item.message

                        # Age messages that have been waiting too long
                        age = (now - message.created_at).total_seconds()
                        if age > self.config.aging_interval_seconds * 2:
                            # Boost priority
                            item.priority_score -= self.config.aging_priority_boost
                            aged_items.append(item.message_id)

                        heapq.heappush(new_heap, item)

                    self._heap = new_heap

                    if aged_items:
                        logger.debug(f"Aged {len(aged_items)} messages in {self.name}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Priority aging error: {e}")

    async def get_by_priority(
        self,
        priority: MessagePriority,
        limit: int = 10,
    ) -> List[Message]:
        """Get messages by specific priority."""
        async with self._lock:
            messages = []
            for item in self._heap:
                if item.message.priority == priority:
                    messages.append(item.message)
                    if len(messages) >= limit:
                        break
            return messages

    async def get_priority_distribution(self) -> Dict[str, int]:
        """Get distribution of messages by priority."""
        return {p.name: count for p, count in self._priority_counts.items()}

    async def requeue_with_priority(
        self,
        message_id: str,
        new_priority: MessagePriority,
    ) -> bool:
        """Change message priority."""
        async with self._lock:
            if message_id not in self._messages:
                return False

            message = self._messages[message_id]
            old_priority = message.priority

            # Update priority
            self._priority_counts[old_priority] -= 1
            message.priority = new_priority
            self._priority_counts[new_priority] += 1

            # Rebuild heap position
            self._sequence += 1
            new_heap = []
            for item in self._heap:
                if item.message_id == message_id:
                    item = PriorityItem(
                        priority_score=self._calculate_priority_score(message),
                        sequence=self._sequence,
                        message_id=message_id,
                        message=message,
                    )
                heapq.heappush(new_heap, item)

            self._heap = new_heap
            return True


class FairPriorityQueue(PriorityQueue):
    """
    Fair priority queue with weighted round-robin.

    Ensures all priority levels get some throughput.
    """

    def __init__(self, config: PriorityQueueConfig):
        super().__init__(config)
        self._priority_weights = {
            MessagePriority.CRITICAL: 5,
            MessagePriority.HIGH: 4,
            MessagePriority.NORMAL: 3,
            MessagePriority.LOW: 2,
            MessagePriority.BACKGROUND: 1,
        }
        self._priority_counters: Dict[MessagePriority, int] = {
            p: 0 for p in MessagePriority
        }
        self._round_robin_state: MessagePriority = MessagePriority.CRITICAL

    async def dequeue(self, count: int = 1) -> List[Message]:
        """Get messages using weighted fair scheduling."""
        async with self._lock:
            messages = []
            attempts = 0
            max_attempts = count * 5  # Prevent infinite loop

            while len(messages) < count and attempts < max_attempts:
                attempts += 1

                # Get message from current priority
                message = await self._dequeue_from_priority(self._round_robin_state)

                if message:
                    messages.append(message)
                    self._priority_counters[self._round_robin_state] += 1

                # Advance round robin
                self._advance_round_robin()

            return messages

    async def _dequeue_from_priority(
        self,
        priority: MessagePriority,
    ) -> Optional[Message]:
        """Dequeue single message from specific priority."""
        now = datetime.utcnow()

        for i, item in enumerate(self._heap):
            if item.message.priority == priority and item.message.is_ready():
                # Remove from heap (inefficient but correct)
                self._heap.pop(i)
                heapq.heapify(self._heap)

                message = item.message
                self._processing[message.id] = now
                message.mark_processing()

                self._stats.pending_messages -= 1
                self._stats.processing_messages += 1
                self._priority_counts[priority] -= 1

                return message

        return None

    def _advance_round_robin(self) -> None:
        """Advance to next priority in weighted round-robin."""
        current = self._round_robin_state
        weight = self._priority_weights[current]
        counter = self._priority_counters[current]

        # Check if we've served enough from this priority
        if counter >= weight:
            # Reset counter and move to next priority
            self._priority_counters[current] = 0

            # Find next priority with messages
            priorities = list(MessagePriority)
            current_idx = priorities.index(current)

            for i in range(1, len(priorities) + 1):
                next_idx = (current_idx + i) % len(priorities)
                next_priority = priorities[next_idx]
                if self._priority_counts.get(next_priority, 0) > 0:
                    self._round_robin_state = next_priority
                    return

            # No messages in any priority, stay on current
            self._round_robin_state = current


class TenantPriorityQueue:
    """
    Multi-tenant priority queue manager.

    Provides isolated priority queues per tenant with global ordering.
    """

    def __init__(
        self,
        base_config: PriorityQueueConfig,
    ):
        self.base_config = base_config
        self._tenant_queues: Dict[str, PriorityQueue] = {}
        self._global_queue: PriorityQueue = PriorityQueue(base_config)
        self._tenant_weights: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def enqueue(
        self,
        tenant_id: str,
        message: Message,
    ) -> str:
        """Enqueue message for tenant."""
        message.tenant_id = tenant_id
        return await self._global_queue.enqueue(message)

    async def dequeue(
        self,
        tenant_id: Optional[str] = None,
        count: int = 1,
    ) -> List[Message]:
        """
        Dequeue messages.

        If tenant_id provided, only get messages for that tenant.
        """
        if tenant_id:
            return await self._dequeue_for_tenant(tenant_id, count)
        return await self._global_queue.dequeue(count)

    async def _dequeue_for_tenant(
        self,
        tenant_id: str,
        count: int,
    ) -> List[Message]:
        """Dequeue messages for specific tenant."""
        messages = []

        async with self._lock:
            items_to_keep = []

            while len(messages) < count and self._global_queue._heap:
                item = heapq.heappop(self._global_queue._heap)

                if item.message.tenant_id == tenant_id and item.message.is_ready():
                    item.message.mark_processing()
                    self._global_queue._processing[item.message.id] = datetime.utcnow()
                    messages.append(item.message)
                else:
                    items_to_keep.append(item)

            # Re-add items for other tenants
            for item in items_to_keep:
                heapq.heappush(self._global_queue._heap, item)

        return messages

    async def set_tenant_weight(
        self,
        tenant_id: str,
        weight: float,
    ) -> None:
        """Set priority weight for tenant."""
        self._tenant_weights[tenant_id] = weight
        await self._global_queue.set_tenant_priority(
            tenant_id,
            int(weight * 10),  # Convert to integer adjustment
        )

    async def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get stats for specific tenant."""
        pending = 0
        processing = 0

        for item in self._global_queue._heap:
            if item.message.tenant_id == tenant_id:
                pending += 1

        for mid, _ in self._global_queue._processing.items():
            msg = self._global_queue._messages.get(mid)
            if msg and msg.tenant_id == tenant_id:
                processing += 1

        return {
            "tenant_id": tenant_id,
            "pending": pending,
            "processing": processing,
            "weight": self._tenant_weights.get(tenant_id, 1.0),
        }
