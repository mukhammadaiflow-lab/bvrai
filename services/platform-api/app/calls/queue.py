"""
Call Queue Management

Queue management for calls:
- Priority queuing
- Wait time management
- Queue overflow handling
- Callback scheduling
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import heapq
import uuid

logger = logging.getLogger(__name__)


class QueuePriority(int, Enum):
    """Queue priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class QueueStatus(str, Enum):
    """Queue status."""
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"
    OVERFLOW = "overflow"


class QueuedCallStatus(str, Enum):
    """Status of a queued call."""
    WAITING = "waiting"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ABANDONED = "abandoned"
    TIMEOUT = "timeout"
    CALLBACK_SCHEDULED = "callback_scheduled"
    OVERFLOW = "overflow"


@dataclass
class QueueConfig:
    """Queue configuration."""
    name: str
    queue_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Capacity
    max_size: int = 100
    max_wait_seconds: int = 600  # 10 minutes

    # Behavior
    wrap_up_seconds: int = 30
    service_level_seconds: int = 20  # Answer 80% within 20 seconds
    service_level_target: float = 0.80

    # Overflow
    overflow_enabled: bool = True
    overflow_queue_id: Optional[str] = None
    overflow_threshold: int = 50

    # Callbacks
    callback_enabled: bool = True
    callback_min_wait_seconds: int = 60
    callback_max_per_hour: int = 50

    # Hours
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    timezone: str = "UTC"

    # Music/Messages
    hold_music_url: Optional[str] = None
    position_announcement_interval: int = 60  # seconds
    estimated_wait_enabled: bool = True


@dataclass(order=True)
class QueuedCall:
    """Represents a call in queue."""
    priority: int
    queued_at: datetime = field(compare=False)
    call_id: str = field(compare=False)
    queue_id: str = field(compare=False)

    # Caller info
    caller_number: str = field(compare=False, default="")
    caller_name: Optional[str] = field(compare=False, default=None)
    caller_language: str = field(compare=False, default="en")

    # Status
    status: QueuedCallStatus = field(compare=False, default=QueuedCallStatus.WAITING)
    position: int = field(compare=False, default=0)

    # Timing
    estimated_wait_seconds: int = field(compare=False, default=0)
    announced_position_at: Optional[datetime] = field(compare=False, default=None)

    # Callback
    callback_number: Optional[str] = field(compare=False, default=None)
    callback_scheduled_at: Optional[datetime] = field(compare=False, default=None)

    # Metadata
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)

    @property
    def wait_time_seconds(self) -> float:
        """Get current wait time in seconds."""
        return (datetime.utcnow() - self.queued_at).total_seconds()


@dataclass
class QueueMetrics:
    """Queue metrics."""
    queue_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Current state
    current_size: int = 0
    active_calls: int = 0
    available_agents: int = 0

    # Wait times
    avg_wait_seconds: float = 0.0
    max_wait_seconds: float = 0.0
    longest_waiting_call_id: Optional[str] = None

    # Service level
    service_level: float = 0.0
    calls_within_sl: int = 0
    calls_outside_sl: int = 0

    # Volumes
    calls_offered: int = 0
    calls_answered: int = 0
    calls_abandoned: int = 0
    calls_overflowed: int = 0
    callbacks_scheduled: int = 0

    # Rates
    abandon_rate: float = 0.0
    answer_rate: float = 0.0

    def calculate_rates(self) -> None:
        """Calculate rates from volumes."""
        total = self.calls_answered + self.calls_abandoned
        if total > 0:
            self.answer_rate = self.calls_answered / total
            self.abandon_rate = self.calls_abandoned / total

        sl_total = self.calls_within_sl + self.calls_outside_sl
        if sl_total > 0:
            self.service_level = self.calls_within_sl / sl_total


class CallQueue:
    """
    Call queue implementation.

    Features:
    - Priority-based queuing
    - Wait time estimation
    - Position announcements
    - Callback scheduling
    - Overflow handling
    """

    def __init__(self, config: QueueConfig):
        self.config = config
        self.queue_id = config.queue_id

        # Queue storage (min-heap by priority and time)
        self._queue: List[QueuedCall] = []
        self._calls_by_id: Dict[str, QueuedCall] = {}

        # Status
        self._status = QueueStatus.ACTIVE
        self._lock = asyncio.Lock()

        # Metrics
        self._metrics = QueueMetrics(queue_id=self.queue_id)
        self._wait_times: List[float] = []

        # Callbacks
        self._on_call_queued: List[Callable[[QueuedCall], Awaitable[None]]] = []
        self._on_call_dequeued: List[Callable[[QueuedCall], Awaitable[None]]] = []
        self._on_overflow: List[Callable[[QueuedCall], Awaitable[None]]] = []

    @property
    def status(self) -> QueueStatus:
        """Get queue status."""
        return self._status

    @property
    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return len(self._queue) >= self.config.max_size

    async def enqueue(
        self,
        call_id: str,
        caller_number: str,
        priority: QueuePriority = QueuePriority.NORMAL,
        caller_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[QueuedCall]:
        """Add call to queue."""
        async with self._lock:
            # Check if queue accepts calls
            if self._status == QueueStatus.CLOSED:
                logger.warning(f"Queue {self.queue_id} is closed")
                return None

            # Check capacity
            if self.is_full:
                if self.config.overflow_enabled:
                    return await self._handle_overflow(
                        call_id, caller_number, priority, caller_name, metadata
                    )
                logger.warning(f"Queue {self.queue_id} is full")
                return None

            # Create queued call
            call = QueuedCall(
                priority=priority.value,
                queued_at=datetime.utcnow(),
                call_id=call_id,
                queue_id=self.queue_id,
                caller_number=caller_number,
                caller_name=caller_name,
                metadata=metadata or {},
            )

            # Calculate estimated wait
            call.estimated_wait_seconds = self._estimate_wait_time()

            # Add to queue
            heapq.heappush(self._queue, call)
            self._calls_by_id[call_id] = call

            # Update positions
            self._update_positions()

            # Update metrics
            self._metrics.calls_offered += 1
            self._metrics.current_size = len(self._queue)

            logger.info(
                f"Call {call_id} queued in {self.queue_id} at position {call.position}"
            )

        # Trigger callbacks
        for callback in self._on_call_queued:
            try:
                await callback(call)
            except Exception as e:
                logger.error(f"Queue callback error: {e}")

        return call

    async def dequeue(self) -> Optional[QueuedCall]:
        """Get next call from queue."""
        async with self._lock:
            if not self._queue:
                return None

            call = heapq.heappop(self._queue)
            del self._calls_by_id[call.call_id]

            call.status = QueuedCallStatus.CONNECTING

            # Track wait time
            wait_time = call.wait_time_seconds
            self._wait_times.append(wait_time)
            if len(self._wait_times) > 100:
                self._wait_times = self._wait_times[-100:]

            # Update metrics
            self._metrics.calls_answered += 1
            self._metrics.current_size = len(self._queue)

            if wait_time <= self.config.service_level_seconds:
                self._metrics.calls_within_sl += 1
            else:
                self._metrics.calls_outside_sl += 1

            self._metrics.calculate_rates()

            # Update positions
            self._update_positions()

            logger.info(
                f"Call {call.call_id} dequeued from {self.queue_id} "
                f"after {wait_time:.1f}s"
            )

        # Trigger callbacks
        for callback in self._on_call_dequeued:
            try:
                await callback(call)
            except Exception as e:
                logger.error(f"Dequeue callback error: {e}")

        return call

    async def remove(self, call_id: str, status: QueuedCallStatus = QueuedCallStatus.ABANDONED) -> bool:
        """Remove call from queue."""
        async with self._lock:
            call = self._calls_by_id.pop(call_id, None)
            if not call:
                return False

            call.status = status

            # Rebuild queue without this call
            self._queue = [c for c in self._queue if c.call_id != call_id]
            heapq.heapify(self._queue)

            # Update metrics
            if status == QueuedCallStatus.ABANDONED:
                self._metrics.calls_abandoned += 1
            elif status == QueuedCallStatus.TIMEOUT:
                self._metrics.calls_abandoned += 1

            self._metrics.current_size = len(self._queue)
            self._metrics.calculate_rates()

            # Update positions
            self._update_positions()

            return True

    async def get_call(self, call_id: str) -> Optional[QueuedCall]:
        """Get call by ID."""
        return self._calls_by_id.get(call_id)

    async def get_position(self, call_id: str) -> int:
        """Get call position in queue (1-based)."""
        call = self._calls_by_id.get(call_id)
        return call.position if call else 0

    async def schedule_callback(
        self,
        call_id: str,
        callback_number: str,
        scheduled_at: Optional[datetime] = None,
    ) -> bool:
        """Schedule callback for queued call."""
        async with self._lock:
            call = self._calls_by_id.get(call_id)
            if not call:
                return False

            call.callback_number = callback_number
            call.callback_scheduled_at = scheduled_at or (
                datetime.utcnow() + timedelta(minutes=5)
            )
            call.status = QueuedCallStatus.CALLBACK_SCHEDULED

            self._metrics.callbacks_scheduled += 1

            logger.info(
                f"Callback scheduled for {call_id} at {call.callback_scheduled_at}"
            )

            return True

    async def pause(self) -> None:
        """Pause queue (stop accepting new calls)."""
        self._status = QueueStatus.PAUSED

    async def resume(self) -> None:
        """Resume queue."""
        self._status = QueueStatus.ACTIVE

    async def close(self) -> None:
        """Close queue."""
        self._status = QueueStatus.CLOSED

    async def _handle_overflow(
        self,
        call_id: str,
        caller_number: str,
        priority: QueuePriority,
        caller_name: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[QueuedCall]:
        """Handle queue overflow."""
        call = QueuedCall(
            priority=priority.value,
            queued_at=datetime.utcnow(),
            call_id=call_id,
            queue_id=self.queue_id,
            caller_number=caller_number,
            caller_name=caller_name,
            status=QueuedCallStatus.OVERFLOW,
            metadata=metadata or {},
        )

        self._metrics.calls_overflowed += 1

        # Trigger overflow callbacks
        for callback in self._on_overflow:
            try:
                await callback(call)
            except Exception as e:
                logger.error(f"Overflow callback error: {e}")

        return call

    def _update_positions(self) -> None:
        """Update position for all calls in queue."""
        sorted_calls = sorted(self._queue)
        for i, call in enumerate(sorted_calls):
            call.position = i + 1

    def _estimate_wait_time(self) -> int:
        """Estimate wait time for new call."""
        if not self._wait_times:
            return 60  # Default 1 minute

        avg_wait = sum(self._wait_times) / len(self._wait_times)
        position = len(self._queue) + 1

        # Simple estimation based on average wait and position
        return int(avg_wait * position / max(1, self._metrics.available_agents))

    def get_metrics(self) -> QueueMetrics:
        """Get queue metrics."""
        self._metrics.current_size = len(self._queue)

        if self._wait_times:
            self._metrics.avg_wait_seconds = sum(self._wait_times) / len(self._wait_times)
            self._metrics.max_wait_seconds = max(self._wait_times)

        # Find longest waiting call
        if self._queue:
            longest = max(self._queue, key=lambda c: c.wait_time_seconds)
            self._metrics.longest_waiting_call_id = longest.call_id

        self._metrics.timestamp = datetime.utcnow()
        return self._metrics

    def on_call_queued(self, callback: Callable[[QueuedCall], Awaitable[None]]) -> None:
        """Register callback for call queued event."""
        self._on_call_queued.append(callback)

    def on_call_dequeued(self, callback: Callable[[QueuedCall], Awaitable[None]]) -> None:
        """Register callback for call dequeued event."""
        self._on_call_dequeued.append(callback)

    def on_overflow(self, callback: Callable[[QueuedCall], Awaitable[None]]) -> None:
        """Register callback for overflow event."""
        self._on_overflow.append(callback)


class QueueManager:
    """
    Manages multiple call queues.
    """

    def __init__(self):
        self._queues: Dict[str, CallQueue] = {}
        self._lock = asyncio.Lock()

        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def create_queue(self, config: QueueConfig) -> CallQueue:
        """Create new queue."""
        async with self._lock:
            if config.queue_id in self._queues:
                return self._queues[config.queue_id]

            queue = CallQueue(config)
            self._queues[config.queue_id] = queue
            return queue

    async def get_queue(self, queue_id: str) -> Optional[CallQueue]:
        """Get queue by ID."""
        return self._queues.get(queue_id)

    async def delete_queue(self, queue_id: str) -> bool:
        """Delete queue."""
        async with self._lock:
            return self._queues.pop(queue_id, None) is not None

    async def enqueue(
        self,
        queue_id: str,
        call_id: str,
        caller_number: str,
        priority: QueuePriority = QueuePriority.NORMAL,
        **kwargs,
    ) -> Optional[QueuedCall]:
        """Enqueue call to specific queue."""
        queue = await self.get_queue(queue_id)
        if not queue:
            logger.warning(f"Queue not found: {queue_id}")
            return None

        return await queue.enqueue(
            call_id=call_id,
            caller_number=caller_number,
            priority=priority,
            **kwargs,
        )

    async def dequeue(self, queue_id: str) -> Optional[QueuedCall]:
        """Dequeue call from specific queue."""
        queue = await self.get_queue(queue_id)
        if not queue:
            return None

        return await queue.dequeue()

    async def find_call(self, call_id: str) -> Optional[tuple]:
        """Find call across all queues."""
        for queue_id, queue in self._queues.items():
            call = await queue.get_call(call_id)
            if call:
                return (queue_id, call)
        return None

    async def start(self) -> None:
        """Start queue manager background tasks."""
        if self._running:
            return

        self._running = True
        self._tasks.append(asyncio.create_task(self._timeout_loop()))
        self._tasks.append(asyncio.create_task(self._metrics_loop()))

    async def stop(self) -> None:
        """Stop queue manager."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

    async def _timeout_loop(self) -> None:
        """Check for timed out calls."""
        while self._running:
            try:
                await asyncio.sleep(10)

                now = datetime.utcnow()

                for queue in self._queues.values():
                    # Check each call
                    timed_out = []
                    for call in list(queue._calls_by_id.values()):
                        if call.wait_time_seconds > queue.config.max_wait_seconds:
                            timed_out.append(call.call_id)

                    for call_id in timed_out:
                        await queue.remove(call_id, QueuedCallStatus.TIMEOUT)
                        logger.info(f"Call {call_id} timed out in queue")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Timeout loop error: {e}")

    async def _metrics_loop(self) -> None:
        """Collect and log metrics."""
        while self._running:
            try:
                await asyncio.sleep(30)

                for queue_id, queue in self._queues.items():
                    metrics = queue.get_metrics()
                    logger.debug(
                        f"Queue {queue_id}: size={metrics.current_size}, "
                        f"avg_wait={metrics.avg_wait_seconds:.1f}s, "
                        f"sl={metrics.service_level:.1%}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")

    def get_all_metrics(self) -> Dict[str, QueueMetrics]:
        """Get metrics for all queues."""
        return {
            queue_id: queue.get_metrics()
            for queue_id, queue in self._queues.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        total_calls = sum(q.size for q in self._queues.values())
        total_offered = sum(q._metrics.calls_offered for q in self._queues.values())
        total_answered = sum(q._metrics.calls_answered for q in self._queues.values())

        return {
            "queue_count": len(self._queues),
            "total_calls_waiting": total_calls,
            "total_calls_offered": total_offered,
            "total_calls_answered": total_answered,
            "overall_answer_rate": total_answered / max(1, total_offered),
        }
