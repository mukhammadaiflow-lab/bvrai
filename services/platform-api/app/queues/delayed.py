"""
Delayed Queue Implementation

Schedule messages for future delivery:
- Scheduled messages
- Recurring tasks
- Cron-like scheduling
- Time-based routing
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import heapq
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


class ScheduleType(str, Enum):
    """Types of schedules."""
    ONCE = "once"
    RECURRING = "recurring"
    CRON = "cron"


@dataclass
class Schedule:
    """Schedule definition."""
    schedule_type: ScheduleType = ScheduleType.ONCE
    run_at: Optional[datetime] = None
    interval_seconds: Optional[int] = None
    cron_expression: Optional[str] = None
    timezone: str = "UTC"
    max_runs: Optional[int] = None
    end_at: Optional[datetime] = None

    def get_next_run(self, from_time: Optional[datetime] = None) -> Optional[datetime]:
        """Calculate next run time."""
        from_time = from_time or datetime.utcnow()

        if self.schedule_type == ScheduleType.ONCE:
            if self.run_at and self.run_at > from_time:
                return self.run_at
            return None

        elif self.schedule_type == ScheduleType.RECURRING:
            if not self.interval_seconds:
                return None
            next_run = from_time + timedelta(seconds=self.interval_seconds)
            if self.end_at and next_run > self.end_at:
                return None
            return next_run

        elif self.schedule_type == ScheduleType.CRON:
            return self._parse_cron(from_time)

        return None

    def _parse_cron(self, from_time: datetime) -> Optional[datetime]:
        """Parse cron expression and get next run time."""
        # Simplified cron parsing
        # Full implementation would use croniter library
        if not self.cron_expression:
            return None

        parts = self.cron_expression.split()
        if len(parts) != 5:
            return None

        # Simple implementation: just add 1 hour for demo
        return from_time + timedelta(hours=1)


@dataclass
class ScheduledMessage:
    """Message with scheduling information."""
    message: Message
    schedule: Schedule
    run_count: int = 0
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __lt__(self, other):
        """Compare by next run time for heap."""
        self_time = self.next_run or datetime.max
        other_time = other.next_run or datetime.max
        return self_time < other_time

    def calculate_next_run(self) -> None:
        """Calculate next run time."""
        self.next_run = self.schedule.get_next_run(self.last_run)

        # Check max runs
        if self.schedule.max_runs and self.run_count >= self.schedule.max_runs:
            self.next_run = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message.to_dict(),
            "schedule_type": self.schedule.schedule_type.value,
            "run_count": self.run_count,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class DelayedQueueConfig(QueueConfig):
    """Configuration for delayed queue."""
    poll_interval_seconds: float = 1.0
    batch_size: int = 100
    enable_recurring: bool = True
    max_schedule_ahead_days: int = 365


class DelayedQueue:
    """
    Delayed message queue.

    Stores messages for future delivery based on schedule.
    """

    def __init__(
        self,
        config: DelayedQueueConfig,
        target_queue: Optional[Queue] = None,
    ):
        self.config = config
        self.name = config.name
        self.target_queue = target_queue
        self._scheduled: List[ScheduledMessage] = []  # Min-heap by next_run
        self._messages: Dict[str, ScheduledMessage] = {}
        self._lock = asyncio.Lock()
        self._poll_task: Optional[asyncio.Task] = None
        self._stats = DelayedQueueStats(queue_name=config.name)

    async def start(self) -> None:
        """Start polling for due messages."""
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(f"Started delayed queue: {self.name}")

    async def stop(self) -> None:
        """Stop polling."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped delayed queue: {self.name}")

    async def schedule(
        self,
        message: Message,
        delay_seconds: Optional[int] = None,
        run_at: Optional[datetime] = None,
        interval_seconds: Optional[int] = None,
        cron_expression: Optional[str] = None,
        max_runs: Optional[int] = None,
    ) -> str:
        """Schedule a message for future delivery."""
        async with self._lock:
            # Determine schedule type
            if interval_seconds or cron_expression:
                schedule_type = ScheduleType.CRON if cron_expression else ScheduleType.RECURRING
            else:
                schedule_type = ScheduleType.ONCE

            # Calculate run_at
            if not run_at:
                if delay_seconds:
                    run_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
                else:
                    run_at = datetime.utcnow()

            # Validate schedule ahead limit
            max_ahead = timedelta(days=self.config.max_schedule_ahead_days)
            if run_at > datetime.utcnow() + max_ahead:
                raise ValueError(f"Cannot schedule more than {self.config.max_schedule_ahead_days} days ahead")

            # Create schedule
            schedule = Schedule(
                schedule_type=schedule_type,
                run_at=run_at,
                interval_seconds=interval_seconds,
                cron_expression=cron_expression,
                max_runs=max_runs,
            )

            # Create scheduled message
            scheduled = ScheduledMessage(
                message=message,
                schedule=schedule,
                next_run=run_at,
            )

            # Store and add to heap
            self._messages[message.id] = scheduled
            heapq.heappush(self._scheduled, scheduled)

            # Update message status
            message.status = MessageStatus.DELAYED
            message.scheduled_at = run_at

            self._stats.total_scheduled += 1
            self._stats.pending_scheduled += 1

            logger.debug(f"Scheduled message {message.id} for {run_at}")
            return message.id

    async def schedule_batch(
        self,
        messages: List[Message],
        delay_seconds: int,
    ) -> List[str]:
        """Schedule multiple messages with same delay."""
        message_ids = []
        for message in messages:
            msg_id = await self.schedule(message, delay_seconds=delay_seconds)
            message_ids.append(msg_id)
        return message_ids

    async def cancel(self, message_id: str) -> bool:
        """Cancel a scheduled message."""
        async with self._lock:
            if message_id not in self._messages:
                return False

            scheduled = self._messages[message_id]
            scheduled.message.status = MessageStatus.CANCELLED
            scheduled.next_run = None

            del self._messages[message_id]
            self._stats.pending_scheduled -= 1
            self._stats.cancelled += 1

            logger.info(f"Cancelled scheduled message: {message_id}")
            return True

    async def get_due_messages(self, limit: int = 100) -> List[ScheduledMessage]:
        """Get messages that are due for processing."""
        async with self._lock:
            now = datetime.utcnow()
            due_messages = []
            to_reschedule = []

            while self._scheduled and len(due_messages) < limit:
                scheduled = heapq.heappop(self._scheduled)

                if scheduled.message.id not in self._messages:
                    # Message was cancelled
                    continue

                if scheduled.next_run and scheduled.next_run <= now:
                    due_messages.append(scheduled)
                else:
                    # Not due yet, put back
                    heapq.heappush(self._scheduled, scheduled)
                    break

            return due_messages

    async def _poll_loop(self) -> None:
        """Poll for due messages and deliver them."""
        while True:
            try:
                await asyncio.sleep(self.config.poll_interval_seconds)

                due = await self.get_due_messages(self.config.batch_size)

                for scheduled in due:
                    await self._deliver(scheduled)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Delayed queue poll error: {e}")

    async def _deliver(self, scheduled: ScheduledMessage) -> None:
        """Deliver a scheduled message."""
        async with self._lock:
            message = scheduled.message
            message.status = MessageStatus.PENDING
            message.scheduled_at = None

            # Update run tracking
            scheduled.run_count += 1
            scheduled.last_run = datetime.utcnow()

            # Deliver to target queue
            if self.target_queue:
                await self.target_queue.enqueue(message)
                self._stats.delivered += 1
            else:
                # No target queue, just mark as ready
                logger.warning(f"No target queue for scheduled message {message.id}")

            # Handle recurring schedules
            if scheduled.schedule.schedule_type in (ScheduleType.RECURRING, ScheduleType.CRON):
                scheduled.calculate_next_run()

                if scheduled.next_run:
                    # Create new message for next run
                    new_message = Message.create(
                        payload=message.payload,
                        queue_name=message.queue_name,
                        priority=message.priority,
                        tenant_id=message.tenant_id,
                        headers=message.headers,
                        metadata=message.metadata,
                    )

                    new_scheduled = ScheduledMessage(
                        message=new_message,
                        schedule=scheduled.schedule,
                        run_count=scheduled.run_count,
                        last_run=scheduled.last_run,
                        next_run=scheduled.next_run,
                    )

                    self._messages[new_message.id] = new_scheduled
                    heapq.heappush(self._scheduled, new_scheduled)

                    logger.debug(f"Rescheduled recurring message, next run: {scheduled.next_run}")
                else:
                    self._stats.pending_scheduled -= 1
            else:
                # One-time message, remove from tracking
                del self._messages[message.id]
                self._stats.pending_scheduled -= 1

    async def get_pending(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ScheduledMessage]:
        """Get pending scheduled messages."""
        sorted_messages = sorted(
            self._messages.values(),
            key=lambda s: s.next_run or datetime.max,
        )
        return sorted_messages[offset:offset + limit]

    async def get_stats(self) -> "DelayedQueueStats":
        """Get queue statistics."""
        self._stats.pending_scheduled = len(self._messages)
        return self._stats


@dataclass
class DelayedQueueStats:
    """Delayed queue statistics."""
    queue_name: str
    total_scheduled: int = 0
    pending_scheduled: int = 0
    delivered: int = 0
    cancelled: int = 0
    recurring_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "queue_name": self.queue_name,
            "total_scheduled": self.total_scheduled,
            "pending_scheduled": self.pending_scheduled,
            "delivered": self.delivered,
            "cancelled": self.cancelled,
            "recurring_count": self.recurring_count,
        }


class TaskScheduler:
    """
    High-level task scheduler.

    Provides convenient API for scheduling tasks.
    """

    def __init__(
        self,
        delayed_queue: DelayedQueue,
        target_queue: Queue,
    ):
        self.delayed_queue = delayed_queue
        self.target_queue = target_queue

    async def schedule_once(
        self,
        task_name: str,
        payload: Any,
        run_at: Optional[datetime] = None,
        delay_seconds: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Schedule a one-time task."""
        message = Message.create(
            payload={
                "task_name": task_name,
                "payload": payload,
                **kwargs,
            },
            queue_name=self.target_queue.name,
        )

        return await self.delayed_queue.schedule(
            message,
            run_at=run_at,
            delay_seconds=delay_seconds,
        )

    async def schedule_recurring(
        self,
        task_name: str,
        payload: Any,
        interval_seconds: int,
        max_runs: Optional[int] = None,
        start_at: Optional[datetime] = None,
        **kwargs,
    ) -> str:
        """Schedule a recurring task."""
        message = Message.create(
            payload={
                "task_name": task_name,
                "payload": payload,
                **kwargs,
            },
            queue_name=self.target_queue.name,
        )

        return await self.delayed_queue.schedule(
            message,
            run_at=start_at,
            interval_seconds=interval_seconds,
            max_runs=max_runs,
        )

    async def schedule_cron(
        self,
        task_name: str,
        payload: Any,
        cron_expression: str,
        **kwargs,
    ) -> str:
        """Schedule a cron-based task."""
        message = Message.create(
            payload={
                "task_name": task_name,
                "payload": payload,
                **kwargs,
            },
            queue_name=self.target_queue.name,
        )

        return await self.delayed_queue.schedule(
            message,
            cron_expression=cron_expression,
        )

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        return await self.delayed_queue.cancel(task_id)

    async def list_scheduled(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List scheduled tasks."""
        scheduled = await self.delayed_queue.get_pending(limit=limit)
        return [s.to_dict() for s in scheduled]
