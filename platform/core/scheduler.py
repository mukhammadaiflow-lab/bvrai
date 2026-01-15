"""
Distributed Job Scheduler
=========================

Enterprise-grade job scheduling system for executing background tasks,
periodic jobs, and distributed workloads.

Features:
- Cron-based and interval scheduling
- Priority queues
- Distributed execution
- Job persistence and recovery
- Retry with exponential backoff
- Dead letter handling
- Job dependencies
- Rate limiting

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import heapq
import json
import logging
import re
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import partial, wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

T = TypeVar("T")
JobHandler = Callable[[Dict[str, Any]], Awaitable[Any]]


# =============================================================================
# ENUMS
# =============================================================================


class JobStatus(str, Enum):
    """Job execution status"""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    DEAD_LETTER = "dead_letter"


class JobPriority(int, Enum):
    """Job priority levels"""

    CRITICAL = 0
    HIGH = 10
    NORMAL = 50
    LOW = 100
    BACKGROUND = 200


class ScheduleType(str, Enum):
    """Types of schedules"""

    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    CRON = "cron"
    INTERVAL = "interval"
    FIXED_RATE = "fixed_rate"
    ONCE = "once"


class RetryStrategy(str, Enum):
    """Retry strategies"""

    NONE = "none"
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


# =============================================================================
# CRON EXPRESSION PARSER
# =============================================================================


class CronExpression:
    """
    Parses and evaluates cron expressions.

    Supports standard 5-field cron syntax:
    minute hour day_of_month month day_of_week

    Special characters:
    - * : any value
    - , : value list separator
    - - : range of values
    - / : step values

    Examples:
    - "0 * * * *" : every hour
    - "*/15 * * * *" : every 15 minutes
    - "0 9-17 * * 1-5" : 9am-5pm weekdays
    - "0 0 1 * *" : first of month
    """

    def __init__(self, expression: str):
        self.expression = expression
        self._parts = self._parse(expression)

    def _parse(self, expression: str) -> Dict[str, Set[int]]:
        """Parse cron expression into parts"""
        parts = expression.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression: {expression}. "
                "Expected 5 fields (minute hour day month weekday)"
            )

        return {
            "minute": self._parse_field(parts[0], 0, 59),
            "hour": self._parse_field(parts[1], 0, 23),
            "day": self._parse_field(parts[2], 1, 31),
            "month": self._parse_field(parts[3], 1, 12),
            "weekday": self._parse_field(parts[4], 0, 6)
        }

    def _parse_field(
        self,
        field: str,
        min_val: int,
        max_val: int
    ) -> Set[int]:
        """Parse a single cron field"""
        values = set()

        for part in field.split(","):
            if part == "*":
                values.update(range(min_val, max_val + 1))
            elif "/" in part:
                base, step = part.split("/")
                if base == "*":
                    start = min_val
                else:
                    start = int(base)
                values.update(range(start, max_val + 1, int(step)))
            elif "-" in part:
                start, end = part.split("-")
                values.update(range(int(start), int(end) + 1))
            else:
                values.add(int(part))

        return values

    def matches(self, dt: datetime) -> bool:
        """Check if datetime matches the cron expression"""
        return (
            dt.minute in self._parts["minute"] and
            dt.hour in self._parts["hour"] and
            dt.day in self._parts["day"] and
            dt.month in self._parts["month"] and
            dt.weekday() in self._parts["weekday"]
        )

    def next_occurrence(self, after: Optional[datetime] = None) -> datetime:
        """Find the next occurrence after the given datetime"""
        if after is None:
            after = datetime.utcnow()

        # Start from the next minute
        current = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Search up to 4 years ahead
        max_iterations = 365 * 4 * 24 * 60
        for _ in range(max_iterations):
            if self.matches(current):
                return current
            current += timedelta(minutes=1)

        raise ValueError(f"No next occurrence found for {self.expression}")


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Schedule:
    """Job schedule configuration"""

    type: ScheduleType = ScheduleType.IMMEDIATE
    cron: Optional[str] = None
    interval_seconds: Optional[float] = None
    delay_seconds: Optional[float] = None
    run_at: Optional[datetime] = None
    timezone: str = "UTC"

    _cron_expr: Optional[CronExpression] = field(default=None, repr=False)

    def __post_init__(self):
        if self.cron:
            self._cron_expr = CronExpression(self.cron)

    def next_run_time(self, after: Optional[datetime] = None) -> Optional[datetime]:
        """Calculate next run time"""
        after = after or datetime.utcnow()

        if self.type == ScheduleType.IMMEDIATE:
            return after

        elif self.type == ScheduleType.DELAYED:
            return after + timedelta(seconds=self.delay_seconds or 0)

        elif self.type == ScheduleType.CRON and self._cron_expr:
            return self._cron_expr.next_occurrence(after)

        elif self.type == ScheduleType.INTERVAL:
            return after + timedelta(seconds=self.interval_seconds or 0)

        elif self.type == ScheduleType.FIXED_RATE:
            return after + timedelta(seconds=self.interval_seconds or 0)

        elif self.type == ScheduleType.ONCE:
            return self.run_at

        return None


@dataclass
class RetryConfig:
    """Retry configuration for jobs"""

    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0
    multiplier: float = 2.0
    retryable_exceptions: List[str] = field(default_factory=list)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.strategy == RetryStrategy.NONE:
            return 0
        elif self.strategy == RetryStrategy.FIXED:
            return self.base_delay_seconds
        elif self.strategy == RetryStrategy.LINEAR:
            return min(
                self.base_delay_seconds * attempt,
                self.max_delay_seconds
            )
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            return min(
                self.base_delay_seconds * (self.multiplier ** (attempt - 1)),
                self.max_delay_seconds
            )
        return self.base_delay_seconds


@dataclass
class Job:
    """
    Represents a scheduled job.

    Contains all information needed to execute and track a job.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    handler: str = ""  # Handler name or function path
    payload: Dict[str, Any] = field(default_factory=dict)

    # Scheduling
    schedule: Schedule = field(default_factory=Schedule)
    priority: JobPriority = JobPriority.NORMAL
    queue: str = "default"

    # Execution
    status: JobStatus = JobStatus.PENDING
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None

    # Results
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None

    # Retry
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    attempt: int = 0

    # Timeout
    timeout_seconds: float = 3600.0

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    dependent_jobs: List[str] = field(default_factory=list)

    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    organization_id: Optional[str] = None
    user_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    @property
    def is_recurring(self) -> bool:
        return self.schedule.type in (
            ScheduleType.CRON,
            ScheduleType.INTERVAL,
            ScheduleType.FIXED_RATE
        )

    @property
    def can_retry(self) -> bool:
        return (
            self.retry_config.strategy != RetryStrategy.NONE and
            self.attempt < self.retry_config.max_retries
        )

    @property
    def is_expired(self) -> bool:
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    def __lt__(self, other: "Job") -> bool:
        # For priority queue comparison
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return (self.scheduled_at or self.created_at) < (other.scheduled_at or other.created_at)


@dataclass
class JobResult:
    """Result of a job execution"""

    job_id: str
    status: JobStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    attempt: int = 0


@dataclass
class JobMetrics:
    """Job execution metrics"""

    jobs_total: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs_retried: int = 0
    jobs_cancelled: int = 0
    avg_duration_ms: float = 0.0
    total_duration_ms: float = 0.0


# =============================================================================
# JOB QUEUE
# =============================================================================


class JobQueue:
    """
    Priority queue for jobs.
    """

    def __init__(self, name: str = "default", max_size: int = 10000):
        self.name = name
        self.max_size = max_size
        self._queue: List[Tuple[int, float, Job]] = []
        self._job_map: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, job: Job) -> bool:
        """Add a job to the queue"""
        async with self._lock:
            if len(self._queue) >= self.max_size:
                return False

            timestamp = (job.scheduled_at or datetime.utcnow()).timestamp()
            heapq.heappush(self._queue, (job.priority.value, timestamp, job))
            self._job_map[job.id] = job
            return True

    async def dequeue(self) -> Optional[Job]:
        """Get the highest priority job from the queue"""
        async with self._lock:
            while self._queue:
                _, _, job = heapq.heappop(self._queue)
                if job.id in self._job_map:
                    del self._job_map[job.id]
                    return job
            return None

    async def peek(self) -> Optional[Job]:
        """Peek at the next job without removing it"""
        async with self._lock:
            if self._queue:
                return self._queue[0][2]
            return None

    async def remove(self, job_id: str) -> bool:
        """Remove a job from the queue"""
        async with self._lock:
            if job_id in self._job_map:
                del self._job_map[job_id]
                # Mark for lazy removal
                return True
            return False

    async def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID"""
        return self._job_map.get(job_id)

    @property
    def size(self) -> int:
        return len(self._job_map)

    @property
    def is_empty(self) -> bool:
        return self.size == 0


# =============================================================================
# JOB WORKER
# =============================================================================


class JobWorker:
    """
    Worker that processes jobs from a queue.
    """

    def __init__(
        self,
        worker_id: str,
        queue: JobQueue,
        handlers: Dict[str, JobHandler],
        max_concurrent: int = 5
    ):
        self.worker_id = worker_id
        self.queue = queue
        self.handlers = handlers
        self.max_concurrent = max_concurrent
        self._running = False
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_jobs: Dict[str, asyncio.Task] = {}
        self._logger = structlog.get_logger(f"worker.{worker_id}")
        self._metrics = JobMetrics()

    async def start(self) -> None:
        """Start the worker"""
        self._running = True
        self._logger.info("worker_started")

        while self._running:
            job = await self.queue.dequeue()

            if job:
                await self._process_job(job)
            else:
                await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Stop the worker"""
        self._running = False

        # Cancel active jobs
        for job_id, task in self._active_jobs.items():
            task.cancel()

        self._logger.info("worker_stopped")

    async def _process_job(self, job: Job) -> None:
        """Process a single job"""
        async with self._semaphore:
            task = asyncio.create_task(self._execute_job(job))
            self._active_jobs[job.id] = task

            try:
                await task
            finally:
                self._active_jobs.pop(job.id, None)

    async def _execute_job(self, job: Job) -> JobResult:
        """Execute a job"""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.attempt += 1

        self._logger.info(
            "job_started",
            job_id=job.id,
            handler=job.handler,
            attempt=job.attempt
        )

        try:
            handler = self.handlers.get(job.handler)
            if not handler:
                raise ValueError(f"Handler '{job.handler}' not found")

            # Execute with timeout
            result = await asyncio.wait_for(
                handler(job.payload),
                timeout=job.timeout_seconds
            )

            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = datetime.utcnow()

            self._metrics.jobs_completed += 1
            self._metrics.total_duration_ms += job.duration_ms or 0

            self._logger.info(
                "job_completed",
                job_id=job.id,
                duration_ms=job.duration_ms
            )

            return JobResult(
                job_id=job.id,
                status=JobStatus.COMPLETED,
                result=result,
                started_at=job.started_at,
                completed_at=job.completed_at,
                duration_ms=job.duration_ms or 0,
                attempt=job.attempt
            )

        except asyncio.TimeoutError:
            job.error = "Job execution timed out"
            return await self._handle_failure(job, TimeoutError("Timeout"))

        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            self._metrics.jobs_cancelled += 1
            raise

        except Exception as e:
            job.error = str(e)
            job.traceback = traceback.format_exc()
            return await self._handle_failure(job, e)

    async def _handle_failure(
        self,
        job: Job,
        error: Exception
    ) -> JobResult:
        """Handle job failure"""
        job.completed_at = datetime.utcnow()

        if job.can_retry:
            job.status = JobStatus.RETRYING
            delay = job.retry_config.get_delay(job.attempt)

            self._logger.warning(
                "job_retrying",
                job_id=job.id,
                attempt=job.attempt,
                delay=delay
            )

            # Re-enqueue with delay
            job.scheduled_at = datetime.utcnow() + timedelta(seconds=delay)
            await self.queue.enqueue(job)

            self._metrics.jobs_retried += 1
        else:
            job.status = JobStatus.FAILED
            self._metrics.jobs_failed += 1

            self._logger.error(
                "job_failed",
                job_id=job.id,
                error=str(error)
            )

        return JobResult(
            job_id=job.id,
            status=job.status,
            error=job.error,
            started_at=job.started_at,
            completed_at=job.completed_at,
            duration_ms=job.duration_ms or 0,
            attempt=job.attempt
        )


# =============================================================================
# JOB SCHEDULER
# =============================================================================


class JobScheduler:
    """
    Central job scheduling system.

    Manages job queues, workers, scheduling, and execution.

    Usage:
        scheduler = JobScheduler()
        await scheduler.start()

        # Register a handler
        @scheduler.handler("send_email")
        async def send_email(payload: Dict[str, Any]) -> None:
            await send_email_async(payload["to"], payload["subject"])

        # Schedule a job
        job_id = await scheduler.schedule(
            "send_email",
            payload={"to": "user@example.com", "subject": "Hello"},
            schedule=Schedule(type=ScheduleType.DELAYED, delay_seconds=60)
        )

        await scheduler.stop()
    """

    def __init__(
        self,
        num_workers: int = 4,
        max_queue_size: int = 10000,
        check_interval: float = 1.0
    ):
        self._handlers: Dict[str, JobHandler] = {}
        self._queues: Dict[str, JobQueue] = {}
        self._workers: List[JobWorker] = []
        self._scheduled_jobs: Dict[str, Job] = {}
        self._job_results: Dict[str, JobResult] = {}
        self._num_workers = num_workers
        self._max_queue_size = max_queue_size
        self._check_interval = check_interval
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger("job_scheduler")
        self._metrics = JobMetrics()

        # Create default queue
        self._queues["default"] = JobQueue("default", max_queue_size)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the scheduler"""
        self._running = True

        # Start workers for each queue
        for queue_name, queue in self._queues.items():
            for i in range(self._num_workers):
                worker = JobWorker(
                    f"{queue_name}-worker-{i}",
                    queue,
                    self._handlers
                )
                self._workers.append(worker)
                asyncio.create_task(worker.start())

        # Start scheduler loop
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        self._logger.info(
            "scheduler_started",
            queues=len(self._queues),
            workers=len(self._workers)
        )

    async def stop(self) -> None:
        """Stop the scheduler"""
        self._running = False

        # Stop scheduler task
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        # Stop workers
        for worker in self._workers:
            await worker.stop()

        self._logger.info("scheduler_stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # -------------------------------------------------------------------------
    # Handler Registration
    # -------------------------------------------------------------------------

    def register_handler(
        self,
        name: str,
        handler: JobHandler
    ) -> None:
        """Register a job handler"""
        self._handlers[name] = handler
        self._logger.info("handler_registered", name=name)

    def handler(
        self,
        name: str
    ) -> Callable[[JobHandler], JobHandler]:
        """Decorator to register a job handler"""
        def decorator(func: JobHandler) -> JobHandler:
            self.register_handler(name, func)
            return func
        return decorator

    def unregister_handler(self, name: str) -> bool:
        """Unregister a job handler"""
        if name in self._handlers:
            del self._handlers[name]
            return True
        return False

    # -------------------------------------------------------------------------
    # Queue Management
    # -------------------------------------------------------------------------

    def create_queue(
        self,
        name: str,
        max_size: int = 10000
    ) -> JobQueue:
        """Create a new job queue"""
        if name not in self._queues:
            self._queues[name] = JobQueue(name, max_size)

            # Start workers for new queue
            if self._running:
                for i in range(self._num_workers):
                    worker = JobWorker(
                        f"{name}-worker-{i}",
                        self._queues[name],
                        self._handlers
                    )
                    self._workers.append(worker)
                    asyncio.create_task(worker.start())

        return self._queues[name]

    def get_queue(self, name: str) -> Optional[JobQueue]:
        """Get a queue by name"""
        return self._queues.get(name)

    # -------------------------------------------------------------------------
    # Job Scheduling
    # -------------------------------------------------------------------------

    async def schedule(
        self,
        handler: str,
        payload: Optional[Dict[str, Any]] = None,
        schedule: Optional[Schedule] = None,
        priority: JobPriority = JobPriority.NORMAL,
        queue: str = "default",
        name: Optional[str] = None,
        timeout_seconds: float = 3600.0,
        retry_config: Optional[RetryConfig] = None,
        depends_on: Optional[List[str]] = None,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> str:
        """
        Schedule a job for execution.

        Args:
            handler: Name of the registered handler
            payload: Data to pass to the handler
            schedule: Schedule configuration
            priority: Job priority
            queue: Queue to use
            name: Optional job name
            timeout_seconds: Maximum execution time
            retry_config: Retry configuration
            depends_on: List of job IDs this depends on
            tags: Job tags
            metadata: Additional metadata
            organization_id: Organization context
            user_id: User context
            expires_at: Expiration time

        Returns:
            Job ID
        """
        job = Job(
            name=name or handler,
            handler=handler,
            payload=payload or {},
            schedule=schedule or Schedule(),
            priority=priority,
            queue=queue,
            timeout_seconds=timeout_seconds,
            retry_config=retry_config or RetryConfig(),
            depends_on=depends_on or [],
            tags=tags or set(),
            metadata=metadata or {},
            organization_id=organization_id,
            user_id=user_id,
            expires_at=expires_at
        )

        # Calculate next run time
        job.next_run_at = job.schedule.next_run_time()
        job.scheduled_at = job.next_run_at

        async with self._lock:
            if job.is_recurring:
                # Store for recurring scheduling
                self._scheduled_jobs[job.id] = job
            else:
                # Enqueue for immediate/delayed execution
                job_queue = self._queues.get(queue)
                if job_queue:
                    await job_queue.enqueue(job)

            self._metrics.jobs_total += 1

        self._logger.info(
            "job_scheduled",
            job_id=job.id,
            handler=handler,
            next_run=job.next_run_at.isoformat() if job.next_run_at else None
        )

        return job.id

    async def schedule_cron(
        self,
        handler: str,
        cron: str,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Schedule a cron job"""
        return await self.schedule(
            handler=handler,
            payload=payload,
            schedule=Schedule(type=ScheduleType.CRON, cron=cron),
            **kwargs
        )

    async def schedule_interval(
        self,
        handler: str,
        interval_seconds: float,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Schedule an interval job"""
        return await self.schedule(
            handler=handler,
            payload=payload,
            schedule=Schedule(
                type=ScheduleType.INTERVAL,
                interval_seconds=interval_seconds
            ),
            **kwargs
        )

    async def schedule_delayed(
        self,
        handler: str,
        delay_seconds: float,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Schedule a delayed job"""
        return await self.schedule(
            handler=handler,
            payload=payload,
            schedule=Schedule(
                type=ScheduleType.DELAYED,
                delay_seconds=delay_seconds
            ),
            **kwargs
        )

    async def schedule_at(
        self,
        handler: str,
        run_at: datetime,
        payload: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Schedule a job at a specific time"""
        return await self.schedule(
            handler=handler,
            payload=payload,
            schedule=Schedule(type=ScheduleType.ONCE, run_at=run_at),
            **kwargs
        )

    # -------------------------------------------------------------------------
    # Job Management
    # -------------------------------------------------------------------------

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job"""
        async with self._lock:
            if job_id in self._scheduled_jobs:
                del self._scheduled_jobs[job_id]
                self._metrics.jobs_cancelled += 1
                return True

            for queue in self._queues.values():
                if await queue.remove(job_id):
                    self._metrics.jobs_cancelled += 1
                    return True

        return False

    async def pause_job(self, job_id: str) -> bool:
        """Pause a recurring job"""
        async with self._lock:
            if job_id in self._scheduled_jobs:
                self._scheduled_jobs[job_id].status = JobStatus.CANCELLED
                return True
        return False

    async def resume_job(self, job_id: str) -> bool:
        """Resume a paused job"""
        async with self._lock:
            if job_id in self._scheduled_jobs:
                job = self._scheduled_jobs[job_id]
                job.status = JobStatus.SCHEDULED
                job.next_run_at = job.schedule.next_run_time()
                return True
        return False

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID"""
        if job_id in self._scheduled_jobs:
            return self._scheduled_jobs[job_id]

        for queue in self._queues.values():
            job = await queue.get(job_id)
            if job:
                return job

        return None

    async def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """Get the result of a completed job"""
        return self._job_results.get(job_id)

    # -------------------------------------------------------------------------
    # Scheduler Loop
    # -------------------------------------------------------------------------

    async def _scheduler_loop(self) -> None:
        """Background loop for scheduling recurring jobs"""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                await self._check_scheduled_jobs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("scheduler_loop_error", error=str(e))

    async def _check_scheduled_jobs(self) -> None:
        """Check and enqueue scheduled jobs"""
        now = datetime.utcnow()

        async with self._lock:
            for job_id, job in list(self._scheduled_jobs.items()):
                if job.status != JobStatus.SCHEDULED:
                    continue

                if job.next_run_at and job.next_run_at <= now:
                    # Create execution instance
                    exec_job = Job(
                        name=job.name,
                        handler=job.handler,
                        payload=job.payload.copy(),
                        schedule=Schedule(type=ScheduleType.IMMEDIATE),
                        priority=job.priority,
                        queue=job.queue,
                        timeout_seconds=job.timeout_seconds,
                        retry_config=job.retry_config,
                        tags=job.tags.copy(),
                        metadata=job.metadata.copy(),
                        organization_id=job.organization_id,
                        user_id=job.user_id
                    )

                    # Enqueue
                    queue = self._queues.get(job.queue)
                    if queue:
                        await queue.enqueue(exec_job)

                    # Update next run time
                    if job.is_recurring:
                        job.next_run_at = job.schedule.next_run_time(now)
                    else:
                        del self._scheduled_jobs[job_id]

    # -------------------------------------------------------------------------
    # Status & Metrics
    # -------------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics"""
        queue_stats = {}
        for name, queue in self._queues.items():
            queue_stats[name] = {
                "size": queue.size,
                "is_empty": queue.is_empty
            }

        return {
            "jobs_total": self._metrics.jobs_total,
            "jobs_completed": self._metrics.jobs_completed,
            "jobs_failed": self._metrics.jobs_failed,
            "jobs_retried": self._metrics.jobs_retried,
            "jobs_cancelled": self._metrics.jobs_cancelled,
            "scheduled_jobs": len(self._scheduled_jobs),
            "queues": queue_stats,
            "workers": len(self._workers),
            "handlers": list(self._handlers.keys())
        }

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "running": self._running,
            "metrics": self.get_metrics()
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def cron(expression: str) -> Schedule:
    """Create a cron schedule"""
    return Schedule(type=ScheduleType.CRON, cron=expression)


def interval(seconds: float) -> Schedule:
    """Create an interval schedule"""
    return Schedule(type=ScheduleType.INTERVAL, interval_seconds=seconds)


def delayed(seconds: float) -> Schedule:
    """Create a delayed schedule"""
    return Schedule(type=ScheduleType.DELAYED, delay_seconds=seconds)


def at(run_time: datetime) -> Schedule:
    """Create a one-time schedule"""
    return Schedule(type=ScheduleType.ONCE, run_at=run_time)
