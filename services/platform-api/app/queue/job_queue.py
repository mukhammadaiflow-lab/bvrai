"""Background job queue system."""

from typing import Optional, Dict, Any, List, Callable, Awaitable, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import uuid
import logging
import traceback
import functools

logger = logging.getLogger(__name__)

T = TypeVar("T")


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class JobPriority(int, Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class JobResult:
    """Result of a job execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    duration_ms: int = 0


@dataclass
class Job:
    """A background job."""
    id: str
    name: str
    queue: str
    payload: Dict[str, Any]
    status: JobStatus
    priority: JobPriority
    max_retries: int
    retry_count: int
    timeout_seconds: int
    result: Optional[JobResult] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    scheduled_for: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "queue": self.queue,
            "payload": self.payload,
            "status": self.status.value,
            "priority": self.priority.value,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "timeout_seconds": self.timeout_seconds,
            "result": self.result.__dict__ if self.result else None,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "scheduled_for": self.scheduled_for.isoformat() if self.scheduled_for else None,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create from dictionary."""
        result_data = data.get("result")
        result = JobResult(**result_data) if result_data else None

        return cls(
            id=data["id"],
            name=data["name"],
            queue=data["queue"],
            payload=data["payload"],
            status=JobStatus(data["status"]),
            priority=JobPriority(data["priority"]),
            max_retries=data["max_retries"],
            retry_count=data["retry_count"],
            timeout_seconds=data["timeout_seconds"],
            result=result,
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            scheduled_for=datetime.fromisoformat(data["scheduled_for"]) if data.get("scheduled_for") else None,
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


JobHandler = Callable[[Job], Awaitable[Any]]


class JobRegistry:
    """Registry of job handlers."""

    def __init__(self):
        self._handlers: Dict[str, JobHandler] = {}

    def register(
        self,
        name: str,
        handler: Optional[JobHandler] = None,
    ) -> Union[Callable, None]:
        """
        Register a job handler.

        Can be used as a decorator or directly.
        """
        if handler is not None:
            self._handlers[name] = handler
            return None

        def decorator(func: JobHandler) -> JobHandler:
            self._handlers[name] = func
            return func

        return decorator

    def get(self, name: str) -> Optional[JobHandler]:
        """Get a handler by name."""
        return self._handlers.get(name)

    def list_handlers(self) -> List[str]:
        """List all registered handlers."""
        return list(self._handlers.keys())


class InMemoryQueue:
    """
    In-memory job queue.

    For development and testing. Use Redis queue in production.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._jobs: Dict[str, Job] = {}
        self._pending: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._delayed: List[Job] = []
        self._lock = asyncio.Lock()

    async def enqueue(self, job: Job) -> None:
        """Add a job to the queue."""
        async with self._lock:
            self._jobs[job.id] = job
            job.status = JobStatus.QUEUED

            if job.scheduled_for and job.scheduled_for > datetime.utcnow():
                self._delayed.append(job)
            else:
                # Priority queue uses (priority, timestamp, job_id) for ordering
                await self._pending.put((
                    -job.priority.value,  # Negative for higher priority first
                    job.created_at.timestamp(),
                    job.id,
                ))

    async def dequeue(self, timeout: float = 1.0) -> Optional[Job]:
        """Get next job from queue."""
        try:
            _, _, job_id = await asyncio.wait_for(
                self._pending.get(),
                timeout=timeout,
            )
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow()
            return job
        except asyncio.TimeoutError:
            return None

    async def complete(self, job_id: str, result: JobResult) -> None:
        """Mark job as completed."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED if result.success else JobStatus.FAILED
                job.result = result
                job.completed_at = datetime.utcnow()

    async def retry(self, job_id: str) -> None:
        """Retry a failed job."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.retry_count += 1
                job.status = JobStatus.RETRYING
                job.started_at = None
                job.completed_at = None
                await self._pending.put((
                    -job.priority.value,
                    datetime.utcnow().timestamp(),
                    job.id,
                ))

    async def cancel(self, job_id: str) -> bool:
        """Cancel a job."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in (JobStatus.PENDING, JobStatus.QUEUED):
                job.status = JobStatus.CANCELLED
                return True
            return False

    async def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    async def process_delayed(self) -> int:
        """Move due delayed jobs to pending queue."""
        async with self._lock:
            now = datetime.utcnow()
            due_jobs = [j for j in self._delayed if j.scheduled_for <= now]
            for job in due_jobs:
                self._delayed.remove(job)
                await self._pending.put((
                    -job.priority.value,
                    job.created_at.timestamp(),
                    job.id,
                ))
            return len(due_jobs)

    @property
    def pending_count(self) -> int:
        """Get number of pending jobs."""
        return self._pending.qsize()

    @property
    def total_count(self) -> int:
        """Get total number of jobs."""
        return len(self._jobs)


class JobWorker:
    """
    Worker that processes jobs from a queue.

    Usage:
        registry = JobRegistry()

        @registry.register("send_email")
        async def send_email(job: Job):
            # Process job
            ...

        queue = InMemoryQueue()
        worker = JobWorker(queue, registry)

        await worker.start()
    """

    def __init__(
        self,
        queue: InMemoryQueue,
        registry: JobRegistry,
        concurrency: int = 5,
        poll_interval: float = 1.0,
    ):
        self.queue = queue
        self.registry = registry
        self.concurrency = concurrency
        self.poll_interval = poll_interval

        self._running = False
        self._workers: List[asyncio.Task] = []
        self._semaphore = asyncio.Semaphore(concurrency)
        self._stats = WorkerStats()

    async def start(self) -> None:
        """Start the worker."""
        if self._running:
            return

        self._running = True

        # Start worker tasks
        for i in range(self.concurrency):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)

        # Start delayed job processor
        delayed_processor = asyncio.create_task(self._delayed_processor())
        self._workers.append(delayed_processor)

        logger.info(f"Job worker started with {self.concurrency} workers")

    async def stop(self, wait: bool = True) -> None:
        """Stop the worker."""
        self._running = False

        if wait:
            # Wait for current jobs to complete
            for worker in self._workers:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass

        self._workers.clear()
        logger.info("Job worker stopped")

    async def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop."""
        logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                async with self._semaphore:
                    job = await self.queue.dequeue(timeout=self.poll_interval)
                    if job:
                        await self._process_job(job)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(self.poll_interval)

        logger.debug(f"Worker {worker_id} stopped")

    async def _delayed_processor(self) -> None:
        """Process delayed jobs."""
        while self._running:
            try:
                moved = await self.queue.process_delayed()
                if moved:
                    logger.debug(f"Moved {moved} delayed jobs to pending")
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Delayed processor error: {e}")

    async def _process_job(self, job: Job) -> None:
        """Process a single job."""
        handler = self.registry.get(job.name)
        if handler is None:
            logger.error(f"No handler for job type: {job.name}")
            await self.queue.complete(job.id, JobResult(
                success=False,
                error=f"No handler for job type: {job.name}",
            ))
            return

        start_time = datetime.utcnow()
        self._stats.jobs_started += 1

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                handler(job),
                timeout=job.timeout_seconds,
            )

            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            await self.queue.complete(job.id, JobResult(
                success=True,
                result=result,
                duration_ms=duration_ms,
            ))

            self._stats.jobs_completed += 1
            logger.info(f"Job {job.id} completed in {duration_ms}ms")

        except asyncio.TimeoutError:
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            if job.retry_count < job.max_retries:
                await self.queue.retry(job.id)
                self._stats.jobs_retried += 1
                logger.warning(f"Job {job.id} timed out, retrying ({job.retry_count + 1}/{job.max_retries})")
            else:
                await self.queue.complete(job.id, JobResult(
                    success=False,
                    error="Job timed out",
                    duration_ms=duration_ms,
                ))
                self._stats.jobs_failed += 1
                logger.error(f"Job {job.id} failed after timeout")

        except Exception as e:
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            tb = traceback.format_exc()

            if job.retry_count < job.max_retries:
                await self.queue.retry(job.id)
                self._stats.jobs_retried += 1
                logger.warning(f"Job {job.id} failed: {e}, retrying ({job.retry_count + 1}/{job.max_retries})")
            else:
                await self.queue.complete(job.id, JobResult(
                    success=False,
                    error=str(e),
                    traceback=tb,
                    duration_ms=duration_ms,
                ))
                self._stats.jobs_failed += 1
                logger.error(f"Job {job.id} failed permanently: {e}")

    @property
    def stats(self) -> "WorkerStats":
        """Get worker statistics."""
        return self._stats


@dataclass
class WorkerStats:
    """Worker statistics."""
    jobs_started: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs_retried: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "jobs_started": self.jobs_started,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "jobs_retried": self.jobs_retried,
            "success_rate": self.jobs_completed / self.jobs_started if self.jobs_started > 0 else 0,
        }


class JobScheduler:
    """
    Job scheduler for recurring and delayed jobs.

    Usage:
        scheduler = JobScheduler(queue)

        # Schedule recurring job
        scheduler.schedule_recurring(
            "cleanup",
            "cleanup_old_records",
            {},
            interval_seconds=3600,  # Every hour
        )

        # Schedule delayed job
        scheduler.schedule_delayed(
            "send_reminder",
            {"user_id": "123"},
            delay_seconds=3600,
        )

        await scheduler.start()
    """

    def __init__(self, queue: InMemoryQueue):
        self.queue = queue
        self._recurring_jobs: Dict[str, RecurringJob] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Job scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Job scheduler stopped")

    def schedule_recurring(
        self,
        schedule_id: str,
        job_name: str,
        payload: Dict[str, Any],
        interval_seconds: int,
        priority: JobPriority = JobPriority.NORMAL,
    ) -> None:
        """Schedule a recurring job."""
        self._recurring_jobs[schedule_id] = RecurringJob(
            schedule_id=schedule_id,
            job_name=job_name,
            payload=payload,
            interval_seconds=interval_seconds,
            priority=priority,
            next_run=datetime.utcnow(),
        )

    def cancel_recurring(self, schedule_id: str) -> bool:
        """Cancel a recurring job."""
        if schedule_id in self._recurring_jobs:
            del self._recurring_jobs[schedule_id]
            return True
        return False

    async def schedule_delayed(
        self,
        job_name: str,
        payload: Dict[str, Any],
        delay_seconds: int,
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3,
        timeout_seconds: int = 300,
    ) -> str:
        """Schedule a delayed job."""
        job = Job(
            id=str(uuid.uuid4()),
            name=job_name,
            queue=self.queue.name,
            payload=payload,
            status=JobStatus.PENDING,
            priority=priority,
            max_retries=max_retries,
            retry_count=0,
            timeout_seconds=timeout_seconds,
            scheduled_for=datetime.utcnow() + timedelta(seconds=delay_seconds),
        )
        await self.queue.enqueue(job)
        return job.id

    async def schedule_at(
        self,
        job_name: str,
        payload: Dict[str, Any],
        run_at: datetime,
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3,
        timeout_seconds: int = 300,
    ) -> str:
        """Schedule a job for a specific time."""
        job = Job(
            id=str(uuid.uuid4()),
            name=job_name,
            queue=self.queue.name,
            payload=payload,
            status=JobStatus.PENDING,
            priority=priority,
            max_retries=max_retries,
            retry_count=0,
            timeout_seconds=timeout_seconds,
            scheduled_for=run_at,
        )
        await self.queue.enqueue(job)
        return job.id

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.utcnow()

                for recurring in list(self._recurring_jobs.values()):
                    if recurring.next_run <= now:
                        # Create job
                        job = Job(
                            id=str(uuid.uuid4()),
                            name=recurring.job_name,
                            queue=self.queue.name,
                            payload=recurring.payload,
                            status=JobStatus.PENDING,
                            priority=recurring.priority,
                            max_retries=3,
                            retry_count=0,
                            timeout_seconds=300,
                            metadata={"schedule_id": recurring.schedule_id},
                        )
                        await self.queue.enqueue(job)

                        # Update next run
                        recurring.next_run = now + timedelta(seconds=recurring.interval_seconds)
                        recurring.last_run = now

                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(5.0)


@dataclass
class RecurringJob:
    """Definition of a recurring job."""
    schedule_id: str
    job_name: str
    payload: Dict[str, Any]
    interval_seconds: int
    priority: JobPriority
    next_run: datetime
    last_run: Optional[datetime] = None


def job(
    name: str,
    registry: JobRegistry,
    priority: JobPriority = JobPriority.NORMAL,
    max_retries: int = 3,
    timeout_seconds: int = 300,
) -> Callable:
    """
    Decorator for registering a job handler.

    Usage:
        @job("send_email", registry)
        async def send_email_handler(job: Job):
            email = job.payload["email"]
            await send_email_to(email)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(j: Job) -> Any:
            return await func(j)

        registry.register(name, wrapper)
        wrapper.job_name = name
        wrapper.priority = priority
        wrapper.max_retries = max_retries
        wrapper.timeout_seconds = timeout_seconds

        return wrapper

    return decorator


async def enqueue_job(
    queue: InMemoryQueue,
    name: str,
    payload: Dict[str, Any],
    priority: JobPriority = JobPriority.NORMAL,
    max_retries: int = 3,
    timeout_seconds: int = 300,
    scheduled_for: Optional[datetime] = None,
) -> str:
    """Helper function to enqueue a job."""
    job = Job(
        id=str(uuid.uuid4()),
        name=name,
        queue=queue.name,
        payload=payload,
        status=JobStatus.PENDING,
        priority=priority,
        max_retries=max_retries,
        retry_count=0,
        timeout_seconds=timeout_seconds,
        scheduled_for=scheduled_for,
    )
    await queue.enqueue(job)
    return job.id
