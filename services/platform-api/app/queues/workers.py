"""
Queue Workers

Message processing workers:
- Single message handlers
- Batch handlers
- Worker pools
- Graceful shutdown
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import time
import logging

from app.queues.base import (
    Message,
    MessageStatus,
    Queue,
    QueueStats,
)
from app.queues.deadletter import DeadLetterQueue, FailureReason

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Type aliases for handlers
MessageHandler = Callable[[Message], Awaitable[Any]]
BatchHandler = Callable[[List[Message]], Awaitable[List[Any]]]


class WorkerStatus(str, Enum):
    """Worker status."""
    IDLE = "idle"
    PROCESSING = "processing"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class WorkerConfig:
    """Worker configuration."""
    name: str = "worker"
    batch_size: int = 1
    poll_interval_seconds: float = 1.0
    processing_timeout_seconds: float = 30.0
    enable_batch_processing: bool = False
    max_batch_wait_seconds: float = 5.0
    graceful_shutdown_seconds: float = 30.0
    enable_dead_letter: bool = True
    prefetch_count: int = 1
    concurrency: int = 1
    enable_metrics: bool = True


@dataclass
class WorkerStats:
    """Worker statistics."""
    worker_name: str
    status: WorkerStatus = WorkerStatus.IDLE
    messages_processed: int = 0
    messages_failed: int = 0
    batches_processed: int = 0
    current_batch_size: int = 0
    avg_processing_time_ms: float = 0.0
    last_processed_at: Optional[datetime] = None
    uptime_seconds: float = 0.0
    started_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worker_name": self.worker_name,
            "status": self.status.value,
            "messages_processed": self.messages_processed,
            "messages_failed": self.messages_failed,
            "batches_processed": self.batches_processed,
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "last_processed_at": self.last_processed_at.isoformat() if self.last_processed_at else None,
            "uptime_seconds": round(self.uptime_seconds, 2),
        }


class Worker:
    """
    Queue message worker.

    Processes messages from a queue using provided handler.
    """

    def __init__(
        self,
        queue: Queue,
        handler: MessageHandler,
        config: Optional[WorkerConfig] = None,
        dead_letter_queue: Optional[DeadLetterQueue] = None,
    ):
        self.queue = queue
        self.handler = handler
        self.config = config or WorkerConfig()
        self.dlq = dead_letter_queue
        self._status = WorkerStatus.STOPPED
        self._task: Optional[asyncio.Task] = None
        self._stats = WorkerStats(worker_name=self.config.name)
        self._processing_times: List[float] = []
        self._stop_event = asyncio.Event()

    @property
    def status(self) -> WorkerStatus:
        """Get worker status."""
        return self._status

    async def start(self) -> None:
        """Start the worker."""
        if self._status == WorkerStatus.PROCESSING:
            return

        self._stop_event.clear()
        self._status = WorkerStatus.IDLE
        self._stats.started_at = datetime.utcnow()
        self._task = asyncio.create_task(self._run())

        logger.info(f"Started worker: {self.config.name}")

    async def stop(self, graceful: bool = True) -> None:
        """Stop the worker."""
        self._status = WorkerStatus.STOPPING
        self._stop_event.set()

        if self._task and graceful:
            try:
                await asyncio.wait_for(
                    self._task,
                    timeout=self.config.graceful_shutdown_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Worker {self.config.name} shutdown timeout, forcing stop")
                self._task.cancel()

        self._status = WorkerStatus.STOPPED
        logger.info(f"Stopped worker: {self.config.name}")

    async def _run(self) -> None:
        """Main worker loop."""
        while not self._stop_event.is_set():
            try:
                # Poll for messages
                messages = await self.queue.dequeue(count=self.config.prefetch_count)

                if not messages:
                    await asyncio.sleep(self.config.poll_interval_seconds)
                    continue

                # Process messages
                for message in messages:
                    await self._process_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {self.config.name} error: {e}")
                self._status = WorkerStatus.ERROR
                await asyncio.sleep(self.config.poll_interval_seconds)

    async def _process_message(self, message: Message) -> None:
        """Process a single message."""
        self._status = WorkerStatus.PROCESSING
        start_time = time.time()

        try:
            # Execute handler with timeout
            await asyncio.wait_for(
                self.handler(message),
                timeout=self.config.processing_timeout_seconds,
            )

            # Acknowledge success
            await self.queue.ack(message.id)

            # Update stats
            duration = (time.time() - start_time) * 1000
            self._record_success(duration)

        except asyncio.TimeoutError:
            logger.error(f"Message {message.id} processing timeout")
            await self._handle_failure(message, "Processing timeout", FailureReason.TIMEOUT)

        except Exception as e:
            logger.error(f"Message {message.id} processing error: {e}")
            await self._handle_failure(message, str(e), FailureReason.HANDLER_ERROR)

        finally:
            self._status = WorkerStatus.IDLE

    async def _handle_failure(
        self,
        message: Message,
        error: str,
        reason: FailureReason,
    ) -> None:
        """Handle message processing failure."""
        await self.queue.nack(message.id, error)
        self._stats.messages_failed += 1

        # Send to DLQ if configured and retries exhausted
        if self.config.enable_dead_letter and self.dlq and not message.can_retry():
            await self.dlq.add(
                message=message,
                failure_reason=reason,
                failure_details=error,
                original_queue=self.queue.name,
            )

    def _record_success(self, duration_ms: float) -> None:
        """Record successful processing."""
        self._stats.messages_processed += 1
        self._stats.last_processed_at = datetime.utcnow()

        self._processing_times.append(duration_ms)
        if len(self._processing_times) > 1000:
            self._processing_times = self._processing_times[-1000:]

        self._stats.avg_processing_time_ms = (
            sum(self._processing_times) / len(self._processing_times)
        )

    def get_stats(self) -> WorkerStats:
        """Get worker statistics."""
        if self._stats.started_at:
            self._stats.uptime_seconds = (
                datetime.utcnow() - self._stats.started_at
            ).total_seconds()
        self._stats.status = self._status
        return self._stats


class BatchWorker(Worker):
    """
    Batch processing worker.

    Processes messages in batches for efficiency.
    """

    def __init__(
        self,
        queue: Queue,
        handler: BatchHandler,
        config: Optional[WorkerConfig] = None,
        dead_letter_queue: Optional[DeadLetterQueue] = None,
    ):
        # Override to enable batch processing
        cfg = config or WorkerConfig()
        cfg.enable_batch_processing = True
        if cfg.batch_size == 1:
            cfg.batch_size = 10

        super().__init__(queue, self._wrap_handler(handler), cfg, dead_letter_queue)
        self.batch_handler = handler
        self._batch: List[Message] = []
        self._batch_start: Optional[datetime] = None

    def _wrap_handler(self, batch_handler: BatchHandler) -> MessageHandler:
        """Wrap batch handler to work with single message interface."""
        async def handler(message: Message):
            # This is called per message, but we accumulate and process in batches
            self._batch.append(message)

            if not self._batch_start:
                self._batch_start = datetime.utcnow()

        return handler

    async def _run(self) -> None:
        """Main worker loop with batch processing."""
        while not self._stop_event.is_set():
            try:
                # Poll for messages
                messages = await self.queue.dequeue(count=self.config.batch_size)

                if messages:
                    self._batch.extend(messages)
                    if not self._batch_start:
                        self._batch_start = datetime.utcnow()

                # Check if we should process the batch
                should_process = False

                if len(self._batch) >= self.config.batch_size:
                    should_process = True
                elif self._batch_start:
                    wait_time = (datetime.utcnow() - self._batch_start).total_seconds()
                    if wait_time >= self.config.max_batch_wait_seconds:
                        should_process = True

                if should_process and self._batch:
                    await self._process_batch()

                if not messages:
                    await asyncio.sleep(self.config.poll_interval_seconds)

            except asyncio.CancelledError:
                # Process remaining batch before exit
                if self._batch:
                    await self._process_batch()
                break
            except Exception as e:
                logger.error(f"Batch worker {self.config.name} error: {e}")
                await asyncio.sleep(self.config.poll_interval_seconds)

    async def _process_batch(self) -> None:
        """Process accumulated batch."""
        if not self._batch:
            return

        batch = self._batch
        self._batch = []
        self._batch_start = None

        self._status = WorkerStatus.PROCESSING
        self._stats.current_batch_size = len(batch)
        start_time = time.time()

        try:
            # Execute batch handler
            results = await asyncio.wait_for(
                self.batch_handler(batch),
                timeout=self.config.processing_timeout_seconds * len(batch),
            )

            # Acknowledge all messages
            for message in batch:
                await self.queue.ack(message.id)

            # Update stats
            duration = (time.time() - start_time) * 1000
            self._stats.batches_processed += 1
            self._record_success(duration / len(batch))

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Nack all messages in batch
            for message in batch:
                await self._handle_failure(message, str(e), FailureReason.HANDLER_ERROR)

        finally:
            self._status = WorkerStatus.IDLE
            self._stats.current_batch_size = 0


class WorkerPool:
    """
    Pool of workers for parallel processing.

    Manages multiple workers for a queue.
    """

    def __init__(
        self,
        queue: Queue,
        handler: MessageHandler,
        config: Optional[WorkerConfig] = None,
        dead_letter_queue: Optional[DeadLetterQueue] = None,
        pool_size: int = 4,
    ):
        self.queue = queue
        self.handler = handler
        self.config = config or WorkerConfig()
        self.dlq = dead_letter_queue
        self.pool_size = pool_size
        self._workers: List[Worker] = []
        self._started = False

    async def start(self) -> None:
        """Start all workers in the pool."""
        if self._started:
            return

        for i in range(self.pool_size):
            worker_config = WorkerConfig(
                name=f"{self.config.name}-{i}",
                batch_size=self.config.batch_size,
                poll_interval_seconds=self.config.poll_interval_seconds,
                processing_timeout_seconds=self.config.processing_timeout_seconds,
            )

            worker = Worker(
                queue=self.queue,
                handler=self.handler,
                config=worker_config,
                dead_letter_queue=self.dlq,
            )

            await worker.start()
            self._workers.append(worker)

        self._started = True
        logger.info(f"Started worker pool with {self.pool_size} workers")

    async def stop(self, graceful: bool = True) -> None:
        """Stop all workers."""
        stop_tasks = [worker.stop(graceful) for worker in self._workers]
        await asyncio.gather(*stop_tasks)
        self._workers.clear()
        self._started = False
        logger.info("Stopped worker pool")

    async def scale(self, new_size: int) -> None:
        """Scale the worker pool."""
        if new_size < 1:
            raise ValueError("Pool size must be at least 1")

        current_size = len(self._workers)

        if new_size > current_size:
            # Add workers
            for i in range(current_size, new_size):
                worker_config = WorkerConfig(
                    name=f"{self.config.name}-{i}",
                    batch_size=self.config.batch_size,
                    poll_interval_seconds=self.config.poll_interval_seconds,
                    processing_timeout_seconds=self.config.processing_timeout_seconds,
                )

                worker = Worker(
                    queue=self.queue,
                    handler=self.handler,
                    config=worker_config,
                    dead_letter_queue=self.dlq,
                )

                await worker.start()
                self._workers.append(worker)

        elif new_size < current_size:
            # Remove workers
            workers_to_stop = self._workers[new_size:]
            self._workers = self._workers[:new_size]

            for worker in workers_to_stop:
                await worker.stop()

        self.pool_size = new_size
        logger.info(f"Scaled worker pool to {new_size} workers")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        worker_stats = [w.get_stats().to_dict() for w in self._workers]

        total_processed = sum(s["messages_processed"] for s in worker_stats)
        total_failed = sum(s["messages_failed"] for s in worker_stats)
        active_workers = sum(1 for w in self._workers if w.status == WorkerStatus.PROCESSING)

        return {
            "pool_size": self.pool_size,
            "active_workers": active_workers,
            "total_messages_processed": total_processed,
            "total_messages_failed": total_failed,
            "workers": worker_stats,
        }


class ConcurrentWorker:
    """
    Concurrent worker with semaphore-based concurrency control.

    Processes multiple messages concurrently within a single worker.
    """

    def __init__(
        self,
        queue: Queue,
        handler: MessageHandler,
        config: Optional[WorkerConfig] = None,
        dead_letter_queue: Optional[DeadLetterQueue] = None,
    ):
        self.queue = queue
        self.handler = handler
        self.config = config or WorkerConfig()
        self.dlq = dead_letter_queue
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._stats = WorkerStats(worker_name=self.config.name)
        self._active_tasks: int = 0
        self._processing_times: List[float] = []

    async def start(self) -> None:
        """Start the concurrent worker."""
        self._stop_event.clear()
        self._stats.started_at = datetime.utcnow()
        self._task = asyncio.create_task(self._run())
        logger.info(f"Started concurrent worker: {self.config.name}")

    async def stop(self, graceful: bool = True) -> None:
        """Stop the concurrent worker."""
        self._stop_event.set()

        if self._task and graceful:
            # Wait for active tasks
            while self._active_tasks > 0:
                await asyncio.sleep(0.1)

            self._task.cancel()

        logger.info(f"Stopped concurrent worker: {self.config.name}")

    async def _run(self) -> None:
        """Main worker loop."""
        while not self._stop_event.is_set():
            try:
                # Wait for semaphore slot
                async with self._semaphore:
                    messages = await self.queue.dequeue(count=1)

                    if not messages:
                        await asyncio.sleep(self.config.poll_interval_seconds)
                        continue

                    # Process message concurrently
                    self._active_tasks += 1
                    asyncio.create_task(self._process_message(messages[0]))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Concurrent worker error: {e}")

    async def _process_message(self, message: Message) -> None:
        """Process a single message."""
        start_time = time.time()

        try:
            await asyncio.wait_for(
                self.handler(message),
                timeout=self.config.processing_timeout_seconds,
            )

            await self.queue.ack(message.id)

            # Record stats
            duration = (time.time() - start_time) * 1000
            self._stats.messages_processed += 1
            self._stats.last_processed_at = datetime.utcnow()
            self._processing_times.append(duration)

            if len(self._processing_times) > 1000:
                self._processing_times = self._processing_times[-1000:]

            self._stats.avg_processing_time_ms = (
                sum(self._processing_times) / len(self._processing_times)
            )

        except Exception as e:
            logger.error(f"Message {message.id} processing error: {e}")
            await self.queue.nack(message.id, str(e))
            self._stats.messages_failed += 1

        finally:
            self._active_tasks -= 1

    def get_stats(self) -> WorkerStats:
        """Get worker statistics."""
        if self._stats.started_at:
            self._stats.uptime_seconds = (
                datetime.utcnow() - self._stats.started_at
            ).total_seconds()
        return self._stats


# Decorator for creating workers
def worker(
    queue: Queue,
    config: Optional[WorkerConfig] = None,
    dlq: Optional[DeadLetterQueue] = None,
):
    """
    Decorator to create a worker from a handler function.

    Usage:
        @worker(my_queue)
        async def handle_message(message: Message):
            ...
    """
    def decorator(func: MessageHandler) -> Worker:
        return Worker(
            queue=queue,
            handler=func,
            config=config,
            dead_letter_queue=dlq,
        )

    return decorator
