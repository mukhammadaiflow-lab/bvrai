"""
Data Pipeline Engine
====================

Enterprise-grade real-time data pipeline for processing voice AI events,
transcriptions, analytics, and streaming data.

Features:
- Real-time stream processing
- Batch processing support
- Multiple source/sink connectors
- Custom transformation stages
- Windowing and aggregation
- Error handling and retry
- Backpressure management
- Metrics and monitoring

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Deque,
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
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

T = TypeVar("T")
DataT = TypeVar("DataT")


# =============================================================================
# ENUMS
# =============================================================================


class PipelineState(str, Enum):
    """Pipeline execution states"""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class ProcessingMode(str, Enum):
    """Data processing modes"""

    STREAM = "stream"      # Real-time streaming
    BATCH = "batch"        # Batch processing
    MICRO_BATCH = "micro_batch"  # Micro-batch


class DeliveryGuarantee(str, Enum):
    """Message delivery guarantees"""

    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


class BackpressureStrategy(str, Enum):
    """Backpressure handling strategies"""

    DROP = "drop"          # Drop new messages
    BLOCK = "block"        # Block producer
    BUFFER = "buffer"      # Buffer messages
    SAMPLE = "sample"      # Sample messages


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class DataRecord:
    """
    Represents a single record in the pipeline.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    key: Optional[str] = None
    value: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    partition: Optional[int] = None
    offset: Optional[int] = None
    source: str = ""

    # Processing info
    processing_time_ms: float = 0.0
    retries: int = 0
    errors: List[str] = field(default_factory=list)

    def with_value(self, value: Any) -> "DataRecord":
        """Create a copy with new value"""
        return DataRecord(
            id=self.id,
            key=self.key,
            value=value,
            timestamp=self.timestamp,
            headers=self.headers.copy(),
            metadata=self.metadata.copy(),
            partition=self.partition,
            offset=self.offset,
            source=self.source
        )

    def with_metadata(self, key: str, value: Any) -> "DataRecord":
        """Create a copy with additional metadata"""
        new_metadata = self.metadata.copy()
        new_metadata[key] = value
        return DataRecord(
            id=self.id,
            key=self.key,
            value=self.value,
            timestamp=self.timestamp,
            headers=self.headers.copy(),
            metadata=new_metadata,
            partition=self.partition,
            offset=self.offset,
            source=self.source
        )


@dataclass
class PipelineConfig:
    """Pipeline configuration"""

    name: str
    description: str = ""

    # Processing
    mode: ProcessingMode = ProcessingMode.STREAM
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE

    # Performance
    parallelism: int = 4
    batch_size: int = 100
    batch_timeout_ms: float = 1000.0
    buffer_size: int = 10000

    # Backpressure
    backpressure_strategy: BackpressureStrategy = BackpressureStrategy.BUFFER
    max_buffer_size: int = 100000

    # Error handling
    max_retries: int = 3
    retry_delay_ms: float = 1000.0
    dead_letter_enabled: bool = True

    # Checkpointing
    checkpoint_enabled: bool = True
    checkpoint_interval_ms: float = 10000.0

    # Monitoring
    metrics_enabled: bool = True
    metrics_interval_ms: float = 5000.0


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics"""

    records_received: int = 0
    records_processed: int = 0
    records_failed: int = 0
    records_dropped: int = 0
    bytes_received: int = 0
    bytes_processed: int = 0
    processing_time_total_ms: float = 0.0
    avg_processing_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    buffer_utilization: float = 0.0
    last_record_time: Optional[datetime] = None

    def record_processed(self, processing_time_ms: float, bytes_count: int = 0) -> None:
        self.records_processed += 1
        self.bytes_processed += bytes_count
        self.processing_time_total_ms += processing_time_ms
        self.avg_processing_time_ms = (
            self.processing_time_total_ms / self.records_processed
        )
        self.last_record_time = datetime.utcnow()


@dataclass
class PipelineResult:
    """Result of pipeline processing"""

    success: bool
    records_processed: int = 0
    records_failed: int = 0
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    metrics: Optional[PipelineMetrics] = None


@dataclass
class Checkpoint:
    """Pipeline checkpoint for recovery"""

    pipeline_name: str
    stage_name: str
    offset: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ABSTRACT CLASSES
# =============================================================================


class DataSource(ABC):
    """Abstract base class for data sources"""

    def __init__(self, name: str):
        self.name = name
        self._running = False
        self._logger = structlog.get_logger(f"source.{name}")

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the data source"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data source"""
        pass

    @abstractmethod
    async def read(self) -> AsyncGenerator[DataRecord, None]:
        """Read records from the source"""
        pass

    @abstractmethod
    async def commit(self, record: DataRecord) -> None:
        """Commit a processed record"""
        pass

    @property
    def is_running(self) -> bool:
        return self._running


class DataSink(ABC):
    """Abstract base class for data sinks"""

    def __init__(self, name: str):
        self.name = name
        self._running = False
        self._logger = structlog.get_logger(f"sink.{name}")

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the data sink"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data sink"""
        pass

    @abstractmethod
    async def write(self, record: DataRecord) -> None:
        """Write a record to the sink"""
        pass

    @abstractmethod
    async def write_batch(self, records: List[DataRecord]) -> None:
        """Write multiple records to the sink"""
        pass

    @abstractmethod
    async def flush(self) -> None:
        """Flush pending writes"""
        pass


class Processor(ABC):
    """Abstract base class for data processors"""

    def __init__(self, name: str):
        self.name = name
        self._logger = structlog.get_logger(f"processor.{name}")

    @abstractmethod
    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        """Process a single record"""
        pass

    async def process_batch(
        self,
        records: List[DataRecord]
    ) -> List[DataRecord]:
        """Process multiple records"""
        results = []
        for record in records:
            result = await self.process(record)
            if result:
                results.append(result)
        return results

    async def initialize(self) -> None:
        """Initialize the processor"""
        pass

    async def cleanup(self) -> None:
        """Cleanup the processor"""
        pass


# =============================================================================
# PIPELINE STAGE
# =============================================================================


@dataclass
class PipelineStage:
    """
    Represents a stage in the pipeline.
    """

    name: str
    processor: Processor
    parallelism: int = 1
    timeout_seconds: float = 30.0
    retry_enabled: bool = True
    max_retries: int = 3

    # Routing
    route_to: Optional[List[str]] = None
    route_condition: Optional[Callable[[DataRecord], str]] = None

    # Metrics
    records_in: int = 0
    records_out: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0


# =============================================================================
# PIPELINE
# =============================================================================


class Pipeline:
    """
    Data processing pipeline.

    Connects sources, processors, and sinks into a directed acyclic graph
    for processing data records.

    Usage:
        pipeline = Pipeline("voice-analytics")

        pipeline.from_source(KafkaSource("calls-topic"))
        pipeline.add_stage("transform", TransformProcessor(...))
        pipeline.add_stage("enrich", EnrichProcessor(...))
        pipeline.to_sink(DatabaseSink("analytics-db"))

        await pipeline.start()
    """

    def __init__(
        self,
        name: str,
        config: Optional[PipelineConfig] = None
    ):
        self.name = name
        self.config = config or PipelineConfig(name=name)
        self._sources: List[DataSource] = []
        self._sinks: List[DataSink] = []
        self._stages: Dict[str, PipelineStage] = {}
        self._stage_order: List[str] = []
        self._state = PipelineState.CREATED
        self._metrics = PipelineMetrics()
        self._dead_letter_queue: Deque[Tuple[DataRecord, str]] = deque(maxlen=10000)
        self._buffer: asyncio.Queue = asyncio.Queue(maxsize=config.buffer_size if config else 10000)
        self._lock = asyncio.Lock()
        self._checkpoint_store: Dict[str, Checkpoint] = {}
        self._tasks: List[asyncio.Task] = []
        self._logger = structlog.get_logger(f"pipeline.{name}")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def metrics(self) -> PipelineMetrics:
        return self._metrics

    @property
    def is_running(self) -> bool:
        return self._state == PipelineState.RUNNING

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def from_source(self, source: DataSource) -> "Pipeline":
        """Add a data source"""
        self._sources.append(source)
        return self

    def to_sink(self, sink: DataSink) -> "Pipeline":
        """Add a data sink"""
        self._sinks.append(sink)
        return self

    def add_stage(
        self,
        name: str,
        processor: Processor,
        parallelism: int = 1,
        timeout_seconds: float = 30.0,
        route_to: Optional[List[str]] = None
    ) -> "Pipeline":
        """Add a processing stage"""
        stage = PipelineStage(
            name=name,
            processor=processor,
            parallelism=parallelism,
            timeout_seconds=timeout_seconds,
            route_to=route_to
        )
        self._stages[name] = stage
        self._stage_order.append(name)
        return self

    def route(
        self,
        condition: Callable[[DataRecord], str]
    ) -> "Pipeline":
        """Add routing logic"""
        if self._stage_order:
            last_stage = self._stages[self._stage_order[-1]]
            last_stage.route_condition = condition
        return self

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the pipeline"""
        if self._state == PipelineState.RUNNING:
            return

        self._state = PipelineState.STARTING
        self._logger.info("pipeline_starting", name=self.name)

        try:
            # Initialize processors
            for stage in self._stages.values():
                await stage.processor.initialize()

            # Connect sources
            for source in self._sources:
                await source.connect()

            # Connect sinks
            for sink in self._sinks:
                await sink.connect()

            # Start processing tasks
            self._state = PipelineState.RUNNING

            # Start source readers
            for source in self._sources:
                task = asyncio.create_task(self._read_source(source))
                self._tasks.append(task)

            # Start stage processors
            for stage_name in self._stage_order:
                stage = self._stages[stage_name]
                for i in range(stage.parallelism):
                    task = asyncio.create_task(
                        self._process_stage(stage)
                    )
                    self._tasks.append(task)

            # Start sink writers
            for sink in self._sinks:
                task = asyncio.create_task(self._write_sink(sink))
                self._tasks.append(task)

            # Start checkpoint task
            if self.config.checkpoint_enabled:
                task = asyncio.create_task(self._checkpoint_loop())
                self._tasks.append(task)

            # Start metrics task
            if self.config.metrics_enabled:
                task = asyncio.create_task(self._metrics_loop())
                self._tasks.append(task)

            self._logger.info("pipeline_started", name=self.name)

        except Exception as e:
            self._state = PipelineState.FAILED
            self._logger.error("pipeline_start_failed", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop the pipeline"""
        if self._state == PipelineState.STOPPED:
            return

        self._state = PipelineState.STOPPING
        self._logger.info("pipeline_stopping", name=self.name)

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Flush sinks
        for sink in self._sinks:
            await sink.flush()
            await sink.disconnect()

        # Disconnect sources
        for source in self._sources:
            await source.disconnect()

        # Cleanup processors
        for stage in self._stages.values():
            await stage.processor.cleanup()

        self._state = PipelineState.STOPPED
        self._logger.info("pipeline_stopped", name=self.name)

    async def pause(self) -> None:
        """Pause the pipeline"""
        self._state = PipelineState.PAUSED
        self._logger.info("pipeline_paused", name=self.name)

    async def resume(self) -> None:
        """Resume the pipeline"""
        if self._state == PipelineState.PAUSED:
            self._state = PipelineState.RUNNING
            self._logger.info("pipeline_resumed", name=self.name)

    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------

    async def _read_source(self, source: DataSource) -> None:
        """Read records from a source"""
        try:
            async for record in source.read():
                if self._state != PipelineState.RUNNING:
                    break

                self._metrics.records_received += 1
                record.source = source.name

                # Handle backpressure
                if self._buffer.full():
                    if self.config.backpressure_strategy == BackpressureStrategy.DROP:
                        self._metrics.records_dropped += 1
                        continue
                    elif self.config.backpressure_strategy == BackpressureStrategy.BLOCK:
                        await self._buffer.put(record)
                    elif self.config.backpressure_strategy == BackpressureStrategy.SAMPLE:
                        if self._metrics.records_received % 10 == 0:
                            await self._buffer.put(record)
                        continue
                else:
                    await self._buffer.put(record)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error(
                "source_read_error",
                source=source.name,
                error=str(e)
            )

    async def _process_stage(self, stage: PipelineStage) -> None:
        """Process records through a stage"""
        try:
            while self._state == PipelineState.RUNNING:
                try:
                    record = await asyncio.wait_for(
                        self._buffer.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                if self._state == PipelineState.PAUSED:
                    await self._buffer.put(record)
                    await asyncio.sleep(0.1)
                    continue

                start_time = time.time()
                stage.records_in += 1

                try:
                    # Process with timeout
                    result = await asyncio.wait_for(
                        stage.processor.process(record),
                        timeout=stage.timeout_seconds
                    )

                    if result:
                        stage.records_out += 1
                        processing_time = (time.time() - start_time) * 1000
                        result.processing_time_ms = processing_time

                        # Update metrics
                        self._metrics.record_processed(processing_time)

                        # Route to next stage or sink
                        await self._route_record(result, stage)

                except asyncio.TimeoutError:
                    stage.errors += 1
                    await self._handle_error(record, stage, "Processing timeout")

                except Exception as e:
                    stage.errors += 1
                    await self._handle_error(record, stage, str(e))

        except asyncio.CancelledError:
            pass

    async def _route_record(
        self,
        record: DataRecord,
        stage: PipelineStage
    ) -> None:
        """Route a processed record"""
        # Use custom routing if defined
        if stage.route_condition:
            next_stage = stage.route_condition(record)
            if next_stage in self._stages:
                await self._buffer.put(record)
                return

        # Route to specified stages
        if stage.route_to:
            for next_stage_name in stage.route_to:
                if next_stage_name in self._stages:
                    await self._buffer.put(record)
            return

        # Default: send to sink buffer
        for sink in self._sinks:
            await self._sink_buffer.put(record)

    async def _write_sink(self, sink: DataSink) -> None:
        """Write records to a sink"""
        batch: List[DataRecord] = []
        last_flush = time.time()

        try:
            while self._state == PipelineState.RUNNING:
                try:
                    record = await asyncio.wait_for(
                        self._sink_buffer.get(),
                        timeout=0.1
                    )
                    batch.append(record)

                except asyncio.TimeoutError:
                    pass

                # Flush batch
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    (time.time() - last_flush) * 1000 >= self.config.batch_timeout_ms
                )

                if should_flush and batch:
                    try:
                        await sink.write_batch(batch)
                        batch = []
                        last_flush = time.time()
                    except Exception as e:
                        self._logger.error(
                            "sink_write_error",
                            sink=sink.name,
                            error=str(e)
                        )
                        # Move to dead letter queue
                        for record in batch:
                            self._dead_letter_queue.append((record, str(e)))
                        batch = []

        except asyncio.CancelledError:
            # Final flush
            if batch:
                try:
                    await sink.write_batch(batch)
                except Exception:
                    pass

    async def _handle_error(
        self,
        record: DataRecord,
        stage: PipelineStage,
        error: str
    ) -> None:
        """Handle processing error"""
        record.errors.append(error)
        record.retries += 1

        if stage.retry_enabled and record.retries <= stage.max_retries:
            # Retry
            await asyncio.sleep(self.config.retry_delay_ms / 1000)
            await self._buffer.put(record)
            self._logger.warning(
                "record_retry",
                record_id=record.id,
                stage=stage.name,
                retry=record.retries
            )
        else:
            # Dead letter
            self._metrics.records_failed += 1
            if self.config.dead_letter_enabled:
                self._dead_letter_queue.append((record, error))
            self._logger.error(
                "record_failed",
                record_id=record.id,
                stage=stage.name,
                error=error
            )

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    async def _checkpoint_loop(self) -> None:
        """Periodically save checkpoints"""
        try:
            while self._state == PipelineState.RUNNING:
                await asyncio.sleep(self.config.checkpoint_interval_ms / 1000)
                await self._save_checkpoint()
        except asyncio.CancelledError:
            await self._save_checkpoint()

    async def _save_checkpoint(self) -> None:
        """Save current checkpoint"""
        for stage_name, stage in self._stages.items():
            checkpoint = Checkpoint(
                pipeline_name=self.name,
                stage_name=stage_name,
                offset={
                    "records_in": stage.records_in,
                    "records_out": stage.records_out
                },
                metadata={
                    "errors": stage.errors,
                    "avg_latency_ms": stage.avg_latency_ms
                }
            )
            self._checkpoint_store[stage_name] = checkpoint

        self._logger.debug("checkpoint_saved", pipeline=self.name)

    async def restore_from_checkpoint(self) -> None:
        """Restore pipeline from checkpoint"""
        for stage_name, checkpoint in self._checkpoint_store.items():
            if stage_name in self._stages:
                stage = self._stages[stage_name]
                stage.records_in = checkpoint.offset.get("records_in", 0)
                stage.records_out = checkpoint.offset.get("records_out", 0)

        self._logger.info("checkpoint_restored", pipeline=self.name)

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    async def _metrics_loop(self) -> None:
        """Periodically collect metrics"""
        try:
            while self._state == PipelineState.RUNNING:
                await asyncio.sleep(self.config.metrics_interval_ms / 1000)
                self._collect_metrics()
        except asyncio.CancelledError:
            pass

    def _collect_metrics(self) -> None:
        """Collect current metrics"""
        self._metrics.buffer_utilization = (
            self._buffer.qsize() / self.config.buffer_size
        )

        if self._metrics.records_received > 0:
            self._metrics.error_rate = (
                self._metrics.records_failed / self._metrics.records_received
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        return {
            "name": self.name,
            "state": self._state.value,
            "records_received": self._metrics.records_received,
            "records_processed": self._metrics.records_processed,
            "records_failed": self._metrics.records_failed,
            "records_dropped": self._metrics.records_dropped,
            "avg_processing_time_ms": self._metrics.avg_processing_time_ms,
            "throughput_per_second": self._metrics.throughput_per_second,
            "error_rate": self._metrics.error_rate,
            "buffer_utilization": self._metrics.buffer_utilization,
            "dead_letter_queue_size": len(self._dead_letter_queue),
            "stages": {
                name: {
                    "records_in": stage.records_in,
                    "records_out": stage.records_out,
                    "errors": stage.errors,
                    "avg_latency_ms": stage.avg_latency_ms
                }
                for name, stage in self._stages.items()
            }
        }


# =============================================================================
# PIPELINE ENGINE
# =============================================================================


class PipelineEngine:
    """
    Central pipeline management engine.

    Manages multiple pipelines, handles deployment, and provides
    monitoring and control.

    Usage:
        engine = PipelineEngine()
        await engine.start()

        # Deploy a pipeline
        await engine.deploy(pipeline)

        # Get metrics
        metrics = engine.get_all_metrics()

        await engine.stop()
    """

    def __init__(self):
        self._pipelines: Dict[str, Pipeline] = {}
        self._running = False
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger("pipeline_engine")

    async def start(self) -> None:
        """Start the pipeline engine"""
        self._running = True
        self._logger.info("pipeline_engine_started")

    async def stop(self) -> None:
        """Stop the pipeline engine and all pipelines"""
        for pipeline in self._pipelines.values():
            if pipeline.is_running:
                await pipeline.stop()

        self._running = False
        self._logger.info("pipeline_engine_stopped")

    async def deploy(self, pipeline: Pipeline) -> None:
        """Deploy a pipeline"""
        async with self._lock:
            self._pipelines[pipeline.name] = pipeline
            await pipeline.start()
            self._logger.info("pipeline_deployed", name=pipeline.name)

    async def undeploy(self, name: str) -> None:
        """Undeploy a pipeline"""
        async with self._lock:
            if name in self._pipelines:
                pipeline = self._pipelines[name]
                await pipeline.stop()
                del self._pipelines[name]
                self._logger.info("pipeline_undeployed", name=name)

    def get_pipeline(self, name: str) -> Optional[Pipeline]:
        """Get a pipeline by name"""
        return self._pipelines.get(name)

    def list_pipelines(self) -> List[str]:
        """List all pipeline names"""
        return list(self._pipelines.keys())

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all pipelines"""
        return {
            name: pipeline.get_metrics()
            for name, pipeline in self._pipelines.items()
        }

    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            "running": self._running,
            "pipelines_count": len(self._pipelines),
            "pipelines": {
                name: pipeline.state.value
                for name, pipeline in self._pipelines.items()
            }
        }
