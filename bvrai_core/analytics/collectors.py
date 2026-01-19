"""
Metrics Collectors Module

This module provides metrics collection capabilities for capturing
call and system metrics in real-time.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .base import (
    MetricType,
    MetricPoint,
    MetricDefinition,
    CallMetricsSnapshot,
    CallOutcome,
)


logger = logging.getLogger(__name__)


class MetricsBackend(ABC):
    """Abstract base class for metrics storage backends."""

    @abstractmethod
    async def write(self, metric_name: str, point: MetricPoint) -> None:
        """Write a metric point."""
        pass

    @abstractmethod
    async def write_batch(
        self,
        points: List[Tuple[str, MetricPoint]],
    ) -> None:
        """Write a batch of metric points."""
        pass

    @abstractmethod
    async def query(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[MetricPoint]:
        """Query metric points."""
        pass


class InMemoryMetricsBackend(MetricsBackend):
    """In-memory metrics backend for development and testing."""

    def __init__(self, max_points_per_metric: int = 10000):
        """
        Initialize backend.

        Args:
            max_points_per_metric: Maximum points to keep per metric
        """
        self.max_points = max_points_per_metric
        self._data: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def write(self, metric_name: str, point: MetricPoint) -> None:
        """Write a metric point."""
        async with self._lock:
            self._data[metric_name].append(point)

            # Trim if needed
            if len(self._data[metric_name]) > self.max_points:
                self._data[metric_name] = self._data[metric_name][-self.max_points:]

    async def write_batch(
        self,
        points: List[Tuple[str, MetricPoint]],
    ) -> None:
        """Write a batch of metric points."""
        async with self._lock:
            for metric_name, point in points:
                self._data[metric_name].append(point)

            # Trim all metrics
            for metric_name in self._data:
                if len(self._data[metric_name]) > self.max_points:
                    self._data[metric_name] = self._data[metric_name][-self.max_points:]

    async def query(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[MetricPoint]:
        """Query metric points."""
        points = self._data.get(metric_name, [])

        result = []
        for point in points:
            # Filter by time
            if point.timestamp < start_time or point.timestamp > end_time:
                continue

            # Filter by tags
            if tags:
                match = all(
                    point.tags.get(k) == v
                    for k, v in tags.items()
                )
                if not match:
                    continue

            result.append(point)

        return result


class InfluxDBBackend(MetricsBackend):
    """InfluxDB metrics backend for production use."""

    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: Optional[str] = None,
        org: str = "default",
        bucket: str = "metrics",
    ):
        """
        Initialize backend.

        Args:
            url: InfluxDB URL
            token: Authentication token
            org: Organization name
            bucket: Bucket name
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self._client: Optional[Any] = None

    async def _get_client(self) -> Any:
        """Get InfluxDB client."""
        if self._client is None:
            try:
                from influxdb_client import InfluxDBClient
                from influxdb_client.client.write_api import ASYNCHRONOUS

                self._client = InfluxDBClient(
                    url=self.url,
                    token=self.token,
                    org=self.org,
                )
            except ImportError:
                raise RuntimeError("influxdb-client package not installed")

        return self._client

    async def write(self, metric_name: str, point: MetricPoint) -> None:
        """Write a metric point to InfluxDB."""
        client = await self._get_client()

        from influxdb_client import Point

        p = (
            Point(metric_name)
            .field("value", point.value)
            .time(point.timestamp)
        )

        for key, value in point.tags.items():
            p.tag(key, value)

        write_api = client.write_api()
        write_api.write(bucket=self.bucket, record=p)

    async def write_batch(
        self,
        points: List[Tuple[str, MetricPoint]],
    ) -> None:
        """Write a batch of metric points."""
        client = await self._get_client()

        from influxdb_client import Point

        records = []
        for metric_name, point in points:
            p = (
                Point(metric_name)
                .field("value", point.value)
                .time(point.timestamp)
            )
            for key, value in point.tags.items():
                p.tag(key, value)
            records.append(p)

        write_api = client.write_api()
        write_api.write(bucket=self.bucket, record=records)

    async def query(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[MetricPoint]:
        """Query metric points from InfluxDB."""
        client = await self._get_client()

        # Build Flux query
        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r._measurement == "{metric_name}")
        '''

        if tags:
            for key, value in tags.items():
                query += f'\n|> filter(fn: (r) => r.{key} == "{value}")'

        query_api = client.query_api()
        tables = query_api.query(query)

        result = []
        for table in tables:
            for record in table.records:
                point = MetricPoint(
                    timestamp=record.get_time(),
                    value=record.get_value(),
                    tags={k: v for k, v in record.values.items() if k.startswith("_") is False},
                )
                result.append(point)

        return result


class Counter:
    """A counter metric that can only be incremented."""

    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Initialize counter.

        Args:
            name: Metric name
            tags: Default tags
        """
        self.name = name
        self.tags = tags or {}
        self._value = 0
        self._lock = asyncio.Lock()

    async def increment(
        self,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> MetricPoint:
        """Increment the counter."""
        async with self._lock:
            self._value += value

            merged_tags = {**self.tags, **(tags or {})}
            return MetricPoint(
                timestamp=datetime.utcnow(),
                value=self._value,
                tags=merged_tags,
            )

    @property
    def value(self) -> float:
        """Get current value."""
        return self._value


class Gauge:
    """A gauge metric that can go up and down."""

    def __init__(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Initialize gauge.

        Args:
            name: Metric name
            tags: Default tags
        """
        self.name = name
        self.tags = tags or {}
        self._value = 0.0
        self._lock = asyncio.Lock()

    async def set(
        self,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> MetricPoint:
        """Set the gauge value."""
        async with self._lock:
            self._value = value

            merged_tags = {**self.tags, **(tags or {})}
            return MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                tags=merged_tags,
            )

    async def increment(
        self,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> MetricPoint:
        """Increment the gauge."""
        async with self._lock:
            self._value += value
            return await self.set(self._value, tags)

    async def decrement(
        self,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> MetricPoint:
        """Decrement the gauge."""
        async with self._lock:
            self._value -= value
            return await self.set(self._value, tags)

    @property
    def value(self) -> float:
        """Get current value."""
        return self._value


class Histogram:
    """A histogram metric for measuring value distributions."""

    DEFAULT_BUCKETS = [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
    ]

    def __init__(
        self,
        name: str,
        buckets: Optional[List[float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize histogram.

        Args:
            name: Metric name
            buckets: Bucket boundaries
            tags: Default tags
        """
        self.name = name
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self.tags = tags or {}

        self._values: List[float] = []
        self._bucket_counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self._bucket_counts[float("inf")] = 0
        self._sum = 0.0
        self._count = 0
        self._lock = asyncio.Lock()

    async def observe(
        self,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> MetricPoint:
        """Record a value observation."""
        async with self._lock:
            self._values.append(value)
            self._sum += value
            self._count += 1

            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
                    break
            else:
                self._bucket_counts[float("inf")] += 1

            merged_tags = {**self.tags, **(tags or {})}
            return MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                tags=merged_tags,
            )

    def get_percentile(self, percentile: float) -> float:
        """Get percentile value."""
        if not self._values:
            return 0.0

        sorted_values = sorted(self._values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    @property
    def sum(self) -> float:
        """Get sum of all values."""
        return self._sum

    @property
    def count(self) -> int:
        """Get count of observations."""
        return self._count

    @property
    def mean(self) -> float:
        """Get mean value."""
        return self._sum / self._count if self._count > 0 else 0.0


class Timer(Histogram):
    """A timer metric for measuring durations."""

    # Buckets in seconds
    DEFAULT_BUCKETS = [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
        1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
    ]

    def time(self) -> "TimerContext":
        """Start a timer context."""
        return TimerContext(self)


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, timer: Timer):
        """Initialize context."""
        self.timer = timer
        self.start_time: Optional[float] = None

    async def __aenter__(self) -> "TimerContext":
        """Enter context."""
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and record duration."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            await self.timer.observe(duration)


class MetricsCollector:
    """
    Main metrics collector for recording and storing metrics.

    Provides:
    - Metric creation and management
    - Automatic flushing to backend
    - Buffering for performance
    """

    def __init__(
        self,
        backend: Optional[MetricsBackend] = None,
        flush_interval_seconds: float = 10.0,
        buffer_size: int = 1000,
    ):
        """
        Initialize collector.

        Args:
            backend: Metrics storage backend
            flush_interval_seconds: Interval for automatic flushing
            buffer_size: Maximum buffer size before auto-flush
        """
        self.backend = backend or InMemoryMetricsBackend()
        self.flush_interval = flush_interval_seconds
        self.buffer_size = buffer_size

        # Metrics registry
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._timers: Dict[str, Timer] = {}

        # Buffer for batching writes
        self._buffer: List[Tuple[str, MetricPoint]] = []
        self._buffer_lock = asyncio.Lock()

        # Background flush task
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the collector."""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Metrics collector started")

    async def stop(self) -> None:
        """Stop the collector."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self.flush()
        logger.info("Metrics collector stopped")

    async def _flush_loop(self) -> None:
        """Background task to flush metrics periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Metrics flush error: {e}")

    async def flush(self) -> int:
        """
        Flush buffered metrics to backend.

        Returns:
            Number of points flushed
        """
        async with self._buffer_lock:
            if not self._buffer:
                return 0

            points = self._buffer.copy()
            self._buffer.clear()

        try:
            await self.backend.write_batch(points)
            return len(points)
        except Exception as e:
            logger.exception(f"Failed to flush metrics: {e}")
            # Put points back in buffer
            async with self._buffer_lock:
                self._buffer.extend(points)
            return 0

    async def _record(self, metric_name: str, point: MetricPoint) -> None:
        """Record a metric point."""
        async with self._buffer_lock:
            self._buffer.append((metric_name, point))

            # Auto-flush if buffer is full
            if len(self._buffer) >= self.buffer_size:
                asyncio.create_task(self.flush())

    def counter(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Counter:
        """Get or create a counter."""
        key = f"{name}:{str(tags)}"
        if key not in self._counters:
            self._counters[key] = Counter(name, tags)
        return self._counters[key]

    def gauge(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Gauge:
        """Get or create a gauge."""
        key = f"{name}:{str(tags)}"
        if key not in self._gauges:
            self._gauges[key] = Gauge(name, tags)
        return self._gauges[key]

    def histogram(
        self,
        name: str,
        buckets: Optional[List[float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Histogram:
        """Get or create a histogram."""
        key = f"{name}:{str(tags)}"
        if key not in self._histograms:
            self._histograms[key] = Histogram(name, buckets, tags)
        return self._histograms[key]

    def timer(
        self,
        name: str,
        buckets: Optional[List[float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Timer:
        """Get or create a timer."""
        key = f"{name}:{str(tags)}"
        if key not in self._timers:
            self._timers[key] = Timer(name, buckets, tags)
        return self._timers[key]

    async def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter and record."""
        counter = self.counter(name, tags)
        point = await counter.increment(value, tags)
        await self._record(name, point)

    async def set_gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge and record."""
        gauge = self.gauge(name, tags)
        point = await gauge.set(value, tags)
        await self._record(name, point)

    async def observe_histogram(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a histogram value and record."""
        histogram = self.histogram(name, tags=tags)
        point = await histogram.observe(value, tags)
        await self._record(name, point)

    async def record_timing(
        self,
        name: str,
        duration_seconds: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a timing measurement."""
        timer = self.timer(name, tags=tags)
        point = await timer.observe(duration_seconds, tags)
        await self._record(name, point)


class CallMetricsCollector:
    """
    Specialized collector for call metrics.

    Provides convenient methods for recording call-specific metrics.
    """

    def __init__(self, collector: MetricsCollector):
        """
        Initialize call metrics collector.

        Args:
            collector: Base metrics collector
        """
        self.collector = collector

    async def record_call_started(
        self,
        organization_id: str,
        agent_id: str,
        direction: str,
        campaign_id: Optional[str] = None,
    ) -> None:
        """Record a call started."""
        tags = {
            "organization_id": organization_id,
            "agent_id": agent_id,
            "direction": direction,
        }
        if campaign_id:
            tags["campaign_id"] = campaign_id

        await self.collector.increment_counter("calls.started", tags=tags)
        await self.collector.set_gauge("calls.active", 1, tags=tags)

    async def record_call_ended(
        self,
        organization_id: str,
        agent_id: str,
        direction: str,
        outcome: CallOutcome,
        duration_seconds: float,
        campaign_id: Optional[str] = None,
    ) -> None:
        """Record a call ended."""
        tags = {
            "organization_id": organization_id,
            "agent_id": agent_id,
            "direction": direction,
            "outcome": outcome.value,
        }
        if campaign_id:
            tags["campaign_id"] = campaign_id

        await self.collector.increment_counter("calls.ended", tags=tags)
        await self.collector.observe_histogram("calls.duration", duration_seconds, tags=tags)

        # Update outcome counters
        await self.collector.increment_counter(f"calls.outcome.{outcome.value}", tags=tags)

    async def record_response_time(
        self,
        organization_id: str,
        agent_id: str,
        response_time_ms: float,
    ) -> None:
        """Record agent response time."""
        tags = {
            "organization_id": organization_id,
            "agent_id": agent_id,
        }
        await self.collector.observe_histogram(
            "agent.response_time_ms",
            response_time_ms,
            tags=tags,
        )

    async def record_llm_usage(
        self,
        organization_id: str,
        agent_id: str,
        tokens: int,
        latency_ms: float,
        model: str,
    ) -> None:
        """Record LLM usage."""
        tags = {
            "organization_id": organization_id,
            "agent_id": agent_id,
            "model": model,
        }
        await self.collector.increment_counter("llm.tokens", tokens, tags=tags)
        await self.collector.observe_histogram("llm.latency_ms", latency_ms, tags=tags)
        await self.collector.increment_counter("llm.requests", tags=tags)

    async def record_function_call(
        self,
        organization_id: str,
        agent_id: str,
        function_name: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record a function call."""
        tags = {
            "organization_id": organization_id,
            "agent_id": agent_id,
            "function": function_name,
            "success": str(success),
        }
        await self.collector.increment_counter("function.calls", tags=tags)
        await self.collector.observe_histogram("function.duration_ms", duration_ms, tags=tags)

    async def record_call_snapshot(self, snapshot: CallMetricsSnapshot) -> None:
        """Record a complete call metrics snapshot."""
        base_tags = {
            "organization_id": snapshot.organization_id,
            "agent_id": snapshot.agent_id,
            "direction": snapshot.direction,
            "outcome": snapshot.outcome.value,
        }

        if snapshot.campaign_id:
            base_tags["campaign_id"] = snapshot.campaign_id

        if snapshot.industry:
            base_tags["industry"] = snapshot.industry

        # Record various metrics
        await self.collector.increment_counter("calls.total", tags=base_tags)
        await self.collector.observe_histogram(
            "calls.duration",
            snapshot.duration_seconds,
            tags=base_tags,
        )
        await self.collector.observe_histogram(
            "calls.response_time",
            snapshot.avg_response_time_ms,
            tags=base_tags,
        )
        await self.collector.increment_counter(
            "calls.tokens",
            snapshot.llm_tokens_used,
            tags=base_tags,
        )
        await self.collector.increment_counter(
            "calls.cost",
            snapshot.cost_cents,
            tags=base_tags,
        )


__all__ = [
    "MetricsBackend",
    "InMemoryMetricsBackend",
    "InfluxDBBackend",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "TimerContext",
    "MetricsCollector",
    "CallMetricsCollector",
]
