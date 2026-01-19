"""Analytics metrics collection and aggregation."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import time
import json


logger = structlog.get_logger()


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class HistogramBucket:
    """Histogram bucket for distribution tracking."""
    le: float  # Less than or equal
    count: int = 0


class MetricsCollector:
    """
    Collects and aggregates metrics.

    Features:
    - Multiple metric types (counter, gauge, histogram)
    - Tag-based filtering
    - Time-series storage
    - Aggregation (sum, avg, percentiles)
    """

    def __init__(
        self,
        redis_client=None,
        retention_hours: int = 24,
        flush_interval_seconds: float = 10.0,
    ):
        self._redis = redis_client
        self._retention = timedelta(hours=retention_hours)
        self._flush_interval = flush_interval_seconds

        # In-memory buffers
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)

        # Time series data
        self._time_series: Dict[str, List[MetricPoint]] = defaultdict(list)

        # State
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Histogram buckets configuration
        self._histogram_buckets = [
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
        ]

    async def start(self) -> None:
        """Start the metrics collector."""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("metrics_collector_started")

    async def stop(self) -> None:
        """Stop and flush remaining metrics."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        await self._flush()
        logger.info("metrics_collector_stopped")

    async def increment(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, tags)
        async with self._lock:
            self._counters[key] += value

    async def gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric."""
        key = self._make_key(name, tags)
        async with self._lock:
            self._gauges[key] = value

    async def histogram(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram value."""
        key = self._make_key(name, tags)
        async with self._lock:
            self._histograms[key].append(value)

    async def timing(
        self,
        name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a timing measurement."""
        await self.histogram(f"{name}_ms", duration_ms, tags)

    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, tags)

    async def record(self, point: MetricPoint) -> None:
        """Record a metric point."""
        async with self._lock:
            self._time_series[point.name].append(point)

    async def get_counter(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get current counter value."""
        key = self._make_key(name, tags)
        return self._counters.get(key, 0.0)

    async def get_gauge(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        """Get current gauge value."""
        key = self._make_key(name, tags)
        return self._gauges.get(key)

    async def get_histogram_stats(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, tags)
        values = self._histograms.get(key, [])

        if not values:
            return {}

        sorted_values = sorted(values)
        count = len(sorted_values)

        return {
            "count": count,
            "sum": sum(sorted_values),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(sorted_values) / count,
            "p50": self._percentile(sorted_values, 50),
            "p90": self._percentile(sorted_values, 90),
            "p95": self._percentile(sorted_values, 95),
            "p99": self._percentile(sorted_values, 99),
        }

    async def get_time_series(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval_seconds: int = 60,
    ) -> List[Dict[str, Any]]:
        """Get time series data with aggregation."""
        points = self._time_series.get(name, [])

        if not points:
            return []

        # Filter by time range
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]

        # Aggregate into buckets
        buckets = defaultdict(list)
        for point in points:
            bucket_ts = int(point.timestamp.timestamp() / interval_seconds) * interval_seconds
            buckets[bucket_ts].append(point.value)

        # Calculate aggregates
        result = []
        for ts, values in sorted(buckets.items()):
            result.append({
                "timestamp": datetime.fromtimestamp(ts).isoformat(),
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            })

        return result

    def _make_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for a metric."""
        if not tags:
            return name

        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"

    def _percentile(self, sorted_values: List[float], p: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0

        k = (len(sorted_values) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f

        if f == c:
            return sorted_values[f]

        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    async def _flush_loop(self) -> None:
        """Periodically flush metrics."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("metrics_flush_error", error=str(e))

    async def _flush(self) -> None:
        """Flush metrics to storage."""
        now = datetime.utcnow()

        async with self._lock:
            # Record counters to time series
            for key, value in self._counters.items():
                name = key.split("{")[0]
                self._time_series[name].append(
                    MetricPoint(name=name, value=value, timestamp=now)
                )

            # Record gauges to time series
            for key, value in self._gauges.items():
                name = key.split("{")[0]
                self._time_series[name].append(
                    MetricPoint(name=name, value=value, timestamp=now)
                )

            # Persist to Redis if available
            if self._redis:
                await self._persist_to_redis()

            # Clean old data
            cutoff = now - self._retention
            for name in list(self._time_series.keys()):
                self._time_series[name] = [
                    p for p in self._time_series[name]
                    if p.timestamp > cutoff
                ]

    async def _persist_to_redis(self) -> None:
        """Persist metrics to Redis."""
        try:
            pipeline = self._redis.pipeline()

            # Store counters
            for key, value in self._counters.items():
                pipeline.hset("metrics:counters", key, value)

            # Store gauges
            for key, value in self._gauges.items():
                pipeline.hset("metrics:gauges", key, value)

            await pipeline.execute()

        except Exception as e:
            logger.error("redis_persist_error", error=str(e))

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: len(v) for k, v in self._histograms.items()
            },
        }


class TimerContext:
    """Context manager for timing operations."""

    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ):
        self._collector = collector
        self._name = name
        self._tags = tags
        self._start_time: Optional[float] = None

    async def __aenter__(self):
        self._start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._start_time:
            duration_ms = (time.perf_counter() - self._start_time) * 1000
            await self._collector.timing(self._name, duration_ms, self._tags)


# Call-specific metrics
class CallMetrics:
    """
    Specialized metrics for call analytics.

    Tracks:
    - Call volume
    - Duration distributions
    - Success/failure rates
    - Latency metrics
    """

    def __init__(self, collector: MetricsCollector):
        self._collector = collector

    async def call_started(
        self,
        agent_id: str,
        direction: str,
    ) -> None:
        """Record call start."""
        await self._collector.increment(
            "calls.started",
            tags={"agent_id": agent_id, "direction": direction},
        )
        await self._collector.increment("calls.active")

    async def call_ended(
        self,
        agent_id: str,
        direction: str,
        duration_seconds: float,
        end_reason: str,
        success: bool,
    ) -> None:
        """Record call end."""
        tags = {"agent_id": agent_id, "direction": direction}

        await self._collector.increment("calls.ended", tags=tags)
        await self._collector.increment("calls.active", value=-1)

        await self._collector.histogram(
            "calls.duration_seconds",
            duration_seconds,
            tags=tags,
        )

        if success:
            await self._collector.increment("calls.success", tags=tags)
        else:
            await self._collector.increment("calls.failed", tags=tags)

        await self._collector.increment(
            f"calls.end_reason.{end_reason}",
            tags=tags,
        )

    async def turn_completed(
        self,
        agent_id: str,
        speaker: str,
        duration_ms: float,
    ) -> None:
        """Record conversation turn."""
        await self._collector.increment(
            "turns.total",
            tags={"agent_id": agent_id, "speaker": speaker},
        )
        await self._collector.histogram(
            "turns.duration_ms",
            duration_ms,
            tags={"agent_id": agent_id, "speaker": speaker},
        )

    async def asr_completed(
        self,
        latency_ms: float,
        confidence: float,
        word_count: int,
    ) -> None:
        """Record ASR metrics."""
        await self._collector.histogram("asr.latency_ms", latency_ms)
        await self._collector.histogram("asr.confidence", confidence)
        await self._collector.histogram("asr.word_count", word_count)

    async def tts_completed(
        self,
        latency_ms: float,
        audio_duration_ms: float,
        character_count: int,
    ) -> None:
        """Record TTS metrics."""
        await self._collector.histogram("tts.latency_ms", latency_ms)
        await self._collector.histogram("tts.audio_duration_ms", audio_duration_ms)
        await self._collector.histogram("tts.character_count", character_count)

    async def llm_completed(
        self,
        model: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record LLM metrics."""
        tags = {"model": model}
        await self._collector.histogram("llm.latency_ms", latency_ms, tags)
        await self._collector.histogram("llm.prompt_tokens", prompt_tokens, tags)
        await self._collector.histogram("llm.completion_tokens", completion_tokens, tags)
        await self._collector.increment("llm.requests", tags=tags)

    async def function_executed(
        self,
        function_name: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Record function execution."""
        tags = {"function": function_name}
        await self._collector.increment("functions.executed", tags=tags)
        await self._collector.histogram("functions.latency_ms", latency_ms, tags)

        if success:
            await self._collector.increment("functions.success", tags=tags)
        else:
            await self._collector.increment("functions.failed", tags=tags)


# Global metrics instance
metrics_collector = MetricsCollector()
call_metrics = CallMetrics(metrics_collector)
