"""Metrics collection and aggregation for call monitoring."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class TimeGranularity(str, Enum):
    """Time granularity for metric aggregation."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
        }


@dataclass
class AggregatedMetric:
    """Aggregated metric over a time period."""
    name: str
    period_start: datetime
    period_end: datetime
    count: int = 0
    sum: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    avg: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "count": self.count,
            "sum": self.sum,
            "min": self.min if self.min != float("inf") else 0,
            "max": self.max if self.max != float("-inf") else 0,
            "avg": self.avg,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "labels": self.labels,
        }


class MetricsCollector:
    """Collects, aggregates, and stores metrics for call monitoring."""

    # Metric definitions
    METRICS = {
        "calls_total": MetricType.COUNTER,
        "calls_active": MetricType.GAUGE,
        "call_duration_seconds": MetricType.HISTOGRAM,
        "call_wait_time_seconds": MetricType.HISTOGRAM,
        "transcription_latency_ms": MetricType.HISTOGRAM,
        "llm_response_time_ms": MetricType.HISTOGRAM,
        "tts_latency_ms": MetricType.HISTOGRAM,
        "stt_latency_ms": MetricType.HISTOGRAM,
        "sentiment_score": MetricType.GAUGE,
        "api_requests_total": MetricType.COUNTER,
        "api_errors_total": MetricType.COUNTER,
        "websocket_connections": MetricType.GAUGE,
        "concurrent_calls": MetricType.GAUGE,
        "queue_length": MetricType.GAUGE,
        "agent_utilization": MetricType.GAUGE,
    }

    # Histogram buckets for different metrics
    HISTOGRAM_BUCKETS = {
        "call_duration_seconds": [30, 60, 120, 300, 600, 900, 1800, 3600],
        "call_wait_time_seconds": [1, 5, 10, 30, 60, 120, 300],
        "transcription_latency_ms": [50, 100, 200, 500, 1000, 2000, 5000],
        "llm_response_time_ms": [100, 250, 500, 1000, 2000, 5000, 10000],
        "tts_latency_ms": [50, 100, 200, 500, 1000, 2000],
        "stt_latency_ms": [50, 100, 200, 500, 1000, 2000],
    }

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        retention_days: int = 30,
        aggregation_interval: int = 60,
    ):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.retention_days = retention_days
        self.aggregation_interval = aggregation_interval

        # In-memory buffers for high-frequency metrics
        self._buffers: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

        # Background tasks
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start the metrics collector."""
        self.redis = redis.from_url(self.redis_url, decode_responses=True)

        # Start background tasks
        self._tasks.append(asyncio.create_task(self._flush_loop()))
        self._tasks.append(asyncio.create_task(self._aggregation_loop()))
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))

        logger.info("Metrics collector started")

    async def stop(self) -> None:
        """Stop the metrics collector."""
        # Final flush
        await self._flush_metrics()

        for task in self._tasks:
            task.cancel()

        if self.redis:
            await self.redis.close()

        logger.info("Metrics collector stopped")

    # Recording methods

    async def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        labels = labels or {}
        key = self._make_key(name, labels)

        async with self._lock:
            self._counters[key] += value
            self._buffers[key].append(MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels,
            ))

    async def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric."""
        labels = labels or {}
        key = self._make_key(name, labels)

        async with self._lock:
            self._gauges[key] = value
            self._buffers[key].append(MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels,
            ))

    async def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram metric."""
        labels = labels or {}
        key = self._make_key(name, labels)

        async with self._lock:
            self._histograms[key].append(value)
            self._buffers[key].append(MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels,
            ))

    # Call-specific metrics

    async def record_call_started(
        self,
        organization_id: str,
        agent_id: str,
        direction: str,
    ) -> None:
        """Record a call started event."""
        labels = {
            "organization_id": organization_id,
            "agent_id": agent_id,
            "direction": direction,
        }

        await self.increment("calls_total", labels=labels)
        await self.increment("calls_active", labels={"organization_id": organization_id})

    async def record_call_ended(
        self,
        organization_id: str,
        agent_id: str,
        duration: float,
        status: str,
        wait_time: float = 0.0,
    ) -> None:
        """Record a call ended event."""
        labels = {
            "organization_id": organization_id,
            "agent_id": agent_id,
            "status": status,
        }

        await self.histogram("call_duration_seconds", duration, labels=labels)
        await self.histogram("call_wait_time_seconds", wait_time, labels=labels)
        await self.increment("calls_active", value=-1, labels={"organization_id": organization_id})

        # Update success/failure counters
        if status in ("completed", "success"):
            await self.increment("calls_success", labels={"organization_id": organization_id})
        else:
            await self.increment("calls_failed", labels={"organization_id": organization_id, "status": status})

    async def record_latency(
        self,
        metric_name: str,
        latency_ms: float,
        organization_id: str,
        call_id: Optional[str] = None,
    ) -> None:
        """Record a latency metric."""
        labels = {"organization_id": organization_id}
        if call_id:
            labels["call_id"] = call_id

        await self.histogram(metric_name, latency_ms, labels=labels)

    async def record_concurrent_calls(
        self,
        organization_id: str,
        count: int,
    ) -> None:
        """Record concurrent call count."""
        await self.gauge(
            "concurrent_calls",
            count,
            labels={"organization_id": organization_id},
        )

    async def record_sentiment(
        self,
        organization_id: str,
        call_id: str,
        sentiment: str,
        score: float,
    ) -> None:
        """Record sentiment score."""
        await self.gauge(
            "sentiment_score",
            score,
            labels={
                "organization_id": organization_id,
                "call_id": call_id,
                "sentiment": sentiment,
            },
        )

    # Querying methods

    async def get_metric(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        granularity: TimeGranularity = TimeGranularity.MINUTE,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AggregatedMetric]:
        """Get aggregated metrics for a time range."""
        if not self.redis:
            return []

        labels = labels or {}
        end_time = end_time or datetime.utcnow()
        start_time = start_time or (end_time - timedelta(hours=1))

        # Build key pattern
        key_pattern = self._make_aggregation_key(name, labels, granularity, "*")

        # Scan for matching keys
        results = []
        async for key in self.redis.scan_iter(match=key_pattern):
            data = await self.redis.hgetall(key)
            if data:
                timestamp_str = key.split(":")[-1]
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if start_time <= timestamp <= end_time:
                        results.append(AggregatedMetric(
                            name=name,
                            period_start=timestamp,
                            period_end=timestamp + self._granularity_delta(granularity),
                            count=int(data.get("count", 0)),
                            sum=float(data.get("sum", 0)),
                            min=float(data.get("min", 0)),
                            max=float(data.get("max", 0)),
                            avg=float(data.get("avg", 0)),
                            p50=float(data.get("p50", 0)),
                            p95=float(data.get("p95", 0)),
                            p99=float(data.get("p99", 0)),
                            labels=labels,
                        ))
                except ValueError:
                    continue

        return sorted(results, key=lambda x: x.period_start)

    async def get_counter_value(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get current counter value."""
        labels = labels or {}
        key = self._make_key(name, labels)

        async with self._lock:
            return self._counters.get(key, 0.0)

    async def get_gauge_value(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get current gauge value."""
        labels = labels or {}
        key = self._make_key(name, labels)

        async with self._lock:
            return self._gauges.get(key, 0.0)

    async def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Get current histogram statistics."""
        labels = labels or {}
        key = self._make_key(name, labels)

        async with self._lock:
            values = self._histograms.get(key, [])

        if not values:
            return {"count": 0, "sum": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}

        sorted_values = sorted(values)
        count = len(sorted_values)

        return {
            "count": count,
            "sum": sum(sorted_values),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(sorted_values) / count,
            "p50": self._percentile(sorted_values, 50),
            "p95": self._percentile(sorted_values, 95),
            "p99": self._percentile(sorted_values, 99),
        }

    async def get_organization_metrics(
        self,
        organization_id: str,
        period: str = "hour",
    ) -> Dict[str, Any]:
        """Get all metrics for an organization."""
        now = datetime.utcnow()

        if period == "hour":
            start_time = now - timedelta(hours=1)
            granularity = TimeGranularity.MINUTE
        elif period == "day":
            start_time = now - timedelta(days=1)
            granularity = TimeGranularity.HOUR
        elif period == "week":
            start_time = now - timedelta(weeks=1)
            granularity = TimeGranularity.DAY
        else:
            start_time = now - timedelta(days=30)
            granularity = TimeGranularity.DAY

        labels = {"organization_id": organization_id}

        # Get various metrics
        calls_total = await self.get_metric(
            "calls_total", labels, granularity, start_time, now
        )
        call_duration = await self.get_metric(
            "call_duration_seconds", labels, granularity, start_time, now
        )
        llm_latency = await self.get_metric(
            "llm_response_time_ms", labels, granularity, start_time, now
        )

        # Calculate aggregates
        total_calls = sum(m.count for m in calls_total)
        avg_duration = (
            sum(m.avg * m.count for m in call_duration) / max(total_calls, 1)
            if call_duration else 0
        )
        avg_llm_latency = (
            sum(m.avg * m.count for m in llm_latency) / max(sum(m.count for m in llm_latency), 1)
            if llm_latency else 0
        )

        return {
            "period": period,
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "total_calls": total_calls,
            "avg_duration": avg_duration,
            "avg_llm_latency": avg_llm_latency,
            "time_series": {
                "calls": [m.to_dict() for m in calls_total],
                "duration": [m.to_dict() for m in call_duration],
                "llm_latency": [m.to_dict() for m in llm_latency],
            },
        }

    # Helper methods

    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for a metric."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _make_aggregation_key(
        self,
        name: str,
        labels: Dict[str, str],
        granularity: TimeGranularity,
        timestamp: str,
    ) -> str:
        """Create a Redis key for aggregated metrics."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"metrics:{name}:{label_str}:{granularity.value}:{timestamp}"

    def _granularity_delta(self, granularity: TimeGranularity) -> timedelta:
        """Get timedelta for a granularity."""
        deltas = {
            TimeGranularity.MINUTE: timedelta(minutes=1),
            TimeGranularity.HOUR: timedelta(hours=1),
            TimeGranularity.DAY: timedelta(days=1),
            TimeGranularity.WEEK: timedelta(weeks=1),
            TimeGranularity.MONTH: timedelta(days=30),
        }
        return deltas.get(granularity, timedelta(minutes=1))

    def _percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    # Background tasks

    async def _flush_loop(self) -> None:
        """Periodically flush buffered metrics to Redis."""
        while True:
            try:
                await asyncio.sleep(10)  # Flush every 10 seconds
                await self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush error: {e}")

    async def _flush_metrics(self) -> None:
        """Flush buffered metrics to Redis."""
        if not self.redis:
            return

        async with self._lock:
            buffers = dict(self._buffers)
            self._buffers.clear()

        for key, points in buffers.items():
            if not points:
                continue

            # Store raw points for recent data
            pipeline = self.redis.pipeline()
            for point in points:
                pipeline.zadd(
                    f"raw:{key}",
                    {point.to_dict().__str__(): point.timestamp.timestamp()},
                )

            # Trim to last hour of raw data
            cutoff = datetime.utcnow() - timedelta(hours=1)
            pipeline.zremrangebyscore(f"raw:{key}", 0, cutoff.timestamp())

            await pipeline.execute()

    async def _aggregation_loop(self) -> None:
        """Periodically aggregate metrics."""
        while True:
            try:
                await asyncio.sleep(self.aggregation_interval)
                await self._aggregate_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aggregation error: {e}")

    async def _aggregate_metrics(self) -> None:
        """Aggregate metrics into time buckets."""
        if not self.redis:
            return

        now = datetime.utcnow()

        # Aggregate histograms
        async with self._lock:
            histograms = dict(self._histograms)
            self._histograms.clear()

        for key, values in histograms.items():
            if not values:
                continue

            # Parse key to get name and labels
            name = key.split("{")[0]
            labels_str = key[key.find("{") + 1 : key.find("}")]
            labels = dict(l.split("=") for l in labels_str.split(",") if "=" in l)

            # Calculate statistics
            sorted_values = sorted(values)
            count = len(sorted_values)

            stats = {
                "count": count,
                "sum": sum(sorted_values),
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "avg": sum(sorted_values) / count,
                "p50": self._percentile(sorted_values, 50),
                "p95": self._percentile(sorted_values, 95),
                "p99": self._percentile(sorted_values, 99),
            }

            # Store aggregated data
            agg_key = self._make_aggregation_key(
                name, labels, TimeGranularity.MINUTE,
                now.strftime("%Y-%m-%dT%H:%M")
            )

            await self.redis.hset(agg_key, mapping=stats)
            await self.redis.expire(agg_key, self.retention_days * 24 * 3600)

    async def _cleanup_loop(self) -> None:
        """Periodically clean up old data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run hourly
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _cleanup_old_data(self) -> None:
        """Clean up data older than retention period."""
        if not self.redis:
            return

        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        # Find and delete old aggregation keys
        async for key in self.redis.scan_iter(match="metrics:*"):
            try:
                timestamp_str = key.split(":")[-1]
                if timestamp_str < cutoff_str:
                    await self.redis.delete(key)
            except Exception:
                continue

        logger.info(f"Cleaned up metrics older than {cutoff_str}")
