"""
Time Series Storage and Querying

Advanced time-series data management with:
- Multiple storage backends
- Efficient querying and aggregation
- Downsampling and retention
- Real-time streaming
"""

from typing import Optional, Dict, Any, List, Union, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
import asyncio
import bisect
import statistics
import logging

logger = logging.getLogger(__name__)


class Granularity(str, Enum):
    """Time granularity for aggregation."""
    SECOND = "second"
    MINUTE = "minute"
    FIVE_MINUTES = "5min"
    FIFTEEN_MINUTES = "15min"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

    def to_seconds(self) -> int:
        """Convert to seconds."""
        mapping = {
            self.SECOND: 1,
            self.MINUTE: 60,
            self.FIVE_MINUTES: 300,
            self.FIFTEEN_MINUTES: 900,
            self.HOUR: 3600,
            self.DAY: 86400,
            self.WEEK: 604800,
            self.MONTH: 2592000,
        }
        return mapping.get(self, 60)


class TimeSeriesAggregation(str, Enum):
    """Aggregation functions."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"


@dataclass
class TimeSeriesPoint:
    """A single time-series data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TimeWindow:
    """Time window for queries."""
    start: datetime
    end: datetime

    @classmethod
    def last_hours(cls, hours: int) -> "TimeWindow":
        """Create window for last N hours."""
        now = datetime.utcnow()
        return cls(start=now - timedelta(hours=hours), end=now)

    @classmethod
    def last_days(cls, days: int) -> "TimeWindow":
        """Create window for last N days."""
        now = datetime.utcnow()
        return cls(start=now - timedelta(days=days), end=now)

    @classmethod
    def today(cls) -> "TimeWindow":
        """Create window for today."""
        now = datetime.utcnow()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return cls(start=start, end=now)


@dataclass
class TimeSeriesQuery:
    """Query for time-series data."""
    metric: str
    window: TimeWindow
    granularity: Granularity = Granularity.MINUTE
    aggregation: TimeSeriesAggregation = TimeSeriesAggregation.AVG
    tags: Optional[Dict[str, str]] = None
    group_by: Optional[List[str]] = None
    fill: Optional[str] = None  # "null", "zero", "previous"


@dataclass
class TimeSeriesResult:
    """Result of time-series query."""
    metric: str
    points: List[TimeSeriesPoint]
    aggregation: TimeSeriesAggregation
    granularity: Granularity
    query_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "aggregation": self.aggregation.value,
            "granularity": self.granularity.value,
            "points": [
                {"timestamp": p.timestamp.isoformat(), "value": p.value, "tags": p.tags}
                for p in self.points
            ],
            "query_time_ms": self.query_time_ms,
        }


class TimeSeriesStore(ABC):
    """Abstract time-series storage backend."""

    @abstractmethod
    async def write(self, metric: str, point: TimeSeriesPoint) -> None:
        """Write a data point."""
        pass

    @abstractmethod
    async def write_batch(self, metric: str, points: List[TimeSeriesPoint]) -> None:
        """Write multiple points."""
        pass

    @abstractmethod
    async def query(self, query: TimeSeriesQuery) -> TimeSeriesResult:
        """Query time-series data."""
        pass

    @abstractmethod
    async def delete(self, metric: str, before: datetime) -> int:
        """Delete old data."""
        pass


class InMemoryTimeSeriesStore(TimeSeriesStore):
    """
    In-memory time-series store.

    Good for development and small datasets.
    Uses sorted list for efficient range queries.
    """

    def __init__(
        self,
        max_points_per_metric: int = 100000,
        retention_days: int = 7,
    ):
        self.max_points = max_points_per_metric
        self.retention = timedelta(days=retention_days)
        self._data: Dict[str, List[TimeSeriesPoint]] = defaultdict(list)
        self._timestamps: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def write(self, metric: str, point: TimeSeriesPoint) -> None:
        """Write a point."""
        async with self._lock:
            ts = point.timestamp.timestamp()
            idx = bisect.bisect_left(self._timestamps[metric], ts)
            self._timestamps[metric].insert(idx, ts)
            self._data[metric].insert(idx, point)

            # Trim if over limit
            if len(self._data[metric]) > self.max_points:
                self._data[metric] = self._data[metric][-self.max_points:]
                self._timestamps[metric] = self._timestamps[metric][-self.max_points:]

    async def write_batch(self, metric: str, points: List[TimeSeriesPoint]) -> None:
        """Write multiple points."""
        for point in points:
            await self.write(metric, point)

    async def query(self, query: TimeSeriesQuery) -> TimeSeriesResult:
        """Query with aggregation."""
        import time
        start_time = time.time()

        async with self._lock:
            # Get points in range
            start_ts = query.window.start.timestamp()
            end_ts = query.window.end.timestamp()

            timestamps = self._timestamps.get(query.metric, [])
            points = self._data.get(query.metric, [])

            start_idx = bisect.bisect_left(timestamps, start_ts)
            end_idx = bisect.bisect_right(timestamps, end_ts)

            range_points = points[start_idx:end_idx]

            # Filter by tags
            if query.tags:
                range_points = [
                    p for p in range_points
                    if all(p.tags.get(k) == v for k, v in query.tags.items())
                ]

            # Aggregate by granularity
            aggregated = self._aggregate_points(
                range_points,
                query.granularity,
                query.aggregation,
            )

            # Fill gaps if requested
            if query.fill:
                aggregated = self._fill_gaps(
                    aggregated,
                    query.window,
                    query.granularity,
                    query.fill,
                )

        return TimeSeriesResult(
            metric=query.metric,
            points=aggregated,
            aggregation=query.aggregation,
            granularity=query.granularity,
            query_time_ms=(time.time() - start_time) * 1000,
        )

    def _aggregate_points(
        self,
        points: List[TimeSeriesPoint],
        granularity: Granularity,
        aggregation: TimeSeriesAggregation,
    ) -> List[TimeSeriesPoint]:
        """Aggregate points by granularity."""
        if not points:
            return []

        bucket_seconds = granularity.to_seconds()
        buckets: Dict[int, List[float]] = defaultdict(list)

        for point in points:
            bucket = int(point.timestamp.timestamp() / bucket_seconds) * bucket_seconds
            buckets[bucket].append(point.value)

        result = []
        for bucket_ts in sorted(buckets.keys()):
            values = buckets[bucket_ts]
            agg_value = self._calculate_aggregation(values, aggregation)
            result.append(TimeSeriesPoint(
                timestamp=datetime.utcfromtimestamp(bucket_ts),
                value=agg_value,
            ))

        return result

    def _calculate_aggregation(
        self,
        values: List[float],
        aggregation: TimeSeriesAggregation,
    ) -> float:
        """Calculate aggregation function."""
        if not values:
            return 0.0

        if aggregation == TimeSeriesAggregation.SUM:
            return sum(values)
        elif aggregation == TimeSeriesAggregation.AVG:
            return statistics.mean(values)
        elif aggregation == TimeSeriesAggregation.MIN:
            return min(values)
        elif aggregation == TimeSeriesAggregation.MAX:
            return max(values)
        elif aggregation == TimeSeriesAggregation.COUNT:
            return len(values)
        elif aggregation == TimeSeriesAggregation.FIRST:
            return values[0]
        elif aggregation == TimeSeriesAggregation.LAST:
            return values[-1]
        elif aggregation == TimeSeriesAggregation.P50:
            return statistics.median(values)
        elif aggregation == TimeSeriesAggregation.P90:
            return self._percentile(values, 90)
        elif aggregation == TimeSeriesAggregation.P95:
            return self._percentile(values, 95)
        elif aggregation == TimeSeriesAggregation.P99:
            return self._percentile(values, 99)
        return 0.0

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def _fill_gaps(
        self,
        points: List[TimeSeriesPoint],
        window: TimeWindow,
        granularity: Granularity,
        fill_type: str,
    ) -> List[TimeSeriesPoint]:
        """Fill gaps in time series."""
        if not points:
            return []

        bucket_seconds = granularity.to_seconds()
        start_bucket = int(window.start.timestamp() / bucket_seconds) * bucket_seconds
        end_bucket = int(window.end.timestamp() / bucket_seconds) * bucket_seconds

        # Create map of existing points
        existing = {int(p.timestamp.timestamp()): p for p in points}

        result = []
        current = start_bucket
        previous_value = 0.0

        while current <= end_bucket:
            if current in existing:
                result.append(existing[current])
                previous_value = existing[current].value
            else:
                if fill_type == "zero":
                    fill_value = 0.0
                elif fill_type == "previous":
                    fill_value = previous_value
                else:  # null
                    fill_value = None

                if fill_value is not None:
                    result.append(TimeSeriesPoint(
                        timestamp=datetime.utcfromtimestamp(current),
                        value=fill_value,
                    ))

            current += bucket_seconds

        return result

    async def delete(self, metric: str, before: datetime) -> int:
        """Delete old data."""
        async with self._lock:
            if metric not in self._data:
                return 0

            before_ts = before.timestamp()
            idx = bisect.bisect_left(self._timestamps[metric], before_ts)

            deleted = idx
            self._data[metric] = self._data[metric][idx:]
            self._timestamps[metric] = self._timestamps[metric][idx:]

            return deleted

    async def cleanup(self) -> int:
        """Remove expired data."""
        cutoff = datetime.utcnow() - self.retention
        total_deleted = 0

        for metric in list(self._data.keys()):
            deleted = await self.delete(metric, cutoff)
            total_deleted += deleted

        return total_deleted


class RedisTimeSeriesStore(TimeSeriesStore):
    """
    Redis-backed time-series store.

    Uses Redis sorted sets for efficient storage and queries.
    """

    def __init__(
        self,
        redis_client,
        key_prefix: str = "ts",
        retention_seconds: int = 604800,  # 7 days
    ):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.retention = retention_seconds

    def _make_key(self, metric: str) -> str:
        """Create Redis key."""
        return f"{self.key_prefix}:{metric}"

    async def write(self, metric: str, point: TimeSeriesPoint) -> None:
        """Write to Redis sorted set."""
        key = self._make_key(metric)
        ts = point.timestamp.timestamp()

        # Store as JSON with tags
        import json
        data = json.dumps({"value": point.value, "tags": point.tags})

        await self.redis.zadd(key, {f"{ts}:{data}": ts})
        await self.redis.expire(key, self.retention)

    async def write_batch(self, metric: str, points: List[TimeSeriesPoint]) -> None:
        """Write batch to Redis."""
        if not points:
            return

        key = self._make_key(metric)
        import json

        mapping = {}
        for point in points:
            ts = point.timestamp.timestamp()
            data = json.dumps({"value": point.value, "tags": point.tags})
            mapping[f"{ts}:{data}"] = ts

        await self.redis.zadd(key, mapping)
        await self.redis.expire(key, self.retention)

    async def query(self, query: TimeSeriesQuery) -> TimeSeriesResult:
        """Query from Redis."""
        import time
        import json
        start_time = time.time()

        key = self._make_key(query.metric)
        start_ts = query.window.start.timestamp()
        end_ts = query.window.end.timestamp()

        # Get range from sorted set
        raw_data = await self.redis.zrangebyscore(key, start_ts, end_ts)

        points = []
        for item in raw_data:
            parts = item.decode().split(":", 1)
            ts = float(parts[0])
            data = json.loads(parts[1]) if len(parts) > 1 else {"value": 0}

            point = TimeSeriesPoint(
                timestamp=datetime.utcfromtimestamp(ts),
                value=data.get("value", 0),
                tags=data.get("tags", {}),
            )

            # Filter by tags
            if query.tags:
                if not all(point.tags.get(k) == v for k, v in query.tags.items()):
                    continue

            points.append(point)

        # Use InMemoryTimeSeriesStore's aggregation logic
        in_memory = InMemoryTimeSeriesStore()
        aggregated = in_memory._aggregate_points(
            points,
            query.granularity,
            query.aggregation,
        )

        return TimeSeriesResult(
            metric=query.metric,
            points=aggregated,
            aggregation=query.aggregation,
            granularity=query.granularity,
            query_time_ms=(time.time() - start_time) * 1000,
        )

    async def delete(self, metric: str, before: datetime) -> int:
        """Delete old data from Redis."""
        key = self._make_key(metric)
        before_ts = before.timestamp()
        return await self.redis.zremrangebyscore(key, "-inf", before_ts)


class TimeSeriesManager:
    """
    High-level time-series management.

    Handles multiple metrics with automatic downsampling and retention.
    """

    def __init__(
        self,
        store: TimeSeriesStore,
        downsample_configs: Optional[Dict[Granularity, int]] = None,
    ):
        """
        Args:
            store: Storage backend
            downsample_configs: Retention days for each granularity
        """
        self.store = store
        self.downsample_configs = downsample_configs or {
            Granularity.SECOND: 1,
            Granularity.MINUTE: 7,
            Granularity.HOUR: 30,
            Granularity.DAY: 365,
        }

        self._buffer: Dict[str, List[TimeSeriesPoint]] = defaultdict(list)
        self._buffer_size = 100
        self._lock = asyncio.Lock()

    async def record(
        self,
        metric: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a metric value."""
        point = TimeSeriesPoint(
            timestamp=timestamp or datetime.utcnow(),
            value=value,
            tags=tags or {},
        )

        async with self._lock:
            self._buffer[metric].append(point)

            if len(self._buffer[metric]) >= self._buffer_size:
                await self._flush_metric(metric)

    async def _flush_metric(self, metric: str) -> None:
        """Flush buffered points for a metric."""
        points = self._buffer[metric]
        self._buffer[metric] = []

        if points:
            await self.store.write_batch(metric, points)

    async def flush_all(self) -> None:
        """Flush all buffered data."""
        async with self._lock:
            for metric in list(self._buffer.keys()):
                await self._flush_metric(metric)

    async def query(
        self,
        metric: str,
        window: TimeWindow,
        granularity: Granularity = Granularity.MINUTE,
        aggregation: TimeSeriesAggregation = TimeSeriesAggregation.AVG,
        tags: Optional[Dict[str, str]] = None,
    ) -> TimeSeriesResult:
        """Query time-series data."""
        query = TimeSeriesQuery(
            metric=metric,
            window=window,
            granularity=granularity,
            aggregation=aggregation,
            tags=tags,
        )
        return await self.store.query(query)

    async def get_latest(self, metric: str) -> Optional[TimeSeriesPoint]:
        """Get the most recent point."""
        result = await self.query(
            metric=metric,
            window=TimeWindow.last_hours(1),
            granularity=Granularity.SECOND,
            aggregation=TimeSeriesAggregation.LAST,
        )
        return result.points[-1] if result.points else None

    async def get_rate(
        self,
        metric: str,
        window: TimeWindow,
    ) -> float:
        """Calculate rate of change."""
        result = await self.query(
            metric=metric,
            window=window,
            granularity=Granularity.MINUTE,
            aggregation=TimeSeriesAggregation.SUM,
        )

        if len(result.points) < 2:
            return 0.0

        first = result.points[0].value
        last = result.points[-1].value
        duration_minutes = (window.end - window.start).total_seconds() / 60

        return (last - first) / duration_minutes if duration_minutes > 0 else 0.0
