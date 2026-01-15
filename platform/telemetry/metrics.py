"""
Core Metrics System
===================

Comprehensive metrics collection with counters, gauges, histograms, and timers.

Author: Platform Observability Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class MetricType(str, Enum):
    """Types of metrics"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A metric value with labels"""

    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def with_labels(self, **labels: str) -> "MetricValue":
        """Create new value with additional labels"""
        new_labels = {**self.labels, **labels}
        return MetricValue(value=self.value, labels=new_labels, timestamp=self.timestamp)


@dataclass
class MetricMetadata:
    """Metadata for a metric"""

    name: str
    type: MetricType
    description: str = ""
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class Metric(ABC):
    """Base class for all metrics"""

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.unit = unit
        self.label_names = labels or []
        self._lock = threading.RLock()
        self._logger = structlog.get_logger(f"metric.{name}")

    @property
    @abstractmethod
    def type(self) -> MetricType:
        """Get the metric type"""
        pass

    @abstractmethod
    def collect(self) -> List[MetricValue]:
        """Collect all metric values"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the metric"""
        pass

    def _validate_labels(self, labels: Dict[str, str]) -> None:
        """Validate label names match declaration"""
        if self.label_names:
            provided = set(labels.keys())
            expected = set(self.label_names)
            if provided != expected:
                raise ValueError(
                    f"Label mismatch for {self.name}: expected {expected}, got {provided}"
                )

    def _labels_key(self, labels: Dict[str, str]) -> str:
        """Create a unique key from labels"""
        if not labels:
            return ""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))

    @property
    def metadata(self) -> MetricMetadata:
        """Get metric metadata"""
        return MetricMetadata(
            name=self.name,
            type=self.type,
            description=self.description,
            unit=self.unit,
            labels=self.label_names,
        )


class Counter(Metric):
    """
    A monotonically increasing counter.

    Usage:
        requests_total = Counter("requests_total", "Total requests", labels=["method", "status"])
        requests_total.inc(labels={"method": "POST", "status": "200"})
        requests_total.inc(5, labels={"method": "GET", "status": "200"})
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: Optional[List[str]] = None,
    ):
        super().__init__(name, description, unit, labels)
        self._values: Dict[str, float] = defaultdict(float)

    @property
    def type(self) -> MetricType:
        return MetricType.COUNTER

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter"""
        if amount < 0:
            raise ValueError("Counter can only be incremented with non-negative values")

        labels = labels or {}
        self._validate_labels(labels)
        key = self._labels_key(labels)

        with self._lock:
            self._values[key] += amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the current counter value"""
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> List[MetricValue]:
        """Collect all counter values"""
        with self._lock:
            values = []
            for key, value in self._values.items():
                labels = self._parse_labels_key(key)
                values.append(MetricValue(value=value, labels=labels))
            return values

    def reset(self) -> None:
        """Reset the counter"""
        with self._lock:
            self._values.clear()

    def _parse_labels_key(self, key: str) -> Dict[str, str]:
        """Parse labels from key"""
        if not key:
            return {}
        labels = {}
        for pair in key.split("|"):
            k, v = pair.split("=", 1)
            labels[k] = v
        return labels


class Gauge(Metric):
    """
    A gauge that can go up or down.

    Usage:
        active_connections = Gauge("active_connections", "Active connections")
        active_connections.set(42)
        active_connections.inc()
        active_connections.dec(5)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: Optional[List[str]] = None,
    ):
        super().__init__(name, description, unit, labels)
        self._values: Dict[str, float] = defaultdict(float)

    @property
    def type(self) -> MetricType:
        return MetricType.GAUGE

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge value"""
        labels = labels or {}
        self._validate_labels(labels)
        key = self._labels_key(labels)

        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the gauge"""
        labels = labels or {}
        self._validate_labels(labels)
        key = self._labels_key(labels)

        with self._lock:
            self._values[key] += amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement the gauge"""
        labels = labels or {}
        self._validate_labels(labels)
        key = self._labels_key(labels)

        with self._lock:
            self._values[key] -= amount

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the current gauge value"""
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            return self._values.get(key, 0.0)

    def set_to_current_time(self, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge to current Unix timestamp"""
        self.set(time.time(), labels)

    @contextmanager
    def track_inprogress(
        self, labels: Optional[Dict[str, str]] = None
    ) -> Iterator[None]:
        """Track in-progress operations"""
        self.inc(labels=labels)
        try:
            yield
        finally:
            self.dec(labels=labels)

    def collect(self) -> List[MetricValue]:
        """Collect all gauge values"""
        with self._lock:
            values = []
            for key, value in self._values.items():
                labels = self._parse_labels_key(key)
                values.append(MetricValue(value=value, labels=labels))
            return values

    def reset(self) -> None:
        """Reset the gauge"""
        with self._lock:
            self._values.clear()

    def _parse_labels_key(self, key: str) -> Dict[str, str]:
        """Parse labels from key"""
        if not key:
            return {}
        labels = {}
        for pair in key.split("|"):
            k, v = pair.split("=", 1)
            labels[k] = v
        return labels


@dataclass
class HistogramBucket:
    """A histogram bucket"""

    upper_bound: float
    count: int = 0


class Histogram(Metric):
    """
    A histogram for measuring distributions.

    Usage:
        request_latency = Histogram(
            "request_latency_seconds",
            "Request latency",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        request_latency.observe(0.123)
    """

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__(name, description, unit, labels)
        self._buckets = buckets or self.DEFAULT_BUCKETS
        self._data: Dict[str, Dict[str, Any]] = {}

    @property
    def type(self) -> MetricType:
        return MetricType.HISTOGRAM

    def _get_or_create_data(self, key: str) -> Dict[str, Any]:
        """Get or create histogram data for a label set"""
        if key not in self._data:
            self._data[key] = {
                "buckets": {b: 0 for b in self._buckets},
                "sum": 0.0,
                "count": 0,
                "min": float("inf"),
                "max": float("-inf"),
            }
        return self._data[key]

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observation"""
        labels = labels or {}
        self._validate_labels(labels)
        key = self._labels_key(labels)

        with self._lock:
            data = self._get_or_create_data(key)
            data["sum"] += value
            data["count"] += 1
            data["min"] = min(data["min"], value)
            data["max"] = max(data["max"], value)

            for bucket in self._buckets:
                if value <= bucket:
                    data["buckets"][bucket] += 1

    def get_snapshot(
        self, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get histogram snapshot"""
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            data = self._data.get(key, {})
            if not data:
                return {
                    "buckets": {b: 0 for b in self._buckets},
                    "sum": 0.0,
                    "count": 0,
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                }

            count = data["count"]
            return {
                "buckets": dict(data["buckets"]),
                "sum": data["sum"],
                "count": count,
                "min": data["min"] if count > 0 else 0.0,
                "max": data["max"] if count > 0 else 0.0,
                "mean": data["sum"] / count if count > 0 else 0.0,
            }

    def get_percentile(
        self, percentile: float, labels: Optional[Dict[str, str]] = None
    ) -> float:
        """Estimate percentile from histogram (approximate)"""
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            data = self._data.get(key)
            if not data or data["count"] == 0:
                return 0.0

            target = data["count"] * percentile
            cumulative = 0
            prev_bound = 0.0

            sorted_buckets = sorted(self._buckets)
            for bound in sorted_buckets:
                bucket_count = data["buckets"][bound]
                if cumulative + bucket_count >= target:
                    # Linear interpolation within bucket
                    fraction = (target - cumulative) / max(bucket_count, 1)
                    return prev_bound + (bound - prev_bound) * fraction
                cumulative += bucket_count
                prev_bound = bound

            return data["max"]

    def collect(self) -> List[MetricValue]:
        """Collect histogram values"""
        with self._lock:
            values = []
            for key, data in self._data.items():
                labels = self._parse_labels_key(key)

                # Bucket counts
                cumulative = 0
                for bucket in sorted(self._buckets):
                    cumulative += data["buckets"][bucket]
                    bucket_labels = {**labels, "le": str(bucket)}
                    values.append(
                        MetricValue(value=cumulative, labels=bucket_labels)
                    )

                # +Inf bucket
                inf_labels = {**labels, "le": "+Inf"}
                values.append(MetricValue(value=data["count"], labels=inf_labels))

                # Sum and count
                sum_labels = {**labels, "_type": "sum"}
                values.append(MetricValue(value=data["sum"], labels=sum_labels))

                count_labels = {**labels, "_type": "count"}
                values.append(MetricValue(value=data["count"], labels=count_labels))

            return values

    def reset(self) -> None:
        """Reset the histogram"""
        with self._lock:
            self._data.clear()

    def _parse_labels_key(self, key: str) -> Dict[str, str]:
        """Parse labels from key"""
        if not key:
            return {}
        labels = {}
        for pair in key.split("|"):
            k, v = pair.split("=", 1)
            labels[k] = v
        return labels


class Timer(Metric):
    """
    A timer for measuring durations.

    Usage:
        with request_timer.time(labels={"endpoint": "/api/v1/calls"}):
            process_request()

        # Or as decorator
        @request_timer.time_function()
        def process_request():
            pass
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__(name, description, "seconds", labels)
        self._histogram = Histogram(
            name=f"{name}_seconds",
            description=description,
            unit="seconds",
            labels=labels,
            buckets=buckets,
        )
        self._active_timers: Dict[str, float] = {}

    @property
    def type(self) -> MetricType:
        return MetricType.TIMER

    @contextmanager
    def time(self, labels: Optional[Dict[str, str]] = None) -> Iterator[None]:
        """Time a block of code"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._histogram.observe(duration, labels)

    def time_function(
        self, labels: Optional[Dict[str, str]] = None
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to time a function"""

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                with self.time(labels):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    async def time_async(
        self, labels: Optional[Dict[str, str]] = None
    ) -> "AsyncTimerContext":
        """Time an async block"""
        return AsyncTimerContext(self, labels)

    def start(self, timer_id: str) -> None:
        """Start a named timer"""
        with self._lock:
            self._active_timers[timer_id] = time.perf_counter()

    def stop(
        self, timer_id: str, labels: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """Stop a named timer and record duration"""
        with self._lock:
            start = self._active_timers.pop(timer_id, None)
            if start is None:
                return None
            duration = time.perf_counter() - start
            self._histogram.observe(duration, labels)
            return duration

    def get_snapshot(
        self, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get timer statistics"""
        return self._histogram.get_snapshot(labels)

    def collect(self) -> List[MetricValue]:
        """Collect timer values"""
        return self._histogram.collect()

    def reset(self) -> None:
        """Reset the timer"""
        self._histogram.reset()
        with self._lock:
            self._active_timers.clear()


class AsyncTimerContext:
    """Async context manager for timing"""

    def __init__(self, timer: Timer, labels: Optional[Dict[str, str]] = None):
        self._timer = timer
        self._labels = labels
        self._start: float = 0

    async def __aenter__(self) -> "AsyncTimerContext":
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration = time.perf_counter() - self._start
        self._timer._histogram.observe(duration, self._labels)


# =============================================================================
# METRICS REGISTRY
# =============================================================================


class MetricsRegistry:
    """
    Central registry for all metrics.

    Usage:
        registry = MetricsRegistry()
        counter = registry.counter("requests_total", "Total requests")
        gauge = registry.gauge("active_connections", "Active connections")
    """

    def __init__(self, prefix: str = ""):
        self._prefix = prefix
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.RLock()
        self._logger = structlog.get_logger("metrics_registry")

    def _full_name(self, name: str) -> str:
        """Get full metric name with prefix"""
        if self._prefix:
            return f"{self._prefix}_{name}"
        return name

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """Create or get a counter"""
        full_name = self._full_name(name)

        with self._lock:
            if full_name in self._metrics:
                metric = self._metrics[full_name]
                if not isinstance(metric, Counter):
                    raise ValueError(
                        f"Metric {full_name} already exists as {metric.type}"
                    )
                return metric

            metric = Counter(full_name, description, labels=labels)
            self._metrics[full_name] = metric
            return metric

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """Create or get a gauge"""
        full_name = self._full_name(name)

        with self._lock:
            if full_name in self._metrics:
                metric = self._metrics[full_name]
                if not isinstance(metric, Gauge):
                    raise ValueError(
                        f"Metric {full_name} already exists as {metric.type}"
                    )
                return metric

            metric = Gauge(full_name, description, labels=labels)
            self._metrics[full_name] = metric
            return metric

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Histogram:
        """Create or get a histogram"""
        full_name = self._full_name(name)

        with self._lock:
            if full_name in self._metrics:
                metric = self._metrics[full_name]
                if not isinstance(metric, Histogram):
                    raise ValueError(
                        f"Metric {full_name} already exists as {metric.type}"
                    )
                return metric

            metric = Histogram(full_name, description, labels=labels, buckets=buckets)
            self._metrics[full_name] = metric
            return metric

    def timer(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Timer:
        """Create or get a timer"""
        full_name = self._full_name(name)

        with self._lock:
            if full_name in self._metrics:
                metric = self._metrics[full_name]
                if not isinstance(metric, Timer):
                    raise ValueError(
                        f"Metric {full_name} already exists as {metric.type}"
                    )
                return metric

            metric = Timer(full_name, description, labels=labels, buckets=buckets)
            self._metrics[full_name] = metric
            return metric

    def register(self, metric: Metric) -> None:
        """Register an existing metric"""
        with self._lock:
            if metric.name in self._metrics:
                raise ValueError(f"Metric {metric.name} already registered")
            self._metrics[metric.name] = metric

    def unregister(self, name: str) -> bool:
        """Unregister a metric"""
        full_name = self._full_name(name)
        with self._lock:
            return self._metrics.pop(full_name, None) is not None

    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name"""
        full_name = self._full_name(name)
        with self._lock:
            return self._metrics.get(full_name)

    def list_metrics(self) -> List[MetricMetadata]:
        """List all registered metrics"""
        with self._lock:
            return [m.metadata for m in self._metrics.values()]

    def collect_all(self) -> Dict[str, List[MetricValue]]:
        """Collect all metric values"""
        with self._lock:
            return {name: metric.collect() for name, metric in self._metrics.items()}

    def reset_all(self) -> None:
        """Reset all metrics"""
        with self._lock:
            for metric in self._metrics.values():
                metric.reset()


# =============================================================================
# GLOBAL REGISTRY AND CONVENIENCE FUNCTIONS
# =============================================================================


_default_registry: Optional[MetricsRegistry] = None
_registry_lock = threading.Lock()


def get_registry(prefix: str = "") -> MetricsRegistry:
    """Get or create the default registry"""
    global _default_registry

    with _registry_lock:
        if _default_registry is None:
            _default_registry = MetricsRegistry(prefix)
        return _default_registry


def counter(
    name: str,
    description: str = "",
    labels: Optional[List[str]] = None,
) -> Counter:
    """Create a counter in the default registry"""
    return get_registry().counter(name, description, labels)


def gauge(
    name: str,
    description: str = "",
    labels: Optional[List[str]] = None,
) -> Gauge:
    """Create a gauge in the default registry"""
    return get_registry().gauge(name, description, labels)


def histogram(
    name: str,
    description: str = "",
    labels: Optional[List[str]] = None,
    buckets: Optional[Tuple[float, ...]] = None,
) -> Histogram:
    """Create a histogram in the default registry"""
    return get_registry().histogram(name, description, labels, buckets)


def timer(
    name: str,
    description: str = "",
    labels: Optional[List[str]] = None,
    buckets: Optional[Tuple[float, ...]] = None,
) -> Timer:
    """Create a timer in the default registry"""
    return get_registry().timer(name, description, labels, buckets)
