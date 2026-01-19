"""Prometheus-style metrics collection."""

from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import asyncio
import time
import logging
import functools
import threading

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricLabels:
    """Labels for a metric."""
    labels: Dict[str, str] = field(default_factory=dict)

    def to_key(self) -> str:
        """Convert labels to a unique key."""
        if not self.labels:
            return ""
        sorted_labels = sorted(self.labels.items())
        return ",".join(f'{k}="{v}"' for k, v in sorted_labels)

    def to_prometheus(self) -> str:
        """Convert to Prometheus format."""
        if not self.labels:
            return ""
        sorted_labels = sorted(self.labels.items())
        return "{" + ",".join(f'{k}="{v}"' for k, v in sorted_labels) + "}"


class Counter:
    """
    Monotonically increasing counter.

    Usage:
        requests = Counter("http_requests_total", "Total HTTP requests")
        requests.inc()
        requests.inc(labels={"method": "GET", "path": "/api/v1/agents"})
    """

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(
        self,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")

        key = MetricLabels(labels or {}).to_key()
        with self._lock:
            self._values[key] += value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current value."""
        key = MetricLabels(labels or {}).to_key()
        return self._values.get(key, 0.0)

    def labels(self, **kwargs) -> "LabeledCounter":
        """Return a labeled counter."""
        return LabeledCounter(self, kwargs)

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all values."""
        result = []
        with self._lock:
            for key, value in self._values.items():
                result.append({
                    "name": self.name,
                    "type": MetricType.COUNTER.value,
                    "labels": key,
                    "value": value,
                })
        return result

    def to_prometheus(self) -> str:
        """Export to Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            for key, value in self._values.items():
                label_str = f"{{{key}}}" if key else ""
                lines.append(f"{self.name}{label_str} {value}")
        return "\n".join(lines)


class LabeledCounter:
    """Counter with pre-set labels."""

    def __init__(self, counter: Counter, labels: Dict[str, str]):
        self._counter = counter
        self._labels = labels

    def inc(self, value: float = 1.0) -> None:
        """Increment counter."""
        self._counter.inc(value, self._labels)

    def get(self) -> float:
        """Get current value."""
        return self._counter.get(self._labels)


class Gauge:
    """
    Metric that can go up and down.

    Usage:
        active_calls = Gauge("active_calls", "Number of active calls")
        active_calls.set(10)
        active_calls.inc()
        active_calls.dec()
    """

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._values: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set gauge value."""
        key = MetricLabels(labels or {}).to_key()
        with self._lock:
            self._values[key] = value

    def inc(
        self,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment gauge."""
        key = MetricLabels(labels or {}).to_key()
        with self._lock:
            self._values[key] += value

    def dec(
        self,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Decrement gauge."""
        key = MetricLabels(labels or {}).to_key()
        with self._lock:
            self._values[key] -= value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current value."""
        key = MetricLabels(labels or {}).to_key()
        return self._values.get(key, 0.0)

    def labels(self, **kwargs) -> "LabeledGauge":
        """Return a labeled gauge."""
        return LabeledGauge(self, kwargs)

    def set_to_current_time(
        self,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set to current Unix timestamp."""
        self.set(time.time(), labels)

    def track_inprogress(self, labels: Optional[Dict[str, str]] = None):
        """Context manager to track in-progress operations."""
        return GaugeInProgress(self, labels)

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all values."""
        result = []
        with self._lock:
            for key, value in self._values.items():
                result.append({
                    "name": self.name,
                    "type": MetricType.GAUGE.value,
                    "labels": key,
                    "value": value,
                })
        return result

    def to_prometheus(self) -> str:
        """Export to Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            for key, value in self._values.items():
                label_str = f"{{{key}}}" if key else ""
                lines.append(f"{self.name}{label_str} {value}")
        return "\n".join(lines)


class LabeledGauge:
    """Gauge with pre-set labels."""

    def __init__(self, gauge: Gauge, labels: Dict[str, str]):
        self._gauge = gauge
        self._labels = labels

    def set(self, value: float) -> None:
        """Set gauge value."""
        self._gauge.set(value, self._labels)

    def inc(self, value: float = 1.0) -> None:
        """Increment gauge."""
        self._gauge.inc(value, self._labels)

    def dec(self, value: float = 1.0) -> None:
        """Decrement gauge."""
        self._gauge.dec(value, self._labels)

    def get(self) -> float:
        """Get current value."""
        return self._gauge.get(self._labels)


class GaugeInProgress:
    """Context manager for tracking in-progress operations."""

    def __init__(self, gauge: Gauge, labels: Optional[Dict[str, str]] = None):
        self._gauge = gauge
        self._labels = labels

    def __enter__(self):
        self._gauge.inc(1.0, self._labels)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._gauge.dec(1.0, self._labels)
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


class Histogram:
    """
    Histogram for measuring distributions.

    Usage:
        latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        latency.observe(0.25)

        # Or use timer
        with latency.time():
            process_request()
    """

    # Default buckets for latency measurements
    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf"),
    )

    def __init__(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        label_names: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.buckets = tuple(sorted(buckets or self.DEFAULT_BUCKETS))
        self.label_names = label_names or []

        self._bucket_counts: Dict[str, Dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self.buckets}
        )
        self._sum: Dict[str, float] = defaultdict(float)
        self._count: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a value."""
        key = MetricLabels(labels or {}).to_key()
        with self._lock:
            self._sum[key] += value
            self._count[key] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[key][bucket] += 1

    def labels(self, **kwargs) -> "LabeledHistogram":
        """Return a labeled histogram."""
        return LabeledHistogram(self, kwargs)

    def time(
        self,
        labels: Optional[Dict[str, str]] = None,
    ) -> "HistogramTimer":
        """Context manager for timing operations."""
        return HistogramTimer(self, labels)

    def get_percentile(
        self,
        percentile: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        """Estimate percentile from histogram."""
        key = MetricLabels(labels or {}).to_key()
        with self._lock:
            if key not in self._count or self._count[key] == 0:
                return None

            target = self._count[key] * percentile
            prev_bucket = 0.0
            prev_count = 0

            for bucket in self.buckets:
                count = self._bucket_counts[key][bucket]
                if count >= target:
                    # Linear interpolation
                    if count == prev_count:
                        return bucket
                    fraction = (target - prev_count) / (count - prev_count)
                    return prev_bucket + (bucket - prev_bucket) * fraction
                prev_bucket = bucket
                prev_count = count

            return self.buckets[-2] if len(self.buckets) > 1 else 0.0

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all values."""
        result = []
        with self._lock:
            for key in self._count.keys():
                result.append({
                    "name": self.name,
                    "type": MetricType.HISTOGRAM.value,
                    "labels": key,
                    "buckets": dict(self._bucket_counts[key]),
                    "sum": self._sum[key],
                    "count": self._count[key],
                })
        return result

    def to_prometheus(self) -> str:
        """Export to Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            for key in self._count.keys():
                base_labels = f"{{{key}," if key else "{"
                cumulative = 0
                for bucket in self.buckets:
                    cumulative += self._bucket_counts[key].get(bucket, 0)
                    le = "+Inf" if bucket == float("inf") else str(bucket)
                    lines.append(
                        f'{self.name}_bucket{base_labels}le="{le}"}} {cumulative}'
                    )
                label_str = f"{{{key}}}" if key else ""
                lines.append(f"{self.name}_sum{label_str} {self._sum[key]}")
                lines.append(f"{self.name}_count{label_str} {self._count[key]}")
        return "\n".join(lines)


class LabeledHistogram:
    """Histogram with pre-set labels."""

    def __init__(self, histogram: Histogram, labels: Dict[str, str]):
        self._histogram = histogram
        self._labels = labels

    def observe(self, value: float) -> None:
        """Observe a value."""
        self._histogram.observe(value, self._labels)

    def time(self) -> "HistogramTimer":
        """Context manager for timing."""
        return HistogramTimer(self._histogram, self._labels)


class HistogramTimer:
    """Timer context manager for histograms."""

    def __init__(
        self,
        histogram: Histogram,
        labels: Optional[Dict[str, str]] = None,
    ):
        self._histogram = histogram
        self._labels = labels
        self._start: Optional[float] = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is not None:
            duration = time.perf_counter() - self._start
            self._histogram.observe(duration, self._labels)
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


class Summary:
    """
    Summary metric that calculates quantiles over a sliding time window.

    Note: This is a simplified implementation. For production use with
    accurate quantiles, consider using a proper streaming algorithm.
    """

    def __init__(
        self,
        name: str,
        description: str,
        max_age_seconds: float = 600.0,
        age_buckets: int = 5,
        label_names: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.max_age_seconds = max_age_seconds
        self.age_buckets = age_buckets
        self.label_names = label_names or []

        self._observations: Dict[str, List[tuple]] = defaultdict(list)
        self._sum: Dict[str, float] = defaultdict(float)
        self._count: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a value."""
        key = MetricLabels(labels or {}).to_key()
        now = time.time()
        with self._lock:
            self._observations[key].append((now, value))
            self._sum[key] += value
            self._count[key] += 1
            # Cleanup old observations
            cutoff = now - self.max_age_seconds
            self._observations[key] = [
                (t, v) for t, v in self._observations[key] if t > cutoff
            ]

    def get_quantile(
        self,
        quantile: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        """Get a quantile value."""
        key = MetricLabels(labels or {}).to_key()
        now = time.time()
        cutoff = now - self.max_age_seconds

        with self._lock:
            values = [
                v for t, v in self._observations.get(key, [])
                if t > cutoff
            ]

        if not values:
            return None

        values.sort()
        idx = int(len(values) * quantile)
        idx = min(idx, len(values) - 1)
        return values[idx]

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all values."""
        result = []
        with self._lock:
            for key in self._count.keys():
                result.append({
                    "name": self.name,
                    "type": MetricType.SUMMARY.value,
                    "labels": key,
                    "sum": self._sum[key],
                    "count": self._count[key],
                })
        return result

    def to_prometheus(self) -> str:
        """Export to Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} summary",
        ]
        with self._lock:
            for key in self._count.keys():
                label_str = f"{{{key}}}" if key else ""
                lines.append(f"{self.name}_sum{label_str} {self._sum[key]}")
                lines.append(f"{self.name}_count{label_str} {self._count[key]}")
        return "\n".join(lines)


class MetricsRegistry:
    """
    Registry for all metrics.

    Usage:
        registry = MetricsRegistry()

        requests = registry.counter("http_requests_total", "Total requests")
        active = registry.gauge("active_connections", "Active connections")
        latency = registry.histogram("request_latency", "Request latency")

        # Export to Prometheus format
        print(registry.to_prometheus())
    """

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self._metrics: Dict[str, Union[Counter, Gauge, Histogram, Summary]] = {}
        self._lock = threading.Lock()

    def _full_name(self, name: str) -> str:
        """Get full metric name with prefix."""
        if self.prefix:
            return f"{self.prefix}_{name}"
        return name

    def counter(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
    ) -> Counter:
        """Create or get a counter."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Counter(
                    full_name, description, label_names
                )
            return self._metrics[full_name]

    def gauge(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
    ) -> Gauge:
        """Create or get a gauge."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Gauge(
                    full_name, description, label_names
                )
            return self._metrics[full_name]

    def histogram(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        label_names: Optional[List[str]] = None,
    ) -> Histogram:
        """Create or get a histogram."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Histogram(
                    full_name, description, buckets, label_names
                )
            return self._metrics[full_name]

    def summary(
        self,
        name: str,
        description: str,
        max_age_seconds: float = 600.0,
        label_names: Optional[List[str]] = None,
    ) -> Summary:
        """Create or get a summary."""
        full_name = self._full_name(name)
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Summary(
                    full_name, description, max_age_seconds, label_names
                )
            return self._metrics[full_name]

    def collect_all(self) -> List[Dict[str, Any]]:
        """Collect all metrics."""
        result = []
        with self._lock:
            for metric in self._metrics.values():
                result.extend(metric.collect())
        return result

    def to_prometheus(self) -> str:
        """Export all metrics to Prometheus format."""
        lines = []
        with self._lock:
            for metric in self._metrics.values():
                lines.append(metric.to_prometheus())
        return "\n\n".join(lines)


# Global registry
_registry: Optional[MetricsRegistry] = None


def get_registry(prefix: str = "bvrai") -> MetricsRegistry:
    """Get or create the global metrics registry."""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry(prefix)
    return _registry


# Convenience functions
def counter(name: str, description: str) -> Counter:
    """Create a counter in the global registry."""
    return get_registry().counter(name, description)


def gauge(name: str, description: str) -> Gauge:
    """Create a gauge in the global registry."""
    return get_registry().gauge(name, description)


def histogram(
    name: str,
    description: str,
    buckets: Optional[List[float]] = None,
) -> Histogram:
    """Create a histogram in the global registry."""
    return get_registry().histogram(name, description, buckets)


def summary(name: str, description: str) -> Summary:
    """Create a summary in the global registry."""
    return get_registry().summary(name, description)


# Pre-defined metrics for the platform
class PlatformMetrics:
    """Pre-defined metrics for the voice AI platform."""

    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or get_registry()

        # HTTP metrics
        self.http_requests_total = self.registry.counter(
            "http_requests_total",
            "Total HTTP requests",
            label_names=["method", "path", "status"],
        )
        self.http_request_duration_seconds = self.registry.histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            label_names=["method", "path"],
        )
        self.http_requests_in_progress = self.registry.gauge(
            "http_requests_in_progress",
            "Number of HTTP requests in progress",
            label_names=["method"],
        )

        # Call metrics
        self.calls_total = self.registry.counter(
            "calls_total",
            "Total voice calls",
            label_names=["agent_id", "direction", "status"],
        )
        self.calls_active = self.registry.gauge(
            "calls_active",
            "Number of active calls",
            label_names=["agent_id"],
        )
        self.call_duration_seconds = self.registry.histogram(
            "call_duration_seconds",
            "Call duration in seconds",
            buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],
            label_names=["agent_id"],
        )

        # ASR metrics
        self.asr_requests_total = self.registry.counter(
            "asr_requests_total",
            "Total ASR requests",
            label_names=["provider", "language"],
        )
        self.asr_latency_seconds = self.registry.histogram(
            "asr_latency_seconds",
            "ASR latency in seconds",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            label_names=["provider"],
        )

        # TTS metrics
        self.tts_requests_total = self.registry.counter(
            "tts_requests_total",
            "Total TTS requests",
            label_names=["provider", "voice"],
        )
        self.tts_latency_seconds = self.registry.histogram(
            "tts_latency_seconds",
            "TTS latency in seconds",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            label_names=["provider"],
        )
        self.tts_characters_total = self.registry.counter(
            "tts_characters_total",
            "Total TTS characters processed",
            label_names=["provider"],
        )

        # LLM metrics
        self.llm_requests_total = self.registry.counter(
            "llm_requests_total",
            "Total LLM requests",
            label_names=["provider", "model"],
        )
        self.llm_latency_seconds = self.registry.histogram(
            "llm_latency_seconds",
            "LLM latency in seconds",
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            label_names=["provider", "model"],
        )
        self.llm_tokens_total = self.registry.counter(
            "llm_tokens_total",
            "Total LLM tokens",
            label_names=["provider", "model", "type"],  # type: prompt/completion
        )

        # WebSocket metrics
        self.websocket_connections = self.registry.gauge(
            "websocket_connections",
            "Active WebSocket connections",
        )
        self.websocket_messages_total = self.registry.counter(
            "websocket_messages_total",
            "Total WebSocket messages",
            label_names=["direction"],  # sent/received
        )

        # Queue metrics
        self.job_queue_size = self.registry.gauge(
            "job_queue_size",
            "Number of jobs in queue",
            label_names=["queue", "status"],
        )
        self.job_processing_time_seconds = self.registry.histogram(
            "job_processing_time_seconds",
            "Job processing time in seconds",
            label_names=["queue", "job_type"],
        )

        # Cache metrics
        self.cache_hits_total = self.registry.counter(
            "cache_hits_total",
            "Cache hits",
            label_names=["cache"],
        )
        self.cache_misses_total = self.registry.counter(
            "cache_misses_total",
            "Cache misses",
            label_names=["cache"],
        )

        # Circuit breaker metrics
        self.circuit_breaker_state = self.registry.gauge(
            "circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half-open, 2=open)",
            label_names=["name"],
        )
        self.circuit_breaker_failures_total = self.registry.counter(
            "circuit_breaker_failures_total",
            "Circuit breaker failures",
            label_names=["name"],
        )


# Global platform metrics
_platform_metrics: Optional[PlatformMetrics] = None


def get_platform_metrics() -> PlatformMetrics:
    """Get or create platform metrics."""
    global _platform_metrics
    if _platform_metrics is None:
        _platform_metrics = PlatformMetrics()
    return _platform_metrics


def timed(histogram: Histogram, labels: Optional[Dict[str, str]] = None):
    """
    Decorator for timing functions.

    Usage:
        @timed(request_latency, {"endpoint": "/api/v1/agents"})
        async def handle_request():
            ...
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with histogram.time(labels):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with histogram.time(labels):
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator


def counted(
    counter: Counter,
    labels: Optional[Dict[str, str]] = None,
    exceptions: Optional[Dict[str, str]] = None,
):
    """
    Decorator for counting function calls.

    Usage:
        @counted(request_counter, {"endpoint": "/api/v1/agents"})
        async def handle_request():
            ...
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    counter.inc(labels=labels)
                    return result
                except Exception as e:
                    if exceptions:
                        error_labels = {**(labels or {}), **exceptions}
                        counter.inc(labels=error_labels)
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    counter.inc(labels=labels)
                    return result
                except Exception as e:
                    if exceptions:
                        error_labels = {**(labels or {}), **exceptions}
                        counter.inc(labels=error_labels)
                    raise
            return sync_wrapper
    return decorator
