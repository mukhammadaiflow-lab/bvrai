"""
Latency Tracking and Optimization Module.

This module provides comprehensive latency tracking, analysis, and
optimization recommendations for the streaming pipeline.
"""

import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LatencyComponent(str, Enum):
    """Components tracked for latency."""

    ASR = "asr"
    LLM_FIRST_TOKEN = "llm_first_token"
    LLM_TOTAL = "llm_total"
    TTS_FIRST_AUDIO = "tts_first_audio"
    TTS_TOTAL = "tts_total"
    E2E = "e2e"
    NETWORK_INBOUND = "network_inbound"
    NETWORK_OUTBOUND = "network_outbound"
    VAD = "vad"
    AUDIO_PROCESSING = "audio_processing"


class LatencyStatus(str, Enum):
    """Status of latency relative to targets."""

    EXCELLENT = "excellent"  # < 50% of target
    GOOD = "good"            # < 100% of target
    WARNING = "warning"      # < 150% of target
    CRITICAL = "critical"    # >= 150% of target


@dataclass
class LatencyMeasurement:
    """A single latency measurement."""

    component: LatencyComponent
    value_ms: float
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    turn_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Statistical summary of latency measurements."""

    component: LatencyComponent
    count: int = 0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    avg_ms: float = 0.0
    median_ms: float = 0.0
    stddev_ms: float = 0.0
    p50_ms: float = 0.0
    p75_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    last_ms: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component.value,
            "count": self.count,
            "min_ms": round(self.min_ms, 2) if self.min_ms != float("inf") else None,
            "max_ms": round(self.max_ms, 2),
            "avg_ms": round(self.avg_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "stddev_ms": round(self.stddev_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p75_ms": round(self.p75_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "last_ms": round(self.last_ms, 2),
        }


@dataclass
class LatencyTarget:
    """Target latency for a component."""

    component: LatencyComponent
    target_ms: float
    warning_ms: float
    critical_ms: float

    def get_status(self, value_ms: float) -> LatencyStatus:
        """Get status for a latency value."""
        if value_ms <= self.target_ms * 0.5:
            return LatencyStatus.EXCELLENT
        elif value_ms <= self.target_ms:
            return LatencyStatus.GOOD
        elif value_ms <= self.warning_ms:
            return LatencyStatus.WARNING
        else:
            return LatencyStatus.CRITICAL


@dataclass
class LatencyAlert:
    """An alert for latency issues."""

    component: LatencyComponent
    status: LatencyStatus
    measured_ms: float
    target_ms: float
    threshold_ms: float
    message: str
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None


class LatencyTracker:
    """
    Comprehensive latency tracking and analysis.

    Features:
    - Rolling window statistics for each component
    - Percentile calculations (p50, p75, p90, p95, p99)
    - Alert generation for threshold violations
    - Optimization recommendations
    - Export to Prometheus metrics format
    """

    def __init__(
        self,
        window_size: int = 1000,
        alert_callback: Optional[Callable[[LatencyAlert], None]] = None,
    ):
        """
        Initialize latency tracker.

        Args:
            window_size: Number of measurements to keep in rolling window
            alert_callback: Optional callback for alerts
        """
        self._window_size = window_size
        self._alert_callback = alert_callback

        # Rolling windows for each component
        self._measurements: Dict[LatencyComponent, Deque[float]] = {
            component: deque(maxlen=window_size)
            for component in LatencyComponent
        }

        # Targets for each component
        self._targets: Dict[LatencyComponent, LatencyTarget] = {}
        self._setup_default_targets()

        # Cached stats (updated periodically)
        self._stats_cache: Dict[LatencyComponent, LatencyStats] = {}
        self._stats_cache_time: float = 0.0
        self._stats_cache_ttl: float = 1.0  # seconds

        # Alert tracking
        self._last_alert_time: Dict[LatencyComponent, float] = {}
        self._alert_cooldown: float = 10.0  # seconds between alerts

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def _setup_default_targets(self) -> None:
        """Setup default latency targets."""
        # Based on ultra-low latency profile
        defaults = {
            LatencyComponent.ASR: (100, 150, 200),
            LatencyComponent.LLM_FIRST_TOKEN: (150, 225, 300),
            LatencyComponent.LLM_TOTAL: (500, 750, 1000),
            LatencyComponent.TTS_FIRST_AUDIO: (75, 112, 150),
            LatencyComponent.TTS_TOTAL: (300, 450, 600),
            LatencyComponent.E2E: (300, 450, 600),
            LatencyComponent.NETWORK_INBOUND: (20, 30, 50),
            LatencyComponent.NETWORK_OUTBOUND: (20, 30, 50),
            LatencyComponent.VAD: (10, 15, 25),
            LatencyComponent.AUDIO_PROCESSING: (5, 10, 20),
        }

        for component, (target, warning, critical) in defaults.items():
            self._targets[component] = LatencyTarget(
                component=component,
                target_ms=target,
                warning_ms=warning,
                critical_ms=critical,
            )

    def set_targets(
        self,
        component: LatencyComponent,
        target_ms: float,
        warning_ms: Optional[float] = None,
        critical_ms: Optional[float] = None,
    ) -> None:
        """Set latency targets for a component."""
        if warning_ms is None:
            warning_ms = target_ms * 1.5
        if critical_ms is None:
            critical_ms = target_ms * 2.0

        self._targets[component] = LatencyTarget(
            component=component,
            target_ms=target_ms,
            warning_ms=warning_ms,
            critical_ms=critical_ms,
        )

    async def record(
        self,
        component: LatencyComponent,
        value_ms: float,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[LatencyAlert]:
        """
        Record a latency measurement.

        Args:
            component: The component being measured
            value_ms: Latency value in milliseconds
            session_id: Optional session identifier
            turn_id: Optional turn identifier
            metadata: Optional additional metadata

        Returns:
            LatencyAlert if threshold exceeded, None otherwise
        """
        async with self._lock:
            # Add to rolling window
            self._measurements[component].append(value_ms)

            # Invalidate stats cache
            self._stats_cache.pop(component, None)

        # Check for alerts
        alert = self._check_alert(component, value_ms, session_id)

        if alert and self._alert_callback:
            try:
                self._alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        return alert

    def _check_alert(
        self,
        component: LatencyComponent,
        value_ms: float,
        session_id: Optional[str] = None,
    ) -> Optional[LatencyAlert]:
        """Check if an alert should be generated."""
        target = self._targets.get(component)
        if not target:
            return None

        status = target.get_status(value_ms)

        # Only alert for warning or critical
        if status not in (LatencyStatus.WARNING, LatencyStatus.CRITICAL):
            return None

        # Check cooldown
        now = time.time()
        last_alert = self._last_alert_time.get(component, 0)
        if now - last_alert < self._alert_cooldown:
            return None

        self._last_alert_time[component] = now

        # Generate alert
        threshold = target.warning_ms if status == LatencyStatus.WARNING else target.critical_ms

        return LatencyAlert(
            component=component,
            status=status,
            measured_ms=value_ms,
            target_ms=target.target_ms,
            threshold_ms=threshold,
            message=f"{component.value} latency {status.value}: {value_ms:.1f}ms (target: {target.target_ms:.1f}ms)",
            session_id=session_id,
        )

    async def get_stats(self, component: LatencyComponent) -> LatencyStats:
        """Get statistics for a component."""
        now = time.time()

        # Check cache
        if (
            component in self._stats_cache
            and now - self._stats_cache_time < self._stats_cache_ttl
        ):
            return self._stats_cache[component]

        async with self._lock:
            measurements = list(self._measurements[component])

        if not measurements:
            return LatencyStats(component=component)

        # Calculate statistics
        sorted_values = sorted(measurements)
        n = len(sorted_values)

        stats = LatencyStats(
            component=component,
            count=n,
            min_ms=min(sorted_values),
            max_ms=max(sorted_values),
            avg_ms=statistics.mean(sorted_values),
            median_ms=statistics.median(sorted_values),
            stddev_ms=statistics.stdev(sorted_values) if n > 1 else 0.0,
            p50_ms=self._percentile(sorted_values, 50),
            p75_ms=self._percentile(sorted_values, 75),
            p90_ms=self._percentile(sorted_values, 90),
            p95_ms=self._percentile(sorted_values, 95),
            p99_ms=self._percentile(sorted_values, 99),
            last_ms=measurements[-1] if measurements else 0.0,
            last_updated=now,
        )

        # Cache stats
        self._stats_cache[component] = stats
        self._stats_cache_time = now

        return stats

    async def get_all_stats(self) -> Dict[str, LatencyStats]:
        """Get statistics for all components."""
        stats = {}
        for component in LatencyComponent:
            stats[component.value] = await self.get_stats(component)
        return stats

    @staticmethod
    def _percentile(sorted_values: List[float], p: int) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0

        n = len(sorted_values)
        idx = (n - 1) * p / 100

        lower = int(idx)
        upper = lower + 1
        weight = idx - lower

        if upper >= n:
            return sorted_values[-1]

        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    async def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all latency metrics."""
        all_stats = await self.get_all_stats()

        # Calculate overall health
        warning_count = 0
        critical_count = 0

        for component, stats in all_stats.items():
            target = self._targets.get(LatencyComponent(component))
            if target and stats.p95_ms > 0:
                status = target.get_status(stats.p95_ms)
                if status == LatencyStatus.WARNING:
                    warning_count += 1
                elif status == LatencyStatus.CRITICAL:
                    critical_count += 1

        if critical_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        return {
            "status": overall_status,
            "warning_count": warning_count,
            "critical_count": critical_count,
            "components": {k: v.to_dict() for k, v in all_stats.items()},
            "targets": {
                k.value: {
                    "target_ms": v.target_ms,
                    "warning_ms": v.warning_ms,
                    "critical_ms": v.critical_ms,
                }
                for k, v in self._targets.items()
            },
        }

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for component in LatencyComponent:
            measurements = list(self._measurements[component])
            if not measurements:
                continue

            name = f"streaming_latency_{component.value}"

            # Histogram
            lines.append(f"# HELP {name}_ms Latency in milliseconds")
            lines.append(f"# TYPE {name}_ms histogram")

            # Calculate buckets
            buckets = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, float("inf")]
            counts = [0] * len(buckets)

            for value in measurements:
                for i, bucket in enumerate(buckets):
                    if value <= bucket:
                        counts[i] += 1

            cumulative = 0
            for bucket, count in zip(buckets, counts):
                cumulative += count
                if bucket == float("inf"):
                    lines.append(f'{name}_ms_bucket{{le="+Inf"}} {cumulative}')
                else:
                    lines.append(f'{name}_ms_bucket{{le="{bucket}"}} {cumulative}')

            lines.append(f"{name}_ms_sum {sum(measurements):.2f}")
            lines.append(f"{name}_ms_count {len(measurements)}")
            lines.append("")

        return "\n".join(lines)

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for latency optimization."""
        recommendations = []

        for component, target in self._targets.items():
            measurements = list(self._measurements[component])
            if not measurements:
                continue

            p95 = self._percentile(sorted(measurements), 95)
            status = target.get_status(p95)

            if status in (LatencyStatus.WARNING, LatencyStatus.CRITICAL):
                rec = {
                    "component": component.value,
                    "severity": status.value,
                    "current_p95_ms": round(p95, 2),
                    "target_ms": target.target_ms,
                    "recommendations": [],
                }

                # Component-specific recommendations
                if component == LatencyComponent.ASR:
                    rec["recommendations"] = [
                        "Consider using Deepgram Nova-2 for faster transcription",
                        "Enable speculative execution to overlap ASR with LLM",
                        "Reduce input audio buffer size",
                        "Check network latency to ASR provider",
                    ]
                elif component == LatencyComponent.LLM_FIRST_TOKEN:
                    rec["recommendations"] = [
                        "Use Groq or Cerebras for lowest latency inference",
                        "Reduce system prompt length",
                        "Enable speculative execution on stable partials",
                        "Consider using smaller, faster models",
                    ]
                elif component == LatencyComponent.TTS_FIRST_AUDIO:
                    rec["recommendations"] = [
                        "Use Cartesia Sonic for 40-95ms latency",
                        "Enable streaming TTS",
                        "Pre-buffer common responses",
                        "Use edge-deployed TTS if available",
                    ]
                elif component == LatencyComponent.E2E:
                    rec["recommendations"] = [
                        "Enable full pipeline streaming",
                        "Use speculative execution",
                        "Consider edge deployment",
                        "Review and optimize all component latencies",
                    ]

                recommendations.append(rec)

        return recommendations

    async def reset(self) -> None:
        """Reset all measurements."""
        async with self._lock:
            for component in LatencyComponent:
                self._measurements[component].clear()
            self._stats_cache.clear()
            self._stats_cache_time = 0.0


class LatencyContext:
    """
    Context manager for measuring latency of operations.

    Usage:
        async with LatencyContext(tracker, LatencyComponent.ASR, session_id="123") as ctx:
            result = await some_operation()
            ctx.add_metadata("tokens", 100)
    """

    def __init__(
        self,
        tracker: LatencyTracker,
        component: LatencyComponent,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ):
        self.tracker = tracker
        self.component = component
        self.session_id = session_id
        self.turn_id = turn_id
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.metadata: Dict[str, Any] = {}

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the measurement."""
        self.metadata[key] = value

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    async def __aenter__(self) -> "LatencyContext":
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.time()
        elapsed_ms = (self.end_time - self.start_time) * 1000

        await self.tracker.record(
            component=self.component,
            value_ms=elapsed_ms,
            session_id=self.session_id,
            turn_id=self.turn_id,
            metadata=self.metadata,
        )


# Convenience function for timing
def timed_section(
    tracker: LatencyTracker,
    component: LatencyComponent,
    session_id: Optional[str] = None,
) -> LatencyContext:
    """Create a latency measurement context."""
    return LatencyContext(tracker, component, session_id)
