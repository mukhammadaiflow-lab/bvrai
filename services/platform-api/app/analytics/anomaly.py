"""
Anomaly Detection

Real-time anomaly detection with:
- Statistical analysis (Z-score, IQR)
- Threshold-based detection
- Trend analysis
- Pattern recognition
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
import asyncio
import statistics
import math
import logging

logger = logging.getLogger(__name__)


class AnomalySeverity(str, Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(str, Enum):
    """Types of anomalies."""
    SPIKE = "spike"  # Sudden increase
    DROP = "drop"  # Sudden decrease
    THRESHOLD_BREACH = "threshold_breach"
    TREND_CHANGE = "trend_change"
    PATTERN_DEVIATION = "pattern_deviation"
    STATISTICAL_OUTLIER = "statistical_outlier"


@dataclass
class AnomalyAlert:
    """An anomaly detection alert."""
    alert_id: str
    metric: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    value: float
    expected_value: float
    deviation: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "metric": self.metric,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "value": self.value,
            "expected_value": self.expected_value,
            "deviation": self.deviation,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "context": self.context,
        }


@dataclass
class DetectorConfig:
    """Configuration for anomaly detector."""
    metric: str
    enabled: bool = True
    sensitivity: float = 2.0  # Standard deviations for statistical
    window_size: int = 60  # Number of samples
    min_samples: int = 10  # Minimum samples before detection
    cooldown_seconds: float = 300.0  # Cooldown between alerts


class AnomalyDetector(ABC):
    """Abstract base for anomaly detectors."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._last_alert: Optional[datetime] = None

    @abstractmethod
    async def analyze(self, value: float, timestamp: Optional[datetime] = None) -> Optional[AnomalyAlert]:
        """Analyze a value for anomalies."""
        pass

    def _can_alert(self) -> bool:
        """Check if cooldown has passed."""
        if self._last_alert is None:
            return True
        elapsed = (datetime.utcnow() - self._last_alert).total_seconds()
        return elapsed >= self.config.cooldown_seconds

    def _record_alert(self) -> None:
        """Record that an alert was sent."""
        self._last_alert = datetime.utcnow()


class StatisticalDetector(AnomalyDetector):
    """
    Statistical anomaly detection using Z-score.

    Detects values that deviate significantly from the mean.
    """

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self._values: deque = deque(maxlen=config.window_size)
        self._mean: float = 0.0
        self._std: float = 0.0

    async def analyze(self, value: float, timestamp: Optional[datetime] = None) -> Optional[AnomalyAlert]:
        """Analyze value using Z-score."""
        if not self.config.enabled:
            return None

        self._values.append(value)

        if len(self._values) < self.config.min_samples:
            return None

        # Calculate statistics
        values = list(self._values)
        self._mean = statistics.mean(values)
        self._std = statistics.stdev(values) if len(values) > 1 else 0.0

        # Avoid division by zero
        if self._std == 0:
            return None

        # Calculate Z-score
        z_score = abs(value - self._mean) / self._std

        if z_score > self.config.sensitivity and self._can_alert():
            self._record_alert()

            # Determine type and severity
            anomaly_type = AnomalyType.SPIKE if value > self._mean else AnomalyType.DROP
            severity = self._calculate_severity(z_score)

            import uuid
            return AnomalyAlert(
                alert_id=str(uuid.uuid4()),
                metric=self.config.metric,
                anomaly_type=anomaly_type,
                severity=severity,
                value=value,
                expected_value=self._mean,
                deviation=z_score,
                timestamp=timestamp or datetime.utcnow(),
                message=f"Value {value:.2f} is {z_score:.1f} std devs from mean {self._mean:.2f}",
                context={
                    "z_score": z_score,
                    "mean": self._mean,
                    "std": self._std,
                    "window_size": len(self._values),
                },
            )

        return None

    def _calculate_severity(self, z_score: float) -> AnomalySeverity:
        """Calculate severity based on Z-score."""
        if z_score >= 4.0:
            return AnomalySeverity.CRITICAL
        elif z_score >= 3.0:
            return AnomalySeverity.HIGH
        elif z_score >= 2.5:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW


class ThresholdDetector(AnomalyDetector):
    """
    Threshold-based anomaly detection.

    Simple but effective for known boundaries.
    """

    def __init__(
        self,
        config: DetectorConfig,
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
        warning_min: Optional[float] = None,
        warning_max: Optional[float] = None,
    ):
        super().__init__(config)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.warning_min = warning_min
        self.warning_max = warning_max

    async def analyze(self, value: float, timestamp: Optional[datetime] = None) -> Optional[AnomalyAlert]:
        """Check value against thresholds."""
        if not self.config.enabled:
            return None

        import uuid
        ts = timestamp or datetime.utcnow()

        # Check critical thresholds
        if self.max_threshold is not None and value > self.max_threshold:
            if self._can_alert():
                self._record_alert()
                return AnomalyAlert(
                    alert_id=str(uuid.uuid4()),
                    metric=self.config.metric,
                    anomaly_type=AnomalyType.THRESHOLD_BREACH,
                    severity=AnomalySeverity.CRITICAL,
                    value=value,
                    expected_value=self.max_threshold,
                    deviation=(value - self.max_threshold) / self.max_threshold if self.max_threshold else 0,
                    timestamp=ts,
                    message=f"Value {value:.2f} exceeds maximum threshold {self.max_threshold:.2f}",
                    context={"threshold_type": "max", "threshold": self.max_threshold},
                )

        if self.min_threshold is not None and value < self.min_threshold:
            if self._can_alert():
                self._record_alert()
                return AnomalyAlert(
                    alert_id=str(uuid.uuid4()),
                    metric=self.config.metric,
                    anomaly_type=AnomalyType.THRESHOLD_BREACH,
                    severity=AnomalySeverity.CRITICAL,
                    value=value,
                    expected_value=self.min_threshold,
                    deviation=(self.min_threshold - value) / self.min_threshold if self.min_threshold else 0,
                    timestamp=ts,
                    message=f"Value {value:.2f} below minimum threshold {self.min_threshold:.2f}",
                    context={"threshold_type": "min", "threshold": self.min_threshold},
                )

        # Check warning thresholds
        if self.warning_max is not None and value > self.warning_max:
            if self._can_alert():
                self._record_alert()
                return AnomalyAlert(
                    alert_id=str(uuid.uuid4()),
                    metric=self.config.metric,
                    anomaly_type=AnomalyType.THRESHOLD_BREACH,
                    severity=AnomalySeverity.MEDIUM,
                    value=value,
                    expected_value=self.warning_max,
                    deviation=(value - self.warning_max) / self.warning_max if self.warning_max else 0,
                    timestamp=ts,
                    message=f"Value {value:.2f} approaching maximum (warning: {self.warning_max:.2f})",
                    context={"threshold_type": "warning_max", "threshold": self.warning_max},
                )

        return None


class TrendDetector(AnomalyDetector):
    """
    Trend-based anomaly detection.

    Detects sudden changes in trend direction or rate.
    """

    def __init__(self, config: DetectorConfig, trend_threshold: float = 0.5):
        super().__init__(config)
        self._values: deque = deque(maxlen=config.window_size)
        self._timestamps: deque = deque(maxlen=config.window_size)
        self.trend_threshold = trend_threshold
        self._last_trend: Optional[float] = None

    async def analyze(self, value: float, timestamp: Optional[datetime] = None) -> Optional[AnomalyAlert]:
        """Analyze trend changes."""
        if not self.config.enabled:
            return None

        ts = timestamp or datetime.utcnow()
        self._values.append(value)
        self._timestamps.append(ts.timestamp())

        if len(self._values) < self.config.min_samples:
            return None

        # Calculate current trend (simple linear regression slope)
        trend = self._calculate_trend()

        if self._last_trend is not None:
            # Check for trend reversal
            if trend * self._last_trend < 0:  # Signs are opposite
                trend_change = abs(trend - self._last_trend)

                if trend_change > self.trend_threshold and self._can_alert():
                    self._record_alert()

                    import uuid
                    return AnomalyAlert(
                        alert_id=str(uuid.uuid4()),
                        metric=self.config.metric,
                        anomaly_type=AnomalyType.TREND_CHANGE,
                        severity=AnomalySeverity.HIGH,
                        value=value,
                        expected_value=self._last_trend,
                        deviation=trend_change,
                        timestamp=ts,
                        message=f"Trend reversal detected: {self._last_trend:.3f} -> {trend:.3f}",
                        context={
                            "previous_trend": self._last_trend,
                            "current_trend": trend,
                            "change": trend_change,
                        },
                    )

        self._last_trend = trend
        return None

    def _calculate_trend(self) -> float:
        """Calculate trend using linear regression slope."""
        if len(self._values) < 2:
            return 0.0

        values = list(self._values)
        times = list(self._timestamps)

        # Normalize timestamps
        t_start = times[0]
        times = [t - t_start for t in times]

        n = len(values)
        sum_x = sum(times)
        sum_y = sum(values)
        sum_xy = sum(t * v for t, v in zip(times, values))
        sum_x2 = sum(t ** 2 for t in times)

        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope


class IQRDetector(AnomalyDetector):
    """
    Interquartile Range (IQR) based detector.

    More robust to outliers than standard deviation.
    """

    def __init__(self, config: DetectorConfig, multiplier: float = 1.5):
        super().__init__(config)
        self._values: deque = deque(maxlen=config.window_size)
        self.multiplier = multiplier

    async def analyze(self, value: float, timestamp: Optional[datetime] = None) -> Optional[AnomalyAlert]:
        """Detect outliers using IQR."""
        if not self.config.enabled:
            return None

        self._values.append(value)

        if len(self._values) < self.config.min_samples:
            return None

        sorted_values = sorted(self._values)
        q1 = sorted_values[len(sorted_values) // 4]
        q3 = sorted_values[3 * len(sorted_values) // 4]
        iqr = q3 - q1

        lower_bound = q1 - self.multiplier * iqr
        upper_bound = q3 + self.multiplier * iqr

        if value < lower_bound or value > upper_bound:
            if self._can_alert():
                self._record_alert()

                import uuid
                return AnomalyAlert(
                    alert_id=str(uuid.uuid4()),
                    metric=self.config.metric,
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=AnomalySeverity.MEDIUM,
                    value=value,
                    expected_value=(q1 + q3) / 2,  # Median
                    deviation=abs(value - (q1 + q3) / 2) / iqr if iqr > 0 else 0,
                    timestamp=timestamp or datetime.utcnow(),
                    message=f"Value {value:.2f} outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                    context={
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                    },
                )

        return None


class CompositeDetector(AnomalyDetector):
    """
    Combines multiple detectors.

    Returns the most severe alert from any detector.
    """

    def __init__(
        self,
        config: DetectorConfig,
        detectors: List[AnomalyDetector],
    ):
        super().__init__(config)
        self.detectors = detectors

    async def analyze(self, value: float, timestamp: Optional[datetime] = None) -> Optional[AnomalyAlert]:
        """Run all detectors and return most severe alert."""
        if not self.config.enabled:
            return None

        alerts = []
        for detector in self.detectors:
            alert = await detector.analyze(value, timestamp)
            if alert:
                alerts.append(alert)

        if not alerts:
            return None

        # Return most severe
        severity_order = {
            AnomalySeverity.CRITICAL: 4,
            AnomalySeverity.HIGH: 3,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.LOW: 1,
        }

        return max(alerts, key=lambda a: severity_order[a.severity])


class AnomalyDetectionManager:
    """
    Manages anomaly detection across multiple metrics.

    Features:
    - Automatic detector configuration
    - Alert routing
    - Alert aggregation
    - Learning mode
    """

    def __init__(
        self,
        alert_callback: Optional[Callable[[AnomalyAlert], Awaitable[None]]] = None,
    ):
        self._detectors: Dict[str, List[AnomalyDetector]] = {}
        self._alert_callback = alert_callback
        self._alert_history: deque = deque(maxlen=1000)
        self._lock = asyncio.Lock()

    def register_detector(self, metric: str, detector: AnomalyDetector) -> None:
        """Register a detector for a metric."""
        if metric not in self._detectors:
            self._detectors[metric] = []
        self._detectors[metric].append(detector)

    def create_default_detectors(self, metric: str, config: DetectorConfig) -> None:
        """Create default set of detectors for a metric."""
        self.register_detector(metric, StatisticalDetector(config))
        self.register_detector(metric, TrendDetector(config))
        self.register_detector(metric, IQRDetector(config))

    async def analyze(
        self,
        metric: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> List[AnomalyAlert]:
        """Analyze a metric value."""
        detectors = self._detectors.get(metric, [])
        alerts = []

        for detector in detectors:
            alert = await detector.analyze(value, timestamp)
            if alert:
                alerts.append(alert)
                self._alert_history.append(alert)

                if self._alert_callback:
                    try:
                        await self._alert_callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")

        return alerts

    def get_alert_history(
        self,
        metric: Optional[str] = None,
        severity: Optional[AnomalySeverity] = None,
        limit: int = 100,
    ) -> List[AnomalyAlert]:
        """Get alert history with filters."""
        alerts = list(self._alert_history)

        if metric:
            alerts = [a for a in alerts if a.metric == metric]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts[-limit:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get detection metrics."""
        alerts = list(self._alert_history)

        return {
            "total_alerts": len(alerts),
            "by_severity": {
                s.value: len([a for a in alerts if a.severity == s])
                for s in AnomalySeverity
            },
            "by_type": {
                t.value: len([a for a in alerts if a.anomaly_type == t])
                for t in AnomalyType
            },
            "monitored_metrics": list(self._detectors.keys()),
        }
