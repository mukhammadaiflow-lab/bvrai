"""
Monitoring System Base Types

Core types, interfaces, and data structures for the monitoring
and alerting system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertState(str, Enum):
    """Alert state."""
    PENDING = "pending"
    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class IncidentStatus(str, Enum):
    """Incident status."""
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"


class IncidentSeverity(str, Enum):
    """Incident severity levels (similar to PagerDuty)."""
    SEV1 = "sev1"  # Critical - immediate action required
    SEV2 = "sev2"  # High - urgent attention needed
    SEV3 = "sev3"  # Medium - needs attention soon
    SEV4 = "sev4"  # Low - minor issue
    SEV5 = "sev5"  # Info - informational


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class ComponentType(str, Enum):
    """Types of monitored components."""
    SERVICE = "service"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    API = "api"
    STORAGE = "storage"
    INTEGRATION = "integration"
    WORKER = "worker"
    CUSTOM = "custom"


class NotificationChannel(str, Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"
    DISCORD = "discord"


@dataclass
class MetricValue:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "type": self.metric_type.value,
            "unit": self.unit,
        }


@dataclass
class MetricSeries:
    """Time series of metric values."""
    name: str
    metric_type: MetricType
    values: List[MetricValue] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    description: str = ""

    @property
    def latest_value(self) -> Optional[float]:
        """Get the most recent value."""
        if self.values:
            return self.values[-1].value
        return None

    @property
    def count(self) -> int:
        """Get number of data points."""
        return len(self.values)

    def add_value(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a new value to the series."""
        self.values.append(MetricValue(
            name=self.name,
            value=value,
            timestamp=timestamp or datetime.utcnow(),
            labels=self.labels,
            metric_type=self.metric_type,
            unit=self.unit,
        ))

    def get_average(self, window_seconds: int = 300) -> Optional[float]:
        """Get average over a time window."""
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        recent = [v.value for v in self.values if v.timestamp >= cutoff]
        if recent:
            return sum(recent) / len(recent)
        return None

    def get_min(self, window_seconds: int = 300) -> Optional[float]:
        """Get minimum over a time window."""
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        recent = [v.value for v in self.values if v.timestamp >= cutoff]
        return min(recent) if recent else None

    def get_max(self, window_seconds: int = 300) -> Optional[float]:
        """Get maximum over a time window."""
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        recent = [v.value for v in self.values if v.timestamp >= cutoff]
        return max(recent) if recent else None

    def get_percentile(self, percentile: float, window_seconds: int = 300) -> Optional[float]:
        """Get percentile over a time window."""
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        recent = sorted([v.value for v in self.values if v.timestamp >= cutoff])
        if not recent:
            return None
        index = int(len(recent) * percentile / 100)
        return recent[min(index, len(recent) - 1)]


@dataclass
class HealthCheck:
    """Health check definition."""
    id: str
    name: str
    component_id: str
    component_type: ComponentType
    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3
    success_threshold: int = 1
    enabled: bool = True
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check execution."""
    check_id: str
    component_id: str
    status: HealthStatus
    latency_ms: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_id": self.check_id,
            "component_id": self.component_id,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "details": self.details,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
        }


@dataclass
class Component:
    """A monitored system component."""
    id: str
    name: str
    component_type: ComponentType
    organization_id: Optional[str] = None  # None for platform components
    description: str = ""
    endpoint: Optional[str] = None
    health_checks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.component_type.value,
            "organization_id": self.organization_id,
            "description": self.description,
            "endpoint": self.endpoint,
            "status": self.status.value,
            "last_check_at": self.last_check_at.isoformat() if self.last_check_at else None,
            "dependencies": self.dependencies,
        }


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    id: str
    name: str
    organization_id: Optional[str]  # None for platform-wide rules
    description: str = ""
    enabled: bool = True

    # Condition
    metric_name: str = ""
    condition: str = ""  # e.g., "value > 90", "avg(5m) > 80"
    threshold: float = 0.0
    comparison: str = ">"  # >, <, >=, <=, ==, !=

    # Timing
    evaluation_interval_seconds: int = 60
    for_duration_seconds: int = 300  # How long condition must be true

    # Alert properties
    severity: AlertSeverity = AlertSeverity.MEDIUM
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # Notification
    notification_channels: List[str] = field(default_factory=list)
    notify_on_resolve: bool = True
    cooldown_seconds: int = 300  # Don't re-alert within this window

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "organization_id": self.organization_id,
            "description": self.description,
            "enabled": self.enabled,
            "metric_name": self.metric_name,
            "condition": f"{self.metric_name} {self.comparison} {self.threshold}",
            "severity": self.severity.value,
            "evaluation_interval_seconds": self.evaluation_interval_seconds,
            "for_duration_seconds": self.for_duration_seconds,
            "notification_channels": self.notification_channels,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Alert:
    """An active or historical alert."""
    id: str
    rule_id: str
    rule_name: str
    organization_id: Optional[str]
    severity: AlertSeverity
    state: AlertState
    title: str
    description: str = ""

    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    last_evaluation_at: Optional[datetime] = None

    # Value tracking
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None

    # Context
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    fingerprint: str = ""  # Unique identifier for deduplication

    # Response
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    incident_id: Optional[str] = None

    # Notification tracking
    notifications_sent: int = 0
    last_notification_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "organization_id": self.organization_id,
            "severity": self.severity.value,
            "state": self.state.value,
            "title": self.title,
            "description": self.description,
            "started_at": self.started_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "labels": self.labels,
            "incident_id": self.incident_id,
        }


@dataclass
class IncidentUpdate:
    """An update to an incident."""
    id: str
    incident_id: str
    status: IncidentStatus
    message: str
    author: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "status": self.status.value,
            "message": self.message,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Incident:
    """An incident record."""
    id: str
    organization_id: Optional[str]
    title: str
    severity: IncidentSeverity
    status: IncidentStatus

    # Description
    description: str = ""
    impact: str = ""
    root_cause: str = ""
    resolution: str = ""

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    detected_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    # Responders
    commander: Optional[str] = None
    responders: List[str] = field(default_factory=list)

    # Related items
    alert_ids: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    updates: List[IncidentUpdate] = field(default_factory=list)

    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    external_url: Optional[str] = None
    postmortem_url: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[int]:
        """Get incident duration in seconds."""
        if self.started_at and self.resolved_at:
            return int((self.resolved_at - self.started_at).total_seconds())
        elif self.started_at:
            return int((datetime.utcnow() - self.started_at).total_seconds())
        return None

    @property
    def time_to_detection_seconds(self) -> Optional[int]:
        """Get time from start to detection."""
        if self.started_at and self.detected_at:
            return int((self.detected_at - self.started_at).total_seconds())
        return None

    @property
    def time_to_resolution_seconds(self) -> Optional[int]:
        """Get time from detection to resolution."""
        if self.detected_at and self.resolved_at:
            return int((self.resolved_at - self.detected_at).total_seconds())
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "title": self.title,
            "severity": self.severity.value,
            "status": self.status.value,
            "description": self.description,
            "impact": self.impact,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "duration_seconds": self.duration_seconds,
            "commander": self.commander,
            "responders": self.responders,
            "affected_components": self.affected_components,
            "alert_ids": self.alert_ids,
            "updates_count": len(self.updates),
        }


@dataclass
class NotificationTarget:
    """A notification target configuration."""
    id: str
    organization_id: Optional[str]
    name: str
    channel: NotificationChannel
    enabled: bool = True

    # Channel-specific config
    config: Dict[str, Any] = field(default_factory=dict)
    # email: {"addresses": ["email@example.com"]}
    # slack: {"webhook_url": "...", "channel": "#alerts"}
    # webhook: {"url": "...", "headers": {...}}

    # Filtering
    severity_filter: List[AlertSeverity] = field(default_factory=list)  # Empty = all
    label_filters: Dict[str, str] = field(default_factory=dict)

    # Rate limiting
    rate_limit_per_hour: int = 100

    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive config)."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "channel": self.channel.value,
            "enabled": self.enabled,
            "severity_filter": [s.value for s in self.severity_filter],
            "rate_limit_per_hour": self.rate_limit_per_hour,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MaintenanceWindow:
    """Scheduled maintenance window for suppressing alerts."""
    id: str
    organization_id: Optional[str]
    name: str
    description: str = ""

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime = field(default_factory=datetime.utcnow)
    timezone: str = "UTC"

    # Scope
    affected_components: List[str] = field(default_factory=list)  # Empty = all
    affected_services: List[str] = field(default_factory=list)
    suppress_alerts: bool = True
    suppress_notifications: bool = True

    # Recurrence
    recurring: bool = False
    recurrence_rule: Optional[str] = None  # iCal RRULE format

    # Metadata
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_active(self) -> bool:
        """Check if maintenance window is currently active."""
        now = datetime.utcnow()
        return self.start_time <= now <= self.end_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "is_active": self.is_active,
            "affected_components": self.affected_components,
            "suppress_alerts": self.suppress_alerts,
            "recurring": self.recurring,
            "created_by": self.created_by,
        }


@dataclass
class StatusPageComponent:
    """Component displayed on public status page."""
    id: str
    name: str
    description: str = ""
    status: HealthStatus = HealthStatus.HEALTHY
    group: Optional[str] = None
    order: int = 0
    visible: bool = True
    last_updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StatusPageIncident:
    """Incident displayed on public status page."""
    id: str
    title: str
    status: IncidentStatus
    severity: IncidentSeverity
    message: str
    affected_components: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None


# =========================================================================
# Abstract Interfaces
# =========================================================================

class HealthChecker(ABC):
    """Abstract interface for health checkers."""

    @abstractmethod
    async def check(self, component: Component) -> HealthCheckResult:
        """
        Perform a health check on a component.

        Args:
            component: Component to check

        Returns:
            Health check result
        """
        pass


class MetricCollector(ABC):
    """Abstract interface for metric collectors."""

    @abstractmethod
    async def collect(self) -> List[MetricValue]:
        """
        Collect metrics.

        Returns:
            List of collected metric values
        """
        pass


class NotificationSender(ABC):
    """Abstract interface for notification senders."""

    @abstractmethod
    async def send(
        self,
        target: NotificationTarget,
        alert: Alert,
        is_resolved: bool = False,
    ) -> bool:
        """
        Send a notification.

        Args:
            target: Notification target
            alert: Alert to notify about
            is_resolved: Whether this is a resolution notification

        Returns:
            True if notification was sent successfully
        """
        pass


class MetricStore(ABC):
    """Abstract interface for metric storage."""

    @abstractmethod
    async def write(self, metrics: List[MetricValue]) -> None:
        """Write metrics to storage."""
        pass

    @abstractmethod
    async def query(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        labels: Optional[Dict[str, str]] = None,
    ) -> List[MetricValue]:
        """Query metrics from storage."""
        pass


class AlertStore(ABC):
    """Abstract interface for alert storage."""

    @abstractmethod
    async def save_alert(self, alert: Alert) -> None:
        """Save an alert."""
        pass

    @abstractmethod
    async def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID."""
        pass

    @abstractmethod
    async def list_alerts(
        self,
        organization_id: Optional[str] = None,
        state: Optional[AlertState] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """List alerts with filters."""
        pass


# =========================================================================
# Errors
# =========================================================================

class MonitoringError(Exception):
    """Base monitoring error."""

    def __init__(self, message: str, code: str = "monitoring_error"):
        self.message = message
        self.code = code
        super().__init__(message)


class HealthCheckError(MonitoringError):
    """Health check failed."""

    def __init__(self, message: str, component_id: str):
        self.component_id = component_id
        super().__init__(message, "health_check_error")


class AlertError(MonitoringError):
    """Alert processing error."""

    def __init__(self, message: str, alert_id: Optional[str] = None):
        self.alert_id = alert_id
        super().__init__(message, "alert_error")


class NotificationError(MonitoringError):
    """Notification delivery error."""

    def __init__(self, message: str, channel: str):
        self.channel = channel
        super().__init__(message, "notification_error")
