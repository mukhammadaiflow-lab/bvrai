"""
Data Models for Analytics Engine Service.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

from .config import (
    EventType,
    MetricType,
    AggregationPeriod,
    CallOutcome,
    SentimentCategory,
    LatencyComponent,
    AlertSeverity,
)


# =============================================================================
# Event Models
# =============================================================================


@dataclass
class AnalyticsEvent:
    """Base analytics event."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.CALL_STARTED
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Context
    organization_id: Optional[str] = None
    agent_id: Optional[str] = None
    call_id: Optional[str] = None
    session_id: Optional[str] = None

    # Data
    data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    source: str = "unknown"
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "organization_id": self.organization_id,
            "agent_id": self.agent_id,
            "call_id": self.call_id,
            "session_id": self.session_id,
            "data": self.data,
            "source": self.source,
            "version": self.version,
        }


@dataclass
class CallEvent(AnalyticsEvent):
    """Call-specific event."""

    caller_number: Optional[str] = None
    called_number: Optional[str] = None
    direction: str = "inbound"
    outcome: Optional[CallOutcome] = None
    duration_ms: Optional[int] = None


@dataclass
class ConversationEvent(AnalyticsEvent):
    """Conversation-specific event."""

    transcript: Optional[str] = None
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    sentiment: Optional[SentimentCategory] = None
    sentiment_score: Optional[float] = None


@dataclass
class LatencyEvent(AnalyticsEvent):
    """Latency measurement event."""

    component: LatencyComponent = LatencyComponent.TOTAL
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# Metric Models
# =============================================================================


@dataclass
class MetricValue:
    """A single metric value."""

    timestamp: datetime
    value: float
    count: int = 1
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric values."""

    metric_type: MetricType
    period: AggregationPeriod
    start_time: datetime
    end_time: datetime
    values: List[MetricValue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_type": self.metric_type.value,
            "period": self.period.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "values": [
                {
                    "timestamp": v.timestamp.isoformat(),
                    "value": v.value,
                    "count": v.count,
                    "tags": v.tags,
                }
                for v in self.values
            ],
        }


@dataclass
class AggregatedMetric:
    """Aggregated metric with statistics."""

    metric_type: MetricType
    period: AggregationPeriod
    period_start: datetime
    period_end: datetime

    # Aggregations
    count: int = 0
    sum: float = 0.0
    avg: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std_dev: float = 0.0

    # Percentiles
    p50: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    # Dimensions
    dimensions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_type": self.metric_type.value,
            "period": self.period.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "count": self.count,
            "sum": self.sum,
            "avg": round(self.avg, 3),
            "min": self.min,
            "max": self.max,
            "std_dev": round(self.std_dev, 3),
            "percentiles": {
                "p50": self.p50,
                "p75": self.p75,
                "p90": self.p90,
                "p95": self.p95,
                "p99": self.p99,
            },
            "dimensions": self.dimensions,
        }


# =============================================================================
# Dashboard Models
# =============================================================================


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""

    widget_id: str
    widget_type: str  # chart, metric, table, map
    title: str
    metrics: List[MetricType]
    period: AggregationPeriod = AggregationPeriod.HOUR
    filters: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 4, "h": 3})


@dataclass
class Dashboard:
    """Dashboard configuration."""

    dashboard_id: str
    name: str
    description: str
    widgets: List[DashboardWidget]
    owner_id: str
    is_public: bool = False
    refresh_interval_s: int = 60
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Alert Models
# =============================================================================


@dataclass
class AlertRule:
    """Alert rule definition."""

    rule_id: str
    name: str
    description: str
    metric_type: MetricType
    condition: str  # gt, lt, gte, lte, eq, ne
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 15
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """Triggered alert."""

    alert_id: str
    rule_id: str
    severity: AlertSeverity
    message: str
    metric_type: MetricType
    current_value: float
    threshold: float
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None


# =============================================================================
# Report Models
# =============================================================================


@dataclass
class ReportConfig:
    """Report configuration."""

    report_id: str
    name: str
    description: str
    metrics: List[MetricType]
    period: AggregationPeriod
    dimensions: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[str] = None  # Cron expression
    format: str = "json"  # json, csv, pdf


@dataclass
class Report:
    """Generated report."""

    report_id: str
    config_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    data: Dict[str, Any]
    summary: Dict[str, Any]


# =============================================================================
# API Models
# =============================================================================


class IngestEventRequest(BaseModel):
    """Request to ingest an analytics event."""

    event_type: str = Field(..., description="Event type")
    timestamp: Optional[datetime] = Field(None, description="Event timestamp")
    organization_id: Optional[str] = None
    agent_id: Optional[str] = None
    call_id: Optional[str] = None
    session_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class IngestBatchRequest(BaseModel):
    """Request to ingest multiple events."""

    events: List[IngestEventRequest]


class QueryMetricsRequest(BaseModel):
    """Request to query metrics."""

    metric_type: str
    start_time: datetime
    end_time: datetime
    period: str = "hour"
    filters: Dict[str, Any] = Field(default_factory=dict)
    dimensions: List[str] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    """Response with metrics data."""

    metric_type: str
    period: str
    start_time: datetime
    end_time: datetime
    data: List[Dict[str, Any]]
    summary: Dict[str, Any]


class DashboardResponse(BaseModel):
    """Response with dashboard data."""

    dashboard_id: str
    name: str
    widgets: List[Dict[str, Any]]
    last_updated: datetime


class AlertsResponse(BaseModel):
    """Response with alerts."""

    alerts: List[Dict[str, Any]]
    total: int
    unacknowledged: int


class ReportResponse(BaseModel):
    """Response with report data."""

    report_id: str
    name: str
    generated_at: datetime
    data: Dict[str, Any]
    download_url: Optional[str] = None


# =============================================================================
# Export
# =============================================================================


__all__ = [
    # Events
    "AnalyticsEvent",
    "CallEvent",
    "ConversationEvent",
    "LatencyEvent",
    # Metrics
    "MetricValue",
    "MetricSeries",
    "AggregatedMetric",
    # Dashboard
    "DashboardWidget",
    "Dashboard",
    # Alerts
    "AlertRule",
    "Alert",
    # Reports
    "ReportConfig",
    "Report",
    # API
    "IngestEventRequest",
    "IngestBatchRequest",
    "QueryMetricsRequest",
    "MetricsResponse",
    "DashboardResponse",
    "AlertsResponse",
    "ReportResponse",
]
