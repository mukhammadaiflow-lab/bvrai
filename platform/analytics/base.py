"""
Analytics Base Types Module

This module defines core types and data structures for the analytics system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


class MetricType(str, Enum):
    """Types of metrics."""

    # Counter - monotonically increasing
    COUNTER = "counter"

    # Gauge - point-in-time value
    GAUGE = "gauge"

    # Histogram - distribution of values
    HISTOGRAM = "histogram"

    # Timer - duration measurements
    TIMER = "timer"

    # Rate - events per time unit
    RATE = "rate"


class TimeGranularity(str, Enum):
    """Time granularity for aggregations."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class ReportType(str, Enum):
    """Types of reports."""

    USAGE_SUMMARY = "usage_summary"
    CALL_PERFORMANCE = "call_performance"
    AGENT_PERFORMANCE = "agent_performance"
    COST_ANALYSIS = "cost_analysis"
    QUALITY_METRICS = "quality_metrics"
    CONVERSION_REPORT = "conversion_report"
    CAMPAIGN_REPORT = "campaign_report"


class CallOutcome(str, Enum):
    """Call outcomes for analytics."""

    COMPLETED = "completed"
    VOICEMAIL = "voicemail"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    FAILED = "failed"
    TRANSFERRED = "transferred"
    CANCELED = "canceled"


@dataclass
class MetricPoint:
    """A single metric data point."""

    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "tags": self.tags,
        }


@dataclass
class MetricDefinition:
    """Definition of a metric."""

    name: str
    type: MetricType
    description: str
    unit: str = ""
    tags: List[str] = field(default_factory=list)

    # For histograms
    buckets: Optional[List[float]] = None


@dataclass
class AggregatedMetric:
    """Aggregated metric data."""

    metric_name: str
    granularity: TimeGranularity
    start_time: datetime
    end_time: datetime

    # Aggregation values
    count: int = 0
    sum: float = 0.0
    min: float = 0.0
    max: float = 0.0
    avg: float = 0.0

    # Percentiles (for histograms/timers)
    p50: Optional[float] = None
    p90: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None

    # Tags used in aggregation
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "metric": self.metric_name,
            "granularity": self.granularity.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "count": self.count,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "avg": self.avg,
            "tags": self.tags,
        }

        if self.p50 is not None:
            result["percentiles"] = {
                "p50": self.p50,
                "p90": self.p90,
                "p95": self.p95,
                "p99": self.p99,
            }

        return result


@dataclass
class CallMetricsSnapshot:
    """Snapshot of call metrics."""

    call_id: str
    organization_id: str
    agent_id: str
    timestamp: datetime

    # Call info
    direction: str  # inbound/outbound
    outcome: CallOutcome
    duration_seconds: float
    wait_time_seconds: float = 0.0

    # Performance
    time_to_answer_ms: float = 0.0
    time_to_first_response_ms: float = 0.0
    avg_response_time_ms: float = 0.0

    # Conversation
    turn_count: int = 0
    interruption_count: int = 0
    silence_seconds: float = 0.0

    # Quality
    transcription_confidence: float = 0.0
    sentiment_score: float = 0.0

    # AI usage
    llm_tokens_used: int = 0
    function_calls: int = 0

    # Cost
    cost_cents: int = 0

    # Tags
    campaign_id: Optional[str] = None
    industry: Optional[str] = None
    custom_tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class UsageSummary:
    """Usage summary for a time period."""

    organization_id: str
    start_date: datetime
    end_date: datetime

    # Call counts
    total_calls: int = 0
    inbound_calls: int = 0
    outbound_calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0

    # Duration
    total_minutes: float = 0.0
    avg_call_duration_seconds: float = 0.0

    # Success rates
    answer_rate: float = 0.0
    completion_rate: float = 0.0
    transfer_rate: float = 0.0

    # Cost
    total_cost_cents: int = 0
    telephony_cost_cents: int = 0
    ai_cost_cents: int = 0

    # Resources
    unique_agents: int = 0
    unique_phone_numbers: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "organization_id": self.organization_id,
            "period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
            },
            "calls": {
                "total": self.total_calls,
                "inbound": self.inbound_calls,
                "outbound": self.outbound_calls,
                "completed": self.completed_calls,
                "failed": self.failed_calls,
            },
            "duration": {
                "total_minutes": self.total_minutes,
                "avg_seconds": self.avg_call_duration_seconds,
            },
            "rates": {
                "answer_rate": self.answer_rate,
                "completion_rate": self.completion_rate,
                "transfer_rate": self.transfer_rate,
            },
            "cost": {
                "total_cents": self.total_cost_cents,
                "telephony_cents": self.telephony_cost_cents,
                "ai_cents": self.ai_cost_cents,
            },
            "resources": {
                "agents": self.unique_agents,
                "phone_numbers": self.unique_phone_numbers,
            },
        }


@dataclass
class AgentAnalytics:
    """Analytics for a specific agent."""

    agent_id: str
    agent_name: str
    organization_id: str
    start_date: datetime
    end_date: datetime

    # Call metrics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    transferred_calls: int = 0

    # Duration
    total_talk_time_minutes: float = 0.0
    avg_call_duration_seconds: float = 0.0

    # Performance
    avg_response_time_ms: float = 0.0
    avg_time_to_first_response_ms: float = 0.0

    # Success rates
    completion_rate: float = 0.0
    transfer_rate: float = 0.0
    customer_satisfaction: Optional[float] = None

    # Quality
    avg_sentiment_score: float = 0.0
    positive_sentiment_rate: float = 0.0
    negative_sentiment_rate: float = 0.0

    # AI usage
    total_tokens_used: int = 0
    avg_tokens_per_call: float = 0.0
    total_function_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
            },
            "calls": {
                "total": self.total_calls,
                "successful": self.successful_calls,
                "failed": self.failed_calls,
                "transferred": self.transferred_calls,
            },
            "duration": {
                "total_minutes": self.total_talk_time_minutes,
                "avg_seconds": self.avg_call_duration_seconds,
            },
            "performance": {
                "avg_response_time_ms": self.avg_response_time_ms,
                "avg_first_response_ms": self.avg_time_to_first_response_ms,
            },
            "rates": {
                "completion": self.completion_rate,
                "transfer": self.transfer_rate,
                "satisfaction": self.customer_satisfaction,
            },
            "sentiment": {
                "avg_score": self.avg_sentiment_score,
                "positive_rate": self.positive_sentiment_rate,
                "negative_rate": self.negative_sentiment_rate,
            },
            "ai_usage": {
                "total_tokens": self.total_tokens_used,
                "avg_tokens_per_call": self.avg_tokens_per_call,
                "function_calls": self.total_function_calls,
            },
        }


@dataclass
class TimeSeriesDataPoint:
    """A point in a time series."""

    timestamp: datetime
    value: float
    label: Optional[str] = None


@dataclass
class TimeSeriesData:
    """Time series data for charting."""

    metric_name: str
    granularity: TimeGranularity
    start_time: datetime
    end_time: datetime
    data_points: List[TimeSeriesDataPoint] = field(default_factory=list)

    # Statistics
    total: float = 0.0
    average: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric_name,
            "granularity": self.granularity.value,
            "period": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
            },
            "data": [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "value": point.value,
                    "label": point.label,
                }
                for point in self.data_points
            ],
            "statistics": {
                "total": self.total,
                "average": self.average,
                "min": self.min_value,
                "max": self.max_value,
            },
        }


@dataclass
class OutcomeDistribution:
    """Distribution of call outcomes."""

    organization_id: str
    start_date: datetime
    end_date: datetime

    completed: int = 0
    voicemail: int = 0
    no_answer: int = 0
    busy: int = 0
    failed: int = 0
    transferred: int = 0
    canceled: int = 0

    @property
    def total(self) -> int:
        """Get total calls."""
        return (
            self.completed + self.voicemail + self.no_answer +
            self.busy + self.failed + self.transferred + self.canceled
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        total = self.total
        return {
            "period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
            },
            "outcomes": {
                "completed": {"count": self.completed, "rate": self.completed / total if total else 0},
                "voicemail": {"count": self.voicemail, "rate": self.voicemail / total if total else 0},
                "no_answer": {"count": self.no_answer, "rate": self.no_answer / total if total else 0},
                "busy": {"count": self.busy, "rate": self.busy / total if total else 0},
                "failed": {"count": self.failed, "rate": self.failed / total if total else 0},
                "transferred": {"count": self.transferred, "rate": self.transferred / total if total else 0},
                "canceled": {"count": self.canceled, "rate": self.canceled / total if total else 0},
            },
            "total": total,
        }


@dataclass
class CostBreakdown:
    """Cost breakdown by category."""

    organization_id: str
    start_date: datetime
    end_date: datetime

    # Cost categories (in cents)
    telephony_inbound_cents: int = 0
    telephony_outbound_cents: int = 0
    transcription_cents: int = 0
    llm_input_cents: int = 0
    llm_output_cents: int = 0
    tts_cents: int = 0
    storage_cents: int = 0

    @property
    def total_cents(self) -> int:
        """Get total cost in cents."""
        return (
            self.telephony_inbound_cents + self.telephony_outbound_cents +
            self.transcription_cents + self.llm_input_cents +
            self.llm_output_cents + self.tts_cents + self.storage_cents
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        total = self.total_cents
        return {
            "period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
            },
            "breakdown": {
                "telephony_inbound": self.telephony_inbound_cents,
                "telephony_outbound": self.telephony_outbound_cents,
                "transcription": self.transcription_cents,
                "llm_input": self.llm_input_cents,
                "llm_output": self.llm_output_cents,
                "tts": self.tts_cents,
                "storage": self.storage_cents,
            },
            "total_cents": total,
            "total_dollars": total / 100,
        }


@dataclass
class QueryParams:
    """Parameters for analytics queries."""

    organization_id: str
    start_date: datetime
    end_date: datetime

    # Filters
    agent_ids: Optional[List[str]] = None
    campaign_ids: Optional[List[str]] = None
    phone_numbers: Optional[List[str]] = None
    directions: Optional[List[str]] = None
    outcomes: Optional[List[CallOutcome]] = None
    industries: Optional[List[str]] = None

    # Grouping
    group_by: Optional[List[str]] = None

    # Pagination
    limit: int = 100
    offset: int = 0


__all__ = [
    "MetricType",
    "TimeGranularity",
    "ReportType",
    "CallOutcome",
    "MetricPoint",
    "MetricDefinition",
    "AggregatedMetric",
    "CallMetricsSnapshot",
    "UsageSummary",
    "AgentAnalytics",
    "TimeSeriesDataPoint",
    "TimeSeriesData",
    "OutcomeDistribution",
    "CostBreakdown",
    "QueryParams",
]
