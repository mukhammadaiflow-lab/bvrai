"""Analytics schemas."""

from datetime import datetime, date
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    """Time range for analytics queries."""

    start: datetime
    end: datetime


class DailyStats(BaseModel):
    """Daily statistics."""

    date: date
    total_calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0
    total_duration_seconds: int = 0
    avg_duration_seconds: float = 0
    unique_callers: int = 0


class HourlyDistribution(BaseModel):
    """Hourly call distribution."""

    hour: int
    call_count: int


class AgentPerformance(BaseModel):
    """Agent performance metrics."""

    agent_id: UUID
    agent_name: str
    total_calls: int
    completed_calls: int
    failed_calls: int
    avg_duration_seconds: float
    completion_rate: float
    avg_response_time_ms: Optional[float] = None


class CallOutcome(BaseModel):
    """Call outcome breakdown."""

    outcome: str
    count: int
    percentage: float


class SentimentBreakdown(BaseModel):
    """Sentiment analysis breakdown."""

    positive: int
    neutral: int
    negative: int
    positive_percentage: float
    neutral_percentage: float
    negative_percentage: float


class TopicFrequency(BaseModel):
    """Topic frequency in calls."""

    topic: str
    count: int
    percentage: float


class DashboardOverview(BaseModel):
    """Main dashboard overview."""

    # Call metrics
    total_calls: int = 0
    calls_today: int = 0
    calls_this_week: int = 0
    calls_this_month: int = 0

    # Duration metrics
    total_minutes: int = 0
    avg_call_duration_seconds: float = 0

    # Success metrics
    completion_rate: float = 0
    avg_response_time_ms: float = 0

    # Trends
    calls_trend_percentage: float = 0  # vs previous period
    duration_trend_percentage: float = 0

    # Agent metrics
    active_agents: int = 0
    total_agents: int = 0


class AnalyticsReport(BaseModel):
    """Full analytics report."""

    time_range: TimeRange
    overview: DashboardOverview
    daily_stats: list[DailyStats] = Field(default_factory=list)
    hourly_distribution: list[HourlyDistribution] = Field(default_factory=list)
    agent_performance: list[AgentPerformance] = Field(default_factory=list)
    call_outcomes: list[CallOutcome] = Field(default_factory=list)
    top_topics: list[TopicFrequency] = Field(default_factory=list)


class UsageMetrics(BaseModel):
    """Platform usage metrics."""

    period_start: datetime
    period_end: datetime

    # Call usage
    total_calls: int
    total_minutes: int
    inbound_calls: int
    outbound_calls: int

    # API usage
    api_calls: int
    llm_tokens_used: int
    tts_characters: int
    asr_minutes: int

    # Storage usage
    recordings_size_mb: float
    knowledge_base_size_mb: float


class CostBreakdown(BaseModel):
    """Cost breakdown for billing."""

    period_start: datetime
    period_end: datetime

    # Telephony costs
    telephony_minutes: int
    telephony_cost: float

    # AI costs
    llm_tokens: int
    llm_cost: float
    tts_characters: int
    tts_cost: float
    asr_minutes: int
    asr_cost: float

    # Storage costs
    storage_gb: float
    storage_cost: float

    # Total
    total_cost: float
    currency: str = "USD"


class RealTimeMetrics(BaseModel):
    """Real-time metrics for live dashboard."""

    timestamp: datetime
    active_calls: int
    calls_per_minute: float
    avg_wait_time_seconds: float
    agents_online: int
    queue_length: int


class AlertConfig(BaseModel):
    """Alert configuration."""

    id: UUID
    name: str
    metric: str
    condition: str  # "greater_than", "less_than", "equals"
    threshold: float
    window_minutes: int
    webhook_url: Optional[str] = None
    email: Optional[str] = None
    enabled: bool = True


class AlertEvent(BaseModel):
    """Alert event."""

    id: UUID
    alert_config_id: UUID
    triggered_at: datetime
    metric_value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None
