"""Analytics API for Builder Engine."""

from typing import Optional, Dict, List, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

if TYPE_CHECKING:
    from bvrai_sdk.client import BvraiClient


class MetricType(str, Enum):
    """Types of metrics."""
    CALL_VOLUME = "call_volume"
    CALL_DURATION = "call_duration"
    SUCCESS_RATE = "success_rate"
    LATENCY = "latency"
    COST = "cost"
    TOKENS = "tokens"
    ERRORS = "errors"


class AggregationType(str, Enum):
    """Aggregation types for metrics."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"


class TimeGranularity(str, Enum):
    """Time granularity for time series."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class TimeSeriesPoint:
    """A single point in a time series."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSeriesPoint":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            value=data["value"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class TimeSeries:
    """Time series data."""
    metric: str
    granularity: TimeGranularity
    points: List[TimeSeriesPoint]
    aggregation: AggregationType = AggregationType.SUM

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSeries":
        return cls(
            metric=data["metric"],
            granularity=TimeGranularity(data["granularity"]),
            points=[TimeSeriesPoint.from_dict(p) for p in data.get("points", [])],
            aggregation=AggregationType(data.get("aggregation", "sum")),
        )


@dataclass
class CallMetrics:
    """Metrics for calls."""
    total_calls: int
    successful_calls: int
    failed_calls: int
    average_duration_seconds: float
    total_duration_seconds: float
    average_latency_ms: float
    total_cost: float
    total_tokens: int
    success_rate: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CallMetrics":
        return cls(
            total_calls=data.get("total_calls", 0),
            successful_calls=data.get("successful_calls", 0),
            failed_calls=data.get("failed_calls", 0),
            average_duration_seconds=data.get("average_duration_seconds", 0.0),
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
            average_latency_ms=data.get("average_latency_ms", 0.0),
            total_cost=data.get("total_cost", 0.0),
            total_tokens=data.get("total_tokens", 0),
            success_rate=data.get("success_rate", 0.0),
        )


@dataclass
class AgentMetrics:
    """Metrics for an agent."""
    agent_id: str
    agent_name: str
    total_calls: int
    average_duration_seconds: float
    success_rate: float
    average_sentiment_score: float
    average_response_time_ms: float
    top_intents: List[Dict[str, Any]]
    total_cost: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMetrics":
        return cls(
            agent_id=data["agent_id"],
            agent_name=data.get("agent_name", ""),
            total_calls=data.get("total_calls", 0),
            average_duration_seconds=data.get("average_duration_seconds", 0.0),
            success_rate=data.get("success_rate", 0.0),
            average_sentiment_score=data.get("average_sentiment_score", 0.0),
            average_response_time_ms=data.get("average_response_time_ms", 0.0),
            top_intents=data.get("top_intents", []),
            total_cost=data.get("total_cost", 0.0),
        )


@dataclass
class UsageMetrics:
    """Usage metrics for billing."""
    period_start: datetime
    period_end: datetime
    total_minutes: float
    total_calls: int
    total_tokens: int
    total_cost: float
    breakdown_by_agent: Dict[str, Dict[str, Any]]
    breakdown_by_service: Dict[str, Dict[str, Any]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageMetrics":
        return cls(
            period_start=datetime.fromisoformat(data["period_start"]),
            period_end=datetime.fromisoformat(data["period_end"]),
            total_minutes=data.get("total_minutes", 0.0),
            total_calls=data.get("total_calls", 0),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            breakdown_by_agent=data.get("breakdown_by_agent", {}),
            breakdown_by_service=data.get("breakdown_by_service", {}),
        )


@dataclass
class RealtimeMetrics:
    """Real-time metrics snapshot."""
    timestamp: datetime
    active_calls: int
    calls_per_minute: float
    average_queue_time_ms: float
    error_rate: float
    active_agents: int
    concurrent_connections: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RealtimeMetrics":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            active_calls=data.get("active_calls", 0),
            calls_per_minute=data.get("calls_per_minute", 0.0),
            average_queue_time_ms=data.get("average_queue_time_ms", 0.0),
            error_rate=data.get("error_rate", 0.0),
            active_agents=data.get("active_agents", 0),
            concurrent_connections=data.get("concurrent_connections", 0),
        )


@dataclass
class Report:
    """Analytics report."""
    id: str
    name: str
    type: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    download_url: Optional[str]
    parameters: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Report":
        return cls(
            id=data["id"],
            name=data["name"],
            type=data["type"],
            status=data["status"],
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            download_url=data.get("download_url"),
            parameters=data.get("parameters", {}),
        )


class AnalyticsAPI:
    """
    Analytics API client.

    Access call metrics, usage data, and generate reports.
    """

    def __init__(self, client: "BvraiClient"):
        self._client = client

    # Call Metrics

    async def get_call_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        agent_id: Optional[str] = None,
    ) -> CallMetrics:
        """Get aggregated call metrics."""
        params = {}
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        if agent_id:
            params["agent_id"] = agent_id

        response = await self._client.get("/v1/analytics/calls/metrics", params=params)
        return CallMetrics.from_dict(response)

    async def get_call_time_series(
        self,
        metric: MetricType,
        granularity: TimeGranularity = TimeGranularity.HOUR,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        agent_id: Optional[str] = None,
        aggregation: AggregationType = AggregationType.SUM,
    ) -> TimeSeries:
        """Get call metrics as time series."""
        params = {
            "metric": metric.value,
            "granularity": granularity.value,
            "aggregation": aggregation.value,
        }
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        if agent_id:
            params["agent_id"] = agent_id

        response = await self._client.get("/v1/analytics/calls/timeseries", params=params)
        return TimeSeries.from_dict(response)

    # Agent Metrics

    async def get_agent_metrics(
        self,
        agent_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> AgentMetrics:
        """Get metrics for a specific agent."""
        params = {}
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()

        response = await self._client.get(f"/v1/analytics/agents/{agent_id}/metrics", params=params)
        return AgentMetrics.from_dict(response)

    async def get_all_agents_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        sort_by: str = "total_calls",
        limit: int = 10,
    ) -> List[AgentMetrics]:
        """Get metrics for all agents."""
        params = {
            "sort_by": sort_by,
            "limit": limit,
        }
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()

        response = await self._client.get("/v1/analytics/agents/metrics", params=params)
        return [AgentMetrics.from_dict(a) for a in response.get("agents", [])]

    # Usage & Billing

    async def get_usage(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> UsageMetrics:
        """Get usage metrics for billing."""
        params = {}
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()

        response = await self._client.get("/v1/analytics/usage", params=params)
        return UsageMetrics.from_dict(response)

    async def get_current_billing_period(self) -> UsageMetrics:
        """Get usage for current billing period."""
        response = await self._client.get("/v1/analytics/usage/current")
        return UsageMetrics.from_dict(response)

    async def get_cost_breakdown(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        group_by: str = "service",
    ) -> Dict[str, Any]:
        """Get cost breakdown by service, agent, or day."""
        params = {"group_by": group_by}
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()

        return await self._client.get("/v1/analytics/costs/breakdown", params=params)

    # Real-time Metrics

    async def get_realtime_metrics(self) -> RealtimeMetrics:
        """Get current real-time metrics."""
        response = await self._client.get("/v1/analytics/realtime")
        return RealtimeMetrics.from_dict(response)

    async def get_active_calls_count(self) -> int:
        """Get count of currently active calls."""
        response = await self._client.get("/v1/analytics/realtime/active-calls")
        return response.get("count", 0)

    # Reports

    async def create_report(
        self,
        name: str,
        report_type: str,
        start_time: datetime,
        end_time: datetime,
        parameters: Optional[Dict[str, Any]] = None,
        format: str = "csv",
    ) -> Report:
        """Create an analytics report."""
        data = {
            "name": name,
            "type": report_type,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "parameters": parameters or {},
            "format": format,
        }
        response = await self._client.post("/v1/analytics/reports", data=data)
        return Report.from_dict(response)

    async def get_report(self, report_id: str) -> Report:
        """Get a report by ID."""
        response = await self._client.get(f"/v1/analytics/reports/{report_id}")
        return Report.from_dict(response)

    async def list_reports(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Report]:
        """List all reports."""
        params = {"limit": limit, "offset": offset}
        response = await self._client.get("/v1/analytics/reports", params=params)
        return [Report.from_dict(r) for r in response.get("reports", [])]

    async def delete_report(self, report_id: str) -> bool:
        """Delete a report."""
        await self._client.delete(f"/v1/analytics/reports/{report_id}")
        return True

    async def download_report(self, report_id: str) -> bytes:
        """Download a report file."""
        return await self._client.get_raw(f"/v1/analytics/reports/{report_id}/download")

    # Convenience Methods

    async def get_today_metrics(self) -> CallMetrics:
        """Get metrics for today."""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return await self.get_call_metrics(start_time=today)

    async def get_last_24h_metrics(self) -> CallMetrics:
        """Get metrics for last 24 hours."""
        end = datetime.now()
        start = end - timedelta(hours=24)
        return await self.get_call_metrics(start_time=start, end_time=end)

    async def get_last_7_days_metrics(self) -> CallMetrics:
        """Get metrics for last 7 days."""
        end = datetime.now()
        start = end - timedelta(days=7)
        return await self.get_call_metrics(start_time=start, end_time=end)

    async def get_last_30_days_metrics(self) -> CallMetrics:
        """Get metrics for last 30 days."""
        end = datetime.now()
        start = end - timedelta(days=30)
        return await self.get_call_metrics(start_time=start, end_time=end)

    async def get_hourly_call_volume(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> TimeSeries:
        """Get hourly call volume."""
        return await self.get_call_time_series(
            metric=MetricType.CALL_VOLUME,
            granularity=TimeGranularity.HOUR,
            start_time=start_time,
            end_time=end_time,
            aggregation=AggregationType.COUNT,
        )

    async def get_daily_cost(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> TimeSeries:
        """Get daily cost time series."""
        return await self.get_call_time_series(
            metric=MetricType.COST,
            granularity=TimeGranularity.DAY,
            start_time=start_time,
            end_time=end_time,
            aggregation=AggregationType.SUM,
        )


class AnalyticsQueryBuilder:
    """
    Builder for complex analytics queries.

    Example:
        query = (AnalyticsQueryBuilder()
            .metric(MetricType.CALL_DURATION)
            .aggregation(AggregationType.AVG)
            .granularity(TimeGranularity.HOUR)
            .filter_agent("agent_123")
            .time_range(start, end)
            .build())
    """

    def __init__(self):
        self._metric: Optional[MetricType] = None
        self._aggregation: AggregationType = AggregationType.SUM
        self._granularity: TimeGranularity = TimeGranularity.HOUR
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._filters: Dict[str, Any] = {}
        self._group_by: List[str] = []

    def metric(self, metric: MetricType) -> "AnalyticsQueryBuilder":
        """Set the metric to query."""
        self._metric = metric
        return self

    def aggregation(self, aggregation: AggregationType) -> "AnalyticsQueryBuilder":
        """Set the aggregation type."""
        self._aggregation = aggregation
        return self

    def granularity(self, granularity: TimeGranularity) -> "AnalyticsQueryBuilder":
        """Set the time granularity."""
        self._granularity = granularity
        return self

    def time_range(
        self,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> "AnalyticsQueryBuilder":
        """Set time range."""
        self._start_time = start
        self._end_time = end or datetime.now()
        return self

    def last_hours(self, hours: int) -> "AnalyticsQueryBuilder":
        """Set time range to last N hours."""
        self._end_time = datetime.now()
        self._start_time = self._end_time - timedelta(hours=hours)
        return self

    def last_days(self, days: int) -> "AnalyticsQueryBuilder":
        """Set time range to last N days."""
        self._end_time = datetime.now()
        self._start_time = self._end_time - timedelta(days=days)
        return self

    def filter_agent(self, agent_id: str) -> "AnalyticsQueryBuilder":
        """Filter by agent."""
        self._filters["agent_id"] = agent_id
        return self

    def filter_status(self, status: str) -> "AnalyticsQueryBuilder":
        """Filter by call status."""
        self._filters["status"] = status
        return self

    def filter_direction(self, direction: str) -> "AnalyticsQueryBuilder":
        """Filter by call direction (inbound/outbound)."""
        self._filters["direction"] = direction
        return self

    def group_by(self, *fields: str) -> "AnalyticsQueryBuilder":
        """Group results by fields."""
        self._group_by.extend(fields)
        return self

    def build(self) -> Dict[str, Any]:
        """Build the query parameters."""
        if not self._metric:
            raise ValueError("Metric is required")

        params = {
            "metric": self._metric.value,
            "aggregation": self._aggregation.value,
            "granularity": self._granularity.value,
        }

        if self._start_time:
            params["start_time"] = self._start_time.isoformat()
        if self._end_time:
            params["end_time"] = self._end_time.isoformat()

        params.update(self._filters)

        if self._group_by:
            params["group_by"] = ",".join(self._group_by)

        return params
