"""
Data Aggregation Module

This module provides data aggregation capabilities for analytics,
including time-based aggregations and statistical calculations.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .base import (
    MetricType,
    TimeGranularity,
    MetricPoint,
    AggregatedMetric,
    CallMetricsSnapshot,
    CallOutcome,
    UsageSummary,
    AgentAnalytics,
    TimeSeriesData,
    TimeSeriesDataPoint,
    OutcomeDistribution,
    CostBreakdown,
    QueryParams,
)
from .collectors import MetricsBackend


logger = logging.getLogger(__name__)


def get_time_bucket(
    timestamp: datetime,
    granularity: TimeGranularity,
) -> datetime:
    """Get the start of the time bucket for a timestamp."""
    if granularity == TimeGranularity.MINUTE:
        return timestamp.replace(second=0, microsecond=0)
    elif granularity == TimeGranularity.HOUR:
        return timestamp.replace(minute=0, second=0, microsecond=0)
    elif granularity == TimeGranularity.DAY:
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    elif granularity == TimeGranularity.WEEK:
        start = timestamp - timedelta(days=timestamp.weekday())
        return start.replace(hour=0, minute=0, second=0, microsecond=0)
    elif granularity == TimeGranularity.MONTH:
        return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        return timestamp


def get_next_bucket(
    timestamp: datetime,
    granularity: TimeGranularity,
) -> datetime:
    """Get the start of the next time bucket."""
    if granularity == TimeGranularity.MINUTE:
        return timestamp + timedelta(minutes=1)
    elif granularity == TimeGranularity.HOUR:
        return timestamp + timedelta(hours=1)
    elif granularity == TimeGranularity.DAY:
        return timestamp + timedelta(days=1)
    elif granularity == TimeGranularity.WEEK:
        return timestamp + timedelta(weeks=1)
    elif granularity == TimeGranularity.MONTH:
        if timestamp.month == 12:
            return timestamp.replace(year=timestamp.year + 1, month=1)
        else:
            return timestamp.replace(month=timestamp.month + 1)
    else:
        return timestamp


class DataStore(ABC):
    """Abstract base class for analytics data storage."""

    @abstractmethod
    async def store_call_metrics(self, snapshot: CallMetricsSnapshot) -> None:
        """Store call metrics snapshot."""
        pass

    @abstractmethod
    async def query_calls(self, params: QueryParams) -> List[CallMetricsSnapshot]:
        """Query call metrics."""
        pass

    @abstractmethod
    async def aggregate_calls(
        self,
        params: QueryParams,
        granularity: TimeGranularity,
    ) -> List[Dict[str, Any]]:
        """Aggregate call metrics."""
        pass


class InMemoryDataStore(DataStore):
    """In-memory data store for development and testing."""

    def __init__(self, max_records: int = 100000):
        """
        Initialize store.

        Args:
            max_records: Maximum records to keep
        """
        self.max_records = max_records
        self._calls: List[CallMetricsSnapshot] = []
        self._lock = asyncio.Lock()

    async def store_call_metrics(self, snapshot: CallMetricsSnapshot) -> None:
        """Store call metrics."""
        async with self._lock:
            self._calls.append(snapshot)

            # Trim if needed
            if len(self._calls) > self.max_records:
                self._calls = self._calls[-self.max_records:]

    async def query_calls(self, params: QueryParams) -> List[CallMetricsSnapshot]:
        """Query call metrics."""
        results = []

        for call in self._calls:
            # Filter by organization
            if call.organization_id != params.organization_id:
                continue

            # Filter by time
            if call.timestamp < params.start_date or call.timestamp > params.end_date:
                continue

            # Filter by agents
            if params.agent_ids and call.agent_id not in params.agent_ids:
                continue

            # Filter by campaigns
            if params.campaign_ids and call.campaign_id not in params.campaign_ids:
                continue

            # Filter by outcomes
            if params.outcomes and call.outcome not in params.outcomes:
                continue

            # Filter by directions
            if params.directions and call.direction not in params.directions:
                continue

            results.append(call)

            # Apply limit
            if len(results) >= params.limit:
                break

        return results[params.offset:params.offset + params.limit]

    async def aggregate_calls(
        self,
        params: QueryParams,
        granularity: TimeGranularity,
    ) -> List[Dict[str, Any]]:
        """Aggregate call metrics by time bucket."""
        calls = await self.query_calls(params)

        # Group by time bucket
        buckets: Dict[datetime, List[CallMetricsSnapshot]] = defaultdict(list)
        for call in calls:
            bucket = get_time_bucket(call.timestamp, granularity)
            buckets[bucket].append(call)

        # Aggregate each bucket
        results = []
        for bucket_time in sorted(buckets.keys()):
            bucket_calls = buckets[bucket_time]

            agg = {
                "timestamp": bucket_time.isoformat(),
                "count": len(bucket_calls),
                "duration_sum": sum(c.duration_seconds for c in bucket_calls),
                "duration_avg": sum(c.duration_seconds for c in bucket_calls) / len(bucket_calls),
                "cost_sum": sum(c.cost_cents for c in bucket_calls),
                "outcomes": {},
            }

            # Count outcomes
            for outcome in CallOutcome:
                count = sum(1 for c in bucket_calls if c.outcome == outcome)
                if count > 0:
                    agg["outcomes"][outcome.value] = count

            results.append(agg)

        return results


class MetricsAggregator:
    """
    Aggregates raw metric points into summarized data.

    Supports:
    - Time-based aggregation
    - Statistical calculations
    - Grouping by tags
    """

    def __init__(self, backend: MetricsBackend):
        """
        Initialize aggregator.

        Args:
            backend: Metrics storage backend
        """
        self.backend = backend

    async def aggregate(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        granularity: TimeGranularity,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[AggregatedMetric]:
        """
        Aggregate a metric over time.

        Args:
            metric_name: Metric name to aggregate
            start_time: Start of time range
            end_time: End of time range
            granularity: Time granularity
            tags: Filter tags

        Returns:
            List of aggregated metrics
        """
        # Query raw points
        points = await self.backend.query(
            metric_name,
            start_time,
            end_time,
            tags,
        )

        # Group by time bucket
        buckets: Dict[datetime, List[float]] = defaultdict(list)
        for point in points:
            bucket = get_time_bucket(point.timestamp, granularity)
            buckets[bucket].append(point.value)

        # Aggregate each bucket
        results = []
        current = get_time_bucket(start_time, granularity)

        while current < end_time:
            next_bucket = get_next_bucket(current, granularity)
            values = buckets.get(current, [])

            if values:
                sorted_values = sorted(values)
                n = len(values)

                agg = AggregatedMetric(
                    metric_name=metric_name,
                    granularity=granularity,
                    start_time=current,
                    end_time=next_bucket,
                    count=n,
                    sum=sum(values),
                    min=sorted_values[0],
                    max=sorted_values[-1],
                    avg=sum(values) / n,
                    p50=sorted_values[n // 2],
                    p90=sorted_values[int(n * 0.9)],
                    p95=sorted_values[int(n * 0.95)],
                    p99=sorted_values[int(n * 0.99)],
                    tags=tags or {},
                )
            else:
                agg = AggregatedMetric(
                    metric_name=metric_name,
                    granularity=granularity,
                    start_time=current,
                    end_time=next_bucket,
                    tags=tags or {},
                )

            results.append(agg)
            current = next_bucket

        return results

    async def get_time_series(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        granularity: TimeGranularity,
        aggregation: str = "sum",
        tags: Optional[Dict[str, str]] = None,
    ) -> TimeSeriesData:
        """
        Get time series data for a metric.

        Args:
            metric_name: Metric name
            start_time: Start time
            end_time: End time
            granularity: Time granularity
            aggregation: Aggregation function (sum, avg, min, max, count)
            tags: Filter tags

        Returns:
            Time series data
        """
        aggregated = await self.aggregate(
            metric_name,
            start_time,
            end_time,
            granularity,
            tags,
        )

        data_points = []
        values = []

        for agg in aggregated:
            if aggregation == "sum":
                value = agg.sum
            elif aggregation == "avg":
                value = agg.avg
            elif aggregation == "min":
                value = agg.min
            elif aggregation == "max":
                value = agg.max
            elif aggregation == "count":
                value = agg.count
            else:
                value = agg.sum

            data_points.append(TimeSeriesDataPoint(
                timestamp=agg.start_time,
                value=value,
            ))
            values.append(value)

        return TimeSeriesData(
            metric_name=metric_name,
            granularity=granularity,
            start_time=start_time,
            end_time=end_time,
            data_points=data_points,
            total=sum(values),
            average=sum(values) / len(values) if values else 0,
            min_value=min(values) if values else 0,
            max_value=max(values) if values else 0,
        )


class CallAnalyticsAggregator:
    """
    Specialized aggregator for call analytics.

    Provides pre-built aggregations for common call metrics.
    """

    def __init__(self, data_store: DataStore):
        """
        Initialize aggregator.

        Args:
            data_store: Data storage backend
        """
        self.data_store = data_store

    async def get_usage_summary(
        self,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
        agent_ids: Optional[List[str]] = None,
    ) -> UsageSummary:
        """
        Get usage summary for a time period.

        Args:
            organization_id: Organization ID
            start_date: Start date
            end_date: End date
            agent_ids: Optional agent filter

        Returns:
            Usage summary
        """
        params = QueryParams(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            agent_ids=agent_ids,
            limit=100000,  # Get all calls
        )

        calls = await self.data_store.query_calls(params)

        summary = UsageSummary(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
        )

        if not calls:
            return summary

        # Aggregate metrics
        agents = set()
        phone_numbers = set()

        for call in calls:
            summary.total_calls += 1

            if call.direction == "inbound":
                summary.inbound_calls += 1
            else:
                summary.outbound_calls += 1

            if call.outcome == CallOutcome.COMPLETED:
                summary.completed_calls += 1
            elif call.outcome in (CallOutcome.FAILED, CallOutcome.BUSY, CallOutcome.NO_ANSWER):
                summary.failed_calls += 1

            summary.total_minutes += call.duration_seconds / 60
            summary.total_cost_cents += call.cost_cents

            agents.add(call.agent_id)

        # Calculate rates
        if summary.total_calls > 0:
            summary.avg_call_duration_seconds = (
                summary.total_minutes * 60 / summary.total_calls
            )

            answered_calls = sum(
                1 for c in calls
                if c.outcome in (CallOutcome.COMPLETED, CallOutcome.TRANSFERRED)
            )
            summary.answer_rate = answered_calls / summary.total_calls

            summary.completion_rate = summary.completed_calls / summary.total_calls

            transferred_calls = sum(
                1 for c in calls if c.outcome == CallOutcome.TRANSFERRED
            )
            summary.transfer_rate = transferred_calls / summary.total_calls

        summary.unique_agents = len(agents)

        return summary

    async def get_agent_analytics(
        self,
        organization_id: str,
        agent_id: str,
        agent_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> AgentAnalytics:
        """
        Get analytics for a specific agent.

        Args:
            organization_id: Organization ID
            agent_id: Agent ID
            agent_name: Agent name
            start_date: Start date
            end_date: End date

        Returns:
            Agent analytics
        """
        params = QueryParams(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            agent_ids=[agent_id],
            limit=100000,
        )

        calls = await self.data_store.query_calls(params)

        analytics = AgentAnalytics(
            agent_id=agent_id,
            agent_name=agent_name,
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
        )

        if not calls:
            return analytics

        # Aggregate metrics
        sentiment_scores = []
        response_times = []
        first_response_times = []

        for call in calls:
            analytics.total_calls += 1
            analytics.total_talk_time_minutes += call.duration_seconds / 60
            analytics.total_tokens_used += call.llm_tokens_used
            analytics.total_function_calls += call.function_calls

            if call.outcome == CallOutcome.COMPLETED:
                analytics.successful_calls += 1
            elif call.outcome == CallOutcome.TRANSFERRED:
                analytics.transferred_calls += 1
            elif call.outcome in (CallOutcome.FAILED, CallOutcome.BUSY, CallOutcome.NO_ANSWER):
                analytics.failed_calls += 1

            if call.avg_response_time_ms > 0:
                response_times.append(call.avg_response_time_ms)

            if call.time_to_first_response_ms > 0:
                first_response_times.append(call.time_to_first_response_ms)

            if call.sentiment_score != 0:
                sentiment_scores.append(call.sentiment_score)

        # Calculate rates and averages
        if analytics.total_calls > 0:
            analytics.avg_call_duration_seconds = (
                analytics.total_talk_time_minutes * 60 / analytics.total_calls
            )
            analytics.completion_rate = analytics.successful_calls / analytics.total_calls
            analytics.transfer_rate = analytics.transferred_calls / analytics.total_calls
            analytics.avg_tokens_per_call = analytics.total_tokens_used / analytics.total_calls

        if response_times:
            analytics.avg_response_time_ms = sum(response_times) / len(response_times)

        if first_response_times:
            analytics.avg_time_to_first_response_ms = (
                sum(first_response_times) / len(first_response_times)
            )

        if sentiment_scores:
            analytics.avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
            analytics.positive_sentiment_rate = sum(
                1 for s in sentiment_scores if s > 0.3
            ) / len(sentiment_scores)
            analytics.negative_sentiment_rate = sum(
                1 for s in sentiment_scores if s < -0.3
            ) / len(sentiment_scores)

        return analytics

    async def get_outcome_distribution(
        self,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
        agent_ids: Optional[List[str]] = None,
    ) -> OutcomeDistribution:
        """
        Get distribution of call outcomes.

        Args:
            organization_id: Organization ID
            start_date: Start date
            end_date: End date
            agent_ids: Optional agent filter

        Returns:
            Outcome distribution
        """
        params = QueryParams(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            agent_ids=agent_ids,
            limit=100000,
        )

        calls = await self.data_store.query_calls(params)

        distribution = OutcomeDistribution(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
        )

        for call in calls:
            if call.outcome == CallOutcome.COMPLETED:
                distribution.completed += 1
            elif call.outcome == CallOutcome.VOICEMAIL:
                distribution.voicemail += 1
            elif call.outcome == CallOutcome.NO_ANSWER:
                distribution.no_answer += 1
            elif call.outcome == CallOutcome.BUSY:
                distribution.busy += 1
            elif call.outcome == CallOutcome.FAILED:
                distribution.failed += 1
            elif call.outcome == CallOutcome.TRANSFERRED:
                distribution.transferred += 1
            elif call.outcome == CallOutcome.CANCELED:
                distribution.canceled += 1

        return distribution

    async def get_cost_breakdown(
        self,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> CostBreakdown:
        """
        Get cost breakdown by category.

        Args:
            organization_id: Organization ID
            start_date: Start date
            end_date: End date

        Returns:
            Cost breakdown
        """
        # In production, this would query a separate cost tracking table
        # For now, estimate based on call metrics

        params = QueryParams(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            limit=100000,
        )

        calls = await self.data_store.query_calls(params)

        breakdown = CostBreakdown(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
        )

        for call in calls:
            # Estimate cost breakdown (in production, use actual cost data)
            total_cost = call.cost_cents

            # Rough estimates
            if call.direction == "inbound":
                breakdown.telephony_inbound_cents += int(total_cost * 0.3)
            else:
                breakdown.telephony_outbound_cents += int(total_cost * 0.3)

            breakdown.transcription_cents += int(total_cost * 0.1)
            breakdown.llm_input_cents += int(total_cost * 0.25)
            breakdown.llm_output_cents += int(total_cost * 0.25)
            breakdown.tts_cents += int(total_cost * 0.1)

        return breakdown

    async def get_call_volume_time_series(
        self,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
        granularity: TimeGranularity,
        agent_ids: Optional[List[str]] = None,
    ) -> TimeSeriesData:
        """
        Get call volume time series.

        Args:
            organization_id: Organization ID
            start_date: Start date
            end_date: End date
            granularity: Time granularity
            agent_ids: Optional agent filter

        Returns:
            Time series data
        """
        params = QueryParams(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            agent_ids=agent_ids,
            limit=100000,
        )

        calls = await self.data_store.query_calls(params)

        # Group by time bucket
        buckets: Dict[datetime, int] = defaultdict(int)
        for call in calls:
            bucket = get_time_bucket(call.timestamp, granularity)
            buckets[bucket] += 1

        # Build time series
        data_points = []
        values = []
        current = get_time_bucket(start_date, granularity)

        while current < end_date:
            value = buckets.get(current, 0)
            data_points.append(TimeSeriesDataPoint(
                timestamp=current,
                value=value,
            ))
            values.append(value)
            current = get_next_bucket(current, granularity)

        return TimeSeriesData(
            metric_name="call_volume",
            granularity=granularity,
            start_time=start_date,
            end_time=end_date,
            data_points=data_points,
            total=sum(values),
            average=sum(values) / len(values) if values else 0,
            min_value=min(values) if values else 0,
            max_value=max(values) if values else 0,
        )


__all__ = [
    "get_time_bucket",
    "get_next_bucket",
    "DataStore",
    "InMemoryDataStore",
    "MetricsAggregator",
    "CallAnalyticsAggregator",
]
