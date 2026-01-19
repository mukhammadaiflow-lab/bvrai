"""
Metrics Aggregator.

Computes aggregated metrics from analytics events.
"""

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import statistics

from ..config import (
    MetricType,
    AggregationPeriod,
    EventType,
    CallOutcome,
    SentimentCategory,
    LatencyComponent,
    get_settings,
)
from ..models import (
    AnalyticsEvent,
    CallEvent,
    ConversationEvent,
    LatencyEvent,
    MetricValue,
    MetricSeries,
    AggregatedMetric,
)

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """
    Aggregates events into metrics.

    Features:
    - Time-based aggregation
    - Dimensional aggregation
    - Statistical calculations
    - Real-time and batch modes
    """

    def __init__(self):
        """Initialize aggregator."""
        self.settings = get_settings()
        self.config = self.settings.aggregator

        # In-memory aggregation buckets
        self._buckets: Dict[str, Dict[datetime, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Dimensional aggregations
        self._dimensional: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )

    def aggregate_events(
        self,
        events: List[AnalyticsEvent],
        period: AggregationPeriod = AggregationPeriod.HOUR,
    ) -> List[AggregatedMetric]:
        """
        Aggregate a batch of events.

        Args:
            events: Events to aggregate
            period: Aggregation period

        Returns:
            List of aggregated metrics
        """
        # Group events by type
        by_type: Dict[EventType, List[AnalyticsEvent]] = defaultdict(list)
        for event in events:
            by_type[event.event_type].append(event)

        metrics = []

        # Call metrics
        call_events = (
            by_type[EventType.CALL_STARTED]
            + by_type[EventType.CALL_ENDED]
            + by_type[EventType.CALL_CONNECTED]
        )
        if call_events:
            metrics.extend(self._aggregate_call_metrics(call_events, period))

        # Latency metrics
        latency_events = by_type[EventType.LATENCY_RECORDED]
        if latency_events:
            metrics.extend(self._aggregate_latency_metrics(latency_events, period))

        # Conversation metrics
        conv_events = (
            by_type[EventType.INTENT_DETECTED]
            + by_type[EventType.SENTIMENT_CHANGED]
        )
        if conv_events:
            metrics.extend(self._aggregate_conversation_metrics(conv_events, period))

        # Business metrics
        business_events = (
            by_type[EventType.GOAL_ACHIEVED]
            + by_type[EventType.CONVERSION]
            + by_type[EventType.CSAT_RECORDED]
        )
        if business_events:
            metrics.extend(self._aggregate_business_metrics(business_events, period))

        return metrics

    def _aggregate_call_metrics(
        self,
        events: List[AnalyticsEvent],
        period: AggregationPeriod,
    ) -> List[AggregatedMetric]:
        """Aggregate call-related metrics."""
        metrics = []

        # Group by period
        buckets = self._group_by_period(events, period)

        for period_start, bucket_events in buckets.items():
            period_end = self._get_period_end(period_start, period)

            # Call volume
            call_started = [
                e for e in bucket_events if e.event_type == EventType.CALL_STARTED
            ]
            if call_started:
                metrics.append(
                    AggregatedMetric(
                        metric_type=MetricType.CALL_VOLUME,
                        period=period,
                        period_start=period_start,
                        period_end=period_end,
                        count=len(call_started),
                        sum=len(call_started),
                        avg=len(call_started),
                        min=len(call_started),
                        max=len(call_started),
                    )
                )

            # Call duration
            call_ended = [
                e for e in bucket_events
                if e.event_type == EventType.CALL_ENDED and isinstance(e, CallEvent)
            ]
            durations = [
                e.duration_ms for e in call_ended
                if e.duration_ms is not None
            ]

            if durations:
                metrics.append(
                    self._create_aggregated_metric(
                        MetricType.CALL_DURATION,
                        period,
                        period_start,
                        period_end,
                        durations,
                    )
                )

            # Call outcomes
            outcomes = [
                e.outcome for e in call_ended
                if e.outcome is not None
            ]
            if outcomes:
                for outcome in CallOutcome:
                    count = sum(1 for o in outcomes if o == outcome)
                    if count > 0:
                        metrics.append(
                            AggregatedMetric(
                                metric_type=MetricType.CALL_OUTCOME,
                                period=period,
                                period_start=period_start,
                                period_end=period_end,
                                count=count,
                                sum=count,
                                avg=count / len(outcomes),
                                min=count,
                                max=count,
                                dimensions={"outcome": outcome.value},
                            )
                        )

        return metrics

    def _aggregate_latency_metrics(
        self,
        events: List[AnalyticsEvent],
        period: AggregationPeriod,
    ) -> List[AggregatedMetric]:
        """Aggregate latency metrics."""
        metrics = []

        # Group by component
        by_component: Dict[LatencyComponent, List[float]] = defaultdict(list)
        for event in events:
            if isinstance(event, LatencyEvent):
                by_component[event.component].append(event.latency_ms)

        # Create metrics for each component
        buckets = self._group_by_period(events, period)

        for period_start, bucket_events in buckets.items():
            period_end = self._get_period_end(period_start, period)

            for component, latencies in by_component.items():
                if latencies:
                    metric = self._create_aggregated_metric(
                        MetricType.LATENCY,
                        period,
                        period_start,
                        period_end,
                        latencies,
                    )
                    metric.dimensions = {"component": component.value}
                    metrics.append(metric)

        return metrics

    def _aggregate_conversation_metrics(
        self,
        events: List[AnalyticsEvent],
        period: AggregationPeriod,
    ) -> List[AggregatedMetric]:
        """Aggregate conversation metrics."""
        metrics = []

        buckets = self._group_by_period(events, period)

        for period_start, bucket_events in buckets.items():
            period_end = self._get_period_end(period_start, period)

            # Intent distribution
            intents: Dict[str, int] = defaultdict(int)
            for event in bucket_events:
                if (
                    isinstance(event, ConversationEvent)
                    and event.event_type == EventType.INTENT_DETECTED
                    and event.intent
                ):
                    intents[event.intent] += 1

            for intent, count in intents.items():
                metrics.append(
                    AggregatedMetric(
                        metric_type=MetricType.INTENT,
                        period=period,
                        period_start=period_start,
                        period_end=period_end,
                        count=count,
                        sum=count,
                        avg=count / len(bucket_events) if bucket_events else 0,
                        min=count,
                        max=count,
                        dimensions={"intent": intent},
                    )
                )

            # Sentiment distribution
            sentiments: Dict[SentimentCategory, int] = defaultdict(int)
            sentiment_scores: List[float] = []

            for event in bucket_events:
                if isinstance(event, ConversationEvent) and event.sentiment:
                    sentiments[event.sentiment] += 1
                    if event.sentiment_score is not None:
                        sentiment_scores.append(event.sentiment_score)

            if sentiment_scores:
                metric = self._create_aggregated_metric(
                    MetricType.SENTIMENT,
                    period,
                    period_start,
                    period_end,
                    sentiment_scores,
                )
                metrics.append(metric)

        return metrics

    def _aggregate_business_metrics(
        self,
        events: List[AnalyticsEvent],
        period: AggregationPeriod,
    ) -> List[AggregatedMetric]:
        """Aggregate business metrics."""
        metrics = []

        buckets = self._group_by_period(events, period)

        for period_start, bucket_events in buckets.items():
            period_end = self._get_period_end(period_start, period)

            # Conversions
            conversions = [
                e for e in bucket_events if e.event_type == EventType.CONVERSION
            ]
            if conversions:
                metrics.append(
                    AggregatedMetric(
                        metric_type=MetricType.CONVERSION,
                        period=period,
                        period_start=period_start,
                        period_end=period_end,
                        count=len(conversions),
                        sum=len(conversions),
                        avg=len(conversions),
                        min=len(conversions),
                        max=len(conversions),
                    )
                )

            # CSAT scores
            csat_events = [
                e for e in bucket_events if e.event_type == EventType.CSAT_RECORDED
            ]
            csat_scores = [
                e.data.get("score") for e in csat_events
                if e.data.get("score") is not None
            ]

            if csat_scores:
                metric = self._create_aggregated_metric(
                    MetricType.CSAT,
                    period,
                    period_start,
                    period_end,
                    csat_scores,
                )
                metrics.append(metric)

            # Goal completions
            goals = [
                e for e in bucket_events if e.event_type == EventType.GOAL_ACHIEVED
            ]
            if goals:
                # Group by goal name
                by_goal: Dict[str, int] = defaultdict(int)
                for goal in goals:
                    goal_name = goal.data.get("goal_name", "unknown")
                    by_goal[goal_name] += 1

                for goal_name, count in by_goal.items():
                    metrics.append(
                        AggregatedMetric(
                            metric_type=MetricType.GOAL_COMPLETION,
                            period=period,
                            period_start=period_start,
                            period_end=period_end,
                            count=count,
                            sum=count,
                            avg=count,
                            min=count,
                            max=count,
                            dimensions={"goal": goal_name},
                        )
                    )

        return metrics

    def _group_by_period(
        self,
        events: List[AnalyticsEvent],
        period: AggregationPeriod,
    ) -> Dict[datetime, List[AnalyticsEvent]]:
        """Group events by time period."""
        buckets: Dict[datetime, List[AnalyticsEvent]] = defaultdict(list)

        for event in events:
            bucket_time = self._truncate_to_period(event.timestamp, period)
            buckets[bucket_time].append(event)

        return buckets

    def _truncate_to_period(
        self,
        dt: datetime,
        period: AggregationPeriod,
    ) -> datetime:
        """Truncate datetime to period start."""
        if period == AggregationPeriod.MINUTE:
            return dt.replace(second=0, microsecond=0)
        elif period == AggregationPeriod.HOUR:
            return dt.replace(minute=0, second=0, microsecond=0)
        elif period == AggregationPeriod.DAY:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == AggregationPeriod.WEEK:
            start = dt - timedelta(days=dt.weekday())
            return start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == AggregationPeriod.MONTH:
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

    def _get_period_end(
        self,
        start: datetime,
        period: AggregationPeriod,
    ) -> datetime:
        """Get period end from start."""
        if period == AggregationPeriod.MINUTE:
            return start + timedelta(minutes=1)
        elif period == AggregationPeriod.HOUR:
            return start + timedelta(hours=1)
        elif period == AggregationPeriod.DAY:
            return start + timedelta(days=1)
        elif period == AggregationPeriod.WEEK:
            return start + timedelta(weeks=1)
        elif period == AggregationPeriod.MONTH:
            if start.month == 12:
                return start.replace(year=start.year + 1, month=1)
            return start.replace(month=start.month + 1)
        else:
            return start.replace(year=start.year + 1)

    def _create_aggregated_metric(
        self,
        metric_type: MetricType,
        period: AggregationPeriod,
        period_start: datetime,
        period_end: datetime,
        values: List[float],
    ) -> AggregatedMetric:
        """Create aggregated metric with statistics."""
        if not values:
            return AggregatedMetric(
                metric_type=metric_type,
                period=period,
                period_start=period_start,
                period_end=period_end,
            )

        sorted_values = sorted(values)
        count = len(values)

        return AggregatedMetric(
            metric_type=metric_type,
            period=period,
            period_start=period_start,
            period_end=period_end,
            count=count,
            sum=sum(values),
            avg=statistics.mean(values),
            min=min(values),
            max=max(values),
            std_dev=statistics.stdev(values) if count > 1 else 0.0,
            p50=self._percentile(sorted_values, 50),
            p75=self._percentile(sorted_values, 75),
            p90=self._percentile(sorted_values, 90),
            p95=self._percentile(sorted_values, 95),
            p99=self._percentile(sorted_values, 99),
        )

    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0

        k = (len(sorted_values) - 1) * (percentile / 100)
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return sorted_values[int(k)]

        return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)

    def get_real_time_metrics(
        self,
        metric_type: MetricType,
        window_seconds: int = 60,
    ) -> Dict[str, Any]:
        """Get real-time metrics for dashboard."""
        # Would query from real-time storage
        return {
            "metric_type": metric_type.value,
            "window_seconds": window_seconds,
            "current_value": 0,
            "trend": "stable",
            "change_percent": 0,
        }
