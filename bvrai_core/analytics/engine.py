"""
Analytics Engine Module

This module provides the main analytics engine that coordinates
all analytics components.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Union,
)

from .base import (
    ReportType,
    TimeGranularity,
    CallOutcome,
    CallMetricsSnapshot,
    UsageSummary,
    AgentAnalytics,
    TimeSeriesData,
    OutcomeDistribution,
    CostBreakdown,
    QueryParams,
)
from .collectors import (
    MetricsCollector,
    CallMetricsCollector,
    MetricsBackend,
    InMemoryMetricsBackend,
)
from .aggregators import (
    DataStore,
    InMemoryDataStore,
    MetricsAggregator,
    CallAnalyticsAggregator,
)
from .reports import (
    ExportFormat,
    ReportConfig,
    GeneratedReport,
    ReportGenerator,
    UsageSummaryReportGenerator,
    AgentPerformanceReportGenerator,
    CostAnalysisReportGenerator,
    ReportExporter,
    ReportScheduler,
)


logger = logging.getLogger(__name__)


@dataclass
class AnalyticsEngineConfig:
    """Configuration for analytics engine."""

    # Storage
    metrics_backend: Optional[MetricsBackend] = None
    data_store: Optional[DataStore] = None

    # Collection settings
    metrics_flush_interval_seconds: float = 10.0
    metrics_buffer_size: int = 1000

    # Aggregation settings
    default_granularity: TimeGranularity = TimeGranularity.DAY
    max_query_range_days: int = 90

    # Report settings
    enable_scheduled_reports: bool = True

    # Cache settings
    cache_ttl_seconds: int = 300


class AnalyticsEngine:
    """
    Main analytics engine.

    Coordinates:
    - Metrics collection
    - Data aggregation
    - Report generation
    - Scheduled reports
    """

    def __init__(self, config: Optional[AnalyticsEngineConfig] = None):
        """
        Initialize analytics engine.

        Args:
            config: Engine configuration
        """
        self.config = config or AnalyticsEngineConfig()

        # Initialize backends
        self._metrics_backend = (
            self.config.metrics_backend or InMemoryMetricsBackend()
        )
        self._data_store = (
            self.config.data_store or InMemoryDataStore()
        )

        # Initialize collectors
        self._metrics_collector = MetricsCollector(
            backend=self._metrics_backend,
            flush_interval_seconds=self.config.metrics_flush_interval_seconds,
            buffer_size=self.config.metrics_buffer_size,
        )
        self._call_collector = CallMetricsCollector(self._metrics_collector)

        # Initialize aggregators
        self._metrics_aggregator = MetricsAggregator(self._metrics_backend)
        self._call_aggregator = CallAnalyticsAggregator(self._data_store)

        # Initialize report generators
        self._report_generators: Dict[ReportType, ReportGenerator] = {}
        self._setup_report_generators()

        # Initialize exporter and scheduler
        self._exporter = ReportExporter()
        self._scheduler: Optional[ReportScheduler] = None

        if self.config.enable_scheduled_reports:
            self._scheduler = ReportScheduler(
                generators=self._report_generators,
                exporter=self._exporter,
            )

        # Cache
        self._cache: Dict[str, Any] = {}
        self._cache_times: Dict[str, datetime] = {}

        # State
        self._running = False

    def _setup_report_generators(self) -> None:
        """Set up default report generators."""
        self._report_generators[ReportType.USAGE_SUMMARY] = (
            UsageSummaryReportGenerator(self._call_aggregator)
        )
        self._report_generators[ReportType.AGENT_PERFORMANCE] = (
            AgentPerformanceReportGenerator(self._call_aggregator)
        )
        self._report_generators[ReportType.COST_ANALYSIS] = (
            CostAnalysisReportGenerator(self._call_aggregator)
        )

    async def start(self) -> None:
        """Start the analytics engine."""
        logger.info("Starting analytics engine...")

        await self._metrics_collector.start()

        if self._scheduler:
            await self._scheduler.start()

        self._running = True
        logger.info("Analytics engine started")

    async def stop(self) -> None:
        """Stop the analytics engine."""
        logger.info("Stopping analytics engine...")

        self._running = False

        await self._metrics_collector.stop()

        if self._scheduler:
            await self._scheduler.stop()

        logger.info("Analytics engine stopped")

    # ==================== Collection Methods ====================

    async def record_call_started(
        self,
        organization_id: str,
        agent_id: str,
        direction: str,
        campaign_id: Optional[str] = None,
    ) -> None:
        """Record a call started event."""
        await self._call_collector.record_call_started(
            organization_id=organization_id,
            agent_id=agent_id,
            direction=direction,
            campaign_id=campaign_id,
        )

    async def record_call_ended(
        self,
        call_id: str,
        organization_id: str,
        agent_id: str,
        direction: str,
        outcome: CallOutcome,
        duration_seconds: float,
        campaign_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a call ended event with full metrics."""
        # Record to metrics collector
        await self._call_collector.record_call_ended(
            organization_id=organization_id,
            agent_id=agent_id,
            direction=direction,
            outcome=outcome,
            duration_seconds=duration_seconds,
            campaign_id=campaign_id,
        )

        # Create and store call metrics snapshot
        snapshot = CallMetricsSnapshot(
            call_id=call_id,
            organization_id=organization_id,
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            direction=direction,
            outcome=outcome,
            duration_seconds=duration_seconds,
            campaign_id=campaign_id,
        )

        # Add additional metrics if provided
        if metrics:
            snapshot.time_to_answer_ms = metrics.get("time_to_answer_ms", 0)
            snapshot.time_to_first_response_ms = metrics.get("time_to_first_response_ms", 0)
            snapshot.avg_response_time_ms = metrics.get("avg_response_time_ms", 0)
            snapshot.turn_count = metrics.get("turn_count", 0)
            snapshot.interruption_count = metrics.get("interruption_count", 0)
            snapshot.silence_seconds = metrics.get("silence_seconds", 0)
            snapshot.transcription_confidence = metrics.get("transcription_confidence", 0)
            snapshot.sentiment_score = metrics.get("sentiment_score", 0)
            snapshot.llm_tokens_used = metrics.get("llm_tokens_used", 0)
            snapshot.function_calls = metrics.get("function_calls", 0)
            snapshot.cost_cents = metrics.get("cost_cents", 0)
            snapshot.industry = metrics.get("industry")
            snapshot.custom_tags = metrics.get("custom_tags", {})

        await self._data_store.store_call_metrics(snapshot)

    async def record_response_time(
        self,
        organization_id: str,
        agent_id: str,
        response_time_ms: float,
    ) -> None:
        """Record agent response time."""
        await self._call_collector.record_response_time(
            organization_id=organization_id,
            agent_id=agent_id,
            response_time_ms=response_time_ms,
        )

    async def record_llm_usage(
        self,
        organization_id: str,
        agent_id: str,
        tokens: int,
        latency_ms: float,
        model: str,
    ) -> None:
        """Record LLM usage."""
        await self._call_collector.record_llm_usage(
            organization_id=organization_id,
            agent_id=agent_id,
            tokens=tokens,
            latency_ms=latency_ms,
            model=model,
        )

    async def record_function_call(
        self,
        organization_id: str,
        agent_id: str,
        function_name: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record a function call."""
        await self._call_collector.record_function_call(
            organization_id=organization_id,
            agent_id=agent_id,
            function_name=function_name,
            success=success,
            duration_ms=duration_ms,
        )

    # ==================== Query Methods ====================

    async def get_usage_summary(
        self,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
        agent_ids: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> UsageSummary:
        """Get usage summary for a time period."""
        cache_key = f"usage:{organization_id}:{start_date}:{end_date}:{agent_ids}"

        if use_cache and cache_key in self._cache:
            if self._is_cache_valid(cache_key):
                return self._cache[cache_key]

        summary = await self._call_aggregator.get_usage_summary(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            agent_ids=agent_ids,
        )

        if use_cache:
            self._cache[cache_key] = summary
            self._cache_times[cache_key] = datetime.utcnow()

        return summary

    async def get_agent_analytics(
        self,
        organization_id: str,
        agent_id: str,
        agent_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> AgentAnalytics:
        """Get analytics for a specific agent."""
        return await self._call_aggregator.get_agent_analytics(
            organization_id=organization_id,
            agent_id=agent_id,
            agent_name=agent_name,
            start_date=start_date,
            end_date=end_date,
        )

    async def get_outcome_distribution(
        self,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
        agent_ids: Optional[List[str]] = None,
    ) -> OutcomeDistribution:
        """Get distribution of call outcomes."""
        return await self._call_aggregator.get_outcome_distribution(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            agent_ids=agent_ids,
        )

    async def get_cost_breakdown(
        self,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> CostBreakdown:
        """Get cost breakdown."""
        return await self._call_aggregator.get_cost_breakdown(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
        )

    async def get_call_volume_time_series(
        self,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
        granularity: Optional[TimeGranularity] = None,
        agent_ids: Optional[List[str]] = None,
    ) -> TimeSeriesData:
        """Get call volume time series."""
        return await self._call_aggregator.get_call_volume_time_series(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity or self.config.default_granularity,
            agent_ids=agent_ids,
        )

    async def get_metric_time_series(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        granularity: Optional[TimeGranularity] = None,
        aggregation: str = "sum",
        tags: Optional[Dict[str, str]] = None,
    ) -> TimeSeriesData:
        """Get time series data for any metric."""
        return await self._metrics_aggregator.get_time_series(
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time,
            granularity=granularity or self.config.default_granularity,
            aggregation=aggregation,
            tags=tags,
        )

    async def query_calls(
        self,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
        agent_ids: Optional[List[str]] = None,
        campaign_ids: Optional[List[str]] = None,
        outcomes: Optional[List[CallOutcome]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CallMetricsSnapshot]:
        """Query call metrics snapshots."""
        params = QueryParams(
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            agent_ids=agent_ids,
            campaign_ids=campaign_ids,
            outcomes=outcomes,
            limit=limit,
            offset=offset,
        )
        return await self._data_store.query_calls(params)

    # ==================== Report Methods ====================

    async def generate_report(
        self,
        report_type: ReportType,
        organization_id: str,
        start_date: datetime,
        end_date: datetime,
        agent_ids: Optional[List[str]] = None,
        granularity: Optional[TimeGranularity] = None,
        format: ExportFormat = ExportFormat.JSON,
    ) -> GeneratedReport:
        """Generate a report."""
        generator = self._report_generators.get(report_type)
        if not generator:
            raise ValueError(f"No generator for report type: {report_type}")

        config = ReportConfig(
            report_type=report_type,
            organization_id=organization_id,
            start_date=start_date,
            end_date=end_date,
            agent_ids=agent_ids,
            granularity=granularity or self.config.default_granularity,
            format=format,
        )

        return await generator.generate(config)

    async def export_report(
        self,
        report: GeneratedReport,
        format: ExportFormat,
    ) -> Union[str, bytes]:
        """Export a report to specified format."""
        return await self._exporter.export(report, format)

    def schedule_report(
        self,
        report_type: ReportType,
        organization_id: str,
        schedule: str,  # "daily", "weekly", "monthly"
        delivery_method: str,
        delivery_config: Dict[str, Any],
        granularity: Optional[TimeGranularity] = None,
        format: ExportFormat = ExportFormat.JSON,
    ) -> Optional[str]:
        """Schedule a recurring report."""
        if not self._scheduler:
            logger.warning("Report scheduler not enabled")
            return None

        # Calculate date range based on schedule
        now = datetime.utcnow()
        if schedule == "daily":
            start_date = now - timedelta(days=1)
        elif schedule == "weekly":
            start_date = now - timedelta(weeks=1)
        else:
            start_date = now - timedelta(days=30)

        config = ReportConfig(
            report_type=report_type,
            organization_id=organization_id,
            start_date=start_date,
            end_date=now,
            granularity=granularity or self.config.default_granularity,
            format=format,
        )

        return self._scheduler.add_schedule(
            config=config,
            schedule=schedule,
            delivery_method=delivery_method,
            delivery_config=delivery_config,
        )

    # ==================== Dashboard Methods ====================

    async def get_dashboard_data(
        self,
        organization_id: str,
        period_days: int = 30,
    ) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)

        # Get all dashboard components in parallel
        usage_task = self.get_usage_summary(organization_id, start_date, end_date)
        outcomes_task = self.get_outcome_distribution(organization_id, start_date, end_date)
        volume_task = self.get_call_volume_time_series(
            organization_id, start_date, end_date,
            TimeGranularity.DAY,
        )

        usage, outcomes, volume = await asyncio.gather(
            usage_task, outcomes_task, volume_task
        )

        # Calculate trends (compare to previous period)
        prev_end = start_date
        prev_start = prev_end - timedelta(days=period_days)
        prev_usage = await self.get_usage_summary(organization_id, prev_start, prev_end)

        def calc_change(current: float, previous: float) -> float:
            if previous == 0:
                return 0
            return ((current - previous) / previous) * 100

        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": period_days,
            },
            "summary": usage.to_dict(),
            "outcomes": outcomes.to_dict(),
            "call_volume": volume.to_dict(),
            "trends": {
                "calls_change_percent": calc_change(usage.total_calls, prev_usage.total_calls),
                "duration_change_percent": calc_change(usage.total_minutes, prev_usage.total_minutes),
                "cost_change_percent": calc_change(usage.total_cost_cents, prev_usage.total_cost_cents),
                "completion_rate_change": (usage.completion_rate - prev_usage.completion_rate) * 100,
            },
        }

    # ==================== Cache Methods ====================

    def _is_cache_valid(self, key: str) -> bool:
        """Check if a cache entry is still valid."""
        if key not in self._cache_times:
            return False
        age = (datetime.utcnow() - self._cache_times[key]).total_seconds()
        return age < self.config.cache_ttl_seconds

    def clear_cache(self) -> None:
        """Clear the analytics cache."""
        self._cache.clear()
        self._cache_times.clear()


def create_analytics_engine(
    config: Optional[AnalyticsEngineConfig] = None,
) -> AnalyticsEngine:
    """
    Create an analytics engine with default configuration.

    Args:
        config: Optional engine configuration

    Returns:
        Configured analytics engine
    """
    return AnalyticsEngine(config)


__all__ = [
    "AnalyticsEngineConfig",
    "AnalyticsEngine",
    "create_analytics_engine",
]
