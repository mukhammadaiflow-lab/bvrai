"""
Analytics Package

This package provides comprehensive analytics capabilities for the
voice agent platform, including:

- Real-time metrics collection
- Time-series data aggregation
- Report generation and scheduling
- Dashboard data APIs

Example usage:

    from bvrai_core.analytics import (
        AnalyticsEngine,
        AnalyticsEngineConfig,
        CallOutcome,
        TimeGranularity,
        ReportType,
        ExportFormat,
    )

    # Create analytics engine
    config = AnalyticsEngineConfig(
        metrics_flush_interval_seconds=10.0,
        enable_scheduled_reports=True,
    )
    engine = AnalyticsEngine(config)

    # Start engine
    await engine.start()

    # Record call metrics
    await engine.record_call_ended(
        call_id="call_123",
        organization_id="org_456",
        agent_id="agent_789",
        direction="inbound",
        outcome=CallOutcome.COMPLETED,
        duration_seconds=180.5,
    )

    # Get usage summary
    summary = await engine.get_usage_summary(
        organization_id="org_456",
        start_date=datetime.utcnow() - timedelta(days=30),
        end_date=datetime.utcnow(),
    )

    # Generate report
    report = await engine.generate_report(
        report_type=ReportType.USAGE_SUMMARY,
        organization_id="org_456",
        start_date=datetime.utcnow() - timedelta(days=30),
        end_date=datetime.utcnow(),
    )

    # Export report
    json_data = await engine.export_report(report, ExportFormat.JSON)

    # Stop engine
    await engine.stop()
"""

# Base types
from .base import (
    MetricType,
    TimeGranularity,
    ReportType,
    CallOutcome,
    MetricPoint,
    MetricDefinition,
    AggregatedMetric,
    CallMetricsSnapshot,
    UsageSummary,
    AgentAnalytics,
    TimeSeriesDataPoint,
    TimeSeriesData,
    OutcomeDistribution,
    CostBreakdown,
    QueryParams,
)

# Collectors
from .collectors import (
    MetricsBackend,
    InMemoryMetricsBackend,
    InfluxDBBackend,
    Counter,
    Gauge,
    Histogram,
    Timer,
    TimerContext,
    MetricsCollector,
    CallMetricsCollector,
)

# Aggregators
from .aggregators import (
    get_time_bucket,
    get_next_bucket,
    DataStore,
    InMemoryDataStore,
    MetricsAggregator,
    CallAnalyticsAggregator,
)

# Reports
from .reports import (
    ExportFormat,
    ReportConfig,
    ReportSection,
    GeneratedReport,
    ReportGenerator,
    UsageSummaryReportGenerator,
    AgentPerformanceReportGenerator,
    CostAnalysisReportGenerator,
    ReportExporter,
    ReportScheduler,
)

# Engine
from .engine import (
    AnalyticsEngineConfig,
    AnalyticsEngine,
    create_analytics_engine,
)


__all__ = [
    # Base types
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
    # Collectors
    "MetricsBackend",
    "InMemoryMetricsBackend",
    "InfluxDBBackend",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "TimerContext",
    "MetricsCollector",
    "CallMetricsCollector",
    # Aggregators
    "get_time_bucket",
    "get_next_bucket",
    "DataStore",
    "InMemoryDataStore",
    "MetricsAggregator",
    "CallAnalyticsAggregator",
    # Reports
    "ExportFormat",
    "ReportConfig",
    "ReportSection",
    "GeneratedReport",
    "ReportGenerator",
    "UsageSummaryReportGenerator",
    "AgentPerformanceReportGenerator",
    "CostAnalysisReportGenerator",
    "ReportExporter",
    "ReportScheduler",
    # Engine
    "AnalyticsEngineConfig",
    "AnalyticsEngine",
    "create_analytics_engine",
]
