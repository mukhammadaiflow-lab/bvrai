"""
Report Generation Module

This module provides report generation capabilities for analytics,
including various report types and export formats.
"""

import asyncio
import csv
import io
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from .base import (
    ReportType,
    TimeGranularity,
    CallOutcome,
    UsageSummary,
    AgentAnalytics,
    TimeSeriesData,
    OutcomeDistribution,
    CostBreakdown,
    QueryParams,
)
from .aggregators import CallAnalyticsAggregator, DataStore


logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Report export formats."""

    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    XLSX = "xlsx"


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    report_type: ReportType
    organization_id: str
    start_date: datetime
    end_date: datetime

    # Filters
    agent_ids: Optional[List[str]] = None
    campaign_ids: Optional[List[str]] = None

    # Time series settings
    granularity: TimeGranularity = TimeGranularity.DAY

    # Output settings
    format: ExportFormat = ExportFormat.JSON
    include_charts: bool = False

    # Custom options
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportSection:
    """A section of a report."""

    title: str
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    chart_data: Optional[TimeSeriesData] = None


@dataclass
class GeneratedReport:
    """A generated report."""

    report_type: ReportType
    title: str
    organization_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime

    # Content
    summary: Dict[str, Any] = field(default_factory=dict)
    sections: List[ReportSection] = field(default_factory=list)

    # Metadata
    generation_time_ms: float = 0.0
    data_points_analyzed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_type": self.report_type.value,
            "title": self.title,
            "organization_id": self.organization_id,
            "generated_at": self.generated_at.isoformat(),
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "summary": self.summary,
            "sections": [
                {
                    "title": section.title,
                    "description": section.description,
                    "data": section.data,
                    "chart": section.chart_data.to_dict() if section.chart_data else None,
                }
                for section in self.sections
            ],
            "metadata": {
                "generation_time_ms": self.generation_time_ms,
                "data_points_analyzed": self.data_points_analyzed,
            },
        }


class ReportGenerator(ABC):
    """Abstract base class for report generators."""

    @property
    @abstractmethod
    def report_type(self) -> ReportType:
        """Report type this generator produces."""
        pass

    @abstractmethod
    async def generate(self, config: ReportConfig) -> GeneratedReport:
        """Generate the report."""
        pass


class UsageSummaryReportGenerator(ReportGenerator):
    """Generates usage summary reports."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.USAGE_SUMMARY

    def __init__(self, aggregator: CallAnalyticsAggregator):
        """
        Initialize generator.

        Args:
            aggregator: Call analytics aggregator
        """
        self.aggregator = aggregator

    async def generate(self, config: ReportConfig) -> GeneratedReport:
        """Generate usage summary report."""
        start_time = datetime.utcnow()

        # Get usage summary
        summary = await self.aggregator.get_usage_summary(
            organization_id=config.organization_id,
            start_date=config.start_date,
            end_date=config.end_date,
            agent_ids=config.agent_ids,
        )

        # Get call volume time series
        call_volume = await self.aggregator.get_call_volume_time_series(
            organization_id=config.organization_id,
            start_date=config.start_date,
            end_date=config.end_date,
            granularity=config.granularity,
            agent_ids=config.agent_ids,
        )

        # Get outcome distribution
        outcomes = await self.aggregator.get_outcome_distribution(
            organization_id=config.organization_id,
            start_date=config.start_date,
            end_date=config.end_date,
            agent_ids=config.agent_ids,
        )

        # Build report
        report = GeneratedReport(
            report_type=self.report_type,
            title="Usage Summary Report",
            organization_id=config.organization_id,
            generated_at=datetime.utcnow(),
            period_start=config.start_date,
            period_end=config.end_date,
            summary=summary.to_dict(),
        )

        # Add sections
        report.sections.append(ReportSection(
            title="Call Volume Over Time",
            description="Number of calls per time period",
            data={"total_calls": summary.total_calls},
            chart_data=call_volume,
        ))

        report.sections.append(ReportSection(
            title="Call Outcomes",
            description="Distribution of call outcomes",
            data=outcomes.to_dict(),
        ))

        report.sections.append(ReportSection(
            title="Key Metrics",
            description="Summary of key performance indicators",
            data={
                "answer_rate": f"{summary.answer_rate * 100:.1f}%",
                "completion_rate": f"{summary.completion_rate * 100:.1f}%",
                "avg_duration": f"{summary.avg_call_duration_seconds:.0f} seconds",
                "total_cost": f"${summary.total_cost_cents / 100:.2f}",
            },
        ))

        report.generation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        report.data_points_analyzed = summary.total_calls

        return report


class AgentPerformanceReportGenerator(ReportGenerator):
    """Generates agent performance reports."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.AGENT_PERFORMANCE

    def __init__(
        self,
        aggregator: CallAnalyticsAggregator,
        agent_lookup: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize generator.

        Args:
            aggregator: Call analytics aggregator
            agent_lookup: Map of agent IDs to names
        """
        self.aggregator = aggregator
        self.agent_lookup = agent_lookup or {}

    async def generate(self, config: ReportConfig) -> GeneratedReport:
        """Generate agent performance report."""
        start_time = datetime.utcnow()

        agent_ids = config.agent_ids or list(self.agent_lookup.keys())

        # Get analytics for each agent
        agent_analytics = []
        for agent_id in agent_ids:
            agent_name = self.agent_lookup.get(agent_id, agent_id)
            analytics = await self.aggregator.get_agent_analytics(
                organization_id=config.organization_id,
                agent_id=agent_id,
                agent_name=agent_name,
                start_date=config.start_date,
                end_date=config.end_date,
            )
            if analytics.total_calls > 0:
                agent_analytics.append(analytics)

        # Sort by total calls
        agent_analytics.sort(key=lambda a: a.total_calls, reverse=True)

        # Build report
        report = GeneratedReport(
            report_type=self.report_type,
            title="Agent Performance Report",
            organization_id=config.organization_id,
            generated_at=datetime.utcnow(),
            period_start=config.start_date,
            period_end=config.end_date,
        )

        # Summary
        total_calls = sum(a.total_calls for a in agent_analytics)
        total_minutes = sum(a.total_talk_time_minutes for a in agent_analytics)
        avg_completion = (
            sum(a.completion_rate for a in agent_analytics) / len(agent_analytics)
            if agent_analytics else 0
        )

        report.summary = {
            "total_agents": len(agent_analytics),
            "total_calls": total_calls,
            "total_minutes": total_minutes,
            "avg_completion_rate": avg_completion,
        }

        # Agent rankings section
        report.sections.append(ReportSection(
            title="Agent Rankings",
            description="Agents ranked by call volume",
            data={
                "rankings": [
                    {
                        "rank": i + 1,
                        "agent_name": a.agent_name,
                        "calls": a.total_calls,
                        "completion_rate": f"{a.completion_rate * 100:.1f}%",
                        "avg_response_ms": f"{a.avg_response_time_ms:.0f}ms",
                    }
                    for i, a in enumerate(agent_analytics[:10])
                ],
            },
        ))

        # Individual agent sections
        for analytics in agent_analytics[:5]:  # Top 5 detailed
            report.sections.append(ReportSection(
                title=f"Agent: {analytics.agent_name}",
                description=f"Performance metrics for {analytics.agent_name}",
                data=analytics.to_dict(),
            ))

        report.generation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        report.data_points_analyzed = total_calls

        return report


class CostAnalysisReportGenerator(ReportGenerator):
    """Generates cost analysis reports."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.COST_ANALYSIS

    def __init__(self, aggregator: CallAnalyticsAggregator):
        """
        Initialize generator.

        Args:
            aggregator: Call analytics aggregator
        """
        self.aggregator = aggregator

    async def generate(self, config: ReportConfig) -> GeneratedReport:
        """Generate cost analysis report."""
        start_time = datetime.utcnow()

        # Get cost breakdown
        breakdown = await self.aggregator.get_cost_breakdown(
            organization_id=config.organization_id,
            start_date=config.start_date,
            end_date=config.end_date,
        )

        # Get usage for context
        usage = await self.aggregator.get_usage_summary(
            organization_id=config.organization_id,
            start_date=config.start_date,
            end_date=config.end_date,
        )

        # Build report
        report = GeneratedReport(
            report_type=self.report_type,
            title="Cost Analysis Report",
            organization_id=config.organization_id,
            generated_at=datetime.utcnow(),
            period_start=config.start_date,
            period_end=config.end_date,
            summary={
                "total_cost_dollars": breakdown.total_cents / 100,
                "total_calls": usage.total_calls,
                "cost_per_call": (
                    breakdown.total_cents / usage.total_calls / 100
                    if usage.total_calls > 0 else 0
                ),
                "cost_per_minute": (
                    breakdown.total_cents / usage.total_minutes / 100
                    if usage.total_minutes > 0 else 0
                ),
            },
        )

        # Cost breakdown section
        report.sections.append(ReportSection(
            title="Cost Breakdown by Category",
            description="Breakdown of costs by service type",
            data=breakdown.to_dict(),
        ))

        # Cost efficiency section
        report.sections.append(ReportSection(
            title="Cost Efficiency Metrics",
            description="Key cost efficiency indicators",
            data={
                "avg_cost_per_call": f"${breakdown.total_cents / max(usage.total_calls, 1) / 100:.3f}",
                "avg_cost_per_minute": f"${breakdown.total_cents / max(usage.total_minutes, 1) / 100:.4f}",
                "ai_cost_percentage": f"{(breakdown.llm_input_cents + breakdown.llm_output_cents) / max(breakdown.total_cents, 1) * 100:.1f}%",
                "telephony_cost_percentage": f"{(breakdown.telephony_inbound_cents + breakdown.telephony_outbound_cents) / max(breakdown.total_cents, 1) * 100:.1f}%",
            },
        ))

        report.generation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        report.data_points_analyzed = usage.total_calls

        return report


class ReportExporter:
    """Exports reports to various formats."""

    async def export(
        self,
        report: GeneratedReport,
        format: ExportFormat,
    ) -> Union[str, bytes]:
        """
        Export report to specified format.

        Args:
            report: Report to export
            format: Export format

        Returns:
            Exported data
        """
        if format == ExportFormat.JSON:
            return await self._export_json(report)
        elif format == ExportFormat.CSV:
            return await self._export_csv(report)
        elif format == ExportFormat.PDF:
            return await self._export_pdf(report)
        elif format == ExportFormat.XLSX:
            return await self._export_xlsx(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def _export_json(self, report: GeneratedReport) -> str:
        """Export to JSON."""
        return json.dumps(report.to_dict(), indent=2, default=str)

    async def _export_csv(self, report: GeneratedReport) -> str:
        """Export to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Report Type", "Organization", "Period Start", "Period End", "Generated At"
        ])
        writer.writerow([
            report.report_type.value,
            report.organization_id,
            report.period_start.isoformat(),
            report.period_end.isoformat(),
            report.generated_at.isoformat(),
        ])
        writer.writerow([])

        # Summary
        writer.writerow(["Summary"])
        for key, value in report.summary.items():
            writer.writerow([key, value])
        writer.writerow([])

        # Sections
        for section in report.sections:
            writer.writerow([section.title])
            writer.writerow([section.description])
            for key, value in section.data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        writer.writerow([f"{key}.{sub_key}", sub_value])
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            for item_key, item_value in item.items():
                                writer.writerow([f"{key}[{i}].{item_key}", item_value])
                        else:
                            writer.writerow([f"{key}[{i}]", item])
                else:
                    writer.writerow([key, value])
            writer.writerow([])

        return output.getvalue()

    async def _export_pdf(self, report: GeneratedReport) -> bytes:
        """Export to PDF."""
        # In production, use a PDF library like reportlab or weasyprint
        # For now, return placeholder
        logger.warning("PDF export not implemented, returning JSON as bytes")
        json_str = await self._export_json(report)
        return json_str.encode("utf-8")

    async def _export_xlsx(self, report: GeneratedReport) -> bytes:
        """Export to Excel."""
        # In production, use openpyxl
        # For now, return CSV as bytes
        logger.warning("XLSX export not implemented, returning CSV as bytes")
        csv_str = await self._export_csv(report)
        return csv_str.encode("utf-8")


class ReportScheduler:
    """
    Schedules recurring report generation.

    Supports:
    - Daily, weekly, monthly schedules
    - Email delivery
    - Storage to various destinations
    """

    def __init__(
        self,
        generators: Dict[ReportType, ReportGenerator],
        exporter: ReportExporter,
    ):
        """
        Initialize scheduler.

        Args:
            generators: Report generators by type
            exporter: Report exporter
        """
        self.generators = generators
        self.exporter = exporter
        self._schedules: List[Dict[str, Any]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def add_schedule(
        self,
        config: ReportConfig,
        schedule: str,  # "daily", "weekly", "monthly"
        delivery_method: str,  # "email", "s3", "webhook"
        delivery_config: Dict[str, Any],
    ) -> str:
        """
        Add a scheduled report.

        Returns:
            Schedule ID
        """
        import uuid
        schedule_id = f"sched_{uuid.uuid4().hex[:12]}"

        self._schedules.append({
            "id": schedule_id,
            "config": config,
            "schedule": schedule,
            "delivery_method": delivery_method,
            "delivery_config": delivery_config,
            "last_run": None,
            "next_run": self._calculate_next_run(schedule),
        })

        return schedule_id

    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calculate next run time."""
        now = datetime.utcnow()

        if schedule == "daily":
            next_run = now.replace(hour=8, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
        elif schedule == "weekly":
            days_until_monday = (7 - now.weekday()) % 7 or 7
            next_run = now + timedelta(days=days_until_monday)
            next_run = next_run.replace(hour=8, minute=0, second=0, microsecond=0)
        elif schedule == "monthly":
            if now.month == 12:
                next_run = now.replace(year=now.year + 1, month=1, day=1)
            else:
                next_run = now.replace(month=now.month + 1, day=1)
            next_run = next_run.replace(hour=8, minute=0, second=0, microsecond=0)
        else:
            next_run = now + timedelta(days=1)

        return next_run

    async def start(self) -> None:
        """Start the scheduler."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Report scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Report scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._check_schedules()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Scheduler error: {e}")

    async def _check_schedules(self) -> None:
        """Check and run due schedules."""
        now = datetime.utcnow()

        for schedule in self._schedules:
            if schedule["next_run"] <= now:
                await self._run_scheduled_report(schedule)
                schedule["last_run"] = now
                schedule["next_run"] = self._calculate_next_run(schedule["schedule"])

    async def _run_scheduled_report(self, schedule: Dict[str, Any]) -> None:
        """Run a scheduled report."""
        config = schedule["config"]

        generator = self.generators.get(config.report_type)
        if not generator:
            logger.error(f"No generator for report type: {config.report_type}")
            return

        try:
            report = await generator.generate(config)
            exported = await self.exporter.export(report, config.format)

            # Deliver report
            await self._deliver_report(
                exported,
                schedule["delivery_method"],
                schedule["delivery_config"],
            )

            logger.info(f"Delivered scheduled report: {schedule['id']}")

        except Exception as e:
            logger.exception(f"Failed to run scheduled report {schedule['id']}: {e}")

    async def _deliver_report(
        self,
        report_data: Union[str, bytes],
        method: str,
        config: Dict[str, Any],
    ) -> None:
        """Deliver a report."""
        if method == "email":
            # In production, send email
            logger.info(f"Would send email to {config.get('recipients')}")
        elif method == "s3":
            # In production, upload to S3
            logger.info(f"Would upload to s3://{config.get('bucket')}/{config.get('key')}")
        elif method == "webhook":
            # In production, call webhook
            logger.info(f"Would call webhook: {config.get('url')}")
        else:
            logger.warning(f"Unknown delivery method: {method}")


__all__ = [
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
]
