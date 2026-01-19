"""Analytics report generator."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
import json


logger = structlog.get_logger()


class ReportPeriod(str, Enum):
    """Report time periods."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class ReportType(str, Enum):
    """Types of reports."""
    CALL_SUMMARY = "call_summary"
    AGENT_PERFORMANCE = "agent_performance"
    USAGE_BREAKDOWN = "usage_breakdown"
    QUALITY_METRICS = "quality_metrics"
    COST_ANALYSIS = "cost_analysis"
    TREND_ANALYSIS = "trend_analysis"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    report_type: ReportType
    period: ReportPeriod
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    agent_ids: List[str] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    include_charts: bool = True
    format: str = "json"  # json, csv, pdf


@dataclass
class CallSummaryReport:
    """Call summary report data."""
    period_start: datetime
    period_end: datetime
    total_calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0
    total_duration_seconds: float = 0.0
    avg_duration_seconds: float = 0.0
    inbound_calls: int = 0
    outbound_calls: int = 0
    success_rate: float = 0.0
    avg_wait_time_seconds: float = 0.0
    peak_concurrent: int = 0
    by_agent: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_hour: Dict[int, int] = field(default_factory=dict)
    by_day: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "summary": {
                "total_calls": self.total_calls,
                "completed_calls": self.completed_calls,
                "failed_calls": self.failed_calls,
                "success_rate": round(self.success_rate * 100, 2),
                "total_duration_minutes": round(self.total_duration_seconds / 60, 2),
                "avg_duration_seconds": round(self.avg_duration_seconds, 2),
            },
            "by_direction": {
                "inbound": self.inbound_calls,
                "outbound": self.outbound_calls,
            },
            "by_agent": self.by_agent,
            "by_hour": self.by_hour,
            "by_day": self.by_day,
        }


@dataclass
class AgentPerformanceReport:
    """Agent performance report data."""
    agent_id: str
    agent_name: str
    period_start: datetime
    period_end: datetime
    total_calls: int = 0
    avg_call_duration: float = 0.0
    avg_response_time_ms: float = 0.0
    successful_outcomes: int = 0
    failed_outcomes: int = 0
    transfers_to_human: int = 0
    avg_sentiment_score: float = 0.0
    avg_customer_satisfaction: float = 0.0
    total_talk_time_seconds: float = 0.0
    interruption_rate: float = 0.0
    function_usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        success_rate = 0.0
        if self.total_calls > 0:
            success_rate = self.successful_outcomes / self.total_calls

        return {
            "agent": {
                "id": self.agent_id,
                "name": self.agent_name,
            },
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "call_metrics": {
                "total_calls": self.total_calls,
                "avg_duration_seconds": round(self.avg_call_duration, 2),
                "total_talk_time_minutes": round(self.total_talk_time_seconds / 60, 2),
            },
            "performance": {
                "success_rate": round(success_rate * 100, 2),
                "avg_response_time_ms": round(self.avg_response_time_ms, 2),
                "transfer_rate": round(
                    self.transfers_to_human / max(self.total_calls, 1) * 100, 2
                ),
                "interruption_rate": round(self.interruption_rate * 100, 2),
            },
            "quality": {
                "avg_sentiment": round(self.avg_sentiment_score, 2),
                "avg_satisfaction": round(self.avg_customer_satisfaction, 2),
            },
            "function_usage": self.function_usage,
        }


@dataclass
class UsageBreakdownReport:
    """Usage breakdown report data."""
    period_start: datetime
    period_end: datetime
    total_minutes: float = 0.0
    asr_minutes: float = 0.0
    tts_characters: int = 0
    llm_tokens: int = 0
    llm_requests: int = 0
    storage_bytes: int = 0
    by_agent: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_provider: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    daily_breakdown: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "totals": {
                "call_minutes": round(self.total_minutes, 2),
                "asr_minutes": round(self.asr_minutes, 2),
                "tts_characters": self.tts_characters,
                "llm_tokens": self.llm_tokens,
                "llm_requests": self.llm_requests,
                "storage_mb": round(self.storage_bytes / (1024 * 1024), 2),
            },
            "by_agent": self.by_agent,
            "by_provider": self.by_provider,
            "daily": self.daily_breakdown,
        }


class ReportGenerator:
    """
    Generates analytics reports.

    Aggregates data from multiple sources to create
    comprehensive reports on calls, performance, and usage.
    """

    def __init__(self, db_session, metrics_collector=None):
        self._db = db_session
        self._metrics = metrics_collector

    async def generate(self, config: ReportConfig) -> Dict[str, Any]:
        """
        Generate a report based on configuration.

        Args:
            config: Report configuration

        Returns:
            Report data
        """
        # Determine date range
        start_date, end_date = self._get_date_range(config)

        logger.info(
            "generating_report",
            type=config.report_type.value,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        )

        if config.report_type == ReportType.CALL_SUMMARY:
            return await self._generate_call_summary(
                start_date, end_date, config.agent_ids
            )
        elif config.report_type == ReportType.AGENT_PERFORMANCE:
            return await self._generate_agent_performance(
                start_date, end_date, config.agent_ids
            )
        elif config.report_type == ReportType.USAGE_BREAKDOWN:
            return await self._generate_usage_breakdown(
                start_date, end_date, config.agent_ids
            )
        elif config.report_type == ReportType.QUALITY_METRICS:
            return await self._generate_quality_metrics(
                start_date, end_date, config.agent_ids
            )
        elif config.report_type == ReportType.COST_ANALYSIS:
            return await self._generate_cost_analysis(
                start_date, end_date, config.agent_ids
            )
        else:
            raise ValueError(f"Unknown report type: {config.report_type}")

    def _get_date_range(
        self,
        config: ReportConfig,
    ) -> tuple[datetime, datetime]:
        """Calculate date range from config."""
        now = datetime.utcnow()

        if config.start_date and config.end_date:
            return config.start_date, config.end_date

        if config.period == ReportPeriod.HOURLY:
            start = now - timedelta(hours=1)
        elif config.period == ReportPeriod.DAILY:
            start = now - timedelta(days=1)
        elif config.period == ReportPeriod.WEEKLY:
            start = now - timedelta(weeks=1)
        elif config.period == ReportPeriod.MONTHLY:
            start = now - timedelta(days=30)
        else:
            start = now - timedelta(days=1)

        return start, now

    async def _generate_call_summary(
        self,
        start_date: datetime,
        end_date: datetime,
        agent_ids: List[str],
    ) -> Dict[str, Any]:
        """Generate call summary report."""
        # Query calls from database
        query = """
            SELECT
                COUNT(*) as total_calls,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(duration_seconds) as total_duration,
                AVG(duration_seconds) as avg_duration,
                SUM(CASE WHEN direction = 'inbound' THEN 1 ELSE 0 END) as inbound,
                SUM(CASE WHEN direction = 'outbound' THEN 1 ELSE 0 END) as outbound
            FROM calls
            WHERE started_at >= :start_date AND started_at < :end_date
        """

        if agent_ids:
            query += " AND agent_id = ANY(:agent_ids)"

        # For now, return mock data (in production, execute query)
        report = CallSummaryReport(
            period_start=start_date,
            period_end=end_date,
            total_calls=150,
            completed_calls=142,
            failed_calls=8,
            total_duration_seconds=18500,
            avg_duration_seconds=123.3,
            inbound_calls=95,
            outbound_calls=55,
            success_rate=0.947,
            avg_wait_time_seconds=2.5,
            peak_concurrent=12,
            by_hour={h: 5 + (h % 8) for h in range(24)},
            by_day={
                "Monday": 25,
                "Tuesday": 28,
                "Wednesday": 22,
                "Thursday": 30,
                "Friday": 26,
                "Saturday": 12,
                "Sunday": 7,
            },
        )

        return {
            "report_type": "call_summary",
            "generated_at": datetime.utcnow().isoformat(),
            "data": report.to_dict(),
        }

    async def _generate_agent_performance(
        self,
        start_date: datetime,
        end_date: datetime,
        agent_ids: List[str],
    ) -> Dict[str, Any]:
        """Generate agent performance report."""
        # Mock data for demonstration
        agents = [
            AgentPerformanceReport(
                agent_id="agent-1",
                agent_name="Sales Assistant",
                period_start=start_date,
                period_end=end_date,
                total_calls=75,
                avg_call_duration=145.5,
                avg_response_time_ms=850,
                successful_outcomes=68,
                failed_outcomes=7,
                transfers_to_human=3,
                avg_sentiment_score=0.72,
                avg_customer_satisfaction=4.2,
                total_talk_time_seconds=10912,
                interruption_rate=0.08,
                function_usage={
                    "check_availability": 45,
                    "book_appointment": 32,
                    "get_pricing": 28,
                },
            ),
            AgentPerformanceReport(
                agent_id="agent-2",
                agent_name="Support Agent",
                period_start=start_date,
                period_end=end_date,
                total_calls=75,
                avg_call_duration=98.2,
                avg_response_time_ms=720,
                successful_outcomes=70,
                failed_outcomes=5,
                transfers_to_human=8,
                avg_sentiment_score=0.65,
                avg_customer_satisfaction=4.0,
                total_talk_time_seconds=7365,
                interruption_rate=0.12,
                function_usage={
                    "lookup_order": 52,
                    "check_status": 48,
                    "submit_ticket": 15,
                },
            ),
        ]

        return {
            "report_type": "agent_performance",
            "generated_at": datetime.utcnow().isoformat(),
            "data": {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "agents": [a.to_dict() for a in agents],
            },
        }

    async def _generate_usage_breakdown(
        self,
        start_date: datetime,
        end_date: datetime,
        agent_ids: List[str],
    ) -> Dict[str, Any]:
        """Generate usage breakdown report."""
        report = UsageBreakdownReport(
            period_start=start_date,
            period_end=end_date,
            total_minutes=308.3,
            asr_minutes=154.2,
            tts_characters=125000,
            llm_tokens=450000,
            llm_requests=1250,
            storage_bytes=52428800,
            by_provider={
                "openai": {"tokens": 350000, "requests": 950},
                "anthropic": {"tokens": 100000, "requests": 300},
            },
        )

        return {
            "report_type": "usage_breakdown",
            "generated_at": datetime.utcnow().isoformat(),
            "data": report.to_dict(),
        }

    async def _generate_quality_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        agent_ids: List[str],
    ) -> Dict[str, Any]:
        """Generate quality metrics report."""
        return {
            "report_type": "quality_metrics",
            "generated_at": datetime.utcnow().isoformat(),
            "data": {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "speech_quality": {
                    "asr_accuracy": 0.94,
                    "avg_confidence": 0.87,
                    "low_confidence_rate": 0.08,
                },
                "response_quality": {
                    "avg_latency_ms": 1250,
                    "p95_latency_ms": 2100,
                    "timeout_rate": 0.02,
                },
                "conversation_quality": {
                    "avg_turns": 8.5,
                    "completion_rate": 0.89,
                    "barge_in_rate": 0.15,
                    "silence_rate": 0.05,
                },
                "customer_satisfaction": {
                    "avg_rating": 4.1,
                    "nps_score": 42,
                    "positive_sentiment": 0.72,
                    "negative_sentiment": 0.12,
                },
            },
        }

    async def _generate_cost_analysis(
        self,
        start_date: datetime,
        end_date: datetime,
        agent_ids: List[str],
    ) -> Dict[str, Any]:
        """Generate cost analysis report."""
        return {
            "report_type": "cost_analysis",
            "generated_at": datetime.utcnow().isoformat(),
            "data": {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "total_cost_cents": 15420,
                "by_category": {
                    "telephony": 4500,
                    "asr": 3200,
                    "tts": 2800,
                    "llm": 4920,
                },
                "by_agent": {
                    "agent-1": 8200,
                    "agent-2": 7220,
                },
                "cost_per_call_cents": 102.8,
                "cost_per_minute_cents": 50.1,
                "trend": {
                    "change_percent": -5.2,
                    "direction": "decreasing",
                },
            },
        }

    async def schedule_report(
        self,
        config: ReportConfig,
        schedule: str,  # cron expression
        recipient_emails: List[str],
    ) -> str:
        """Schedule a recurring report."""
        # In production: store schedule in database
        report_id = f"report-{datetime.utcnow().timestamp()}"

        logger.info(
            "report_scheduled",
            report_id=report_id,
            type=config.report_type.value,
            schedule=schedule,
            recipients=recipient_emails,
        )

        return report_id

    async def export_report(
        self,
        report_data: Dict[str, Any],
        format: str = "json",
    ) -> bytes:
        """Export report to specified format."""
        if format == "json":
            return json.dumps(report_data, indent=2).encode()
        elif format == "csv":
            return self._to_csv(report_data)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _to_csv(self, data: Dict[str, Any]) -> bytes:
        """Convert report data to CSV."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Flatten and write data
        if "data" in data:
            self._write_dict_to_csv(writer, data["data"])

        return output.getvalue().encode()

    def _write_dict_to_csv(
        self,
        writer,
        data: Dict[str, Any],
        prefix: str = "",
    ) -> None:
        """Recursively write dict to CSV."""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._write_dict_to_csv(writer, value, full_key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._write_dict_to_csv(writer, item, f"{full_key}[{i}]")
                    else:
                        writer.writerow([f"{full_key}[{i}]", item])
            else:
                writer.writerow([full_key, value])
