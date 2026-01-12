"""Analytics service."""

from datetime import datetime, timedelta, date
from typing import Optional
from uuid import UUID

import structlog
from sqlalchemy import select, func, and_, extract, cast, Date
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    Agent,
    AgentStatus,
    Call,
    CallStatus,
    CallDirection,
)
from app.analytics.schemas import (
    DashboardOverview,
    DailyStats,
    HourlyDistribution,
    AgentPerformance,
    CallOutcome,
    AnalyticsReport,
    TimeRange,
    UsageMetrics,
    RealTimeMetrics,
)

logger = structlog.get_logger()


class AnalyticsService:
    """Service for analytics and reporting."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.logger = logger.bind(service="analytics")

    async def get_dashboard_overview(
        self,
        owner_id: Optional[UUID] = None,
    ) -> DashboardOverview:
        """Get main dashboard overview metrics."""
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=today_start.weekday())
        month_start = today_start.replace(day=1)

        # Base condition for owner filtering
        base_conditions = []
        if owner_id:
            # Get agent IDs for this owner
            agent_query = select(Agent.id).where(Agent.owner_id == owner_id)
            agent_result = await self.db.execute(agent_query)
            agent_ids = [r[0] for r in agent_result.fetchall()]
            if agent_ids:
                base_conditions.append(Call.agent_id.in_(agent_ids))
            else:
                # No agents, return empty overview
                return DashboardOverview()

        # Total calls
        total_query = select(func.count()).select_from(Call)
        if base_conditions:
            total_query = total_query.where(and_(*base_conditions))
        total_result = await self.db.execute(total_query)
        total_calls = total_result.scalar() or 0

        # Today's calls
        today_query = select(func.count()).select_from(Call).where(
            and_(Call.created_at >= today_start, *base_conditions)
        )
        today_result = await self.db.execute(today_query)
        calls_today = today_result.scalar() or 0

        # This week's calls
        week_query = select(func.count()).select_from(Call).where(
            and_(Call.created_at >= week_start, *base_conditions)
        )
        week_result = await self.db.execute(week_query)
        calls_this_week = week_result.scalar() or 0

        # This month's calls
        month_query = select(func.count()).select_from(Call).where(
            and_(Call.created_at >= month_start, *base_conditions)
        )
        month_result = await self.db.execute(month_query)
        calls_this_month = month_result.scalar() or 0

        # Total duration
        duration_query = select(func.sum(Call.duration_seconds)).where(
            and_(Call.duration_seconds.isnot(None), *base_conditions)
        )
        duration_result = await self.db.execute(duration_query)
        total_seconds = duration_result.scalar() or 0
        total_minutes = total_seconds // 60

        # Average duration
        avg_query = select(func.avg(Call.duration_seconds)).where(
            and_(
                Call.status == CallStatus.COMPLETED,
                Call.duration_seconds.isnot(None),
                *base_conditions,
            )
        )
        avg_result = await self.db.execute(avg_query)
        avg_duration = avg_result.scalar() or 0

        # Completion rate
        completed_query = select(func.count()).select_from(Call).where(
            and_(Call.status == CallStatus.COMPLETED, *base_conditions)
        )
        completed_result = await self.db.execute(completed_query)
        completed_calls = completed_result.scalar() or 0
        completion_rate = completed_calls / total_calls if total_calls > 0 else 0

        # Agent counts
        if owner_id:
            active_agents_query = select(func.count()).select_from(Agent).where(
                and_(Agent.owner_id == owner_id, Agent.status == AgentStatus.ACTIVE)
            )
            total_agents_query = select(func.count()).select_from(Agent).where(
                Agent.owner_id == owner_id
            )
        else:
            active_agents_query = select(func.count()).select_from(Agent).where(
                Agent.status == AgentStatus.ACTIVE
            )
            total_agents_query = select(func.count()).select_from(Agent)

        active_agents_result = await self.db.execute(active_agents_query)
        active_agents = active_agents_result.scalar() or 0

        total_agents_result = await self.db.execute(total_agents_query)
        total_agents = total_agents_result.scalar() or 0

        # Calculate trends (vs previous period)
        prev_week_start = week_start - timedelta(days=7)
        prev_week_query = select(func.count()).select_from(Call).where(
            and_(
                Call.created_at >= prev_week_start,
                Call.created_at < week_start,
                *base_conditions,
            )
        )
        prev_week_result = await self.db.execute(prev_week_query)
        prev_week_calls = prev_week_result.scalar() or 0

        calls_trend = 0
        if prev_week_calls > 0:
            calls_trend = ((calls_this_week - prev_week_calls) / prev_week_calls) * 100

        return DashboardOverview(
            total_calls=total_calls,
            calls_today=calls_today,
            calls_this_week=calls_this_week,
            calls_this_month=calls_this_month,
            total_minutes=total_minutes,
            avg_call_duration_seconds=float(avg_duration),
            completion_rate=completion_rate,
            avg_response_time_ms=0,  # TODO: Calculate from logs
            calls_trend_percentage=calls_trend,
            duration_trend_percentage=0,
            active_agents=active_agents,
            total_agents=total_agents,
        )

    async def get_daily_stats(
        self,
        start_date: date,
        end_date: date,
        owner_id: Optional[UUID] = None,
    ) -> list[DailyStats]:
        """Get daily statistics for a date range."""
        # Build agent filter
        agent_filter = []
        if owner_id:
            agent_query = select(Agent.id).where(Agent.owner_id == owner_id)
            agent_result = await self.db.execute(agent_query)
            agent_ids = [r[0] for r in agent_result.fetchall()]
            if agent_ids:
                agent_filter.append(Call.agent_id.in_(agent_ids))

        # Query daily stats
        query = (
            select(
                cast(Call.created_at, Date).label("date"),
                func.count().label("total_calls"),
                func.count().filter(Call.status == CallStatus.COMPLETED).label("completed_calls"),
                func.count().filter(Call.status == CallStatus.FAILED).label("failed_calls"),
                func.sum(Call.duration_seconds).label("total_duration"),
                func.avg(Call.duration_seconds).filter(
                    Call.status == CallStatus.COMPLETED
                ).label("avg_duration"),
            )
            .where(
                and_(
                    Call.created_at >= datetime.combine(start_date, datetime.min.time()),
                    Call.created_at < datetime.combine(end_date + timedelta(days=1), datetime.min.time()),
                    *agent_filter,
                )
            )
            .group_by(cast(Call.created_at, Date))
            .order_by(cast(Call.created_at, Date))
        )

        result = await self.db.execute(query)
        rows = result.fetchall()

        return [
            DailyStats(
                date=row.date,
                total_calls=row.total_calls or 0,
                completed_calls=row.completed_calls or 0,
                failed_calls=row.failed_calls or 0,
                total_duration_seconds=int(row.total_duration or 0),
                avg_duration_seconds=float(row.avg_duration or 0),
                unique_callers=0,  # TODO: Count distinct callers
            )
            for row in rows
        ]

    async def get_hourly_distribution(
        self,
        start_date: datetime,
        end_date: datetime,
        owner_id: Optional[UUID] = None,
    ) -> list[HourlyDistribution]:
        """Get hourly call distribution."""
        agent_filter = []
        if owner_id:
            agent_query = select(Agent.id).where(Agent.owner_id == owner_id)
            agent_result = await self.db.execute(agent_query)
            agent_ids = [r[0] for r in agent_result.fetchall()]
            if agent_ids:
                agent_filter.append(Call.agent_id.in_(agent_ids))

        query = (
            select(
                extract("hour", Call.created_at).label("hour"),
                func.count().label("count"),
            )
            .where(
                and_(
                    Call.created_at >= start_date,
                    Call.created_at < end_date,
                    *agent_filter,
                )
            )
            .group_by(extract("hour", Call.created_at))
            .order_by(extract("hour", Call.created_at))
        )

        result = await self.db.execute(query)
        rows = result.fetchall()

        # Fill in missing hours with 0
        hour_counts = {int(row.hour): row.count for row in rows}
        return [
            HourlyDistribution(hour=h, call_count=hour_counts.get(h, 0))
            for h in range(24)
        ]

    async def get_agent_performance(
        self,
        start_date: datetime,
        end_date: datetime,
        owner_id: Optional[UUID] = None,
    ) -> list[AgentPerformance]:
        """Get performance metrics per agent."""
        agent_filter = []
        if owner_id:
            agent_filter.append(Agent.owner_id == owner_id)

        query = (
            select(
                Agent.id,
                Agent.name,
                func.count(Call.id).label("total_calls"),
                func.count().filter(Call.status == CallStatus.COMPLETED).label("completed_calls"),
                func.count().filter(Call.status == CallStatus.FAILED).label("failed_calls"),
                func.avg(Call.duration_seconds).filter(
                    Call.status == CallStatus.COMPLETED
                ).label("avg_duration"),
            )
            .join(Call, Call.agent_id == Agent.id, isouter=True)
            .where(
                and_(
                    *agent_filter,
                    Call.created_at >= start_date,
                    Call.created_at < end_date,
                )
            )
            .group_by(Agent.id, Agent.name)
        )

        result = await self.db.execute(query)
        rows = result.fetchall()

        return [
            AgentPerformance(
                agent_id=row.id,
                agent_name=row.name,
                total_calls=row.total_calls or 0,
                completed_calls=row.completed_calls or 0,
                failed_calls=row.failed_calls or 0,
                avg_duration_seconds=float(row.avg_duration or 0),
                completion_rate=(
                    row.completed_calls / row.total_calls
                    if row.total_calls > 0
                    else 0
                ),
            )
            for row in rows
        ]

    async def get_call_outcomes(
        self,
        start_date: datetime,
        end_date: datetime,
        owner_id: Optional[UUID] = None,
    ) -> list[CallOutcome]:
        """Get call outcome distribution."""
        agent_filter = []
        if owner_id:
            agent_query = select(Agent.id).where(Agent.owner_id == owner_id)
            agent_result = await self.db.execute(agent_query)
            agent_ids = [r[0] for r in agent_result.fetchall()]
            if agent_ids:
                agent_filter.append(Call.agent_id.in_(agent_ids))

        query = (
            select(
                Call.status,
                func.count().label("count"),
            )
            .where(
                and_(
                    Call.created_at >= start_date,
                    Call.created_at < end_date,
                    *agent_filter,
                )
            )
            .group_by(Call.status)
        )

        result = await self.db.execute(query)
        rows = result.fetchall()

        total = sum(row.count for row in rows)

        return [
            CallOutcome(
                outcome=row.status.value,
                count=row.count,
                percentage=row.count / total * 100 if total > 0 else 0,
            )
            for row in rows
        ]

    async def get_full_report(
        self,
        start_date: datetime,
        end_date: datetime,
        owner_id: Optional[UUID] = None,
    ) -> AnalyticsReport:
        """Generate a full analytics report."""
        overview = await self.get_dashboard_overview(owner_id)
        daily_stats = await self.get_daily_stats(
            start_date.date(),
            end_date.date(),
            owner_id,
        )
        hourly_dist = await self.get_hourly_distribution(start_date, end_date, owner_id)
        agent_perf = await self.get_agent_performance(start_date, end_date, owner_id)
        outcomes = await self.get_call_outcomes(start_date, end_date, owner_id)

        return AnalyticsReport(
            time_range=TimeRange(start=start_date, end=end_date),
            overview=overview,
            daily_stats=daily_stats,
            hourly_distribution=hourly_dist,
            agent_performance=agent_perf,
            call_outcomes=outcomes,
            top_topics=[],  # TODO: Implement topic extraction
        )

    async def get_real_time_metrics(
        self,
        owner_id: Optional[UUID] = None,
    ) -> RealTimeMetrics:
        """Get real-time metrics for live dashboard."""
        now = datetime.utcnow()
        one_minute_ago = now - timedelta(minutes=1)

        agent_filter = []
        if owner_id:
            agent_query = select(Agent.id).where(Agent.owner_id == owner_id)
            agent_result = await self.db.execute(agent_query)
            agent_ids = [r[0] for r in agent_result.fetchall()]
            if agent_ids:
                agent_filter.append(Call.agent_id.in_(agent_ids))

        # Active calls
        active_query = select(func.count()).select_from(Call).where(
            and_(
                Call.status.in_([
                    CallStatus.QUEUED,
                    CallStatus.RINGING,
                    CallStatus.IN_PROGRESS,
                ]),
                *agent_filter,
            )
        )
        active_result = await self.db.execute(active_query)
        active_calls = active_result.scalar() or 0

        # Calls per minute
        cpm_query = select(func.count()).select_from(Call).where(
            and_(
                Call.created_at >= one_minute_ago,
                *agent_filter,
            )
        )
        cpm_result = await self.db.execute(cpm_query)
        calls_per_minute = float(cpm_result.scalar() or 0)

        # Queued calls
        queue_query = select(func.count()).select_from(Call).where(
            and_(
                Call.status == CallStatus.QUEUED,
                *agent_filter,
            )
        )
        queue_result = await self.db.execute(queue_query)
        queue_length = queue_result.scalar() or 0

        # Active agents
        if owner_id:
            agents_query = select(func.count()).select_from(Agent).where(
                and_(Agent.owner_id == owner_id, Agent.status == AgentStatus.ACTIVE)
            )
        else:
            agents_query = select(func.count()).select_from(Agent).where(
                Agent.status == AgentStatus.ACTIVE
            )
        agents_result = await self.db.execute(agents_query)
        agents_online = agents_result.scalar() or 0

        return RealTimeMetrics(
            timestamp=now,
            active_calls=active_calls,
            calls_per_minute=calls_per_minute,
            avg_wait_time_seconds=0,  # TODO: Calculate from queue times
            agents_online=agents_online,
            queue_length=queue_length,
        )

    async def get_usage_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        owner_id: Optional[UUID] = None,
    ) -> UsageMetrics:
        """Get usage metrics for billing."""
        agent_filter = []
        if owner_id:
            agent_query = select(Agent.id).where(Agent.owner_id == owner_id)
            agent_result = await self.db.execute(agent_query)
            agent_ids = [r[0] for r in agent_result.fetchall()]
            if agent_ids:
                agent_filter.append(Call.agent_id.in_(agent_ids))

        # Total calls and minutes
        calls_query = (
            select(
                func.count().label("total_calls"),
                func.sum(Call.duration_seconds).label("total_seconds"),
                func.count().filter(Call.direction == CallDirection.INBOUND).label("inbound"),
                func.count().filter(Call.direction == CallDirection.OUTBOUND).label("outbound"),
            )
            .where(
                and_(
                    Call.created_at >= start_date,
                    Call.created_at < end_date,
                    *agent_filter,
                )
            )
        )

        result = await self.db.execute(calls_query)
        row = result.fetchone()

        return UsageMetrics(
            period_start=start_date,
            period_end=end_date,
            total_calls=row.total_calls or 0,
            total_minutes=(row.total_seconds or 0) // 60,
            inbound_calls=row.inbound or 0,
            outbound_calls=row.outbound or 0,
            api_calls=0,  # TODO: Track API usage
            llm_tokens_used=0,  # TODO: Track LLM usage
            tts_characters=0,  # TODO: Track TTS usage
            asr_minutes=(row.total_seconds or 0) // 60,  # Approximate
            recordings_size_mb=0,  # TODO: Track storage
            knowledge_base_size_mb=0,
        )
