"""
Analytics API Routes

This module provides REST API endpoints for analytics and reporting.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_, case
from sqlalchemy.ext.asyncio import AsyncSession

from ..base import (
    APIResponse,
    success_response,
)
from ..auth import AuthContext, Permission
from ..dependencies import get_db_session
from ...database.models import Call, Agent


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


class TimeGranularity(str):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class UsageSummary(BaseModel):
    """Usage summary for a time period."""

    # Time period
    start_date: datetime
    end_date: datetime

    # Call metrics
    total_calls: int = 0
    inbound_calls: int = 0
    outbound_calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0

    # Duration metrics
    total_minutes: float = 0.0
    average_call_duration_seconds: float = 0.0

    # Success metrics
    answer_rate: float = 0.0
    completion_rate: float = 0.0

    # Cost
    total_cost_cents: int = 0

    # Agent metrics
    unique_agents_used: int = 0


class TimeSeriesPoint(BaseModel):
    """A point in a time series."""

    timestamp: datetime
    value: float


class TimeSeriesData(BaseModel):
    """Time series data."""

    metric: str
    granularity: str
    data: List[TimeSeriesPoint]
    total: float
    average: float
    min_value: float
    max_value: float


class AgentPerformance(BaseModel):
    """Agent performance metrics."""

    agent_id: str
    agent_name: str

    # Call metrics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0

    # Duration
    total_minutes: float = 0.0
    average_call_duration_seconds: float = 0.0

    # Success rates
    answer_rate: float = 0.0
    completion_rate: float = 0.0
    transfer_rate: float = 0.0

    # Sentiment (if analyzed)
    average_sentiment_score: Optional[float] = None
    positive_sentiment_rate: Optional[float] = None


class CallOutcomeBreakdown(BaseModel):
    """Breakdown of call outcomes."""

    completed: int = 0
    voicemail: int = 0
    no_answer: int = 0
    busy: int = 0
    failed: int = 0
    transferred: int = 0


class TopIntentsReport(BaseModel):
    """Report of top detected intents."""

    intent: str
    count: int
    percentage: float
    avg_handling_time_seconds: float


class CallDurationDistribution(BaseModel):
    """Distribution of call durations."""

    bucket: str  # e.g., "0-30s", "30s-1m", "1-2m", etc.
    count: int
    percentage: float


@router.get(
    "/usage",
    response_model=APIResponse[UsageSummary],
    summary="Get Usage Summary",
    description="Get usage summary for a time period.",
)
async def get_usage_summary(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    agent_id: Optional[str] = Query(None),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Get usage summary."""
    auth.require_permission(Permission.ANALYTICS_READ)

    # Default to last 30 days
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Build query conditions
    conditions = [
        Call.organization_id == auth.organization_id,
        Call.initiated_at >= start_date,
        Call.initiated_at <= end_date,
        Call.is_deleted == False,
    ]
    if agent_id:
        conditions.append(Call.agent_id == agent_id)

    # Query call stats
    result = await db.execute(
        select(
            func.count(Call.id).label("total_calls"),
            func.count(case((Call.direction == "inbound", 1))).label("inbound_calls"),
            func.count(case((Call.direction == "outbound", 1))).label("outbound_calls"),
            func.count(case((Call.status == "completed", 1))).label("completed_calls"),
            func.count(case((Call.status == "failed", 1))).label("failed_calls"),
            func.coalesce(func.sum(Call.duration_seconds), 0).label("total_seconds"),
            func.coalesce(func.avg(Call.duration_seconds), 0).label("avg_duration"),
            func.coalesce(func.sum(Call.cost_amount), 0).label("total_cost"),
            func.count(func.distinct(Call.agent_id)).label("unique_agents"),
        ).where(and_(*conditions))
    )
    row = result.one()

    total_calls = row.total_calls or 0
    completed_calls = row.completed_calls or 0
    inbound_calls = row.inbound_calls or 0

    summary = UsageSummary(
        start_date=start_date,
        end_date=end_date,
        total_calls=total_calls,
        inbound_calls=inbound_calls,
        outbound_calls=row.outbound_calls or 0,
        completed_calls=completed_calls,
        failed_calls=row.failed_calls or 0,
        total_minutes=float(row.total_seconds or 0) / 60.0,
        average_call_duration_seconds=float(row.avg_duration or 0),
        answer_rate=(inbound_calls / total_calls * 100) if total_calls > 0 else 0.0,
        completion_rate=(completed_calls / total_calls * 100) if total_calls > 0 else 0.0,
        total_cost_cents=int(float(row.total_cost or 0) * 100),
        unique_agents_used=row.unique_agents or 0,
    )

    return success_response(summary.model_dump())


@router.get(
    "/calls/timeseries",
    response_model=APIResponse[TimeSeriesData],
    summary="Get Call Time Series",
    description="Get call metrics over time.",
)
async def get_call_timeseries(
    metric: str = Query("total_calls", description="Metric to chart"),
    granularity: str = Query(TimeGranularity.DAY),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    agent_id: Optional[str] = Query(None),
    auth: AuthContext = Depends(),
):
    """Get call metrics time series."""
    auth.require_permission(Permission.ANALYTICS_READ)

    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # In production, this would query time-series database
    data = TimeSeriesData(
        metric=metric,
        granularity=granularity,
        data=[],
        total=0,
        average=0,
        min_value=0,
        max_value=0,
    )

    return success_response(data.dict())


@router.get(
    "/agents/performance",
    response_model=APIResponse[List[AgentPerformance]],
    summary="Get Agent Performance",
    description="Get performance metrics for all agents.",
)
async def get_agent_performance(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    sort_by: str = Query("total_calls", description="Sort by metric"),
    limit: int = Query(10, ge=1, le=50),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Get agent performance rankings."""
    auth.require_permission(Permission.ANALYTICS_READ)

    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Query agent performance
    result = await db.execute(
        select(
            Agent.id,
            Agent.name,
            func.count(Call.id).label("total_calls"),
            func.count(case((Call.status == "completed", 1))).label("successful_calls"),
            func.count(case((Call.status == "failed", 1))).label("failed_calls"),
            func.coalesce(func.sum(Call.duration_seconds), 0).label("total_seconds"),
            func.coalesce(func.avg(Call.duration_seconds), 0).label("avg_duration"),
            func.count(case((Call.transferred == True, 1))).label("transferred_calls"),
        )
        .join(Call, Call.agent_id == Agent.id, isouter=True)
        .where(
            and_(
                Agent.organization_id == auth.organization_id,
                Agent.is_deleted == False,
                Call.initiated_at >= start_date,
                Call.initiated_at <= end_date,
            )
        )
        .group_by(Agent.id, Agent.name)
        .order_by(func.count(Call.id).desc())
        .limit(limit)
    )
    rows = result.all()

    performances = []
    for row in rows:
        total = row.total_calls or 0
        successful = row.successful_calls or 0
        transferred = row.transferred_calls or 0

        performances.append(AgentPerformance(
            agent_id=row.id,
            agent_name=row.name,
            total_calls=total,
            successful_calls=successful,
            failed_calls=row.failed_calls or 0,
            total_minutes=float(row.total_seconds or 0) / 60.0,
            average_call_duration_seconds=float(row.avg_duration or 0),
            answer_rate=0.0,  # Would need more data to calculate
            completion_rate=(successful / total * 100) if total > 0 else 0.0,
            transfer_rate=(transferred / total * 100) if total > 0 else 0.0,
        ))

    return success_response([p.model_dump() for p in performances])


@router.get(
    "/calls/outcomes",
    response_model=APIResponse[CallOutcomeBreakdown],
    summary="Get Call Outcomes",
    description="Get breakdown of call outcomes.",
)
async def get_call_outcomes(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    agent_id: Optional[str] = Query(None),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Get call outcome breakdown."""
    auth.require_permission(Permission.ANALYTICS_READ)

    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    conditions = [
        Call.organization_id == auth.organization_id,
        Call.initiated_at >= start_date,
        Call.initiated_at <= end_date,
        Call.is_deleted == False,
    ]
    if agent_id:
        conditions.append(Call.agent_id == agent_id)

    result = await db.execute(
        select(
            func.count(case((Call.status == "completed", 1))).label("completed"),
            func.count(case((Call.end_reason == "voicemail", 1))).label("voicemail"),
            func.count(case((Call.status == "no_answer", 1))).label("no_answer"),
            func.count(case((Call.status == "busy", 1))).label("busy"),
            func.count(case((Call.status == "failed", 1))).label("failed"),
            func.count(case((Call.transferred == True, 1))).label("transferred"),
        ).where(and_(*conditions))
    )
    row = result.one()

    breakdown = CallOutcomeBreakdown(
        completed=row.completed or 0,
        voicemail=row.voicemail or 0,
        no_answer=row.no_answer or 0,
        busy=row.busy or 0,
        failed=row.failed or 0,
        transferred=row.transferred or 0,
    )
    return success_response(breakdown.model_dump())


@router.get(
    "/calls/durations",
    response_model=APIResponse[List[CallDurationDistribution]],
    summary="Get Duration Distribution",
    description="Get distribution of call durations.",
)
async def get_duration_distribution(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    agent_id: Optional[str] = Query(None),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Get call duration distribution."""
    auth.require_permission(Permission.ANALYTICS_READ)

    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    conditions = [
        Call.organization_id == auth.organization_id,
        Call.initiated_at >= start_date,
        Call.initiated_at <= end_date,
        Call.is_deleted == False,
        Call.status == "completed",
    ]
    if agent_id:
        conditions.append(Call.agent_id == agent_id)

    # Query duration buckets
    result = await db.execute(
        select(
            func.count(case((Call.duration_seconds < 30, 1))).label("bucket_0_30"),
            func.count(case((and_(Call.duration_seconds >= 30, Call.duration_seconds < 60), 1))).label("bucket_30_60"),
            func.count(case((and_(Call.duration_seconds >= 60, Call.duration_seconds < 120), 1))).label("bucket_1_2"),
            func.count(case((and_(Call.duration_seconds >= 120, Call.duration_seconds < 300), 1))).label("bucket_2_5"),
            func.count(case((and_(Call.duration_seconds >= 300, Call.duration_seconds < 600), 1))).label("bucket_5_10"),
            func.count(case((Call.duration_seconds >= 600, 1))).label("bucket_10plus"),
            func.count(Call.id).label("total"),
        ).where(and_(*conditions))
    )
    row = result.one()

    total = row.total or 1  # Avoid division by zero

    buckets = [
        CallDurationDistribution(
            bucket="0-30s",
            count=row.bucket_0_30 or 0,
            percentage=round((row.bucket_0_30 or 0) / total * 100, 2),
        ),
        CallDurationDistribution(
            bucket="30s-1m",
            count=row.bucket_30_60 or 0,
            percentage=round((row.bucket_30_60 or 0) / total * 100, 2),
        ),
        CallDurationDistribution(
            bucket="1-2m",
            count=row.bucket_1_2 or 0,
            percentage=round((row.bucket_1_2 or 0) / total * 100, 2),
        ),
        CallDurationDistribution(
            bucket="2-5m",
            count=row.bucket_2_5 or 0,
            percentage=round((row.bucket_2_5 or 0) / total * 100, 2),
        ),
        CallDurationDistribution(
            bucket="5-10m",
            count=row.bucket_5_10 or 0,
            percentage=round((row.bucket_5_10 or 0) / total * 100, 2),
        ),
        CallDurationDistribution(
            bucket="10m+",
            count=row.bucket_10plus or 0,
            percentage=round((row.bucket_10plus or 0) / total * 100, 2),
        ),
    ]

    return success_response([b.model_dump() for b in buckets])


@router.get(
    "/intents/top",
    response_model=APIResponse[List[TopIntentsReport]],
    summary="Get Top Intents",
    description="Get top detected customer intents.",
)
async def get_top_intents(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    agent_id: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    auth: AuthContext = Depends(),
):
    """Get top detected intents."""
    auth.require_permission(Permission.ANALYTICS_READ)

    return success_response([])


@router.get(
    "/cost",
    response_model=APIResponse[Dict[str, Any]],
    summary="Get Cost Breakdown",
    description="Get cost breakdown by category.",
)
async def get_cost_breakdown(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    group_by: str = Query("day", description="Group by: day, week, month, agent"),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Get cost breakdown."""
    auth.require_permission(Permission.BILLING_READ)

    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    conditions = [
        Call.organization_id == auth.organization_id,
        Call.initiated_at >= start_date,
        Call.initiated_at <= end_date,
        Call.is_deleted == False,
    ]

    result = await db.execute(
        select(
            func.coalesce(func.sum(Call.cost_amount), 0).label("total_cost"),
        ).where(and_(*conditions))
    )
    row = result.one()
    total_cost_cents = int(float(row.total_cost or 0) * 100)

    # Estimate breakdown (in real system this would be tracked per-service)
    telephony_pct = 0.40
    transcription_pct = 0.25
    llm_pct = 0.25
    tts_pct = 0.10

    cost_data = {
        "total_cost_cents": total_cost_cents,
        "telephony_cost_cents": int(total_cost_cents * telephony_pct),
        "transcription_cost_cents": int(total_cost_cents * transcription_pct),
        "llm_cost_cents": int(total_cost_cents * llm_pct),
        "tts_cost_cents": int(total_cost_cents * tts_pct),
        "breakdown": [],
    }

    return success_response(cost_data)


@router.get(
    "/dashboard",
    response_model=APIResponse[Dict[str, Any]],
    summary="Get Dashboard Data",
    description="Get all data needed for main analytics dashboard.",
)
async def get_dashboard_data(
    period: str = Query("30d", description="Period: 7d, 30d, 90d"),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Get comprehensive dashboard data."""
    auth.require_permission(Permission.ANALYTICS_READ)

    # Parse period
    days = {"7d": 7, "30d": 30, "90d": 90}.get(period, 30)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    prev_start_date = start_date - timedelta(days=days)

    # Current period stats
    conditions = [
        Call.organization_id == auth.organization_id,
        Call.initiated_at >= start_date,
        Call.initiated_at <= end_date,
        Call.is_deleted == False,
    ]

    result = await db.execute(
        select(
            func.count(Call.id).label("total_calls"),
            func.count(case((Call.status == "completed", 1))).label("completed_calls"),
            func.coalesce(func.sum(Call.duration_seconds), 0).label("total_seconds"),
            func.coalesce(func.avg(Call.duration_seconds), 0).label("avg_duration"),
            func.count(func.distinct(Call.agent_id)).label("active_agents"),
        ).where(and_(*conditions))
    )
    current = result.one()

    # Previous period stats for trends
    prev_conditions = [
        Call.organization_id == auth.organization_id,
        Call.initiated_at >= prev_start_date,
        Call.initiated_at < start_date,
        Call.is_deleted == False,
    ]
    result = await db.execute(
        select(
            func.count(Call.id).label("total_calls"),
            func.count(case((Call.status == "completed", 1))).label("completed_calls"),
            func.coalesce(func.sum(Call.duration_seconds), 0).label("total_seconds"),
        ).where(and_(*prev_conditions))
    )
    previous = result.one()

    # Calculate trends
    current_calls = current.total_calls or 0
    prev_calls = previous.total_calls or 0
    calls_change = ((current_calls - prev_calls) / prev_calls * 100) if prev_calls > 0 else 0

    current_minutes = float(current.total_seconds or 0) / 60.0
    prev_minutes = float(previous.total_seconds or 0) / 60.0
    duration_change = ((current_minutes - prev_minutes) / prev_minutes * 100) if prev_minutes > 0 else 0

    current_success = (current.completed_calls or 0) / current_calls * 100 if current_calls > 0 else 0
    prev_success = (previous.completed_calls or 0) / prev_calls * 100 if prev_calls > 0 else 0
    success_change = current_success - prev_success

    # Recent calls
    result = await db.execute(
        select(Call)
        .where(
            and_(
                Call.organization_id == auth.organization_id,
                Call.is_deleted == False,
            )
        )
        .order_by(Call.initiated_at.desc())
        .limit(10)
    )
    recent_calls = []
    for call in result.scalars().all():
        recent_calls.append({
            "id": call.id,
            "direction": call.direction,
            "status": call.status,
            "from_number": call.from_number,
            "to_number": call.to_number,
            "duration_seconds": call.duration_seconds,
            "initiated_at": call.initiated_at.isoformat() if call.initiated_at else None,
        })

    # Top agents
    result = await db.execute(
        select(
            Agent.id,
            Agent.name,
            func.count(Call.id).label("call_count"),
        )
        .join(Call, Call.agent_id == Agent.id, isouter=True)
        .where(
            and_(
                Agent.organization_id == auth.organization_id,
                Agent.is_deleted == False,
                Call.initiated_at >= start_date,
            )
        )
        .group_by(Agent.id, Agent.name)
        .order_by(func.count(Call.id).desc())
        .limit(5)
    )
    top_agents = [
        {"id": row.id, "name": row.name, "calls": row.call_count or 0}
        for row in result.all()
    ]

    dashboard = {
        "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "summary": {
            "total_calls": current_calls,
            "total_minutes": round(current_minutes, 2),
            "success_rate": round(current_success, 2),
            "average_duration": round(float(current.avg_duration or 0), 2),
            "active_agents": current.active_agents or 0,
        },
        "trends": {
            "calls_change_percent": round(calls_change, 2),
            "duration_change_percent": round(duration_change, 2),
            "success_rate_change": round(success_change, 2),
        },
        "recent_calls": recent_calls,
        "top_agents": top_agents,
        "call_volume_chart": [],  # Would need time-series aggregation
    }

    return success_response(dashboard)


@router.get(
    "/export",
    response_model=APIResponse[Dict[str, Any]],
    summary="Export Analytics",
    description="Export analytics data in various formats.",
)
async def export_analytics(
    format: str = Query("csv", description="Export format: csv, json, xlsx"),
    report_type: str = Query("calls", description="Report type: calls, agents, usage"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    auth: AuthContext = Depends(),
):
    """Export analytics data."""
    auth.require_permission(Permission.ANALYTICS_READ)

    # In production, this would generate a download URL
    return success_response({
        "status": "processing",
        "download_url": None,
        "expires_at": None,
        "message": "Report generation would be queued here",
    })
