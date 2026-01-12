"""Analytics API routes."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.analytics.schemas import (
    DashboardOverview,
    DailyStats,
    HourlyDistribution,
    AgentPerformance,
    CallOutcome,
    AnalyticsReport,
    UsageMetrics,
    RealTimeMetrics,
)
from app.analytics.service import AnalyticsService
from app.auth.dependencies import get_current_user_id

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview", response_model=DashboardOverview)
async def get_dashboard_overview(
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get main dashboard overview metrics."""
    service = AnalyticsService(db)
    return await service.get_dashboard_overview(owner_id=user_id)


@router.get("/daily", response_model=list[DailyStats])
async def get_daily_stats(
    start_date: datetime = Query(..., description="Start date"),
    end_date: datetime = Query(..., description="End date"),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get daily statistics for a date range."""
    service = AnalyticsService(db)
    return await service.get_daily_stats(
        start_date.date(),
        end_date.date(),
        owner_id=user_id,
    )


@router.get("/hourly", response_model=list[HourlyDistribution])
async def get_hourly_distribution(
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get hourly call distribution."""
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=7)
    if not end_date:
        end_date = datetime.utcnow()

    service = AnalyticsService(db)
    return await service.get_hourly_distribution(
        start_date,
        end_date,
        owner_id=user_id,
    )


@router.get("/agents", response_model=list[AgentPerformance])
async def get_agent_performance(
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get performance metrics per agent."""
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()

    service = AnalyticsService(db)
    return await service.get_agent_performance(
        start_date,
        end_date,
        owner_id=user_id,
    )


@router.get("/outcomes", response_model=list[CallOutcome])
async def get_call_outcomes(
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get call outcome distribution."""
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()

    service = AnalyticsService(db)
    return await service.get_call_outcomes(
        start_date,
        end_date,
        owner_id=user_id,
    )


@router.get("/report", response_model=AnalyticsReport)
async def get_full_report(
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Generate a full analytics report."""
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()

    service = AnalyticsService(db)
    return await service.get_full_report(
        start_date,
        end_date,
        owner_id=user_id,
    )


@router.get("/realtime", response_model=RealTimeMetrics)
async def get_realtime_metrics(
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get real-time metrics for live dashboard."""
    service = AnalyticsService(db)
    return await service.get_real_time_metrics(owner_id=user_id)


@router.get("/usage", response_model=UsageMetrics)
async def get_usage_metrics(
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get usage metrics for billing."""
    if not start_date:
        # Default to current billing period (month)
        now = datetime.utcnow()
        start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if not end_date:
        end_date = datetime.utcnow()

    service = AnalyticsService(db)
    return await service.get_usage_metrics(
        start_date,
        end_date,
        owner_id=user_id,
    )


@router.get("/export")
async def export_report(
    format: str = Query("json", description="Export format (json, csv)"),
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Export analytics report in various formats."""
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()

    service = AnalyticsService(db)
    report = await service.get_full_report(start_date, end_date, owner_id=user_id)

    if format == "csv":
        # TODO: Implement CSV export
        return {"message": "CSV export not yet implemented"}

    return report.model_dump()
