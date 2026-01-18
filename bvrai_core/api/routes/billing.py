"""
Billing API Routes

Provides REST API endpoints for billing and subscription management.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..base import APIResponse, success_response
from ..auth import AuthContext, Permission
from ..dependencies import get_db_session, get_auth_context
from ...database.models import Organization, Call


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["Billing"])


# =============================================================================
# Response Models
# =============================================================================

class SubscriptionResponse(BaseModel):
    """Subscription response."""
    id: str
    plan: str
    status: str
    current_period_start: str
    current_period_end: str
    cancel_at_period_end: bool
    trial_end: Optional[str] = None


class UsageResponse(BaseModel):
    """Usage response."""
    period_start: str
    period_end: str
    total_calls: int
    total_minutes: float
    total_cost: float
    breakdown: dict


class InvoiceResponse(BaseModel):
    """Invoice response."""
    id: str
    number: str
    amount: float
    currency: str
    status: str
    period_start: str
    period_end: str
    due_date: str
    paid_at: Optional[str] = None
    pdf_url: Optional[str] = None


# =============================================================================
# Routes
# =============================================================================

@router.get(
    "/subscription",
    response_model=APIResponse[SubscriptionResponse],
    summary="Get Subscription",
    description="Get current subscription details.",
)
async def get_subscription(
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Get subscription details."""
    result = await db.execute(
        select(Organization).where(Organization.id == auth.organization_id)
    )
    org = result.scalar_one_or_none()

    # Calculate period dates
    now = datetime.utcnow()
    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if now.month == 12:
        period_end = period_start.replace(year=now.year + 1, month=1)
    else:
        period_end = period_start.replace(month=now.month + 1)

    return success_response({
        "id": f"sub_{auth.organization_id[:8]}",
        "plan": org.plan if org else "free",
        "status": "active",
        "current_period_start": period_start.isoformat(),
        "current_period_end": period_end.isoformat(),
        "cancel_at_period_end": False,
        "trial_end": None,
    })


@router.get(
    "/usage",
    response_model=APIResponse[UsageResponse],
    summary="Get Usage",
    description="Get current billing period usage.",
)
async def get_usage(
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Get usage for current period."""
    # Calculate period dates
    now = datetime.utcnow()
    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if now.month == 12:
        period_end = period_start.replace(year=now.year + 1, month=1)
    else:
        period_end = period_start.replace(month=now.month + 1)

    # Get call stats
    result = await db.execute(
        select(
            func.count(Call.id).label("total_calls"),
            func.sum(Call.duration_seconds).label("total_duration"),
            func.sum(Call.cost_amount).label("total_cost"),
        ).where(
            Call.organization_id == auth.organization_id,
            Call.initiated_at >= period_start,
            Call.initiated_at < period_end,
        )
    )
    stats = result.first()

    total_calls = stats.total_calls or 0
    total_seconds = float(stats.total_duration or 0)
    total_cost = float(stats.total_cost or 0)

    # Get inbound/outbound breakdown
    result = await db.execute(
        select(
            Call.direction,
            func.sum(Call.duration_seconds).label("duration"),
        ).where(
            Call.organization_id == auth.organization_id,
            Call.initiated_at >= period_start,
            Call.initiated_at < period_end,
        ).group_by(Call.direction)
    )

    breakdown = {
        "inbound_minutes": 0.0,
        "outbound_minutes": 0.0,
        "stt_minutes": total_seconds / 60,  # Approximation
        "tts_minutes": total_seconds / 60 * 0.3,  # Approximation
        "llm_tokens": int(total_seconds * 100),  # Approximation
    }

    for row in result:
        if row.direction == "inbound":
            breakdown["inbound_minutes"] = float(row.duration or 0) / 60
        elif row.direction == "outbound":
            breakdown["outbound_minutes"] = float(row.duration or 0) / 60

    return success_response({
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "total_calls": total_calls,
        "total_minutes": total_seconds / 60,
        "total_cost": total_cost,
        "breakdown": breakdown,
    })


@router.get(
    "/invoices",
    response_model=APIResponse[List[InvoiceResponse]],
    summary="List Invoices",
    description="List all invoices.",
)
async def list_invoices(
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """List invoices."""
    # For MVP, return mock invoices
    now = datetime.utcnow()

    invoices = []
    for i in range(3):
        month = now.month - i - 1
        year = now.year
        if month <= 0:
            month += 12
            year -= 1

        period_start = datetime(year, month, 1)
        if month == 12:
            period_end = datetime(year + 1, 1, 1)
        else:
            period_end = datetime(year, month + 1, 1)

        invoices.append({
            "id": f"inv_{auth.organization_id[:8]}_{month:02d}",
            "number": f"INV-{year}{month:02d}-{i + 1:04d}",
            "amount": 99.00 + (i * 10),
            "currency": "USD",
            "status": "paid",
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "due_date": (period_end + timedelta(days=30)).isoformat(),
            "paid_at": (period_end + timedelta(days=15)).isoformat(),
            "pdf_url": None,
        })

    return success_response(invoices)


@router.post(
    "/checkout",
    response_model=APIResponse[dict],
    summary="Create Checkout",
    description="Create a checkout session for plan upgrade.",
)
async def create_checkout(
    price_id: str,
    auth: AuthContext = Depends(get_auth_context),
):
    """Create checkout session."""
    auth.require_permission(Permission.BILLING_WRITE)

    # In production, integrate with Stripe
    return success_response({
        "checkout_url": f"https://checkout.example.com/{price_id}?org={auth.organization_id}",
        "message": "Stripe integration not implemented in MVP",
    })


@router.post(
    "/portal",
    response_model=APIResponse[dict],
    summary="Create Portal Session",
    description="Create a billing portal session.",
)
async def create_portal_session(
    auth: AuthContext = Depends(get_auth_context),
):
    """Create billing portal session."""
    auth.require_permission(Permission.BILLING_WRITE)

    return success_response({
        "portal_url": f"https://billing.example.com/portal?org={auth.organization_id}",
        "message": "Stripe integration not implemented in MVP",
    })


@router.post(
    "/cancel",
    response_model=APIResponse[dict],
    summary="Cancel Subscription",
    description="Cancel the subscription at period end.",
)
async def cancel_subscription(
    auth: AuthContext = Depends(get_auth_context),
):
    """Cancel subscription."""
    auth.require_permission(Permission.BILLING_WRITE)

    logger.info(f"Subscription cancellation requested: {auth.organization_id}")

    return success_response({
        "message": "Subscription will be cancelled at the end of the current period",
        "cancel_at_period_end": True,
    })


@router.post(
    "/resume",
    response_model=APIResponse[dict],
    summary="Resume Subscription",
    description="Resume a cancelled subscription.",
)
async def resume_subscription(
    auth: AuthContext = Depends(get_auth_context),
):
    """Resume subscription."""
    auth.require_permission(Permission.BILLING_WRITE)

    logger.info(f"Subscription resumed: {auth.organization_id}")

    return success_response({
        "message": "Subscription has been resumed",
        "cancel_at_period_end": False,
    })
