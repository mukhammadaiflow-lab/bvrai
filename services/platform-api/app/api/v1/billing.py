"""
Billing API Routes

Handles:
- Subscription management
- Usage tracking
- Invoice generation
- Payment methods
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, date
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_db_session,
    get_current_user,
    get_current_tenant,
    UserContext,
    TenantContext,
    require_permissions,
    verify_webhook_signature,
)

router = APIRouter(prefix="/billing")


# ============================================================================
# Schemas
# ============================================================================

class PlanTier(str, Enum):
    """Plan tiers."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class BillingInterval(str, Enum):
    """Billing interval."""
    MONTHLY = "monthly"
    YEARLY = "yearly"


class SubscriptionStatus(str, Enum):
    """Subscription status."""
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    TRIALING = "trialing"
    PAUSED = "paused"


class PaymentMethodType(str, Enum):
    """Payment method type."""
    CARD = "card"
    BANK_ACCOUNT = "bank_account"
    ACH = "ach"


class PlanResponse(BaseModel):
    """Plan response."""
    id: str
    name: str
    tier: PlanTier
    description: str
    price_monthly: float
    price_yearly: float
    features: List[str]
    limits: Dict[str, int]
    is_popular: bool = False


class SubscriptionResponse(BaseModel):
    """Subscription response."""
    id: str
    plan_id: str
    plan_name: str
    tier: PlanTier
    status: SubscriptionStatus
    interval: BillingInterval
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool
    trial_end: Optional[datetime]
    created_at: datetime


class UsageResponse(BaseModel):
    """Usage response."""
    period_start: datetime
    period_end: datetime
    calls_count: int
    calls_limit: int
    minutes_used: float
    minutes_limit: float
    agents_count: int
    agents_limit: int
    phone_numbers_count: int
    phone_numbers_limit: int
    storage_used_gb: float
    storage_limit_gb: float
    api_requests: int
    api_requests_limit: int


class InvoiceResponse(BaseModel):
    """Invoice response."""
    id: str
    number: str
    status: str
    amount_due: float
    amount_paid: float
    currency: str
    period_start: datetime
    period_end: datetime
    due_date: datetime
    paid_at: Optional[datetime]
    invoice_url: Optional[str]
    pdf_url: Optional[str]
    line_items: List[Dict[str, Any]]


class InvoiceListResponse(BaseModel):
    """Invoice list response."""
    invoices: List[InvoiceResponse]
    total: int


class PaymentMethodResponse(BaseModel):
    """Payment method response."""
    id: str
    type: PaymentMethodType
    last_four: str
    brand: Optional[str]
    exp_month: Optional[int]
    exp_year: Optional[int]
    is_default: bool
    created_at: datetime


class CreateSubscriptionRequest(BaseModel):
    """Create subscription request."""
    plan_id: str
    interval: BillingInterval = BillingInterval.MONTHLY
    payment_method_id: Optional[str] = None
    trial_days: Optional[int] = None


class UpdateSubscriptionRequest(BaseModel):
    """Update subscription request."""
    plan_id: Optional[str] = None
    interval: Optional[BillingInterval] = None
    cancel_at_period_end: Optional[bool] = None


class CreatePaymentMethodRequest(BaseModel):
    """Create payment method request."""
    payment_method_token: str
    set_as_default: bool = True


class CreateCheckoutRequest(BaseModel):
    """Create checkout session request."""
    plan_id: str
    interval: BillingInterval = BillingInterval.MONTHLY
    success_url: str
    cancel_url: str


class CheckoutResponse(BaseModel):
    """Checkout session response."""
    checkout_url: str
    session_id: str


# ============================================================================
# Plans
# ============================================================================

@router.get("/plans", response_model=List[PlanResponse])
async def list_plans(
    user: UserContext = Depends(get_current_user),
):
    """List available plans."""
    plans = [
        PlanResponse(
            id="free",
            name="Free",
            tier=PlanTier.FREE,
            description="Get started with basic features",
            price_monthly=0,
            price_yearly=0,
            features=[
                "1 AI Agent",
                "50 minutes/month",
                "Basic analytics",
                "Community support",
            ],
            limits={
                "agents": 1,
                "minutes": 50,
                "phone_numbers": 0,
                "api_requests": 1000,
            },
        ),
        PlanResponse(
            id="starter",
            name="Starter",
            tier=PlanTier.STARTER,
            description="Perfect for small teams",
            price_monthly=49,
            price_yearly=470,
            features=[
                "5 AI Agents",
                "500 minutes/month",
                "1 Phone number",
                "Advanced analytics",
                "Email support",
                "Custom prompts",
            ],
            limits={
                "agents": 5,
                "minutes": 500,
                "phone_numbers": 1,
                "api_requests": 10000,
            },
            is_popular=True,
        ),
        PlanResponse(
            id="professional",
            name="Professional",
            tier=PlanTier.PROFESSIONAL,
            description="For growing businesses",
            price_monthly=199,
            price_yearly=1990,
            features=[
                "25 AI Agents",
                "2,500 minutes/month",
                "5 Phone numbers",
                "Full analytics suite",
                "Priority support",
                "Workflows",
                "Integrations",
                "Voice cloning",
            ],
            limits={
                "agents": 25,
                "minutes": 2500,
                "phone_numbers": 5,
                "api_requests": 100000,
            },
        ),
        PlanResponse(
            id="enterprise",
            name="Enterprise",
            tier=PlanTier.ENTERPRISE,
            description="Custom solutions for large organizations",
            price_monthly=0,
            price_yearly=0,
            features=[
                "Unlimited AI Agents",
                "Unlimited minutes",
                "Unlimited phone numbers",
                "Custom integrations",
                "Dedicated support",
                "SLA guarantee",
                "On-premise deployment",
                "Custom training",
            ],
            limits={
                "agents": -1,
                "minutes": -1,
                "phone_numbers": -1,
                "api_requests": -1,
            },
        ),
    ]

    return plans


@router.get("/plans/{plan_id}", response_model=PlanResponse)
async def get_plan(
    plan_id: str,
    user: UserContext = Depends(get_current_user),
):
    """Get plan details."""
    plans = await list_plans(user)

    for plan in plans:
        if plan.id == plan_id:
            return plan

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Plan not found",
    )


# ============================================================================
# Subscription
# ============================================================================

@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get current subscription."""
    from app.billing import BillingService

    service = BillingService(db)
    subscription = await service.get_subscription(tenant.tenant_id)

    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription",
        )

    return SubscriptionResponse(
        id=subscription.id,
        plan_id=subscription.plan_id,
        plan_name=subscription.plan_name,
        tier=PlanTier(subscription.tier),
        status=SubscriptionStatus(subscription.status),
        interval=BillingInterval(subscription.interval),
        current_period_start=subscription.current_period_start,
        current_period_end=subscription.current_period_end,
        cancel_at_period_end=subscription.cancel_at_period_end,
        trial_end=subscription.trial_end,
        created_at=subscription.created_at,
    )


@router.post("/subscription", response_model=SubscriptionResponse)
async def create_subscription(
    data: CreateSubscriptionRequest,
    user: UserContext = Depends(require_permissions("billing:manage")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new subscription."""
    from app.billing import BillingService

    service = BillingService(db)

    try:
        subscription = await service.create_subscription(
            tenant_id=tenant.tenant_id,
            plan_id=data.plan_id,
            interval=data.interval.value,
            payment_method_id=data.payment_method_id,
            trial_days=data.trial_days,
        )

        return SubscriptionResponse(
            id=subscription.id,
            plan_id=subscription.plan_id,
            plan_name=subscription.plan_name,
            tier=PlanTier(subscription.tier),
            status=SubscriptionStatus(subscription.status),
            interval=BillingInterval(subscription.interval),
            current_period_start=subscription.current_period_start,
            current_period_end=subscription.current_period_end,
            cancel_at_period_end=subscription.cancel_at_period_end,
            trial_end=subscription.trial_end,
            created_at=subscription.created_at,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.patch("/subscription", response_model=SubscriptionResponse)
async def update_subscription(
    data: UpdateSubscriptionRequest,
    user: UserContext = Depends(require_permissions("billing:manage")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Update subscription (change plan or cancel)."""
    from app.billing import BillingService

    service = BillingService(db)

    try:
        subscription = await service.update_subscription(
            tenant_id=tenant.tenant_id,
            plan_id=data.plan_id,
            interval=data.interval.value if data.interval else None,
            cancel_at_period_end=data.cancel_at_period_end,
        )

        return SubscriptionResponse(
            id=subscription.id,
            plan_id=subscription.plan_id,
            plan_name=subscription.plan_name,
            tier=PlanTier(subscription.tier),
            status=SubscriptionStatus(subscription.status),
            interval=BillingInterval(subscription.interval),
            current_period_start=subscription.current_period_start,
            current_period_end=subscription.current_period_end,
            cancel_at_period_end=subscription.cancel_at_period_end,
            trial_end=subscription.trial_end,
            created_at=subscription.created_at,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete("/subscription", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_subscription(
    immediately: bool = False,
    user: UserContext = Depends(require_permissions("billing:manage")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Cancel subscription."""
    from app.billing import BillingService

    service = BillingService(db)
    await service.cancel_subscription(tenant.tenant_id, immediately=immediately)


# ============================================================================
# Usage
# ============================================================================

@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get current period usage."""
    from app.billing import BillingService

    service = BillingService(db)
    usage = await service.get_usage(tenant.tenant_id)

    return UsageResponse(**usage)


@router.get("/usage/history")
async def get_usage_history(
    months: int = Query(6, ge=1, le=24),
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get usage history."""
    from app.billing import BillingService

    service = BillingService(db)
    history = await service.get_usage_history(tenant.tenant_id, months=months)

    return {"history": history}


# ============================================================================
# Invoices
# ============================================================================

@router.get("/invoices", response_model=InvoiceListResponse)
async def list_invoices(
    limit: int = Query(10, ge=1, le=100),
    starting_after: Optional[str] = None,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """List invoices."""
    from app.billing import BillingService

    service = BillingService(db)
    invoices = await service.list_invoices(
        tenant_id=tenant.tenant_id,
        limit=limit,
        starting_after=starting_after,
    )

    return InvoiceListResponse(
        invoices=[
            InvoiceResponse(
                id=inv.id,
                number=inv.number,
                status=inv.status,
                amount_due=inv.amount_due,
                amount_paid=inv.amount_paid,
                currency=inv.currency,
                period_start=inv.period_start,
                period_end=inv.period_end,
                due_date=inv.due_date,
                paid_at=inv.paid_at,
                invoice_url=inv.invoice_url,
                pdf_url=inv.pdf_url,
                line_items=inv.line_items or [],
            )
            for inv in invoices
        ],
        total=len(invoices),
    )


@router.get("/invoices/{invoice_id}", response_model=InvoiceResponse)
async def get_invoice(
    invoice_id: str,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get invoice details."""
    from app.billing import BillingService

    service = BillingService(db)
    invoice = await service.get_invoice(invoice_id, tenant.tenant_id)

    if not invoice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invoice not found",
        )

    return InvoiceResponse(
        id=invoice.id,
        number=invoice.number,
        status=invoice.status,
        amount_due=invoice.amount_due,
        amount_paid=invoice.amount_paid,
        currency=invoice.currency,
        period_start=invoice.period_start,
        period_end=invoice.period_end,
        due_date=invoice.due_date,
        paid_at=invoice.paid_at,
        invoice_url=invoice.invoice_url,
        pdf_url=invoice.pdf_url,
        line_items=invoice.line_items or [],
    )


@router.get("/invoices/upcoming", response_model=InvoiceResponse)
async def get_upcoming_invoice(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get upcoming invoice preview."""
    from app.billing import BillingService

    service = BillingService(db)
    invoice = await service.get_upcoming_invoice(tenant.tenant_id)

    if not invoice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No upcoming invoice",
        )

    return InvoiceResponse(
        id=invoice.id,
        number=invoice.number or "DRAFT",
        status="draft",
        amount_due=invoice.amount_due,
        amount_paid=0,
        currency=invoice.currency,
        period_start=invoice.period_start,
        period_end=invoice.period_end,
        due_date=invoice.due_date,
        paid_at=None,
        invoice_url=None,
        pdf_url=None,
        line_items=invoice.line_items or [],
    )


# ============================================================================
# Payment Methods
# ============================================================================

@router.get("/payment-methods", response_model=List[PaymentMethodResponse])
async def list_payment_methods(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """List payment methods."""
    from app.billing import BillingService

    service = BillingService(db)
    methods = await service.list_payment_methods(tenant.tenant_id)

    return [
        PaymentMethodResponse(
            id=m.id,
            type=PaymentMethodType(m.type),
            last_four=m.last_four,
            brand=m.brand,
            exp_month=m.exp_month,
            exp_year=m.exp_year,
            is_default=m.is_default,
            created_at=m.created_at,
        )
        for m in methods
    ]


@router.post("/payment-methods", response_model=PaymentMethodResponse)
async def add_payment_method(
    data: CreatePaymentMethodRequest,
    user: UserContext = Depends(require_permissions("billing:manage")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Add a payment method."""
    from app.billing import BillingService

    service = BillingService(db)

    method = await service.add_payment_method(
        tenant_id=tenant.tenant_id,
        payment_method_token=data.payment_method_token,
        set_as_default=data.set_as_default,
    )

    return PaymentMethodResponse(
        id=method.id,
        type=PaymentMethodType(method.type),
        last_four=method.last_four,
        brand=method.brand,
        exp_month=method.exp_month,
        exp_year=method.exp_year,
        is_default=method.is_default,
        created_at=method.created_at,
    )


@router.delete("/payment-methods/{method_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_payment_method(
    method_id: str,
    user: UserContext = Depends(require_permissions("billing:manage")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a payment method."""
    from app.billing import BillingService

    service = BillingService(db)
    await service.delete_payment_method(method_id, tenant.tenant_id)


@router.post("/payment-methods/{method_id}/default", response_model=PaymentMethodResponse)
async def set_default_payment_method(
    method_id: str,
    user: UserContext = Depends(require_permissions("billing:manage")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Set default payment method."""
    from app.billing import BillingService

    service = BillingService(db)
    method = await service.set_default_payment_method(method_id, tenant.tenant_id)

    return PaymentMethodResponse(
        id=method.id,
        type=PaymentMethodType(method.type),
        last_four=method.last_four,
        brand=method.brand,
        exp_month=method.exp_month,
        exp_year=method.exp_year,
        is_default=True,
        created_at=method.created_at,
    )


# ============================================================================
# Checkout
# ============================================================================

@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout_session(
    data: CreateCheckoutRequest,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Create a Stripe checkout session."""
    from app.billing import BillingService

    service = BillingService(db)

    session = await service.create_checkout_session(
        tenant_id=tenant.tenant_id,
        plan_id=data.plan_id,
        interval=data.interval.value,
        success_url=data.success_url,
        cancel_url=data.cancel_url,
    )

    return CheckoutResponse(
        checkout_url=session.url,
        session_id=session.id,
    )


# ============================================================================
# Webhooks
# ============================================================================

@router.post("/webhooks/stripe")
async def handle_stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """Handle Stripe webhooks."""
    from app.billing import BillingService
    import stripe

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        from app.config import get_settings
        settings = get_settings()

        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            settings.stripe_webhook_secret,
        )

        service = BillingService(db)
        await service.handle_stripe_webhook(event)

        return {"status": "success"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
