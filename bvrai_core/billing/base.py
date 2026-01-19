"""
Billing Base Types

Core types and data structures for the billing system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class BillingPeriod(str, Enum):
    """Billing period options."""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class PlanTier(str, Enum):
    """Subscription plan tiers."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class UsageType(str, Enum):
    """Types of billable usage."""
    CALL_MINUTES = "call_minutes"
    INBOUND_CALLS = "inbound_calls"
    OUTBOUND_CALLS = "outbound_calls"
    SMS_MESSAGES = "sms_messages"
    PHONE_NUMBERS = "phone_numbers"
    CONCURRENT_CALLS = "concurrent_calls"
    API_REQUESTS = "api_requests"
    STORAGE_GB = "storage_gb"
    KNOWLEDGE_QUERIES = "knowledge_queries"
    AGENTS = "agents"
    TRANSCRIPTION_MINUTES = "transcription_minutes"


class PaymentStatus(str, Enum):
    """Payment status states."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"
    DISPUTED = "disputed"
    CANCELED = "canceled"


class InvoiceStatus(str, Enum):
    """Invoice status states."""
    DRAFT = "draft"
    OPEN = "open"
    PAID = "paid"
    VOID = "void"
    UNCOLLECTIBLE = "uncollectible"
    PAST_DUE = "past_due"


class SubscriptionStatus(str, Enum):
    """Subscription status states."""
    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    PAUSED = "paused"
    EXPIRED = "expired"


class CreditType(str, Enum):
    """Types of billing credits."""
    PROMOTIONAL = "promotional"
    REFERRAL = "referral"
    COMPENSATION = "compensation"
    PREPAID = "prepaid"
    ADJUSTMENT = "adjustment"


class DiscountType(str, Enum):
    """Types of discounts."""
    PERCENTAGE = "percentage"
    FIXED_AMOUNT = "fixed_amount"


@dataclass
class PricingTier:
    """Pricing tier for usage-based billing."""

    min_units: int
    max_units: Optional[int]  # None for unlimited
    price_per_unit_cents: int
    flat_fee_cents: int = 0

    def calculate_cost(self, units: int) -> int:
        """Calculate cost for units in this tier."""
        if self.max_units is not None:
            billable_units = min(units - self.min_units + 1, self.max_units - self.min_units + 1)
        else:
            billable_units = units - self.min_units + 1

        return self.flat_fee_cents + (billable_units * self.price_per_unit_cents)


@dataclass
class UsagePricing:
    """Pricing configuration for a usage type."""

    usage_type: UsageType
    tiers: List[PricingTier]
    included_units: int = 0
    minimum_charge_cents: int = 0
    currency: str = "usd"

    def calculate_cost(self, units: int) -> int:
        """Calculate total cost for given usage."""
        billable_units = max(0, units - self.included_units)

        if billable_units == 0:
            return 0

        total_cost = 0
        remaining_units = billable_units

        for tier in self.tiers:
            if remaining_units <= 0:
                break

            tier_start = tier.min_units
            tier_end = tier.max_units if tier.max_units else float('inf')

            if billable_units < tier_start:
                continue

            units_in_tier = min(remaining_units, tier_end - tier_start + 1)
            total_cost += tier.flat_fee_cents + (units_in_tier * tier.price_per_unit_cents)
            remaining_units -= units_in_tier

        return max(total_cost, self.minimum_charge_cents)


@dataclass
class PlanFeatures:
    """Features included in a subscription plan."""

    max_agents: int
    max_concurrent_calls: int
    max_phone_numbers: int
    max_knowledge_base_size_mb: int
    included_call_minutes: int
    included_sms_messages: int
    included_api_requests: int

    custom_voices: bool = False
    priority_support: bool = False
    dedicated_infrastructure: bool = False
    sla_guarantee: bool = False
    white_label: bool = False
    custom_integrations: bool = False
    advanced_analytics: bool = False
    api_access: bool = True
    webhook_support: bool = True
    call_recording: bool = True
    real_time_transcription: bool = True
    sentiment_analysis: bool = False
    multi_language: bool = False


@dataclass
class SubscriptionPlan:
    """Subscription plan definition."""

    plan_id: str
    name: str
    tier: PlanTier
    description: str
    billing_period: BillingPeriod
    base_price_cents: int
    features: PlanFeatures
    usage_pricing: Dict[UsageType, UsagePricing]
    currency: str = "usd"
    trial_days: int = 0
    is_active: bool = True
    is_public: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_monthly_equivalent(self) -> int:
        """Get monthly equivalent price."""
        if self.billing_period == BillingPeriod.MONTHLY:
            return self.base_price_cents
        elif self.billing_period == BillingPeriod.QUARTERLY:
            return self.base_price_cents // 3
        elif self.billing_period == BillingPeriod.ANNUAL:
            return self.base_price_cents // 12
        return self.base_price_cents


@dataclass
class UsageRecord:
    """Record of billable usage."""

    id: str
    organization_id: str
    usage_type: UsageType
    quantity: Decimal
    timestamp: datetime
    idempotency_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Related entities
    call_id: Optional[str] = None
    agent_id: Optional[str] = None
    phone_number_id: Optional[str] = None

    # Billing
    billed: bool = False
    invoice_id: Optional[str] = None
    unit_price_cents: Optional[int] = None
    total_cents: Optional[int] = None


@dataclass
class UsageSummary:
    """Summary of usage for a billing period."""

    organization_id: str
    period_start: datetime
    period_end: datetime
    usage_by_type: Dict[UsageType, Decimal]
    cost_by_type: Dict[UsageType, int]
    total_cost_cents: int
    currency: str = "usd"


@dataclass
class Discount:
    """Discount applied to billing."""

    id: str
    name: str
    discount_type: DiscountType
    value: int  # Percentage (0-100) or cents
    applies_to: Optional[Set[UsageType]] = None  # None = all
    min_purchase_cents: int = 0
    max_discount_cents: Optional[int] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    usage_limit: Optional[int] = None
    times_used: int = 0
    coupon_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if discount is currently valid."""
        now = datetime.utcnow()

        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        if self.usage_limit and self.times_used >= self.usage_limit:
            return False

        return True

    def calculate_discount(self, amount_cents: int) -> int:
        """Calculate discount amount."""
        if not self.is_valid():
            return 0

        if amount_cents < self.min_purchase_cents:
            return 0

        if self.discount_type == DiscountType.PERCENTAGE:
            discount = (amount_cents * self.value) // 100
        else:
            discount = self.value

        if self.max_discount_cents:
            discount = min(discount, self.max_discount_cents)

        return min(discount, amount_cents)


@dataclass
class Credit:
    """Billing credit for an organization."""

    id: str
    organization_id: str
    credit_type: CreditType
    amount_cents: int
    remaining_cents: int
    description: str
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if credit is still valid and has remaining balance."""
        if self.remaining_cents <= 0:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def apply(self, amount_cents: int) -> int:
        """Apply credit and return amount applied."""
        if not self.is_valid():
            return 0

        applied = min(amount_cents, self.remaining_cents)
        self.remaining_cents -= applied
        return applied


@dataclass
class InvoiceLineItem:
    """Line item on an invoice."""

    id: str
    description: str
    quantity: Decimal
    unit_price_cents: int
    total_cents: int
    usage_type: Optional[UsageType] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Invoice:
    """Billing invoice."""

    id: str
    invoice_number: str
    organization_id: str
    subscription_id: Optional[str]
    status: InvoiceStatus
    currency: str

    # Amounts
    subtotal_cents: int
    discount_cents: int
    credit_applied_cents: int
    tax_cents: int
    total_cents: int
    amount_due_cents: int
    amount_paid_cents: int

    # Line items
    line_items: List[InvoiceLineItem]

    # Discounts and credits applied
    discounts_applied: List[str] = field(default_factory=list)
    credits_applied: List[str] = field(default_factory=list)

    # Dates
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    due_date: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    paid_at: Optional[datetime] = None
    voided_at: Optional[datetime] = None

    # External references
    stripe_invoice_id: Optional[str] = None
    payment_intent_id: Optional[str] = None

    # PDF
    pdf_url: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_past_due(self) -> bool:
        """Check if invoice is past due."""
        if self.status in [InvoiceStatus.PAID, InvoiceStatus.VOID]:
            return False
        return datetime.utcnow() > self.due_date


@dataclass
class PaymentMethod:
    """Payment method on file."""

    id: str
    organization_id: str
    type: str  # card, bank_account, etc.
    is_default: bool = False

    # Card details (masked)
    card_brand: Optional[str] = None
    card_last4: Optional[str] = None
    card_exp_month: Optional[int] = None
    card_exp_year: Optional[int] = None

    # Bank details (masked)
    bank_name: Optional[str] = None
    account_last4: Optional[str] = None

    # Stripe
    stripe_payment_method_id: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Payment:
    """Payment transaction record."""

    id: str
    organization_id: str
    invoice_id: str
    payment_method_id: str
    status: PaymentStatus
    amount_cents: int
    currency: str

    # Stripe
    stripe_payment_intent_id: Optional[str] = None
    stripe_charge_id: Optional[str] = None

    # Details
    failure_reason: Optional[str] = None
    refund_reason: Optional[str] = None
    refunded_amount_cents: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    refunded_at: Optional[datetime] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subscription:
    """Customer subscription."""

    id: str
    organization_id: str
    plan_id: str
    status: SubscriptionStatus

    # Dates
    current_period_start: datetime
    current_period_end: datetime
    trial_start: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Billing
    default_payment_method_id: Optional[str] = None

    # Options
    cancel_at_period_end: bool = False
    auto_renew: bool = True

    # Stripe
    stripe_subscription_id: Optional[str] = None
    stripe_customer_id: Optional[str] = None

    # Usage
    current_usage: Dict[UsageType, Decimal] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if subscription is in active state."""
        return self.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]

    def is_in_trial(self) -> bool:
        """Check if subscription is in trial period."""
        if self.status != SubscriptionStatus.TRIALING:
            return False
        if not self.trial_end:
            return False
        return datetime.utcnow() < self.trial_end

    def days_until_renewal(self) -> int:
        """Get days until next renewal."""
        delta = self.current_period_end - datetime.utcnow()
        return max(0, delta.days)


@dataclass
class BillingCustomer:
    """Billing customer record."""

    id: str
    organization_id: str
    email: str
    name: str

    # Address
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: str = "US"

    # Tax
    tax_id: Optional[str] = None
    tax_exempt: bool = False

    # Stripe
    stripe_customer_id: Optional[str] = None

    # Balance
    balance_cents: int = 0  # Positive = credit, negative = owes

    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BillingAlert:
    """Billing alert configuration."""

    id: str
    organization_id: str
    alert_type: str  # usage_threshold, budget_exceeded, payment_failed
    threshold_value: Optional[int] = None
    threshold_type: Optional[UsageType] = None
    enabled: bool = True
    notify_emails: List[str] = field(default_factory=list)
    notify_webhook: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


# Abstract interfaces

class PaymentProcessor(ABC):
    """Abstract payment processor interface."""

    @abstractmethod
    async def create_customer(self, customer: BillingCustomer) -> str:
        """Create customer in payment processor."""
        pass

    @abstractmethod
    async def update_customer(self, customer: BillingCustomer) -> None:
        """Update customer in payment processor."""
        pass

    @abstractmethod
    async def delete_customer(self, customer_id: str) -> None:
        """Delete customer from payment processor."""
        pass

    @abstractmethod
    async def create_payment_method(
        self,
        customer_id: str,
        payment_method_token: str,
    ) -> PaymentMethod:
        """Create payment method for customer."""
        pass

    @abstractmethod
    async def delete_payment_method(self, payment_method_id: str) -> None:
        """Delete payment method."""
        pass

    @abstractmethod
    async def create_subscription(
        self,
        customer_id: str,
        plan: SubscriptionPlan,
        payment_method_id: Optional[str] = None,
        trial_days: Optional[int] = None,
    ) -> Subscription:
        """Create subscription for customer."""
        pass

    @abstractmethod
    async def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True,
    ) -> Subscription:
        """Cancel subscription."""
        pass

    @abstractmethod
    async def charge_payment(
        self,
        customer_id: str,
        amount_cents: int,
        currency: str,
        payment_method_id: str,
        description: str,
    ) -> Payment:
        """Charge payment to customer."""
        pass

    @abstractmethod
    async def refund_payment(
        self,
        payment_id: str,
        amount_cents: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> Payment:
        """Refund payment."""
        pass

    @abstractmethod
    async def create_invoice(
        self,
        customer_id: str,
        line_items: List[InvoiceLineItem],
        auto_charge: bool = True,
    ) -> Invoice:
        """Create and optionally charge invoice."""
        pass


class UsageStore(ABC):
    """Abstract usage storage interface."""

    @abstractmethod
    async def record_usage(self, record: UsageRecord) -> None:
        """Record usage event."""
        pass

    @abstractmethod
    async def get_usage(
        self,
        organization_id: str,
        usage_type: UsageType,
        start_time: datetime,
        end_time: datetime,
    ) -> List[UsageRecord]:
        """Get usage records for a period."""
        pass

    @abstractmethod
    async def get_usage_summary(
        self,
        organization_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> UsageSummary:
        """Get aggregated usage summary."""
        pass

    @abstractmethod
    async def mark_as_billed(
        self,
        organization_id: str,
        invoice_id: str,
        end_time: datetime,
    ) -> int:
        """Mark usage records as billed."""
        pass


class SubscriptionStore(ABC):
    """Abstract subscription storage interface."""

    @abstractmethod
    async def create(self, subscription: Subscription) -> None:
        """Create subscription."""
        pass

    @abstractmethod
    async def get(self, subscription_id: str) -> Optional[Subscription]:
        """Get subscription by ID."""
        pass

    @abstractmethod
    async def get_by_organization(self, organization_id: str) -> Optional[Subscription]:
        """Get active subscription for organization."""
        pass

    @abstractmethod
    async def update(self, subscription: Subscription) -> None:
        """Update subscription."""
        pass

    @abstractmethod
    async def list_expiring(self, within_days: int) -> List[Subscription]:
        """List subscriptions expiring within days."""
        pass


class InvoiceStore(ABC):
    """Abstract invoice storage interface."""

    @abstractmethod
    async def create(self, invoice: Invoice) -> None:
        """Create invoice."""
        pass

    @abstractmethod
    async def get(self, invoice_id: str) -> Optional[Invoice]:
        """Get invoice by ID."""
        pass

    @abstractmethod
    async def get_by_number(self, invoice_number: str) -> Optional[Invoice]:
        """Get invoice by number."""
        pass

    @abstractmethod
    async def list_for_organization(
        self,
        organization_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Invoice]:
        """List invoices for organization."""
        pass

    @abstractmethod
    async def update(self, invoice: Invoice) -> None:
        """Update invoice."""
        pass

    @abstractmethod
    async def list_unpaid(self) -> List[Invoice]:
        """List all unpaid invoices."""
        pass


# Billing errors

class BillingError(Exception):
    """Base billing error."""

    def __init__(self, message: str, code: str = "billing_error"):
        self.message = message
        self.code = code
        super().__init__(message)


class PaymentError(BillingError):
    """Payment processing error."""

    def __init__(self, message: str, decline_code: Optional[str] = None):
        self.decline_code = decline_code
        super().__init__(message, "payment_error")


class SubscriptionError(BillingError):
    """Subscription management error."""

    def __init__(self, message: str):
        super().__init__(message, "subscription_error")


class UsageLimitError(BillingError):
    """Usage limit exceeded error."""

    def __init__(self, usage_type: UsageType, limit: int, current: int):
        self.usage_type = usage_type
        self.limit = limit
        self.current = current
        super().__init__(
            f"Usage limit exceeded for {usage_type.value}: {current}/{limit}",
            "usage_limit_exceeded"
        )


class InvoiceError(BillingError):
    """Invoice processing error."""

    def __init__(self, message: str, invoice_id: Optional[str] = None):
        self.invoice_id = invoice_id
        super().__init__(message, "invoice_error")
