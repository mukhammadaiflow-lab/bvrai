"""
Pricing System

Pricing configuration, calculation, and plan management.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    BillingPeriod,
    Credit,
    CreditType,
    Discount,
    DiscountType,
    PlanFeatures,
    PlanTier,
    PricingTier,
    SubscriptionPlan,
    UsagePricing,
    UsageType,
)


logger = logging.getLogger(__name__)


# Default pricing configuration

def create_default_call_minute_pricing() -> UsagePricing:
    """Create default call minute pricing (tiered)."""
    return UsagePricing(
        usage_type=UsageType.CALL_MINUTES,
        tiers=[
            PricingTier(min_units=0, max_units=1000, price_per_unit_cents=5),
            PricingTier(min_units=1001, max_units=5000, price_per_unit_cents=4),
            PricingTier(min_units=5001, max_units=20000, price_per_unit_cents=3),
            PricingTier(min_units=20001, max_units=None, price_per_unit_cents=2),
        ],
        included_units=0,
        minimum_charge_cents=0,
    )


def create_default_sms_pricing() -> UsagePricing:
    """Create default SMS pricing."""
    return UsagePricing(
        usage_type=UsageType.SMS_MESSAGES,
        tiers=[
            PricingTier(min_units=0, max_units=None, price_per_unit_cents=2),
        ],
        included_units=0,
    )


def create_default_phone_number_pricing() -> UsagePricing:
    """Create default phone number pricing."""
    return UsagePricing(
        usage_type=UsageType.PHONE_NUMBERS,
        tiers=[
            PricingTier(min_units=0, max_units=None, price_per_unit_cents=200),  # $2/month per number
        ],
        included_units=1,  # First number included
    )


def create_default_api_pricing() -> UsagePricing:
    """Create default API request pricing."""
    return UsagePricing(
        usage_type=UsageType.API_REQUESTS,
        tiers=[
            PricingTier(min_units=0, max_units=100000, price_per_unit_cents=0),  # First 100k free
            PricingTier(min_units=100001, max_units=None, price_per_unit_cents=0),  # Then $0.001 each
        ],
        included_units=0,
    )


def create_default_usage_pricing() -> Dict[UsageType, UsagePricing]:
    """Create default usage pricing for all types."""
    return {
        UsageType.CALL_MINUTES: create_default_call_minute_pricing(),
        UsageType.SMS_MESSAGES: create_default_sms_pricing(),
        UsageType.PHONE_NUMBERS: create_default_phone_number_pricing(),
        UsageType.API_REQUESTS: create_default_api_pricing(),
    }


# Default plans

def create_free_plan() -> SubscriptionPlan:
    """Create free tier plan."""
    return SubscriptionPlan(
        plan_id="plan_free",
        name="Free",
        tier=PlanTier.FREE,
        description="Get started with AI voice agents for free",
        billing_period=BillingPeriod.MONTHLY,
        base_price_cents=0,
        features=PlanFeatures(
            max_agents=1,
            max_concurrent_calls=1,
            max_phone_numbers=1,
            max_knowledge_base_size_mb=50,
            included_call_minutes=100,
            included_sms_messages=10,
            included_api_requests=10000,
            custom_voices=False,
            priority_support=False,
            advanced_analytics=False,
            sentiment_analysis=False,
            multi_language=False,
        ),
        usage_pricing={
            UsageType.CALL_MINUTES: UsagePricing(
                usage_type=UsageType.CALL_MINUTES,
                tiers=[PricingTier(min_units=0, max_units=None, price_per_unit_cents=7)],
                included_units=100,
            ),
            UsageType.SMS_MESSAGES: UsagePricing(
                usage_type=UsageType.SMS_MESSAGES,
                tiers=[PricingTier(min_units=0, max_units=None, price_per_unit_cents=3)],
                included_units=10,
            ),
        },
        trial_days=0,
    )


def create_starter_plan() -> SubscriptionPlan:
    """Create starter tier plan."""
    return SubscriptionPlan(
        plan_id="plan_starter",
        name="Starter",
        tier=PlanTier.STARTER,
        description="For small businesses getting started with AI voice agents",
        billing_period=BillingPeriod.MONTHLY,
        base_price_cents=4900,  # $49/month
        features=PlanFeatures(
            max_agents=3,
            max_concurrent_calls=2,
            max_phone_numbers=3,
            max_knowledge_base_size_mb=500,
            included_call_minutes=500,
            included_sms_messages=100,
            included_api_requests=100000,
            custom_voices=False,
            priority_support=False,
            advanced_analytics=True,
            sentiment_analysis=True,
            multi_language=False,
        ),
        usage_pricing={
            UsageType.CALL_MINUTES: UsagePricing(
                usage_type=UsageType.CALL_MINUTES,
                tiers=[
                    PricingTier(min_units=0, max_units=5000, price_per_unit_cents=5),
                    PricingTier(min_units=5001, max_units=None, price_per_unit_cents=4),
                ],
                included_units=500,
            ),
            UsageType.SMS_MESSAGES: UsagePricing(
                usage_type=UsageType.SMS_MESSAGES,
                tiers=[PricingTier(min_units=0, max_units=None, price_per_unit_cents=2)],
                included_units=100,
            ),
            UsageType.PHONE_NUMBERS: UsagePricing(
                usage_type=UsageType.PHONE_NUMBERS,
                tiers=[PricingTier(min_units=0, max_units=None, price_per_unit_cents=200)],
                included_units=3,
            ),
        },
        trial_days=14,
    )


def create_professional_plan() -> SubscriptionPlan:
    """Create professional tier plan."""
    return SubscriptionPlan(
        plan_id="plan_professional",
        name="Professional",
        tier=PlanTier.PROFESSIONAL,
        description="For growing businesses with higher volume needs",
        billing_period=BillingPeriod.MONTHLY,
        base_price_cents=19900,  # $199/month
        features=PlanFeatures(
            max_agents=10,
            max_concurrent_calls=10,
            max_phone_numbers=10,
            max_knowledge_base_size_mb=2000,
            included_call_minutes=2500,
            included_sms_messages=500,
            included_api_requests=500000,
            custom_voices=True,
            priority_support=True,
            advanced_analytics=True,
            sentiment_analysis=True,
            multi_language=True,
        ),
        usage_pricing={
            UsageType.CALL_MINUTES: UsagePricing(
                usage_type=UsageType.CALL_MINUTES,
                tiers=[
                    PricingTier(min_units=0, max_units=10000, price_per_unit_cents=4),
                    PricingTier(min_units=10001, max_units=50000, price_per_unit_cents=3),
                    PricingTier(min_units=50001, max_units=None, price_per_unit_cents=2),
                ],
                included_units=2500,
            ),
            UsageType.SMS_MESSAGES: UsagePricing(
                usage_type=UsageType.SMS_MESSAGES,
                tiers=[PricingTier(min_units=0, max_units=None, price_per_unit_cents=2)],
                included_units=500,
            ),
            UsageType.PHONE_NUMBERS: UsagePricing(
                usage_type=UsageType.PHONE_NUMBERS,
                tiers=[PricingTier(min_units=0, max_units=None, price_per_unit_cents=150)],
                included_units=10,
            ),
        },
        trial_days=14,
    )


def create_enterprise_plan() -> SubscriptionPlan:
    """Create enterprise tier plan."""
    return SubscriptionPlan(
        plan_id="plan_enterprise",
        name="Enterprise",
        tier=PlanTier.ENTERPRISE,
        description="For large organizations with custom requirements",
        billing_period=BillingPeriod.MONTHLY,
        base_price_cents=99900,  # $999/month
        features=PlanFeatures(
            max_agents=100,
            max_concurrent_calls=50,
            max_phone_numbers=50,
            max_knowledge_base_size_mb=10000,
            included_call_minutes=10000,
            included_sms_messages=2000,
            included_api_requests=2000000,
            custom_voices=True,
            priority_support=True,
            dedicated_infrastructure=True,
            sla_guarantee=True,
            white_label=True,
            custom_integrations=True,
            advanced_analytics=True,
            sentiment_analysis=True,
            multi_language=True,
        ),
        usage_pricing={
            UsageType.CALL_MINUTES: UsagePricing(
                usage_type=UsageType.CALL_MINUTES,
                tiers=[
                    PricingTier(min_units=0, max_units=None, price_per_unit_cents=2),
                ],
                included_units=10000,
            ),
            UsageType.SMS_MESSAGES: UsagePricing(
                usage_type=UsageType.SMS_MESSAGES,
                tiers=[PricingTier(min_units=0, max_units=None, price_per_unit_cents=1)],
                included_units=2000,
            ),
            UsageType.PHONE_NUMBERS: UsagePricing(
                usage_type=UsageType.PHONE_NUMBERS,
                tiers=[PricingTier(min_units=0, max_units=None, price_per_unit_cents=100)],
                included_units=50,
            ),
        },
        trial_days=30,
    )


def get_default_plans() -> List[SubscriptionPlan]:
    """Get all default subscription plans."""
    return [
        create_free_plan(),
        create_starter_plan(),
        create_professional_plan(),
        create_enterprise_plan(),
    ]


@dataclass
class PlanCatalog:
    """Catalog of available subscription plans."""

    plans: Dict[str, SubscriptionPlan] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize with default plans if empty."""
        if not self.plans:
            for plan in get_default_plans():
                self.plans[plan.plan_id] = plan

    def get_plan(self, plan_id: str) -> Optional[SubscriptionPlan]:
        """Get plan by ID."""
        return self.plans.get(plan_id)

    def get_plan_by_tier(self, tier: PlanTier) -> Optional[SubscriptionPlan]:
        """Get plan by tier."""
        for plan in self.plans.values():
            if plan.tier == tier and plan.is_active and plan.is_public:
                return plan
        return None

    def list_public_plans(self) -> List[SubscriptionPlan]:
        """List all public plans."""
        return [
            plan for plan in self.plans.values()
            if plan.is_active and plan.is_public
        ]

    def add_plan(self, plan: SubscriptionPlan) -> None:
        """Add or update a plan."""
        self.plans[plan.plan_id] = plan

    def remove_plan(self, plan_id: str) -> bool:
        """Remove a plan."""
        if plan_id in self.plans:
            del self.plans[plan_id]
            return True
        return False


@dataclass
class PriceQuote:
    """Price quote for a billing period."""

    plan: SubscriptionPlan
    billing_period: BillingPeriod

    # Base costs
    base_price_cents: int
    usage_estimates: Dict[UsageType, Tuple[int, int]] = field(default_factory=dict)  # (units, cost)

    # Adjustments
    discounts: List[Tuple[Discount, int]] = field(default_factory=list)  # (discount, amount)
    credits: List[Tuple[Credit, int]] = field(default_factory=list)  # (credit, amount)

    # Totals
    subtotal_cents: int = 0
    total_discount_cents: int = 0
    total_credit_cents: int = 0
    tax_cents: int = 0
    total_cents: int = 0

    currency: str = "usd"
    created_at: datetime = field(default_factory=datetime.utcnow)
    valid_until: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))

    def is_valid(self) -> bool:
        """Check if quote is still valid."""
        return datetime.utcnow() < self.valid_until


class PricingCalculator:
    """Calculator for pricing and billing."""

    def __init__(self, catalog: Optional[PlanCatalog] = None):
        """Initialize pricing calculator."""
        self.catalog = catalog or PlanCatalog()
        self._tax_rates: Dict[str, Decimal] = {}

    def set_tax_rate(self, country: str, state: Optional[str], rate: Decimal) -> None:
        """Set tax rate for a region."""
        key = f"{country}:{state}" if state else country
        self._tax_rates[key] = rate

    def get_tax_rate(self, country: str, state: Optional[str] = None) -> Decimal:
        """Get tax rate for a region."""
        if state:
            key = f"{country}:{state}"
            if key in self._tax_rates:
                return self._tax_rates[key]
        return self._tax_rates.get(country, Decimal("0"))

    def calculate_usage_cost(
        self,
        plan: SubscriptionPlan,
        usage_type: UsageType,
        quantity: int,
    ) -> int:
        """Calculate cost for usage."""
        pricing = plan.usage_pricing.get(usage_type)

        if not pricing:
            logger.warning(f"No pricing configured for {usage_type} on plan {plan.plan_id}")
            return 0

        return pricing.calculate_cost(quantity)

    def calculate_period_cost(
        self,
        plan: SubscriptionPlan,
        usage: Dict[UsageType, int],
        discounts: Optional[List[Discount]] = None,
        credits: Optional[List[Credit]] = None,
    ) -> PriceQuote:
        """Calculate total cost for a billing period."""
        quote = PriceQuote(
            plan=plan,
            billing_period=plan.billing_period,
            base_price_cents=plan.base_price_cents,
        )

        # Calculate usage costs
        total_usage_cost = 0
        for usage_type, quantity in usage.items():
            cost = self.calculate_usage_cost(plan, usage_type, quantity)
            quote.usage_estimates[usage_type] = (quantity, cost)
            total_usage_cost += cost

        quote.subtotal_cents = plan.base_price_cents + total_usage_cost

        # Apply discounts
        remaining = quote.subtotal_cents
        if discounts:
            for discount in sorted(discounts, key=lambda d: d.discount_type == DiscountType.PERCENTAGE):
                if discount.is_valid():
                    amount = discount.calculate_discount(remaining)
                    if amount > 0:
                        quote.discounts.append((discount, amount))
                        quote.total_discount_cents += amount
                        remaining -= amount

        # Apply credits
        if credits:
            for credit in sorted(credits, key=lambda c: c.expires_at or datetime.max):
                if credit.is_valid() and remaining > 0:
                    amount = min(remaining, credit.remaining_cents)
                    if amount > 0:
                        quote.credits.append((credit, amount))
                        quote.total_credit_cents += amount
                        remaining -= amount

        # Calculate final total (before tax)
        pre_tax = quote.subtotal_cents - quote.total_discount_cents - quote.total_credit_cents
        quote.total_cents = max(0, pre_tax)

        return quote

    def calculate_quote_with_tax(
        self,
        quote: PriceQuote,
        country: str,
        state: Optional[str] = None,
    ) -> PriceQuote:
        """Add tax to a price quote."""
        tax_rate = self.get_tax_rate(country, state)
        quote.tax_cents = int(quote.total_cents * tax_rate / 100)
        quote.total_cents += quote.tax_cents
        return quote

    def calculate_proration(
        self,
        from_plan: SubscriptionPlan,
        to_plan: SubscriptionPlan,
        days_remaining: int,
        total_days: int,
    ) -> Tuple[int, int]:
        """
        Calculate proration for plan change.

        Returns:
            Tuple of (credit_cents, charge_cents)
        """
        if total_days <= 0:
            return (0, 0)

        # Credit for unused time on old plan
        old_daily_rate = from_plan.base_price_cents / total_days
        credit = int(old_daily_rate * days_remaining)

        # Charge for remaining time on new plan
        new_daily_rate = to_plan.base_price_cents / total_days
        charge = int(new_daily_rate * days_remaining)

        return (credit, charge)

    def estimate_monthly_cost(
        self,
        plan: SubscriptionPlan,
        estimated_usage: Dict[UsageType, int],
    ) -> int:
        """Estimate monthly cost based on usage."""
        base_cost = plan.get_monthly_equivalent()

        usage_cost = 0
        for usage_type, quantity in estimated_usage.items():
            usage_cost += self.calculate_usage_cost(plan, usage_type, quantity)

        return base_cost + usage_cost

    def compare_plans(
        self,
        estimated_usage: Dict[UsageType, int],
    ) -> List[Tuple[SubscriptionPlan, int]]:
        """Compare plans for given usage estimate."""
        results = []

        for plan in self.catalog.list_public_plans():
            cost = self.estimate_monthly_cost(plan, estimated_usage)
            results.append((plan, cost))

        return sorted(results, key=lambda x: x[1])

    def recommend_plan(
        self,
        estimated_usage: Dict[UsageType, int],
    ) -> Optional[SubscriptionPlan]:
        """Recommend best plan for given usage."""
        comparisons = self.compare_plans(estimated_usage)

        if not comparisons:
            return None

        # Find cheapest plan that can handle the usage
        for plan, cost in comparisons:
            can_handle = True

            # Check if plan features support the usage
            if estimated_usage.get(UsageType.AGENTS, 0) > plan.features.max_agents:
                can_handle = False
            if estimated_usage.get(UsageType.CONCURRENT_CALLS, 0) > plan.features.max_concurrent_calls:
                can_handle = False
            if estimated_usage.get(UsageType.PHONE_NUMBERS, 0) > plan.features.max_phone_numbers:
                can_handle = False

            if can_handle:
                return plan

        # Return enterprise if nothing else fits
        return self.catalog.get_plan_by_tier(PlanTier.ENTERPRISE)


class DiscountManager:
    """Manager for discount codes and promotions."""

    def __init__(self):
        """Initialize discount manager."""
        self._discounts: Dict[str, Discount] = {}
        self._coupon_codes: Dict[str, str] = {}  # code -> discount_id

    def create_discount(
        self,
        name: str,
        discount_type: DiscountType,
        value: int,
        coupon_code: Optional[str] = None,
        valid_days: Optional[int] = None,
        usage_limit: Optional[int] = None,
        applies_to: Optional[List[UsageType]] = None,
        min_purchase_cents: int = 0,
        max_discount_cents: Optional[int] = None,
    ) -> Discount:
        """Create a new discount."""
        import uuid

        discount_id = f"disc_{uuid.uuid4().hex[:12]}"

        discount = Discount(
            id=discount_id,
            name=name,
            discount_type=discount_type,
            value=value,
            coupon_code=coupon_code,
            valid_from=datetime.utcnow(),
            valid_until=datetime.utcnow() + timedelta(days=valid_days) if valid_days else None,
            usage_limit=usage_limit,
            applies_to=set(applies_to) if applies_to else None,
            min_purchase_cents=min_purchase_cents,
            max_discount_cents=max_discount_cents,
        )

        self._discounts[discount_id] = discount

        if coupon_code:
            self._coupon_codes[coupon_code.upper()] = discount_id

        return discount

    def get_discount(self, discount_id: str) -> Optional[Discount]:
        """Get discount by ID."""
        return self._discounts.get(discount_id)

    def get_by_coupon(self, coupon_code: str) -> Optional[Discount]:
        """Get discount by coupon code."""
        discount_id = self._coupon_codes.get(coupon_code.upper())
        if discount_id:
            return self._discounts.get(discount_id)
        return None

    def validate_coupon(self, coupon_code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a coupon code.

        Returns:
            Tuple of (is_valid, error_message)
        """
        discount = self.get_by_coupon(coupon_code)

        if not discount:
            return (False, "Invalid coupon code")

        if not discount.is_valid():
            if discount.valid_until and datetime.utcnow() > discount.valid_until:
                return (False, "Coupon has expired")
            if discount.usage_limit and discount.times_used >= discount.usage_limit:
                return (False, "Coupon usage limit reached")
            return (False, "Coupon is not valid")

        return (True, None)

    def use_discount(self, discount_id: str) -> bool:
        """Record usage of a discount."""
        discount = self._discounts.get(discount_id)
        if discount and discount.is_valid():
            discount.times_used += 1
            return True
        return False

    def list_active(self) -> List[Discount]:
        """List all active discounts."""
        return [d for d in self._discounts.values() if d.is_valid()]


class CreditManager:
    """Manager for billing credits."""

    def __init__(self):
        """Initialize credit manager."""
        self._credits: Dict[str, List[Credit]] = {}  # org_id -> credits

    def add_credit(
        self,
        organization_id: str,
        amount_cents: int,
        credit_type: CreditType,
        description: str,
        expires_in_days: Optional[int] = None,
    ) -> Credit:
        """Add credit to organization."""
        import uuid

        credit = Credit(
            id=f"cred_{uuid.uuid4().hex[:12]}",
            organization_id=organization_id,
            credit_type=credit_type,
            amount_cents=amount_cents,
            remaining_cents=amount_cents,
            description=description,
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days) if expires_in_days else None,
        )

        if organization_id not in self._credits:
            self._credits[organization_id] = []

        self._credits[organization_id].append(credit)
        return credit

    def get_credits(self, organization_id: str) -> List[Credit]:
        """Get all credits for organization."""
        return self._credits.get(organization_id, [])

    def get_valid_credits(self, organization_id: str) -> List[Credit]:
        """Get valid credits for organization."""
        credits = self._credits.get(organization_id, [])
        return [c for c in credits if c.is_valid()]

    def get_total_balance(self, organization_id: str) -> int:
        """Get total credit balance for organization."""
        valid_credits = self.get_valid_credits(organization_id)
        return sum(c.remaining_cents for c in valid_credits)

    def apply_credits(
        self,
        organization_id: str,
        amount_cents: int,
    ) -> Tuple[int, List[Tuple[Credit, int]]]:
        """
        Apply credits to an amount.

        Returns:
            Tuple of (remaining_amount, list of (credit, amount_applied))
        """
        applied: List[Tuple[Credit, int]] = []
        remaining = amount_cents

        # Sort by expiration (soonest first)
        valid_credits = sorted(
            self.get_valid_credits(organization_id),
            key=lambda c: c.expires_at or datetime.max
        )

        for credit in valid_credits:
            if remaining <= 0:
                break

            amount = credit.apply(remaining)
            if amount > 0:
                applied.append((credit, amount))
                remaining -= amount

        return (remaining, applied)

    def refund_credit(self, credit_id: str, amount_cents: int) -> bool:
        """Refund credit back to a credit record."""
        for org_credits in self._credits.values():
            for credit in org_credits:
                if credit.id == credit_id:
                    credit.remaining_cents += amount_cents
                    if credit.remaining_cents > credit.amount_cents:
                        credit.remaining_cents = credit.amount_cents
                    return True
        return False
