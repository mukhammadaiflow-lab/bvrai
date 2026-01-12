"""Pricing and plans for billing."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import logging

from app.billing.usage import UsageType

logger = logging.getLogger(__name__)


class PricingTier(str, Enum):
    """Pricing tiers."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class PriceComponent:
    """A component of pricing."""
    usage_type: UsageType
    unit_price: Decimal  # Price per unit
    unit_name: str  # e.g., "minute", "1000 tokens", "GB"
    included_quantity: float = 0  # Included in base price
    minimum_charge: Decimal = Decimal("0")
    maximum_charge: Optional[Decimal] = None
    tiered_pricing: Optional[List[Dict[str, Any]]] = None

    def calculate_cost(self, quantity: float) -> Decimal:
        """Calculate cost for quantity."""
        # Subtract included quantity
        billable = max(0, quantity - self.included_quantity)

        if billable == 0:
            return Decimal("0")

        # Check for tiered pricing
        if self.tiered_pricing:
            return self._calculate_tiered(billable)

        # Simple pricing
        cost = self.unit_price * Decimal(str(billable))

        # Apply minimum
        if cost < self.minimum_charge:
            cost = self.minimum_charge

        # Apply maximum
        if self.maximum_charge and cost > self.maximum_charge:
            cost = self.maximum_charge

        return cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_tiered(self, quantity: float) -> Decimal:
        """Calculate cost using tiered pricing."""
        remaining = Decimal(str(quantity))
        total_cost = Decimal("0")

        for tier in self.tiered_pricing:
            tier_limit = Decimal(str(tier.get("up_to", float("inf"))))
            tier_price = Decimal(str(tier["price"]))

            if remaining <= 0:
                break

            tier_quantity = min(remaining, tier_limit)
            total_cost += tier_quantity * tier_price
            remaining -= tier_quantity

        return total_cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


@dataclass
class PricingPlan:
    """A pricing plan."""
    tier: PricingTier
    name: str
    description: str
    base_price: Decimal  # Monthly base price
    components: Dict[UsageType, PriceComponent] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    limits: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tier": self.tier.value,
            "name": self.name,
            "description": self.description,
            "base_price": float(self.base_price),
            "features": self.features,
            "limits": self.limits,
        }

    def get_price(self, usage_type: UsageType) -> Optional[PriceComponent]:
        """Get price component for usage type."""
        return self.components.get(usage_type)


# Default pricing plans
DEFAULT_PLANS: Dict[PricingTier, PricingPlan] = {
    PricingTier.FREE: PricingPlan(
        tier=PricingTier.FREE,
        name="Free",
        description="Get started with voice AI",
        base_price=Decimal("0"),
        components={
            UsageType.CALL_MINUTES: PriceComponent(
                usage_type=UsageType.CALL_MINUTES,
                unit_price=Decimal("0.05"),
                unit_name="minute",
                included_quantity=100,
            ),
            UsageType.LLM_INPUT_TOKENS: PriceComponent(
                usage_type=UsageType.LLM_INPUT_TOKENS,
                unit_price=Decimal("0.0001"),
                unit_name="1000 tokens",
                included_quantity=100000,
            ),
            UsageType.LLM_OUTPUT_TOKENS: PriceComponent(
                usage_type=UsageType.LLM_OUTPUT_TOKENS,
                unit_price=Decimal("0.0003"),
                unit_name="1000 tokens",
                included_quantity=50000,
            ),
            UsageType.TTS_CHARACTERS: PriceComponent(
                usage_type=UsageType.TTS_CHARACTERS,
                unit_price=Decimal("0.000015"),
                unit_name="character",
                included_quantity=100000,
            ),
        },
        features=[
            "100 minutes/month included",
            "1 phone number",
            "Basic analytics",
            "Community support",
        ],
        limits={
            "agents": 3,
            "phone_numbers": 1,
            "concurrent_calls": 2,
        },
    ),
    PricingTier.STARTER: PricingPlan(
        tier=PricingTier.STARTER,
        name="Starter",
        description="For small teams",
        base_price=Decimal("49"),
        components={
            UsageType.CALL_MINUTES: PriceComponent(
                usage_type=UsageType.CALL_MINUTES,
                unit_price=Decimal("0.04"),
                unit_name="minute",
                included_quantity=500,
            ),
            UsageType.LLM_INPUT_TOKENS: PriceComponent(
                usage_type=UsageType.LLM_INPUT_TOKENS,
                unit_price=Decimal("0.00008"),
                unit_name="1000 tokens",
                included_quantity=500000,
            ),
            UsageType.LLM_OUTPUT_TOKENS: PriceComponent(
                usage_type=UsageType.LLM_OUTPUT_TOKENS,
                unit_price=Decimal("0.00024"),
                unit_name="1000 tokens",
                included_quantity=250000,
            ),
            UsageType.TTS_CHARACTERS: PriceComponent(
                usage_type=UsageType.TTS_CHARACTERS,
                unit_price=Decimal("0.000012"),
                unit_name="character",
                included_quantity=500000,
            ),
            UsageType.STORAGE_GB: PriceComponent(
                usage_type=UsageType.STORAGE_GB,
                unit_price=Decimal("0.10"),
                unit_name="GB",
                included_quantity=10,
            ),
        },
        features=[
            "500 minutes/month included",
            "3 phone numbers",
            "Advanced analytics",
            "Email support",
            "Webhook integrations",
        ],
        limits={
            "agents": 10,
            "phone_numbers": 3,
            "concurrent_calls": 5,
            "team_members": 3,
        },
    ),
    PricingTier.PROFESSIONAL: PricingPlan(
        tier=PricingTier.PROFESSIONAL,
        name="Professional",
        description="For growing businesses",
        base_price=Decimal("199"),
        components={
            UsageType.CALL_MINUTES: PriceComponent(
                usage_type=UsageType.CALL_MINUTES,
                unit_price=Decimal("0.03"),
                unit_name="minute",
                included_quantity=2000,
            ),
            UsageType.LLM_INPUT_TOKENS: PriceComponent(
                usage_type=UsageType.LLM_INPUT_TOKENS,
                unit_price=Decimal("0.00006"),
                unit_name="1000 tokens",
                included_quantity=2000000,
            ),
            UsageType.LLM_OUTPUT_TOKENS: PriceComponent(
                usage_type=UsageType.LLM_OUTPUT_TOKENS,
                unit_price=Decimal("0.00018"),
                unit_name="1000 tokens",
                included_quantity=1000000,
            ),
            UsageType.TTS_CHARACTERS: PriceComponent(
                usage_type=UsageType.TTS_CHARACTERS,
                unit_price=Decimal("0.00001"),
                unit_name="character",
                included_quantity=2000000,
            ),
            UsageType.STORAGE_GB: PriceComponent(
                usage_type=UsageType.STORAGE_GB,
                unit_price=Decimal("0.08"),
                unit_name="GB",
                included_quantity=50,
            ),
            UsageType.TRANSCRIPTION_MINUTES: PriceComponent(
                usage_type=UsageType.TRANSCRIPTION_MINUTES,
                unit_price=Decimal("0.02"),
                unit_name="minute",
                included_quantity=500,
            ),
        },
        features=[
            "2,000 minutes/month included",
            "10 phone numbers",
            "Custom analytics",
            "Priority support",
            "Custom integrations",
            "SLA guarantee",
        ],
        limits={
            "agents": 50,
            "phone_numbers": 10,
            "concurrent_calls": 20,
            "team_members": 10,
        },
    ),
    PricingTier.ENTERPRISE: PricingPlan(
        tier=PricingTier.ENTERPRISE,
        name="Enterprise",
        description="For large organizations",
        base_price=Decimal("999"),
        components={
            UsageType.CALL_MINUTES: PriceComponent(
                usage_type=UsageType.CALL_MINUTES,
                unit_price=Decimal("0.02"),
                unit_name="minute",
                included_quantity=10000,
            ),
            UsageType.LLM_INPUT_TOKENS: PriceComponent(
                usage_type=UsageType.LLM_INPUT_TOKENS,
                unit_price=Decimal("0.00004"),
                unit_name="1000 tokens",
                included_quantity=10000000,
            ),
            UsageType.LLM_OUTPUT_TOKENS: PriceComponent(
                usage_type=UsageType.LLM_OUTPUT_TOKENS,
                unit_price=Decimal("0.00012"),
                unit_name="1000 tokens",
                included_quantity=5000000,
            ),
            UsageType.TTS_CHARACTERS: PriceComponent(
                usage_type=UsageType.TTS_CHARACTERS,
                unit_price=Decimal("0.000008"),
                unit_name="character",
                included_quantity=10000000,
            ),
            UsageType.STORAGE_GB: PriceComponent(
                usage_type=UsageType.STORAGE_GB,
                unit_price=Decimal("0.05"),
                unit_name="GB",
                included_quantity=200,
            ),
            UsageType.TRANSCRIPTION_MINUTES: PriceComponent(
                usage_type=UsageType.TRANSCRIPTION_MINUTES,
                unit_price=Decimal("0.015"),
                unit_name="minute",
                included_quantity=2000,
            ),
        },
        features=[
            "10,000 minutes/month included",
            "Unlimited phone numbers",
            "Dedicated support",
            "Custom SLA",
            "On-premise deployment",
            "Custom features",
        ],
        limits={
            "agents": -1,  # Unlimited
            "phone_numbers": -1,
            "concurrent_calls": 100,
            "team_members": -1,
        },
    ),
}


class PricingCalculator:
    """
    Calculates costs based on usage and pricing plans.

    Usage:
        calculator = PricingCalculator()

        # Calculate cost for usage
        cost = calculator.calculate_cost(
            tier=PricingTier.STARTER,
            usage={
                UsageType.CALL_MINUTES: 750,
                UsageType.LLM_INPUT_TOKENS: 1000000,
            },
        )
    """

    def __init__(self, plans: Optional[Dict[PricingTier, PricingPlan]] = None):
        self.plans = plans or DEFAULT_PLANS

    def get_plan(self, tier: PricingTier) -> Optional[PricingPlan]:
        """Get pricing plan by tier."""
        return self.plans.get(tier)

    def calculate_cost(
        self,
        tier: PricingTier,
        usage: Dict[UsageType, float],
    ) -> Dict[str, Any]:
        """Calculate total cost for usage."""
        plan = self.get_plan(tier)
        if not plan:
            raise ValueError(f"Unknown tier: {tier}")

        breakdown = {}
        total_usage_cost = Decimal("0")

        for usage_type, quantity in usage.items():
            component = plan.get_price(usage_type)
            if component:
                cost = component.calculate_cost(quantity)
                total_usage_cost += cost
                breakdown[usage_type.value] = {
                    "quantity": quantity,
                    "included": component.included_quantity,
                    "billable": max(0, quantity - component.included_quantity),
                    "unit_price": float(component.unit_price),
                    "cost": float(cost),
                }

        total_cost = plan.base_price + total_usage_cost

        return {
            "tier": tier.value,
            "base_price": float(plan.base_price),
            "usage_cost": float(total_usage_cost),
            "total_cost": float(total_cost),
            "breakdown": breakdown,
        }

    def estimate_monthly_cost(
        self,
        tier: PricingTier,
        estimated_calls: int,
        avg_call_duration_minutes: float,
        avg_tokens_per_call: int = 5000,
    ) -> Dict[str, Any]:
        """Estimate monthly cost based on call volume."""
        total_minutes = estimated_calls * avg_call_duration_minutes
        total_tokens = estimated_calls * avg_tokens_per_call
        tts_chars = estimated_calls * 1000  # Estimate

        usage = {
            UsageType.CALL_MINUTES: total_minutes,
            UsageType.CALL_COUNT: estimated_calls,
            UsageType.LLM_INPUT_TOKENS: total_tokens * 0.6,  # 60% input
            UsageType.LLM_OUTPUT_TOKENS: total_tokens * 0.4,  # 40% output
            UsageType.TTS_CHARACTERS: tts_chars,
        }

        return self.calculate_cost(tier, usage)

    def compare_plans(
        self,
        usage: Dict[UsageType, float],
    ) -> List[Dict[str, Any]]:
        """Compare costs across all plans."""
        comparisons = []

        for tier in PricingTier:
            if tier == PricingTier.CUSTOM:
                continue

            try:
                cost = self.calculate_cost(tier, usage)
                plan = self.get_plan(tier)
                comparisons.append({
                    **cost,
                    "name": plan.name,
                    "description": plan.description,
                    "features": plan.features,
                })
            except Exception as e:
                logger.error(f"Error calculating {tier}: {e}")

        return sorted(comparisons, key=lambda x: x["total_cost"])

    def get_recommended_plan(
        self,
        usage: Dict[UsageType, float],
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Recommend a plan based on usage and budget."""
        comparisons = self.compare_plans(usage)

        if budget:
            # Filter by budget
            within_budget = [c for c in comparisons if c["total_cost"] <= budget]
            if within_budget:
                return within_budget[-1]  # Best plan within budget

        # Return cheapest option
        return comparisons[0]


class DiscountCalculator:
    """
    Calculates discounts for billing.

    Supports:
    - Volume discounts
    - Promotional codes
    - Annual billing discounts
    """

    def __init__(self):
        self._promo_codes: Dict[str, Dict[str, Any]] = {}

    def add_promo_code(
        self,
        code: str,
        discount_percent: float,
        valid_until: Optional[str] = None,
        applicable_tiers: Optional[List[PricingTier]] = None,
    ) -> None:
        """Add a promotional code."""
        self._promo_codes[code.upper()] = {
            "discount_percent": discount_percent,
            "valid_until": valid_until,
            "applicable_tiers": applicable_tiers,
        }

    def apply_promo_code(
        self,
        code: str,
        amount: Decimal,
        tier: PricingTier,
    ) -> Dict[str, Any]:
        """Apply promotional code to amount."""
        code = code.upper()
        promo = self._promo_codes.get(code)

        if not promo:
            return {
                "valid": False,
                "error": "Invalid promo code",
                "original": float(amount),
                "final": float(amount),
            }

        # Check tier applicability
        if promo["applicable_tiers"] and tier not in promo["applicable_tiers"]:
            return {
                "valid": False,
                "error": "Promo code not valid for this plan",
                "original": float(amount),
                "final": float(amount),
            }

        discount = amount * Decimal(str(promo["discount_percent"])) / 100
        final = amount - discount

        return {
            "valid": True,
            "code": code,
            "discount_percent": promo["discount_percent"],
            "discount_amount": float(discount),
            "original": float(amount),
            "final": float(final),
        }

    def calculate_volume_discount(
        self,
        quantity: float,
        thresholds: List[Dict[str, Any]],
    ) -> float:
        """Calculate volume discount based on thresholds."""
        discount = 0.0

        for threshold in sorted(thresholds, key=lambda x: x["min_quantity"]):
            if quantity >= threshold["min_quantity"]:
                discount = threshold["discount_percent"]

        return discount

    def calculate_annual_discount(
        self,
        monthly_amount: Decimal,
        discount_percent: float = 20,
    ) -> Dict[str, Any]:
        """Calculate annual billing discount."""
        annual_without_discount = monthly_amount * 12
        discount = annual_without_discount * Decimal(str(discount_percent)) / 100
        annual_with_discount = annual_without_discount - discount

        return {
            "monthly_amount": float(monthly_amount),
            "annual_without_discount": float(annual_without_discount),
            "discount_percent": discount_percent,
            "discount_amount": float(discount),
            "annual_with_discount": float(annual_with_discount),
            "effective_monthly": float(annual_with_discount / 12),
        }


# Global pricing calculator
_pricing_calculator: Optional[PricingCalculator] = None


def get_pricing_calculator() -> PricingCalculator:
    """Get or create the global pricing calculator."""
    global _pricing_calculator
    if _pricing_calculator is None:
        _pricing_calculator = PricingCalculator()
    return _pricing_calculator
