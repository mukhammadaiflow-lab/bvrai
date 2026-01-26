"""Unit tests for billing functionality."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class TestSubscriptionPlan:
    """Tests for subscription plans."""

    def test_create_plan(self):
        """Test creating a subscription plan."""
        from bvrai_core.billing.pricing import SubscriptionPlan, PlanFeatures

        features = PlanFeatures(
            max_agents=5,
            max_calls_per_month=1000,
            max_minutes_per_month=5000,
            has_custom_voices=True,
            has_knowledge_base=True,
            support_level="standard",
        )

        plan = SubscriptionPlan(
            id="plan_pro",
            name="Professional",
            description="For growing businesses",
            price_cents_monthly=9900,
            features=features,
        )

        assert plan.id == "plan_pro"
        assert plan.price_cents_monthly == 9900
        assert plan.features.max_agents == 5

    def test_plan_annual_pricing(self):
        """Test annual pricing calculation."""
        from bvrai_core.billing.pricing import SubscriptionPlan, PlanFeatures

        plan = SubscriptionPlan(
            id="plan_test",
            name="Test",
            price_cents_monthly=10000,  # $100/month
            features=PlanFeatures(),
            annual_discount_percent=20,  # 20% discount
        )

        # Annual price should be monthly * 12 * (1 - discount)
        expected = 10000 * 12 * 0.8  # $960/year
        assert plan.price_cents_annual == expected

    def test_plan_feature_limits(self):
        """Test checking feature limits."""
        from bvrai_core.billing.pricing import PlanFeatures

        features = PlanFeatures(
            max_agents=3,
            max_calls_per_month=500,
        )

        # Within limits
        assert features.check_limit("agents", 2) is True
        assert features.check_limit("calls", 400) is True

        # Exceeds limits
        assert features.check_limit("agents", 5) is False
        assert features.check_limit("calls", 600) is False


class TestPlanCatalog:
    """Tests for plan catalog."""

    def test_list_public_plans(self):
        """Test listing public plans."""
        from bvrai_core.billing.pricing import PlanCatalog

        catalog = PlanCatalog()

        # Should have default plans
        plans = catalog.list_public_plans()

        assert len(plans) > 0
        assert all(p.is_public for p in plans)

    def test_get_plan_by_id(self):
        """Test getting plan by ID."""
        from bvrai_core.billing.pricing import PlanCatalog

        catalog = PlanCatalog()

        # Get a known plan
        plan = catalog.get_plan("starter")

        assert plan is not None
        assert plan.id == "starter"

        # Get non-existent plan
        assert catalog.get_plan("non_existent") is None

    def test_compare_plans(self):
        """Test comparing two plans."""
        from bvrai_core.billing.pricing import PlanCatalog

        catalog = PlanCatalog()

        comparison = catalog.compare_plans("starter", "professional")

        assert "features" in comparison
        assert "price_difference" in comparison


class TestUsageTracking:
    """Tests for usage tracking."""

    @pytest.mark.asyncio
    async def test_record_usage(self):
        """Test recording usage."""
        from bvrai_core.billing.usage import UsageTracker, InMemoryUsageStore, UsageMeter
        from bvrai_core.billing.base import UsageType

        store = InMemoryUsageStore()
        meter = UsageMeter(store)
        tracker = UsageTracker(meter, store)

        record = await tracker.record_usage(
            organization_id="org_123",
            usage_type=UsageType.CALL_MINUTES,
            quantity=Decimal("15.5"),
            metadata={"call_id": "call_456"},
        )

        assert record.organization_id == "org_123"
        assert record.usage_type == UsageType.CALL_MINUTES
        assert record.quantity == Decimal("15.5")

    @pytest.mark.asyncio
    async def test_get_usage_summary(self):
        """Test getting usage summary."""
        from bvrai_core.billing.usage import UsageTracker, InMemoryUsageStore, UsageMeter
        from bvrai_core.billing.base import UsageType

        store = InMemoryUsageStore()
        meter = UsageMeter(store)
        tracker = UsageTracker(meter, store)

        # Record some usage
        await tracker.record_usage(
            organization_id="org_123",
            usage_type=UsageType.CALL_MINUTES,
            quantity=Decimal("10"),
        )
        await tracker.record_usage(
            organization_id="org_123",
            usage_type=UsageType.CALL_MINUTES,
            quantity=Decimal("20"),
        )
        await tracker.record_usage(
            organization_id="org_123",
            usage_type=UsageType.API_CALLS,
            quantity=Decimal("100"),
        )

        summary = await tracker.get_usage_summary("org_123")

        assert summary.usage_by_type[UsageType.CALL_MINUTES] == Decimal("30")
        assert summary.usage_by_type[UsageType.API_CALLS] == Decimal("100")

    @pytest.mark.asyncio
    async def test_check_quota(self):
        """Test checking quota limits."""
        from bvrai_core.billing.usage import UsageTracker, InMemoryUsageStore, UsageMeter
        from bvrai_core.billing.base import UsageType
        from bvrai_core.billing.pricing import PlanFeatures

        store = InMemoryUsageStore()
        meter = UsageMeter(store)
        tracker = UsageTracker(meter, store)

        # Set limits
        tracker.set_organization_limits(
            "org_123",
            PlanFeatures(max_calls_per_month=100),
        )

        # Within quota
        allowed, reason = await tracker.check_quota(
            "org_123",
            UsageType.CALLS,
            Decimal("50"),
        )
        assert allowed is True

        # Exceeds quota
        allowed, reason = await tracker.check_quota(
            "org_123",
            UsageType.CALLS,
            Decimal("150"),
        )
        assert allowed is False
        assert "quota" in reason.lower()


class TestInvoiceGeneration:
    """Tests for invoice generation."""

    @pytest.mark.asyncio
    async def test_generate_subscription_invoice(self):
        """Test generating subscription invoice."""
        from bvrai_core.billing.invoice import InvoiceGenerator
        from bvrai_core.billing.pricing import PlanCatalog, PricingCalculator
        from bvrai_core.billing.base import Subscription, SubscriptionStatus

        catalog = PlanCatalog()
        calculator = PricingCalculator(catalog)
        generator = InvoiceGenerator(catalog, calculator)

        subscription = Subscription(
            id="sub_123",
            organization_id="org_123",
            plan_id="starter",
            status=SubscriptionStatus.ACTIVE,
            current_period_start=datetime.utcnow() - timedelta(days=30),
            current_period_end=datetime.utcnow(),
        )

        invoice = await generator.generate_subscription_invoice(
            subscription=subscription,
            usage_summary=None,
        )

        assert invoice.organization_id == "org_123"
        assert invoice.subscription_id == "sub_123"
        assert invoice.total_cents > 0

    def test_invoice_line_items(self):
        """Test invoice line item calculations."""
        from bvrai_core.billing.invoice import InvoiceLineItem

        item = InvoiceLineItem(
            description="Professional Plan (Monthly)",
            quantity=1,
            unit_price_cents=9900,
        )

        assert item.total_cents == 9900

        # Multiple quantity
        item2 = InvoiceLineItem(
            description="Additional Agents",
            quantity=3,
            unit_price_cents=1000,
        )

        assert item2.total_cents == 3000


class TestCreditManagement:
    """Tests for credit management."""

    def test_add_credit(self):
        """Test adding credits."""
        from bvrai_core.billing.pricing import CreditManager
        from bvrai_core.billing.base import CreditType

        manager = CreditManager()

        credit = manager.add_credit(
            organization_id="org_123",
            amount_cents=5000,
            credit_type=CreditType.PROMOTIONAL,
            description="Welcome bonus",
        )

        assert credit.amount_cents == 5000
        assert credit.credit_type == CreditType.PROMOTIONAL

    def test_get_credit_balance(self):
        """Test getting credit balance."""
        from bvrai_core.billing.pricing import CreditManager
        from bvrai_core.billing.base import CreditType

        manager = CreditManager()

        manager.add_credit("org_123", 5000, CreditType.PROMOTIONAL, "Welcome")
        manager.add_credit("org_123", 3000, CreditType.REFERRAL, "Referral bonus")

        balance = manager.get_total_balance("org_123")

        assert balance == 8000

    def test_use_credit(self):
        """Test using credits."""
        from bvrai_core.billing.pricing import CreditManager
        from bvrai_core.billing.base import CreditType

        manager = CreditManager()

        manager.add_credit("org_123", 5000, CreditType.PROMOTIONAL, "Welcome")

        # Use some credit
        used = manager.use_credit("org_123", 2000)
        assert used == 2000

        # Balance should be reduced
        balance = manager.get_total_balance("org_123")
        assert balance == 3000

    def test_credit_expiration(self):
        """Test credit expiration."""
        from bvrai_core.billing.pricing import CreditManager
        from bvrai_core.billing.base import CreditType

        manager = CreditManager()

        # Add credit that expires in -1 days (already expired)
        manager.add_credit(
            "org_123",
            5000,
            CreditType.PROMOTIONAL,
            "Expired promo",
            expires_in_days=-1,
        )

        # Balance should exclude expired credits
        balance = manager.get_total_balance("org_123")
        assert balance == 0


class TestDiscountManagement:
    """Tests for discount management."""

    def test_create_discount(self):
        """Test creating a discount."""
        from bvrai_core.billing.pricing import DiscountManager
        from bvrai_core.billing.base import Discount, DiscountType

        manager = DiscountManager()

        discount = manager.create_discount(
            code="SAVE20",
            discount_type=DiscountType.PERCENTAGE,
            value=20,
            max_uses=100,
        )

        assert discount.code == "SAVE20"
        assert discount.value == 20

    def test_validate_coupon(self):
        """Test validating a coupon code."""
        from bvrai_core.billing.pricing import DiscountManager
        from bvrai_core.billing.base import DiscountType

        manager = DiscountManager()

        manager.create_discount(
            code="VALID20",
            discount_type=DiscountType.PERCENTAGE,
            value=20,
        )

        # Valid coupon
        valid, error = manager.validate_coupon("VALID20")
        assert valid is True

        # Invalid coupon
        valid, error = manager.validate_coupon("INVALID")
        assert valid is False
        assert "not found" in error.lower()

    def test_apply_discount(self):
        """Test applying a discount."""
        from bvrai_core.billing.pricing import DiscountManager, PricingCalculator, PlanCatalog
        from bvrai_core.billing.base import DiscountType

        manager = DiscountManager()
        catalog = PlanCatalog()
        calculator = PricingCalculator(catalog, discount_manager=manager)

        manager.create_discount(
            code="HALF",
            discount_type=DiscountType.PERCENTAGE,
            value=50,
        )

        # Original price
        plan = catalog.get_plan("starter")
        original = plan.price_cents_monthly

        # With discount
        discounted = calculator.calculate_price(plan, discount_code="HALF")

        assert discounted == original * 0.5


class TestPaymentProcessing:
    """Tests for payment processing."""

    @pytest.mark.asyncio
    async def test_mock_payment_processor(self):
        """Test mock payment processor."""
        from bvrai_core.billing.payment import MockPaymentProcessor
        from bvrai_core.billing.base import PaymentStatus

        processor = MockPaymentProcessor()

        result = await processor.charge(
            amount_cents=9900,
            currency="usd",
            payment_method_id="pm_test",
            customer_id="cus_test",
        )

        assert result.status == PaymentStatus.SUCCEEDED
        assert result.amount_cents == 9900

    @pytest.mark.asyncio
    async def test_payment_refund(self):
        """Test payment refund."""
        from bvrai_core.billing.payment import MockPaymentProcessor
        from bvrai_core.billing.base import PaymentStatus

        processor = MockPaymentProcessor()

        # First charge
        payment = await processor.charge(
            amount_cents=5000,
            currency="usd",
            payment_method_id="pm_test",
            customer_id="cus_test",
        )

        # Then refund
        refund = await processor.refund(
            payment_id=payment.id,
            amount_cents=2500,  # Partial refund
        )

        assert refund.status == PaymentStatus.SUCCEEDED
        assert refund.amount_cents == 2500


class TestBillingEngine:
    """Tests for billing engine."""

    @pytest.mark.asyncio
    async def test_create_subscription(self):
        """Test creating a subscription."""
        from bvrai_core.billing.engine import BillingEngine, BillingEngineConfig
        from bvrai_core.billing.base import SubscriptionStatus

        config = BillingEngineConfig()
        engine = BillingEngine(config)

        subscription = await engine.create_subscription(
            organization_id="org_123",
            plan_id="starter",
            trial_days=14,
        )

        assert subscription.organization_id == "org_123"
        assert subscription.plan_id == "starter"
        assert subscription.status == SubscriptionStatus.TRIALING

    @pytest.mark.asyncio
    async def test_record_call_usage(self):
        """Test recording call usage through billing engine."""
        from bvrai_core.billing.engine import BillingEngine, BillingEngineConfig

        config = BillingEngineConfig()
        engine = BillingEngine(config)

        # Start a call
        await engine.record_call_start(
            call_id="call_123",
            organization_id="org_123",
            agent_id="agt_456",
            direction="inbound",
        )

        # End the call
        record = await engine.record_call_end(
            call_id="call_123",
            duration_seconds=180.0,
        )

        assert record is not None
        assert record.quantity == Decimal("3")  # 180 seconds = 3 minutes

    @pytest.mark.asyncio
    async def test_get_billing_dashboard(self):
        """Test getting billing dashboard data."""
        from bvrai_core.billing.engine import BillingEngine, BillingEngineConfig

        config = BillingEngineConfig()
        engine = BillingEngine(config)

        # Create subscription first
        await engine.create_subscription(
            organization_id="org_123",
            plan_id="starter",
        )

        dashboard = await engine.get_billing_dashboard("org_123")

        assert "subscription" in dashboard
        assert "usage" in dashboard
        assert "quotas" in dashboard
        assert "credit_balance_cents" in dashboard
