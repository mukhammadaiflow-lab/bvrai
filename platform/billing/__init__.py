"""
Billing Package

This package provides comprehensive billing capabilities for the
voice agent platform, including:

- Subscription management with plan tiers
- Usage-based metering and tracking
- Invoice generation and processing
- Payment processing with Stripe integration
- Credits and discount management

Example usage:

    from platform.billing import (
        BillingEngine,
        BillingEngineConfig,
        UsageType,
        CreditType,
    )

    # Create billing engine
    config = BillingEngineConfig(
        stripe_api_key="sk_test_...",
        stripe_webhook_secret="whsec_...",
    )
    engine = BillingEngine(config)

    # Start engine
    await engine.start()

    # Create customer
    customer = await engine.create_customer(
        organization_id="org_123",
        email="billing@company.com",
        name="Company Inc",
    )

    # Create subscription
    subscription = await engine.create_subscription(
        organization_id="org_123",
        plan_id="plan_professional",
        trial_days=14,
    )

    # Record usage
    await engine.record_call_start(
        call_id="call_abc",
        organization_id="org_123",
        agent_id="agent_xyz",
    )
    await engine.record_call_end(
        call_id="call_abc",
        duration_seconds=180.5,
    )

    # Get billing dashboard
    dashboard = await engine.get_billing_dashboard("org_123")

    # Stop engine
    await engine.stop()
"""

# Base types
from .base import (
    BillingPeriod,
    PlanTier,
    UsageType,
    PaymentStatus,
    InvoiceStatus,
    SubscriptionStatus,
    CreditType,
    DiscountType,
    PricingTier,
    UsagePricing,
    PlanFeatures,
    SubscriptionPlan,
    UsageRecord,
    UsageSummary,
    Discount,
    Credit,
    InvoiceLineItem,
    Invoice,
    PaymentMethod,
    Payment,
    Subscription,
    BillingCustomer,
    BillingAlert,
    PaymentProcessor,
    UsageStore,
    SubscriptionStore,
    InvoiceStore,
    BillingError,
    PaymentError,
    SubscriptionError,
    UsageLimitError,
    InvoiceError,
)

# Pricing
from .pricing import (
    create_default_call_minute_pricing,
    create_default_usage_pricing,
    create_free_plan,
    create_starter_plan,
    create_professional_plan,
    create_enterprise_plan,
    get_default_plans,
    PlanCatalog,
    PriceQuote,
    PricingCalculator,
    DiscountManager,
    CreditManager,
)

# Usage tracking
from .usage import (
    InMemoryUsageStore,
    UsageMeter,
    UsageTracker,
    UsageQuota,
    UsageThreshold,
    CallUsageTracker,
    ResourceUsageTracker,
    APIUsageTracker,
)

# Subscriptions
from .subscription import (
    SubscriptionEvent,
    InMemorySubscriptionStore,
    SubscriptionChange,
    SubscriptionManager,
    SubscriptionBillingService,
)

# Payments
from .payment import (
    StripeConfig,
    StripePaymentProcessor,
    MockPaymentProcessor,
    WebhookEvent,
    WebhookHandler,
    PaymentService,
)

# Invoices
from .invoice import (
    InMemoryInvoiceStore,
    InvoiceGenerationResult,
    InvoiceGenerator,
    InvoiceProcessor,
    InvoiceService,
    InvoicePDFGenerator,
)

# Engine
from .engine import (
    BillingEngineConfig,
    BillingEngine,
    create_billing_engine,
)


__all__ = [
    # Base types
    "BillingPeriod",
    "PlanTier",
    "UsageType",
    "PaymentStatus",
    "InvoiceStatus",
    "SubscriptionStatus",
    "CreditType",
    "DiscountType",
    "PricingTier",
    "UsagePricing",
    "PlanFeatures",
    "SubscriptionPlan",
    "UsageRecord",
    "UsageSummary",
    "Discount",
    "Credit",
    "InvoiceLineItem",
    "Invoice",
    "PaymentMethod",
    "Payment",
    "Subscription",
    "BillingCustomer",
    "BillingAlert",
    "PaymentProcessor",
    "UsageStore",
    "SubscriptionStore",
    "InvoiceStore",
    "BillingError",
    "PaymentError",
    "SubscriptionError",
    "UsageLimitError",
    "InvoiceError",
    # Pricing
    "create_default_call_minute_pricing",
    "create_default_usage_pricing",
    "create_free_plan",
    "create_starter_plan",
    "create_professional_plan",
    "create_enterprise_plan",
    "get_default_plans",
    "PlanCatalog",
    "PriceQuote",
    "PricingCalculator",
    "DiscountManager",
    "CreditManager",
    # Usage
    "InMemoryUsageStore",
    "UsageMeter",
    "UsageTracker",
    "UsageQuota",
    "UsageThreshold",
    "CallUsageTracker",
    "ResourceUsageTracker",
    "APIUsageTracker",
    # Subscriptions
    "SubscriptionEvent",
    "InMemorySubscriptionStore",
    "SubscriptionChange",
    "SubscriptionManager",
    "SubscriptionBillingService",
    # Payments
    "StripeConfig",
    "StripePaymentProcessor",
    "MockPaymentProcessor",
    "WebhookEvent",
    "WebhookHandler",
    "PaymentService",
    # Invoices
    "InMemoryInvoiceStore",
    "InvoiceGenerationResult",
    "InvoiceGenerator",
    "InvoiceProcessor",
    "InvoiceService",
    "InvoicePDFGenerator",
    # Engine
    "BillingEngineConfig",
    "BillingEngine",
    "create_billing_engine",
]
