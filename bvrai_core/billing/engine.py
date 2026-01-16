"""
Billing Engine

Main billing engine that orchestrates all billing operations.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import (
    BillingAlert,
    BillingCustomer,
    BillingError,
    Credit,
    CreditType,
    Discount,
    Invoice,
    InvoiceStatus,
    Payment,
    PaymentMethod,
    PaymentStatus,
    PlanFeatures,
    Subscription,
    SubscriptionStatus,
    UsageRecord,
    UsageSummary,
    UsageType,
)
from .invoice import (
    InMemoryInvoiceStore,
    InvoiceGenerator,
    InvoiceProcessor,
    InvoiceService,
)
from .payment import (
    MockPaymentProcessor,
    PaymentProcessor,
    PaymentService,
    StripeConfig,
    StripePaymentProcessor,
    WebhookHandler,
)
from .pricing import (
    CreditManager,
    DiscountManager,
    PlanCatalog,
    PricingCalculator,
    SubscriptionPlan,
)
from .subscription import (
    InMemorySubscriptionStore,
    SubscriptionBillingService,
    SubscriptionManager,
)
from .usage import (
    CallUsageTracker,
    InMemoryUsageStore,
    UsageMeter,
    UsageTracker,
)


logger = logging.getLogger(__name__)


@dataclass
class BillingEngineConfig:
    """Configuration for the billing engine."""

    # Stripe configuration
    stripe_api_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    stripe_publishable_key: Optional[str] = None

    # Usage tracking
    usage_flush_interval_seconds: float = 10.0
    usage_buffer_size: int = 1000

    # Invoice settings
    auto_finalize_invoices: bool = True
    invoice_due_days: int = 30
    send_invoice_reminders: bool = True
    reminder_days_before_due: List[int] = field(default_factory=lambda: [7, 3, 1])

    # Tax settings
    default_tax_rate: Decimal = Decimal("0")
    tax_rates_by_region: Dict[str, Decimal] = field(default_factory=dict)

    # Retry settings
    payment_retry_attempts: int = 3
    payment_retry_delay_hours: int = 24

    # Background tasks
    enable_background_tasks: bool = True
    renewal_check_interval_seconds: float = 3600.0  # 1 hour
    past_due_check_interval_seconds: float = 86400.0  # 24 hours


class BillingEngine:
    """
    Main billing engine for the platform.

    Orchestrates all billing operations including:
    - Subscription management
    - Usage tracking and metering
    - Invoice generation and processing
    - Payment processing
    - Credits and discounts
    """

    def __init__(self, config: BillingEngineConfig):
        """Initialize billing engine."""
        self._config = config
        self._running = False

        # Initialize components
        self._init_components()

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Customer cache
        self._customers: Dict[str, BillingCustomer] = {}

    def _init_components(self) -> None:
        """Initialize all billing components."""
        # Pricing
        self._catalog = PlanCatalog()
        self._calculator = PricingCalculator(self._catalog)
        self._credit_manager = CreditManager()
        self._discount_manager = DiscountManager()

        # Set tax rates
        for region, rate in self._config.tax_rates_by_region.items():
            parts = region.split(":")
            if len(parts) == 2:
                self._calculator.set_tax_rate(parts[0], parts[1], rate)
            else:
                self._calculator.set_tax_rate(region, None, rate)

        # Usage tracking
        self._usage_store = InMemoryUsageStore()
        self._usage_meter = UsageMeter(
            self._usage_store,
            flush_interval_seconds=self._config.usage_flush_interval_seconds,
            max_buffer_size=self._config.usage_buffer_size,
        )
        self._usage_tracker = UsageTracker(self._usage_meter, self._usage_store)
        self._call_tracker = CallUsageTracker(self._usage_tracker)

        # Subscriptions
        self._subscription_store = InMemorySubscriptionStore()
        self._subscription_manager = SubscriptionManager(
            self._subscription_store,
            self._catalog,
            self._calculator,
        )
        self._subscription_billing = SubscriptionBillingService(
            self._subscription_manager,
            self._catalog,
            self._calculator,
        )

        # Invoices
        self._invoice_store = InMemoryInvoiceStore()
        self._invoice_generator = InvoiceGenerator(
            self._catalog,
            self._calculator,
            self._credit_manager,
            self._discount_manager,
        )
        self._invoice_processor = InvoiceProcessor(self._invoice_store)
        self._invoice_service = InvoiceService(
            self._invoice_generator,
            self._invoice_processor,
            self._invoice_store,
        )

        # Payments
        if self._config.stripe_api_key:
            stripe_config = StripeConfig(
                api_key=self._config.stripe_api_key,
                webhook_secret=self._config.stripe_webhook_secret or "",
                publishable_key=self._config.stripe_publishable_key or "",
            )
            self._payment_processor: PaymentProcessor = StripePaymentProcessor(stripe_config)
        else:
            self._payment_processor = MockPaymentProcessor()

        self._webhook_handler = WebhookHandler()
        self._payment_service = PaymentService(self._payment_processor, self._webhook_handler)

    async def start(self) -> None:
        """Start the billing engine."""
        if self._running:
            return

        self._running = True

        # Start usage meter
        await self._usage_meter.start()

        # Start background tasks
        if self._config.enable_background_tasks:
            self._background_tasks.append(
                asyncio.create_task(self._renewal_check_loop())
            )
            self._background_tasks.append(
                asyncio.create_task(self._past_due_check_loop())
            )

        logger.info("Billing engine started")

    async def stop(self) -> None:
        """Stop the billing engine."""
        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop usage meter
        await self._usage_meter.stop()

        logger.info("Billing engine stopped")

    async def _renewal_check_loop(self) -> None:
        """Background loop for checking renewals."""
        while self._running:
            try:
                await asyncio.sleep(self._config.renewal_check_interval_seconds)
                await self._process_renewals()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in renewal check: {e}")

    async def _past_due_check_loop(self) -> None:
        """Background loop for checking past due invoices."""
        while self._running:
            try:
                await asyncio.sleep(self._config.past_due_check_interval_seconds)
                await self._invoice_processor.check_past_due()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in past due check: {e}")

    async def _process_renewals(self) -> None:
        """Process subscription renewals."""
        expiring = await self._subscription_manager.list_expiring_subscriptions(within_days=1)

        for subscription in expiring:
            try:
                # Generate invoice
                usage_summary = await self._usage_tracker.get_usage_summary(
                    subscription.organization_id,
                    subscription.current_period_start,
                    subscription.current_period_end,
                )

                invoice = await self._invoice_service.create_subscription_invoice(
                    organization_id=subscription.organization_id,
                    subscription=subscription,
                    usage_summary=usage_summary,
                )

                # Try to charge
                customer = self._customers.get(subscription.organization_id)
                if customer and subscription.default_payment_method_id:
                    try:
                        payment = await self._payment_service.charge_customer(
                            customer=customer,
                            amount_cents=invoice.total_cents,
                            description=f"Invoice {invoice.invoice_number}",
                            payment_method_id=subscription.default_payment_method_id,
                        )

                        if payment.status == PaymentStatus.SUCCEEDED:
                            await self._invoice_processor.mark_paid(invoice.id, payment)
                            await self._subscription_manager.renew_subscription(subscription.id)
                        else:
                            await self._subscription_manager.mark_past_due(subscription.id)
                    except Exception as e:
                        logger.error(f"Payment failed for subscription {subscription.id}: {e}")
                        await self._subscription_manager.mark_past_due(subscription.id)

            except Exception as e:
                logger.error(f"Error processing renewal for {subscription.id}: {e}")

    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def _emit_event(self, event_type: str, data: Any) -> None:
        """Emit billing event."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    # Customer management

    async def create_customer(
        self,
        organization_id: str,
        email: str,
        name: str,
        **kwargs,
    ) -> BillingCustomer:
        """Create billing customer."""
        customer = await self._payment_service.setup_customer(
            organization_id=organization_id,
            email=email,
            name=name,
            **kwargs,
        )
        self._customers[organization_id] = customer
        return customer

    async def get_customer(self, organization_id: str) -> Optional[BillingCustomer]:
        """Get customer by organization ID."""
        return self._customers.get(organization_id)

    async def add_payment_method(
        self,
        organization_id: str,
        payment_method_token: str,
    ) -> PaymentMethod:
        """Add payment method for customer."""
        customer = self._customers.get(organization_id)
        if not customer:
            raise BillingError(f"Customer not found: {organization_id}")

        return await self._payment_service.add_payment_method(customer, payment_method_token)

    # Subscription management

    async def create_subscription(
        self,
        organization_id: str,
        plan_id: str,
        payment_method_id: Optional[str] = None,
        trial_days: Optional[int] = None,
    ) -> Subscription:
        """Create subscription for organization."""
        subscription = await self._subscription_manager.create_subscription(
            organization_id=organization_id,
            plan_id=plan_id,
            payment_method_id=payment_method_id,
            trial_days=trial_days,
        )

        # Set usage limits from plan
        plan = self._catalog.get_plan(plan_id)
        if plan:
            self._usage_tracker.set_organization_limits(organization_id, plan.features)

        await self._emit_event("subscription.created", subscription)
        return subscription

    async def get_subscription(self, organization_id: str) -> Optional[Subscription]:
        """Get subscription for organization."""
        return await self._subscription_manager.get_organization_subscription(organization_id)

    async def change_plan(
        self,
        organization_id: str,
        new_plan_id: str,
        prorate: bool = True,
        immediate: bool = True,
    ) -> Subscription:
        """Change subscription plan."""
        subscription = await self._subscription_manager.get_organization_subscription(organization_id)
        if not subscription:
            raise BillingError(f"No active subscription for {organization_id}")

        subscription, change = await self._subscription_manager.change_plan(
            subscription_id=subscription.id,
            new_plan_id=new_plan_id,
            prorate=prorate,
            immediate=immediate,
        )

        # Update usage limits
        plan = self._catalog.get_plan(new_plan_id)
        if plan:
            self._usage_tracker.set_organization_limits(organization_id, plan.features)

        # Generate proration invoice if needed
        if immediate and (change.charge_cents - change.credit_cents) > 0:
            invoice = await self._subscription_billing.generate_proration_invoice(
                subscription, change
            )
            if invoice:
                await self._invoice_processor.create(invoice)
                await self._invoice_processor.finalize(invoice.id)

        await self._emit_event("subscription.plan_changed", subscription)
        return subscription

    async def cancel_subscription(
        self,
        organization_id: str,
        at_period_end: bool = True,
        reason: Optional[str] = None,
    ) -> Subscription:
        """Cancel subscription."""
        subscription = await self._subscription_manager.get_organization_subscription(organization_id)
        if not subscription:
            raise BillingError(f"No active subscription for {organization_id}")

        subscription = await self._subscription_manager.cancel_subscription(
            subscription_id=subscription.id,
            at_period_end=at_period_end,
            reason=reason,
        )

        await self._emit_event("subscription.canceled", subscription)
        return subscription

    async def get_plan_features(self, organization_id: str) -> Optional[PlanFeatures]:
        """Get plan features for organization."""
        return await self._subscription_manager.get_plan_features(organization_id)

    # Usage tracking

    async def record_call_start(
        self,
        call_id: str,
        organization_id: str,
        agent_id: str,
        direction: str = "inbound",
    ) -> None:
        """Record call start for billing."""
        await self._call_tracker.start_call(call_id, organization_id, agent_id, direction)

    async def record_call_end(
        self,
        call_id: str,
        duration_seconds: float,
    ) -> UsageRecord:
        """Record call end for billing."""
        return await self._call_tracker.end_call(call_id, duration_seconds)

    async def record_usage(
        self,
        organization_id: str,
        usage_type: UsageType,
        quantity: Decimal,
        **kwargs,
    ) -> UsageRecord:
        """Record arbitrary usage."""
        return await self._usage_tracker.record_usage(
            organization_id=organization_id,
            usage_type=usage_type,
            quantity=quantity,
            **kwargs,
        )

    async def get_usage_summary(
        self,
        organization_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> UsageSummary:
        """Get usage summary for organization."""
        return await self._usage_tracker.get_usage_summary(
            organization_id,
            start_date,
            end_date,
        )

    async def get_quota_status(
        self,
        organization_id: str,
    ) -> Dict[UsageType, Dict[str, Any]]:
        """Get quota status for organization."""
        return await self._usage_tracker.get_quota_status(organization_id)

    async def check_quota(
        self,
        organization_id: str,
        usage_type: UsageType,
        quantity: Decimal = Decimal("1"),
    ) -> Tuple[bool, Optional[str]]:
        """Check if usage is within quota."""
        return await self._usage_tracker.check_quota(organization_id, usage_type, quantity)

    # Invoice management

    async def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        """Get invoice by ID."""
        return await self._invoice_service.get_invoice(invoice_id)

    async def list_invoices(
        self,
        organization_id: str,
        limit: int = 50,
    ) -> List[Invoice]:
        """List invoices for organization."""
        return await self._invoice_service.list_invoices(organization_id, limit)

    async def get_organization_balance(
        self,
        organization_id: str,
    ) -> Dict[str, Any]:
        """Get billing balance for organization."""
        return await self._invoice_service.get_organization_balance(organization_id)

    # Credits and discounts

    async def add_credit(
        self,
        organization_id: str,
        amount_cents: int,
        credit_type: CreditType,
        description: str,
        expires_in_days: Optional[int] = None,
    ) -> Credit:
        """Add billing credit to organization."""
        return self._credit_manager.add_credit(
            organization_id=organization_id,
            amount_cents=amount_cents,
            credit_type=credit_type,
            description=description,
            expires_in_days=expires_in_days,
        )

    async def get_credit_balance(self, organization_id: str) -> int:
        """Get credit balance for organization."""
        return self._credit_manager.get_total_balance(organization_id)

    async def validate_coupon(self, coupon_code: str) -> Tuple[bool, Optional[str]]:
        """Validate a coupon code."""
        return self._discount_manager.validate_coupon(coupon_code)

    async def apply_coupon(
        self,
        organization_id: str,
        coupon_code: str,
    ) -> Optional[Discount]:
        """Apply coupon to organization."""
        discount = self._discount_manager.get_by_coupon(coupon_code)
        if discount and discount.is_valid():
            # Store discount association with organization
            return discount
        return None

    # Pricing

    def get_plans(self) -> List[SubscriptionPlan]:
        """Get available subscription plans."""
        return self._catalog.list_public_plans()

    def get_plan(self, plan_id: str) -> Optional[SubscriptionPlan]:
        """Get plan by ID."""
        return self._catalog.get_plan(plan_id)

    async def estimate_cost(
        self,
        plan_id: str,
        estimated_usage: Dict[UsageType, int],
    ) -> int:
        """Estimate monthly cost for plan and usage."""
        plan = self._catalog.get_plan(plan_id)
        if not plan:
            return 0
        return self._calculator.estimate_monthly_cost(plan, estimated_usage)

    async def recommend_plan(
        self,
        estimated_usage: Dict[UsageType, int],
    ) -> Optional[SubscriptionPlan]:
        """Recommend best plan for estimated usage."""
        return self._calculator.recommend_plan(estimated_usage)

    # Webhooks

    async def process_webhook(
        self,
        payload: bytes,
        signature: str,
    ) -> Tuple[bool, Optional[str]]:
        """Process payment processor webhook."""
        return await self._payment_service.process_webhook(payload, signature)

    # Dashboard data

    async def get_billing_dashboard(
        self,
        organization_id: str,
    ) -> Dict[str, Any]:
        """Get billing dashboard data for organization."""
        subscription = await self.get_subscription(organization_id)
        balance = await self.get_organization_balance(organization_id)
        usage = await self.get_usage_summary(organization_id)
        quotas = await self.get_quota_status(organization_id)
        credits = await self.get_credit_balance(organization_id)
        invoices = await self.list_invoices(organization_id, limit=5)

        plan = None
        if subscription:
            plan = self._catalog.get_plan(subscription.plan_id)

        return {
            "subscription": {
                "id": subscription.id if subscription else None,
                "status": subscription.status.value if subscription else None,
                "plan_id": subscription.plan_id if subscription else None,
                "plan_name": plan.name if plan else None,
                "current_period_end": subscription.current_period_end.isoformat() if subscription else None,
                "days_until_renewal": subscription.days_until_renewal() if subscription else None,
                "cancel_at_period_end": subscription.cancel_at_period_end if subscription else False,
            } if subscription else None,
            "balance": balance,
            "usage": {
                "period_start": usage.period_start.isoformat(),
                "period_end": usage.period_end.isoformat(),
                "by_type": {
                    ut.value: float(qty) for ut, qty in usage.usage_by_type.items()
                },
            },
            "quotas": {
                ut.value: status for ut, status in quotas.items()
            },
            "credit_balance_cents": credits,
            "recent_invoices": [
                {
                    "id": inv.id,
                    "number": inv.invoice_number,
                    "status": inv.status.value,
                    "total_cents": inv.total_cents,
                    "created_at": inv.created_at.isoformat(),
                }
                for inv in invoices
            ],
        }


def create_billing_engine(
    stripe_api_key: Optional[str] = None,
    stripe_webhook_secret: Optional[str] = None,
    **kwargs,
) -> BillingEngine:
    """Create billing engine with configuration."""
    config = BillingEngineConfig(
        stripe_api_key=stripe_api_key,
        stripe_webhook_secret=stripe_webhook_secret,
        **kwargs,
    )
    return BillingEngine(config)
