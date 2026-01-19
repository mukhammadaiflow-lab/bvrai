"""
Payment Processing

Payment processing with Stripe integration and webhook handling.
"""

import asyncio
import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import (
    BillingCustomer,
    BillingError,
    Invoice,
    InvoiceLineItem,
    InvoiceStatus,
    Payment,
    PaymentError,
    PaymentMethod,
    PaymentProcessor,
    PaymentStatus,
    Subscription,
    SubscriptionPlan,
)


logger = logging.getLogger(__name__)


@dataclass
class StripeConfig:
    """Stripe configuration."""

    api_key: str
    webhook_secret: str
    publishable_key: str = ""
    api_version: str = "2023-10-16"
    connect_enabled: bool = False


class StripePaymentProcessor(PaymentProcessor):
    """
    Stripe payment processor implementation.

    Handles all Stripe API interactions for billing.
    """

    def __init__(self, config: StripeConfig):
        """Initialize Stripe processor."""
        self._config = config
        self._stripe = None

    def _get_stripe(self):
        """Get Stripe client (lazy load)."""
        if self._stripe is None:
            try:
                import stripe
                stripe.api_key = self._config.api_key
                stripe.api_version = self._config.api_version
                self._stripe = stripe
            except ImportError:
                raise BillingError("stripe package not installed")
        return self._stripe

    async def create_customer(self, customer: BillingCustomer) -> str:
        """Create customer in Stripe."""
        stripe = self._get_stripe()

        try:
            stripe_customer = stripe.Customer.create(
                email=customer.email,
                name=customer.name,
                address={
                    "line1": customer.address_line1,
                    "line2": customer.address_line2,
                    "city": customer.city,
                    "state": customer.state,
                    "postal_code": customer.postal_code,
                    "country": customer.country,
                } if customer.address_line1 else None,
                metadata={
                    "organization_id": customer.organization_id,
                    "customer_id": customer.id,
                },
            )
            return stripe_customer.id
        except stripe.error.StripeError as e:
            raise PaymentError(f"Failed to create Stripe customer: {str(e)}")

    async def update_customer(self, customer: BillingCustomer) -> None:
        """Update customer in Stripe."""
        if not customer.stripe_customer_id:
            return

        stripe = self._get_stripe()

        try:
            stripe.Customer.modify(
                customer.stripe_customer_id,
                email=customer.email,
                name=customer.name,
                address={
                    "line1": customer.address_line1,
                    "line2": customer.address_line2,
                    "city": customer.city,
                    "state": customer.state,
                    "postal_code": customer.postal_code,
                    "country": customer.country,
                } if customer.address_line1 else None,
            )
        except stripe.error.StripeError as e:
            raise PaymentError(f"Failed to update Stripe customer: {str(e)}")

    async def delete_customer(self, customer_id: str) -> None:
        """Delete customer from Stripe."""
        stripe = self._get_stripe()

        try:
            stripe.Customer.delete(customer_id)
        except stripe.error.StripeError as e:
            logger.warning(f"Failed to delete Stripe customer {customer_id}: {e}")

    async def create_payment_method(
        self,
        customer_id: str,
        payment_method_token: str,
    ) -> PaymentMethod:
        """Attach payment method to customer."""
        import uuid

        stripe = self._get_stripe()

        try:
            # Attach payment method to customer
            payment_method = stripe.PaymentMethod.attach(
                payment_method_token,
                customer=customer_id,
            )

            # Set as default
            stripe.Customer.modify(
                customer_id,
                invoice_settings={
                    "default_payment_method": payment_method.id,
                },
            )

            return PaymentMethod(
                id=f"pm_{uuid.uuid4().hex[:12]}",
                organization_id="",  # Will be set by caller
                type=payment_method.type,
                is_default=True,
                card_brand=payment_method.card.brand if payment_method.card else None,
                card_last4=payment_method.card.last4 if payment_method.card else None,
                card_exp_month=payment_method.card.exp_month if payment_method.card else None,
                card_exp_year=payment_method.card.exp_year if payment_method.card else None,
                stripe_payment_method_id=payment_method.id,
            )
        except stripe.error.CardError as e:
            raise PaymentError(
                f"Card error: {str(e)}",
                decline_code=e.code if hasattr(e, 'code') else None,
            )
        except stripe.error.StripeError as e:
            raise PaymentError(f"Failed to create payment method: {str(e)}")

    async def delete_payment_method(self, payment_method_id: str) -> None:
        """Detach payment method."""
        stripe = self._get_stripe()

        try:
            stripe.PaymentMethod.detach(payment_method_id)
        except stripe.error.StripeError as e:
            logger.warning(f"Failed to delete payment method {payment_method_id}: {e}")

    async def create_subscription(
        self,
        customer_id: str,
        plan: SubscriptionPlan,
        payment_method_id: Optional[str] = None,
        trial_days: Optional[int] = None,
    ) -> Subscription:
        """Create subscription in Stripe."""
        import uuid

        stripe = self._get_stripe()

        try:
            # Create price if needed or use existing
            items = [{
                "price_data": {
                    "currency": plan.currency,
                    "product_data": {
                        "name": plan.name,
                        "metadata": {"plan_id": plan.plan_id},
                    },
                    "unit_amount": plan.base_price_cents,
                    "recurring": {
                        "interval": "month" if plan.billing_period.value == "monthly" else "year",
                    },
                },
            }]

            subscription_params = {
                "customer": customer_id,
                "items": items,
                "metadata": {"plan_id": plan.plan_id},
            }

            if payment_method_id:
                subscription_params["default_payment_method"] = payment_method_id

            use_trial = trial_days or plan.trial_days
            if use_trial > 0:
                subscription_params["trial_period_days"] = use_trial

            stripe_sub = stripe.Subscription.create(**subscription_params)

            from .base import SubscriptionStatus

            status_map = {
                "trialing": SubscriptionStatus.TRIALING,
                "active": SubscriptionStatus.ACTIVE,
                "past_due": SubscriptionStatus.PAST_DUE,
                "canceled": SubscriptionStatus.CANCELED,
            }

            return Subscription(
                id=f"sub_{uuid.uuid4().hex[:16]}",
                organization_id="",  # Set by caller
                plan_id=plan.plan_id,
                status=status_map.get(stripe_sub.status, SubscriptionStatus.ACTIVE),
                current_period_start=datetime.fromtimestamp(stripe_sub.current_period_start),
                current_period_end=datetime.fromtimestamp(stripe_sub.current_period_end),
                trial_start=datetime.fromtimestamp(stripe_sub.trial_start) if stripe_sub.trial_start else None,
                trial_end=datetime.fromtimestamp(stripe_sub.trial_end) if stripe_sub.trial_end else None,
                stripe_subscription_id=stripe_sub.id,
                stripe_customer_id=customer_id,
            )
        except stripe.error.StripeError as e:
            raise PaymentError(f"Failed to create subscription: {str(e)}")

    async def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True,
    ) -> Subscription:
        """Cancel Stripe subscription."""
        stripe = self._get_stripe()

        try:
            if at_period_end:
                stripe_sub = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True,
                )
            else:
                stripe_sub = stripe.Subscription.delete(subscription_id)

            from .base import SubscriptionStatus

            return Subscription(
                id="",
                organization_id="",
                plan_id="",
                status=SubscriptionStatus.CANCELED,
                current_period_start=datetime.fromtimestamp(stripe_sub.current_period_start),
                current_period_end=datetime.fromtimestamp(stripe_sub.current_period_end),
                canceled_at=datetime.utcnow(),
                stripe_subscription_id=stripe_sub.id,
            )
        except stripe.error.StripeError as e:
            raise PaymentError(f"Failed to cancel subscription: {str(e)}")

    async def charge_payment(
        self,
        customer_id: str,
        amount_cents: int,
        currency: str,
        payment_method_id: str,
        description: str,
    ) -> Payment:
        """Charge payment to customer."""
        import uuid

        stripe = self._get_stripe()

        try:
            payment_intent = stripe.PaymentIntent.create(
                amount=amount_cents,
                currency=currency,
                customer=customer_id,
                payment_method=payment_method_id,
                description=description,
                confirm=True,
                off_session=True,
            )

            return Payment(
                id=f"pay_{uuid.uuid4().hex[:16]}",
                organization_id="",
                invoice_id="",
                payment_method_id=payment_method_id,
                status=PaymentStatus.SUCCEEDED if payment_intent.status == "succeeded" else PaymentStatus.PENDING,
                amount_cents=amount_cents,
                currency=currency,
                stripe_payment_intent_id=payment_intent.id,
                stripe_charge_id=payment_intent.latest_charge,
                processed_at=datetime.utcnow() if payment_intent.status == "succeeded" else None,
            )
        except stripe.error.CardError as e:
            return Payment(
                id=f"pay_{uuid.uuid4().hex[:16]}",
                organization_id="",
                invoice_id="",
                payment_method_id=payment_method_id,
                status=PaymentStatus.FAILED,
                amount_cents=amount_cents,
                currency=currency,
                failure_reason=str(e),
            )
        except stripe.error.StripeError as e:
            raise PaymentError(f"Failed to charge payment: {str(e)}")

    async def refund_payment(
        self,
        payment_id: str,
        amount_cents: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> Payment:
        """Refund a payment."""
        import uuid

        stripe = self._get_stripe()

        try:
            refund_params = {"payment_intent": payment_id}
            if amount_cents:
                refund_params["amount"] = amount_cents
            if reason:
                refund_params["reason"] = reason

            refund = stripe.Refund.create(**refund_params)

            return Payment(
                id=f"pay_{uuid.uuid4().hex[:16]}",
                organization_id="",
                invoice_id="",
                payment_method_id="",
                status=PaymentStatus.REFUNDED,
                amount_cents=refund.amount,
                currency=refund.currency,
                refund_reason=reason,
                refunded_amount_cents=refund.amount,
                refunded_at=datetime.utcnow(),
            )
        except stripe.error.StripeError as e:
            raise PaymentError(f"Failed to refund payment: {str(e)}")

    async def create_invoice(
        self,
        customer_id: str,
        line_items: List[InvoiceLineItem],
        auto_charge: bool = True,
    ) -> Invoice:
        """Create invoice in Stripe."""
        import uuid

        stripe = self._get_stripe()

        try:
            # Create invoice
            stripe_invoice = stripe.Invoice.create(
                customer=customer_id,
                auto_advance=auto_charge,
            )

            # Add line items
            for item in line_items:
                stripe.InvoiceItem.create(
                    customer=customer_id,
                    invoice=stripe_invoice.id,
                    description=item.description,
                    amount=item.total_cents,
                    currency="usd",
                )

            # Finalize and optionally pay
            stripe_invoice = stripe.Invoice.finalize_invoice(stripe_invoice.id)

            if auto_charge:
                try:
                    stripe_invoice = stripe.Invoice.pay(stripe_invoice.id)
                except stripe.error.CardError:
                    pass  # Will be handled by webhook

            status_map = {
                "draft": InvoiceStatus.DRAFT,
                "open": InvoiceStatus.OPEN,
                "paid": InvoiceStatus.PAID,
                "void": InvoiceStatus.VOID,
                "uncollectible": InvoiceStatus.UNCOLLECTIBLE,
            }

            return Invoice(
                id=f"inv_{uuid.uuid4().hex[:16]}",
                invoice_number=stripe_invoice.number or "",
                organization_id="",
                subscription_id=None,
                status=status_map.get(stripe_invoice.status, InvoiceStatus.OPEN),
                currency=stripe_invoice.currency,
                subtotal_cents=stripe_invoice.subtotal,
                discount_cents=stripe_invoice.total_discount_amounts[0].amount if stripe_invoice.total_discount_amounts else 0,
                credit_applied_cents=0,
                tax_cents=stripe_invoice.tax or 0,
                total_cents=stripe_invoice.total,
                amount_due_cents=stripe_invoice.amount_due,
                amount_paid_cents=stripe_invoice.amount_paid,
                line_items=line_items,
                stripe_invoice_id=stripe_invoice.id,
                payment_intent_id=stripe_invoice.payment_intent,
                pdf_url=stripe_invoice.invoice_pdf,
            )
        except stripe.error.StripeError as e:
            raise PaymentError(f"Failed to create invoice: {str(e)}")

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify Stripe webhook signature."""
        stripe = self._get_stripe()

        try:
            stripe.Webhook.construct_event(
                payload,
                signature,
                self._config.webhook_secret,
            )
            return True
        except ValueError:
            return False
        except stripe.error.SignatureVerificationError:
            return False


class MockPaymentProcessor(PaymentProcessor):
    """Mock payment processor for testing."""

    def __init__(self):
        """Initialize mock processor."""
        self._customers: Dict[str, BillingCustomer] = {}
        self._payment_methods: Dict[str, PaymentMethod] = {}
        self._subscriptions: Dict[str, Subscription] = {}
        self._payments: Dict[str, Payment] = {}
        self._invoices: Dict[str, Invoice] = {}
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        """Generate next ID."""
        self._counter += 1
        return f"{prefix}_{self._counter:012x}"

    async def create_customer(self, customer: BillingCustomer) -> str:
        """Create mock customer."""
        customer_id = self._next_id("cus")
        customer.stripe_customer_id = customer_id
        self._customers[customer_id] = customer
        return customer_id

    async def update_customer(self, customer: BillingCustomer) -> None:
        """Update mock customer."""
        if customer.stripe_customer_id:
            self._customers[customer.stripe_customer_id] = customer

    async def delete_customer(self, customer_id: str) -> None:
        """Delete mock customer."""
        self._customers.pop(customer_id, None)

    async def create_payment_method(
        self,
        customer_id: str,
        payment_method_token: str,
    ) -> PaymentMethod:
        """Create mock payment method."""
        import uuid

        pm = PaymentMethod(
            id=f"pm_{uuid.uuid4().hex[:12]}",
            organization_id="",
            type="card",
            is_default=True,
            card_brand="visa",
            card_last4="4242",
            card_exp_month=12,
            card_exp_year=2030,
            stripe_payment_method_id=self._next_id("pm"),
        )
        self._payment_methods[pm.stripe_payment_method_id] = pm
        return pm

    async def delete_payment_method(self, payment_method_id: str) -> None:
        """Delete mock payment method."""
        self._payment_methods.pop(payment_method_id, None)

    async def create_subscription(
        self,
        customer_id: str,
        plan: SubscriptionPlan,
        payment_method_id: Optional[str] = None,
        trial_days: Optional[int] = None,
    ) -> Subscription:
        """Create mock subscription."""
        import uuid
        from datetime import timedelta
        from .base import SubscriptionStatus

        now = datetime.utcnow()
        use_trial = trial_days or plan.trial_days

        sub = Subscription(
            id=f"sub_{uuid.uuid4().hex[:16]}",
            organization_id="",
            plan_id=plan.plan_id,
            status=SubscriptionStatus.TRIALING if use_trial > 0 else SubscriptionStatus.ACTIVE,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
            trial_start=now if use_trial > 0 else None,
            trial_end=now + timedelta(days=use_trial) if use_trial > 0 else None,
            stripe_subscription_id=self._next_id("sub"),
            stripe_customer_id=customer_id,
        )
        self._subscriptions[sub.stripe_subscription_id] = sub
        return sub

    async def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True,
    ) -> Subscription:
        """Cancel mock subscription."""
        from .base import SubscriptionStatus

        sub = self._subscriptions.get(subscription_id)
        if sub:
            if at_period_end:
                sub.cancel_at_period_end = True
            else:
                sub.status = SubscriptionStatus.CANCELED
            sub.canceled_at = datetime.utcnow()
        return sub

    async def charge_payment(
        self,
        customer_id: str,
        amount_cents: int,
        currency: str,
        payment_method_id: str,
        description: str,
    ) -> Payment:
        """Create mock payment."""
        import uuid

        payment = Payment(
            id=f"pay_{uuid.uuid4().hex[:16]}",
            organization_id="",
            invoice_id="",
            payment_method_id=payment_method_id,
            status=PaymentStatus.SUCCEEDED,
            amount_cents=amount_cents,
            currency=currency,
            stripe_payment_intent_id=self._next_id("pi"),
            stripe_charge_id=self._next_id("ch"),
            processed_at=datetime.utcnow(),
        )
        self._payments[payment.stripe_payment_intent_id] = payment
        return payment

    async def refund_payment(
        self,
        payment_id: str,
        amount_cents: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> Payment:
        """Create mock refund."""
        import uuid

        payment = self._payments.get(payment_id)
        if payment:
            payment.status = PaymentStatus.REFUNDED
            payment.refunded_amount_cents = amount_cents or payment.amount_cents
            payment.refund_reason = reason
            payment.refunded_at = datetime.utcnow()
        return payment

    async def create_invoice(
        self,
        customer_id: str,
        line_items: List[InvoiceLineItem],
        auto_charge: bool = True,
    ) -> Invoice:
        """Create mock invoice."""
        import uuid

        subtotal = sum(item.total_cents for item in line_items)

        invoice = Invoice(
            id=f"inv_{uuid.uuid4().hex[:16]}",
            invoice_number=f"INV-{datetime.utcnow().strftime('%Y%m%d')}-{self._counter:04d}",
            organization_id="",
            subscription_id=None,
            status=InvoiceStatus.PAID if auto_charge else InvoiceStatus.OPEN,
            currency="usd",
            subtotal_cents=subtotal,
            discount_cents=0,
            credit_applied_cents=0,
            tax_cents=0,
            total_cents=subtotal,
            amount_due_cents=0 if auto_charge else subtotal,
            amount_paid_cents=subtotal if auto_charge else 0,
            line_items=line_items,
            stripe_invoice_id=self._next_id("in"),
            paid_at=datetime.utcnow() if auto_charge else None,
        )
        self._invoices[invoice.stripe_invoice_id] = invoice
        return invoice


@dataclass
class WebhookEvent:
    """Parsed webhook event."""

    event_type: str
    event_id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processed: bool = False


class WebhookHandler:
    """Handler for payment processor webhooks."""

    def __init__(self):
        """Initialize webhook handler."""
        self._handlers: Dict[str, List[Callable]] = {}
        self._processed_events: set = set()

    def register(self, event_type: str, handler: Callable) -> None:
        """Register handler for event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def handle_stripe_webhook(
        self,
        payload: bytes,
        signature: str,
        processor: StripePaymentProcessor,
    ) -> Tuple[bool, Optional[str]]:
        """Handle Stripe webhook."""
        # Verify signature
        if not processor.verify_webhook_signature(payload, signature):
            return (False, "Invalid signature")

        stripe = processor._get_stripe()

        try:
            event = stripe.Event.construct_from(
                stripe.util.json.loads(payload),
                stripe.api_key,
            )
        except Exception as e:
            return (False, f"Invalid payload: {str(e)}")

        # Check for duplicate
        if event.id in self._processed_events:
            return (True, "Duplicate event")

        # Parse event
        webhook_event = WebhookEvent(
            event_type=event.type,
            event_id=event.id,
            data=event.data.object.to_dict() if hasattr(event.data, 'object') else {},
            timestamp=datetime.fromtimestamp(event.created),
        )

        # Handle event
        await self._process_event(webhook_event)

        self._processed_events.add(event.id)

        return (True, None)

    async def _process_event(self, event: WebhookEvent) -> None:
        """Process a webhook event."""
        handlers = self._handlers.get(event.event_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error handling webhook {event.event_type}: {e}")

        # Also call wildcard handlers
        for handler in self._handlers.get("*", []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in wildcard webhook handler: {e}")


class PaymentService:
    """
    High-level payment service.

    Provides convenient methods for common payment operations.
    """

    def __init__(
        self,
        processor: PaymentProcessor,
        webhook_handler: Optional[WebhookHandler] = None,
    ):
        """Initialize payment service."""
        self._processor = processor
        self._webhook_handler = webhook_handler or WebhookHandler()

        # Payment method cache
        self._payment_methods: Dict[str, List[PaymentMethod]] = {}

        # Setup default webhook handlers
        self._setup_webhooks()

    def _setup_webhooks(self) -> None:
        """Setup default webhook handlers."""
        self._webhook_handler.register("invoice.payment_succeeded", self._handle_invoice_paid)
        self._webhook_handler.register("invoice.payment_failed", self._handle_invoice_failed)
        self._webhook_handler.register("customer.subscription.updated", self._handle_subscription_updated)
        self._webhook_handler.register("customer.subscription.deleted", self._handle_subscription_deleted)

    async def _handle_invoice_paid(self, event: WebhookEvent) -> None:
        """Handle invoice paid event."""
        logger.info(f"Invoice paid: {event.data.get('id')}")

    async def _handle_invoice_failed(self, event: WebhookEvent) -> None:
        """Handle invoice payment failed."""
        logger.warning(f"Invoice payment failed: {event.data.get('id')}")

    async def _handle_subscription_updated(self, event: WebhookEvent) -> None:
        """Handle subscription updated."""
        logger.info(f"Subscription updated: {event.data.get('id')}")

    async def _handle_subscription_deleted(self, event: WebhookEvent) -> None:
        """Handle subscription deleted."""
        logger.info(f"Subscription deleted: {event.data.get('id')}")

    async def setup_customer(
        self,
        organization_id: str,
        email: str,
        name: str,
        **kwargs,
    ) -> BillingCustomer:
        """Setup billing customer."""
        import uuid

        customer = BillingCustomer(
            id=f"cust_{uuid.uuid4().hex[:12]}",
            organization_id=organization_id,
            email=email,
            name=name,
            **kwargs,
        )

        stripe_id = await self._processor.create_customer(customer)
        customer.stripe_customer_id = stripe_id

        return customer

    async def add_payment_method(
        self,
        customer: BillingCustomer,
        payment_method_token: str,
    ) -> PaymentMethod:
        """Add payment method to customer."""
        if not customer.stripe_customer_id:
            raise PaymentError("Customer has no Stripe ID")

        pm = await self._processor.create_payment_method(
            customer.stripe_customer_id,
            payment_method_token,
        )
        pm.organization_id = customer.organization_id

        # Cache
        if customer.organization_id not in self._payment_methods:
            self._payment_methods[customer.organization_id] = []
        self._payment_methods[customer.organization_id].append(pm)

        return pm

    async def charge_customer(
        self,
        customer: BillingCustomer,
        amount_cents: int,
        description: str,
        payment_method_id: Optional[str] = None,
    ) -> Payment:
        """Charge a customer."""
        if not customer.stripe_customer_id:
            raise PaymentError("Customer has no Stripe ID")

        # Get default payment method if not specified
        if not payment_method_id:
            methods = self._payment_methods.get(customer.organization_id, [])
            default_methods = [m for m in methods if m.is_default]
            if not default_methods:
                raise PaymentError("No default payment method")
            payment_method_id = default_methods[0].stripe_payment_method_id

        payment = await self._processor.charge_payment(
            customer_id=customer.stripe_customer_id,
            amount_cents=amount_cents,
            currency="usd",
            payment_method_id=payment_method_id,
            description=description,
        )
        payment.organization_id = customer.organization_id

        return payment

    async def process_webhook(
        self,
        payload: bytes,
        signature: str,
    ) -> Tuple[bool, Optional[str]]:
        """Process incoming webhook."""
        if isinstance(self._processor, StripePaymentProcessor):
            return await self._webhook_handler.handle_stripe_webhook(
                payload,
                signature,
                self._processor,
            )
        return (False, "Unsupported processor")
