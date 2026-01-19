"""
Subscription Management

Subscription lifecycle management, upgrades, downgrades, and renewals.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import (
    BillingCustomer,
    BillingError,
    BillingPeriod,
    Invoice,
    InvoiceLineItem,
    InvoiceStatus,
    PaymentMethod,
    PlanFeatures,
    Subscription,
    SubscriptionError,
    SubscriptionStatus,
    SubscriptionStore,
    UsageType,
)
from .pricing import PlanCatalog, PricingCalculator, SubscriptionPlan


logger = logging.getLogger(__name__)


@dataclass
class SubscriptionEvent:
    """Event related to subscription changes."""

    event_type: str
    subscription_id: str
    organization_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)


class InMemorySubscriptionStore(SubscriptionStore):
    """In-memory subscription store implementation."""

    def __init__(self):
        """Initialize in-memory store."""
        self._subscriptions: Dict[str, Subscription] = {}
        self._by_organization: Dict[str, str] = {}  # org_id -> subscription_id

    async def create(self, subscription: Subscription) -> None:
        """Create subscription."""
        self._subscriptions[subscription.id] = subscription
        self._by_organization[subscription.organization_id] = subscription.id

    async def get(self, subscription_id: str) -> Optional[Subscription]:
        """Get subscription by ID."""
        return self._subscriptions.get(subscription_id)

    async def get_by_organization(self, organization_id: str) -> Optional[Subscription]:
        """Get active subscription for organization."""
        subscription_id = self._by_organization.get(organization_id)
        if subscription_id:
            sub = self._subscriptions.get(subscription_id)
            if sub and sub.status in [
                SubscriptionStatus.ACTIVE,
                SubscriptionStatus.TRIALING,
                SubscriptionStatus.PAST_DUE,
            ]:
                return sub
        return None

    async def update(self, subscription: Subscription) -> None:
        """Update subscription."""
        self._subscriptions[subscription.id] = subscription

    async def list_expiring(self, within_days: int) -> List[Subscription]:
        """List subscriptions expiring within days."""
        cutoff = datetime.utcnow() + timedelta(days=within_days)

        return [
            sub for sub in self._subscriptions.values()
            if sub.status == SubscriptionStatus.ACTIVE
            and sub.current_period_end <= cutoff
        ]

    async def list_by_status(self, status: SubscriptionStatus) -> List[Subscription]:
        """List subscriptions by status."""
        return [
            sub for sub in self._subscriptions.values()
            if sub.status == status
        ]

    async def delete(self, subscription_id: str) -> None:
        """Delete subscription."""
        if subscription_id in self._subscriptions:
            sub = self._subscriptions[subscription_id]
            if sub.organization_id in self._by_organization:
                del self._by_organization[sub.organization_id]
            del self._subscriptions[subscription_id]


@dataclass
class SubscriptionChange:
    """Represents a subscription change request."""

    from_plan_id: str
    to_plan_id: str
    effective_date: datetime
    prorate: bool = True
    credit_cents: int = 0
    charge_cents: int = 0
    reason: Optional[str] = None


class SubscriptionManager:
    """
    Manager for subscription lifecycle.

    Handles subscription creation, upgrades, downgrades,
    cancellations, and renewals.
    """

    def __init__(
        self,
        store: SubscriptionStore,
        catalog: PlanCatalog,
        calculator: PricingCalculator,
    ):
        """Initialize subscription manager."""
        self._store = store
        self._catalog = catalog
        self._calculator = calculator

        # Event handlers
        self._event_handlers: List[Callable] = []

        # Pending changes (scheduled)
        self._pending_changes: Dict[str, SubscriptionChange] = {}

    def add_event_handler(self, handler: Callable) -> None:
        """Add subscription event handler."""
        self._event_handlers.append(handler)

    async def _emit_event(self, event: SubscriptionEvent) -> None:
        """Emit subscription event to handlers."""
        for handler in self._event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    async def create_subscription(
        self,
        organization_id: str,
        plan_id: str,
        payment_method_id: Optional[str] = None,
        trial_days: Optional[int] = None,
        stripe_customer_id: Optional[str] = None,
        stripe_subscription_id: Optional[str] = None,
    ) -> Subscription:
        """Create a new subscription."""
        import uuid

        # Check for existing subscription
        existing = await self._store.get_by_organization(organization_id)
        if existing:
            raise SubscriptionError(
                f"Organization {organization_id} already has an active subscription"
            )

        # Get plan
        plan = self._catalog.get_plan(plan_id)
        if not plan:
            raise SubscriptionError(f"Plan not found: {plan_id}")

        # Calculate dates
        now = datetime.utcnow()
        use_trial = trial_days or plan.trial_days

        if use_trial > 0:
            status = SubscriptionStatus.TRIALING
            trial_start = now
            trial_end = now + timedelta(days=use_trial)
            period_start = trial_end
            period_end = self._calculate_period_end(trial_end, plan.billing_period)
        else:
            status = SubscriptionStatus.ACTIVE
            trial_start = None
            trial_end = None
            period_start = now
            period_end = self._calculate_period_end(now, plan.billing_period)

        subscription = Subscription(
            id=f"sub_{uuid.uuid4().hex[:16]}",
            organization_id=organization_id,
            plan_id=plan_id,
            status=status,
            current_period_start=period_start,
            current_period_end=period_end,
            trial_start=trial_start,
            trial_end=trial_end,
            default_payment_method_id=payment_method_id,
            stripe_customer_id=stripe_customer_id,
            stripe_subscription_id=stripe_subscription_id,
        )

        await self._store.create(subscription)

        await self._emit_event(SubscriptionEvent(
            event_type="subscription.created",
            subscription_id=subscription.id,
            organization_id=organization_id,
            data={
                "plan_id": plan_id,
                "status": status.value,
                "trial_days": use_trial,
            }
        ))

        return subscription

    def _calculate_period_end(
        self,
        start: datetime,
        billing_period: BillingPeriod,
    ) -> datetime:
        """Calculate period end date."""
        if billing_period == BillingPeriod.MONTHLY:
            # Add one month
            month = start.month + 1
            year = start.year
            if month > 12:
                month = 1
                year += 1
            day = min(start.day, 28)  # Safe for all months
            return start.replace(year=year, month=month, day=day)

        elif billing_period == BillingPeriod.QUARTERLY:
            # Add three months
            month = start.month + 3
            year = start.year
            while month > 12:
                month -= 12
                year += 1
            day = min(start.day, 28)
            return start.replace(year=year, month=month, day=day)

        elif billing_period == BillingPeriod.ANNUAL:
            # Add one year
            return start.replace(year=start.year + 1)

        return start + timedelta(days=30)

    async def get_subscription(
        self,
        subscription_id: str,
    ) -> Optional[Subscription]:
        """Get subscription by ID."""
        return await self._store.get(subscription_id)

    async def get_organization_subscription(
        self,
        organization_id: str,
    ) -> Optional[Subscription]:
        """Get subscription for organization."""
        return await self._store.get_by_organization(organization_id)

    async def get_plan_features(
        self,
        organization_id: str,
    ) -> Optional[PlanFeatures]:
        """Get plan features for organization."""
        subscription = await self._store.get_by_organization(organization_id)
        if not subscription:
            return None

        plan = self._catalog.get_plan(subscription.plan_id)
        if not plan:
            return None

        return plan.features

    async def change_plan(
        self,
        subscription_id: str,
        new_plan_id: str,
        prorate: bool = True,
        immediate: bool = False,
    ) -> Tuple[Subscription, SubscriptionChange]:
        """
        Change subscription plan.

        Args:
            subscription_id: Subscription to change
            new_plan_id: New plan ID
            prorate: Whether to prorate charges
            immediate: Apply immediately or at period end

        Returns:
            Tuple of (updated subscription, change details)
        """
        subscription = await self._store.get(subscription_id)
        if not subscription:
            raise SubscriptionError(f"Subscription not found: {subscription_id}")

        if subscription.status not in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]:
            raise SubscriptionError(
                f"Cannot change plan for subscription in {subscription.status.value} status"
            )

        old_plan = self._catalog.get_plan(subscription.plan_id)
        new_plan = self._catalog.get_plan(new_plan_id)

        if not old_plan or not new_plan:
            raise SubscriptionError("Invalid plan")

        # Calculate proration
        credit_cents = 0
        charge_cents = 0

        if prorate and subscription.status == SubscriptionStatus.ACTIVE:
            days_remaining = (subscription.current_period_end - datetime.utcnow()).days
            total_days = (subscription.current_period_end - subscription.current_period_start).days

            credit_cents, charge_cents = self._calculator.calculate_proration(
                old_plan, new_plan, days_remaining, total_days
            )

        change = SubscriptionChange(
            from_plan_id=subscription.plan_id,
            to_plan_id=new_plan_id,
            effective_date=datetime.utcnow() if immediate else subscription.current_period_end,
            prorate=prorate,
            credit_cents=credit_cents,
            charge_cents=charge_cents,
        )

        if immediate:
            subscription.plan_id = new_plan_id
            await self._store.update(subscription)

            await self._emit_event(SubscriptionEvent(
                event_type="subscription.plan_changed",
                subscription_id=subscription.id,
                organization_id=subscription.organization_id,
                data={
                    "from_plan_id": old_plan.plan_id,
                    "to_plan_id": new_plan_id,
                    "credit_cents": credit_cents,
                    "charge_cents": charge_cents,
                }
            ))
        else:
            # Schedule for period end
            self._pending_changes[subscription_id] = change

            await self._emit_event(SubscriptionEvent(
                event_type="subscription.plan_change_scheduled",
                subscription_id=subscription.id,
                organization_id=subscription.organization_id,
                data={
                    "from_plan_id": old_plan.plan_id,
                    "to_plan_id": new_plan_id,
                    "effective_date": change.effective_date.isoformat(),
                }
            ))

        return subscription, change

    async def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True,
        reason: Optional[str] = None,
    ) -> Subscription:
        """Cancel subscription."""
        subscription = await self._store.get(subscription_id)
        if not subscription:
            raise SubscriptionError(f"Subscription not found: {subscription_id}")

        if subscription.status in [SubscriptionStatus.CANCELED, SubscriptionStatus.EXPIRED]:
            raise SubscriptionError("Subscription is already canceled")

        subscription.canceled_at = datetime.utcnow()

        if at_period_end:
            subscription.cancel_at_period_end = True
        else:
            subscription.status = SubscriptionStatus.CANCELED
            subscription.ended_at = datetime.utcnow()

        await self._store.update(subscription)

        await self._emit_event(SubscriptionEvent(
            event_type="subscription.canceled",
            subscription_id=subscription.id,
            organization_id=subscription.organization_id,
            data={
                "at_period_end": at_period_end,
                "reason": reason,
            }
        ))

        return subscription

    async def reactivate_subscription(
        self,
        subscription_id: str,
    ) -> Subscription:
        """Reactivate a canceled subscription."""
        subscription = await self._store.get(subscription_id)
        if not subscription:
            raise SubscriptionError(f"Subscription not found: {subscription_id}")

        if subscription.status == SubscriptionStatus.EXPIRED:
            raise SubscriptionError("Cannot reactivate expired subscription")

        if not subscription.cancel_at_period_end:
            raise SubscriptionError("Subscription is not scheduled for cancellation")

        subscription.cancel_at_period_end = False
        subscription.canceled_at = None

        await self._store.update(subscription)

        await self._emit_event(SubscriptionEvent(
            event_type="subscription.reactivated",
            subscription_id=subscription.id,
            organization_id=subscription.organization_id,
        ))

        return subscription

    async def pause_subscription(
        self,
        subscription_id: str,
        resume_date: Optional[datetime] = None,
    ) -> Subscription:
        """Pause subscription."""
        subscription = await self._store.get(subscription_id)
        if not subscription:
            raise SubscriptionError(f"Subscription not found: {subscription_id}")

        if subscription.status != SubscriptionStatus.ACTIVE:
            raise SubscriptionError("Only active subscriptions can be paused")

        subscription.status = SubscriptionStatus.PAUSED
        subscription.metadata["paused_at"] = datetime.utcnow().isoformat()
        if resume_date:
            subscription.metadata["resume_date"] = resume_date.isoformat()

        await self._store.update(subscription)

        await self._emit_event(SubscriptionEvent(
            event_type="subscription.paused",
            subscription_id=subscription.id,
            organization_id=subscription.organization_id,
            data={"resume_date": resume_date.isoformat() if resume_date else None}
        ))

        return subscription

    async def resume_subscription(
        self,
        subscription_id: str,
    ) -> Subscription:
        """Resume a paused subscription."""
        subscription = await self._store.get(subscription_id)
        if not subscription:
            raise SubscriptionError(f"Subscription not found: {subscription_id}")

        if subscription.status != SubscriptionStatus.PAUSED:
            raise SubscriptionError("Subscription is not paused")

        subscription.status = SubscriptionStatus.ACTIVE
        subscription.metadata.pop("paused_at", None)
        subscription.metadata.pop("resume_date", None)

        await self._store.update(subscription)

        await self._emit_event(SubscriptionEvent(
            event_type="subscription.resumed",
            subscription_id=subscription.id,
            organization_id=subscription.organization_id,
        ))

        return subscription

    async def renew_subscription(
        self,
        subscription_id: str,
    ) -> Subscription:
        """Renew subscription for next period."""
        subscription = await self._store.get(subscription_id)
        if not subscription:
            raise SubscriptionError(f"Subscription not found: {subscription_id}")

        if subscription.cancel_at_period_end:
            subscription.status = SubscriptionStatus.CANCELED
            subscription.ended_at = subscription.current_period_end
            await self._store.update(subscription)

            await self._emit_event(SubscriptionEvent(
                event_type="subscription.expired",
                subscription_id=subscription.id,
                organization_id=subscription.organization_id,
            ))

            return subscription

        # Apply pending plan change if any
        pending_change = self._pending_changes.pop(subscription_id, None)
        if pending_change:
            subscription.plan_id = pending_change.to_plan_id

        plan = self._catalog.get_plan(subscription.plan_id)
        if not plan:
            raise SubscriptionError(f"Plan not found: {subscription.plan_id}")

        # Update period
        subscription.current_period_start = subscription.current_period_end
        subscription.current_period_end = self._calculate_period_end(
            subscription.current_period_start,
            plan.billing_period,
        )

        # Reset usage
        subscription.current_usage = {}

        # Update status if coming from trial
        if subscription.status == SubscriptionStatus.TRIALING:
            subscription.status = SubscriptionStatus.ACTIVE

        await self._store.update(subscription)

        await self._emit_event(SubscriptionEvent(
            event_type="subscription.renewed",
            subscription_id=subscription.id,
            organization_id=subscription.organization_id,
            data={
                "period_start": subscription.current_period_start.isoformat(),
                "period_end": subscription.current_period_end.isoformat(),
            }
        ))

        return subscription

    async def process_trial_ending(
        self,
        subscription_id: str,
    ) -> Subscription:
        """Process trial ending - convert to paid or cancel."""
        subscription = await self._store.get(subscription_id)
        if not subscription:
            raise SubscriptionError(f"Subscription not found: {subscription_id}")

        if subscription.status != SubscriptionStatus.TRIALING:
            raise SubscriptionError("Subscription is not in trial")

        if subscription.default_payment_method_id:
            # Has payment method - activate
            subscription.status = SubscriptionStatus.ACTIVE
            await self._emit_event(SubscriptionEvent(
                event_type="subscription.trial_converted",
                subscription_id=subscription.id,
                organization_id=subscription.organization_id,
            ))
        else:
            # No payment method - expire
            subscription.status = SubscriptionStatus.EXPIRED
            subscription.ended_at = datetime.utcnow()
            await self._emit_event(SubscriptionEvent(
                event_type="subscription.trial_expired",
                subscription_id=subscription.id,
                organization_id=subscription.organization_id,
            ))

        await self._store.update(subscription)
        return subscription

    async def mark_past_due(
        self,
        subscription_id: str,
    ) -> Subscription:
        """Mark subscription as past due."""
        subscription = await self._store.get(subscription_id)
        if not subscription:
            raise SubscriptionError(f"Subscription not found: {subscription_id}")

        subscription.status = SubscriptionStatus.PAST_DUE

        await self._store.update(subscription)

        await self._emit_event(SubscriptionEvent(
            event_type="subscription.past_due",
            subscription_id=subscription.id,
            organization_id=subscription.organization_id,
        ))

        return subscription

    async def list_expiring_subscriptions(
        self,
        within_days: int = 7,
    ) -> List[Subscription]:
        """List subscriptions expiring soon."""
        return await self._store.list_expiring(within_days)

    async def list_ending_trials(
        self,
        within_days: int = 3,
    ) -> List[Subscription]:
        """List trials ending soon."""
        cutoff = datetime.utcnow() + timedelta(days=within_days)

        if isinstance(self._store, InMemorySubscriptionStore):
            return [
                sub for sub in (await self._store.list_by_status(SubscriptionStatus.TRIALING))
                if sub.trial_end and sub.trial_end <= cutoff
            ]
        return []


class SubscriptionBillingService:
    """
    Service for subscription billing operations.

    Handles generating invoices for subscriptions.
    """

    def __init__(
        self,
        subscription_manager: SubscriptionManager,
        catalog: PlanCatalog,
        calculator: PricingCalculator,
    ):
        """Initialize billing service."""
        self._subscriptions = subscription_manager
        self._catalog = catalog
        self._calculator = calculator

    async def generate_renewal_invoice(
        self,
        subscription: Subscription,
    ) -> Invoice:
        """Generate invoice for subscription renewal."""
        import uuid

        plan = self._catalog.get_plan(subscription.plan_id)
        if not plan:
            raise SubscriptionError(f"Plan not found: {subscription.plan_id}")

        # Base subscription charge
        line_items = [
            InvoiceLineItem(
                id=f"li_{uuid.uuid4().hex[:12]}",
                description=f"{plan.name} Subscription ({plan.billing_period.value})",
                quantity=Decimal("1"),
                unit_price_cents=plan.base_price_cents,
                total_cents=plan.base_price_cents,
                period_start=subscription.current_period_start,
                period_end=subscription.current_period_end,
            )
        ]

        # Calculate usage charges
        for usage_type, quantity in subscription.current_usage.items():
            if quantity > 0:
                cost = self._calculator.calculate_usage_cost(
                    plan, usage_type, int(quantity)
                )
                if cost > 0:
                    pricing = plan.usage_pricing.get(usage_type)
                    included = pricing.included_units if pricing else 0
                    billable = max(0, int(quantity) - included)

                    if billable > 0:
                        line_items.append(InvoiceLineItem(
                            id=f"li_{uuid.uuid4().hex[:12]}",
                            description=f"{usage_type.value.replace('_', ' ').title()} Overage",
                            quantity=Decimal(str(billable)),
                            unit_price_cents=cost // billable if billable > 0 else 0,
                            total_cents=cost,
                            usage_type=usage_type,
                            period_start=subscription.current_period_start,
                            period_end=subscription.current_period_end,
                        ))

        subtotal = sum(item.total_cents for item in line_items)

        invoice = Invoice(
            id=f"inv_{uuid.uuid4().hex[:16]}",
            invoice_number=f"INV-{datetime.utcnow().strftime('%Y%m')}-{uuid.uuid4().hex[:6].upper()}",
            organization_id=subscription.organization_id,
            subscription_id=subscription.id,
            status=InvoiceStatus.DRAFT,
            currency=plan.currency,
            subtotal_cents=subtotal,
            discount_cents=0,
            credit_applied_cents=0,
            tax_cents=0,
            total_cents=subtotal,
            amount_due_cents=subtotal,
            amount_paid_cents=0,
            line_items=line_items,
            period_start=subscription.current_period_start,
            period_end=subscription.current_period_end,
        )

        return invoice

    async def generate_proration_invoice(
        self,
        subscription: Subscription,
        change: SubscriptionChange,
    ) -> Optional[Invoice]:
        """Generate proration invoice for plan change."""
        import uuid

        net_charge = change.charge_cents - change.credit_cents

        if net_charge <= 0:
            return None  # Credit will be applied to next invoice

        old_plan = self._catalog.get_plan(change.from_plan_id)
        new_plan = self._catalog.get_plan(change.to_plan_id)

        if not old_plan or not new_plan:
            raise SubscriptionError("Invalid plan")

        line_items = []

        if change.credit_cents > 0:
            line_items.append(InvoiceLineItem(
                id=f"li_{uuid.uuid4().hex[:12]}",
                description=f"Credit: Unused time on {old_plan.name}",
                quantity=Decimal("1"),
                unit_price_cents=-change.credit_cents,
                total_cents=-change.credit_cents,
            ))

        line_items.append(InvoiceLineItem(
            id=f"li_{uuid.uuid4().hex[:12]}",
            description=f"Prorated charge: {new_plan.name}",
            quantity=Decimal("1"),
            unit_price_cents=change.charge_cents,
            total_cents=change.charge_cents,
            period_start=change.effective_date,
            period_end=subscription.current_period_end,
        ))

        invoice = Invoice(
            id=f"inv_{uuid.uuid4().hex[:16]}",
            invoice_number=f"INV-{datetime.utcnow().strftime('%Y%m')}-{uuid.uuid4().hex[:6].upper()}",
            organization_id=subscription.organization_id,
            subscription_id=subscription.id,
            status=InvoiceStatus.DRAFT,
            currency="usd",
            subtotal_cents=net_charge,
            discount_cents=0,
            credit_applied_cents=0,
            tax_cents=0,
            total_cents=net_charge,
            amount_due_cents=net_charge,
            amount_paid_cents=0,
            line_items=line_items,
            period_start=change.effective_date,
            period_end=subscription.current_period_end,
        )

        return invoice
