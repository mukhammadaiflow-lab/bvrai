"""
Invoice Management

Invoice generation, processing, and management.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from .base import (
    BillingCustomer,
    BillingError,
    Credit,
    Discount,
    Invoice,
    InvoiceError,
    InvoiceLineItem,
    InvoiceStatus,
    InvoiceStore,
    Payment,
    PaymentStatus,
    Subscription,
    UsageSummary,
    UsageType,
)
from .pricing import CreditManager, DiscountManager, PlanCatalog, PricingCalculator


logger = logging.getLogger(__name__)


class InMemoryInvoiceStore(InvoiceStore):
    """In-memory invoice store implementation."""

    def __init__(self):
        """Initialize in-memory store."""
        self._invoices: Dict[str, Invoice] = {}
        self._by_number: Dict[str, str] = {}
        self._by_organization: Dict[str, List[str]] = {}

    async def create(self, invoice: Invoice) -> None:
        """Create invoice."""
        self._invoices[invoice.id] = invoice
        self._by_number[invoice.invoice_number] = invoice.id

        if invoice.organization_id not in self._by_organization:
            self._by_organization[invoice.organization_id] = []
        self._by_organization[invoice.organization_id].append(invoice.id)

    async def get(self, invoice_id: str) -> Optional[Invoice]:
        """Get invoice by ID."""
        return self._invoices.get(invoice_id)

    async def get_by_number(self, invoice_number: str) -> Optional[Invoice]:
        """Get invoice by number."""
        invoice_id = self._by_number.get(invoice_number)
        if invoice_id:
            return self._invoices.get(invoice_id)
        return None

    async def list_for_organization(
        self,
        organization_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Invoice]:
        """List invoices for organization."""
        invoice_ids = self._by_organization.get(organization_id, [])

        # Sort by created date descending
        invoices = [self._invoices[id] for id in invoice_ids if id in self._invoices]
        invoices.sort(key=lambda x: x.created_at, reverse=True)

        return invoices[offset:offset + limit]

    async def update(self, invoice: Invoice) -> None:
        """Update invoice."""
        self._invoices[invoice.id] = invoice

    async def list_unpaid(self) -> List[Invoice]:
        """List all unpaid invoices."""
        return [
            inv for inv in self._invoices.values()
            if inv.status in [InvoiceStatus.OPEN, InvoiceStatus.PAST_DUE]
        ]

    async def list_past_due(self) -> List[Invoice]:
        """List all past due invoices."""
        return [
            inv for inv in self._invoices.values()
            if inv.status == InvoiceStatus.PAST_DUE
            or (inv.status == InvoiceStatus.OPEN and inv.is_past_due())
        ]


@dataclass
class InvoiceGenerationResult:
    """Result of invoice generation."""

    invoice: Invoice
    credits_applied: List[tuple]  # (credit_id, amount)
    discounts_applied: List[tuple]  # (discount_id, amount)
    warnings: List[str] = field(default_factory=list)


class InvoiceGenerator:
    """
    Generator for creating invoices.

    Handles line item creation, discounts, credits, and tax calculation.
    """

    def __init__(
        self,
        catalog: PlanCatalog,
        calculator: PricingCalculator,
        credit_manager: Optional[CreditManager] = None,
        discount_manager: Optional[DiscountManager] = None,
    ):
        """Initialize invoice generator."""
        self._catalog = catalog
        self._calculator = calculator
        self._credit_manager = credit_manager
        self._discount_manager = discount_manager

        self._invoice_counter = 0

    def _generate_invoice_number(self) -> str:
        """Generate unique invoice number."""
        self._invoice_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m")
        return f"INV-{timestamp}-{self._invoice_counter:06d}"

    async def generate_subscription_invoice(
        self,
        organization_id: str,
        subscription: Subscription,
        usage_summary: Optional[UsageSummary] = None,
        discounts: Optional[List[Discount]] = None,
        apply_credits: bool = True,
        tax_rate: Decimal = Decimal("0"),
    ) -> InvoiceGenerationResult:
        """Generate invoice for subscription billing."""
        import uuid

        plan = self._catalog.get_plan(subscription.plan_id)
        if not plan:
            raise InvoiceError(f"Plan not found: {subscription.plan_id}")

        line_items: List[InvoiceLineItem] = []
        warnings: List[str] = []

        # Add base subscription charge
        line_items.append(InvoiceLineItem(
            id=f"li_{uuid.uuid4().hex[:12]}",
            description=f"{plan.name} - {plan.billing_period.value.title()} Subscription",
            quantity=Decimal("1"),
            unit_price_cents=plan.base_price_cents,
            total_cents=plan.base_price_cents,
            period_start=subscription.current_period_start,
            period_end=subscription.current_period_end,
        ))

        # Add usage charges
        if usage_summary:
            for usage_type, quantity in usage_summary.usage_by_type.items():
                pricing = plan.usage_pricing.get(usage_type)
                if not pricing:
                    continue

                included = pricing.included_units
                billable = max(0, int(quantity) - included)

                if billable > 0:
                    cost = self._calculator.calculate_usage_cost(plan, usage_type, int(quantity))

                    if cost > 0:
                        unit_price = cost // billable if billable > 0 else 0
                        line_items.append(InvoiceLineItem(
                            id=f"li_{uuid.uuid4().hex[:12]}",
                            description=f"{usage_type.value.replace('_', ' ').title()} Usage ({billable} units)",
                            quantity=Decimal(str(billable)),
                            unit_price_cents=unit_price,
                            total_cents=cost,
                            usage_type=usage_type,
                            period_start=usage_summary.period_start,
                            period_end=usage_summary.period_end,
                        ))

        # Calculate subtotal
        subtotal = sum(item.total_cents for item in line_items)

        # Apply discounts
        discount_total = 0
        discounts_applied: List[tuple] = []

        if discounts:
            remaining = subtotal
            for discount in discounts:
                if discount.is_valid():
                    amount = discount.calculate_discount(remaining)
                    if amount > 0:
                        discount_total += amount
                        remaining -= amount
                        discounts_applied.append((discount.id, amount))

                        # Track usage
                        if self._discount_manager:
                            self._discount_manager.use_discount(discount.id)

        # Apply credits
        credit_total = 0
        credits_applied: List[tuple] = []

        if apply_credits and self._credit_manager:
            remaining_after_discount = subtotal - discount_total
            remaining, applied = self._credit_manager.apply_credits(
                organization_id,
                remaining_after_discount,
            )
            credit_total = remaining_after_discount - remaining
            credits_applied = [(c.id, amount) for c, amount in applied]

        # Calculate tax
        taxable = subtotal - discount_total - credit_total
        tax = int(taxable * tax_rate / 100) if tax_rate > 0 else 0

        # Calculate total
        total = subtotal - discount_total - credit_total + tax
        total = max(0, total)

        invoice = Invoice(
            id=f"inv_{uuid.uuid4().hex[:16]}",
            invoice_number=self._generate_invoice_number(),
            organization_id=organization_id,
            subscription_id=subscription.id,
            status=InvoiceStatus.DRAFT,
            currency=plan.currency,
            subtotal_cents=subtotal,
            discount_cents=discount_total,
            credit_applied_cents=credit_total,
            tax_cents=tax,
            total_cents=total,
            amount_due_cents=total,
            amount_paid_cents=0,
            line_items=line_items,
            discounts_applied=[d[0] for d in discounts_applied],
            credits_applied=[c[0] for c in credits_applied],
            period_start=subscription.current_period_start,
            period_end=subscription.current_period_end,
        )

        return InvoiceGenerationResult(
            invoice=invoice,
            credits_applied=credits_applied,
            discounts_applied=discounts_applied,
            warnings=warnings,
        )

    async def generate_one_time_invoice(
        self,
        organization_id: str,
        line_items: List[InvoiceLineItem],
        tax_rate: Decimal = Decimal("0"),
        description: Optional[str] = None,
    ) -> Invoice:
        """Generate a one-time invoice."""
        import uuid

        subtotal = sum(item.total_cents for item in line_items)
        tax = int(subtotal * tax_rate / 100) if tax_rate > 0 else 0
        total = subtotal + tax

        return Invoice(
            id=f"inv_{uuid.uuid4().hex[:16]}",
            invoice_number=self._generate_invoice_number(),
            organization_id=organization_id,
            subscription_id=None,
            status=InvoiceStatus.DRAFT,
            currency="usd",
            subtotal_cents=subtotal,
            discount_cents=0,
            credit_applied_cents=0,
            tax_cents=tax,
            total_cents=total,
            amount_due_cents=total,
            amount_paid_cents=0,
            line_items=line_items,
            metadata={"description": description} if description else {},
        )


class InvoiceProcessor:
    """
    Processor for invoice operations.

    Handles finalizing, paying, voiding, and managing invoice lifecycle.
    """

    def __init__(
        self,
        store: InvoiceStore,
    ):
        """Initialize invoice processor."""
        self._store = store

        # Payment handlers
        self._payment_handlers: List[Callable] = []

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

    def add_payment_handler(self, handler: Callable) -> None:
        """Add handler for processing payments."""
        self._payment_handlers.append(handler)

    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def _emit_event(self, event_type: str, invoice: Invoice) -> None:
        """Emit invoice event."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(invoice)
                else:
                    handler(invoice)
            except Exception as e:
                logger.error(f"Error in invoice event handler: {e}")

    async def create(self, invoice: Invoice) -> Invoice:
        """Create and store invoice."""
        await self._store.create(invoice)
        await self._emit_event("invoice.created", invoice)
        return invoice

    async def finalize(self, invoice_id: str) -> Invoice:
        """Finalize invoice (make it payable)."""
        invoice = await self._store.get(invoice_id)
        if not invoice:
            raise InvoiceError(f"Invoice not found: {invoice_id}", invoice_id=invoice_id)

        if invoice.status != InvoiceStatus.DRAFT:
            raise InvoiceError(
                f"Cannot finalize invoice in {invoice.status.value} status",
                invoice_id=invoice_id,
            )

        invoice.status = InvoiceStatus.OPEN
        await self._store.update(invoice)
        await self._emit_event("invoice.finalized", invoice)

        return invoice

    async def mark_paid(
        self,
        invoice_id: str,
        payment: Payment,
    ) -> Invoice:
        """Mark invoice as paid."""
        invoice = await self._store.get(invoice_id)
        if not invoice:
            raise InvoiceError(f"Invoice not found: {invoice_id}", invoice_id=invoice_id)

        if invoice.status in [InvoiceStatus.PAID, InvoiceStatus.VOID]:
            raise InvoiceError(
                f"Cannot pay invoice in {invoice.status.value} status",
                invoice_id=invoice_id,
            )

        invoice.amount_paid_cents = payment.amount_cents
        invoice.amount_due_cents = max(0, invoice.total_cents - payment.amount_cents)
        invoice.payment_intent_id = payment.stripe_payment_intent_id
        invoice.paid_at = datetime.utcnow()

        if invoice.amount_due_cents == 0:
            invoice.status = InvoiceStatus.PAID
        else:
            # Partial payment
            invoice.status = InvoiceStatus.OPEN

        await self._store.update(invoice)
        await self._emit_event("invoice.paid", invoice)

        return invoice

    async def void(
        self,
        invoice_id: str,
        reason: Optional[str] = None,
    ) -> Invoice:
        """Void an invoice."""
        invoice = await self._store.get(invoice_id)
        if not invoice:
            raise InvoiceError(f"Invoice not found: {invoice_id}", invoice_id=invoice_id)

        if invoice.status == InvoiceStatus.PAID:
            raise InvoiceError("Cannot void a paid invoice", invoice_id=invoice_id)

        invoice.status = InvoiceStatus.VOID
        invoice.voided_at = datetime.utcnow()
        if reason:
            invoice.metadata["void_reason"] = reason

        await self._store.update(invoice)
        await self._emit_event("invoice.voided", invoice)

        return invoice

    async def mark_uncollectible(
        self,
        invoice_id: str,
        reason: Optional[str] = None,
    ) -> Invoice:
        """Mark invoice as uncollectible."""
        invoice = await self._store.get(invoice_id)
        if not invoice:
            raise InvoiceError(f"Invoice not found: {invoice_id}", invoice_id=invoice_id)

        invoice.status = InvoiceStatus.UNCOLLECTIBLE
        if reason:
            invoice.metadata["uncollectible_reason"] = reason

        await self._store.update(invoice)
        await self._emit_event("invoice.uncollectible", invoice)

        return invoice

    async def check_past_due(self) -> List[Invoice]:
        """Check and update past due invoices."""
        updated = []

        unpaid = await self._store.list_unpaid()
        for invoice in unpaid:
            if invoice.status == InvoiceStatus.OPEN and invoice.is_past_due():
                invoice.status = InvoiceStatus.PAST_DUE
                await self._store.update(invoice)
                await self._emit_event("invoice.past_due", invoice)
                updated.append(invoice)

        return updated

    async def get_outstanding_balance(
        self,
        organization_id: str,
    ) -> int:
        """Get total outstanding balance for organization."""
        invoices = await self._store.list_for_organization(organization_id)
        return sum(
            inv.amount_due_cents for inv in invoices
            if inv.status in [InvoiceStatus.OPEN, InvoiceStatus.PAST_DUE]
        )


class InvoiceService:
    """
    High-level invoice service.

    Provides convenient methods for invoice operations.
    """

    def __init__(
        self,
        generator: InvoiceGenerator,
        processor: InvoiceProcessor,
        store: InvoiceStore,
    ):
        """Initialize invoice service."""
        self._generator = generator
        self._processor = processor
        self._store = store

    async def create_subscription_invoice(
        self,
        organization_id: str,
        subscription: Subscription,
        usage_summary: Optional[UsageSummary] = None,
        auto_finalize: bool = True,
    ) -> Invoice:
        """Create and optionally finalize subscription invoice."""
        result = await self._generator.generate_subscription_invoice(
            organization_id=organization_id,
            subscription=subscription,
            usage_summary=usage_summary,
        )

        invoice = await self._processor.create(result.invoice)

        if auto_finalize:
            invoice = await self._processor.finalize(invoice.id)

        return invoice

    async def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        """Get invoice by ID."""
        return await self._store.get(invoice_id)

    async def get_invoice_by_number(self, invoice_number: str) -> Optional[Invoice]:
        """Get invoice by number."""
        return await self._store.get_by_number(invoice_number)

    async def list_invoices(
        self,
        organization_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Invoice]:
        """List invoices for organization."""
        return await self._store.list_for_organization(organization_id, limit, offset)

    async def void_invoice(
        self,
        invoice_id: str,
        reason: Optional[str] = None,
    ) -> Invoice:
        """Void an invoice."""
        return await self._processor.void(invoice_id, reason)

    async def record_payment(
        self,
        invoice_id: str,
        payment: Payment,
    ) -> Invoice:
        """Record payment for invoice."""
        return await self._processor.mark_paid(invoice_id, payment)

    async def get_organization_balance(
        self,
        organization_id: str,
    ) -> Dict[str, Any]:
        """Get organization billing balance."""
        invoices = await self._store.list_for_organization(organization_id)

        total_invoiced = sum(inv.total_cents for inv in invoices)
        total_paid = sum(inv.amount_paid_cents for inv in invoices)
        outstanding = sum(
            inv.amount_due_cents for inv in invoices
            if inv.status in [InvoiceStatus.OPEN, InvoiceStatus.PAST_DUE]
        )
        past_due = sum(
            inv.amount_due_cents for inv in invoices
            if inv.status == InvoiceStatus.PAST_DUE or (inv.status == InvoiceStatus.OPEN and inv.is_past_due())
        )

        return {
            "total_invoiced_cents": total_invoiced,
            "total_paid_cents": total_paid,
            "outstanding_cents": outstanding,
            "past_due_cents": past_due,
            "invoice_count": len(invoices),
            "unpaid_count": sum(
                1 for inv in invoices
                if inv.status in [InvoiceStatus.OPEN, InvoiceStatus.PAST_DUE]
            ),
        }

    async def send_invoice_reminder(
        self,
        invoice_id: str,
    ) -> bool:
        """Send payment reminder for invoice."""
        invoice = await self._store.get(invoice_id)
        if not invoice:
            return False

        if invoice.status not in [InvoiceStatus.OPEN, InvoiceStatus.PAST_DUE]:
            return False

        # This would integrate with notification system
        logger.info(f"Sending reminder for invoice {invoice.invoice_number}")
        return True


class InvoicePDFGenerator:
    """Generator for invoice PDF documents."""

    def __init__(self, company_info: Dict[str, str]):
        """Initialize PDF generator."""
        self._company_info = company_info

    async def generate(
        self,
        invoice: Invoice,
        customer: BillingCustomer,
    ) -> bytes:
        """Generate PDF for invoice."""
        # In a real implementation, this would use a PDF library
        # like reportlab or weasyprint

        # For now, return a placeholder
        content = f"""
INVOICE
=======

Invoice Number: {invoice.invoice_number}
Date: {invoice.created_at.strftime('%Y-%m-%d')}
Due Date: {invoice.due_date.strftime('%Y-%m-%d')}

From:
{self._company_info.get('name', 'Company')}
{self._company_info.get('address', '')}

To:
{customer.name}
{customer.email}
{customer.address_line1 or ''}
{customer.city or ''}, {customer.state or ''} {customer.postal_code or ''}

Line Items:
-----------
"""
        for item in invoice.line_items:
            content += f"{item.description}: ${item.total_cents / 100:.2f}\n"

        content += f"""
-----------
Subtotal: ${invoice.subtotal_cents / 100:.2f}
Discount: -${invoice.discount_cents / 100:.2f}
Credit: -${invoice.credit_applied_cents / 100:.2f}
Tax: ${invoice.tax_cents / 100:.2f}
-----------
Total: ${invoice.total_cents / 100:.2f}
Amount Due: ${invoice.amount_due_cents / 100:.2f}
"""

        return content.encode('utf-8')
