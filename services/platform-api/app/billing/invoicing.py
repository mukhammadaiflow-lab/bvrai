"""Invoice generation for billing."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import uuid
import logging

from app.billing.usage import UsageType, UsageTracker, get_usage_tracker
from app.billing.pricing import (
    PricingTier,
    PricingCalculator,
    get_pricing_calculator,
)

logger = logging.getLogger(__name__)


class InvoiceStatus(str, Enum):
    """Invoice status."""
    DRAFT = "draft"
    PENDING = "pending"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


@dataclass
class InvoiceLineItem:
    """A line item on an invoice."""
    description: str
    quantity: float
    unit_price: Decimal
    amount: Decimal
    usage_type: Optional[UsageType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": float(self.unit_price),
            "amount": float(self.amount),
            "usage_type": self.usage_type.value if self.usage_type else None,
            "metadata": self.metadata,
        }


@dataclass
class Invoice:
    """Invoice for billing."""
    invoice_id: str
    user_id: str
    status: InvoiceStatus = InvoiceStatus.DRAFT
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    line_items: List[InvoiceLineItem] = field(default_factory=list)
    subtotal: Decimal = Decimal("0")
    tax_rate: Decimal = Decimal("0")
    tax_amount: Decimal = Decimal("0")
    discount_amount: Decimal = Decimal("0")
    total: Decimal = Decimal("0")
    currency: str = "USD"
    due_date: Optional[date] = None
    paid_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "invoice_id": self.invoice_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "line_items": [item.to_dict() for item in self.line_items],
            "subtotal": float(self.subtotal),
            "tax_rate": float(self.tax_rate),
            "tax_amount": float(self.tax_amount),
            "discount_amount": float(self.discount_amount),
            "total": float(self.total),
            "currency": self.currency,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    def add_line_item(
        self,
        description: str,
        quantity: float,
        unit_price: Decimal,
        usage_type: Optional[UsageType] = None,
        **metadata,
    ) -> None:
        """Add a line item."""
        amount = (Decimal(str(quantity)) * unit_price).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        self.line_items.append(InvoiceLineItem(
            description=description,
            quantity=quantity,
            unit_price=unit_price,
            amount=amount,
            usage_type=usage_type,
            metadata=metadata,
        ))
        self._recalculate_totals()

    def _recalculate_totals(self) -> None:
        """Recalculate invoice totals."""
        self.subtotal = sum(
            item.amount for item in self.line_items
        )
        self.tax_amount = (self.subtotal * self.tax_rate / 100).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        self.total = self.subtotal + self.tax_amount - self.discount_amount

    def apply_discount(self, amount: Decimal) -> None:
        """Apply discount to invoice."""
        self.discount_amount = amount
        self._recalculate_totals()

    def set_tax_rate(self, rate: Decimal) -> None:
        """Set tax rate."""
        self.tax_rate = rate
        self._recalculate_totals()


class InvoiceGenerator:
    """
    Generates invoices from usage data.

    Usage:
        generator = InvoiceGenerator()

        # Generate invoice for a billing period
        invoice = await generator.generate(
            user_id="user-123",
            tier=PricingTier.STARTER,
            period_start=date(2024, 1, 1),
            period_end=date(2024, 1, 31),
        )
    """

    def __init__(
        self,
        tracker: Optional[UsageTracker] = None,
        calculator: Optional[PricingCalculator] = None,
    ):
        self.tracker = tracker or get_usage_tracker()
        self.calculator = calculator or get_pricing_calculator()
        self._invoices: Dict[str, Invoice] = {}

    async def generate(
        self,
        user_id: str,
        tier: PricingTier,
        period_start: date,
        period_end: date,
        tax_rate: Decimal = Decimal("0"),
        discount: Decimal = Decimal("0"),
    ) -> Invoice:
        """Generate invoice for billing period."""
        # Get plan
        plan = self.calculator.get_plan(tier)
        if not plan:
            raise ValueError(f"Unknown tier: {tier}")

        # Create invoice
        invoice = Invoice(
            invoice_id=str(uuid.uuid4()),
            user_id=user_id,
            period_start=period_start,
            period_end=period_end,
            tax_rate=tax_rate,
        )

        # Add base plan fee
        if plan.base_price > 0:
            invoice.add_line_item(
                description=f"{plan.name} Plan - Monthly Fee",
                quantity=1,
                unit_price=plan.base_price,
            )

        # Get usage for period
        start_dt = datetime.combine(period_start, datetime.min.time())
        end_dt = datetime.combine(period_end, datetime.max.time())
        usage_totals = await self.tracker.get_totals(user_id, start_dt, end_dt)

        # Add usage line items
        for usage_type, quantity in usage_totals.items():
            try:
                ut = UsageType(usage_type)
                component = plan.get_price(ut)

                if component:
                    billable = max(0, quantity - component.included_quantity)
                    if billable > 0:
                        invoice.add_line_item(
                            description=f"{ut.value.replace('_', ' ').title()} ({quantity:.2f} {component.unit_name}s, {component.included_quantity:.2f} included)",
                            quantity=billable,
                            unit_price=component.unit_price,
                            usage_type=ut,
                            total_quantity=quantity,
                            included=component.included_quantity,
                        )
            except ValueError:
                logger.warning(f"Unknown usage type: {usage_type}")

        # Apply discount
        if discount > 0:
            invoice.apply_discount(discount)

        # Set due date (15 days from now)
        from datetime import timedelta
        invoice.due_date = date.today() + timedelta(days=15)

        # Store invoice
        self._invoices[invoice.invoice_id] = invoice

        return invoice

    async def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        """Get invoice by ID."""
        return self._invoices.get(invoice_id)

    async def get_user_invoices(
        self,
        user_id: str,
        status: Optional[InvoiceStatus] = None,
    ) -> List[Invoice]:
        """Get invoices for user."""
        invoices = [
            inv for inv in self._invoices.values()
            if inv.user_id == user_id
        ]

        if status:
            invoices = [inv for inv in invoices if inv.status == status]

        return sorted(invoices, key=lambda x: x.created_at, reverse=True)

    async def mark_paid(
        self,
        invoice_id: str,
        payment_method: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Optional[Invoice]:
        """Mark invoice as paid."""
        invoice = self._invoices.get(invoice_id)
        if not invoice:
            return None

        invoice.status = InvoiceStatus.PAID
        invoice.paid_at = datetime.utcnow()
        invoice.metadata["payment_method"] = payment_method
        invoice.metadata["transaction_id"] = transaction_id

        return invoice

    async def cancel(self, invoice_id: str, reason: str = "") -> Optional[Invoice]:
        """Cancel an invoice."""
        invoice = self._invoices.get(invoice_id)
        if not invoice:
            return None

        if invoice.status == InvoiceStatus.PAID:
            raise ValueError("Cannot cancel a paid invoice")

        invoice.status = InvoiceStatus.CANCELLED
        invoice.metadata["cancellation_reason"] = reason

        return invoice

    async def refund(
        self,
        invoice_id: str,
        amount: Optional[Decimal] = None,
        reason: str = "",
    ) -> Optional[Invoice]:
        """Refund an invoice."""
        invoice = self._invoices.get(invoice_id)
        if not invoice:
            return None

        if invoice.status != InvoiceStatus.PAID:
            raise ValueError("Can only refund paid invoices")

        refund_amount = amount or invoice.total
        invoice.status = InvoiceStatus.REFUNDED
        invoice.metadata["refund_amount"] = float(refund_amount)
        invoice.metadata["refund_reason"] = reason

        return invoice


class InvoiceFormatter:
    """
    Formats invoices for display/export.

    Supports HTML, PDF, and plain text formats.
    """

    @staticmethod
    def to_html(invoice: Invoice, company_info: Optional[Dict[str, str]] = None) -> str:
        """Format invoice as HTML."""
        company = company_info or {
            "name": "Builder Engine",
            "address": "123 AI Street, San Francisco, CA 94102",
            "email": "billing@bvrai.com",
        }

        items_html = ""
        for item in invoice.line_items:
            items_html += f"""
            <tr>
                <td>{item.description}</td>
                <td class="text-right">{item.quantity:.2f}</td>
                <td class="text-right">${item.unit_price:.4f}</td>
                <td class="text-right">${item.amount:.2f}</td>
            </tr>
            """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Invoice {invoice.invoice_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ display: flex; justify-content: space-between; margin-bottom: 40px; }}
                .invoice-title {{ font-size: 24px; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                .text-right {{ text-align: right; }}
                .totals {{ margin-top: 20px; }}
                .totals td {{ border: none; }}
                .total-row {{ font-weight: bold; font-size: 1.2em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div>
                    <div class="invoice-title">INVOICE</div>
                    <div>{company['name']}</div>
                    <div>{company['address']}</div>
                    <div>{company['email']}</div>
                </div>
                <div>
                    <div><strong>Invoice #:</strong> {invoice.invoice_id[:8]}</div>
                    <div><strong>Date:</strong> {invoice.created_at.strftime('%Y-%m-%d')}</div>
                    <div><strong>Due:</strong> {invoice.due_date.isoformat() if invoice.due_date else 'N/A'}</div>
                    <div><strong>Status:</strong> {invoice.status.value.upper()}</div>
                </div>
            </div>

            <div>
                <strong>Bill To:</strong><br>
                User ID: {invoice.user_id}<br>
                Period: {invoice.period_start} to {invoice.period_end}
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Description</th>
                        <th class="text-right">Quantity</th>
                        <th class="text-right">Unit Price</th>
                        <th class="text-right">Amount</th>
                    </tr>
                </thead>
                <tbody>
                    {items_html}
                </tbody>
            </table>

            <table class="totals">
                <tr>
                    <td colspan="3" class="text-right">Subtotal:</td>
                    <td class="text-right">${invoice.subtotal:.2f}</td>
                </tr>
                <tr>
                    <td colspan="3" class="text-right">Tax ({invoice.tax_rate}%):</td>
                    <td class="text-right">${invoice.tax_amount:.2f}</td>
                </tr>
                <tr>
                    <td colspan="3" class="text-right">Discount:</td>
                    <td class="text-right">-${invoice.discount_amount:.2f}</td>
                </tr>
                <tr class="total-row">
                    <td colspan="3" class="text-right">Total ({invoice.currency}):</td>
                    <td class="text-right">${invoice.total:.2f}</td>
                </tr>
            </table>
        </body>
        </html>
        """

    @staticmethod
    def to_text(invoice: Invoice) -> str:
        """Format invoice as plain text."""
        lines = [
            "=" * 60,
            f"INVOICE #{invoice.invoice_id[:8]}",
            "=" * 60,
            f"Status: {invoice.status.value.upper()}",
            f"Date: {invoice.created_at.strftime('%Y-%m-%d')}",
            f"Due: {invoice.due_date.isoformat() if invoice.due_date else 'N/A'}",
            f"Period: {invoice.period_start} to {invoice.period_end}",
            "",
            "-" * 60,
            "LINE ITEMS",
            "-" * 60,
        ]

        for item in invoice.line_items:
            lines.append(f"{item.description}")
            lines.append(f"  {item.quantity:.2f} x ${item.unit_price:.4f} = ${item.amount:.2f}")

        lines.extend([
            "",
            "-" * 60,
            f"{'Subtotal:':>50} ${invoice.subtotal:.2f}",
            f"{'Tax:':>50} ${invoice.tax_amount:.2f}",
            f"{'Discount:':>50} -${invoice.discount_amount:.2f}",
            "=" * 60,
            f"{'TOTAL:':>50} ${invoice.total:.2f} {invoice.currency}",
            "=" * 60,
        ])

        return "\n".join(lines)


# Global invoice generator
_invoice_generator: Optional[InvoiceGenerator] = None


def get_invoice_generator() -> InvoiceGenerator:
    """Get or create the global invoice generator."""
    global _invoice_generator
    if _invoice_generator is None:
        _invoice_generator = InvoiceGenerator()
    return _invoice_generator
