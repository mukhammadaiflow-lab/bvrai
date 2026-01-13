"""
Builder Engine Python SDK - Billing Resource

This module provides methods for managing billing and subscriptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List

from builderengine.resources.base import BaseResource, PaginatedResponse
from builderengine.models import Subscription, Invoice, PaymentMethod, Usage
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class BillingResource(BaseResource):
    """
    Resource for managing billing and subscriptions.

    This resource provides access to subscription management, invoices,
    payment methods, and usage tracking.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> usage = client.billing.get_usage()
        >>> print(f"Minutes used: {usage.minutes_used}/{usage.minutes_limit}")
    """

    def get_subscription(self) -> Subscription:
        """
        Get the current subscription.

        Returns:
            Subscription object
        """
        response = self._get(Endpoints.BILLING_SUBSCRIPTION)
        return Subscription.from_dict(response)

    def get_plans(self) -> List[Dict[str, Any]]:
        """
        Get available subscription plans.

        Returns:
            List of plans with pricing and features
        """
        response = self._get(f"{Endpoints.BILLING_SUBSCRIPTION}/plans")
        return response.get("plans", [])

    def change_plan(
        self,
        plan_id: str,
        prorate: bool = True,
    ) -> Subscription:
        """
        Change subscription plan.

        Args:
            plan_id: ID of the new plan
            prorate: Whether to prorate the change

        Returns:
            Updated Subscription object

        Example:
            >>> subscription = client.billing.change_plan("plan_pro")
            >>> print(f"New plan: {subscription.plan_name}")
        """
        response = self._post(f"{Endpoints.BILLING_SUBSCRIPTION}/change", json={
            "plan_id": plan_id,
            "prorate": prorate,
        })
        return Subscription.from_dict(response)

    def cancel_subscription(
        self,
        at_period_end: bool = True,
        reason: Optional[str] = None,
    ) -> Subscription:
        """
        Cancel the subscription.

        Args:
            at_period_end: Cancel at end of billing period (vs immediately)
            reason: Reason for cancellation

        Returns:
            Updated Subscription object
        """
        data = {"at_period_end": at_period_end}
        if reason:
            data["reason"] = reason

        response = self._post(f"{Endpoints.BILLING_SUBSCRIPTION}/cancel", json=data)
        return Subscription.from_dict(response)

    def resume_subscription(self) -> Subscription:
        """
        Resume a canceled subscription.

        Returns:
            Updated Subscription object
        """
        response = self._post(f"{Endpoints.BILLING_SUBSCRIPTION}/resume")
        return Subscription.from_dict(response)

    # Usage

    def get_usage(
        self,
        period: str = "current",
    ) -> Usage:
        """
        Get usage for a billing period.

        Args:
            period: Billing period (current, previous, or date)

        Returns:
            Usage object with limits and current usage
        """
        response = self._get(Endpoints.BILLING_USAGE, params={"period": period})
        return Usage.from_dict(response)

    def get_usage_history(
        self,
        page: int = 1,
        page_size: int = 12,
    ) -> Dict[str, Any]:
        """
        Get usage history.

        Args:
            page: Page number
            page_size: Items per page

        Returns:
            Paginated usage history
        """
        return self._get(f"{Endpoints.BILLING_USAGE}/history", params={
            "page": page,
            "page_size": page_size,
        })

    # Invoices

    def list_invoices(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
    ) -> PaginatedResponse[Invoice]:
        """
        List invoices.

        Args:
            page: Page number
            page_size: Items per page
            status: Filter by status (draft, open, paid, void)

        Returns:
            PaginatedResponse containing Invoice objects
        """
        params = {
            "page": page,
            "page_size": page_size,
        }
        if status:
            params["status"] = status

        response = self._get(Endpoints.BILLING_INVOICES, params=params)
        return self._parse_paginated_response(response, Invoice)

    def get_invoice(self, invoice_id: str) -> Invoice:
        """
        Get an invoice by ID.

        Args:
            invoice_id: The invoice's unique identifier

        Returns:
            Invoice object
        """
        path = Endpoints.BILLING_INVOICE.format(invoice_id=invoice_id)
        response = self._get(path)
        return Invoice.from_dict(response)

    def download_invoice(self, invoice_id: str) -> Dict[str, Any]:
        """
        Get invoice PDF download URL.

        Args:
            invoice_id: The invoice's unique identifier

        Returns:
            Download URL
        """
        path = f"{Endpoints.BILLING_INVOICE.format(invoice_id=invoice_id)}/pdf"
        return self._get(path)

    def pay_invoice(self, invoice_id: str, payment_method_id: Optional[str] = None) -> Invoice:
        """
        Pay an open invoice.

        Args:
            invoice_id: The invoice's unique identifier
            payment_method_id: Payment method to use (default if not specified)

        Returns:
            Updated Invoice object
        """
        path = f"{Endpoints.BILLING_INVOICE.format(invoice_id=invoice_id)}/pay"
        data = {}
        if payment_method_id:
            data["payment_method_id"] = payment_method_id
        response = self._post(path, json=data)
        return Invoice.from_dict(response)

    # Payment Methods

    def list_payment_methods(self) -> List[PaymentMethod]:
        """
        List payment methods.

        Returns:
            List of PaymentMethod objects
        """
        response = self._get(Endpoints.BILLING_PAYMENT_METHODS)
        return [PaymentMethod.from_dict(pm) for pm in response.get("payment_methods", [])]

    def get_payment_method(self, payment_method_id: str) -> PaymentMethod:
        """
        Get a payment method by ID.

        Args:
            payment_method_id: The payment method's unique identifier

        Returns:
            PaymentMethod object
        """
        path = Endpoints.BILLING_PAYMENT_METHOD.format(payment_method_id=payment_method_id)
        response = self._get(path)
        return PaymentMethod.from_dict(response)

    def add_payment_method(
        self,
        type: str,
        token: str,
        set_default: bool = False,
        billing_address: Optional[Dict[str, str]] = None,
    ) -> PaymentMethod:
        """
        Add a payment method.

        Args:
            type: Payment method type (card, bank_account)
            token: Payment token from Stripe.js
            set_default: Set as default payment method
            billing_address: Billing address

        Returns:
            Created PaymentMethod object

        Example:
            >>> # Token from Stripe.js
            >>> pm = client.billing.add_payment_method(
            ...     type="card",
            ...     token="tok_visa",
            ...     set_default=True
            ... )
        """
        data = {
            "type": type,
            "token": token,
            "set_default": set_default,
        }
        if billing_address:
            data["billing_address"] = billing_address

        response = self._post(Endpoints.BILLING_PAYMENT_METHODS, json=data)
        return PaymentMethod.from_dict(response)

    def set_default_payment_method(self, payment_method_id: str) -> PaymentMethod:
        """
        Set a payment method as default.

        Args:
            payment_method_id: The payment method's unique identifier

        Returns:
            Updated PaymentMethod object
        """
        path = f"{Endpoints.BILLING_PAYMENT_METHOD.format(payment_method_id=payment_method_id)}/default"
        response = self._post(path)
        return PaymentMethod.from_dict(response)

    def delete_payment_method(self, payment_method_id: str) -> None:
        """
        Delete a payment method.

        Args:
            payment_method_id: The payment method's unique identifier
        """
        path = Endpoints.BILLING_PAYMENT_METHOD.format(payment_method_id=payment_method_id)
        self._delete(path)

    # Checkout and Portal

    def create_checkout_session(
        self,
        plan_id: str,
        success_url: str,
        cancel_url: str,
    ) -> Dict[str, Any]:
        """
        Create a Stripe checkout session.

        Args:
            plan_id: Plan to subscribe to
            success_url: URL to redirect on success
            cancel_url: URL to redirect on cancel

        Returns:
            Checkout session with URL
        """
        return self._post(Endpoints.BILLING_CHECKOUT, json={
            "plan_id": plan_id,
            "success_url": success_url,
            "cancel_url": cancel_url,
        })

    def create_portal_session(self, return_url: str) -> Dict[str, Any]:
        """
        Create a Stripe customer portal session.

        Args:
            return_url: URL to return to after portal

        Returns:
            Portal session with URL
        """
        return self._post(Endpoints.BILLING_PORTAL, json={"return_url": return_url})

    # Credits and Add-ons

    def get_credits(self) -> Dict[str, Any]:
        """
        Get available credits.

        Returns:
            Credit balance and history
        """
        return self._get(f"{Endpoints.BILLING_USAGE}/credits")

    def purchase_credits(
        self,
        amount: float,
        payment_method_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Purchase additional credits.

        Args:
            amount: Amount in dollars
            payment_method_id: Payment method to use

        Returns:
            Purchase confirmation
        """
        data = {"amount": amount}
        if payment_method_id:
            data["payment_method_id"] = payment_method_id

        return self._post(f"{Endpoints.BILLING_USAGE}/credits/purchase", json=data)

    def apply_promo_code(self, code: str) -> Dict[str, Any]:
        """
        Apply a promotional code.

        Args:
            code: Promo code to apply

        Returns:
            Result with discount details
        """
        return self._post(f"{Endpoints.BILLING_SUBSCRIPTION}/promo", json={"code": code})

    def get_spending_limit(self) -> Dict[str, Any]:
        """
        Get spending limit settings.

        Returns:
            Current spending limit configuration
        """
        return self._get(f"{Endpoints.BILLING_USAGE}/spending-limit")

    def set_spending_limit(
        self,
        limit: float,
        action: str = "notify",
    ) -> Dict[str, Any]:
        """
        Set spending limit.

        Args:
            limit: Monthly spending limit in dollars
            action: Action when limit reached (notify, pause, none)

        Returns:
            Updated spending limit configuration
        """
        return self._post(f"{Endpoints.BILLING_USAGE}/spending-limit", json={
            "limit": limit,
            "action": action,
        })
