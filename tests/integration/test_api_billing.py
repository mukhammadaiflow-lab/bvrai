"""Integration tests for billing API."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient
from uuid import uuid4


class TestBillingAPI:
    """Tests for billing management endpoints."""

    @pytest.mark.asyncio
    async def test_get_subscription(self, authenticated_client: AsyncClient, mock_stripe):
        """Test getting current subscription."""
        response = await authenticated_client.get("/api/v1/billing/subscription")

        assert response.status_code == 200
        data = response.json()
        assert "plan" in data
        assert "status" in data

    @pytest.mark.asyncio
    async def test_get_usage(self, authenticated_client: AsyncClient):
        """Test getting usage data."""
        response = await authenticated_client.get("/api/v1/billing/usage")

        assert response.status_code == 200
        data = response.json()
        assert "minutes_used" in data
        assert "calls_made" in data
        assert "minutes_limit" in data

    @pytest.mark.asyncio
    async def test_get_invoices(self, authenticated_client: AsyncClient, mock_stripe):
        """Test getting invoices."""
        response = await authenticated_client.get("/api/v1/billing/invoices")

        assert response.status_code == 200
        data = response.json()
        assert "invoices" in data

    @pytest.mark.asyncio
    async def test_get_plans(self, authenticated_client: AsyncClient):
        """Test getting available plans."""
        response = await authenticated_client.get("/api/v1/billing/plans")

        assert response.status_code == 200
        data = response.json()
        assert "plans" in data
        assert len(data["plans"]) > 0

        # Check plan structure
        plan = data["plans"][0]
        assert "id" in plan
        assert "name" in plan
        assert "price" in plan
        assert "features" in plan

    @pytest.mark.asyncio
    async def test_upgrade_plan(self, authenticated_client: AsyncClient, mock_stripe):
        """Test upgrading subscription plan."""
        response = await authenticated_client.post(
            "/api/v1/billing/upgrade",
            json={"plan": "professional"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["plan"] == "professional"

    @pytest.mark.asyncio
    async def test_cancel_subscription(self, authenticated_client: AsyncClient, mock_stripe):
        """Test canceling subscription."""
        response = await authenticated_client.post("/api/v1/billing/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["cancel_at_period_end"] is True

    @pytest.mark.asyncio
    async def test_resume_subscription(self, authenticated_client: AsyncClient, mock_stripe):
        """Test resuming canceled subscription."""
        response = await authenticated_client.post("/api/v1/billing/resume")

        assert response.status_code == 200
        data = response.json()
        assert data["cancel_at_period_end"] is False

    @pytest.mark.asyncio
    async def test_create_checkout_session(
        self, authenticated_client: AsyncClient, mock_stripe
    ):
        """Test creating checkout session for new subscription."""
        with patch("stripe.checkout.Session.create") as mock_create:
            mock_create.return_value = MagicMock(
                id="cs_test_123",
                url="https://checkout.stripe.com/session/cs_test_123",
            )

            response = await authenticated_client.post(
                "/api/v1/billing/checkout",
                json={
                    "plan_id": "professional",
                    "interval": "monthly",
                    "success_url": "https://app.example.com/billing?success=true",
                    "cancel_url": "https://app.example.com/billing?canceled=true",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "checkout_url" in data

    @pytest.mark.asyncio
    async def test_get_payment_methods(
        self, authenticated_client: AsyncClient, mock_stripe
    ):
        """Test getting payment methods."""
        response = await authenticated_client.get("/api/v1/billing/payment-methods")

        assert response.status_code == 200
        data = response.json()
        assert "payment_methods" in data

    @pytest.mark.asyncio
    async def test_add_payment_method(
        self, authenticated_client: AsyncClient, mock_stripe
    ):
        """Test adding a payment method."""
        with patch("stripe.PaymentMethod.attach") as mock_attach:
            mock_attach.return_value = MagicMock(id="pm_test_123")

            response = await authenticated_client.post(
                "/api/v1/billing/payment-methods",
                json={"payment_method_id": "pm_test_123"},
            )

            assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_set_default_payment_method(
        self, authenticated_client: AsyncClient, mock_stripe
    ):
        """Test setting default payment method."""
        response = await authenticated_client.post(
            "/api/v1/billing/payment-methods/pm_test_123/default"
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_remove_payment_method(
        self, authenticated_client: AsyncClient, mock_stripe
    ):
        """Test removing a payment method."""
        with patch("stripe.PaymentMethod.detach") as mock_detach:
            mock_detach.return_value = MagicMock(id="pm_test_123")

            response = await authenticated_client.delete(
                "/api/v1/billing/payment-methods/pm_test_123"
            )

            assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_get_setup_intent(
        self, authenticated_client: AsyncClient, mock_stripe
    ):
        """Test getting setup intent for adding payment method."""
        with patch("stripe.SetupIntent.create") as mock_create:
            mock_create.return_value = MagicMock(
                id="seti_test_123",
                client_secret="seti_test_123_secret_xxx",
            )

            response = await authenticated_client.post("/api/v1/billing/setup-intent")

            assert response.status_code == 200
            data = response.json()
            assert "client_secret" in data


class TestBillingWebhooks:
    """Tests for Stripe webhook handling."""

    @pytest.mark.asyncio
    async def test_invoice_paid_webhook(self, async_client: AsyncClient):
        """Test handling invoice.paid webhook."""
        payload = {
            "type": "invoice.paid",
            "data": {
                "object": {
                    "id": "inv_test_123",
                    "customer": "cus_test_123",
                    "amount_paid": 9900,
                    "lines": {
                        "data": [
                            {"price": {"product": "prod_professional"}}
                        ]
                    },
                },
            },
        }

        with patch("stripe.Webhook.construct_event") as mock_construct:
            mock_construct.return_value = payload

            response = await async_client.post(
                "/api/v1/billing/webhooks/stripe",
                content='{"test": "payload"}',
                headers={
                    "stripe-signature": "test_sig",
                    "content-type": "application/json",
                },
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_subscription_updated_webhook(self, async_client: AsyncClient):
        """Test handling customer.subscription.updated webhook."""
        payload = {
            "type": "customer.subscription.updated",
            "data": {
                "object": {
                    "id": "sub_test_123",
                    "customer": "cus_test_123",
                    "status": "active",
                    "items": {
                        "data": [
                            {"price": {"product": "prod_professional"}}
                        ]
                    },
                },
            },
        }

        with patch("stripe.Webhook.construct_event") as mock_construct:
            mock_construct.return_value = payload

            response = await async_client.post(
                "/api/v1/billing/webhooks/stripe",
                content='{"test": "payload"}',
                headers={
                    "stripe-signature": "test_sig",
                    "content-type": "application/json",
                },
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_payment_failed_webhook(self, async_client: AsyncClient):
        """Test handling invoice.payment_failed webhook."""
        payload = {
            "type": "invoice.payment_failed",
            "data": {
                "object": {
                    "id": "inv_test_123",
                    "customer": "cus_test_123",
                    "attempt_count": 1,
                },
            },
        }

        with patch("stripe.Webhook.construct_event") as mock_construct, \
             patch("app.services.email.send_payment_failed_email", new_callable=AsyncMock):
            mock_construct.return_value = payload

            response = await async_client.post(
                "/api/v1/billing/webhooks/stripe",
                content='{"test": "payload"}',
                headers={
                    "stripe-signature": "test_sig",
                    "content-type": "application/json",
                },
            )

            assert response.status_code == 200


class TestUsageTracking:
    """Tests for usage tracking."""

    @pytest.mark.asyncio
    async def test_usage_increments_on_call(
        self, authenticated_client: AsyncClient, test_agent, db_session, test_tenant
    ):
        """Test that usage increments when calls are made."""
        from app.models.usage import UsageRecord

        # Get initial usage
        initial_response = await authenticated_client.get("/api/v1/billing/usage")
        initial_minutes = initial_response.json().get("minutes_used", 0)

        # Simulate a completed call that updates usage
        usage_record = UsageRecord(
            id=str(uuid4()),
            tenant_id=test_tenant.id,
            type="minutes",
            amount=5.0,  # 5 minutes
            call_id=str(uuid4()),
            recorded_at=datetime.utcnow(),
        )
        db_session.add(usage_record)
        await db_session.commit()

        # Check updated usage
        updated_response = await authenticated_client.get("/api/v1/billing/usage")
        updated_minutes = updated_response.json().get("minutes_used", 0)

        assert updated_minutes >= initial_minutes

    @pytest.mark.asyncio
    async def test_usage_alerts(self, authenticated_client: AsyncClient, mock_redis):
        """Test usage alert thresholds."""
        with patch("app.services.billing.check_usage_alerts", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = {
                "alerts": [
                    {"type": "minutes", "threshold": 80, "current": 85, "message": "Usage at 85%"}
                ]
            }

            response = await authenticated_client.get("/api/v1/billing/usage/alerts")

            assert response.status_code == 200
            data = response.json()
            assert "alerts" in data
