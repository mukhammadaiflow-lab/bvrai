"""Unit tests for webhook functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import json
import hmac
import hashlib


class TestWebhookManager:
    """Tests for WebhookManager."""

    def test_create_webhook(self):
        """Test creating a new webhook."""
        from bvrai_core.webhooks.manager import Webhook, WebhookConfig

        config = WebhookConfig(
            url="https://example.com/webhook",
            events=["call.started", "call.ended"],
            secret="test_secret_123",
        )

        webhook = Webhook(
            id=f"whk_{uuid4().hex[:12]}",
            organization_id="org_test123",
            config=config,
            is_active=True,
        )

        assert webhook.config.url == "https://example.com/webhook"
        assert "call.started" in webhook.config.events
        assert webhook.is_active is True

    def test_webhook_event_filtering(self):
        """Test filtering events for webhook."""
        from bvrai_core.webhooks.manager import Webhook, WebhookConfig

        config = WebhookConfig(
            url="https://example.com/webhook",
            events=["call.started", "call.ended"],
        )

        webhook = Webhook(
            id="whk_test",
            organization_id="org_test",
            config=config,
        )

        # Should match
        assert webhook.should_receive_event("call.started") is True
        assert webhook.should_receive_event("call.ended") is True

        # Should not match
        assert webhook.should_receive_event("agent.created") is False
        assert webhook.should_receive_event("billing.invoice") is False

    def test_webhook_wildcard_events(self):
        """Test wildcard event matching."""
        from bvrai_core.webhooks.manager import Webhook, WebhookConfig

        config = WebhookConfig(
            url="https://example.com/webhook",
            events=["call.*"],  # Wildcard for all call events
        )

        webhook = Webhook(
            id="whk_test",
            organization_id="org_test",
            config=config,
        )

        assert webhook.should_receive_event("call.started") is True
        assert webhook.should_receive_event("call.ended") is True
        assert webhook.should_receive_event("call.failed") is True
        assert webhook.should_receive_event("agent.created") is False


class TestWebhookSignature:
    """Tests for webhook signature verification."""

    def test_generate_signature(self):
        """Test generating webhook signature."""
        from bvrai_core.webhooks.delivery import WebhookSigner

        signer = WebhookSigner(secret="test_secret")
        payload = json.dumps({"event": "test", "data": {"id": "123"}})
        timestamp = "1234567890"

        signature = signer.sign(payload, timestamp)

        assert signature is not None
        assert signature.startswith("v1=")

    def test_verify_signature(self):
        """Test verifying webhook signature."""
        from bvrai_core.webhooks.delivery import WebhookSigner

        secret = "test_secret"
        signer = WebhookSigner(secret=secret)
        payload = json.dumps({"event": "test"})
        timestamp = str(int(datetime.utcnow().timestamp()))

        signature = signer.sign(payload, timestamp)

        # Valid signature
        assert signer.verify(payload, timestamp, signature) is True

        # Invalid signature
        assert signer.verify(payload, timestamp, "v1=invalid") is False

    def test_signature_replay_protection(self):
        """Test replay attack protection."""
        from bvrai_core.webhooks.delivery import WebhookSigner

        signer = WebhookSigner(secret="test_secret", max_age_seconds=300)
        payload = json.dumps({"event": "test"})

        # Old timestamp (more than 5 minutes ago)
        old_timestamp = str(int((datetime.utcnow() - timedelta(minutes=10)).timestamp()))
        signature = signer.sign(payload, old_timestamp)

        # Should reject old timestamp
        assert signer.verify(payload, old_timestamp, signature, check_timestamp=True) is False


class TestWebhookDelivery:
    """Tests for webhook delivery."""

    @pytest.mark.asyncio
    async def test_deliver_webhook(self):
        """Test delivering a webhook."""
        from bvrai_core.webhooks.delivery import WebhookDeliveryService

        service = WebhookDeliveryService()

        with patch('aiohttp.ClientSession.post', new_callable=AsyncMock) as mock_post:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="OK")
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await service.deliver(
                url="https://example.com/webhook",
                payload={"event": "test", "data": {}},
                secret="test_secret",
            )

            assert result.success is True
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_delivery_retry_on_failure(self):
        """Test retry logic on delivery failure."""
        from bvrai_core.webhooks.delivery import WebhookDeliveryService

        service = WebhookDeliveryService(max_retries=3)

        with patch('aiohttp.ClientSession.post', new_callable=AsyncMock) as mock_post:
            # First two calls fail, third succeeds
            mock_fail = MagicMock()
            mock_fail.status = 500
            mock_fail.text = AsyncMock(return_value="Error")

            mock_success = MagicMock()
            mock_success.status = 200
            mock_success.text = AsyncMock(return_value="OK")

            mock_post.return_value.__aenter__.side_effect = [
                mock_fail, mock_fail, mock_success
            ]

            result = await service.deliver(
                url="https://example.com/webhook",
                payload={"event": "test"},
                secret="test_secret",
            )

            assert result.success is True
            assert result.attempts == 3


class TestWebhookPayload:
    """Tests for webhook payload formatting."""

    def test_format_call_started_payload(self):
        """Test formatting call.started payload."""
        from bvrai_core.webhooks.payloads import format_call_event

        call_data = {
            "id": "call_123",
            "agent_id": "agt_456",
            "direction": "inbound",
            "from_number": "+15551234567",
            "to_number": "+15559876543",
            "started_at": datetime.utcnow().isoformat(),
        }

        payload = format_call_event("call.started", call_data)

        assert payload["event"] == "call.started"
        assert payload["data"]["call"]["id"] == "call_123"
        assert "timestamp" in payload

    def test_format_call_ended_payload(self):
        """Test formatting call.ended payload."""
        from bvrai_core.webhooks.payloads import format_call_event

        call_data = {
            "id": "call_123",
            "agent_id": "agt_456",
            "direction": "inbound",
            "duration_seconds": 180,
            "ended_reason": "completed",
            "ended_at": datetime.utcnow().isoformat(),
        }

        payload = format_call_event("call.ended", call_data)

        assert payload["event"] == "call.ended"
        assert payload["data"]["call"]["duration_seconds"] == 180
        assert payload["data"]["call"]["ended_reason"] == "completed"

    def test_format_agent_event_payload(self):
        """Test formatting agent event payload."""
        from bvrai_core.webhooks.payloads import format_agent_event

        agent_data = {
            "id": "agt_123",
            "name": "Test Agent",
            "status": "active",
        }

        payload = format_agent_event("agent.updated", agent_data)

        assert payload["event"] == "agent.updated"
        assert payload["data"]["agent"]["id"] == "agt_123"


class TestWebhookRetryPolicy:
    """Tests for webhook retry policy."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        from bvrai_core.webhooks.delivery import calculate_backoff

        # First retry: 1 second base
        assert calculate_backoff(1, base=1) == 1

        # Second retry: 2 seconds
        assert calculate_backoff(2, base=1) == 2

        # Third retry: 4 seconds
        assert calculate_backoff(3, base=1) == 4

    def test_max_backoff(self):
        """Test maximum backoff limit."""
        from bvrai_core.webhooks.delivery import calculate_backoff

        # Should not exceed max_backoff
        result = calculate_backoff(10, base=1, max_backoff=60)
        assert result <= 60

    def test_jitter(self):
        """Test jitter in backoff."""
        from bvrai_core.webhooks.delivery import calculate_backoff

        # Multiple calls should produce slightly different results with jitter
        results = [calculate_backoff(3, base=1, jitter=True) for _ in range(10)]

        # Not all results should be identical due to jitter
        assert len(set(results)) > 1


class TestWebhookStorage:
    """Tests for webhook delivery storage."""

    @pytest.mark.asyncio
    async def test_store_delivery_attempt(self):
        """Test storing delivery attempt."""
        from bvrai_core.webhooks.storage import InMemoryDeliveryStore

        store = InMemoryDeliveryStore()

        await store.record_attempt(
            webhook_id="whk_123",
            event="call.started",
            status_code=200,
            success=True,
            response_body="OK",
        )

        attempts = await store.get_attempts("whk_123", limit=10)

        assert len(attempts) == 1
        assert attempts[0]["status_code"] == 200
        assert attempts[0]["success"] is True

    @pytest.mark.asyncio
    async def test_get_recent_failures(self):
        """Test getting recent failures."""
        from bvrai_core.webhooks.storage import InMemoryDeliveryStore

        store = InMemoryDeliveryStore()

        # Record some failures
        for i in range(5):
            await store.record_attempt(
                webhook_id="whk_123",
                event="call.started",
                status_code=500,
                success=False,
                error="Server error",
            )

        failures = await store.get_recent_failures("whk_123", hours=24)

        assert len(failures) == 5
        assert all(f["success"] is False for f in failures)
