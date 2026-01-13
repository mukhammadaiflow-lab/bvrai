"""
Webhook Package

This package provides comprehensive webhook functionality for the
voice agent platform, including:

- Webhook endpoint management
- Event dispatching and delivery
- Signature generation and verification
- Retry with exponential backoff
- Background worker processing
- Delivery logging and statistics

Example usage:

    from platform.webhooks import (
        WebhookDeliveryService,
        WebhookWorker,
        WebhookEventEmitter,
        Webhook,
        WebhookSigner,
    )

    # Create delivery service
    delivery_service = WebhookDeliveryService(redis_url="redis://localhost:6379")
    await delivery_service.start()

    # Register a webhook
    webhook = Webhook(
        id="wh_123",
        organization_id="org_456",
        url="https://example.com/webhook",
        events=["call.*", "agent.speech.*"],
        secret=WebhookSigner.generate_secret(),
    )
    await delivery_service.register_webhook(webhook)

    # Dispatch events
    await delivery_service.dispatch_event(
        organization_id="org_456",
        event_type="call.started",
        data={"call_id": "call_789", "agent_id": "agent_abc"},
    )

    # Or use the worker for queued processing
    worker = WebhookWorker(redis_url="redis://localhost:6379")
    await worker.start(delivery_service)

    emitter = WebhookEventEmitter(worker)
    await emitter.emit_call_started(
        organization_id="org_456",
        call_id="call_789",
        agent_id="agent_abc",
        to_number="+15551234567",
        from_number="+15559876543",
    )

    # Verify webhook signatures in your endpoint
    signer = WebhookSigner(secret)
    is_valid, error = signer.verify(payload, signature_header)
"""

# Models
from .models import (
    WebhookEventType,
    DeliveryStatus,
    WebhookStatus,
    Webhook,
    WebhookEvent,
    DeliveryAttempt,
    DeliveryLog,
    WebhookStats,
)

# Delivery
from .delivery import (
    DeliveryResult,
    WebhookDeliveryService,
)

# Signing
from .signing import (
    SignatureComponents,
    WebhookSigner,
    SignatureVerifier,
    create_signed_payload,
)

# Worker
from .worker import (
    WebhookWorker,
    WebhookEventEmitter,
)


__all__ = [
    # Models
    "WebhookEventType",
    "DeliveryStatus",
    "WebhookStatus",
    "Webhook",
    "WebhookEvent",
    "DeliveryAttempt",
    "DeliveryLog",
    "WebhookStats",
    # Delivery
    "DeliveryResult",
    "WebhookDeliveryService",
    # Signing
    "SignatureComponents",
    "WebhookSigner",
    "SignatureVerifier",
    "create_signed_payload",
    # Worker
    "WebhookWorker",
    "WebhookEventEmitter",
]
