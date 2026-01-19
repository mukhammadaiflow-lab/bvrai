"""Webhook management API routes."""

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, HttpUrl

from .delivery import WebhookDeliveryService
from .models import (
    Webhook,
    WebhookStatus,
    WebhookEventType,
    DeliveryStatus,
)
from .signing import WebhookSigner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


# Request/Response models

class WebhookCreate(BaseModel):
    """Create webhook request."""
    url: HttpUrl
    events: List[str]
    description: str = ""
    headers: Dict[str, str] = Field(default_factory=dict)
    secret: Optional[str] = None
    retry_enabled: bool = True
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=30, ge=5, le=120)


class WebhookUpdate(BaseModel):
    """Update webhook request."""
    url: Optional[HttpUrl] = None
    events: Optional[List[str]] = None
    description: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    status: Optional[str] = None
    retry_enabled: Optional[bool] = None
    max_retries: Optional[int] = Field(default=None, ge=0, le=10)
    timeout_seconds: Optional[int] = Field(default=None, ge=5, le=120)


class WebhookResponse(BaseModel):
    """Webhook response model."""
    id: str
    organization_id: str
    url: str
    events: List[str]
    status: str
    description: str
    retry_enabled: bool
    max_retries: int
    timeout_seconds: int
    consecutive_failures: int
    last_triggered_at: Optional[str] = None
    last_success_at: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class WebhookCreateResponse(WebhookResponse):
    """Webhook create response (includes secret)."""
    secret: str


class DeliveryLogResponse(BaseModel):
    """Delivery log response model."""
    id: str
    webhook_id: str
    event_id: str
    event_type: str
    url: str
    status: str
    attempts: int
    response_status: Optional[int] = None
    response_time_ms: Optional[int] = None
    created_at: str
    completed_at: Optional[str] = None


class DeliveryLogDetailResponse(DeliveryLogResponse):
    """Detailed delivery log response."""
    payload: Dict[str, Any]
    response_body: Optional[str] = None
    error_message: Optional[str] = None


class WebhookStatsResponse(BaseModel):
    """Webhook statistics response."""
    webhook_id: str
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    pending_deliveries: int
    avg_response_time_ms: float
    success_rate: float
    last_24h_deliveries: int
    last_24h_failures: int


class TestWebhookResponse(BaseModel):
    """Test webhook response."""
    success: bool
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[int] = None


class EventTypeInfo(BaseModel):
    """Event type information."""
    name: str
    description: str
    category: str


# Dependency injection
_delivery_service: Optional[WebhookDeliveryService] = None


def get_delivery_service() -> WebhookDeliveryService:
    global _delivery_service
    if _delivery_service is None:
        _delivery_service = WebhookDeliveryService()
    return _delivery_service


# Routes

@router.get("", response_model=List[WebhookResponse])
async def list_webhooks(
    organization_id: str = Query(..., description="Organization ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    service: WebhookDeliveryService = Depends(get_delivery_service),
):
    """List all webhooks for an organization."""
    webhooks = await service.list_webhooks(organization_id)

    if status:
        webhooks = [w for w in webhooks if w.status.value == status]

    return [
        WebhookResponse(
            id=w.id,
            organization_id=w.organization_id,
            url=w.url,
            events=w.events,
            status=w.status.value,
            description=w.description,
            retry_enabled=w.retry_enabled,
            max_retries=w.max_retries,
            timeout_seconds=w.timeout_seconds,
            consecutive_failures=w.consecutive_failures,
            last_triggered_at=w.last_triggered_at.isoformat() if w.last_triggered_at else None,
            last_success_at=w.last_success_at.isoformat() if w.last_success_at else None,
            created_at=w.created_at.isoformat(),
            updated_at=w.updated_at.isoformat(),
        )
        for w in webhooks
    ]


@router.post("", response_model=WebhookCreateResponse, status_code=201)
async def create_webhook(
    data: WebhookCreate,
    organization_id: str = Query(..., description="Organization ID"),
    service: WebhookDeliveryService = Depends(get_delivery_service),
):
    """Create a new webhook endpoint."""
    # Validate event types
    for event in data.events:
        if not event.endswith("*"):
            try:
                WebhookEventType(event)
            except ValueError:
                # Allow custom events but warn
                logger.warning(f"Unknown event type: {event}")

    # Generate secret if not provided
    secret = data.secret or WebhookSigner.generate_secret()

    webhook = Webhook(
        id=f"wh_{uuid4().hex[:12]}",
        organization_id=organization_id,
        url=str(data.url),
        events=data.events,
        secret=secret,
        description=data.description,
        headers=data.headers,
        retry_enabled=data.retry_enabled,
        max_retries=data.max_retries,
        timeout_seconds=data.timeout_seconds,
    )

    created = await service.register_webhook(webhook)

    return WebhookCreateResponse(
        id=created.id,
        organization_id=created.organization_id,
        url=created.url,
        events=created.events,
        status=created.status.value,
        description=created.description,
        retry_enabled=created.retry_enabled,
        max_retries=created.max_retries,
        timeout_seconds=created.timeout_seconds,
        consecutive_failures=created.consecutive_failures,
        last_triggered_at=None,
        last_success_at=None,
        created_at=created.created_at.isoformat(),
        updated_at=created.updated_at.isoformat(),
        secret=created.secret,
    )


@router.get("/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(
    webhook_id: str,
    service: WebhookDeliveryService = Depends(get_delivery_service),
):
    """Get webhook details."""
    webhook = await service.get_webhook(webhook_id)

    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")

    return WebhookResponse(
        id=webhook.id,
        organization_id=webhook.organization_id,
        url=webhook.url,
        events=webhook.events,
        status=webhook.status.value,
        description=webhook.description,
        retry_enabled=webhook.retry_enabled,
        max_retries=webhook.max_retries,
        timeout_seconds=webhook.timeout_seconds,
        consecutive_failures=webhook.consecutive_failures,
        last_triggered_at=webhook.last_triggered_at.isoformat() if webhook.last_triggered_at else None,
        last_success_at=webhook.last_success_at.isoformat() if webhook.last_success_at else None,
        created_at=webhook.created_at.isoformat(),
        updated_at=webhook.updated_at.isoformat(),
    )


@router.patch("/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(
    webhook_id: str,
    updates: WebhookUpdate,
    service: WebhookDeliveryService = Depends(get_delivery_service),
):
    """Update a webhook."""
    update_dict = updates.model_dump(exclude_unset=True)

    # Convert URL to string
    if "url" in update_dict and update_dict["url"]:
        update_dict["url"] = str(update_dict["url"])

    # Convert status to enum
    if "status" in update_dict:
        update_dict["status"] = WebhookStatus(update_dict["status"])

    webhook = await service.update_webhook(webhook_id, update_dict)

    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")

    return WebhookResponse(
        id=webhook.id,
        organization_id=webhook.organization_id,
        url=webhook.url,
        events=webhook.events,
        status=webhook.status.value,
        description=webhook.description,
        retry_enabled=webhook.retry_enabled,
        max_retries=webhook.max_retries,
        timeout_seconds=webhook.timeout_seconds,
        consecutive_failures=webhook.consecutive_failures,
        last_triggered_at=webhook.last_triggered_at.isoformat() if webhook.last_triggered_at else None,
        last_success_at=webhook.last_success_at.isoformat() if webhook.last_success_at else None,
        created_at=webhook.created_at.isoformat(),
        updated_at=webhook.updated_at.isoformat(),
    )


@router.delete("/{webhook_id}", status_code=204)
async def delete_webhook(
    webhook_id: str,
    service: WebhookDeliveryService = Depends(get_delivery_service),
):
    """Delete a webhook."""
    deleted = await service.delete_webhook(webhook_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Webhook not found")

    return None


@router.post("/{webhook_id}/test", response_model=TestWebhookResponse)
async def test_webhook(
    webhook_id: str,
    event_type: str = Query("test.ping", description="Event type to send"),
    service: WebhookDeliveryService = Depends(get_delivery_service),
):
    """Send a test event to a webhook."""
    result = await service.test_webhook(webhook_id, event_type)

    return TestWebhookResponse(
        success=result.success,
        status_code=result.status_code,
        response_body=result.response_body[:1000] if result.response_body else None,
        error_message=result.error_message,
        response_time_ms=result.response_time_ms,
    )


@router.get("/{webhook_id}/logs", response_model=List[DeliveryLogResponse])
async def get_delivery_logs(
    webhook_id: str,
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    service: WebhookDeliveryService = Depends(get_delivery_service),
):
    """Get delivery logs for a webhook."""
    status_enum = DeliveryStatus(status) if status else None
    logs = await service.get_delivery_logs(webhook_id, limit, status_enum)

    return [
        DeliveryLogResponse(
            id=log.id,
            webhook_id=log.webhook_id,
            event_id=log.event_id,
            event_type=log.event_type,
            url=log.url,
            status=log.status.value,
            attempts=log.total_attempts,
            response_status=log.response_status,
            response_time_ms=log.response_time_ms,
            created_at=log.created_at.isoformat(),
            completed_at=log.completed_at.isoformat() if log.completed_at else None,
        )
        for log in logs
    ]


@router.get("/{webhook_id}/stats", response_model=WebhookStatsResponse)
async def get_webhook_stats(
    webhook_id: str,
    service: WebhookDeliveryService = Depends(get_delivery_service),
):
    """Get statistics for a webhook."""
    stats = await service.get_webhook_stats(webhook_id)
    return WebhookStatsResponse(**stats.to_dict())


@router.post("/{webhook_id}/regenerate-secret")
async def regenerate_secret(
    webhook_id: str,
    service: WebhookDeliveryService = Depends(get_delivery_service),
):
    """Regenerate the signing secret for a webhook."""
    webhook = await service.get_webhook(webhook_id)

    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")

    new_secret = WebhookSigner.generate_secret()

    await service.update_webhook(webhook_id, {"secret": new_secret})

    return {"secret": new_secret}


@router.get("/events/types", response_model=List[EventTypeInfo])
async def list_event_types():
    """List all available webhook event types."""
    event_info = {
        # Call events
        "call.started": ("Call events", "A new call has been initiated"),
        "call.ringing": ("Call events", "The phone is ringing"),
        "call.answered": ("Call events", "The call was answered"),
        "call.ended": ("Call events", "The call has ended"),
        "call.failed": ("Call events", "The call failed to connect"),

        # Transcription events
        "transcription.partial": ("Transcription", "Partial transcription available"),
        "transcription.final": ("Transcription", "Final transcription for an utterance"),

        # Agent events
        "agent.speech.start": ("Agent events", "Agent started speaking"),
        "agent.speech.end": ("Agent events", "Agent finished speaking"),
        "agent.thinking": ("Agent events", "Agent is processing response"),
        "agent.tool_call": ("Agent events", "Agent invoked a tool/function"),

        # Conversation events
        "conversation.created": ("Conversation", "New conversation started"),
        "conversation.updated": ("Conversation", "Conversation was updated"),
        "conversation.ended": ("Conversation", "Conversation ended"),

        # Campaign events
        "campaign.started": ("Campaign", "Campaign started processing"),
        "campaign.completed": ("Campaign", "Campaign finished all calls"),
        "campaign.paused": ("Campaign", "Campaign was paused"),
        "campaign.resumed": ("Campaign", "Campaign was resumed"),
        "campaign.canceled": ("Campaign", "Campaign was canceled"),

        # Phone number events
        "phone_number.purchased": ("Phone numbers", "A phone number was purchased"),
        "phone_number.released": ("Phone numbers", "A phone number was released"),
        "phone_number.configured": ("Phone numbers", "Phone number configuration changed"),

        # Account events
        "account.credits_low": ("Account", "Account credits are running low"),
        "account.usage_limit": ("Account", "Usage limit was reached"),

        # Test
        "test.ping": ("Testing", "Test event for webhook verification"),
    }

    return [
        EventTypeInfo(name=name, category=info[0], description=info[1])
        for name, info in event_info.items()
    ]


# Webhook receiver endpoint (for testing/examples)
@router.post("/receive")
async def receive_webhook(request: Request):
    """
    Example webhook receiver endpoint.

    This endpoint demonstrates how to verify and process incoming webhooks.
    """
    # Get signature header
    signature = request.headers.get("X-Webhook-Signature")
    if not signature:
        raise HTTPException(status_code=401, detail="Missing signature")

    # Get raw body
    body = await request.body()
    payload = body.decode("utf-8")

    # In a real implementation, you would:
    # 1. Look up the webhook secret for this endpoint
    # 2. Verify the signature
    # 3. Process the event

    # For demo, just parse and return
    try:
        data = await request.json()
        return {
            "received": True,
            "event_type": data.get("type"),
            "event_id": data.get("id"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")
