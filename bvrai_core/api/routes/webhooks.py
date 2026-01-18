"""
Webhook API Routes

This module provides REST API endpoints for managing webhooks.
"""

import logging
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Path, Body
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..base import (
    APIResponse,
    ListResponse,
    NotFoundError,
    success_response,
    paginated_response,
)
from ..auth import AuthContext, Permission
from ..dependencies import get_db_session, get_auth_context
from ...database.repositories import WebhookRepository


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


# =============================================================================
# Event Types
# =============================================================================


WEBHOOK_EVENTS = [
    {"type": "call.started", "description": "Call has been initiated"},
    {"type": "call.answered", "description": "Call was answered"},
    {"type": "call.ended", "description": "Call has ended"},
    {"type": "call.failed", "description": "Call failed to connect"},
    {"type": "transcript.ready", "description": "Call transcript is available"},
    {"type": "recording.ready", "description": "Call recording is available"},
    {"type": "function.call", "description": "Agent triggered a function call"},
    {"type": "agent.updated", "description": "Agent configuration was updated"},
    {"type": "agent.deleted", "description": "Agent was deleted"},
    {"type": "campaign.started", "description": "Campaign has started"},
    {"type": "campaign.completed", "description": "Campaign has completed"},
    {"type": "*", "description": "All events (wildcard)"},
]

VALID_EVENT_TYPES = [e["type"] for e in WEBHOOK_EVENTS]


# =============================================================================
# Request/Response Models
# =============================================================================


class WebhookCreateRequest(BaseModel):
    """Request to create a webhook."""

    name: str = Field(..., description="Webhook name", max_length=255)
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to subscribe to")
    description: Optional[str] = Field(default=None, max_length=500)

    # Authentication
    auth_type: str = Field(default="hmac", description="Auth type: none, basic, bearer, hmac")
    auth_value: Optional[str] = Field(default=None, description="Auth value (for basic/bearer)")

    # Settings
    is_active: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: int = Field(default=60, ge=10, le=3600)
    timeout_seconds: int = Field(default=30, ge=5, le=120)

    # Filtering
    agent_ids: Optional[List[str]] = Field(default=None, description="Filter by specific agents")

    # Metadata
    extra_data: Dict[str, Any] = Field(default_factory=dict)


class WebhookUpdateRequest(BaseModel):
    """Request to update a webhook."""

    name: Optional[str] = Field(default=None, max_length=255)
    url: Optional[str] = Field(default=None)
    events: Optional[List[str]] = Field(default=None)
    description: Optional[str] = Field(default=None, max_length=500)
    auth_type: Optional[str] = Field(default=None)
    auth_value: Optional[str] = Field(default=None)
    is_active: Optional[bool] = Field(default=None)
    max_retries: Optional[int] = Field(default=None, ge=0, le=10)
    retry_delay_seconds: Optional[int] = Field(default=None, ge=10, le=3600)
    timeout_seconds: Optional[int] = Field(default=None, ge=5, le=120)
    agent_ids: Optional[List[str]] = Field(default=None)


class WebhookResponse(BaseModel):
    """Webhook response."""

    id: str
    organization_id: str
    name: str
    url: str
    events: List[str]
    description: Optional[str] = None

    # Authentication (secret is write-only, never returned in full)
    auth_type: str = "hmac"
    has_secret: bool = False

    # Settings
    is_active: bool = True
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 30

    # Filtering
    agent_ids: Optional[List[str]] = None

    # Stats
    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    last_triggered_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None

    # Timestamps
    created_at: datetime
    updated_at: datetime


class WebhookDeliveryResponse(BaseModel):
    """Webhook delivery record."""

    id: str
    webhook_id: str
    event_type: str
    event_id: str

    # Request info
    request_url: str

    # Response info
    status: str  # pending, success, failed, retrying
    response_status: Optional[int] = None
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None

    # Retry info
    attempt_number: int = 1
    next_retry_at: Optional[datetime] = None

    # Timestamps
    created_at: datetime


# =============================================================================
# Helper Functions
# =============================================================================


def webhook_to_response(webhook) -> dict:
    """Convert database webhook model to response dict."""
    return {
        "id": webhook.id,
        "organization_id": webhook.organization_id,
        "name": webhook.name,
        "url": webhook.url,
        "events": webhook.events or [],
        "description": webhook.description,
        "auth_type": webhook.auth_type,
        "has_secret": bool(webhook.secret),
        "is_active": webhook.is_active,
        "max_retries": webhook.max_retries,
        "retry_delay_seconds": webhook.retry_delay_seconds,
        "timeout_seconds": webhook.timeout_seconds,
        "agent_ids": webhook.agent_ids,
        "total_deliveries": webhook.total_deliveries,
        "successful_deliveries": webhook.successful_deliveries,
        "failed_deliveries": webhook.failed_deliveries,
        "last_triggered_at": webhook.last_triggered_at,
        "last_success_at": webhook.last_success_at,
        "last_failure_at": webhook.last_failure_at,
        "created_at": webhook.created_at,
        "updated_at": webhook.updated_at,
    }


def delivery_to_response(delivery) -> dict:
    """Convert database webhook delivery model to response dict."""
    return {
        "id": delivery.id,
        "webhook_id": delivery.webhook_id,
        "event_type": delivery.event_type,
        "event_id": delivery.event_id,
        "request_url": delivery.request_url,
        "status": delivery.status,
        "response_status": delivery.response_status,
        "error_message": delivery.error_message,
        "duration_ms": delivery.duration_ms,
        "attempt_number": delivery.attempt_number,
        "next_retry_at": delivery.next_retry_at,
        "created_at": delivery.created_at,
    }


# =============================================================================
# Routes
# =============================================================================


@router.get(
    "/events",
    response_model=APIResponse[List[Dict[str, str]]],
    summary="List Event Types",
    description="List all available webhook event types.",
)
async def list_event_types(auth: AuthContext = Depends(get_auth_context)):
    """List available webhook event types."""
    auth.require_permission(Permission.WEBHOOKS_READ)
    return success_response(WEBHOOK_EVENTS)


@router.post(
    "",
    response_model=APIResponse[WebhookResponse],
    status_code=201,
    summary="Create Webhook",
)
async def create_webhook(
    request: WebhookCreateRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new webhook endpoint."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)

    # Generate signing secret for HMAC auth
    secret = None
    if request.auth_type == "hmac":
        secret = secrets.token_urlsafe(32)

    repo = WebhookRepository(db)
    webhook = await repo.create(
        organization_id=auth.organization_id,
        name=request.name,
        description=request.description,
        url=request.url,
        events=request.events,
        secret=secret,
        auth_type=request.auth_type,
        auth_value=request.auth_value,
        is_active=request.is_active,
        max_retries=request.max_retries,
        retry_delay_seconds=request.retry_delay_seconds,
        timeout_seconds=request.timeout_seconds,
        agent_ids=request.agent_ids,
        extra_data=request.extra_data,
    )

    await db.commit()

    logger.info(f"Created webhook {webhook.id} for org {auth.organization_id}")

    response = webhook_to_response(webhook)
    # Include the secret in the response only on creation
    if secret:
        response["secret"] = secret

    return success_response(response)


@router.get(
    "",
    response_model=ListResponse[WebhookResponse],
    summary="List Webhooks",
)
async def list_webhooks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    is_active: Optional[bool] = Query(None),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """List webhooks."""
    auth.require_permission(Permission.WEBHOOKS_READ)

    repo = WebhookRepository(db)

    skip = (page - 1) * page_size
    webhooks = await repo.list_by_organization(
        organization_id=auth.organization_id,
        is_active=is_active,
        skip=skip,
        limit=page_size,
    )

    total = await repo.count_by_organization(
        organization_id=auth.organization_id,
        is_active=is_active,
    )

    items = [webhook_to_response(w) for w in webhooks]

    return paginated_response(
        items=items,
        page=page,
        page_size=page_size,
        total_items=total,
    )


@router.get(
    "/{webhook_id}",
    response_model=APIResponse[WebhookResponse],
    summary="Get Webhook",
)
async def get_webhook(
    webhook_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Get webhook details."""
    auth.require_permission(Permission.WEBHOOKS_READ)

    repo = WebhookRepository(db)
    webhook = await repo.get_by_id(webhook_id)

    if not webhook or webhook.organization_id != auth.organization_id or webhook.is_deleted:
        raise NotFoundError("Webhook", webhook_id)

    return success_response(webhook_to_response(webhook))


@router.patch(
    "/{webhook_id}",
    response_model=APIResponse[WebhookResponse],
    summary="Update Webhook",
)
async def update_webhook(
    webhook_id: str = Path(...),
    request: WebhookUpdateRequest = Body(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Update a webhook."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)

    repo = WebhookRepository(db)
    webhook = await repo.get_by_id(webhook_id)

    if not webhook or webhook.organization_id != auth.organization_id or webhook.is_deleted:
        raise NotFoundError("Webhook", webhook_id)

    # Build update data
    update_data = {}
    if request.name is not None:
        update_data["name"] = request.name
    if request.url is not None:
        update_data["url"] = request.url
    if request.events is not None:
        update_data["events"] = request.events
    if request.description is not None:
        update_data["description"] = request.description
    if request.auth_type is not None:
        update_data["auth_type"] = request.auth_type
    if request.auth_value is not None:
        update_data["auth_value"] = request.auth_value
    if request.is_active is not None:
        update_data["is_active"] = request.is_active
    if request.max_retries is not None:
        update_data["max_retries"] = request.max_retries
    if request.retry_delay_seconds is not None:
        update_data["retry_delay_seconds"] = request.retry_delay_seconds
    if request.timeout_seconds is not None:
        update_data["timeout_seconds"] = request.timeout_seconds
    if request.agent_ids is not None:
        update_data["agent_ids"] = request.agent_ids

    if update_data:
        webhook = await repo.update(webhook_id, **update_data)

    await db.commit()

    logger.info(f"Updated webhook {webhook_id}")

    return success_response(webhook_to_response(webhook))


@router.delete(
    "/{webhook_id}",
    status_code=204,
    summary="Delete Webhook",
)
async def delete_webhook(
    webhook_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a webhook."""
    auth.require_permission(Permission.WEBHOOKS_DELETE)

    repo = WebhookRepository(db)
    webhook = await repo.get_by_id(webhook_id)

    if not webhook or webhook.organization_id != auth.organization_id or webhook.is_deleted:
        raise NotFoundError("Webhook", webhook_id)

    await repo.soft_delete(webhook_id)
    await db.commit()

    logger.info(f"Deleted webhook {webhook_id}")

    return None


@router.post(
    "/{webhook_id}/enable",
    response_model=APIResponse[WebhookResponse],
    summary="Enable Webhook",
)
async def enable_webhook(
    webhook_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Enable a webhook."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)

    repo = WebhookRepository(db)
    webhook = await repo.get_by_id(webhook_id)

    if not webhook or webhook.organization_id != auth.organization_id or webhook.is_deleted:
        raise NotFoundError("Webhook", webhook_id)

    webhook = await repo.update(webhook_id, is_active=True)
    await db.commit()

    logger.info(f"Enabled webhook {webhook_id}")

    return success_response(webhook_to_response(webhook))


@router.post(
    "/{webhook_id}/disable",
    response_model=APIResponse[WebhookResponse],
    summary="Disable Webhook",
)
async def disable_webhook(
    webhook_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Disable a webhook."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)

    repo = WebhookRepository(db)
    webhook = await repo.get_by_id(webhook_id)

    if not webhook or webhook.organization_id != auth.organization_id or webhook.is_deleted:
        raise NotFoundError("Webhook", webhook_id)

    webhook = await repo.update(webhook_id, is_active=False)
    await db.commit()

    logger.info(f"Disabled webhook {webhook_id}")

    return success_response(webhook_to_response(webhook))


@router.post(
    "/{webhook_id}/rotate-secret",
    response_model=APIResponse[Dict[str, Any]],
    summary="Rotate Secret",
    description="Generate a new signing secret for the webhook.",
)
async def rotate_secret(
    webhook_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Rotate the webhook signing secret."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)

    repo = WebhookRepository(db)
    webhook = await repo.get_by_id(webhook_id)

    if not webhook or webhook.organization_id != auth.organization_id or webhook.is_deleted:
        raise NotFoundError("Webhook", webhook_id)

    # Generate new secret
    new_secret = secrets.token_urlsafe(32)
    await repo.update(webhook_id, secret=new_secret)
    await db.commit()

    logger.info(f"Rotated secret for webhook {webhook_id}")

    return success_response({
        "webhook_id": webhook_id,
        "secret": new_secret,
        "message": "Secret rotated successfully. Store this secret securely - it won't be shown again.",
    })


@router.post(
    "/{webhook_id}/test",
    response_model=APIResponse[Dict[str, Any]],
    summary="Test Webhook",
    description="Send a test event to the webhook.",
)
async def test_webhook(
    webhook_id: str = Path(...),
    event_type: str = Body(default="test.ping", embed=True),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Send a test event to webhook."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)

    repo = WebhookRepository(db)
    webhook = await repo.get_by_id(webhook_id)

    if not webhook or webhook.organization_id != auth.organization_id or webhook.is_deleted:
        raise NotFoundError("Webhook", webhook_id)

    # Create test delivery record
    import uuid
    event_id = str(uuid.uuid4())
    delivery = await repo.create_delivery(
        webhook_id=webhook_id,
        event_type=event_type,
        event_id=event_id,
        request_url=webhook.url,
        request_body={
            "event": event_type,
            "event_id": event_id,
            "organization_id": auth.organization_id,
            "webhook_id": webhook_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "test": True,
                "message": "This is a test webhook delivery",
            },
        },
    )

    await db.commit()

    # In production, this would actually send the webhook via background task
    # For now, we just create the delivery record

    return success_response({
        "delivery_id": delivery.id,
        "webhook_id": webhook_id,
        "event_type": event_type,
        "status": "queued",
        "message": "Test webhook delivery queued",
    })


@router.get(
    "/{webhook_id}/deliveries",
    response_model=ListResponse[WebhookDeliveryResponse],
    summary="List Deliveries",
    description="List webhook delivery history.",
)
async def list_deliveries(
    webhook_id: str = Path(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """List webhook deliveries."""
    auth.require_permission(Permission.WEBHOOKS_READ)

    repo = WebhookRepository(db)
    webhook = await repo.get_by_id(webhook_id)

    if not webhook or webhook.organization_id != auth.organization_id or webhook.is_deleted:
        raise NotFoundError("Webhook", webhook_id)

    skip = (page - 1) * page_size
    deliveries = await repo.get_deliveries(
        webhook_id=webhook_id,
        status=status,
        skip=skip,
        limit=page_size,
    )

    # For total count, we'd need another query - for now estimate
    total = len(deliveries) if len(deliveries) < page_size else page_size * 2

    items = [delivery_to_response(d) for d in deliveries]

    return paginated_response(
        items=items,
        page=page,
        page_size=page_size,
        total_items=total,
    )


@router.post(
    "/{webhook_id}/deliveries/{delivery_id}/retry",
    response_model=APIResponse[WebhookDeliveryResponse],
    summary="Retry Delivery",
    description="Retry a failed webhook delivery.",
)
async def retry_delivery(
    webhook_id: str = Path(...),
    delivery_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Retry a failed delivery."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)

    repo = WebhookRepository(db)
    webhook = await repo.get_by_id(webhook_id)

    if not webhook or webhook.organization_id != auth.organization_id or webhook.is_deleted:
        raise NotFoundError("Webhook", webhook_id)

    # Get delivery and mark for retry
    delivery = await repo.update_delivery(
        delivery_id=delivery_id,
        status="retrying",
    )

    if not delivery:
        raise NotFoundError("WebhookDelivery", delivery_id)

    await db.commit()

    # In production, this would trigger actual retry via background task

    return success_response(delivery_to_response(delivery))
