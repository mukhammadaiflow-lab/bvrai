"""
Webhook API Routes

This module provides REST API endpoints for managing webhooks.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Path, Body
from pydantic import BaseModel, Field

from ..base import (
    APIResponse,
    ListResponse,
    NotFoundError,
    success_response,
    paginated_response,
)
from ..auth import AuthContext, Permission


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


class WebhookEvent(str):
    """Webhook event types."""

    # Call events
    CALL_STARTED = "call.started"
    CALL_ANSWERED = "call.answered"
    CALL_ENDED = "call.ended"
    CALL_FAILED = "call.failed"

    # Transcript events
    TRANSCRIPT_READY = "transcript.ready"

    # Recording events
    RECORDING_READY = "recording.ready"

    # Agent events
    AGENT_UPDATED = "agent.updated"
    AGENT_DELETED = "agent.deleted"

    # Campaign events
    CAMPAIGN_STARTED = "campaign.started"
    CAMPAIGN_COMPLETED = "campaign.completed"

    # Function call events
    FUNCTION_CALL = "function.call"


class WebhookCreateRequest(BaseModel):
    """Request to create a webhook."""

    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to subscribe to")
    description: Optional[str] = Field(default=None, max_length=200)

    # Security
    secret: Optional[str] = Field(
        default=None,
        description="Signing secret (auto-generated if not provided)",
    )

    # Settings
    enabled: bool = Field(default=True)
    retry_on_failure: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0, le=10)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WebhookResponse(BaseModel):
    """Webhook response."""

    id: str
    organization_id: str
    url: str
    events: List[str]
    description: Optional[str] = None

    # Security (secret is write-only, never returned)
    secret_last4: Optional[str] = None

    # Settings
    enabled: bool = True
    retry_on_failure: bool = True
    max_retries: int = 3

    # Stats
    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    last_delivery_at: Optional[datetime] = None
    last_delivery_status: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = {}

    # Timestamps
    created_at: datetime
    updated_at: datetime


class WebhookDelivery(BaseModel):
    """Webhook delivery record."""

    id: str
    webhook_id: str
    event_type: str
    payload: Dict[str, Any]

    # Delivery info
    status: str  # pending, delivered, failed
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None

    # Timing
    attempts: int = 0
    next_retry_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None

    # Timestamps
    created_at: datetime


@router.post(
    "",
    response_model=APIResponse[WebhookResponse],
    status_code=201,
    summary="Create Webhook",
)
async def create_webhook(
    request: WebhookCreateRequest,
    auth: AuthContext = Depends(),
):
    """Create a new webhook endpoint."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)

    webhook = WebhookResponse(
        id="wh_" + "x" * 24,
        organization_id=auth.organization_id,
        url=request.url,
        events=request.events,
        description=request.description,
        enabled=request.enabled,
        retry_on_failure=request.retry_on_failure,
        max_retries=request.max_retries,
        secret_last4="****",  # Would show last 4 of generated secret
        metadata=request.metadata,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    return success_response(webhook.dict())


@router.get(
    "",
    response_model=ListResponse[WebhookResponse],
    summary="List Webhooks",
)
async def list_webhooks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    enabled: Optional[bool] = Query(None),
    auth: AuthContext = Depends(),
):
    """List webhooks."""
    auth.require_permission(Permission.WEBHOOKS_READ)

    return paginated_response(items=[], page=page, page_size=page_size, total_items=0)


@router.get("/{webhook_id}", response_model=APIResponse[WebhookResponse])
async def get_webhook(webhook_id: str = Path(...), auth: AuthContext = Depends()):
    """Get webhook details."""
    auth.require_permission(Permission.WEBHOOKS_READ)
    raise NotFoundError("Webhook", webhook_id)


@router.patch("/{webhook_id}", response_model=APIResponse[WebhookResponse])
async def update_webhook(
    webhook_id: str = Path(...),
    request: WebhookCreateRequest = Body(...),
    auth: AuthContext = Depends(),
):
    """Update a webhook."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)
    raise NotFoundError("Webhook", webhook_id)


@router.delete("/{webhook_id}", status_code=204)
async def delete_webhook(webhook_id: str = Path(...), auth: AuthContext = Depends()):
    """Delete a webhook."""
    auth.require_permission(Permission.WEBHOOKS_DELETE)
    raise NotFoundError("Webhook", webhook_id)


@router.post("/{webhook_id}/enable", response_model=APIResponse[WebhookResponse])
async def enable_webhook(webhook_id: str = Path(...), auth: AuthContext = Depends()):
    """Enable a webhook."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)
    raise NotFoundError("Webhook", webhook_id)


@router.post("/{webhook_id}/disable", response_model=APIResponse[WebhookResponse])
async def disable_webhook(webhook_id: str = Path(...), auth: AuthContext = Depends()):
    """Disable a webhook."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)
    raise NotFoundError("Webhook", webhook_id)


@router.post(
    "/{webhook_id}/test",
    response_model=APIResponse[Dict[str, Any]],
    summary="Test Webhook",
    description="Send a test event to the webhook.",
)
async def test_webhook(
    webhook_id: str = Path(...),
    event_type: str = Body(default="test.event", embed=True),
    auth: AuthContext = Depends(),
):
    """Send a test event to webhook."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)
    raise NotFoundError("Webhook", webhook_id)


@router.post("/{webhook_id}/rotate-secret", response_model=APIResponse[Dict[str, Any]])
async def rotate_secret(webhook_id: str = Path(...), auth: AuthContext = Depends()):
    """Rotate the webhook signing secret."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)
    raise NotFoundError("Webhook", webhook_id)


@router.get(
    "/{webhook_id}/deliveries",
    response_model=ListResponse[WebhookDelivery],
    summary="List Deliveries",
)
async def list_deliveries(
    webhook_id: str = Path(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None),
    auth: AuthContext = Depends(),
):
    """List webhook deliveries."""
    auth.require_permission(Permission.WEBHOOKS_READ)
    raise NotFoundError("Webhook", webhook_id)


@router.post(
    "/{webhook_id}/deliveries/{delivery_id}/retry",
    response_model=APIResponse[WebhookDelivery],
)
async def retry_delivery(
    webhook_id: str = Path(...),
    delivery_id: str = Path(...),
    auth: AuthContext = Depends(),
):
    """Retry a failed delivery."""
    auth.require_permission(Permission.WEBHOOKS_WRITE)
    raise NotFoundError("WebhookDelivery", delivery_id)


@router.get(
    "/events",
    response_model=APIResponse[List[Dict[str, str]]],
    summary="List Event Types",
    description="List all available webhook event types.",
)
async def list_event_types(auth: AuthContext = Depends()):
    """List available webhook event types."""
    auth.require_permission(Permission.WEBHOOKS_READ)

    events = [
        {"type": "call.started", "description": "Call has been initiated"},
        {"type": "call.answered", "description": "Call was answered"},
        {"type": "call.ended", "description": "Call has ended"},
        {"type": "call.failed", "description": "Call failed to connect"},
        {"type": "transcript.ready", "description": "Call transcript is available"},
        {"type": "recording.ready", "description": "Call recording is available"},
        {"type": "function.call", "description": "Agent triggered a function call"},
        {"type": "campaign.started", "description": "Campaign has started"},
        {"type": "campaign.completed", "description": "Campaign has completed"},
    ]

    return success_response(events)
