"""
Integrations API Routes

Handles:
- Third-party integrations
- OAuth connections
- Webhooks configuration
- Integration data sync
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_db_session,
    get_current_user,
    get_current_tenant,
    UserContext,
    TenantContext,
    require_permissions,
    require_feature,
)

router = APIRouter(prefix="/integrations")


# ============================================================================
# Schemas
# ============================================================================

class IntegrationType(str, Enum):
    """Integration types."""
    CRM = "crm"
    CALENDAR = "calendar"
    HELPDESK = "helpdesk"
    PAYMENT = "payment"
    MESSAGING = "messaging"
    STORAGE = "storage"
    ANALYTICS = "analytics"
    CUSTOM = "custom"


class IntegrationProvider(str, Enum):
    """Integration providers."""
    # CRM
    SALESFORCE = "salesforce"
    HUBSPOT = "hubspot"
    PIPEDRIVE = "pipedrive"
    ZOHO = "zoho"
    # Calendar
    GOOGLE_CALENDAR = "google_calendar"
    OUTLOOK = "outlook"
    CALENDLY = "calendly"
    # Helpdesk
    ZENDESK = "zendesk"
    FRESHDESK = "freshdesk"
    INTERCOM = "intercom"
    # Payment
    STRIPE = "stripe"
    SQUARE = "square"
    # Messaging
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    # Storage
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    # Custom
    WEBHOOK = "webhook"
    API = "api"


class IntegrationStatus(str, Enum):
    """Integration status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    PENDING = "pending"


class AvailableIntegration(BaseModel):
    """Available integration info."""
    provider: IntegrationProvider
    name: str
    type: IntegrationType
    description: str
    logo_url: str
    features: List[str]
    requires_oauth: bool
    is_available: bool


class IntegrationResponse(BaseModel):
    """Integration response."""
    id: str
    provider: IntegrationProvider
    name: str
    type: IntegrationType
    status: IntegrationStatus
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    last_sync_at: Optional[datetime]
    error_message: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]


class IntegrationListResponse(BaseModel):
    """Integration list response."""
    integrations: List[IntegrationResponse]
    total: int


class ConnectIntegrationRequest(BaseModel):
    """Connect integration request."""
    provider: IntegrationProvider
    config: Dict[str, Any] = {}
    name: Optional[str] = None


class UpdateIntegrationRequest(BaseModel):
    """Update integration request."""
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None


class WebhookConfig(BaseModel):
    """Webhook configuration."""
    url: HttpUrl
    secret: Optional[str] = None
    events: List[str] = []
    headers: Dict[str, str] = {}
    retry_count: int = Field(3, ge=0, le=10)
    timeout_seconds: int = Field(30, ge=5, le=120)


class WebhookResponse(BaseModel):
    """Webhook response."""
    id: str
    url: str
    events: List[str]
    status: str
    last_triggered_at: Optional[datetime]
    success_count: int
    failure_count: int
    created_at: datetime


class WebhookListResponse(BaseModel):
    """Webhook list response."""
    webhooks: List[WebhookResponse]
    total: int


class WebhookTestResponse(BaseModel):
    """Webhook test response."""
    success: bool
    status_code: int
    response_time_ms: int
    error: Optional[str]


class SyncRequest(BaseModel):
    """Sync request."""
    full_sync: bool = False
    entity_types: Optional[List[str]] = None


class SyncResponse(BaseModel):
    """Sync response."""
    sync_id: str
    status: str
    entities_synced: int
    started_at: datetime
    completed_at: Optional[datetime]


# ============================================================================
# Available Integrations
# ============================================================================

@router.get("/available", response_model=List[AvailableIntegration])
async def list_available_integrations(
    type_filter: Optional[IntegrationType] = Query(None, alias="type"),
    user: UserContext = Depends(get_current_user),
):
    """List available integrations."""
    integrations = [
        # CRM
        AvailableIntegration(
            provider=IntegrationProvider.SALESFORCE,
            name="Salesforce",
            type=IntegrationType.CRM,
            description="Sync contacts, leads, and opportunities",
            logo_url="/integrations/salesforce.svg",
            features=["Contact sync", "Lead creation", "Opportunity tracking"],
            requires_oauth=True,
            is_available=True,
        ),
        AvailableIntegration(
            provider=IntegrationProvider.HUBSPOT,
            name="HubSpot",
            type=IntegrationType.CRM,
            description="Integrate with HubSpot CRM",
            logo_url="/integrations/hubspot.svg",
            features=["Contact sync", "Deal tracking", "Activity logging"],
            requires_oauth=True,
            is_available=True,
        ),
        # Calendar
        AvailableIntegration(
            provider=IntegrationProvider.GOOGLE_CALENDAR,
            name="Google Calendar",
            type=IntegrationType.CALENDAR,
            description="Schedule appointments and sync events",
            logo_url="/integrations/google-calendar.svg",
            features=["Event creation", "Availability check", "Reminders"],
            requires_oauth=True,
            is_available=True,
        ),
        AvailableIntegration(
            provider=IntegrationProvider.CALENDLY,
            name="Calendly",
            type=IntegrationType.CALENDAR,
            description="Schedule meetings with Calendly",
            logo_url="/integrations/calendly.svg",
            features=["Booking links", "Availability sync", "Confirmations"],
            requires_oauth=True,
            is_available=True,
        ),
        # Helpdesk
        AvailableIntegration(
            provider=IntegrationProvider.ZENDESK,
            name="Zendesk",
            type=IntegrationType.HELPDESK,
            description="Create and manage support tickets",
            logo_url="/integrations/zendesk.svg",
            features=["Ticket creation", "Status updates", "Agent routing"],
            requires_oauth=True,
            is_available=True,
        ),
        # Messaging
        AvailableIntegration(
            provider=IntegrationProvider.SLACK,
            name="Slack",
            type=IntegrationType.MESSAGING,
            description="Send notifications to Slack channels",
            logo_url="/integrations/slack.svg",
            features=["Channel messages", "Direct messages", "Rich formatting"],
            requires_oauth=True,
            is_available=True,
        ),
        # Custom
        AvailableIntegration(
            provider=IntegrationProvider.WEBHOOK,
            name="Custom Webhook",
            type=IntegrationType.CUSTOM,
            description="Send events to your own endpoints",
            logo_url="/integrations/webhook.svg",
            features=["Custom events", "Retry logic", "Signature verification"],
            requires_oauth=False,
            is_available=True,
        ),
    ]

    if type_filter:
        integrations = [i for i in integrations if i.type == type_filter]

    return integrations


# ============================================================================
# Connected Integrations
# ============================================================================

@router.get("", response_model=IntegrationListResponse)
async def list_integrations(
    type_filter: Optional[IntegrationType] = Query(None, alias="type"),
    status_filter: Optional[IntegrationStatus] = Query(None, alias="status"),
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """List connected integrations."""
    from app.integrations import IntegrationService

    service = IntegrationService(db)

    integrations = await service.list_integrations(
        tenant_id=tenant.tenant_id,
        type=type_filter.value if type_filter else None,
        status=status_filter.value if status_filter else None,
    )

    return IntegrationListResponse(
        integrations=[
            IntegrationResponse(
                id=i.id,
                provider=IntegrationProvider(i.provider),
                name=i.name,
                type=IntegrationType(i.type),
                status=IntegrationStatus(i.status),
                config=i.config or {},
                metadata=i.metadata or {},
                last_sync_at=i.last_sync_at,
                error_message=i.error_message,
                created_at=i.created_at,
                updated_at=i.updated_at,
            )
            for i in integrations
        ],
        total=len(integrations),
    )


@router.post("", response_model=IntegrationResponse, status_code=status.HTTP_201_CREATED)
async def connect_integration(
    data: ConnectIntegrationRequest,
    user: UserContext = Depends(require_permissions("integrations:create")),
    tenant: TenantContext = Depends(require_feature("integrations")),
    db: AsyncSession = Depends(get_db_session),
):
    """Connect a new integration."""
    from app.integrations import IntegrationService

    service = IntegrationService(db)

    # Check if already connected
    existing = await service.get_by_provider(tenant.tenant_id, data.provider.value)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Integration {data.provider.value} already connected",
        )

    integration = await service.connect(
        tenant_id=tenant.tenant_id,
        provider=data.provider.value,
        config=data.config,
        name=data.name,
    )

    return IntegrationResponse(
        id=integration.id,
        provider=IntegrationProvider(integration.provider),
        name=integration.name,
        type=IntegrationType(integration.type),
        status=IntegrationStatus(integration.status),
        config=integration.config or {},
        metadata=integration.metadata or {},
        last_sync_at=integration.last_sync_at,
        error_message=integration.error_message,
        created_at=integration.created_at,
        updated_at=integration.updated_at,
    )


@router.get("/{integration_id}", response_model=IntegrationResponse)
async def get_integration(
    integration_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get integration details."""
    from app.integrations import IntegrationService

    service = IntegrationService(db)
    integration = await service.get(str(integration_id), tenant.tenant_id)

    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Integration not found",
        )

    return IntegrationResponse(
        id=integration.id,
        provider=IntegrationProvider(integration.provider),
        name=integration.name,
        type=IntegrationType(integration.type),
        status=IntegrationStatus(integration.status),
        config=integration.config or {},
        metadata=integration.metadata or {},
        last_sync_at=integration.last_sync_at,
        error_message=integration.error_message,
        created_at=integration.created_at,
        updated_at=integration.updated_at,
    )


@router.patch("/{integration_id}", response_model=IntegrationResponse)
async def update_integration(
    integration_id: UUID,
    data: UpdateIntegrationRequest,
    user: UserContext = Depends(require_permissions("integrations:update")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Update integration configuration."""
    from app.integrations import IntegrationService

    service = IntegrationService(db)

    integration = await service.update(
        integration_id=str(integration_id),
        tenant_id=tenant.tenant_id,
        **data.model_dump(exclude_unset=True),
    )

    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Integration not found",
        )

    return IntegrationResponse(
        id=integration.id,
        provider=IntegrationProvider(integration.provider),
        name=integration.name,
        type=IntegrationType(integration.type),
        status=IntegrationStatus(integration.status),
        config=integration.config or {},
        metadata=integration.metadata or {},
        last_sync_at=integration.last_sync_at,
        error_message=integration.error_message,
        created_at=integration.created_at,
        updated_at=integration.updated_at,
    )


@router.delete("/{integration_id}", status_code=status.HTTP_204_NO_CONTENT)
async def disconnect_integration(
    integration_id: UUID,
    user: UserContext = Depends(require_permissions("integrations:delete")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Disconnect an integration."""
    from app.integrations import IntegrationService

    service = IntegrationService(db)
    disconnected = await service.disconnect(str(integration_id), tenant.tenant_id)

    if not disconnected:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Integration not found",
        )


# ============================================================================
# OAuth Flow
# ============================================================================

@router.get("/{provider}/oauth/authorize")
async def oauth_authorize(
    provider: IntegrationProvider,
    request: Request,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Initiate OAuth flow for integration."""
    from app.integrations import IntegrationService

    service = IntegrationService()

    redirect_uri = str(request.url_for(
        "oauth_callback",
        provider=provider.value,
    ))

    auth_url = await service.get_oauth_url(
        provider=provider.value,
        tenant_id=tenant.tenant_id,
        redirect_uri=redirect_uri,
    )

    return {"authorization_url": auth_url}


@router.get("/{provider}/oauth/callback")
async def oauth_callback(
    provider: IntegrationProvider,
    code: str,
    state: Optional[str] = None,
    request: Request = None,
    db: AsyncSession = Depends(get_db_session),
):
    """Handle OAuth callback."""
    from app.integrations import IntegrationService

    service = IntegrationService(db)

    try:
        integration = await service.handle_oauth_callback(
            provider=provider.value,
            code=code,
            state=state,
        )

        return {
            "status": "success",
            "integration_id": integration.id,
            "message": f"Successfully connected {provider.value}",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth failed: {str(e)}",
        )


# ============================================================================
# Sync Operations
# ============================================================================

@router.post("/{integration_id}/sync", response_model=SyncResponse)
async def sync_integration(
    integration_id: UUID,
    data: SyncRequest,
    user: UserContext = Depends(require_permissions("integrations:sync")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Trigger data sync for integration."""
    from app.integrations import IntegrationService

    service = IntegrationService(db)

    integration = await service.get(str(integration_id), tenant.tenant_id)
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Integration not found",
        )

    sync = await service.start_sync(
        integration_id=str(integration_id),
        full_sync=data.full_sync,
        entity_types=data.entity_types,
    )

    return SyncResponse(
        sync_id=sync.id,
        status=sync.status,
        entities_synced=sync.entities_synced,
        started_at=sync.started_at,
        completed_at=sync.completed_at,
    )


@router.get("/{integration_id}/sync/status", response_model=SyncResponse)
async def get_sync_status(
    integration_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get sync status for integration."""
    from app.integrations import IntegrationService

    service = IntegrationService(db)

    sync = await service.get_last_sync(str(integration_id), tenant.tenant_id)

    if not sync:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No sync history found",
        )

    return SyncResponse(
        sync_id=sync.id,
        status=sync.status,
        entities_synced=sync.entities_synced,
        started_at=sync.started_at,
        completed_at=sync.completed_at,
    )


# ============================================================================
# Webhooks
# ============================================================================

@router.get("/webhooks", response_model=WebhookListResponse)
async def list_webhooks(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """List configured webhooks."""
    from app.webhooks import WebhookService

    service = WebhookService(db)
    webhooks = await service.list_webhooks(tenant.tenant_id)

    return WebhookListResponse(
        webhooks=[
            WebhookResponse(
                id=w.id,
                url=w.url,
                events=w.events or [],
                status=w.status,
                last_triggered_at=w.last_triggered_at,
                success_count=w.success_count,
                failure_count=w.failure_count,
                created_at=w.created_at,
            )
            for w in webhooks
        ],
        total=len(webhooks),
    )


@router.post("/webhooks", response_model=WebhookResponse, status_code=status.HTTP_201_CREATED)
async def create_webhook(
    data: WebhookConfig,
    user: UserContext = Depends(require_permissions("webhooks:create")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new webhook."""
    from app.webhooks import WebhookService

    service = WebhookService(db)

    webhook = await service.create_webhook(
        tenant_id=tenant.tenant_id,
        url=str(data.url),
        events=data.events,
        secret=data.secret,
        headers=data.headers,
        retry_count=data.retry_count,
        timeout_seconds=data.timeout_seconds,
    )

    return WebhookResponse(
        id=webhook.id,
        url=webhook.url,
        events=webhook.events or [],
        status=webhook.status,
        last_triggered_at=webhook.last_triggered_at,
        success_count=webhook.success_count,
        failure_count=webhook.failure_count,
        created_at=webhook.created_at,
    )


@router.delete("/webhooks/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook(
    webhook_id: UUID,
    user: UserContext = Depends(require_permissions("webhooks:delete")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a webhook."""
    from app.webhooks import WebhookService

    service = WebhookService(db)
    await service.delete_webhook(str(webhook_id), tenant.tenant_id)


@router.post("/webhooks/{webhook_id}/test", response_model=WebhookTestResponse)
async def test_webhook(
    webhook_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Test a webhook with a sample payload."""
    from app.webhooks import WebhookService

    service = WebhookService(db)

    result = await service.test_webhook(str(webhook_id), tenant.tenant_id)

    return WebhookTestResponse(
        success=result["success"],
        status_code=result["status_code"],
        response_time_ms=result["response_time_ms"],
        error=result.get("error"),
    )


@router.get("/webhooks/events")
async def list_webhook_events(
    user: UserContext = Depends(get_current_user),
):
    """List available webhook events."""
    events = [
        {"event": "call.started", "description": "Call has started"},
        {"event": "call.ended", "description": "Call has ended"},
        {"event": "call.transferred", "description": "Call was transferred"},
        {"event": "transcription.completed", "description": "Transcription is ready"},
        {"event": "agent.message", "description": "Agent sent a message"},
        {"event": "user.message", "description": "User sent a message"},
        {"event": "intent.detected", "description": "Intent was detected"},
        {"event": "function.called", "description": "Function was called"},
        {"event": "error.occurred", "description": "Error occurred"},
    ]
    return {"events": events}
