"""
Campaign API Routes

This module provides REST API endpoints for managing outbound campaigns.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Path, Body, UploadFile, File
from pydantic import BaseModel, Field

from ..base import (
    APIResponse,
    ListResponse,
    NotFoundError,
    ValidationError,
    success_response,
    paginated_response,
)
from ..auth import AuthContext, Permission


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/campaigns", tags=["Campaigns"])


class CampaignStatus(str):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELED = "canceled"


class ScheduleConfig(BaseModel):
    """Campaign schedule configuration."""

    start_time: datetime = Field(..., description="Campaign start time")
    end_time: Optional[datetime] = Field(default=None, description="Campaign end time")
    timezone: str = Field(default="UTC", description="Timezone")

    # Daily calling window
    daily_start_hour: int = Field(default=9, ge=0, le=23)
    daily_end_hour: int = Field(default=17, ge=0, le=23)

    # Days of week (0=Monday, 6=Sunday)
    days_of_week: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])

    # Concurrency
    max_concurrent_calls: int = Field(default=10, ge=1, le=100)
    calls_per_minute: int = Field(default=5, ge=1, le=60)


class RetryConfig(BaseModel):
    """Retry configuration for failed calls."""

    max_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay_minutes: int = Field(default=30, ge=5, le=1440)
    retry_on_busy: bool = True
    retry_on_no_answer: bool = True
    retry_on_voicemail: bool = False


class CampaignCreateRequest(BaseModel):
    """Request to create a campaign."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)

    # Agent and phone number
    agent_id: str = Field(..., description="Agent to use for calls")
    from_phone_number_id: str = Field(..., description="Caller ID phone number")

    # Schedule
    schedule: ScheduleConfig

    # Retry settings
    retry: RetryConfig = Field(default_factory=RetryConfig)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CampaignContact(BaseModel):
    """Contact in a campaign."""

    phone_number: str = Field(..., description="E.164 format")
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class CampaignResponse(BaseModel):
    """Campaign response."""

    id: str
    organization_id: str
    name: str
    description: Optional[str] = None

    # Configuration
    agent_id: str
    from_phone_number_id: str
    schedule: ScheduleConfig
    retry: RetryConfig

    # Status
    status: str = CampaignStatus.DRAFT

    # Stats
    total_contacts: int = 0
    calls_completed: int = 0
    calls_successful: int = 0
    calls_failed: int = 0
    calls_pending: int = 0

    # Metadata
    metadata: Dict[str, Any] = {}

    # Timestamps
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class CampaignStats(BaseModel):
    """Campaign statistics."""

    total_contacts: int
    calls_completed: int
    calls_successful: int
    calls_failed: int
    calls_pending: int
    calls_in_progress: int

    # Rates
    completion_rate: float
    success_rate: float
    answer_rate: float
    voicemail_rate: float

    # Duration
    total_talk_time_minutes: float
    average_call_duration_seconds: float

    # Cost
    total_cost_cents: int


@router.post(
    "",
    response_model=APIResponse[CampaignResponse],
    status_code=201,
    summary="Create Campaign",
)
async def create_campaign(
    request: CampaignCreateRequest,
    auth: AuthContext = Depends(),
):
    """Create a new campaign."""
    auth.require_permission(Permission.CAMPAIGNS_WRITE)

    campaign = CampaignResponse(
        id="cmp_" + "x" * 24,
        organization_id=auth.organization_id,
        name=request.name,
        description=request.description,
        agent_id=request.agent_id,
        from_phone_number_id=request.from_phone_number_id,
        schedule=request.schedule,
        retry=request.retry,
        metadata=request.metadata,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    return success_response(campaign.dict())


@router.get(
    "",
    response_model=ListResponse[CampaignResponse],
    summary="List Campaigns",
)
async def list_campaigns(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    auth: AuthContext = Depends(),
):
    """List campaigns."""
    auth.require_permission(Permission.CAMPAIGNS_READ)

    return paginated_response(items=[], page=page, page_size=page_size, total_items=0)


@router.get("/{campaign_id}", response_model=APIResponse[CampaignResponse])
async def get_campaign(campaign_id: str = Path(...), auth: AuthContext = Depends()):
    """Get campaign details."""
    auth.require_permission(Permission.CAMPAIGNS_READ)
    raise NotFoundError("Campaign", campaign_id)


@router.delete("/{campaign_id}", status_code=204)
async def delete_campaign(campaign_id: str = Path(...), auth: AuthContext = Depends()):
    """Delete a campaign."""
    auth.require_permission(Permission.CAMPAIGNS_DELETE)
    raise NotFoundError("Campaign", campaign_id)


# Contact management
@router.post(
    "/{campaign_id}/contacts",
    response_model=APIResponse[Dict[str, Any]],
    summary="Add Contacts",
)
async def add_contacts(
    campaign_id: str = Path(...),
    contacts: List[CampaignContact] = Body(...),
    auth: AuthContext = Depends(),
):
    """Add contacts to campaign."""
    auth.require_permission(Permission.CAMPAIGNS_WRITE)

    if len(contacts) > 1000:
        raise ValidationError("Maximum 1000 contacts per request")

    raise NotFoundError("Campaign", campaign_id)


@router.post(
    "/{campaign_id}/contacts/upload",
    response_model=APIResponse[Dict[str, Any]],
    summary="Upload Contacts CSV",
)
async def upload_contacts(
    campaign_id: str = Path(...),
    file: UploadFile = File(...),
    auth: AuthContext = Depends(),
):
    """Upload contacts from CSV."""
    auth.require_permission(Permission.CAMPAIGNS_WRITE)

    if not file.filename.endswith(".csv"):
        raise ValidationError("File must be CSV format")

    raise NotFoundError("Campaign", campaign_id)


@router.get(
    "/{campaign_id}/contacts",
    response_model=ListResponse[CampaignContact],
    summary="List Contacts",
)
async def list_contacts(
    campaign_id: str = Path(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None),
    auth: AuthContext = Depends(),
):
    """List contacts in campaign."""
    auth.require_permission(Permission.CAMPAIGNS_READ)
    raise NotFoundError("Campaign", campaign_id)


# Campaign actions
@router.post("/{campaign_id}/start", response_model=APIResponse[CampaignResponse])
async def start_campaign(campaign_id: str = Path(...), auth: AuthContext = Depends()):
    """Start a campaign."""
    auth.require_permission(Permission.CAMPAIGNS_EXECUTE)
    raise NotFoundError("Campaign", campaign_id)


@router.post("/{campaign_id}/pause", response_model=APIResponse[CampaignResponse])
async def pause_campaign(campaign_id: str = Path(...), auth: AuthContext = Depends()):
    """Pause a running campaign."""
    auth.require_permission(Permission.CAMPAIGNS_EXECUTE)
    raise NotFoundError("Campaign", campaign_id)


@router.post("/{campaign_id}/resume", response_model=APIResponse[CampaignResponse])
async def resume_campaign(campaign_id: str = Path(...), auth: AuthContext = Depends()):
    """Resume a paused campaign."""
    auth.require_permission(Permission.CAMPAIGNS_EXECUTE)
    raise NotFoundError("Campaign", campaign_id)


@router.post("/{campaign_id}/cancel", response_model=APIResponse[CampaignResponse])
async def cancel_campaign(campaign_id: str = Path(...), auth: AuthContext = Depends()):
    """Cancel a campaign."""
    auth.require_permission(Permission.CAMPAIGNS_EXECUTE)
    raise NotFoundError("Campaign", campaign_id)


@router.get("/{campaign_id}/stats", response_model=APIResponse[CampaignStats])
async def get_campaign_stats(campaign_id: str = Path(...), auth: AuthContext = Depends()):
    """Get campaign statistics."""
    auth.require_permission(Permission.ANALYTICS_READ)
    raise NotFoundError("Campaign", campaign_id)
