"""
Campaign API Routes

This module provides REST API endpoints for managing outbound campaigns.
"""

import csv
import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Path, Body, UploadFile, File
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..base import (
    APIResponse,
    ListResponse,
    NotFoundError,
    ValidationError,
    success_response,
    paginated_response,
)
from ..auth import AuthContext, Permission
from ..dependencies import get_db_session, get_auth_context
from ...database.repositories import CampaignRepository, AgentRepository, PhoneNumberRepository


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/campaigns", tags=["Campaigns"])


# =============================================================================
# Request/Response Models
# =============================================================================


class ScheduleConfig(BaseModel):
    """Campaign schedule configuration."""

    start_time: Optional[datetime] = Field(default=None, description="Campaign start time")
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

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=500)

    # Agent and phone number
    agent_id: str = Field(..., description="Agent to use for calls")
    phone_number_id: Optional[str] = Field(default=None, description="Caller ID phone number")

    # Schedule
    schedule_config: Optional[ScheduleConfig] = Field(default=None)

    # Retry settings
    retry_config: Optional[RetryConfig] = Field(default=None)

    # Metadata
    extra_data: Dict[str, Any] = Field(default_factory=dict)


class CampaignUpdateRequest(BaseModel):
    """Request to update a campaign."""

    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None, max_length=500)
    agent_id: Optional[str] = Field(default=None)
    phone_number_id: Optional[str] = Field(default=None)
    schedule_config: Optional[ScheduleConfig] = Field(default=None)
    retry_config: Optional[RetryConfig] = Field(default=None)


class ContactInput(BaseModel):
    """Contact input for campaign."""

    phone_number: str = Field(..., description="E.164 format")
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class ContactResponse(BaseModel):
    """Contact response."""

    id: str
    campaign_id: str
    phone_number: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    context: Optional[Dict] = None
    status: str
    call_id: Optional[str] = None
    call_outcome: Optional[str] = None
    attempt_count: int = 0
    last_attempt_at: Optional[datetime] = None
    next_attempt_at: Optional[datetime] = None
    call_duration_seconds: float = 0.0
    notes: Optional[str] = None
    created_at: datetime


class CampaignResponse(BaseModel):
    """Campaign response."""

    id: str
    organization_id: str
    name: str
    description: Optional[str] = None

    # Configuration
    agent_id: Optional[str] = None
    phone_number_id: Optional[str] = None
    schedule_config: Optional[Dict] = None
    retry_config: Optional[Dict] = None

    # Status
    status: str = "draft"

    # Stats
    total_contacts: int = 0
    calls_completed: int = 0
    calls_successful: int = 0
    calls_failed: int = 0
    calls_pending: int = 0
    calls_in_progress: int = 0

    # Cost/Duration
    total_cost: float = 0.0
    total_minutes: float = 0.0

    # Timestamps
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None


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

    # Duration and cost
    total_minutes: float
    total_cost: float


# =============================================================================
# Helper Functions
# =============================================================================


def campaign_to_response(campaign) -> dict:
    """Convert database campaign model to response dict."""
    return {
        "id": campaign.id,
        "organization_id": campaign.organization_id,
        "name": campaign.name,
        "description": campaign.description,
        "agent_id": campaign.agent_id,
        "phone_number_id": campaign.phone_number_id,
        "schedule_config": campaign.schedule_config,
        "retry_config": campaign.retry_config,
        "status": campaign.status,
        "total_contacts": campaign.total_contacts,
        "calls_completed": campaign.calls_completed,
        "calls_successful": campaign.calls_successful,
        "calls_failed": campaign.calls_failed,
        "calls_pending": campaign.calls_pending,
        "calls_in_progress": campaign.calls_in_progress,
        "total_cost": campaign.total_cost,
        "total_minutes": campaign.total_minutes,
        "created_at": campaign.created_at,
        "updated_at": campaign.updated_at,
        "started_at": campaign.started_at,
        "completed_at": campaign.completed_at,
        "paused_at": campaign.paused_at,
    }


def contact_to_response(contact) -> dict:
    """Convert database contact model to response dict."""
    return {
        "id": contact.id,
        "campaign_id": contact.campaign_id,
        "phone_number": contact.phone_number,
        "first_name": contact.first_name,
        "last_name": contact.last_name,
        "email": contact.email,
        "context": contact.context,
        "status": contact.status,
        "call_id": contact.call_id,
        "call_outcome": contact.call_outcome,
        "attempt_count": contact.attempt_count,
        "last_attempt_at": contact.last_attempt_at,
        "next_attempt_at": contact.next_attempt_at,
        "call_duration_seconds": contact.call_duration_seconds,
        "notes": contact.notes,
        "created_at": contact.created_at,
    }


# =============================================================================
# Routes
# =============================================================================


@router.post(
    "",
    response_model=APIResponse[CampaignResponse],
    status_code=201,
    summary="Create Campaign",
)
async def create_campaign(
    request: CampaignCreateRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new campaign."""
    auth.require_permission(Permission.CAMPAIGNS_WRITE)

    # Validate agent exists
    agent_repo = AgentRepository(db)
    agent = await agent_repo.get_by_id(request.agent_id)
    if not agent or agent.organization_id != auth.organization_id:
        raise ValidationError("Agent not found", {"agent_id": request.agent_id})

    # Validate phone number if provided
    if request.phone_number_id:
        phone_repo = PhoneNumberRepository(db)
        phone = await phone_repo.get_by_id(request.phone_number_id)
        if not phone or phone.organization_id != auth.organization_id:
            raise ValidationError("Phone number not found", {"phone_number_id": request.phone_number_id})

    repo = CampaignRepository(db)
    campaign = await repo.create(
        organization_id=auth.organization_id,
        name=request.name,
        description=request.description,
        agent_id=request.agent_id,
        phone_number_id=request.phone_number_id,
        schedule_config=request.schedule_config.model_dump() if request.schedule_config else None,
        retry_config=request.retry_config.model_dump() if request.retry_config else None,
        extra_data=request.extra_data,
        status="draft",
    )

    await db.commit()

    logger.info(f"Created campaign {campaign.id} for org {auth.organization_id}")

    return success_response(campaign_to_response(campaign))


@router.get(
    "",
    response_model=ListResponse[CampaignResponse],
    summary="List Campaigns",
)
async def list_campaigns(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """List campaigns."""
    auth.require_permission(Permission.CAMPAIGNS_READ)

    repo = CampaignRepository(db)

    skip = (page - 1) * page_size
    campaigns = await repo.list_by_organization(
        organization_id=auth.organization_id,
        status=status,
        skip=skip,
        limit=page_size,
    )

    total = await repo.count_by_organization(
        organization_id=auth.organization_id,
        status=status,
    )

    items = [campaign_to_response(c) for c in campaigns]

    return paginated_response(
        items=items,
        page=page,
        page_size=page_size,
        total_items=total,
    )


@router.get(
    "/{campaign_id}",
    response_model=APIResponse[CampaignResponse],
    summary="Get Campaign",
)
async def get_campaign(
    campaign_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Get campaign details."""
    auth.require_permission(Permission.CAMPAIGNS_READ)

    repo = CampaignRepository(db)
    campaign = await repo.get_by_id(campaign_id)

    if not campaign or campaign.organization_id != auth.organization_id or campaign.is_deleted:
        raise NotFoundError("Campaign", campaign_id)

    return success_response(campaign_to_response(campaign))


@router.patch(
    "/{campaign_id}",
    response_model=APIResponse[CampaignResponse],
    summary="Update Campaign",
)
async def update_campaign(
    campaign_id: str = Path(...),
    request: CampaignUpdateRequest = Body(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Update a campaign."""
    auth.require_permission(Permission.CAMPAIGNS_WRITE)

    repo = CampaignRepository(db)
    campaign = await repo.get_by_id(campaign_id)

    if not campaign or campaign.organization_id != auth.organization_id or campaign.is_deleted:
        raise NotFoundError("Campaign", campaign_id)

    if campaign.status not in ["draft", "paused"]:
        raise ValidationError("Can only update draft or paused campaigns", {"status": campaign.status})

    update_data = {}
    if request.name is not None:
        update_data["name"] = request.name
    if request.description is not None:
        update_data["description"] = request.description
    if request.agent_id is not None:
        # Validate agent
        agent_repo = AgentRepository(db)
        agent = await agent_repo.get_by_id(request.agent_id)
        if not agent or agent.organization_id != auth.organization_id:
            raise ValidationError("Agent not found", {"agent_id": request.agent_id})
        update_data["agent_id"] = request.agent_id
    if request.phone_number_id is not None:
        update_data["phone_number_id"] = request.phone_number_id
    if request.schedule_config is not None:
        update_data["schedule_config"] = request.schedule_config.model_dump()
    if request.retry_config is not None:
        update_data["retry_config"] = request.retry_config.model_dump()

    if update_data:
        campaign = await repo.update(campaign_id, **update_data)

    await db.commit()

    logger.info(f"Updated campaign {campaign_id}")

    return success_response(campaign_to_response(campaign))


@router.delete(
    "/{campaign_id}",
    status_code=204,
    summary="Delete Campaign",
)
async def delete_campaign(
    campaign_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a campaign."""
    auth.require_permission(Permission.CAMPAIGNS_DELETE)

    repo = CampaignRepository(db)
    campaign = await repo.get_by_id(campaign_id)

    if not campaign or campaign.organization_id != auth.organization_id or campaign.is_deleted:
        raise NotFoundError("Campaign", campaign_id)

    if campaign.status == "running":
        raise ValidationError("Cannot delete a running campaign. Pause or cancel it first.", {"status": campaign.status})

    await repo.soft_delete(campaign_id)
    await db.commit()

    logger.info(f"Deleted campaign {campaign_id}")

    return None


# =============================================================================
# Contact Management
# =============================================================================


@router.post(
    "/{campaign_id}/contacts",
    response_model=APIResponse[Dict[str, Any]],
    summary="Add Contacts",
)
async def add_contacts(
    campaign_id: str = Path(...),
    contacts: List[ContactInput] = Body(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Add contacts to campaign."""
    auth.require_permission(Permission.CAMPAIGNS_WRITE)

    if len(contacts) > 1000:
        raise ValidationError("Maximum 1000 contacts per request", {"contacts": "too_many"})

    repo = CampaignRepository(db)
    campaign = await repo.get_by_id(campaign_id)

    if not campaign or campaign.organization_id != auth.organization_id or campaign.is_deleted:
        raise NotFoundError("Campaign", campaign_id)

    # Add contacts
    contact_dicts = [c.model_dump() for c in contacts]
    added = await repo.add_contacts_bulk(
        campaign_id=campaign_id,
        organization_id=auth.organization_id,
        contacts=contact_dicts,
    )

    await db.commit()

    logger.info(f"Added {added} contacts to campaign {campaign_id}")

    return success_response({
        "campaign_id": campaign_id,
        "contacts_added": added,
        "contacts_submitted": len(contacts),
    })


@router.post(
    "/{campaign_id}/contacts/upload",
    response_model=APIResponse[Dict[str, Any]],
    summary="Upload Contacts CSV",
)
async def upload_contacts(
    campaign_id: str = Path(...),
    file: UploadFile = File(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Upload contacts from CSV."""
    auth.require_permission(Permission.CAMPAIGNS_WRITE)

    if not file.filename or not file.filename.endswith(".csv"):
        raise ValidationError("File must be CSV format", {"file": "invalid_type"})

    repo = CampaignRepository(db)
    campaign = await repo.get_by_id(campaign_id)

    if not campaign or campaign.organization_id != auth.organization_id or campaign.is_deleted:
        raise NotFoundError("Campaign", campaign_id)

    # Parse CSV
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))

    contacts = []
    errors = []
    for i, row in enumerate(reader):
        phone = row.get("phone_number") or row.get("phone") or row.get("Phone")
        if not phone:
            errors.append(f"Row {i+1}: missing phone_number")
            continue

        contacts.append({
            "phone_number": phone.strip(),
            "first_name": row.get("first_name") or row.get("First Name"),
            "last_name": row.get("last_name") or row.get("Last Name"),
            "email": row.get("email") or row.get("Email"),
            "context": {k: v for k, v in row.items() if k not in ["phone_number", "phone", "Phone", "first_name", "First Name", "last_name", "Last Name", "email", "Email"]},
        })

        if len(contacts) >= 10000:
            break

    added = await repo.add_contacts_bulk(
        campaign_id=campaign_id,
        organization_id=auth.organization_id,
        contacts=contacts,
    )

    await db.commit()

    logger.info(f"Uploaded {added} contacts to campaign {campaign_id}")

    return success_response({
        "campaign_id": campaign_id,
        "contacts_added": added,
        "contacts_parsed": len(contacts),
        "errors": errors[:10] if errors else [],  # Return first 10 errors
        "total_errors": len(errors),
    })


@router.get(
    "/{campaign_id}/contacts",
    response_model=ListResponse[ContactResponse],
    summary="List Contacts",
)
async def list_contacts(
    campaign_id: str = Path(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query(None),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """List contacts in campaign."""
    auth.require_permission(Permission.CAMPAIGNS_READ)

    repo = CampaignRepository(db)
    campaign = await repo.get_by_id(campaign_id)

    if not campaign or campaign.organization_id != auth.organization_id or campaign.is_deleted:
        raise NotFoundError("Campaign", campaign_id)

    skip = (page - 1) * page_size
    contacts = await repo.list_contacts(
        campaign_id=campaign_id,
        status=status,
        skip=skip,
        limit=page_size,
    )

    total = await repo.count_contacts(
        campaign_id=campaign_id,
        status=status,
    )

    items = [contact_to_response(c) for c in contacts]

    return paginated_response(
        items=items,
        page=page,
        page_size=page_size,
        total_items=total,
    )


# =============================================================================
# Campaign Actions
# =============================================================================


@router.post(
    "/{campaign_id}/start",
    response_model=APIResponse[CampaignResponse],
    summary="Start Campaign",
)
async def start_campaign(
    campaign_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Start a campaign."""
    auth.require_permission(Permission.CAMPAIGNS_EXECUTE)

    repo = CampaignRepository(db)
    campaign = await repo.get_by_id(campaign_id)

    if not campaign or campaign.organization_id != auth.organization_id or campaign.is_deleted:
        raise NotFoundError("Campaign", campaign_id)

    if campaign.status not in ["draft", "scheduled"]:
        raise ValidationError(
            f"Cannot start campaign with status '{campaign.status}'",
            {"status": campaign.status},
        )

    if campaign.total_contacts == 0:
        raise ValidationError("Cannot start campaign with no contacts", {"total_contacts": 0})

    campaign = await repo.update_status(campaign_id, "running")
    await db.commit()

    logger.info(f"Started campaign {campaign_id}")

    # In production, this would trigger the campaign worker to start making calls

    return success_response(campaign_to_response(campaign))


@router.post(
    "/{campaign_id}/pause",
    response_model=APIResponse[CampaignResponse],
    summary="Pause Campaign",
)
async def pause_campaign(
    campaign_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Pause a running campaign."""
    auth.require_permission(Permission.CAMPAIGNS_EXECUTE)

    repo = CampaignRepository(db)
    campaign = await repo.get_by_id(campaign_id)

    if not campaign or campaign.organization_id != auth.organization_id or campaign.is_deleted:
        raise NotFoundError("Campaign", campaign_id)

    if campaign.status != "running":
        raise ValidationError(
            f"Cannot pause campaign with status '{campaign.status}'",
            {"status": campaign.status},
        )

    campaign = await repo.update_status(campaign_id, "paused")
    await db.commit()

    logger.info(f"Paused campaign {campaign_id}")

    return success_response(campaign_to_response(campaign))


@router.post(
    "/{campaign_id}/resume",
    response_model=APIResponse[CampaignResponse],
    summary="Resume Campaign",
)
async def resume_campaign(
    campaign_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Resume a paused campaign."""
    auth.require_permission(Permission.CAMPAIGNS_EXECUTE)

    repo = CampaignRepository(db)
    campaign = await repo.get_by_id(campaign_id)

    if not campaign or campaign.organization_id != auth.organization_id or campaign.is_deleted:
        raise NotFoundError("Campaign", campaign_id)

    if campaign.status != "paused":
        raise ValidationError(
            f"Cannot resume campaign with status '{campaign.status}'",
            {"status": campaign.status},
        )

    campaign = await repo.update_status(campaign_id, "running")
    await db.commit()

    logger.info(f"Resumed campaign {campaign_id}")

    return success_response(campaign_to_response(campaign))


@router.post(
    "/{campaign_id}/cancel",
    response_model=APIResponse[CampaignResponse],
    summary="Cancel Campaign",
)
async def cancel_campaign(
    campaign_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Cancel a campaign."""
    auth.require_permission(Permission.CAMPAIGNS_EXECUTE)

    repo = CampaignRepository(db)
    campaign = await repo.get_by_id(campaign_id)

    if not campaign or campaign.organization_id != auth.organization_id or campaign.is_deleted:
        raise NotFoundError("Campaign", campaign_id)

    if campaign.status in ["completed", "canceled"]:
        raise ValidationError(
            f"Cannot cancel campaign with status '{campaign.status}'",
            {"status": campaign.status},
        )

    campaign = await repo.update_status(campaign_id, "canceled")
    await db.commit()

    logger.info(f"Canceled campaign {campaign_id}")

    return success_response(campaign_to_response(campaign))


@router.get(
    "/{campaign_id}/stats",
    response_model=APIResponse[CampaignStats],
    summary="Get Campaign Stats",
)
async def get_campaign_stats(
    campaign_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Get campaign statistics."""
    auth.require_permission(Permission.ANALYTICS_READ)

    repo = CampaignRepository(db)
    campaign = await repo.get_by_id(campaign_id)

    if not campaign or campaign.organization_id != auth.organization_id or campaign.is_deleted:
        raise NotFoundError("Campaign", campaign_id)

    stats = await repo.get_stats(campaign_id)

    return success_response(stats)
