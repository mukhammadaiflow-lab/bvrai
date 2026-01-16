"""
Phone Number API Routes

This module provides REST API endpoints for managing phone numbers.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Path, Body
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
from ..dependencies import get_db_session
from ...database.repositories import PhoneNumberRepository, AgentRepository


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phone-numbers", tags=["Phone Numbers"])


# =============================================================================
# Request/Response Models
# =============================================================================


class PhoneNumberCapabilities(BaseModel):
    """Phone number capabilities."""

    voice: bool = True
    sms: bool = False
    mms: bool = False
    fax: bool = False


class PhoneNumberPurchaseRequest(BaseModel):
    """Request to purchase/add a phone number."""

    phone_number: str = Field(..., description="Phone number in E.164 format")
    friendly_name: Optional[str] = Field(default=None, description="Friendly name")
    country_code: str = Field(default="US", description="Country code")
    number_type: str = Field(default="local", description="Number type: local, toll_free, mobile")
    provider: str = Field(default="twilio", description="Provider: twilio, vonage, bandwidth")
    provider_id: Optional[str] = Field(default=None, description="Provider's ID for this number")
    capabilities: PhoneNumberCapabilities = Field(default_factory=PhoneNumberCapabilities)
    monthly_cost: float = Field(default=0.0, description="Monthly cost")


class PhoneNumberConfigRequest(BaseModel):
    """Request to configure a phone number."""

    agent_id: Optional[str] = Field(default=None, description="Agent to route calls to")
    friendly_name: Optional[str] = Field(default=None, description="Friendly name")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL")
    fallback_url: Optional[str] = Field(default=None, description="Fallback URL")
    status_callback_url: Optional[str] = Field(default=None, description="Status callback URL")
    status: Optional[str] = Field(default=None, description="Status: active, inactive")


class PhoneNumberResponse(BaseModel):
    """Phone number response."""

    id: str
    organization_id: str
    phone_number: str  # E.164 format
    friendly_name: Optional[str] = None
    country_code: str
    number_type: str
    provider: str
    provider_id: Optional[str] = None
    capabilities: PhoneNumberCapabilities

    # Configuration
    agent_id: Optional[str] = None
    webhook_url: Optional[str] = None
    fallback_url: Optional[str] = None
    status_callback_url: Optional[str] = None

    # Status
    status: str = "active"

    # Billing
    monthly_cost: float = 0.0
    currency: str = "USD"

    # Timestamps
    created_at: datetime
    updated_at: datetime


class AvailablePhoneNumber(BaseModel):
    """Available phone number for purchase."""

    phone_number: str
    friendly_name: str
    region: str
    capabilities: PhoneNumberCapabilities
    monthly_cost_cents: int


# =============================================================================
# Helper Functions
# =============================================================================


def phone_to_response(phone) -> dict:
    """Convert database phone number model to response dict."""
    return {
        "id": phone.id,
        "organization_id": phone.organization_id,
        "phone_number": phone.number,
        "friendly_name": phone.friendly_name,
        "country_code": phone.country_code,
        "number_type": phone.number_type,
        "provider": phone.provider,
        "provider_id": phone.provider_id,
        "capabilities": PhoneNumberCapabilities(
            voice=phone.voice_enabled,
            sms=phone.sms_enabled,
            mms=phone.mms_enabled,
        ),
        "agent_id": phone.agent_id,
        "webhook_url": phone.webhook_url,
        "fallback_url": phone.fallback_url,
        "status_callback_url": phone.status_callback_url,
        "status": phone.status,
        "monthly_cost": phone.monthly_cost,
        "currency": phone.currency,
        "created_at": phone.created_at,
        "updated_at": phone.updated_at,
    }


# =============================================================================
# Routes
# =============================================================================


@router.get(
    "/available",
    response_model=APIResponse[List[AvailablePhoneNumber]],
    summary="Search Available Numbers",
    description="Search for available phone numbers to purchase from provider.",
)
async def search_available_numbers(
    country_code: str = Query("US"),
    area_code: Optional[str] = Query(None),
    contains: Optional[str] = Query(None),
    voice: bool = Query(True),
    limit: int = Query(20, ge=1, le=100),
    auth: AuthContext = Depends(),
):
    """Search for available phone numbers from provider (Twilio, etc)."""
    auth.require_permission(Permission.PHONE_NUMBERS_READ)

    # This would integrate with Twilio/Vonage to search available numbers
    # For MVP, return empty list - users can manually add numbers
    return success_response([])


@router.post(
    "",
    response_model=APIResponse[PhoneNumberResponse],
    status_code=201,
    summary="Add Phone Number",
    description="Add a phone number to the organization.",
)
async def add_phone_number(
    request: PhoneNumberPurchaseRequest,
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Add a phone number (can be purchased externally or via provider)."""
    auth.require_permission(Permission.PHONE_NUMBERS_WRITE)

    repo = PhoneNumberRepository(db)

    # Check if number already exists
    existing = await repo.get_by_number(request.phone_number)
    if existing:
        raise ValidationError("Phone number already exists", {"phone_number": request.phone_number})

    # Create phone number
    phone = await repo.create(
        organization_id=auth.organization_id,
        number=request.phone_number,
        friendly_name=request.friendly_name,
        country_code=request.country_code,
        number_type=request.number_type,
        provider=request.provider,
        provider_id=request.provider_id,
        voice_enabled=request.capabilities.voice,
        sms_enabled=request.capabilities.sms,
        mms_enabled=request.capabilities.mms,
        monthly_cost=request.monthly_cost,
        status="active",
    )

    await db.commit()

    logger.info(f"Added phone number {phone.number} for org {auth.organization_id}")

    return success_response(phone_to_response(phone))


@router.get(
    "",
    response_model=ListResponse[PhoneNumberResponse],
    summary="List Phone Numbers",
)
async def list_phone_numbers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    agent_id: Optional[str] = Query(None),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """List phone numbers for the organization."""
    auth.require_permission(Permission.PHONE_NUMBERS_READ)

    repo = PhoneNumberRepository(db)

    # Get phone numbers
    skip = (page - 1) * page_size
    phones = await repo.list_by_organization(
        organization_id=auth.organization_id,
        status=status,
        agent_id=agent_id,
        skip=skip,
        limit=page_size,
    )

    # Get total count
    total = await repo.count_by_organization(
        organization_id=auth.organization_id,
        status=status,
    )

    items = [phone_to_response(p) for p in phones]

    return paginated_response(
        items=items,
        page=page,
        page_size=page_size,
        total_items=total,
    )


@router.get(
    "/{phone_number_id}",
    response_model=APIResponse[PhoneNumberResponse],
    summary="Get Phone Number",
)
async def get_phone_number(
    phone_number_id: str = Path(...),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Get phone number details."""
    auth.require_permission(Permission.PHONE_NUMBERS_READ)

    repo = PhoneNumberRepository(db)
    phone = await repo.get_by_id(phone_number_id)

    if not phone or phone.organization_id != auth.organization_id or phone.is_deleted:
        raise NotFoundError("PhoneNumber", phone_number_id)

    return success_response(phone_to_response(phone))


@router.patch(
    "/{phone_number_id}",
    response_model=APIResponse[PhoneNumberResponse],
    summary="Configure Phone Number",
)
async def configure_phone_number(
    phone_number_id: str = Path(...),
    request: PhoneNumberConfigRequest = Body(...),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Configure a phone number."""
    auth.require_permission(Permission.PHONE_NUMBERS_WRITE)

    repo = PhoneNumberRepository(db)
    phone = await repo.get_by_id(phone_number_id)

    if not phone or phone.organization_id != auth.organization_id or phone.is_deleted:
        raise NotFoundError("PhoneNumber", phone_number_id)

    # Build update data
    update_data = {}
    if request.agent_id is not None:
        # Validate agent exists if provided
        if request.agent_id:
            agent_repo = AgentRepository(db)
            agent = await agent_repo.get_by_id(request.agent_id)
            if not agent or agent.organization_id != auth.organization_id:
                raise ValidationError("Agent not found", {"agent_id": request.agent_id})
        update_data["agent_id"] = request.agent_id if request.agent_id else None

    if request.friendly_name is not None:
        update_data["friendly_name"] = request.friendly_name
    if request.webhook_url is not None:
        update_data["webhook_url"] = request.webhook_url
    if request.fallback_url is not None:
        update_data["fallback_url"] = request.fallback_url
    if request.status_callback_url is not None:
        update_data["status_callback_url"] = request.status_callback_url
    if request.status is not None:
        update_data["status"] = request.status

    if update_data:
        phone = await repo.update(phone_number_id, **update_data)

    await db.commit()

    logger.info(f"Configured phone number {phone_number_id}")

    return success_response(phone_to_response(phone))


@router.delete(
    "/{phone_number_id}",
    status_code=204,
    summary="Release Phone Number",
    description="Release a phone number (soft delete).",
)
async def release_phone_number(
    phone_number_id: str = Path(...),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Release a phone number."""
    auth.require_permission(Permission.PHONE_NUMBERS_DELETE)

    repo = PhoneNumberRepository(db)
    phone = await repo.get_by_id(phone_number_id)

    if not phone or phone.organization_id != auth.organization_id or phone.is_deleted:
        raise NotFoundError("PhoneNumber", phone_number_id)

    # Soft delete
    await repo.soft_delete(phone_number_id)
    await db.commit()

    logger.info(f"Released phone number {phone_number_id}")

    return None


@router.post(
    "/{phone_number_id}/assign",
    response_model=APIResponse[PhoneNumberResponse],
    summary="Assign to Agent",
    description="Assign a phone number to an agent.",
)
async def assign_to_agent(
    phone_number_id: str = Path(...),
    agent_id: str = Body(..., embed=True),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Assign phone number to agent."""
    auth.require_permission(Permission.PHONE_NUMBERS_WRITE)

    repo = PhoneNumberRepository(db)
    phone = await repo.get_by_id(phone_number_id)

    if not phone or phone.organization_id != auth.organization_id or phone.is_deleted:
        raise NotFoundError("PhoneNumber", phone_number_id)

    # Validate agent
    agent_repo = AgentRepository(db)
    agent = await agent_repo.get_by_id(agent_id)
    if not agent or agent.organization_id != auth.organization_id:
        raise ValidationError("Agent not found", {"agent_id": agent_id})

    # Assign
    phone = await repo.assign_to_agent(phone_number_id, agent_id)
    await db.commit()

    logger.info(f"Assigned phone number {phone_number_id} to agent {agent_id}")

    return success_response(phone_to_response(phone))


@router.post(
    "/{phone_number_id}/unassign",
    response_model=APIResponse[PhoneNumberResponse],
    summary="Unassign from Agent",
)
async def unassign_from_agent(
    phone_number_id: str = Path(...),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Unassign phone number from agent."""
    auth.require_permission(Permission.PHONE_NUMBERS_WRITE)

    repo = PhoneNumberRepository(db)
    phone = await repo.get_by_id(phone_number_id)

    if not phone or phone.organization_id != auth.organization_id or phone.is_deleted:
        raise NotFoundError("PhoneNumber", phone_number_id)

    # Unassign
    phone = await repo.unassign_from_agent(phone_number_id)
    await db.commit()

    logger.info(f"Unassigned phone number {phone_number_id} from agent")

    return success_response(phone_to_response(phone))


@router.get(
    "/available-for-assignment",
    response_model=APIResponse[List[PhoneNumberResponse]],
    summary="Get Available for Assignment",
    description="Get phone numbers available for assignment to agents.",
)
async def get_available_for_assignment(
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Get phone numbers not assigned to any agent."""
    auth.require_permission(Permission.PHONE_NUMBERS_READ)

    repo = PhoneNumberRepository(db)
    phones = await repo.get_available(auth.organization_id)

    return success_response([phone_to_response(p) for p in phones])
