"""
Phone Number API Routes

This module provides REST API endpoints for managing phone numbers.
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

router = APIRouter(prefix="/phone-numbers", tags=["Phone Numbers"])


class PhoneNumberCapabilities(BaseModel):
    """Phone number capabilities."""

    voice: bool = True
    sms: bool = False
    mms: bool = False
    fax: bool = False


class PhoneNumberPurchaseRequest(BaseModel):
    """Request to purchase a phone number."""

    country_code: str = Field(default="US", description="Country code")
    area_code: Optional[str] = Field(default=None, description="Area code")
    contains: Optional[str] = Field(default=None, description="Pattern to match")
    capabilities: PhoneNumberCapabilities = Field(default_factory=PhoneNumberCapabilities)


class PhoneNumberConfigRequest(BaseModel):
    """Request to configure a phone number."""

    agent_id: Optional[str] = Field(default=None, description="Agent to route calls to")
    friendly_name: Optional[str] = Field(default=None, description="Friendly name")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL")


class PhoneNumberResponse(BaseModel):
    """Phone number response."""

    id: str
    organization_id: str
    phone_number: str  # E.164 format
    friendly_name: Optional[str] = None
    country_code: str
    capabilities: PhoneNumberCapabilities

    # Configuration
    agent_id: Optional[str] = None
    webhook_url: Optional[str] = None

    # Status
    status: str = "active"  # active, pending, released

    # Billing
    monthly_cost_cents: int = 0

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


@router.get(
    "/available",
    response_model=APIResponse[List[AvailablePhoneNumber]],
    summary="Search Available Numbers",
    description="Search for available phone numbers to purchase.",
)
async def search_available_numbers(
    country_code: str = Query("US"),
    area_code: Optional[str] = Query(None),
    contains: Optional[str] = Query(None),
    voice: bool = Query(True),
    limit: int = Query(20, ge=1, le=100),
    auth: AuthContext = Depends(),
):
    """Search for available phone numbers."""
    auth.require_permission(Permission.PHONE_NUMBERS_READ)

    # In production, query Twilio or other provider
    return success_response([])


@router.post(
    "",
    response_model=APIResponse[PhoneNumberResponse],
    status_code=201,
    summary="Purchase Phone Number",
    description="Purchase a new phone number.",
)
async def purchase_phone_number(
    request: PhoneNumberPurchaseRequest,
    auth: AuthContext = Depends(),
):
    """Purchase a phone number."""
    auth.require_permission(Permission.PHONE_NUMBERS_WRITE)

    # In production:
    # 1. Search for available number
    # 2. Purchase from provider
    # 3. Store in database

    return success_response({
        "id": "pn_" + "x" * 24,
        "phone_number": "+1234567890",
        "message": "Phone number purchase would happen here",
    })


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
):
    """List phone numbers."""
    auth.require_permission(Permission.PHONE_NUMBERS_READ)

    return paginated_response(items=[], page=page, page_size=page_size, total_items=0)


@router.get(
    "/{phone_number_id}",
    response_model=APIResponse[PhoneNumberResponse],
    summary="Get Phone Number",
)
async def get_phone_number(
    phone_number_id: str = Path(...),
    auth: AuthContext = Depends(),
):
    """Get phone number details."""
    auth.require_permission(Permission.PHONE_NUMBERS_READ)

    raise NotFoundError("PhoneNumber", phone_number_id)


@router.patch(
    "/{phone_number_id}",
    response_model=APIResponse[PhoneNumberResponse],
    summary="Configure Phone Number",
)
async def configure_phone_number(
    phone_number_id: str = Path(...),
    request: PhoneNumberConfigRequest = Body(...),
    auth: AuthContext = Depends(),
):
    """Configure a phone number."""
    auth.require_permission(Permission.PHONE_NUMBERS_WRITE)

    raise NotFoundError("PhoneNumber", phone_number_id)


@router.delete(
    "/{phone_number_id}",
    status_code=204,
    summary="Release Phone Number",
    description="Release a phone number back to the provider.",
)
async def release_phone_number(
    phone_number_id: str = Path(...),
    auth: AuthContext = Depends(),
):
    """Release a phone number."""
    auth.require_permission(Permission.PHONE_NUMBERS_DELETE)

    raise NotFoundError("PhoneNumber", phone_number_id)


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
):
    """Assign phone number to agent."""
    auth.require_permission(Permission.PHONE_NUMBERS_WRITE)

    raise NotFoundError("PhoneNumber", phone_number_id)


@router.post(
    "/{phone_number_id}/unassign",
    response_model=APIResponse[PhoneNumberResponse],
    summary="Unassign from Agent",
)
async def unassign_from_agent(
    phone_number_id: str = Path(...),
    auth: AuthContext = Depends(),
):
    """Unassign phone number from agent."""
    auth.require_permission(Permission.PHONE_NUMBERS_WRITE)

    raise NotFoundError("PhoneNumber", phone_number_id)
