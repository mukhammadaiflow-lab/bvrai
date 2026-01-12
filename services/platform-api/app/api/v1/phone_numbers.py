"""
Phone Numbers API Routes

Handles:
- Phone number provisioning
- Number management and configuration
- Agent assignment
- Carrier management
"""

from typing import Optional, List
from datetime import datetime
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_db_session,
    get_current_user,
    get_current_tenant,
    UserContext,
    TenantContext,
    require_permissions,
    get_pagination,
    PaginationParams,
)

router = APIRouter(prefix="/phone-numbers")


# ============================================================================
# Schemas
# ============================================================================

class PhoneNumberType(str, Enum):
    """Phone number types."""
    LOCAL = "local"
    TOLL_FREE = "toll_free"
    MOBILE = "mobile"


class PhoneNumberStatus(str, Enum):
    """Phone number status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    RELEASED = "released"


class Carrier(str, Enum):
    """Supported carriers."""
    TWILIO = "twilio"
    TELNYX = "telnyx"
    VONAGE = "vonage"


class SearchNumbersRequest(BaseModel):
    """Search available numbers request."""
    country: str = Field("US", description="ISO country code")
    area_code: Optional[str] = None
    contains: Optional[str] = None
    number_type: PhoneNumberType = PhoneNumberType.LOCAL
    carrier: Carrier = Carrier.TWILIO
    limit: int = Field(20, ge=1, le=100)


class AvailableNumber(BaseModel):
    """Available phone number."""
    phone_number: str
    formatted: str
    country: str
    region: Optional[str]
    locality: Optional[str]
    number_type: PhoneNumberType
    capabilities: List[str]
    monthly_cost: float
    carrier: Carrier


class SearchNumbersResponse(BaseModel):
    """Search results."""
    numbers: List[AvailableNumber]
    total: int


class PurchaseNumberRequest(BaseModel):
    """Purchase number request."""
    phone_number: str
    carrier: Carrier = Carrier.TWILIO
    agent_id: Optional[UUID] = None
    friendly_name: Optional[str] = None


class PhoneNumberUpdate(BaseModel):
    """Update phone number request."""
    friendly_name: Optional[str] = None
    agent_id: Optional[UUID] = None
    voice_url: Optional[str] = None
    sms_url: Optional[str] = None
    status_callback_url: Optional[str] = None


class PhoneNumberResponse(BaseModel):
    """Phone number response."""
    id: str
    phone_number: str
    formatted: str
    friendly_name: Optional[str]
    country: str
    number_type: PhoneNumberType
    status: PhoneNumberStatus
    carrier: Carrier
    capabilities: List[str]
    agent_id: Optional[str]
    agent_name: Optional[str]
    voice_url: Optional[str]
    monthly_cost: float
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class PhoneNumberListResponse(BaseModel):
    """List response with pagination."""
    numbers: List[PhoneNumberResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class PhoneNumberStats(BaseModel):
    """Phone number statistics."""
    total_numbers: int
    active_numbers: int
    assigned_numbers: int
    total_calls_today: int
    total_calls_month: int
    total_minutes_month: float
    monthly_cost: float


# ============================================================================
# Search & Purchase
# ============================================================================

@router.post("/search", response_model=SearchNumbersResponse)
async def search_available_numbers(
    data: SearchNumbersRequest,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """
    Search for available phone numbers.

    Search across carriers for available numbers to purchase.
    """
    from app.phone_numbers.service import get_phone_number_service

    service = get_phone_number_service()

    numbers = await service.search_available(
        country=data.country,
        area_code=data.area_code,
        contains=data.contains,
        number_type=data.number_type.value,
        carrier=data.carrier.value,
        limit=data.limit,
    )

    return SearchNumbersResponse(
        numbers=[
            AvailableNumber(
                phone_number=n["phone_number"],
                formatted=n["formatted"],
                country=n["country"],
                region=n.get("region"),
                locality=n.get("locality"),
                number_type=PhoneNumberType(n["number_type"]),
                capabilities=n.get("capabilities", []),
                monthly_cost=n.get("monthly_cost", 0),
                carrier=Carrier(n["carrier"]),
            )
            for n in numbers
        ],
        total=len(numbers),
    )


@router.post("/purchase", response_model=PhoneNumberResponse, status_code=status.HTTP_201_CREATED)
async def purchase_phone_number(
    data: PurchaseNumberRequest,
    user: UserContext = Depends(require_permissions("phone_numbers:create")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Purchase a phone number.

    Provisions the number with the carrier and adds it to your account.
    """
    from app.phone_numbers.service import get_phone_number_service

    # Check tenant limits
    limit = tenant.get_limit("phone_numbers", 10)
    service = get_phone_number_service()

    current_count = await service.count_by_tenant(tenant.tenant_id)
    if current_count >= limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Phone number limit reached ({limit}). Please upgrade your plan.",
        )

    try:
        number = await service.purchase(
            tenant_id=tenant.tenant_id,
            phone_number=data.phone_number,
            carrier=data.carrier.value,
            agent_id=str(data.agent_id) if data.agent_id else None,
            friendly_name=data.friendly_name,
        )

        return PhoneNumberResponse(
            id=number.id,
            phone_number=number.phone_number,
            formatted=number.formatted,
            friendly_name=number.friendly_name,
            country=number.country,
            number_type=PhoneNumberType(number.number_type),
            status=PhoneNumberStatus(number.status),
            carrier=Carrier(number.carrier),
            capabilities=number.capabilities or [],
            agent_id=number.agent_id,
            agent_name=None,
            voice_url=number.voice_url,
            monthly_cost=number.monthly_cost,
            created_at=number.created_at,
            updated_at=number.updated_at,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to purchase number: {str(e)}",
        )


# ============================================================================
# CRUD Operations
# ============================================================================

@router.get("", response_model=PhoneNumberListResponse)
async def list_phone_numbers(
    status_filter: Optional[PhoneNumberStatus] = Query(None, alias="status"),
    carrier: Optional[Carrier] = None,
    assigned: Optional[bool] = None,
    pagination: PaginationParams = Depends(get_pagination),
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """List all phone numbers for the tenant."""
    from app.phone_numbers.service import get_phone_number_service

    service = get_phone_number_service()

    numbers, total = await service.list_by_tenant(
        tenant_id=tenant.tenant_id,
        status=status_filter.value if status_filter else None,
        carrier=carrier.value if carrier else None,
        assigned=assigned,
        offset=pagination.offset,
        limit=pagination.limit,
    )

    return PhoneNumberListResponse(
        numbers=[
            PhoneNumberResponse(
                id=n.id,
                phone_number=n.phone_number,
                formatted=n.formatted,
                friendly_name=n.friendly_name,
                country=n.country,
                number_type=PhoneNumberType(n.number_type),
                status=PhoneNumberStatus(n.status),
                carrier=Carrier(n.carrier),
                capabilities=n.capabilities or [],
                agent_id=n.agent_id,
                agent_name=getattr(n, 'agent_name', None),
                voice_url=n.voice_url,
                monthly_cost=n.monthly_cost,
                created_at=n.created_at,
                updated_at=n.updated_at,
            )
            for n in numbers
        ],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
        total_pages=(total + pagination.page_size - 1) // pagination.page_size,
    )


@router.get("/stats", response_model=PhoneNumberStats)
async def get_phone_number_stats(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get phone number statistics."""
    from app.phone_numbers.service import get_phone_number_service

    service = get_phone_number_service()
    stats = await service.get_stats(tenant.tenant_id)

    return PhoneNumberStats(**stats)


@router.get("/{number_id}", response_model=PhoneNumberResponse)
async def get_phone_number(
    number_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get a phone number by ID."""
    from app.phone_numbers.service import get_phone_number_service

    service = get_phone_number_service()
    number = await service.get(str(number_id), tenant.tenant_id)

    if not number:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Phone number not found",
        )

    return PhoneNumberResponse(
        id=number.id,
        phone_number=number.phone_number,
        formatted=number.formatted,
        friendly_name=number.friendly_name,
        country=number.country,
        number_type=PhoneNumberType(number.number_type),
        status=PhoneNumberStatus(number.status),
        carrier=Carrier(number.carrier),
        capabilities=number.capabilities or [],
        agent_id=number.agent_id,
        agent_name=getattr(number, 'agent_name', None),
        voice_url=number.voice_url,
        monthly_cost=number.monthly_cost,
        created_at=number.created_at,
        updated_at=number.updated_at,
    )


@router.patch("/{number_id}", response_model=PhoneNumberResponse)
async def update_phone_number(
    number_id: UUID,
    data: PhoneNumberUpdate,
    user: UserContext = Depends(require_permissions("phone_numbers:update")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Update a phone number."""
    from app.phone_numbers.service import get_phone_number_service

    service = get_phone_number_service()

    number = await service.update(
        number_id=str(number_id),
        tenant_id=tenant.tenant_id,
        **data.model_dump(exclude_unset=True),
    )

    if not number:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Phone number not found",
        )

    return PhoneNumberResponse(
        id=number.id,
        phone_number=number.phone_number,
        formatted=number.formatted,
        friendly_name=number.friendly_name,
        country=number.country,
        number_type=PhoneNumberType(number.number_type),
        status=PhoneNumberStatus(number.status),
        carrier=Carrier(number.carrier),
        capabilities=number.capabilities or [],
        agent_id=number.agent_id,
        agent_name=getattr(number, 'agent_name', None),
        voice_url=number.voice_url,
        monthly_cost=number.monthly_cost,
        created_at=number.created_at,
        updated_at=number.updated_at,
    )


@router.delete("/{number_id}", status_code=status.HTTP_204_NO_CONTENT)
async def release_phone_number(
    number_id: UUID,
    user: UserContext = Depends(require_permissions("phone_numbers:delete")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Release a phone number.

    Releases the number back to the carrier and removes it from your account.
    """
    from app.phone_numbers.service import get_phone_number_service

    service = get_phone_number_service()

    released = await service.release(str(number_id), tenant.tenant_id)
    if not released:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Phone number not found",
        )


# ============================================================================
# Agent Assignment
# ============================================================================

@router.post("/{number_id}/assign/{agent_id}", response_model=PhoneNumberResponse)
async def assign_agent(
    number_id: UUID,
    agent_id: UUID,
    user: UserContext = Depends(require_permissions("phone_numbers:update")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Assign an agent to a phone number."""
    from app.phone_numbers.service import get_phone_number_service

    service = get_phone_number_service()

    number = await service.assign_agent(
        number_id=str(number_id),
        agent_id=str(agent_id),
        tenant_id=tenant.tenant_id,
    )

    if not number:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Phone number not found",
        )

    return PhoneNumberResponse(
        id=number.id,
        phone_number=number.phone_number,
        formatted=number.formatted,
        friendly_name=number.friendly_name,
        country=number.country,
        number_type=PhoneNumberType(number.number_type),
        status=PhoneNumberStatus(number.status),
        carrier=Carrier(number.carrier),
        capabilities=number.capabilities or [],
        agent_id=number.agent_id,
        agent_name=getattr(number, 'agent_name', None),
        voice_url=number.voice_url,
        monthly_cost=number.monthly_cost,
        created_at=number.created_at,
        updated_at=number.updated_at,
    )


@router.post("/{number_id}/unassign", response_model=PhoneNumberResponse)
async def unassign_agent(
    number_id: UUID,
    user: UserContext = Depends(require_permissions("phone_numbers:update")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Unassign agent from a phone number."""
    from app.phone_numbers.service import get_phone_number_service

    service = get_phone_number_service()

    number = await service.unassign_agent(
        number_id=str(number_id),
        tenant_id=tenant.tenant_id,
    )

    if not number:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Phone number not found",
        )

    return PhoneNumberResponse(
        id=number.id,
        phone_number=number.phone_number,
        formatted=number.formatted,
        friendly_name=number.friendly_name,
        country=number.country,
        number_type=PhoneNumberType(number.number_type),
        status=PhoneNumberStatus(number.status),
        carrier=Carrier(number.carrier),
        capabilities=number.capabilities or [],
        agent_id=None,
        agent_name=None,
        voice_url=number.voice_url,
        monthly_cost=number.monthly_cost,
        created_at=number.created_at,
        updated_at=number.updated_at,
    )
