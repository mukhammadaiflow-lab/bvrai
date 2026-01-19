"""
Tenants API Routes

Handles:
- Tenant management
- Team members
- Settings and preferences
- Feature flags
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_db_session,
    get_current_user,
    get_current_tenant,
    UserContext,
    TenantContext,
    require_permissions,
    require_roles,
)

router = APIRouter(prefix="/tenants")


# ============================================================================
# Schemas
# ============================================================================

class MemberRole(str, Enum):
    """Member roles."""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class InviteStatus(str, Enum):
    """Invite status."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    EXPIRED = "expired"
    REVOKED = "revoked"


class TenantResponse(BaseModel):
    """Tenant response."""
    id: str
    name: str
    slug: str
    plan: str
    features: List[str]
    limits: Dict[str, int]
    settings: Dict[str, Any]
    member_count: int
    created_at: datetime
    updated_at: Optional[datetime]


class TenantUpdate(BaseModel):
    """Update tenant request."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    settings: Optional[Dict[str, Any]] = None


class MemberResponse(BaseModel):
    """Team member response."""
    id: str
    user_id: str
    email: str
    name: str
    role: MemberRole
    joined_at: datetime
    last_active_at: Optional[datetime]


class MemberListResponse(BaseModel):
    """Member list response."""
    members: List[MemberResponse]
    total: int


class InviteRequest(BaseModel):
    """Invite member request."""
    email: EmailStr
    role: MemberRole = MemberRole.MEMBER
    message: Optional[str] = None


class InviteResponse(BaseModel):
    """Invite response."""
    id: str
    email: str
    role: MemberRole
    status: InviteStatus
    invited_by: str
    expires_at: datetime
    created_at: datetime


class InviteListResponse(BaseModel):
    """Invite list response."""
    invites: List[InviteResponse]
    total: int


class UpdateMemberRoleRequest(BaseModel):
    """Update member role request."""
    role: MemberRole


class SettingsUpdate(BaseModel):
    """Settings update request."""
    settings: Dict[str, Any]


class APISettingsResponse(BaseModel):
    """API settings response."""
    webhook_url: Optional[str]
    webhook_secret: str
    allowed_origins: List[str]
    rate_limit_override: Optional[int]


class BrandingSettings(BaseModel):
    """Branding settings."""
    logo_url: Optional[str]
    primary_color: Optional[str]
    accent_color: Optional[str]
    company_name: Optional[str]
    support_email: Optional[str]


# ============================================================================
# Tenant Info
# ============================================================================

@router.get("/current", response_model=TenantResponse)
async def get_current_tenant_info(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get current tenant information."""
    from app.tenancy import TenantService

    service = TenantService(db)
    tenant_data = await service.get(tenant.tenant_id)

    return TenantResponse(
        id=tenant_data.id,
        name=tenant_data.name,
        slug=tenant_data.slug,
        plan=tenant_data.plan,
        features=tenant_data.features or [],
        limits=tenant_data.limits or {},
        settings=tenant_data.settings or {},
        member_count=await service.get_member_count(tenant.tenant_id),
        created_at=tenant_data.created_at,
        updated_at=tenant_data.updated_at,
    )


@router.patch("/current", response_model=TenantResponse)
async def update_tenant(
    data: TenantUpdate,
    user: UserContext = Depends(require_permissions("tenant:update")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Update tenant information."""
    from app.tenancy import TenantService

    service = TenantService(db)

    tenant_data = await service.update(
        tenant_id=tenant.tenant_id,
        **data.model_dump(exclude_unset=True),
    )

    return TenantResponse(
        id=tenant_data.id,
        name=tenant_data.name,
        slug=tenant_data.slug,
        plan=tenant_data.plan,
        features=tenant_data.features or [],
        limits=tenant_data.limits or {},
        settings=tenant_data.settings or {},
        member_count=await service.get_member_count(tenant.tenant_id),
        created_at=tenant_data.created_at,
        updated_at=tenant_data.updated_at,
    )


# ============================================================================
# Team Members
# ============================================================================

@router.get("/members", response_model=MemberListResponse)
async def list_members(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """List team members."""
    from app.tenancy import TenantService

    service = TenantService(db)
    members = await service.list_members(tenant.tenant_id)

    return MemberListResponse(
        members=[
            MemberResponse(
                id=m.id,
                user_id=m.user_id,
                email=m.email,
                name=m.name,
                role=MemberRole(m.role),
                joined_at=m.joined_at,
                last_active_at=m.last_active_at,
            )
            for m in members
        ],
        total=len(members),
    )


@router.get("/members/{member_id}", response_model=MemberResponse)
async def get_member(
    member_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Get team member details."""
    from app.tenancy import TenantService

    service = TenantService(db)
    member = await service.get_member(str(member_id), tenant.tenant_id)

    if not member:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found",
        )

    return MemberResponse(
        id=member.id,
        user_id=member.user_id,
        email=member.email,
        name=member.name,
        role=MemberRole(member.role),
        joined_at=member.joined_at,
        last_active_at=member.last_active_at,
    )


@router.patch("/members/{member_id}", response_model=MemberResponse)
async def update_member_role(
    member_id: UUID,
    data: UpdateMemberRoleRequest,
    user: UserContext = Depends(require_roles("owner", "admin")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Update member role."""
    from app.tenancy import TenantService

    service = TenantService(db)

    # Can't change owner role
    member = await service.get_member(str(member_id), tenant.tenant_id)
    if not member:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found",
        )

    if member.role == "owner" and data.role != MemberRole.OWNER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change owner role. Transfer ownership first.",
        )

    updated = await service.update_member_role(
        member_id=str(member_id),
        tenant_id=tenant.tenant_id,
        role=data.role.value,
    )

    return MemberResponse(
        id=updated.id,
        user_id=updated.user_id,
        email=updated.email,
        name=updated.name,
        role=MemberRole(updated.role),
        joined_at=updated.joined_at,
        last_active_at=updated.last_active_at,
    )


@router.delete("/members/{member_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    member_id: UUID,
    user: UserContext = Depends(require_roles("owner", "admin")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Remove a team member."""
    from app.tenancy import TenantService

    service = TenantService(db)

    member = await service.get_member(str(member_id), tenant.tenant_id)
    if not member:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found",
        )

    if member.role == "owner":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove owner",
        )

    await service.remove_member(str(member_id), tenant.tenant_id)


# ============================================================================
# Invitations
# ============================================================================

@router.get("/invites", response_model=InviteListResponse)
async def list_invites(
    status_filter: Optional[InviteStatus] = Query(None, alias="status"),
    user: UserContext = Depends(require_roles("owner", "admin")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """List pending invitations."""
    from app.tenancy import TenantService

    service = TenantService(db)
    invites = await service.list_invites(
        tenant_id=tenant.tenant_id,
        status=status_filter.value if status_filter else None,
    )

    return InviteListResponse(
        invites=[
            InviteResponse(
                id=inv.id,
                email=inv.email,
                role=MemberRole(inv.role),
                status=InviteStatus(inv.status),
                invited_by=inv.invited_by,
                expires_at=inv.expires_at,
                created_at=inv.created_at,
            )
            for inv in invites
        ],
        total=len(invites),
    )


@router.post("/invites", response_model=InviteResponse, status_code=status.HTTP_201_CREATED)
async def invite_member(
    data: InviteRequest,
    user: UserContext = Depends(require_roles("owner", "admin")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Invite a new team member."""
    from app.tenancy import TenantService

    service = TenantService(db)

    # Check if already a member
    existing = await service.get_member_by_email(data.email, tenant.tenant_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User is already a team member",
        )

    # Check for pending invite
    pending = await service.get_pending_invite(data.email, tenant.tenant_id)
    if pending:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Invitation already pending for this email",
        )

    invite = await service.create_invite(
        tenant_id=tenant.tenant_id,
        email=data.email,
        role=data.role.value,
        invited_by=user.user_id,
        message=data.message,
    )

    return InviteResponse(
        id=invite.id,
        email=invite.email,
        role=MemberRole(invite.role),
        status=InviteStatus(invite.status),
        invited_by=invite.invited_by,
        expires_at=invite.expires_at,
        created_at=invite.created_at,
    )


@router.delete("/invites/{invite_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_invite(
    invite_id: UUID,
    user: UserContext = Depends(require_roles("owner", "admin")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Revoke an invitation."""
    from app.tenancy import TenantService

    service = TenantService(db)
    await service.revoke_invite(str(invite_id), tenant.tenant_id)


@router.post("/invites/{invite_id}/resend", response_model=InviteResponse)
async def resend_invite(
    invite_id: UUID,
    user: UserContext = Depends(require_roles("owner", "admin")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Resend an invitation."""
    from app.tenancy import TenantService

    service = TenantService(db)

    invite = await service.resend_invite(str(invite_id), tenant.tenant_id)

    if not invite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invitation not found",
        )

    return InviteResponse(
        id=invite.id,
        email=invite.email,
        role=MemberRole(invite.role),
        status=InviteStatus(invite.status),
        invited_by=invite.invited_by,
        expires_at=invite.expires_at,
        created_at=invite.created_at,
    )


# ============================================================================
# Settings
# ============================================================================

@router.get("/settings")
async def get_settings(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Get tenant settings."""
    return {"settings": tenant.settings}


@router.patch("/settings")
async def update_settings(
    data: SettingsUpdate,
    user: UserContext = Depends(require_permissions("tenant:update")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Update tenant settings."""
    from app.tenancy import TenantService

    service = TenantService(db)

    # Merge settings
    current = tenant.settings.copy()
    current.update(data.settings)

    await service.update(tenant.tenant_id, settings=current)

    return {"settings": current}


@router.get("/settings/api", response_model=APISettingsResponse)
async def get_api_settings(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Get API settings."""
    return APISettingsResponse(
        webhook_url=tenant.settings.get("webhook_url"),
        webhook_secret=tenant.settings.get("webhook_secret", ""),
        allowed_origins=tenant.settings.get("allowed_origins", []),
        rate_limit_override=tenant.settings.get("rate_limit_override"),
    )


@router.get("/settings/branding", response_model=BrandingSettings)
async def get_branding_settings(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Get branding settings."""
    branding = tenant.settings.get("branding", {})
    return BrandingSettings(**branding)


@router.patch("/settings/branding", response_model=BrandingSettings)
async def update_branding_settings(
    data: BrandingSettings,
    user: UserContext = Depends(require_permissions("tenant:update")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Update branding settings."""
    from app.tenancy import TenantService

    service = TenantService(db)

    current = tenant.settings.copy()
    current["branding"] = data.model_dump(exclude_unset=True)

    await service.update(tenant.tenant_id, settings=current)

    return data


# ============================================================================
# Features
# ============================================================================

@router.get("/features")
async def get_features(
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Get enabled features."""
    return {
        "features": tenant.features,
        "limits": tenant.limits,
    }


# ============================================================================
# Danger Zone
# ============================================================================

@router.post("/transfer-ownership")
async def transfer_ownership(
    new_owner_id: UUID,
    user: UserContext = Depends(require_roles("owner")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Transfer tenant ownership."""
    from app.tenancy import TenantService

    service = TenantService(db)

    # Verify new owner is a member
    member = await service.get_member(str(new_owner_id), tenant.tenant_id)
    if not member:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New owner must be a team member",
        )

    await service.transfer_ownership(
        tenant_id=tenant.tenant_id,
        current_owner_id=user.user_id,
        new_owner_id=str(new_owner_id),
    )

    return {"message": "Ownership transferred successfully"}


@router.delete("/current", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tenant(
    confirm: str = Query(..., description="Type 'DELETE' to confirm"),
    user: UserContext = Depends(require_roles("owner")),
    tenant: TenantContext = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete tenant and all data. This action is irreversible."""
    if confirm != "DELETE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please type 'DELETE' to confirm",
        )

    from app.tenancy import TenantService

    service = TenantService(db)
    await service.delete_tenant(tenant.tenant_id)
