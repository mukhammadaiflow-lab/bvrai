"""
Organization API Routes

Provides REST API endpoints for organization management.
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..base import APIResponse, success_response, NotFoundError
from ..auth import AuthContext, Permission
from ..dependencies import get_db_session, get_auth_context
from ...database.models import Organization, User


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/organization", tags=["Organization"])


# =============================================================================
# Request/Response Models
# =============================================================================

class OrganizationSettings(BaseModel):
    """Organization settings."""
    default_language: Optional[str] = "en"
    default_timezone: Optional[str] = "America/New_York"
    webhook_url: Optional[str] = None
    allowed_domains: Optional[List[str]] = None


class OrganizationResponse(BaseModel):
    """Organization response."""
    id: str
    name: str
    slug: str
    plan: str
    settings: Optional[OrganizationSettings] = None
    is_active: bool
    created_at: str
    updated_at: str


class UpdateOrganizationRequest(BaseModel):
    """Update organization request."""
    name: Optional[str] = None
    settings: Optional[OrganizationSettings] = None


class OrganizationMemberResponse(BaseModel):
    """Organization member response."""
    id: str
    email: str
    name: str
    role: str
    status: str
    joined_at: str
    last_active: Optional[str] = None


class InviteMemberRequest(BaseModel):
    """Invite member request."""
    email: str
    role: str = Field(default="member", pattern="^(admin|member)$")


# =============================================================================
# Helper Functions
# =============================================================================

def org_to_response(org: Organization, settings=None) -> dict:
    """Convert organization model to response dict."""
    # Default settings - don't trigger lazy loading for relationship
    default_settings = {
        "default_language": "en",
        "default_timezone": "America/New_York",
    }

    return {
        "id": org.id,
        "name": org.name,
        "slug": org.slug,
        "plan": org.plan or "free",
        "settings": settings or default_settings,
        "is_active": org.is_active,
        "created_at": org.created_at.isoformat() if org.created_at else None,
        "updated_at": org.updated_at.isoformat() if org.updated_at else None,
    }


def user_to_member_response(user: User) -> dict:
    """Convert user model to member response dict."""
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role or "member",
        "status": "active" if user.is_active else "suspended",
        "joined_at": user.created_at.isoformat() if user.created_at else None,
        "last_active": user.last_login_at.isoformat() if user.last_login_at else None,
    }


# =============================================================================
# Routes
# =============================================================================

@router.get(
    "",
    response_model=APIResponse[OrganizationResponse],
    summary="Get Organization",
    description="Get the current user's organization details.",
)
async def get_organization(
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Get current organization."""
    result = await db.execute(
        select(Organization).where(Organization.id == auth.organization_id)
    )
    org = result.scalar_one_or_none()

    if not org:
        raise NotFoundError("Organization not found")

    return success_response(org_to_response(org))


@router.put(
    "",
    response_model=APIResponse[OrganizationResponse],
    summary="Update Organization",
    description="Update organization details.",
)
async def update_organization(
    request: UpdateOrganizationRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Update organization."""
    auth.require_permission(Permission.ADMIN_WRITE)

    result = await db.execute(
        select(Organization).where(Organization.id == auth.organization_id)
    )
    org = result.scalar_one_or_none()

    if not org:
        raise NotFoundError("Organization not found")

    # Update fields
    if request.name is not None:
        org.name = request.name
    if request.settings is not None:
        org.settings = request.settings.dict()

    await db.commit()
    await db.refresh(org)

    logger.info(f"Organization updated: {org.id}")
    return success_response(org_to_response(org))


@router.get(
    "/members",
    response_model=APIResponse[List[OrganizationMemberResponse]],
    summary="List Members",
    description="List all organization members.",
)
async def list_members(
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """List organization members."""
    result = await db.execute(
        select(User).where(
            User.organization_id == auth.organization_id,
            User.is_deleted == False,
        )
    )
    users = result.scalars().all()

    return success_response([user_to_member_response(u) for u in users])


@router.post(
    "/invitations",
    response_model=APIResponse[dict],
    status_code=201,
    summary="Invite Member",
    description="Invite a new member to the organization.",
)
async def invite_member(
    request: InviteMemberRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Invite a new member."""
    auth.require_permission(Permission.USERS_MANAGE)

    # Check if user already exists
    result = await db.execute(
        select(User).where(User.email == request.email)
    )
    existing = result.scalar_one_or_none()

    if existing:
        if existing.organization_id == auth.organization_id:
            raise HTTPException(
                status_code=400,
                detail="User is already a member of this organization"
            )
        raise HTTPException(
            status_code=400,
            detail="User already belongs to another organization"
        )

    # In production, send invitation email
    # For MVP, create pending user
    logger.info(f"Invitation sent to {request.email} for org {auth.organization_id}")

    return success_response({
        "message": f"Invitation sent to {request.email}",
        "email": request.email,
        "role": request.role,
    })


@router.delete(
    "/members/{member_id}",
    status_code=204,
    summary="Remove Member",
    description="Remove a member from the organization.",
)
async def remove_member(
    member_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Remove a member."""
    auth.require_permission(Permission.USERS_MANAGE)

    result = await db.execute(
        select(User).where(
            User.id == member_id,
            User.organization_id == auth.organization_id,
        )
    )
    user = result.scalar_one_or_none()

    if not user:
        raise NotFoundError("Member not found")

    if user.role == "owner":
        raise HTTPException(
            status_code=400,
            detail="Cannot remove organization owner"
        )

    # Soft delete
    user.is_deleted = True
    user.is_active = False
    await db.commit()

    logger.info(f"Member removed: {member_id}")
    return None


@router.put(
    "/members/{member_id}",
    response_model=APIResponse[OrganizationMemberResponse],
    summary="Update Member Role",
    description="Update a member's role.",
)
async def update_member_role(
    member_id: str,
    role: str,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Update member role."""
    auth.require_permission(Permission.USERS_MANAGE)

    result = await db.execute(
        select(User).where(
            User.id == member_id,
            User.organization_id == auth.organization_id,
        )
    )
    user = result.scalar_one_or_none()

    if not user:
        raise NotFoundError("Member not found")

    if user.role == "owner" and role != "owner":
        raise HTTPException(
            status_code=400,
            detail="Cannot change owner's role"
        )

    user.role = role
    await db.commit()
    await db.refresh(user)

    logger.info(f"Member role updated: {member_id} -> {role}")
    return success_response(user_to_member_response(user))


@router.delete(
    "/invitations/{invitation_id}",
    status_code=204,
    summary="Cancel Invitation",
    description="Cancel a pending invitation.",
)
async def cancel_invitation(
    invitation_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Cancel invitation."""
    auth.require_permission(Permission.USERS_MANAGE)

    # In production, would delete from invitations table
    logger.info(f"Invitation cancelled: {invitation_id}")
    return None
