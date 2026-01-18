"""
API Keys Management Routes

Provides REST API endpoints for managing API keys.
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..base import APIResponse, success_response, NotFoundError, hash_api_key
from ..auth import AuthContext, Permission
from ..dependencies import get_db_session, get_auth_context
from ...database.models import APIKey


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api-keys", tags=["API Keys"])


# =============================================================================
# Request/Response Models
# =============================================================================

class APIKeyResponse(BaseModel):
    """API key response (without full key)."""
    id: str
    name: str
    key_prefix: str
    scopes: Optional[List[str]] = None
    last_used: Optional[str] = None
    expires_at: Optional[str] = None
    created_at: str
    is_active: bool


class APIKeyWithSecretResponse(APIKeyResponse):
    """API key response with full key (only on creation)."""
    key: str


class CreateAPIKeyRequest(BaseModel):
    """Create API key request."""
    name: str
    scopes: Optional[List[str]] = None
    expires_in_days: Optional[int] = None


# =============================================================================
# Helper Functions
# =============================================================================

def generate_api_key() -> tuple[str, str]:
    """Generate an API key and its hash."""
    key = f"bvr_{secrets.token_urlsafe(32)}"
    key_hash = hash_api_key(key)
    return key, key_hash


def api_key_to_response(api_key: APIKey, include_key: bool = False, full_key: str = None) -> dict:
    """Convert API key model to response dict."""
    response = {
        "id": api_key.id,
        "name": api_key.name,
        "key_prefix": api_key.key_prefix,
        "scopes": api_key.scopes,
        "last_used": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
        "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
        "created_at": api_key.created_at.isoformat() if api_key.created_at else None,
        "is_active": api_key.is_active,
    }

    if include_key and full_key:
        response["key"] = full_key

    return response


# =============================================================================
# Routes
# =============================================================================

@router.get(
    "",
    response_model=APIResponse[List[APIKeyResponse]],
    summary="List API Keys",
    description="List all API keys for the organization.",
)
async def list_api_keys(
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """List API keys."""
    auth.require_permission(Permission.API_KEYS_MANAGE)

    result = await db.execute(
        select(APIKey).where(
            APIKey.organization_id == auth.organization_id,
            APIKey.is_deleted == False,
        ).order_by(APIKey.created_at.desc())
    )
    keys = result.scalars().all()

    return success_response([api_key_to_response(k) for k in keys])


@router.post(
    "",
    response_model=APIResponse[APIKeyWithSecretResponse],
    status_code=201,
    summary="Create API Key",
    description="Create a new API key. The full key is only shown once.",
)
async def create_api_key(
    request: CreateAPIKeyRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new API key."""
    auth.require_permission(Permission.API_KEYS_MANAGE)

    # Generate key
    key, key_hash = generate_api_key()

    # Calculate expiration
    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)

    # Create API key
    api_key = APIKey(
        organization_id=auth.organization_id,
        created_by_user_id=auth.user_id,
        name=request.name,
        key_hash=key_hash,
        key_prefix=key[:12],
        scopes=request.scopes or ["*"],
        is_active=True,
        expires_at=expires_at,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    logger.info(f"API key created: {api_key.id}")

    # Return with full key (only time it's shown)
    return success_response(api_key_to_response(api_key, include_key=True, full_key=key))


@router.delete(
    "/{key_id}",
    status_code=204,
    summary="Revoke API Key",
    description="Revoke an API key.",
)
async def revoke_api_key(
    key_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Revoke an API key."""
    auth.require_permission(Permission.API_KEYS_MANAGE)

    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.organization_id == auth.organization_id,
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise NotFoundError("API key not found")

    # Don't allow revoking the current key
    if api_key.id == auth.api_key_id:
        raise HTTPException(
            status_code=400,
            detail="Cannot revoke the API key currently in use"
        )

    api_key.is_active = False
    api_key.is_deleted = True
    await db.commit()

    logger.info(f"API key revoked: {key_id}")
    return None


@router.post(
    "/{key_id}/regenerate",
    response_model=APIResponse[APIKeyWithSecretResponse],
    summary="Regenerate API Key",
    description="Regenerate an API key. The old key will stop working.",
)
async def regenerate_api_key(
    key_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Regenerate an API key."""
    auth.require_permission(Permission.API_KEYS_MANAGE)

    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.organization_id == auth.organization_id,
        )
    )
    old_key = result.scalar_one_or_none()

    if not old_key:
        raise NotFoundError("API key not found")

    # Generate new key
    key, key_hash = generate_api_key()

    # Update key
    old_key.key_hash = key_hash
    old_key.key_prefix = key[:12]
    old_key.is_active = True

    await db.commit()
    await db.refresh(old_key)

    logger.info(f"API key regenerated: {key_id}")

    return success_response(api_key_to_response(old_key, include_key=True, full_key=key))
