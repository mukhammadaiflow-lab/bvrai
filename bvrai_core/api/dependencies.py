"""
API Dependencies

This module provides FastAPI dependencies for database sessions,
authentication, and other shared resources.
"""

import logging
import os
from dataclasses import field
from datetime import datetime
from typing import AsyncGenerator, Optional, Set

from fastapi import Depends, Header, Request
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..database.base import get_session
from ..database.models import APIKey, Organization, User

from .auth import (
    AuthContext,
    AuthMethod,
    Permission,
    Role,
    ROLE_PERMISSIONS,
)
from .base import (
    AuthenticationError,
    AuthorizationError,
    ErrorCode,
    hash_api_key,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Database Session Dependency
# =============================================================================

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Usage in routes:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db_session)):
            repo = ItemRepository(db)
            return await repo.get_all()
    """
    async with get_session() as session:
        yield session


# Alias for shorter import
get_db = get_db_session


# =============================================================================
# Security Schemes
# =============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


# =============================================================================
# Authentication Dependencies
# =============================================================================

async def get_api_key(
    api_key: Optional[str] = Depends(api_key_header),
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[str]:
    """Extract API key from header or Bearer token."""
    if api_key:
        return api_key
    if authorization and authorization.scheme.lower() == "bearer":
        # Check if it's an API key (starts with bvr_)
        if authorization.credentials.startswith("bvr_"):
            return authorization.credentials
    return None


async def get_auth_context(
    request: Request,
    api_key: Optional[str] = Depends(get_api_key),
    db: AsyncSession = Depends(get_db_session),
) -> AuthContext:
    """
    Get authentication context from request.

    This is the main authentication dependency used in API routes.

    Usage in routes:
        @router.get("/agents")
        async def list_agents(auth: AuthContext = Depends(get_auth_context)):
            auth.require_permission(Permission.AGENTS_READ)
            # ... use auth.organization_id, etc.
    """
    # Get client info
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    # Development mode - allow unauthenticated access
    dev_mode = os.getenv("BVRAI_DEV_MODE", "false").lower() == "true"
    dev_org_id = os.getenv("BVRAI_DEV_ORG_ID", "dev-org-001")

    if dev_mode and not api_key:
        logger.debug("Dev mode: Creating default auth context")
        return AuthContext(
            method=AuthMethod.API_KEY,
            is_authenticated=True,
            user_id="dev-user-001",
            organization_id=dev_org_id,
            role=Role.OWNER,
            permissions=set(Permission),  # All permissions
            client_ip=client_ip,
            user_agent=user_agent,
        )

    if not api_key:
        raise AuthenticationError(
            message="API key required. Provide via X-API-Key header or Bearer token.",
            code=ErrorCode.AUTHENTICATION_REQUIRED,
        )

    # Validate API key format
    if not api_key.startswith("bvr_"):
        raise AuthenticationError(
            message="Invalid API key format. Keys should start with 'bvr_'",
            code=ErrorCode.INVALID_API_KEY,
        )

    # Look up API key in database
    key_hash = hash_api_key(api_key)

    result = await db.execute(
        select(APIKey).where(APIKey.key_hash == key_hash)
    )
    db_key = result.scalar_one_or_none()

    if not db_key:
        raise AuthenticationError(
            message="Invalid API key",
            code=ErrorCode.INVALID_API_KEY,
        )

    # Check if key is active
    if not db_key.is_active:
        raise AuthenticationError(
            message="API key has been revoked",
            code=ErrorCode.INVALID_API_KEY,
        )

    # Check expiration
    if db_key.expires_at and datetime.utcnow() > db_key.expires_at:
        raise AuthenticationError(
            message="API key has expired",
            code=ErrorCode.EXPIRED_TOKEN,
        )

    # Build permissions set
    permissions: Set[Permission] = set()

    # Get role permissions
    role = None
    if db_key.role:
        try:
            role = Role(db_key.role)
            role_perms = ROLE_PERMISSIONS.get(role, set())
            permissions.update(role_perms)
        except ValueError:
            pass

    # Add explicit permissions
    if db_key.permissions:
        for perm_str in db_key.permissions:
            try:
                permissions.add(Permission(perm_str))
            except ValueError:
                pass

    # Update last used timestamp
    db_key.last_used_at = datetime.utcnow()
    await db.commit()

    return AuthContext(
        method=AuthMethod.API_KEY,
        is_authenticated=True,
        user_id=db_key.user_id,
        organization_id=db_key.organization_id,
        api_key_id=db_key.id,
        role=role,
        permissions=permissions,
        client_ip=client_ip,
        user_agent=user_agent,
    )


# =============================================================================
# Simplified Auth Dependency (for backward compatibility)
# =============================================================================

# This allows routes to use `auth: AuthContext = Depends()` pattern
# by making AuthContext itself a callable dependency
AuthContext.__call__ = staticmethod(get_auth_context)


# =============================================================================
# Permission Check Dependencies
# =============================================================================

def require_permission(permission: Permission):
    """
    Dependency factory that requires a specific permission.

    Usage:
        @router.get("/agents")
        async def list_agents(
            auth: AuthContext = Depends(require_permission(Permission.AGENTS_READ))
        ):
            ...
    """
    async def dependency(
        auth: AuthContext = Depends(get_auth_context),
    ) -> AuthContext:
        auth.require_permission(permission)
        return auth
    return dependency


def require_any_permission(*permissions: Permission):
    """Dependency factory that requires any of the specified permissions."""
    async def dependency(
        auth: AuthContext = Depends(get_auth_context),
    ) -> AuthContext:
        auth.require_any_permission(list(permissions))
        return auth
    return dependency


def require_role(role: Role):
    """Dependency factory that requires a specific role or higher."""
    role_hierarchy = [
        Role.VIEWER, Role.ANALYST, Role.OPERATOR,
        Role.DEVELOPER, Role.ADMIN, Role.OWNER
    ]

    async def dependency(
        auth: AuthContext = Depends(get_auth_context),
    ) -> AuthContext:
        if not auth.role:
            raise AuthorizationError(
                message=f"Role '{role.value}' or higher required",
            )

        try:
            required_level = role_hierarchy.index(role)
            current_level = role_hierarchy.index(auth.role)
            if current_level < required_level:
                raise AuthorizationError(
                    message=f"Role '{role.value}' or higher required",
                )
        except ValueError:
            raise AuthorizationError(
                message=f"Role '{role.value}' or higher required",
            )

        return auth

    return dependency


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "get_db_session",
    "get_db",
    "get_auth_context",
    "require_permission",
    "require_any_permission",
    "require_role",
]
