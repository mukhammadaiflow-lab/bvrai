"""Authentication dependencies."""

from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Header, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.database.models import User, APIKey

import structlog

logger = structlog.get_logger()


async def get_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> Optional[str]:
    """Extract API key from headers."""
    # Check X-API-Key header first
    if x_api_key:
        return x_api_key

    # Check Authorization header (Bearer token)
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]

    return None


async def get_current_user(
    api_key: Optional[str] = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Get current user from API key."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Look up API key
    query = select(APIKey).where(
        APIKey.key_hash == api_key,  # TODO: Hash the key properly
        APIKey.revoked_at.is_(None),
    )
    result = await db.execute(query)
    key_record = result.scalar_one_or_none()

    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Update last used
    key_record.last_used_at = key_record.last_used_at  # TODO: Update timestamp

    # Get user
    user_query = select(User).where(User.id == key_record.user_id)
    user_result = await db.execute(user_query)
    user = user_result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return user


async def get_current_user_id(
    api_key: Optional[str] = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
) -> UUID:
    """Get current user ID from API key.

    For development, returns a default user ID if no key provided.
    """
    if not api_key:
        # Development mode - return a default user ID
        # In production, this should raise 401
        logger.warning("No API key provided, using development user")
        return UUID("00000000-0000-0000-0000-000000000001")

    # Look up API key
    query = select(APIKey).where(
        APIKey.key_hash == api_key,
        APIKey.revoked_at.is_(None),
    )
    result = await db.execute(query)
    key_record = result.scalar_one_or_none()

    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return key_record.user_id


async def get_optional_user_id(
    api_key: Optional[str] = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
) -> Optional[UUID]:
    """Get current user ID if authenticated, None otherwise."""
    if not api_key:
        return None

    query = select(APIKey).where(
        APIKey.key_hash == api_key,
        APIKey.revoked_at.is_(None),
    )
    result = await db.execute(query)
    key_record = result.scalar_one_or_none()

    if not key_record:
        return None

    return key_record.user_id


def require_scopes(*scopes: str):
    """Dependency that requires specific API key scopes."""

    async def check_scopes(
        api_key: Optional[str] = Depends(get_api_key),
        db: AsyncSession = Depends(get_db),
    ) -> APIKey:
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
            )

        query = select(APIKey).where(
            APIKey.key_hash == api_key,
            APIKey.revoked_at.is_(None),
        )
        result = await db.execute(query)
        key_record = result.scalar_one_or_none()

        if not key_record:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        # Check scopes
        key_scopes = set(key_record.scopes or [])
        required_scopes = set(scopes)

        if not required_scopes.issubset(key_scopes):
            missing = required_scopes - key_scopes
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scopes: {missing}",
            )

        return key_record

    return check_scopes
