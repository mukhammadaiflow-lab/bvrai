"""Authentication dependencies."""

from typing import Optional
from uuid import UUID
from datetime import datetime, timezone
import hashlib
import hmac
import os

from fastapi import Depends, HTTPException, Header, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.database.models import User, APIKey

import structlog

logger = structlog.get_logger()

# Environment-based configuration
ENFORCE_AUTH = os.getenv("ENFORCE_AUTH", "true").lower() == "true"
API_KEY_PEPPER = os.getenv("API_KEY_PEPPER", "bvrai-secure-pepper-change-in-production")


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key using SHA-256 with pepper for secure storage.

    Uses HMAC-SHA256 with a pepper for additional security.
    This provides:
    - Consistent hashing for lookups
    - Protection against rainbow table attacks via pepper
    - Fast verification for API authentication
    """
    return hmac.new(
        API_KEY_PEPPER.encode(),
        api_key.encode(),
        hashlib.sha256
    ).hexdigest()


def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """
    Securely compare an API key against its stored hash.
    Uses constant-time comparison to prevent timing attacks.
    """
    provided_hash = hash_api_key(provided_key)
    return hmac.compare_digest(provided_hash, stored_hash)


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

    # Hash the provided API key for comparison
    api_key_hash = hash_api_key(api_key)

    # Look up API key by hash
    query = select(APIKey).where(
        APIKey.key_hash == api_key_hash,
        APIKey.revoked_at.is_(None),
    )
    result = await db.execute(query)
    key_record = result.scalar_one_or_none()

    if not key_record:
        logger.warning("Invalid API key attempt", api_key_prefix=api_key[:8] + "...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Check expiration
    if key_record.expires_at and key_record.expires_at < datetime.now(timezone.utc):
        logger.warning("Expired API key used", key_id=str(key_record.id))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key expired",
        )

    # Update last used timestamp
    key_record.last_used_at = datetime.now(timezone.utc)
    await db.commit()

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

    In development mode (ENFORCE_AUTH=false), returns a default user ID.
    In production mode (ENFORCE_AUTH=true), requires valid API key.
    """
    if not api_key:
        if not ENFORCE_AUTH:
            # Development mode only - return a default user ID
            logger.warning("No API key provided, using development user (ENFORCE_AUTH=false)")
            return UUID("00000000-0000-0000-0000-000000000001")
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Hash the provided API key for comparison
    api_key_hash = hash_api_key(api_key)

    # Look up API key by hash
    query = select(APIKey).where(
        APIKey.key_hash == api_key_hash,
        APIKey.revoked_at.is_(None),
    )
    result = await db.execute(query)
    key_record = result.scalar_one_or_none()

    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Check expiration
    if key_record.expires_at and key_record.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key expired",
        )

    # Update last used timestamp
    key_record.last_used_at = datetime.now(timezone.utc)
    await db.commit()

    return key_record.user_id


async def get_optional_user_id(
    api_key: Optional[str] = Depends(get_api_key),
    db: AsyncSession = Depends(get_db),
) -> Optional[UUID]:
    """Get current user ID if authenticated, None otherwise."""
    if not api_key:
        return None

    # Hash the provided API key for comparison
    api_key_hash = hash_api_key(api_key)

    query = select(APIKey).where(
        APIKey.key_hash == api_key_hash,
        APIKey.revoked_at.is_(None),
    )
    result = await db.execute(query)
    key_record = result.scalar_one_or_none()

    if not key_record:
        return None

    # Check expiration
    if key_record.expires_at and key_record.expires_at < datetime.now(timezone.utc):
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
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Hash the provided API key for comparison
        api_key_hash = hash_api_key(api_key)

        query = select(APIKey).where(
            APIKey.key_hash == api_key_hash,
            APIKey.revoked_at.is_(None),
        )
        result = await db.execute(query)
        key_record = result.scalar_one_or_none()

        if not key_record:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        # Check expiration
        if key_record.expires_at and key_record.expires_at < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key expired",
            )

        # Check scopes
        key_scopes = set(key_record.scopes or [])
        required_scopes = set(scopes)

        if not required_scopes.issubset(key_scopes):
            missing = required_scopes - key_scopes
            logger.warning(
                "Insufficient scopes",
                key_id=str(key_record.id),
                required=list(required_scopes),
                provided=list(key_scopes),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scopes: {list(missing)}",
            )

        # Update last used timestamp
        key_record.last_used_at = datetime.now(timezone.utc)
        await db.commit()

        return key_record

    return check_scopes
