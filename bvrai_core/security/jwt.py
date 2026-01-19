"""
JWT Token Management with Key Rotation

This module provides secure JWT token generation, validation,
and key rotation support for the platform.
"""

import json
import os
import secrets
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import jwt
from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================


DEFAULT_ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS = 7
KEY_ID_LENGTH = 8  # Bytes


# =============================================================================
# Models
# =============================================================================


class JWTKey(BaseModel):
    """A JWT signing key with metadata."""

    kid: str  # Key ID
    secret: str  # The actual secret
    algorithm: str = DEFAULT_ALGORITHM
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = True


class TokenPayload(BaseModel):
    """JWT token payload."""

    sub: str  # Subject (user ID)
    org: Optional[str] = None  # Organization ID
    email: Optional[str] = None
    role: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    iat: datetime = Field(default_factory=datetime.utcnow)  # Issued at
    exp: datetime  # Expiration
    jti: Optional[str] = None  # JWT ID (for blacklisting)
    type: str = "access"  # Token type: access, refresh


class TokenPair(BaseModel):
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int  # Seconds until access token expires


# =============================================================================
# Key Manager
# =============================================================================


class JWTKeyManager:
    """
    Manages JWT signing keys with rotation support.

    Supports multiple active keys for zero-downtime rotation.
    """

    def __init__(
        self,
        primary_secret: Optional[str] = None,
        algorithm: str = DEFAULT_ALGORITHM,
    ):
        """
        Initialize the key manager.

        Args:
            primary_secret: Primary signing secret (from env or generated)
            algorithm: JWT algorithm to use
        """
        self.algorithm = algorithm
        self._keys: Dict[str, JWTKey] = {}

        # Initialize with primary key
        primary_secret = primary_secret or os.getenv("JWT_SECRET")
        if not primary_secret:
            primary_secret = secrets.token_urlsafe(32)

        primary_kid = self._generate_key_id()
        self._keys[primary_kid] = JWTKey(
            kid=primary_kid,
            secret=primary_secret,
            algorithm=algorithm,
        )
        self._primary_kid = primary_kid

    def _generate_key_id(self) -> str:
        """Generate a unique key ID."""
        return secrets.token_hex(KEY_ID_LENGTH)

    @property
    def primary_key(self) -> JWTKey:
        """Get the primary signing key."""
        return self._keys[self._primary_kid]

    @property
    def active_keys(self) -> List[JWTKey]:
        """Get all active keys."""
        return [k for k in self._keys.values() if k.is_active]

    def add_key(
        self,
        secret: Optional[str] = None,
        make_primary: bool = False,
        expires_in_days: Optional[int] = None,
    ) -> JWTKey:
        """
        Add a new signing key.

        Args:
            secret: The key secret (generated if not provided)
            make_primary: Make this the primary signing key
            expires_in_days: Key expiration in days

        Returns:
            The new JWTKey
        """
        kid = self._generate_key_id()
        secret = secret or secrets.token_urlsafe(32)

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        key = JWTKey(
            kid=kid,
            secret=secret,
            algorithm=self.algorithm,
            expires_at=expires_at,
        )

        self._keys[kid] = key

        if make_primary:
            self._primary_kid = kid

        return key

    def rotate_primary(self) -> JWTKey:
        """
        Rotate the primary key.

        Creates a new primary key and marks the old one as non-primary
        (but still valid for verification).

        Returns:
            The new primary key
        """
        # Create new key
        new_key = self.add_key(make_primary=True)

        # Set old key to expire in 24 hours (enough time for refresh tokens)
        old_kid = self._primary_kid
        if old_kid != new_key.kid:
            old_key = self._keys.get(old_kid)
            if old_key:
                old_key.expires_at = datetime.utcnow() + timedelta(hours=24)

        return new_key

    def deactivate_key(self, kid: str) -> bool:
        """Deactivate a key by ID."""
        if kid in self._keys:
            if kid == self._primary_kid:
                return False  # Can't deactivate primary
            self._keys[kid].is_active = False
            return True
        return False

    def cleanup_expired(self) -> int:
        """
        Remove expired keys.

        Returns:
            Number of keys removed
        """
        now = datetime.utcnow()
        expired = [
            kid for kid, key in self._keys.items()
            if key.expires_at and key.expires_at < now and kid != self._primary_kid
        ]

        for kid in expired:
            del self._keys[kid]

        return len(expired)

    def get_key(self, kid: str) -> Optional[JWTKey]:
        """Get a key by ID."""
        return self._keys.get(kid)

    def export_keys(self) -> Dict[str, Any]:
        """Export keys for persistence (secrets should be encrypted)."""
        return {
            "primary_kid": self._primary_kid,
            "keys": {kid: key.model_dump() for kid, key in self._keys.items()},
        }

    def import_keys(self, data: Dict[str, Any]) -> None:
        """Import keys from persistence."""
        self._primary_kid = data.get("primary_kid", self._primary_kid)
        for kid, key_data in data.get("keys", {}).items():
            self._keys[kid] = JWTKey(**key_data)


# =============================================================================
# Token Blacklist
# =============================================================================


class TokenBlacklist(ABC):
    """Abstract base class for token blacklist storage."""

    @abstractmethod
    async def add(self, jti: str, exp: datetime) -> bool:
        """Add a token JTI to the blacklist."""
        pass

    @abstractmethod
    async def contains(self, jti: str) -> bool:
        """Check if a token JTI is blacklisted."""
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired entries from the blacklist."""
        pass


class InMemoryTokenBlacklist(TokenBlacklist):
    """
    In-memory token blacklist for development/testing.

    For production, use RedisTokenBlacklist instead.
    """

    def __init__(self):
        self._blacklist: Dict[str, datetime] = {}

    async def add(self, jti: str, exp: datetime) -> bool:
        """Add a token JTI to the blacklist with its expiration."""
        self._blacklist[jti] = exp
        return True

    async def contains(self, jti: str) -> bool:
        """Check if a token JTI is blacklisted."""
        if jti not in self._blacklist:
            return False
        # Check if the blacklist entry has expired
        if self._blacklist[jti] < datetime.utcnow():
            del self._blacklist[jti]
            return False
        return True

    async def cleanup_expired(self) -> int:
        """Remove expired entries from the blacklist."""
        now = datetime.utcnow()
        expired = [jti for jti, exp in self._blacklist.items() if exp < now]
        for jti in expired:
            del self._blacklist[jti]
        return len(expired)

    def sync_add(self, jti: str, exp: datetime) -> bool:
        """Synchronous version of add for compatibility."""
        self._blacklist[jti] = exp
        return True

    def sync_contains(self, jti: str) -> bool:
        """Synchronous version of contains for compatibility."""
        if jti not in self._blacklist:
            return False
        if self._blacklist[jti] < datetime.utcnow():
            del self._blacklist[jti]
            return False
        return True


class RedisTokenBlacklist(TokenBlacklist):
    """
    Redis-backed token blacklist for production.

    Uses Redis TTL for automatic expiration of blacklisted tokens.
    """

    def __init__(self, redis_url: Optional[str] = None, prefix: str = "token_blacklist:"):
        self.prefix = prefix
        self._redis = None
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")

    async def _get_redis(self):
        """Lazy initialization of Redis client."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self._redis_url)
            except ImportError:
                raise RuntimeError("redis package is required for RedisTokenBlacklist")
        return self._redis

    async def add(self, jti: str, exp: datetime) -> bool:
        """Add a token JTI to the blacklist with TTL based on expiration."""
        redis = await self._get_redis()
        ttl = int((exp - datetime.utcnow()).total_seconds())
        if ttl > 0:
            await redis.setex(f"{self.prefix}{jti}", ttl, "1")
            return True
        return False

    async def contains(self, jti: str) -> bool:
        """Check if a token JTI is blacklisted."""
        redis = await self._get_redis()
        result = await redis.exists(f"{self.prefix}{jti}")
        return bool(result)

    async def cleanup_expired(self) -> int:
        """Redis handles expiration automatically via TTL."""
        return 0


# =============================================================================
# Token Service
# =============================================================================


class JWTService:
    """
    Service for JWT token operations.

    Usage:
        service = JWTService()
        tokens = service.create_tokens(user_id, org_id, email, role)
        payload = service.verify_token(tokens.access_token)
    """

    def __init__(
        self,
        key_manager: Optional[JWTKeyManager] = None,
        blacklist: Optional[TokenBlacklist] = None,
        access_token_expire_minutes: int = DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES,
        refresh_token_expire_days: int = DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS,
    ):
        """
        Initialize the JWT service.

        Args:
            key_manager: JWT key manager (created if not provided)
            blacklist: Token blacklist for revocation (in-memory by default)
            access_token_expire_minutes: Access token expiration
            refresh_token_expire_days: Refresh token expiration
        """
        self.key_manager = key_manager or JWTKeyManager()
        self.blacklist = blacklist or InMemoryTokenBlacklist()
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days

    def create_access_token(
        self,
        user_id: str,
        organization_id: Optional[str] = None,
        email: Optional[str] = None,
        role: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        Create an access token.

        Args:
            user_id: User ID (subject)
            organization_id: Organization ID
            email: User email
            role: User role
            scopes: Permission scopes
            expires_delta: Custom expiration time

        Returns:
            Encoded JWT token
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=self.access_token_expire_minutes)

        now = datetime.utcnow()
        exp = now + expires_delta

        payload = TokenPayload(
            sub=user_id,
            org=organization_id,
            email=email,
            role=role,
            scopes=scopes or [],
            iat=now,
            exp=exp,
            jti=secrets.token_urlsafe(16),
            type="access",
        )

        key = self.key_manager.primary_key

        token = jwt.encode(
            payload.model_dump(),
            key.secret,
            algorithm=key.algorithm,
            headers={"kid": key.kid},
        )

        return token

    def create_refresh_token(
        self,
        user_id: str,
        organization_id: Optional[str] = None,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        Create a refresh token.

        Args:
            user_id: User ID (subject)
            organization_id: Organization ID
            expires_delta: Custom expiration time

        Returns:
            Encoded JWT token
        """
        if expires_delta is None:
            expires_delta = timedelta(days=self.refresh_token_expire_days)

        now = datetime.utcnow()
        exp = now + expires_delta

        payload = TokenPayload(
            sub=user_id,
            org=organization_id,
            iat=now,
            exp=exp,
            jti=secrets.token_urlsafe(16),
            type="refresh",
        )

        key = self.key_manager.primary_key

        token = jwt.encode(
            payload.model_dump(),
            key.secret,
            algorithm=key.algorithm,
            headers={"kid": key.kid},
        )

        return token

    def create_tokens(
        self,
        user_id: str,
        organization_id: Optional[str] = None,
        email: Optional[str] = None,
        role: Optional[str] = None,
        scopes: Optional[List[str]] = None,
    ) -> TokenPair:
        """
        Create an access/refresh token pair.

        Args:
            user_id: User ID
            organization_id: Organization ID
            email: User email
            role: User role
            scopes: Permission scopes

        Returns:
            TokenPair with access and refresh tokens
        """
        access_token = self.create_access_token(
            user_id=user_id,
            organization_id=organization_id,
            email=email,
            role=role,
            scopes=scopes,
        )

        refresh_token = self.create_refresh_token(
            user_id=user_id,
            organization_id=organization_id,
        )

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_expire_minutes * 60,
        )

    def verify_token(
        self,
        token: str,
        token_type: str = "access",
        verify_exp: bool = True,
        check_blacklist: bool = True,
    ) -> Optional[TokenPayload]:
        """
        Verify and decode a token.

        Args:
            token: The JWT token to verify
            token_type: Expected token type (access or refresh)
            verify_exp: Verify expiration
            check_blacklist: Check if token is blacklisted (revoked)

        Returns:
            TokenPayload if valid, None otherwise
        """
        try:
            # Get the key ID from header
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")

            # Find the key
            if kid:
                key = self.key_manager.get_key(kid)
                if not key or not key.is_active:
                    return None
            else:
                # Fall back to primary key
                key = self.key_manager.primary_key

            # Decode and verify
            payload_dict = jwt.decode(
                token,
                key.secret,
                algorithms=[key.algorithm],
                options={"verify_exp": verify_exp},
            )

            payload = TokenPayload(**payload_dict)

            # Verify token type
            if payload.type != token_type:
                return None

            # Check blacklist (synchronous check for in-memory)
            if check_blacklist and payload.jti:
                if isinstance(self.blacklist, InMemoryTokenBlacklist):
                    if self.blacklist.sync_contains(payload.jti):
                        return None

            return payload

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception:
            return None

    async def verify_token_async(
        self,
        token: str,
        token_type: str = "access",
        verify_exp: bool = True,
        check_blacklist: bool = True,
    ) -> Optional[TokenPayload]:
        """
        Async version of verify_token that properly checks Redis blacklist.

        Args:
            token: The JWT token to verify
            token_type: Expected token type (access or refresh)
            verify_exp: Verify expiration
            check_blacklist: Check if token is blacklisted

        Returns:
            TokenPayload if valid, None otherwise
        """
        try:
            # Get the key ID from header
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")

            # Find the key
            if kid:
                key = self.key_manager.get_key(kid)
                if not key or not key.is_active:
                    return None
            else:
                key = self.key_manager.primary_key

            # Decode and verify
            payload_dict = jwt.decode(
                token,
                key.secret,
                algorithms=[key.algorithm],
                options={"verify_exp": verify_exp},
            )

            payload = TokenPayload(**payload_dict)

            # Verify token type
            if payload.type != token_type:
                return None

            # Check blacklist asynchronously
            if check_blacklist and payload.jti:
                if await self.blacklist.contains(payload.jti):
                    return None

            return payload

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception:
            return None

    def refresh_tokens(
        self,
        refresh_token: str,
    ) -> Optional[TokenPair]:
        """
        Refresh tokens using a refresh token.

        Args:
            refresh_token: The refresh token

        Returns:
            New TokenPair if refresh token is valid, None otherwise
        """
        # Verify refresh token
        payload = self.verify_token(refresh_token, token_type="refresh")
        if not payload:
            return None

        # Create new tokens
        return self.create_tokens(
            user_id=payload.sub,
            organization_id=payload.org,
        )

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token by adding it to the blacklist.

        Args:
            token: The JWT token to revoke

        Returns:
            True if revoked successfully, False otherwise
        """
        try:
            # Decode token without verifying expiration (we want to blacklist even expired tokens)
            payload = self.verify_token(token, verify_exp=False, check_blacklist=False)
            if not payload or not payload.jti:
                return False

            # Add to blacklist synchronously for in-memory
            if isinstance(self.blacklist, InMemoryTokenBlacklist):
                return self.blacklist.sync_add(payload.jti, payload.exp)

            # For other implementations, use async in sync context
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context - this shouldn't happen in sync call
                    return False
                return loop.run_until_complete(
                    self.blacklist.add(payload.jti, payload.exp)
                )
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(self.blacklist.add(payload.jti, payload.exp))

        except Exception:
            return False

    async def revoke_token_async(self, token: str) -> bool:
        """
        Async version of revoke_token.

        Args:
            token: The JWT token to revoke

        Returns:
            True if revoked successfully, False otherwise
        """
        try:
            # Decode token without verifying expiration
            payload = self.verify_token(token, verify_exp=False, check_blacklist=False)
            if not payload or not payload.jti:
                return False

            return await self.blacklist.add(payload.jti, payload.exp)
        except Exception:
            return False

    async def revoke_all_user_tokens(self, user_id: str, tokens: List[str]) -> int:
        """
        Revoke all tokens for a user (e.g., on password change or logout-all).

        Args:
            user_id: The user ID
            tokens: List of tokens to revoke

        Returns:
            Number of tokens successfully revoked
        """
        revoked = 0
        for token in tokens:
            if await self.revoke_token_async(token):
                revoked += 1
        return revoked


# =============================================================================
# Singleton Instances
# =============================================================================


_key_manager: Optional[JWTKeyManager] = None
_token_blacklist: Optional[TokenBlacklist] = None
_jwt_service: Optional[JWTService] = None


def get_key_manager() -> JWTKeyManager:
    """Get the JWT key manager singleton."""
    global _key_manager
    if _key_manager is None:
        _key_manager = JWTKeyManager()
    return _key_manager


def get_token_blacklist() -> TokenBlacklist:
    """
    Get the token blacklist singleton.

    Uses Redis if REDIS_URL is set, otherwise uses in-memory.
    """
    global _token_blacklist
    if _token_blacklist is None:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            _token_blacklist = RedisTokenBlacklist(redis_url=redis_url)
        else:
            _token_blacklist = InMemoryTokenBlacklist()
    return _token_blacklist


def get_jwt_service() -> JWTService:
    """Get the JWT service singleton."""
    global _jwt_service
    if _jwt_service is None:
        _jwt_service = JWTService(
            key_manager=get_key_manager(),
            blacklist=get_token_blacklist(),
        )
    return _jwt_service


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "JWTKey",
    "TokenPayload",
    "TokenPair",
    "JWTKeyManager",
    "TokenBlacklist",
    "InMemoryTokenBlacklist",
    "RedisTokenBlacklist",
    "JWTService",
    "get_key_manager",
    "get_jwt_service",
    "get_token_blacklist",
]
