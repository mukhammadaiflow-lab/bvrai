"""
JWT Token Management with Key Rotation

This module provides secure JWT token generation, validation,
and key rotation support for the platform.
"""

import json
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

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
        access_token_expire_minutes: int = DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES,
        refresh_token_expire_days: int = DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS,
    ):
        """
        Initialize the JWT service.

        Args:
            key_manager: JWT key manager (created if not provided)
            access_token_expire_minutes: Access token expiration
            refresh_token_expire_days: Refresh token expiration
        """
        self.key_manager = key_manager or JWTKeyManager()
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
    ) -> Optional[TokenPayload]:
        """
        Verify and decode a token.

        Args:
            token: The JWT token to verify
            token_type: Expected token type (access or refresh)
            verify_exp: Verify expiration

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

    def revoke_token(self, jti: str) -> bool:
        """
        Revoke a token by its JTI.

        Note: This requires a token blacklist implementation.
        The blacklist should be stored in Redis or database.

        Args:
            jti: The token's JTI claim

        Returns:
            True if revoked successfully
        """
        # TODO: Implement token blacklist
        # This should add the JTI to a blacklist in Redis/DB
        # The blacklist should be checked in verify_token()
        pass


# =============================================================================
# Singleton Instances
# =============================================================================


_key_manager: Optional[JWTKeyManager] = None
_jwt_service: Optional[JWTService] = None


def get_key_manager() -> JWTKeyManager:
    """Get the JWT key manager singleton."""
    global _key_manager
    if _key_manager is None:
        _key_manager = JWTKeyManager()
    return _key_manager


def get_jwt_service() -> JWTService:
    """Get the JWT service singleton."""
    global _jwt_service
    if _jwt_service is None:
        _jwt_service = JWTService(key_manager=get_key_manager())
    return _jwt_service


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "JWTKey",
    "TokenPayload",
    "TokenPair",
    "JWTKeyManager",
    "JWTService",
    "get_key_manager",
    "get_jwt_service",
]
