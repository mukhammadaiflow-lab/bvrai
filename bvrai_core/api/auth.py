"""
API Authentication Module

This module provides authentication mechanisms for the REST API,
including API key validation, JWT handling, and permission checking.
"""

import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import jwt
from pydantic import BaseModel, Field

from .base import (
    APIException,
    AuthenticationError,
    AuthorizationError,
    ErrorCode,
    hash_api_key,
)


logger = logging.getLogger(__name__)


class AuthMethod(str, Enum):
    """Authentication methods."""

    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    WEBHOOK_SIGNATURE = "webhook_signature"


class Permission(str, Enum):
    """API permissions."""

    # Agent permissions
    AGENTS_READ = "agents:read"
    AGENTS_WRITE = "agents:write"
    AGENTS_DELETE = "agents:delete"
    AGENTS_EXECUTE = "agents:execute"

    # Call permissions
    CALLS_READ = "calls:read"
    CALLS_WRITE = "calls:write"
    CALLS_DELETE = "calls:delete"

    # Campaign permissions
    CAMPAIGNS_READ = "campaigns:read"
    CAMPAIGNS_WRITE = "campaigns:write"
    CAMPAIGNS_DELETE = "campaigns:delete"
    CAMPAIGNS_EXECUTE = "campaigns:execute"

    # Knowledge base permissions
    KNOWLEDGE_READ = "knowledge:read"
    KNOWLEDGE_WRITE = "knowledge:write"
    KNOWLEDGE_DELETE = "knowledge:delete"

    # Phone number permissions
    PHONE_NUMBERS_READ = "phone_numbers:read"
    PHONE_NUMBERS_WRITE = "phone_numbers:write"
    PHONE_NUMBERS_DELETE = "phone_numbers:delete"

    # Webhook permissions
    WEBHOOKS_READ = "webhooks:read"
    WEBHOOKS_WRITE = "webhooks:write"
    WEBHOOKS_DELETE = "webhooks:delete"

    # Recording/Transcript permissions
    RECORDINGS_READ = "recordings:read"
    RECORDINGS_DELETE = "recordings:delete"
    TRANSCRIPTS_READ = "transcripts:read"

    # Analytics permissions
    ANALYTICS_READ = "analytics:read"

    # Admin permissions
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    USERS_MANAGE = "users:manage"
    API_KEYS_MANAGE = "api_keys:manage"
    BILLING_READ = "billing:read"
    BILLING_WRITE = "billing:write"


class Role(str, Enum):
    """Predefined roles with permission sets."""

    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"
    OPERATOR = "operator"
    ANALYST = "analyst"
    VIEWER = "viewer"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.OWNER: set(Permission),  # All permissions
    Role.ADMIN: {
        Permission.AGENTS_READ,
        Permission.AGENTS_WRITE,
        Permission.AGENTS_DELETE,
        Permission.AGENTS_EXECUTE,
        Permission.CALLS_READ,
        Permission.CALLS_WRITE,
        Permission.CALLS_DELETE,
        Permission.CAMPAIGNS_READ,
        Permission.CAMPAIGNS_WRITE,
        Permission.CAMPAIGNS_DELETE,
        Permission.CAMPAIGNS_EXECUTE,
        Permission.KNOWLEDGE_READ,
        Permission.KNOWLEDGE_WRITE,
        Permission.KNOWLEDGE_DELETE,
        Permission.PHONE_NUMBERS_READ,
        Permission.PHONE_NUMBERS_WRITE,
        Permission.PHONE_NUMBERS_DELETE,
        Permission.WEBHOOKS_READ,
        Permission.WEBHOOKS_WRITE,
        Permission.WEBHOOKS_DELETE,
        Permission.RECORDINGS_READ,
        Permission.RECORDINGS_DELETE,
        Permission.TRANSCRIPTS_READ,
        Permission.ANALYTICS_READ,
        Permission.USERS_MANAGE,
        Permission.API_KEYS_MANAGE,
    },
    Role.DEVELOPER: {
        Permission.AGENTS_READ,
        Permission.AGENTS_WRITE,
        Permission.AGENTS_EXECUTE,
        Permission.CALLS_READ,
        Permission.KNOWLEDGE_READ,
        Permission.KNOWLEDGE_WRITE,
        Permission.WEBHOOKS_READ,
        Permission.WEBHOOKS_WRITE,
        Permission.RECORDINGS_READ,
        Permission.TRANSCRIPTS_READ,
        Permission.ANALYTICS_READ,
    },
    Role.OPERATOR: {
        Permission.AGENTS_READ,
        Permission.AGENTS_EXECUTE,
        Permission.CALLS_READ,
        Permission.CALLS_WRITE,
        Permission.CAMPAIGNS_READ,
        Permission.CAMPAIGNS_WRITE,
        Permission.CAMPAIGNS_EXECUTE,
        Permission.PHONE_NUMBERS_READ,
        Permission.RECORDINGS_READ,
        Permission.TRANSCRIPTS_READ,
        Permission.ANALYTICS_READ,
    },
    Role.ANALYST: {
        Permission.AGENTS_READ,
        Permission.CALLS_READ,
        Permission.CAMPAIGNS_READ,
        Permission.RECORDINGS_READ,
        Permission.TRANSCRIPTS_READ,
        Permission.ANALYTICS_READ,
    },
    Role.VIEWER: {
        Permission.AGENTS_READ,
        Permission.CALLS_READ,
        Permission.CAMPAIGNS_READ,
        Permission.ANALYTICS_READ,
    },
}


@dataclass
class APIKeyInfo:
    """Information about an API key."""

    key_id: str
    key_hash: str
    organization_id: str
    user_id: Optional[str] = None
    name: str = ""
    permissions: Set[Permission] = field(default_factory=set)
    role: Optional[Role] = None

    # Rate limits
    rate_limit_per_minute: int = 60
    rate_limit_per_day: int = 10000

    # Status
    is_active: bool = True
    expires_at: Optional[datetime] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    allowed_ips: List[str] = field(default_factory=list)

    def has_permission(self, permission: Permission) -> bool:
        """Check if key has a specific permission."""
        # If role is set, use role permissions
        if self.role:
            role_perms = ROLE_PERMISSIONS.get(self.role, set())
            if permission in role_perms:
                return True

        return permission in self.permissions

    def is_valid(self) -> bool:
        """Check if key is valid."""
        if not self.is_active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


@dataclass
class JWTPayload:
    """JWT token payload."""

    sub: str  # Subject (user ID)
    org: str  # Organization ID
    role: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    exp: Optional[datetime] = None
    iat: datetime = field(default_factory=datetime.utcnow)
    jti: str = field(default_factory=lambda: secrets.token_hex(16))

    # Additional claims
    email: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AuthContext:
    """Authentication context for a request."""

    method: AuthMethod
    is_authenticated: bool = False

    # Identity
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    api_key_id: Optional[str] = None

    # Permissions
    role: Optional[Role] = None
    permissions: Set[Permission] = field(default_factory=set)

    # Metadata
    authenticated_at: datetime = field(default_factory=datetime.utcnow)
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None

    def has_permission(self, permission: Permission) -> bool:
        """Check if context has a specific permission."""
        # Owner role has all permissions
        if self.role == Role.OWNER:
            return True

        # Check role permissions
        if self.role:
            role_perms = ROLE_PERMISSIONS.get(self.role, set())
            if permission in role_perms:
                return True

        return permission in self.permissions

    def require_permission(self, permission: Permission) -> None:
        """Require a specific permission, raise if not present."""
        if not self.has_permission(permission):
            raise AuthorizationError(
                message=f"Permission '{permission.value}' required",
                details={"required_permission": permission.value},
            )

    def require_any_permission(self, permissions: List[Permission]) -> None:
        """Require any of the specified permissions."""
        for perm in permissions:
            if self.has_permission(perm):
                return

        raise AuthorizationError(
            message="Insufficient permissions",
            details={"required_any": [p.value for p in permissions]},
        )

    def require_all_permissions(self, permissions: List[Permission]) -> None:
        """Require all of the specified permissions."""
        missing = [p for p in permissions if not self.has_permission(p)]
        if missing:
            raise AuthorizationError(
                message="Insufficient permissions",
                details={"missing_permissions": [p.value for p in missing]},
            )


class JWTConfig(BaseModel):
    """JWT configuration."""

    secret_key: str = Field(..., description="Secret key for signing")
    algorithm: str = Field(default="HS256", description="Signing algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration in minutes",
    )
    refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiration in days",
    )
    issuer: str = Field(default="bvrai", description="Token issuer")
    audience: str = Field(default="bvrai-api", description="Token audience")


class APIAuthenticator:
    """
    Handles API authentication.

    Supports:
    - API key authentication
    - JWT bearer tokens
    - Webhook signature verification
    """

    def __init__(
        self,
        jwt_config: Optional[JWTConfig] = None,
        api_key_lookup: Optional[Callable[[str], Optional[APIKeyInfo]]] = None,
    ):
        """
        Initialize authenticator.

        Args:
            jwt_config: JWT configuration
            api_key_lookup: Function to look up API key info
        """
        self.jwt_config = jwt_config
        self._api_key_lookup = api_key_lookup
        self._api_key_cache: Dict[str, Tuple[APIKeyInfo, datetime]] = {}
        # SECURITY: Reduced from 5 minutes to 30 seconds to minimize window
        # for revoked keys to remain valid. For high-security environments,
        # consider using event-based invalidation instead of TTL.
        self._cache_ttl = timedelta(seconds=30)

    def authenticate_api_key(
        self,
        api_key: str,
        client_ip: Optional[str] = None,
    ) -> AuthContext:
        """
        Authenticate using API key.

        Args:
            api_key: API key string
            client_ip: Client IP address

        Returns:
            Authentication context
        """
        if not api_key:
            raise AuthenticationError(
                message="API key required",
                code=ErrorCode.AUTHENTICATION_REQUIRED,
            )

        # Validate format
        if not api_key.startswith("bvr_"):
            raise AuthenticationError(
                message="Invalid API key format",
                code=ErrorCode.INVALID_API_KEY,
            )

        # Look up key info
        key_hash = hash_api_key(api_key)
        key_info = self._get_api_key_info(key_hash)

        if not key_info:
            raise AuthenticationError(
                message="Invalid API key",
                code=ErrorCode.INVALID_API_KEY,
            )

        # Validate key
        if not key_info.is_valid():
            if not key_info.is_active:
                raise AuthenticationError(
                    message="API key has been revoked",
                    code=ErrorCode.INVALID_API_KEY,
                )
            if key_info.expires_at and datetime.utcnow() > key_info.expires_at:
                raise AuthenticationError(
                    message="API key has expired",
                    code=ErrorCode.EXPIRED_TOKEN,
                )

        # Check IP restrictions
        if key_info.allowed_ips and client_ip:
            if client_ip not in key_info.allowed_ips:
                raise AuthenticationError(
                    message="Request from unauthorized IP",
                    code=ErrorCode.AUTHENTICATION_REQUIRED,
                    details={"client_ip": client_ip},
                )

        # Build permissions set
        permissions = set(key_info.permissions)
        if key_info.role:
            permissions.update(ROLE_PERMISSIONS.get(key_info.role, set()))

        return AuthContext(
            method=AuthMethod.API_KEY,
            is_authenticated=True,
            user_id=key_info.user_id,
            organization_id=key_info.organization_id,
            api_key_id=key_info.key_id,
            role=key_info.role,
            permissions=permissions,
            client_ip=client_ip,
        )

    def authenticate_jwt(
        self,
        token: str,
        client_ip: Optional[str] = None,
    ) -> AuthContext:
        """
        Authenticate using JWT token.

        Args:
            token: JWT token string
            client_ip: Client IP address

        Returns:
            Authentication context
        """
        if not self.jwt_config:
            raise AuthenticationError(
                message="JWT authentication not configured",
                code=ErrorCode.AUTHENTICATION_REQUIRED,
            )

        try:
            payload = jwt.decode(
                token,
                self.jwt_config.secret_key,
                algorithms=[self.jwt_config.algorithm],
                audience=self.jwt_config.audience,
                issuer=self.jwt_config.issuer,
            )
        except jwt.ExpiredSignatureError:
            raise AuthenticationError(
                message="Token has expired",
                code=ErrorCode.EXPIRED_TOKEN,
            )
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(
                message=f"Invalid token: {str(e)}",
                code=ErrorCode.INVALID_API_KEY,
            )

        # Extract permissions
        permissions = set()
        for perm_str in payload.get("permissions", []):
            try:
                permissions.add(Permission(perm_str))
            except ValueError:
                pass

        # Extract role
        role = None
        role_str = payload.get("role")
        if role_str:
            try:
                role = Role(role_str)
                permissions.update(ROLE_PERMISSIONS.get(role, set()))
            except ValueError:
                pass

        return AuthContext(
            method=AuthMethod.JWT,
            is_authenticated=True,
            user_id=payload.get("sub"),
            organization_id=payload.get("org"),
            role=role,
            permissions=permissions,
            client_ip=client_ip,
        )

    def create_jwt(
        self,
        user_id: str,
        organization_id: str,
        role: Optional[Role] = None,
        permissions: Optional[List[Permission]] = None,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a JWT token.

        Args:
            user_id: User ID
            organization_id: Organization ID
            role: User role
            permissions: Explicit permissions
            additional_claims: Additional JWT claims

        Returns:
            JWT token string
        """
        if not self.jwt_config:
            raise ValueError("JWT configuration required")

        now = datetime.utcnow()
        exp = now + timedelta(minutes=self.jwt_config.access_token_expire_minutes)

        payload = {
            "sub": user_id,
            "org": organization_id,
            "iat": now,
            "exp": exp,
            "jti": secrets.token_hex(16),
            "iss": self.jwt_config.issuer,
            "aud": self.jwt_config.audience,
        }

        if role:
            payload["role"] = role.value

        if permissions:
            payload["permissions"] = [p.value for p in permissions]

        if additional_claims:
            payload.update(additional_claims)

        return jwt.encode(
            payload,
            self.jwt_config.secret_key,
            algorithm=self.jwt_config.algorithm,
        )

    def create_refresh_token(
        self,
        user_id: str,
        organization_id: str,
    ) -> str:
        """Create a refresh token."""
        if not self.jwt_config:
            raise ValueError("JWT configuration required")

        now = datetime.utcnow()
        exp = now + timedelta(days=self.jwt_config.refresh_token_expire_days)

        payload = {
            "sub": user_id,
            "org": organization_id,
            "type": "refresh",
            "iat": now,
            "exp": exp,
            "jti": secrets.token_hex(16),
            "iss": self.jwt_config.issuer,
            "aud": self.jwt_config.audience,
        }

        return jwt.encode(
            payload,
            self.jwt_config.secret_key,
            algorithm=self.jwt_config.algorithm,
        )

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
        secret: str,
        timestamp: Optional[int] = None,
        tolerance_seconds: int = 300,
    ) -> bool:
        """
        Verify webhook signature.

        Args:
            payload: Request payload bytes
            signature: Signature header value
            secret: Webhook secret
            timestamp: Request timestamp
            tolerance_seconds: Allowed time tolerance

        Returns:
            True if signature is valid
        """
        # Check timestamp if provided
        if timestamp:
            current_time = int(time.time())
            if abs(current_time - timestamp) > tolerance_seconds:
                raise AuthenticationError(
                    message="Webhook timestamp outside tolerance",
                    code=ErrorCode.INVALID_SIGNATURE,
                )

        # Build signed payload
        if timestamp:
            signed_payload = f"{timestamp}.{payload.decode()}"
        else:
            signed_payload = payload.decode()

        # Compute expected signature
        expected_sig = hmac.new(
            secret.encode(),
            signed_payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        # Compare signatures
        if not hmac.compare_digest(signature, expected_sig):
            raise AuthenticationError(
                message="Invalid webhook signature",
                code=ErrorCode.INVALID_SIGNATURE,
            )

        return True

    def _get_api_key_info(self, key_hash: str) -> Optional[APIKeyInfo]:
        """Get API key info with caching."""
        # Check cache
        if key_hash in self._api_key_cache:
            info, cached_at = self._api_key_cache[key_hash]
            if datetime.utcnow() - cached_at < self._cache_ttl:
                return info

        # Look up key
        if self._api_key_lookup:
            info = self._api_key_lookup(key_hash)
            if info:
                self._api_key_cache[key_hash] = (info, datetime.utcnow())
            return info

        return None

    def clear_cache(self) -> None:
        """Clear the API key cache."""
        self._api_key_cache.clear()

    def invalidate_key(self, key_hash: str) -> bool:
        """
        Invalidate a specific API key from cache.

        Call this when an API key is revoked to immediately
        prevent further use.

        Args:
            key_hash: The hash of the API key to invalidate

        Returns:
            True if the key was in cache and removed
        """
        if key_hash in self._api_key_cache:
            del self._api_key_cache[key_hash]
            return True
        return False

    def invalidate_organization_keys(self, organization_id: str) -> int:
        """
        Invalidate all cached keys for an organization.

        Call this when organization access is revoked.

        Args:
            organization_id: The organization ID

        Returns:
            Number of keys invalidated
        """
        keys_to_remove = [
            key_hash
            for key_hash, (info, _) in self._api_key_cache.items()
            if info.organization_id == organization_id
        ]
        for key_hash in keys_to_remove:
            del self._api_key_cache[key_hash]
        return len(keys_to_remove)


def require_permission(permission: Permission) -> Callable:
    """
    Decorator to require a specific permission.

    Usage:
        @require_permission(Permission.AGENTS_READ)
        async def get_agents(auth: AuthContext):
            ...
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, auth: AuthContext, **kwargs):
            auth.require_permission(permission)
            return await func(*args, auth=auth, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def require_any_permission(*permissions: Permission) -> Callable:
    """Decorator to require any of the specified permissions."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, auth: AuthContext, **kwargs):
            auth.require_any_permission(list(permissions))
            return await func(*args, auth=auth, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def require_role(role: Role) -> Callable:
    """Decorator to require a specific role or higher."""
    role_hierarchy = [Role.VIEWER, Role.ANALYST, Role.OPERATOR, Role.DEVELOPER, Role.ADMIN, Role.OWNER]

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, auth: AuthContext, **kwargs):
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

            return await func(*args, auth=auth, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


__all__ = [
    # Enums
    "AuthMethod",
    "Permission",
    "Role",
    # Data classes
    "APIKeyInfo",
    "JWTPayload",
    "AuthContext",
    # Configuration
    "JWTConfig",
    # Classes
    "APIAuthenticator",
    # Constants
    "ROLE_PERMISSIONS",
    # Decorators
    "require_permission",
    "require_any_permission",
    "require_role",
]
