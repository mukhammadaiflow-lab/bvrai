"""
API Dependencies

Common dependencies for FastAPI routes including:
- Authentication and authorization
- Database sessions
- Rate limiting
- Caching
"""

from typing import Optional, List, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
import time

from fastapi import Depends, HTTPException, status, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
import jwt
import redis.asyncio as redis
import structlog

logger = structlog.get_logger()


# ============================================================================
# Security Schemes
# ============================================================================

bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ============================================================================
# User Context
# ============================================================================

@dataclass
class UserContext:
    """Current user context."""
    user_id: str
    tenant_id: str
    email: str
    roles: List[str]
    permissions: List[str]
    api_key_id: Optional[str] = None
    is_service_account: bool = False
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        if "admin" in self.roles:
            return True
        return permission in self.permissions

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles


@dataclass
class TenantContext:
    """Current tenant context."""
    tenant_id: str
    name: str
    plan: str
    features: List[str]
    limits: dict
    settings: dict
    created_at: datetime

    def has_feature(self, feature: str) -> bool:
        """Check if tenant has access to a feature."""
        return feature in self.features

    def get_limit(self, limit_name: str, default: int = 0) -> int:
        """Get a specific limit value."""
        return self.limits.get(limit_name, default)


# ============================================================================
# Database Dependency
# ============================================================================

async def get_db_session() -> AsyncSession:
    """
    Get database session dependency.

    Usage:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    from app.database.session import async_session_maker

    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ============================================================================
# Redis Dependency
# ============================================================================

_redis_pool: Optional[redis.Redis] = None


async def get_redis() -> redis.Redis:
    """
    Get Redis connection dependency.

    Usage:
        @router.get("/cached")
        async def get_cached(cache: redis.Redis = Depends(get_redis)):
            ...
    """
    global _redis_pool

    if _redis_pool is None:
        from app.config import get_settings
        settings = get_settings()
        _redis_pool = redis.Redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    return _redis_pool


# ============================================================================
# Authentication
# ============================================================================

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    db: AsyncSession = Depends(get_db_session),
) -> UserContext:
    """
    Get current authenticated user.

    Supports:
    - JWT Bearer token
    - API Key

    Usage:
        @router.get("/me")
        async def get_me(user: UserContext = Depends(get_current_user)):
            return {"user_id": user.user_id}
    """
    from app.config import get_settings
    settings = get_settings()

    # Try JWT token first
    if credentials:
        try:
            payload = jwt.decode(
                credentials.credentials,
                settings.jwt_secret,
                algorithms=["HS256"],
            )

            return UserContext(
                user_id=payload["sub"],
                tenant_id=payload.get("tenant_id", ""),
                email=payload.get("email", ""),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                metadata=payload.get("metadata", {}),
            )

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Try API key
    if api_key:
        from app.auth.service import AuthService
        auth_service = AuthService(db)

        key_data = await auth_service.validate_api_key(api_key)
        if key_data:
            return UserContext(
                user_id=key_data["user_id"],
                tenant_id=key_data["tenant_id"],
                email=key_data.get("email", ""),
                roles=key_data.get("roles", ["api"]),
                permissions=key_data.get("permissions", []),
                api_key_id=key_data["key_id"],
                is_service_account=key_data.get("is_service_account", False),
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    db: AsyncSession = Depends(get_db_session),
) -> Optional[UserContext]:
    """Get current user if authenticated, None otherwise."""
    try:
        return await get_current_user(request, credentials, api_key, db)
    except HTTPException:
        return None


# ============================================================================
# Tenant Context
# ============================================================================

async def get_current_tenant(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> TenantContext:
    """
    Get current tenant context.

    Usage:
        @router.get("/tenant")
        async def get_tenant_info(tenant: TenantContext = Depends(get_current_tenant)):
            return {"tenant_id": tenant.tenant_id, "plan": tenant.plan}
    """
    from app.tenancy.service import TenantService
    tenant_service = TenantService(db)

    tenant = await tenant_service.get(user.tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    return TenantContext(
        tenant_id=tenant.id,
        name=tenant.name,
        plan=tenant.plan,
        features=tenant.features or [],
        limits=tenant.limits or {},
        settings=tenant.settings or {},
        created_at=tenant.created_at,
    )


# ============================================================================
# Permission Checking
# ============================================================================

def require_permissions(*permissions: str):
    """
    Dependency to require specific permissions.

    Usage:
        @router.delete("/agents/{id}")
        async def delete_agent(
            user: UserContext = Depends(require_permissions("agents:delete"))
        ):
            ...
    """
    async def check_permissions(
        user: UserContext = Depends(get_current_user),
    ) -> UserContext:
        for permission in permissions:
            if not user.has_permission(permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing permission: {permission}",
                )
        return user

    return check_permissions


def require_roles(*roles: str):
    """
    Dependency to require specific roles.

    Usage:
        @router.get("/admin/users")
        async def list_users(
            user: UserContext = Depends(require_roles("admin"))
        ):
            ...
    """
    async def check_roles(
        user: UserContext = Depends(get_current_user),
    ) -> UserContext:
        for role in roles:
            if not user.has_role(role):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing role: {role}",
                )
        return user

    return check_roles


def require_feature(feature: str):
    """
    Dependency to require a tenant feature.

    Usage:
        @router.post("/workflows")
        async def create_workflow(
            tenant: TenantContext = Depends(require_feature("workflows"))
        ):
            ...
    """
    async def check_feature(
        tenant: TenantContext = Depends(get_current_tenant),
    ) -> TenantContext:
        if not tenant.has_feature(feature):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Feature not available: {feature}. Please upgrade your plan.",
            )
        return tenant

    return check_feature


# ============================================================================
# Rate Limiting
# ============================================================================

@dataclass
class RateLimitResult:
    """Rate limit check result."""
    allowed: bool
    limit: int
    remaining: int
    reset_at: int
    retry_after: Optional[int] = None


class RateLimitDep:
    """
    Rate limiting dependency.

    Usage:
        rate_limit = RateLimitDep(requests=100, window=60)

        @router.get("/data")
        async def get_data(
            _: None = Depends(rate_limit),
            user: UserContext = Depends(get_current_user),
        ):
            ...
    """

    def __init__(
        self,
        requests: int = 100,
        window: int = 60,
        key_func: Optional[Callable] = None,
        scope: str = "default",
    ):
        self.requests = requests
        self.window = window
        self.key_func = key_func
        self.scope = scope

    async def __call__(
        self,
        request: Request,
        user: UserContext = Depends(get_current_user),
        cache: redis.Redis = Depends(get_redis),
    ) -> RateLimitResult:
        # Build rate limit key
        if self.key_func:
            key_suffix = self.key_func(request, user)
        else:
            key_suffix = f"{user.tenant_id}:{user.user_id}"

        key = f"ratelimit:{self.scope}:{key_suffix}"
        now = int(time.time())
        window_start = now - self.window

        # Use sliding window with Redis sorted set
        pipe = cache.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, self.window)
        results = await pipe.execute()

        request_count = results[2]
        remaining = max(0, self.requests - request_count)
        reset_at = now + self.window

        # Add rate limit headers to response
        request.state.rate_limit_limit = self.requests
        request.state.rate_limit_remaining = remaining
        request.state.rate_limit_reset = reset_at

        if request_count > self.requests:
            retry_after = self.window
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(self.requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "Retry-After": str(retry_after),
                },
            )

        return RateLimitResult(
            allowed=True,
            limit=self.requests,
            remaining=remaining,
            reset_at=reset_at,
        )


# ============================================================================
# Pagination
# ============================================================================

@dataclass
class PaginationParams:
    """Pagination parameters."""
    page: int = 1
    page_size: int = 20

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        return self.page_size


def get_pagination(
    page: int = 1,
    page_size: int = 20,
) -> PaginationParams:
    """
    Get pagination parameters dependency.

    Usage:
        @router.get("/items")
        async def list_items(pagination: PaginationParams = Depends(get_pagination)):
            items = await service.list(
                offset=pagination.offset,
                limit=pagination.limit,
            )
    """
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 1
    if page_size > 100:
        page_size = 100

    return PaginationParams(page=page, page_size=page_size)


# ============================================================================
# Request Context
# ============================================================================

async def get_request_id(
    request: Request,
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
) -> str:
    """Get or generate request ID."""
    import uuid
    request_id = x_request_id or str(uuid.uuid4())
    request.state.request_id = request_id
    return request_id


async def get_client_ip(request: Request) -> str:
    """Get client IP address."""
    # Check for forwarded headers
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    return request.client.host if request.client else "unknown"


# ============================================================================
# Webhook Verification
# ============================================================================

async def verify_webhook_signature(
    request: Request,
    x_signature: str = Header(..., alias="X-Webhook-Signature"),
) -> bool:
    """
    Verify webhook signature.

    Usage:
        @router.post("/webhooks/incoming")
        async def handle_webhook(
            verified: bool = Depends(verify_webhook_signature),
        ):
            ...
    """
    import hmac
    import hashlib
    from app.config import get_settings

    settings = get_settings()
    body = await request.body()

    expected = hmac.new(
        settings.webhook_secret.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(f"sha256={expected}", x_signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        )

    return True


# ============================================================================
# Feature Flags
# ============================================================================

class FeatureFlag:
    """
    Feature flag dependency.

    Usage:
        beta_feature = FeatureFlag("beta_workflows")

        @router.post("/workflows/beta")
        async def create_beta_workflow(
            enabled: bool = Depends(beta_feature),
        ):
            if not enabled:
                raise HTTPException(status_code=404)
    """

    def __init__(self, flag_name: str):
        self.flag_name = flag_name

    async def __call__(
        self,
        user: UserContext = Depends(get_current_user),
        tenant: TenantContext = Depends(get_current_tenant),
        cache: redis.Redis = Depends(get_redis),
    ) -> bool:
        # Check tenant features first
        if self.flag_name in tenant.features:
            return True

        # Check user-specific flags
        user_flag = await cache.get(f"feature:{self.flag_name}:user:{user.user_id}")
        if user_flag == "1":
            return True

        # Check tenant-specific flags
        tenant_flag = await cache.get(f"feature:{self.flag_name}:tenant:{tenant.tenant_id}")
        if tenant_flag == "1":
            return True

        # Check global flag
        global_flag = await cache.get(f"feature:{self.flag_name}:global")
        return global_flag == "1"
