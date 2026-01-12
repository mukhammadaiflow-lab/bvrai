"""Auth module."""

from app.auth.dependencies import (
    get_current_user,
    get_current_user_id,
    get_optional_user_id,
    require_scopes,
)

from app.auth.jwt import (
    TokenType,
    TokenClaims,
    JWTError,
    TokenExpiredError,
    InvalidTokenError,
    JWTManager,
    APIKeyGenerator,
    get_jwt_manager,
    get_api_key_generator,
)

from app.auth.middleware import (
    AuthenticatedUser,
    AuthenticationMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    CORSMiddleware,
    get_current_user as get_current_user_from_request,
    require_auth,
    require_scopes as require_middleware_scopes,
)

from app.auth.rbac import (
    Permission,
    Role,
    RoleDefinition,
    ROLE_PERMISSIONS,
    RBACManager,
    get_rbac_manager,
    ResourcePolicy,
    PolicyEnforcer,
    require_permission,
    check_permission,
    scopes_to_permissions,
    permissions_to_scopes,
)

__all__ = [
    # Dependencies
    "get_current_user",
    "get_current_user_id",
    "get_optional_user_id",
    "require_scopes",
    # JWT
    "TokenType",
    "TokenClaims",
    "JWTError",
    "TokenExpiredError",
    "InvalidTokenError",
    "JWTManager",
    "APIKeyGenerator",
    "get_jwt_manager",
    "get_api_key_generator",
    # Middleware
    "AuthenticatedUser",
    "AuthenticationMiddleware",
    "RateLimitMiddleware",
    "RequestLoggingMiddleware",
    "CORSMiddleware",
    "get_current_user_from_request",
    "require_auth",
    "require_middleware_scopes",
    # RBAC
    "Permission",
    "Role",
    "RoleDefinition",
    "ROLE_PERMISSIONS",
    "RBACManager",
    "get_rbac_manager",
    "ResourcePolicy",
    "PolicyEnforcer",
    "require_permission",
    "check_permission",
    "scopes_to_permissions",
    "permissions_to_scopes",
]
