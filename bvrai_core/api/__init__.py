"""
REST API Module

This module provides a comprehensive REST API for the Builder Engine
voice agent platform.

Features:
- FastAPI-based REST endpoints
- API key and JWT authentication
- Rate limiting with multiple strategies
- Request logging and tracking
- Distributed tracing support
- OpenAPI documentation
"""

from .base import (
    # Enums
    APIVersion,
    SortOrder,
    ResourceType,
    ErrorCode,
    # Request models
    PaginationParams,
    DateRangeFilter,
    SearchParams,
    # Response models
    APIError,
    PaginationMeta,
    APIResponse,
    ListResponse,
    BatchOperationResult,
    # Resource models
    ResourceBase,
    ResourceReference,
    AuditInfo,
    # Exceptions
    APIException,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ValidationError,
    ConflictError,
    RateLimitError,
    ServiceError,
    # Utilities
    generate_api_key,
    hash_api_key,
    generate_request_id,
    success_response,
    error_response,
    paginated_response,
)

from .auth import (
    AuthMethod,
    Permission,
    Role,
    APIKeyInfo,
    JWTPayload,
    AuthContext,
    JWTConfig,
    APIAuthenticator,
    ROLE_PERMISSIONS,
    require_permission,
    require_any_permission,
    require_role,
)

from .app import (
    AppConfig,
    create_app,
    app,
    run_server,
)


__all__ = [
    # Base - Enums
    "APIVersion",
    "SortOrder",
    "ResourceType",
    "ErrorCode",
    # Base - Request models
    "PaginationParams",
    "DateRangeFilter",
    "SearchParams",
    # Base - Response models
    "APIError",
    "PaginationMeta",
    "APIResponse",
    "ListResponse",
    "BatchOperationResult",
    # Base - Resource models
    "ResourceBase",
    "ResourceReference",
    "AuditInfo",
    # Base - Exceptions
    "APIException",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "ValidationError",
    "ConflictError",
    "RateLimitError",
    "ServiceError",
    # Base - Utilities
    "generate_api_key",
    "hash_api_key",
    "generate_request_id",
    "success_response",
    "error_response",
    "paginated_response",
    # Auth
    "AuthMethod",
    "Permission",
    "Role",
    "APIKeyInfo",
    "JWTPayload",
    "AuthContext",
    "JWTConfig",
    "APIAuthenticator",
    "ROLE_PERMISSIONS",
    "require_permission",
    "require_any_permission",
    "require_role",
    # App
    "AppConfig",
    "create_app",
    "app",
    "run_server",
]
