"""
API Base Types and Models Module

This module provides core types, response models, error handling,
and common utilities for the REST API layer.
"""

import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, validator


# Generic type for response data
T = TypeVar("T")


class APIVersion(str, Enum):
    """API version identifiers."""

    V1 = "v1"
    V2 = "v2"


class SortOrder(str, Enum):
    """Sort order options."""

    ASC = "asc"
    DESC = "desc"


class ResourceType(str, Enum):
    """API resource types."""

    AGENT = "agent"
    CALL = "call"
    CAMPAIGN = "campaign"
    KNOWLEDGE_BASE = "knowledge_base"
    PHONE_NUMBER = "phone_number"
    WEBHOOK = "webhook"
    USER = "user"
    ORGANIZATION = "organization"
    API_KEY = "api_key"
    RECORDING = "recording"
    TRANSCRIPT = "transcript"


class ErrorCode(str, Enum):
    """Standardized error codes."""

    # Authentication errors (1xxx)
    AUTHENTICATION_REQUIRED = "AUTH_1001"
    INVALID_API_KEY = "AUTH_1002"
    EXPIRED_TOKEN = "AUTH_1003"
    INSUFFICIENT_PERMISSIONS = "AUTH_1004"
    ACCOUNT_SUSPENDED = "AUTH_1005"
    INVALID_SIGNATURE = "AUTH_1006"

    # Validation errors (2xxx)
    VALIDATION_ERROR = "VAL_2001"
    INVALID_REQUEST_BODY = "VAL_2002"
    MISSING_REQUIRED_FIELD = "VAL_2003"
    INVALID_FIELD_VALUE = "VAL_2004"
    INVALID_QUERY_PARAMETER = "VAL_2005"

    # Resource errors (3xxx)
    RESOURCE_NOT_FOUND = "RES_3001"
    RESOURCE_ALREADY_EXISTS = "RES_3002"
    RESOURCE_CONFLICT = "RES_3003"
    RESOURCE_LOCKED = "RES_3004"
    RESOURCE_LIMIT_EXCEEDED = "RES_3005"

    # Rate limiting (4xxx)
    RATE_LIMIT_EXCEEDED = "RATE_4001"
    QUOTA_EXCEEDED = "RATE_4002"
    CONCURRENT_LIMIT_EXCEEDED = "RATE_4003"

    # Server errors (5xxx)
    INTERNAL_ERROR = "SRV_5001"
    SERVICE_UNAVAILABLE = "SRV_5002"
    DEPENDENCY_FAILURE = "SRV_5003"
    TIMEOUT = "SRV_5004"

    # Business logic errors (6xxx)
    INVALID_STATE_TRANSITION = "BIZ_6001"
    OPERATION_NOT_ALLOWED = "BIZ_6002"
    INSUFFICIENT_BALANCE = "BIZ_6003"
    COMPLIANCE_VIOLATION = "BIZ_6004"


# =============================================================================
# Request Models
# =============================================================================


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of items per page",
    )
    sort_by: Optional[str] = Field(
        default=None,
        description="Field to sort by",
    )
    sort_order: SortOrder = Field(
        default=SortOrder.DESC,
        description="Sort order",
    )

    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Get limit for database queries."""
        return self.page_size


class DateRangeFilter(BaseModel):
    """Date range filter for queries."""

    start_date: Optional[datetime] = Field(
        default=None,
        description="Start date (inclusive)",
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="End date (inclusive)",
    )

    @validator("end_date")
    def validate_date_range(cls, v, values):
        if v and values.get("start_date") and v < values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class SearchParams(BaseModel):
    """Search parameters."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query",
    )
    fields: Optional[List[str]] = Field(
        default=None,
        description="Fields to search in",
    )
    exact_match: bool = Field(
        default=False,
        description="Require exact match",
    )


# =============================================================================
# Response Models
# =============================================================================


class APIError(BaseModel):
    """API error response model."""

    code: ErrorCode = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details",
    )
    field: Optional[str] = Field(
        default=None,
        description="Field that caused the error (for validation errors)",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp",
    )

    class Config:
        use_enum_values = True


class PaginationMeta(BaseModel):
    """Pagination metadata for list responses."""

    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there's a next page")
    has_previous: bool = Field(..., description="Whether there's a previous page")

    @classmethod
    def create(
        cls,
        page: int,
        page_size: int,
        total_items: int,
    ) -> "PaginationMeta":
        """Create pagination metadata."""
        total_pages = (total_items + page_size - 1) // page_size if page_size > 0 else 0
        return cls(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""

    success: bool = Field(default=True, description="Whether request succeeded")
    data: Optional[T] = Field(default=None, description="Response data")
    error: Optional[APIError] = Field(default=None, description="Error information")
    meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata",
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp",
    )

    class Config:
        arbitrary_types_allowed = True


class ListResponse(BaseModel, Generic[T]):
    """Paginated list response."""

    success: bool = Field(default=True)
    data: List[T] = Field(default_factory=list, description="List of items")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")
    error: Optional[APIError] = Field(default=None)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True


class BatchOperationResult(BaseModel):
    """Result of a batch operation."""

    total: int = Field(..., description="Total items in batch")
    successful: int = Field(..., description="Successfully processed items")
    failed: int = Field(..., description="Failed items")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Errors for failed items",
    )


# =============================================================================
# Resource Models
# =============================================================================


class ResourceBase(BaseModel):
    """Base model for all resources."""

    id: str = Field(..., description="Unique resource identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        orm_mode = True


class ResourceReference(BaseModel):
    """Reference to another resource."""

    id: str = Field(..., description="Resource ID")
    type: ResourceType = Field(..., description="Resource type")
    name: Optional[str] = Field(default=None, description="Resource name")


class AuditInfo(BaseModel):
    """Audit information for resources."""

    created_by: Optional[str] = Field(default=None, description="Creator user ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_by: Optional[str] = Field(default=None, description="Last updater user ID")
    updated_at: datetime = Field(..., description="Last update timestamp")
    version: int = Field(default=1, description="Resource version")


# =============================================================================
# Exception Classes
# =============================================================================


class APIException(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
        field: Optional[str] = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        self.field = field
        super().__init__(message)

    def to_error(self, request_id: Optional[str] = None) -> APIError:
        """Convert to APIError model."""
        return APIError(
            code=self.code,
            message=self.message,
            details=self.details,
            field=self.field,
            request_id=request_id,
        )


class AuthenticationError(APIException):
    """Authentication failure."""

    def __init__(
        self,
        message: str = "Authentication required",
        code: ErrorCode = ErrorCode.AUTHENTICATION_REQUIRED,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            code=code,
            message=message,
            status_code=401,
            details=details,
        )


class AuthorizationError(APIException):
    """Authorization failure."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            code=ErrorCode.INSUFFICIENT_PERMISSIONS,
            message=message,
            status_code=403,
            details=details,
        )


class NotFoundError(APIException):
    """Resource not found."""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
    ):
        super().__init__(
            code=ErrorCode.RESOURCE_NOT_FOUND,
            message=f"{resource_type} with ID '{resource_id}' not found",
            status_code=404,
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class ValidationError(APIException):
    """Validation failure."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            status_code=422,
            details=details,
            field=field,
        )


class ConflictError(APIException):
    """Resource conflict."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            code=ErrorCode.RESOURCE_CONFLICT,
            message=message,
            status_code=409,
            details=details,
        )


class RateLimitError(APIException):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
    ):
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after

        super().__init__(
            code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message=message,
            status_code=429,
            details=details if details else None,
        )


class ServiceError(APIException):
    """Internal service error."""

    def __init__(
        self,
        message: str = "Internal server error",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            code=ErrorCode.INTERNAL_ERROR,
            message=message,
            status_code=500,
            details=details,
        )


# =============================================================================
# Utility Functions
# =============================================================================


def generate_api_key() -> str:
    """Generate a secure API key."""
    prefix = "bvr"
    key_bytes = secrets.token_bytes(32)
    key_hex = key_bytes.hex()
    return f"{prefix}_{key_hex}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_request_id() -> str:
    """Generate a unique request ID."""
    timestamp = int(time.time() * 1000)
    random_part = secrets.token_hex(8)
    return f"req_{timestamp}_{random_part}"


def success_response(
    data: Any,
    meta: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a success response."""
    return {
        "success": True,
        "data": data,
        "error": None,
        "meta": meta,
        "request_id": request_id or generate_request_id(),
        "timestamp": datetime.utcnow().isoformat(),
    }


def error_response(
    error: APIException,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an error response."""
    return {
        "success": False,
        "data": None,
        "error": error.to_error(request_id).dict(),
        "meta": None,
        "request_id": request_id or generate_request_id(),
        "timestamp": datetime.utcnow().isoformat(),
    }


def paginated_response(
    items: List[Any],
    page: int,
    page_size: int,
    total_items: int,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a paginated response."""
    pagination = PaginationMeta.create(page, page_size, total_items)

    return {
        "success": True,
        "data": items,
        "pagination": pagination.dict(),
        "error": None,
        "request_id": request_id or generate_request_id(),
        "timestamp": datetime.utcnow().isoformat(),
    }


__all__ = [
    # Enums
    "APIVersion",
    "SortOrder",
    "ResourceType",
    "ErrorCode",
    # Request models
    "PaginationParams",
    "DateRangeFilter",
    "SearchParams",
    # Response models
    "APIError",
    "PaginationMeta",
    "APIResponse",
    "ListResponse",
    "BatchOperationResult",
    # Resource models
    "ResourceBase",
    "ResourceReference",
    "AuditInfo",
    # Exceptions
    "APIException",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "ValidationError",
    "ConflictError",
    "RateLimitError",
    "ServiceError",
    # Utilities
    "generate_api_key",
    "hash_api_key",
    "generate_request_id",
    "success_response",
    "error_response",
    "paginated_response",
]
