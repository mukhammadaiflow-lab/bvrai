"""
Developer Platform Base Types Module

This module defines core types for developer platform features including
API keys, rate limiting, usage tracking, and SDK management.
"""

import hashlib
import hmac
import secrets
import uuid
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
    TypeVar,
    Union,
)


# =============================================================================
# Enums
# =============================================================================


class APIKeyType(str, Enum):
    """Types of API keys."""

    LIVE = "live"  # Production key
    TEST = "test"  # Sandbox/test key
    RESTRICTED = "restricted"  # Limited scope key


class APIKeyStatus(str, Enum):
    """Status of an API key."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"


class RateLimitScope(str, Enum):
    """Scope of rate limiting."""

    GLOBAL = "global"  # Per API key
    ENDPOINT = "endpoint"  # Per endpoint
    IP = "ip"  # Per IP address
    USER = "user"  # Per user


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class QuotaPeriod(str, Enum):
    """Quota period types."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"


class WebhookEventType(str, Enum):
    """Types of webhook events."""

    # Call events
    CALL_STARTED = "call.started"
    CALL_ENDED = "call.ended"
    CALL_TRANSFERRED = "call.transferred"
    CALL_FAILED = "call.failed"

    # Recording events
    RECORDING_READY = "recording.ready"
    TRANSCRIPTION_READY = "transcription.ready"

    # Agent events
    AGENT_CREATED = "agent.created"
    AGENT_UPDATED = "agent.updated"
    AGENT_DELETED = "agent.deleted"

    # Campaign events
    CAMPAIGN_STARTED = "campaign.started"
    CAMPAIGN_COMPLETED = "campaign.completed"
    CAMPAIGN_PAUSED = "campaign.paused"

    # Billing events
    USAGE_THRESHOLD = "usage.threshold"
    INVOICE_CREATED = "invoice.created"
    PAYMENT_RECEIVED = "payment.received"


class SDKLanguage(str, Enum):
    """SDK supported languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    RUBY = "ruby"
    PHP = "php"
    CSHARP = "csharp"


# =============================================================================
# API Key Types
# =============================================================================


@dataclass
class APIKeyScope:
    """Defines the scope/permissions of an API key."""

    # Resource access
    resources: List[str] = field(default_factory=lambda: ["*"])  # e.g., ["calls", "agents"]

    # Action permissions
    actions: List[str] = field(default_factory=lambda: ["*"])  # e.g., ["read", "write", "delete"]

    # IP restrictions
    allowed_ips: List[str] = field(default_factory=list)
    blocked_ips: List[str] = field(default_factory=list)

    # Rate limits (override defaults)
    custom_rate_limit: Optional[int] = None  # requests per minute
    custom_quota: Optional[int] = None  # requests per month

    # Time restrictions
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    def allows_resource(self, resource: str) -> bool:
        """Check if resource access is allowed."""
        if "*" in self.resources:
            return True
        return resource in self.resources

    def allows_action(self, action: str) -> bool:
        """Check if action is allowed."""
        if "*" in self.actions:
            return True
        return action in self.actions

    def allows_ip(self, ip: str) -> bool:
        """Check if IP is allowed."""
        if ip in self.blocked_ips:
            return False
        if not self.allowed_ips:
            return True
        return ip in self.allowed_ips

    def is_time_valid(self) -> bool:
        """Check if key is within valid time period."""
        now = datetime.utcnow()
        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resources": self.resources,
            "actions": self.actions,
            "allowed_ips": self.allowed_ips,
            "blocked_ips": self.blocked_ips,
            "custom_rate_limit": self.custom_rate_limit,
            "custom_quota": self.custom_quota,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIKeyScope":
        """Create from dictionary."""
        return cls(
            resources=data.get("resources", ["*"]),
            actions=data.get("actions", ["*"]),
            allowed_ips=data.get("allowed_ips", []),
            blocked_ips=data.get("blocked_ips", []),
            custom_rate_limit=data.get("custom_rate_limit"),
            custom_quota=data.get("custom_quota"),
            valid_from=datetime.fromisoformat(data["valid_from"]) if data.get("valid_from") else None,
            valid_until=datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None,
        )


@dataclass
class APIKey:
    """An API key for authentication."""

    id: str
    organization_id: str
    name: str

    # Key value (hashed for storage)
    key_prefix: str  # First 8 chars for identification
    key_hash: str  # SHA-256 hash of full key

    # Type and status
    key_type: APIKeyType = APIKeyType.LIVE
    status: APIKeyStatus = APIKeyStatus.ACTIVE

    # Scope
    scope: APIKeyScope = field(default_factory=APIKeyScope)

    # Metadata
    description: str = ""
    created_by: Optional[str] = None
    last_used_at: Optional[datetime] = None
    last_used_ip: Optional[str] = None

    # Usage tracking
    request_count: int = 0
    error_count: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"key_{uuid.uuid4().hex[:24]}"

    def is_valid(self) -> bool:
        """Check if key is valid for use."""
        if self.status != APIKeyStatus.ACTIVE:
            return False
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        if not self.scope.is_time_valid():
            return False
        return True

    def verify(self, key: str) -> bool:
        """Verify a key against stored hash."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return hmac.compare_digest(key_hash, self.key_hash)

    @staticmethod
    def generate_key(key_type: APIKeyType = APIKeyType.LIVE) -> Tuple[str, str, str]:
        """
        Generate a new API key.

        Returns:
            Tuple of (full_key, key_prefix, key_hash)
        """
        prefix = "sk_live_" if key_type == APIKeyType.LIVE else "sk_test_"
        random_part = secrets.token_urlsafe(32)
        full_key = f"{prefix}{random_part}"
        key_prefix = full_key[:16]
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        return full_key, key_prefix, key_hash

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "key_prefix": self.key_prefix,
            "key_type": self.key_type.value,
            "status": self.status.value,
            "scope": self.scope.to_dict(),
            "description": self.description,
            "created_by": self.created_by,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "last_used_ip": self.last_used_ip,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
        }

        if include_sensitive:
            data["key_hash"] = self.key_hash

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIKey":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            organization_id=data["organization_id"],
            name=data["name"],
            key_prefix=data["key_prefix"],
            key_hash=data.get("key_hash", ""),
            key_type=APIKeyType(data.get("key_type", "live")),
            status=APIKeyStatus(data.get("status", "active")),
            scope=APIKeyScope.from_dict(data.get("scope", {})),
            description=data.get("description", ""),
            created_by=data.get("created_by"),
            last_used_at=datetime.fromisoformat(data["last_used_at"]) if data.get("last_used_at") else None,
            last_used_ip=data.get("last_used_ip"),
            request_count=data.get("request_count", 0),
            error_count=data.get("error_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            revoked_at=datetime.fromisoformat(data["revoked_at"]) if data.get("revoked_at") else None,
        )


# =============================================================================
# Rate Limiting Types
# =============================================================================


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Limits
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000

    # Burst settings
    burst_size: int = 100  # Max burst requests
    burst_recovery_rate: float = 10.0  # Tokens recovered per second

    # Strategy
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW

    # Response behavior
    return_retry_after: bool = True
    include_remaining: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "requests_per_day": self.requests_per_day,
            "burst_size": self.burst_size,
            "burst_recovery_rate": self.burst_recovery_rate,
            "strategy": self.strategy.value,
            "return_retry_after": self.return_retry_after,
            "include_remaining": self.include_remaining,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RateLimitConfig":
        """Create from dictionary."""
        return cls(
            requests_per_minute=data.get("requests_per_minute", 60),
            requests_per_hour=data.get("requests_per_hour", 1000),
            requests_per_day=data.get("requests_per_day", 10000),
            burst_size=data.get("burst_size", 100),
            burst_recovery_rate=data.get("burst_recovery_rate", 10.0),
            strategy=RateLimitStrategy(data.get("strategy", "sliding_window")),
            return_retry_after=data.get("return_retry_after", True),
            include_remaining=data.get("include_remaining", True),
        )


@dataclass
class RateLimitState:
    """Current state of rate limiting for an entity."""

    key: str  # Rate limit key (e.g., "api_key:xxx")
    requests_count: int = 0
    window_start: datetime = field(default_factory=datetime.utcnow)

    # Token bucket state
    tokens: float = 0.0
    last_refill: datetime = field(default_factory=datetime.utcnow)

    # Burst tracking
    burst_count: int = 0
    burst_window_start: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "requests_count": self.requests_count,
            "window_start": self.window_start.isoformat(),
            "tokens": self.tokens,
            "last_refill": self.last_refill.isoformat(),
            "burst_count": self.burst_count,
            "burst_window_start": self.burst_window_start.isoformat() if self.burst_window_start else None,
        }


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int = 0
    limit: int = 0
    reset_at: Optional[datetime] = None
    retry_after_seconds: Optional[int] = None

    # Headers to return
    headers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
        }
        if self.reset_at:
            self.headers["X-RateLimit-Reset"] = str(int(self.reset_at.timestamp()))
        if self.retry_after_seconds:
            self.headers["Retry-After"] = str(self.retry_after_seconds)


# =============================================================================
# Usage Tracking Types
# =============================================================================


@dataclass
class UsageQuota:
    """Usage quota configuration."""

    id: str
    organization_id: str
    name: str

    # Limits
    api_calls_limit: int = 100000
    minutes_limit: int = 10000
    storage_mb_limit: int = 10000

    # Current usage
    api_calls_used: int = 0
    minutes_used: float = 0.0
    storage_mb_used: float = 0.0

    # Period
    period: QuotaPeriod = QuotaPeriod.MONTH
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: Optional[datetime] = None

    # Alerts
    alert_threshold_percent: int = 80
    alert_sent: bool = False

    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"quota_{uuid.uuid4().hex[:24]}"
        if not self.period_end:
            self._set_period_end()

    def _set_period_end(self):
        """Set period end based on period type."""
        if self.period == QuotaPeriod.MINUTE:
            self.period_end = self.period_start + timedelta(minutes=1)
        elif self.period == QuotaPeriod.HOUR:
            self.period_end = self.period_start + timedelta(hours=1)
        elif self.period == QuotaPeriod.DAY:
            self.period_end = self.period_start + timedelta(days=1)
        else:  # MONTH
            # Add approximately 30 days
            self.period_end = self.period_start + timedelta(days=30)

    @property
    def api_calls_remaining(self) -> int:
        """Get remaining API calls."""
        return max(0, self.api_calls_limit - self.api_calls_used)

    @property
    def minutes_remaining(self) -> float:
        """Get remaining minutes."""
        return max(0.0, self.minutes_limit - self.minutes_used)

    @property
    def api_calls_percent(self) -> float:
        """Get API calls usage percentage."""
        if self.api_calls_limit == 0:
            return 0.0
        return (self.api_calls_used / self.api_calls_limit) * 100

    def should_alert(self) -> bool:
        """Check if usage alert should be sent."""
        if self.alert_sent:
            return False
        return self.api_calls_percent >= self.alert_threshold_percent

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "api_calls_limit": self.api_calls_limit,
            "minutes_limit": self.minutes_limit,
            "storage_mb_limit": self.storage_mb_limit,
            "api_calls_used": self.api_calls_used,
            "minutes_used": self.minutes_used,
            "storage_mb_used": self.storage_mb_used,
            "api_calls_remaining": self.api_calls_remaining,
            "minutes_remaining": self.minutes_remaining,
            "api_calls_percent": self.api_calls_percent,
            "period": self.period.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "alert_threshold_percent": self.alert_threshold_percent,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class APIUsageEvent:
    """A single API usage event."""

    id: str
    api_key_id: str
    organization_id: str

    # Request info
    endpoint: str
    method: str
    status_code: int

    # Timing
    request_time: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: float = 0.0

    # Usage
    tokens_used: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0

    # Context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"usage_{uuid.uuid4().hex[:24]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "api_key_id": self.api_key_id,
            "organization_id": self.organization_id,
            "endpoint": self.endpoint,
            "method": self.method,
            "status_code": self.status_code,
            "request_time": self.request_time.isoformat(),
            "response_time_ms": self.response_time_ms,
            "tokens_used": self.tokens_used,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
        }


# =============================================================================
# Webhook Types
# =============================================================================


@dataclass
class WebhookEndpoint:
    """A webhook endpoint configuration."""

    id: str
    organization_id: str
    url: str

    # Events
    events: List[WebhookEventType] = field(default_factory=list)
    event_filters: Dict[str, Any] = field(default_factory=dict)

    # Security
    secret: str = ""
    signature_header: str = "X-Webhook-Signature"

    # Settings
    enabled: bool = True
    retry_count: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 30

    # Status
    last_delivery_at: Optional[datetime] = None
    last_delivery_status: Optional[str] = None
    consecutive_failures: int = 0

    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"wh_{uuid.uuid4().hex[:24]}"
        if not self.secret:
            self.secret = secrets.token_urlsafe(32)

    def should_receive(self, event_type: WebhookEventType) -> bool:
        """Check if endpoint should receive event."""
        if not self.enabled:
            return False
        if not self.events:
            return True  # All events
        return event_type in self.events

    def sign_payload(self, payload: str, timestamp: int) -> str:
        """Generate signature for payload."""
        signature_data = f"{timestamp}.{payload}"
        signature = hmac.new(
            self.secret.encode(),
            signature_data.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"t={timestamp},v1={signature}"

    def to_dict(self, include_secret: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "organization_id": self.organization_id,
            "url": self.url,
            "events": [e.value for e in self.events],
            "event_filters": self.event_filters,
            "signature_header": self.signature_header,
            "enabled": self.enabled,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_seconds": self.timeout_seconds,
            "last_delivery_at": self.last_delivery_at.isoformat() if self.last_delivery_at else None,
            "last_delivery_status": self.last_delivery_status,
            "consecutive_failures": self.consecutive_failures,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

        if include_secret:
            data["secret"] = self.secret

        return data


# =============================================================================
# Exceptions
# =============================================================================


class DeveloperPlatformError(Exception):
    """Base exception for developer platform errors."""
    pass


class APIKeyError(DeveloperPlatformError):
    """Error with API key operations."""
    pass


class RateLimitError(DeveloperPlatformError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExceededError(DeveloperPlatformError):
    """Usage quota exceeded."""
    pass


class WebhookError(DeveloperPlatformError):
    """Error with webhook operations."""
    pass


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "APIKeyType",
    "APIKeyStatus",
    "RateLimitScope",
    "RateLimitStrategy",
    "QuotaPeriod",
    "WebhookEventType",
    "SDKLanguage",
    # API Key types
    "APIKeyScope",
    "APIKey",
    # Rate limiting types
    "RateLimitConfig",
    "RateLimitState",
    "RateLimitResult",
    # Usage types
    "UsageQuota",
    "APIUsageEvent",
    # Webhook types
    "WebhookEndpoint",
    # Exceptions
    "DeveloperPlatformError",
    "APIKeyError",
    "RateLimitError",
    "QuotaExceededError",
    "WebhookError",
]
