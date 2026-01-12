"""
Tenant Quota Management

Resource quotas and limits for multi-tenant:
- Per-tenant resource limits
- Usage tracking
- Quota enforcement
- Overage handling
"""

from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging

logger = logging.getLogger(__name__)


class QuotaType(str, Enum):
    """Types of quotas."""
    # API quotas
    API_CALLS = "api_calls"
    API_CALLS_PER_MINUTE = "api_calls_per_minute"
    API_CALLS_PER_DAY = "api_calls_per_day"

    # Voice quotas
    VOICE_CALLS = "voice_calls"
    CONCURRENT_CALLS = "concurrent_calls"
    CALL_MINUTES = "call_minutes"
    CALL_MINUTES_PER_MONTH = "call_minutes_per_month"

    # Resource quotas
    STORAGE_BYTES = "storage_bytes"
    AGENTS = "agents"
    USERS = "users"
    INTEGRATIONS = "integrations"
    WEBHOOKS = "webhooks"

    # Feature quotas
    CUSTOM_VOICES = "custom_voices"
    PHONE_NUMBERS = "phone_numbers"
    RECORDINGS = "recordings"

    # Custom
    CUSTOM = "custom"


class QuotaPeriod(str, Enum):
    """Quota reset periods."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    LIFETIME = "lifetime"  # No reset


class QuotaAction(str, Enum):
    """Actions when quota exceeded."""
    BLOCK = "block"  # Block operation
    WARN = "warn"  # Warn but allow
    THROTTLE = "throttle"  # Slow down
    CHARGE = "charge"  # Allow with overage charge
    QUEUE = "queue"  # Queue for later


@dataclass
class QuotaLimit:
    """Quota limit definition."""
    quota_type: QuotaType
    limit: int
    period: QuotaPeriod = QuotaPeriod.MONTH
    action_on_exceed: QuotaAction = QuotaAction.BLOCK
    warning_threshold: float = 0.8  # Warn at 80%
    soft_limit: Optional[int] = None  # Soft limit before hard block
    overage_rate: Optional[float] = None  # Cost per unit over limit

    def is_unlimited(self) -> bool:
        """Check if quota is unlimited."""
        return self.limit == -1


@dataclass
class QuotaUsage:
    """Current quota usage."""
    quota_type: QuotaType
    tenant_id: str
    current_usage: int = 0
    limit: int = 0
    period: QuotaPeriod = QuotaPeriod.MONTH
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def remaining(self) -> int:
        """Get remaining quota."""
        if self.limit == -1:
            return -1  # Unlimited
        return max(0, self.limit - self.current_usage)

    @property
    def percentage_used(self) -> float:
        """Get percentage of quota used."""
        if self.limit == -1 or self.limit == 0:
            return 0.0
        return (self.current_usage / self.limit) * 100

    @property
    def is_exceeded(self) -> bool:
        """Check if quota is exceeded."""
        if self.limit == -1:
            return False
        return self.current_usage >= self.limit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quota_type": self.quota_type.value,
            "tenant_id": self.tenant_id,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "remaining": self.remaining,
            "percentage_used": round(self.percentage_used, 2),
            "is_exceeded": self.is_exceeded,
            "period": self.period.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat() if self.period_end else None,
        }


@dataclass
class TenantQuota:
    """Complete quota configuration for tenant."""
    tenant_id: str
    limits: Dict[QuotaType, QuotaLimit] = field(default_factory=dict)
    usages: Dict[QuotaType, QuotaUsage] = field(default_factory=dict)
    custom_limits: Dict[str, QuotaLimit] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_limit(self, quota_type: QuotaType) -> Optional[QuotaLimit]:
        """Get limit for quota type."""
        return self.limits.get(quota_type)

    def get_usage(self, quota_type: QuotaType) -> Optional[QuotaUsage]:
        """Get usage for quota type."""
        return self.usages.get(quota_type)

    def check_quota(self, quota_type: QuotaType, amount: int = 1) -> bool:
        """Check if quota allows operation."""
        limit = self.get_limit(quota_type)
        if not limit or limit.is_unlimited():
            return True

        usage = self.get_usage(quota_type)
        if not usage:
            return True

        return usage.current_usage + amount <= limit.limit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "quotas": {
                qt.value: {
                    "limit": self.limits[qt].limit if qt in self.limits else -1,
                    "usage": self.usages[qt].current_usage if qt in self.usages else 0,
                    "remaining": self.usages[qt].remaining if qt in self.usages else -1,
                }
                for qt in QuotaType
                if qt in self.limits or qt in self.usages
            },
        }


class QuotaStore(ABC):
    """Abstract quota storage."""

    @abstractmethod
    async def get_usage(
        self,
        tenant_id: str,
        quota_type: QuotaType,
    ) -> Optional[QuotaUsage]:
        """Get current usage."""
        pass

    @abstractmethod
    async def increment_usage(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: int = 1,
    ) -> QuotaUsage:
        """Increment usage."""
        pass

    @abstractmethod
    async def reset_usage(
        self,
        tenant_id: str,
        quota_type: QuotaType,
    ) -> None:
        """Reset usage."""
        pass


class InMemoryQuotaStore(QuotaStore):
    """In-memory quota store for development."""

    def __init__(self):
        self._usages: Dict[str, Dict[QuotaType, QuotaUsage]] = {}
        self._lock = asyncio.Lock()

    def _get_key(self, tenant_id: str, quota_type: QuotaType) -> str:
        return f"{tenant_id}:{quota_type.value}"

    async def get_usage(
        self,
        tenant_id: str,
        quota_type: QuotaType,
    ) -> Optional[QuotaUsage]:
        """Get current usage."""
        if tenant_id not in self._usages:
            return None
        return self._usages[tenant_id].get(quota_type)

    async def increment_usage(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: int = 1,
    ) -> QuotaUsage:
        """Increment usage."""
        async with self._lock:
            if tenant_id not in self._usages:
                self._usages[tenant_id] = {}

            if quota_type not in self._usages[tenant_id]:
                self._usages[tenant_id][quota_type] = QuotaUsage(
                    quota_type=quota_type,
                    tenant_id=tenant_id,
                )

            self._usages[tenant_id][quota_type].current_usage += amount
            self._usages[tenant_id][quota_type].last_updated = datetime.utcnow()

            return self._usages[tenant_id][quota_type]

    async def reset_usage(
        self,
        tenant_id: str,
        quota_type: QuotaType,
    ) -> None:
        """Reset usage."""
        async with self._lock:
            if tenant_id in self._usages and quota_type in self._usages[tenant_id]:
                self._usages[tenant_id][quota_type].current_usage = 0
                self._usages[tenant_id][quota_type].period_start = datetime.utcnow()


class RedisQuotaStore(QuotaStore):
    """Redis-backed quota store for production."""

    def __init__(self, redis_client: Any = None):
        self._redis = redis_client
        self._key_prefix = "quota:"

    def _make_key(self, tenant_id: str, quota_type: QuotaType) -> str:
        return f"{self._key_prefix}{tenant_id}:{quota_type.value}"

    async def get_usage(
        self,
        tenant_id: str,
        quota_type: QuotaType,
    ) -> Optional[QuotaUsage]:
        """Get usage from Redis."""
        if not self._redis:
            return None

        key = self._make_key(tenant_id, quota_type)
        value = await self._redis.get(key)

        if value is None:
            return None

        return QuotaUsage(
            quota_type=quota_type,
            tenant_id=tenant_id,
            current_usage=int(value),
        )

    async def increment_usage(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: int = 1,
    ) -> QuotaUsage:
        """Increment usage atomically."""
        if not self._redis:
            return QuotaUsage(quota_type=quota_type, tenant_id=tenant_id)

        key = self._make_key(tenant_id, quota_type)
        new_value = await self._redis.incrby(key, amount)

        return QuotaUsage(
            quota_type=quota_type,
            tenant_id=tenant_id,
            current_usage=new_value,
        )

    async def reset_usage(
        self,
        tenant_id: str,
        quota_type: QuotaType,
    ) -> None:
        """Reset usage in Redis."""
        if not self._redis:
            return

        key = self._make_key(tenant_id, quota_type)
        await self._redis.delete(key)


class TenantQuotaManager:
    """
    Manages tenant quotas.

    Provides quota checking, enforcement, and tracking.
    """

    def __init__(
        self,
        store: Optional[QuotaStore] = None,
        default_limits: Optional[Dict[QuotaType, QuotaLimit]] = None,
    ):
        self.store = store or InMemoryQuotaStore()
        self.default_limits = default_limits or self._get_default_limits()
        self._tenant_limits: Dict[str, Dict[QuotaType, QuotaLimit]] = {}
        self._warning_callbacks: List[callable] = []

    def _get_default_limits(self) -> Dict[QuotaType, QuotaLimit]:
        """Get default quota limits."""
        return {
            QuotaType.API_CALLS_PER_DAY: QuotaLimit(
                quota_type=QuotaType.API_CALLS_PER_DAY,
                limit=10000,
                period=QuotaPeriod.DAY,
                warning_threshold=0.8,
            ),
            QuotaType.API_CALLS_PER_MINUTE: QuotaLimit(
                quota_type=QuotaType.API_CALLS_PER_MINUTE,
                limit=60,
                period=QuotaPeriod.MINUTE,
                action_on_exceed=QuotaAction.THROTTLE,
            ),
            QuotaType.VOICE_CALLS: QuotaLimit(
                quota_type=QuotaType.VOICE_CALLS,
                limit=1000,
                period=QuotaPeriod.MONTH,
                warning_threshold=0.9,
            ),
            QuotaType.CONCURRENT_CALLS: QuotaLimit(
                quota_type=QuotaType.CONCURRENT_CALLS,
                limit=10,
                period=QuotaPeriod.LIFETIME,  # Real-time limit
            ),
            QuotaType.CALL_MINUTES_PER_MONTH: QuotaLimit(
                quota_type=QuotaType.CALL_MINUTES_PER_MONTH,
                limit=5000,
                period=QuotaPeriod.MONTH,
                action_on_exceed=QuotaAction.CHARGE,
                overage_rate=0.05,  # $0.05 per minute
            ),
            QuotaType.STORAGE_BYTES: QuotaLimit(
                quota_type=QuotaType.STORAGE_BYTES,
                limit=10 * 1024 * 1024 * 1024,  # 10 GB
                period=QuotaPeriod.LIFETIME,
            ),
            QuotaType.AGENTS: QuotaLimit(
                quota_type=QuotaType.AGENTS,
                limit=10,
                period=QuotaPeriod.LIFETIME,
            ),
            QuotaType.USERS: QuotaLimit(
                quota_type=QuotaType.USERS,
                limit=25,
                period=QuotaPeriod.LIFETIME,
            ),
        }

    def set_tenant_limits(
        self,
        tenant_id: str,
        limits: Dict[QuotaType, QuotaLimit],
    ) -> None:
        """Set custom limits for tenant."""
        self._tenant_limits[tenant_id] = limits

    def get_limit(
        self,
        tenant_id: str,
        quota_type: QuotaType,
    ) -> QuotaLimit:
        """Get limit for tenant and quota type."""
        # Check tenant-specific limits
        if tenant_id in self._tenant_limits:
            if quota_type in self._tenant_limits[tenant_id]:
                return self._tenant_limits[tenant_id][quota_type]

        # Fall back to default
        return self.default_limits.get(
            quota_type,
            QuotaLimit(quota_type=quota_type, limit=-1),  # Unlimited default
        )

    async def check_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: int = 1,
    ) -> Dict[str, Any]:
        """
        Check if quota allows operation.

        Returns dict with allowed, remaining, and action.
        """
        limit = self.get_limit(tenant_id, quota_type)

        # Unlimited
        if limit.is_unlimited():
            return {
                "allowed": True,
                "remaining": -1,
                "action": None,
            }

        # Get current usage
        usage = await self.store.get_usage(tenant_id, quota_type)
        current = usage.current_usage if usage else 0

        remaining = limit.limit - current
        new_total = current + amount

        # Check if exceeds
        if new_total > limit.limit:
            return {
                "allowed": limit.action_on_exceed != QuotaAction.BLOCK,
                "remaining": remaining,
                "action": limit.action_on_exceed.value,
                "overage": new_total - limit.limit,
                "overage_cost": (new_total - limit.limit) * (limit.overage_rate or 0),
            }

        # Check warning threshold
        percentage = new_total / limit.limit
        if percentage >= limit.warning_threshold:
            await self._emit_warning(tenant_id, quota_type, percentage)

        return {
            "allowed": True,
            "remaining": remaining - amount,
            "action": None,
        }

    async def consume(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: int = 1,
    ) -> QuotaUsage:
        """Consume quota."""
        # Check first
        check_result = await self.check_quota(tenant_id, quota_type, amount)

        if not check_result["allowed"]:
            raise QuotaExceededException(
                tenant_id=tenant_id,
                quota_type=quota_type,
                limit=self.get_limit(tenant_id, quota_type).limit,
                current=check_result.get("remaining", 0) + amount,
            )

        # Update limit in usage
        limit = self.get_limit(tenant_id, quota_type)

        # Increment usage
        usage = await self.store.increment_usage(tenant_id, quota_type, amount)
        usage.limit = limit.limit
        usage.period = limit.period

        return usage

    async def release(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: int = 1,
    ) -> QuotaUsage:
        """Release consumed quota (for concurrent quotas)."""
        return await self.store.increment_usage(tenant_id, quota_type, -amount)

    async def get_usage(
        self,
        tenant_id: str,
        quota_type: QuotaType,
    ) -> QuotaUsage:
        """Get current usage."""
        usage = await self.store.get_usage(tenant_id, quota_type)
        if not usage:
            limit = self.get_limit(tenant_id, quota_type)
            usage = QuotaUsage(
                quota_type=quota_type,
                tenant_id=tenant_id,
                limit=limit.limit,
                period=limit.period,
            )
        else:
            limit = self.get_limit(tenant_id, quota_type)
            usage.limit = limit.limit
            usage.period = limit.period

        return usage

    async def get_all_usage(self, tenant_id: str) -> TenantQuota:
        """Get all quota usage for tenant."""
        tenant_quota = TenantQuota(tenant_id=tenant_id)

        for quota_type in QuotaType:
            limit = self.get_limit(tenant_id, quota_type)
            tenant_quota.limits[quota_type] = limit

            usage = await self.store.get_usage(tenant_id, quota_type)
            if usage:
                usage.limit = limit.limit
                tenant_quota.usages[quota_type] = usage

        return tenant_quota

    async def reset_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
    ) -> None:
        """Reset quota for period."""
        await self.store.reset_usage(tenant_id, quota_type)
        logger.info(f"Reset quota {quota_type.value} for tenant {tenant_id}")

    def on_warning(self, callback: callable) -> None:
        """Register warning callback."""
        self._warning_callbacks.append(callback)

    async def _emit_warning(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        percentage: float,
    ) -> None:
        """Emit quota warning."""
        for callback in self._warning_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(tenant_id, quota_type, percentage)
                else:
                    callback(tenant_id, quota_type, percentage)
            except Exception as e:
                logger.error(f"Warning callback error: {e}")


class QuotaExceededException(Exception):
    """Raised when quota is exceeded."""

    def __init__(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        limit: int,
        current: int,
    ):
        self.tenant_id = tenant_id
        self.quota_type = quota_type
        self.limit = limit
        self.current = current
        super().__init__(
            f"Quota exceeded for {tenant_id}: {quota_type.value} "
            f"({current}/{limit})"
        )


class QuotaEnforcer:
    """
    Enforces quotas at operation boundaries.

    Use as decorator or context manager.
    """

    def __init__(
        self,
        quota_manager: TenantQuotaManager,
        quota_type: QuotaType,
        amount: int = 1,
    ):
        self.manager = quota_manager
        self.quota_type = quota_type
        self.amount = amount

    async def __aenter__(self):
        """Check and consume quota on enter."""
        from app.tenancy.context import get_current_tenant

        ctx = get_current_tenant()
        if not ctx:
            raise RuntimeError("No tenant context available")

        await self.manager.consume(
            ctx.tenant_id,
            self.quota_type,
            self.amount,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release quota if needed (for concurrent quotas)."""
        # Only release for concurrent quotas
        if self.quota_type == QuotaType.CONCURRENT_CALLS:
            from app.tenancy.context import get_current_tenant

            ctx = get_current_tenant()
            if ctx:
                await self.manager.release(
                    ctx.tenant_id,
                    self.quota_type,
                    self.amount,
                )

    def __call__(self, func):
        """Use as decorator."""
        from functools import wraps

        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with self:
                return await func(*args, **kwargs)

        return wrapper


def enforce_quota(
    quota_type: QuotaType,
    amount: int = 1,
    quota_manager: Optional[TenantQuotaManager] = None,
):
    """
    Decorator to enforce quota on function.

    Usage:
        @enforce_quota(QuotaType.API_CALLS_PER_DAY)
        async def my_endpoint():
            ...
    """
    def decorator(func):
        from functools import wraps

        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = quota_manager or TenantQuotaManager()
            async with QuotaEnforcer(manager, quota_type, amount):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
