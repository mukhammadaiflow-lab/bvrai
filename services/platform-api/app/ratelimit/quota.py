"""
Quota Management System

Enterprise quota management with:
- Multi-tenant quota allocation
- Usage tracking and enforcement
- Quota policies and overages
- Billing integration hooks
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class QuotaPeriod(str, Enum):
    """Quota period types."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class QuotaUnit(str, Enum):
    """Units for quota measurement."""
    REQUESTS = "requests"
    TOKENS = "tokens"
    MINUTES = "minutes"
    BYTES = "bytes"
    CREDITS = "credits"
    CALLS = "calls"
    MESSAGES = "messages"


class OveragePolicy(str, Enum):
    """Policy when quota is exceeded."""
    HARD_LIMIT = "hard_limit"  # Deny requests
    SOFT_LIMIT = "soft_limit"  # Allow with warning
    PAY_AS_YOU_GO = "pay_as_you_go"  # Allow and charge
    THROTTLE = "throttle"  # Reduce rate but allow
    QUEUE = "queue"  # Queue for later processing


@dataclass
class QuotaDefinition:
    """Definition of a quota limit."""
    name: str
    unit: QuotaUnit
    limit: int
    period: QuotaPeriod
    period_seconds: Optional[int] = None  # For CUSTOM period
    overage_policy: OveragePolicy = OveragePolicy.HARD_LIMIT
    overage_rate: Optional[float] = None  # Cost per unit over limit
    burst_allowance: float = 0.0  # Allow temporary burst (0-1)
    warning_threshold: float = 0.8  # Warn at this percentage

    def get_period_seconds(self) -> int:
        """Get period duration in seconds."""
        if self.period == QuotaPeriod.HOURLY:
            return 3600
        elif self.period == QuotaPeriod.DAILY:
            return 86400
        elif self.period == QuotaPeriod.WEEKLY:
            return 604800
        elif self.period == QuotaPeriod.MONTHLY:
            return 2592000  # 30 days
        elif self.period == QuotaPeriod.YEARLY:
            return 31536000  # 365 days
        elif self.period == QuotaPeriod.CUSTOM and self.period_seconds:
            return self.period_seconds
        return 86400  # Default to daily


@dataclass
class QuotaUsage:
    """Current quota usage."""
    quota_name: str
    tenant_id: str
    used: int
    limit: int
    period_start: datetime
    period_end: datetime
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def remaining(self) -> int:
        """Remaining quota."""
        return max(0, self.limit - self.used)

    @property
    def percentage_used(self) -> float:
        """Percentage of quota used."""
        return (self.used / self.limit * 100) if self.limit > 0 else 100.0

    @property
    def is_exceeded(self) -> bool:
        """Check if quota is exceeded."""
        return self.used >= self.limit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quota_name": self.quota_name,
            "tenant_id": self.tenant_id,
            "used": self.used,
            "limit": self.limit,
            "remaining": self.remaining,
            "percentage_used": self.percentage_used,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "is_exceeded": self.is_exceeded,
        }


@dataclass
class QuotaCheckResult:
    """Result of a quota check."""
    allowed: bool
    quota_name: str
    usage: QuotaUsage
    overage_amount: int = 0
    overage_cost: float = 0.0
    warning: Optional[str] = None
    policy_applied: Optional[OveragePolicy] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "quota_name": self.quota_name,
            "usage": self.usage.to_dict(),
            "overage_amount": self.overage_amount,
            "overage_cost": self.overage_cost,
            "warning": self.warning,
            "policy_applied": self.policy_applied.value if self.policy_applied else None,
        }


class QuotaStorage(ABC):
    """Abstract storage backend for quotas."""

    @abstractmethod
    async def get_usage(
        self,
        tenant_id: str,
        quota_name: str,
        period_start: datetime,
    ) -> Optional[QuotaUsage]:
        """Get current usage."""
        pass

    @abstractmethod
    async def increment_usage(
        self,
        tenant_id: str,
        quota_name: str,
        amount: int,
        period_start: datetime,
        period_end: datetime,
        limit: int,
    ) -> QuotaUsage:
        """Increment usage and return updated."""
        pass

    @abstractmethod
    async def reset_usage(
        self,
        tenant_id: str,
        quota_name: str,
    ) -> None:
        """Reset usage for a quota."""
        pass

    @abstractmethod
    async def get_all_usage(
        self,
        tenant_id: str,
    ) -> Dict[str, QuotaUsage]:
        """Get all quota usage for a tenant."""
        pass


class InMemoryQuotaStorage(QuotaStorage):
    """In-memory quota storage for development/testing."""

    def __init__(self):
        self._storage: Dict[str, Dict[str, QuotaUsage]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def get_usage(
        self,
        tenant_id: str,
        quota_name: str,
        period_start: datetime,
    ) -> Optional[QuotaUsage]:
        """Get current usage."""
        async with self._lock:
            key = f"{tenant_id}:{quota_name}"
            usage = self._storage.get(tenant_id, {}).get(quota_name)

            # Check if period is current
            if usage and usage.period_start <= datetime.utcnow() < usage.period_end:
                return usage

            return None

    async def increment_usage(
        self,
        tenant_id: str,
        quota_name: str,
        amount: int,
        period_start: datetime,
        period_end: datetime,
        limit: int,
    ) -> QuotaUsage:
        """Increment usage."""
        async with self._lock:
            current = self._storage.get(tenant_id, {}).get(quota_name)

            if current and current.period_start == period_start:
                current.used += amount
                current.last_updated = datetime.utcnow()
            else:
                current = QuotaUsage(
                    quota_name=quota_name,
                    tenant_id=tenant_id,
                    used=amount,
                    limit=limit,
                    period_start=period_start,
                    period_end=period_end,
                )
                if tenant_id not in self._storage:
                    self._storage[tenant_id] = {}
                self._storage[tenant_id][quota_name] = current

            return current

    async def reset_usage(
        self,
        tenant_id: str,
        quota_name: str,
    ) -> None:
        """Reset usage."""
        async with self._lock:
            if tenant_id in self._storage:
                self._storage[tenant_id].pop(quota_name, None)

    async def get_all_usage(
        self,
        tenant_id: str,
    ) -> Dict[str, QuotaUsage]:
        """Get all usage for a tenant."""
        async with self._lock:
            return dict(self._storage.get(tenant_id, {}))


class RedisQuotaStorage(QuotaStorage):
    """Redis-backed quota storage for production."""

    def __init__(
        self,
        redis_client: Any,
        key_prefix: str = "quota",
    ):
        self.redis = redis_client
        self.key_prefix = key_prefix

    def _make_key(self, tenant_id: str, quota_name: str) -> str:
        """Create Redis key."""
        return f"{self.key_prefix}:{tenant_id}:{quota_name}"

    async def get_usage(
        self,
        tenant_id: str,
        quota_name: str,
        period_start: datetime,
    ) -> Optional[QuotaUsage]:
        """Get current usage from Redis."""
        key = self._make_key(tenant_id, quota_name)

        try:
            data = await self.redis.hgetall(key)
            if not data:
                return None

            # Parse stored data
            stored_start = datetime.fromisoformat(data[b"period_start"].decode())
            if stored_start != period_start:
                return None

            return QuotaUsage(
                quota_name=quota_name,
                tenant_id=tenant_id,
                used=int(data[b"used"]),
                limit=int(data[b"limit"]),
                period_start=stored_start,
                period_end=datetime.fromisoformat(data[b"period_end"].decode()),
                last_updated=datetime.fromisoformat(data[b"last_updated"].decode()),
            )
        except Exception as e:
            logger.error(f"Failed to get quota usage: {e}")
            return None

    async def increment_usage(
        self,
        tenant_id: str,
        quota_name: str,
        amount: int,
        period_start: datetime,
        period_end: datetime,
        limit: int,
    ) -> QuotaUsage:
        """Increment usage atomically in Redis."""
        key = self._make_key(tenant_id, quota_name)

        # Lua script for atomic increment
        script = """
        local key = KEYS[1]
        local amount = tonumber(ARGV[1])
        local period_start = ARGV[2]
        local period_end = ARGV[3]
        local limit = tonumber(ARGV[4])
        local now = ARGV[5]
        local ttl = tonumber(ARGV[6])

        -- Check if period matches
        local stored_start = redis.call('HGET', key, 'period_start')
        if stored_start and stored_start ~= period_start then
            -- New period, reset
            redis.call('DEL', key)
        end

        -- Increment or create
        local new_used = redis.call('HINCRBY', key, 'used', amount)
        redis.call('HMSET', key,
            'limit', limit,
            'period_start', period_start,
            'period_end', period_end,
            'last_updated', now
        )
        redis.call('EXPIRE', key, ttl)

        return new_used
        """

        try:
            ttl = int((period_end - datetime.utcnow()).total_seconds()) + 3600
            now = datetime.utcnow().isoformat()

            new_used = await self.redis.eval(
                script,
                1,
                key,
                amount,
                period_start.isoformat(),
                period_end.isoformat(),
                limit,
                now,
                ttl,
            )

            return QuotaUsage(
                quota_name=quota_name,
                tenant_id=tenant_id,
                used=int(new_used),
                limit=limit,
                period_start=period_start,
                period_end=period_end,
            )
        except Exception as e:
            logger.error(f"Failed to increment quota: {e}")
            raise

    async def reset_usage(
        self,
        tenant_id: str,
        quota_name: str,
    ) -> None:
        """Reset usage in Redis."""
        key = self._make_key(tenant_id, quota_name)
        await self.redis.delete(key)

    async def get_all_usage(
        self,
        tenant_id: str,
    ) -> Dict[str, QuotaUsage]:
        """Get all usage for a tenant."""
        pattern = f"{self.key_prefix}:{tenant_id}:*"
        usage = {}

        async for key in self.redis.scan_iter(match=pattern):
            data = await self.redis.hgetall(key)
            if data:
                quota_name = key.decode().split(":")[-1]
                usage[quota_name] = QuotaUsage(
                    quota_name=quota_name,
                    tenant_id=tenant_id,
                    used=int(data[b"used"]),
                    limit=int(data[b"limit"]),
                    period_start=datetime.fromisoformat(data[b"period_start"].decode()),
                    period_end=datetime.fromisoformat(data[b"period_end"].decode()),
                )

        return usage


@dataclass
class TenantQuota:
    """Quota configuration for a tenant."""
    tenant_id: str
    quotas: Dict[str, QuotaDefinition] = field(default_factory=dict)
    custom_limits: Dict[str, int] = field(default_factory=dict)  # Override default limits
    is_unlimited: bool = False
    suspended: bool = False
    suspension_reason: Optional[str] = None

    def get_effective_limit(self, quota_name: str) -> Optional[int]:
        """Get effective limit for a quota."""
        if self.is_unlimited:
            return None

        if quota_name in self.custom_limits:
            return self.custom_limits[quota_name]

        quota = self.quotas.get(quota_name)
        return quota.limit if quota else None


class QuotaManager:
    """
    Enterprise quota management system.

    Features:
    - Multi-tenant quota management
    - Multiple quota types (requests, tokens, minutes, etc.)
    - Flexible period configurations
    - Overage handling policies
    - Usage tracking and reporting
    - Billing integration hooks
    """

    def __init__(
        self,
        storage: Optional[QuotaStorage] = None,
        default_quotas: Optional[Dict[str, QuotaDefinition]] = None,
        billing_callback: Optional[Callable[[str, str, int, float], Awaitable[None]]] = None,
    ):
        """
        Args:
            storage: Quota storage backend
            default_quotas: Default quota definitions
            billing_callback: Callback for overage billing (tenant_id, quota_name, amount, cost)
        """
        self.storage = storage or InMemoryQuotaStorage()
        self.default_quotas = default_quotas or self._default_quotas()
        self.billing_callback = billing_callback

        self._tenant_configs: Dict[str, TenantQuota] = {}
        self._lock = asyncio.Lock()

        # Event hooks
        self._on_warning_hooks: List[Callable] = []
        self._on_exceeded_hooks: List[Callable] = []
        self._on_overage_hooks: List[Callable] = []

    def _default_quotas(self) -> Dict[str, QuotaDefinition]:
        """Default quota definitions."""
        return {
            "api_requests": QuotaDefinition(
                name="api_requests",
                unit=QuotaUnit.REQUESTS,
                limit=10000,
                period=QuotaPeriod.DAILY,
                overage_policy=OveragePolicy.HARD_LIMIT,
                warning_threshold=0.8,
            ),
            "voice_minutes": QuotaDefinition(
                name="voice_minutes",
                unit=QuotaUnit.MINUTES,
                limit=1000,
                period=QuotaPeriod.MONTHLY,
                overage_policy=OveragePolicy.PAY_AS_YOU_GO,
                overage_rate=0.02,  # $0.02 per minute overage
                warning_threshold=0.9,
            ),
            "ai_tokens": QuotaDefinition(
                name="ai_tokens",
                unit=QuotaUnit.TOKENS,
                limit=1000000,
                period=QuotaPeriod.MONTHLY,
                overage_policy=OveragePolicy.PAY_AS_YOU_GO,
                overage_rate=0.00001,  # $0.00001 per token
                warning_threshold=0.8,
            ),
            "concurrent_calls": QuotaDefinition(
                name="concurrent_calls",
                unit=QuotaUnit.CALLS,
                limit=10,
                period=QuotaPeriod.CUSTOM,
                period_seconds=1,  # Per-second limit (effectively concurrent)
                overage_policy=OveragePolicy.HARD_LIMIT,
            ),
            "storage_bytes": QuotaDefinition(
                name="storage_bytes",
                unit=QuotaUnit.BYTES,
                limit=10 * 1024 * 1024 * 1024,  # 10 GB
                period=QuotaPeriod.MONTHLY,
                overage_policy=OveragePolicy.PAY_AS_YOU_GO,
                overage_rate=0.0000000001,  # $0.10 per GB
            ),
        }

    def _get_period_boundaries(
        self,
        quota: QuotaDefinition,
        reference_time: Optional[datetime] = None,
    ) -> tuple[datetime, datetime]:
        """Get period start and end times."""
        now = reference_time or datetime.utcnow()
        period_seconds = quota.get_period_seconds()

        if quota.period == QuotaPeriod.HOURLY:
            start = now.replace(minute=0, second=0, microsecond=0)
        elif quota.period == QuotaPeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif quota.period == QuotaPeriod.WEEKLY:
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif quota.period == QuotaPeriod.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif quota.period == QuotaPeriod.YEARLY:
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            # Custom period - align to epoch
            epoch = datetime(1970, 1, 1)
            seconds_since_epoch = (now - epoch).total_seconds()
            period_start_seconds = (seconds_since_epoch // period_seconds) * period_seconds
            start = epoch + timedelta(seconds=period_start_seconds)

        end = start + timedelta(seconds=period_seconds)
        return start, end

    async def configure_tenant(
        self,
        tenant_id: str,
        quotas: Optional[Dict[str, QuotaDefinition]] = None,
        custom_limits: Optional[Dict[str, int]] = None,
        is_unlimited: bool = False,
    ) -> TenantQuota:
        """Configure quotas for a tenant."""
        async with self._lock:
            config = TenantQuota(
                tenant_id=tenant_id,
                quotas=quotas or self.default_quotas.copy(),
                custom_limits=custom_limits or {},
                is_unlimited=is_unlimited,
            )
            self._tenant_configs[tenant_id] = config
            return config

    async def get_tenant_config(self, tenant_id: str) -> TenantQuota:
        """Get tenant configuration, creating default if not exists."""
        if tenant_id not in self._tenant_configs:
            return await self.configure_tenant(tenant_id)
        return self._tenant_configs[tenant_id]

    async def check_quota(
        self,
        tenant_id: str,
        quota_name: str,
        amount: int = 1,
        consume: bool = True,
    ) -> QuotaCheckResult:
        """
        Check and optionally consume quota.

        Args:
            tenant_id: Tenant identifier
            quota_name: Quota to check
            amount: Amount to consume
            consume: Whether to actually consume (False for peek)

        Returns:
            QuotaCheckResult with decision and details
        """
        config = await self.get_tenant_config(tenant_id)

        # Check if tenant is suspended
        if config.suspended:
            return QuotaCheckResult(
                allowed=False,
                quota_name=quota_name,
                usage=QuotaUsage(
                    quota_name=quota_name,
                    tenant_id=tenant_id,
                    used=0,
                    limit=0,
                    period_start=datetime.utcnow(),
                    period_end=datetime.utcnow(),
                ),
                warning=f"Tenant suspended: {config.suspension_reason}",
            )

        # Check if unlimited
        if config.is_unlimited:
            return QuotaCheckResult(
                allowed=True,
                quota_name=quota_name,
                usage=QuotaUsage(
                    quota_name=quota_name,
                    tenant_id=tenant_id,
                    used=amount if consume else 0,
                    limit=-1,  # Unlimited
                    period_start=datetime.utcnow(),
                    period_end=datetime.utcnow() + timedelta(days=365),
                ),
            )

        # Get quota definition
        quota = config.quotas.get(quota_name) or self.default_quotas.get(quota_name)
        if not quota:
            logger.warning(f"Unknown quota: {quota_name}")
            return QuotaCheckResult(
                allowed=True,
                quota_name=quota_name,
                usage=QuotaUsage(
                    quota_name=quota_name,
                    tenant_id=tenant_id,
                    used=0,
                    limit=-1,
                    period_start=datetime.utcnow(),
                    period_end=datetime.utcnow(),
                ),
            )

        # Get effective limit
        effective_limit = config.get_effective_limit(quota_name) or quota.limit

        # Calculate period boundaries
        period_start, period_end = self._get_period_boundaries(quota)

        # Get current usage
        current_usage = await self.storage.get_usage(tenant_id, quota_name, period_start)
        current_used = current_usage.used if current_usage else 0

        # Calculate with burst allowance
        effective_limit_with_burst = int(effective_limit * (1 + quota.burst_allowance))

        # Check if would exceed
        new_used = current_used + amount
        is_exceeded = new_used > effective_limit
        is_burst_exceeded = new_used > effective_limit_with_burst

        # Initialize result
        result = QuotaCheckResult(
            allowed=True,
            quota_name=quota_name,
            usage=QuotaUsage(
                quota_name=quota_name,
                tenant_id=tenant_id,
                used=current_used,
                limit=effective_limit,
                period_start=period_start,
                period_end=period_end,
            ),
        )

        # Check warning threshold
        warning_used = current_used / effective_limit if effective_limit > 0 else 0
        if warning_used >= quota.warning_threshold:
            result.warning = (
                f"Quota {quota_name} at {warning_used*100:.1f}% "
                f"({current_used}/{effective_limit})"
            )
            await self._trigger_warning(tenant_id, quota_name, result)

        # Apply overage policy
        if is_exceeded:
            overage = new_used - effective_limit

            if quota.overage_policy == OveragePolicy.HARD_LIMIT:
                if is_burst_exceeded:
                    result.allowed = False
                    result.policy_applied = OveragePolicy.HARD_LIMIT
                else:
                    # Within burst allowance
                    result.allowed = True
                    result.warning = f"Using burst allowance for {quota_name}"

            elif quota.overage_policy == OveragePolicy.SOFT_LIMIT:
                result.allowed = True
                result.warning = f"Soft limit exceeded for {quota_name}"
                result.policy_applied = OveragePolicy.SOFT_LIMIT

            elif quota.overage_policy == OveragePolicy.PAY_AS_YOU_GO:
                result.allowed = True
                result.overage_amount = overage
                result.overage_cost = overage * (quota.overage_rate or 0)
                result.policy_applied = OveragePolicy.PAY_AS_YOU_GO

            elif quota.overage_policy == OveragePolicy.THROTTLE:
                result.allowed = True
                result.policy_applied = OveragePolicy.THROTTLE
                result.warning = f"Throttled due to {quota_name} quota"

            elif quota.overage_policy == OveragePolicy.QUEUE:
                result.allowed = True
                result.policy_applied = OveragePolicy.QUEUE
                result.warning = f"Request queued due to {quota_name} quota"

        # Consume if allowed and requested
        if result.allowed and consume:
            updated_usage = await self.storage.increment_usage(
                tenant_id=tenant_id,
                quota_name=quota_name,
                amount=amount,
                period_start=period_start,
                period_end=period_end,
                limit=effective_limit,
            )
            result.usage = updated_usage

            # Handle billing for overage
            if result.overage_cost > 0 and self.billing_callback:
                await self.billing_callback(
                    tenant_id,
                    quota_name,
                    result.overage_amount,
                    result.overage_cost,
                )
                await self._trigger_overage(tenant_id, quota_name, result)

        # Trigger exceeded hook if not allowed
        if not result.allowed:
            await self._trigger_exceeded(tenant_id, quota_name, result)

        return result

    async def get_usage(
        self,
        tenant_id: str,
        quota_name: Optional[str] = None,
    ) -> Dict[str, QuotaUsage]:
        """Get current usage for a tenant."""
        if quota_name:
            config = await self.get_tenant_config(tenant_id)
            quota = config.quotas.get(quota_name) or self.default_quotas.get(quota_name)
            if not quota:
                return {}

            period_start, _ = self._get_period_boundaries(quota)
            usage = await self.storage.get_usage(tenant_id, quota_name, period_start)
            return {quota_name: usage} if usage else {}

        return await self.storage.get_all_usage(tenant_id)

    async def reset_quota(
        self,
        tenant_id: str,
        quota_name: str,
    ) -> None:
        """Reset a quota for a tenant."""
        await self.storage.reset_usage(tenant_id, quota_name)
        logger.info(f"Reset quota {quota_name} for tenant {tenant_id}")

    async def suspend_tenant(
        self,
        tenant_id: str,
        reason: str,
    ) -> None:
        """Suspend a tenant's quota access."""
        config = await self.get_tenant_config(tenant_id)
        config.suspended = True
        config.suspension_reason = reason
        logger.warning(f"Suspended tenant {tenant_id}: {reason}")

    async def unsuspend_tenant(
        self,
        tenant_id: str,
    ) -> None:
        """Unsuspend a tenant."""
        config = await self.get_tenant_config(tenant_id)
        config.suspended = False
        config.suspension_reason = None
        logger.info(f"Unsuspended tenant {tenant_id}")

    async def set_custom_limit(
        self,
        tenant_id: str,
        quota_name: str,
        limit: int,
    ) -> None:
        """Set a custom limit for a tenant."""
        config = await self.get_tenant_config(tenant_id)
        config.custom_limits[quota_name] = limit
        logger.info(f"Set custom limit for {tenant_id}/{quota_name}: {limit}")

    def on_warning(self, callback: Callable) -> None:
        """Register warning callback."""
        self._on_warning_hooks.append(callback)

    def on_exceeded(self, callback: Callable) -> None:
        """Register exceeded callback."""
        self._on_exceeded_hooks.append(callback)

    def on_overage(self, callback: Callable) -> None:
        """Register overage callback."""
        self._on_overage_hooks.append(callback)

    async def _trigger_warning(
        self,
        tenant_id: str,
        quota_name: str,
        result: QuotaCheckResult,
    ) -> None:
        """Trigger warning hooks."""
        for hook in self._on_warning_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(tenant_id, quota_name, result)
                else:
                    hook(tenant_id, quota_name, result)
            except Exception as e:
                logger.error(f"Warning hook error: {e}")

    async def _trigger_exceeded(
        self,
        tenant_id: str,
        quota_name: str,
        result: QuotaCheckResult,
    ) -> None:
        """Trigger exceeded hooks."""
        for hook in self._on_exceeded_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(tenant_id, quota_name, result)
                else:
                    hook(tenant_id, quota_name, result)
            except Exception as e:
                logger.error(f"Exceeded hook error: {e}")

    async def _trigger_overage(
        self,
        tenant_id: str,
        quota_name: str,
        result: QuotaCheckResult,
    ) -> None:
        """Trigger overage hooks."""
        for hook in self._on_overage_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(tenant_id, quota_name, result)
                else:
                    hook(tenant_id, quota_name, result)
            except Exception as e:
                logger.error(f"Overage hook error: {e}")


class QuotaReporter:
    """
    Quota reporting and analytics.

    Generates usage reports, forecasts, and recommendations.
    """

    def __init__(self, quota_manager: QuotaManager):
        self.quota_manager = quota_manager

    async def generate_usage_report(
        self,
        tenant_id: str,
    ) -> Dict[str, Any]:
        """Generate comprehensive usage report."""
        config = await self.quota_manager.get_tenant_config(tenant_id)
        all_usage = await self.quota_manager.get_usage(tenant_id)

        report = {
            "tenant_id": tenant_id,
            "generated_at": datetime.utcnow().isoformat(),
            "is_unlimited": config.is_unlimited,
            "suspended": config.suspended,
            "quotas": {},
        }

        for quota_name, quota in config.quotas.items():
            usage = all_usage.get(quota_name)
            effective_limit = config.get_effective_limit(quota_name) or quota.limit

            quota_report = {
                "name": quota_name,
                "unit": quota.unit.value,
                "period": quota.period.value,
                "limit": effective_limit,
                "used": usage.used if usage else 0,
                "remaining": usage.remaining if usage else effective_limit,
                "percentage_used": usage.percentage_used if usage else 0,
                "overage_policy": quota.overage_policy.value,
                "overage_rate": quota.overage_rate,
            }

            # Add forecast if we have usage data
            if usage:
                quota_report["forecast"] = self._forecast_usage(usage, quota)

            report["quotas"][quota_name] = quota_report

        return report

    def _forecast_usage(
        self,
        usage: QuotaUsage,
        quota: QuotaDefinition,
    ) -> Dict[str, Any]:
        """Forecast usage to end of period."""
        now = datetime.utcnow()
        elapsed = (now - usage.period_start).total_seconds()
        remaining_seconds = (usage.period_end - now).total_seconds()

        if elapsed <= 0 or remaining_seconds <= 0:
            return {"projected_total": usage.used, "will_exceed": usage.is_exceeded}

        # Simple linear projection
        rate = usage.used / elapsed
        projected_total = usage.used + (rate * remaining_seconds)

        return {
            "current_rate_per_hour": rate * 3600,
            "projected_total": int(projected_total),
            "will_exceed": projected_total > usage.limit,
            "projected_overage": max(0, int(projected_total - usage.limit)),
        }

    async def get_top_consumers(
        self,
        quota_name: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get top quota consumers."""
        consumers = []

        for tenant_id, config in self.quota_manager._tenant_configs.items():
            if config.is_unlimited:
                continue

            usage_dict = await self.quota_manager.get_usage(tenant_id, quota_name)
            usage = usage_dict.get(quota_name)

            if usage:
                consumers.append({
                    "tenant_id": tenant_id,
                    "used": usage.used,
                    "limit": usage.limit,
                    "percentage": usage.percentage_used,
                })

        # Sort by usage
        consumers.sort(key=lambda x: x["used"], reverse=True)
        return consumers[:limit]


class Quota:
    """Simplified quota interface for common operations."""

    def __init__(self, manager: QuotaManager, tenant_id: str):
        self.manager = manager
        self.tenant_id = tenant_id

    async def check(self, quota_name: str, amount: int = 1) -> bool:
        """Check and consume quota, returning True if allowed."""
        result = await self.manager.check_quota(self.tenant_id, quota_name, amount)
        return result.allowed

    async def peek(self, quota_name: str, amount: int = 1) -> bool:
        """Check quota without consuming."""
        result = await self.manager.check_quota(
            self.tenant_id, quota_name, amount, consume=False
        )
        return result.allowed

    async def usage(self, quota_name: str) -> Optional[QuotaUsage]:
        """Get current usage."""
        usage_dict = await self.manager.get_usage(self.tenant_id, quota_name)
        return usage_dict.get(quota_name)

    async def remaining(self, quota_name: str) -> int:
        """Get remaining quota."""
        usage = await self.usage(quota_name)
        return usage.remaining if usage else 0
