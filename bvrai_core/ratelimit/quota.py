"""
Quota Management System
=======================

Comprehensive quota tracking and enforcement for resource usage management.

Author: Platform Engineering Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

logger = structlog.get_logger(__name__)


class QuotaExceeded(Exception):
    """Raised when quota is exceeded"""

    def __init__(
        self,
        message: str,
        resource: str,
        limit: int,
        used: int,
        reset_at: datetime,
    ):
        super().__init__(message)
        self.resource = resource
        self.limit = limit
        self.used = used
        self.reset_at = reset_at


class QuotaPeriod(str, Enum):
    """Quota period types"""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    LIFETIME = "lifetime"

    @property
    def seconds(self) -> int:
        """Get period in seconds"""
        mapping = {
            QuotaPeriod.MINUTE: 60,
            QuotaPeriod.HOUR: 3600,
            QuotaPeriod.DAY: 86400,
            QuotaPeriod.WEEK: 604800,
            QuotaPeriod.MONTH: 2592000,  # 30 days
            QuotaPeriod.YEAR: 31536000,  # 365 days
            QuotaPeriod.LIFETIME: 0,
        }
        return mapping.get(self, 0)


@dataclass
class QuotaLimit:
    """A quota limit definition"""

    resource: str
    limit: int
    period: QuotaPeriod
    description: str = ""
    soft_limit: Optional[int] = None  # Warning threshold
    overage_allowed: bool = False
    overage_rate: float = 0.0  # Cost per unit over limit


@dataclass
class QuotaUsage:
    """Current quota usage"""

    resource: str
    used: int
    limit: int
    period: QuotaPeriod
    period_start: datetime
    period_end: datetime
    soft_limit_reached: bool = False
    hard_limit_reached: bool = False

    @property
    def remaining(self) -> int:
        """Get remaining quota"""
        return max(0, self.limit - self.used)

    @property
    def percentage_used(self) -> float:
        """Get percentage of quota used"""
        if self.limit == 0:
            return 100.0
        return (self.used / self.limit) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "resource": self.resource,
            "used": self.used,
            "limit": self.limit,
            "remaining": self.remaining,
            "percentage_used": self.percentage_used,
            "period": self.period.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "soft_limit_reached": self.soft_limit_reached,
            "hard_limit_reached": self.hard_limit_reached,
        }


@dataclass
class Quota:
    """Quota configuration for an entity"""

    entity_id: str  # Organization, user, or API key ID
    entity_type: str = "organization"  # organization, user, api_key
    limits: List[QuotaLimit] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def get_limit(self, resource: str) -> Optional[QuotaLimit]:
        """Get limit for a resource"""
        for limit in self.limits:
            if limit.resource == resource:
                return limit
        return None

    def add_limit(self, limit: QuotaLimit) -> None:
        """Add or update a limit"""
        for i, existing in enumerate(self.limits):
            if existing.resource == limit.resource:
                self.limits[i] = limit
                return
        self.limits.append(limit)


@dataclass
class QuotaConfig:
    """Configuration for quota manager"""

    default_limits: List[QuotaLimit] = field(default_factory=list)
    enable_soft_limits: bool = True
    enable_overage: bool = False
    cleanup_interval: int = 3600  # Seconds
    alert_on_soft_limit: bool = True


class QuotaStore(ABC):
    """Base class for quota storage backends"""

    @abstractmethod
    async def get_quota(self, entity_id: str) -> Optional[Quota]:
        """Get quota for an entity"""
        pass

    @abstractmethod
    async def set_quota(self, quota: Quota) -> None:
        """Set quota for an entity"""
        pass

    @abstractmethod
    async def get_usage(
        self,
        entity_id: str,
        resource: str,
        period: QuotaPeriod,
    ) -> int:
        """Get current usage"""
        pass

    @abstractmethod
    async def increment_usage(
        self,
        entity_id: str,
        resource: str,
        period: QuotaPeriod,
        amount: int = 1,
    ) -> int:
        """Increment usage and return new value"""
        pass

    @abstractmethod
    async def reset_usage(
        self,
        entity_id: str,
        resource: str,
        period: QuotaPeriod,
    ) -> None:
        """Reset usage for a period"""
        pass


class InMemoryQuotaStore(QuotaStore):
    """In-memory quota storage"""

    def __init__(self):
        self._quotas: Dict[str, Quota] = {}
        self._usage: Dict[str, Dict[str, int]] = {}
        self._period_starts: Dict[str, datetime] = {}
        self._lock = threading.RLock()

    async def get_quota(self, entity_id: str) -> Optional[Quota]:
        """Get quota for an entity"""
        with self._lock:
            return self._quotas.get(entity_id)

    async def set_quota(self, quota: Quota) -> None:
        """Set quota for an entity"""
        with self._lock:
            self._quotas[quota.entity_id] = quota

    async def get_usage(
        self,
        entity_id: str,
        resource: str,
        period: QuotaPeriod,
    ) -> int:
        """Get current usage"""
        key = self._make_key(entity_id, resource, period)

        with self._lock:
            self._check_period_reset(key, period)
            return self._usage.get(entity_id, {}).get(key, 0)

    async def increment_usage(
        self,
        entity_id: str,
        resource: str,
        period: QuotaPeriod,
        amount: int = 1,
    ) -> int:
        """Increment usage and return new value"""
        key = self._make_key(entity_id, resource, period)

        with self._lock:
            self._check_period_reset(key, period)

            if entity_id not in self._usage:
                self._usage[entity_id] = {}

            current = self._usage[entity_id].get(key, 0)
            new_value = current + amount
            self._usage[entity_id][key] = new_value

            return new_value

    async def reset_usage(
        self,
        entity_id: str,
        resource: str,
        period: QuotaPeriod,
    ) -> None:
        """Reset usage for a period"""
        key = self._make_key(entity_id, resource, period)

        with self._lock:
            if entity_id in self._usage:
                self._usage[entity_id].pop(key, None)
            self._period_starts.pop(key, None)

    def _make_key(self, entity_id: str, resource: str, period: QuotaPeriod) -> str:
        """Create a unique key"""
        return f"{entity_id}:{resource}:{period.value}"

    def _check_period_reset(self, key: str, period: QuotaPeriod) -> None:
        """Check if period should reset"""
        if period == QuotaPeriod.LIFETIME:
            return

        now = datetime.utcnow()
        period_start = self._period_starts.get(key)

        if period_start is None:
            self._period_starts[key] = self._get_period_start(now, period)
            return

        current_period_start = self._get_period_start(now, period)
        if current_period_start > period_start:
            # Reset usage for new period
            entity_id = key.split(":")[0]
            if entity_id in self._usage:
                self._usage[entity_id].pop(key, None)
            self._period_starts[key] = current_period_start

    def _get_period_start(self, dt: datetime, period: QuotaPeriod) -> datetime:
        """Get the start of the current period"""
        if period == QuotaPeriod.MINUTE:
            return dt.replace(second=0, microsecond=0)
        elif period == QuotaPeriod.HOUR:
            return dt.replace(minute=0, second=0, microsecond=0)
        elif period == QuotaPeriod.DAY:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == QuotaPeriod.WEEK:
            days_since_monday = dt.weekday()
            return (dt - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif period == QuotaPeriod.MONTH:
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == QuotaPeriod.YEAR:
            return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return dt


class RedisQuotaStore(QuotaStore):
    """Redis-backed quota storage"""

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        key_prefix: str = "quota:",
    ):
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._local_store = InMemoryQuotaStore()
        self._logger = structlog.get_logger("redis_quota_store")

    async def get_quota(self, entity_id: str) -> Optional[Quota]:
        """Get quota for an entity"""
        if self._redis is None:
            return await self._local_store.get_quota(entity_id)

        try:
            key = f"{self._key_prefix}config:{entity_id}"
            data = await self._redis.get(key)
            if data:
                return self._deserialize_quota(json.loads(data))
            return None
        except Exception as e:
            self._logger.warning("redis_get_quota_error", error=str(e))
            return await self._local_store.get_quota(entity_id)

    async def set_quota(self, quota: Quota) -> None:
        """Set quota for an entity"""
        if self._redis is None:
            return await self._local_store.set_quota(quota)

        try:
            key = f"{self._key_prefix}config:{quota.entity_id}"
            data = self._serialize_quota(quota)
            await self._redis.set(key, json.dumps(data))
        except Exception as e:
            self._logger.warning("redis_set_quota_error", error=str(e))
            await self._local_store.set_quota(quota)

    async def get_usage(
        self,
        entity_id: str,
        resource: str,
        period: QuotaPeriod,
    ) -> int:
        """Get current usage"""
        if self._redis is None:
            return await self._local_store.get_usage(entity_id, resource, period)

        try:
            key = self._make_usage_key(entity_id, resource, period)
            value = await self._redis.get(key)
            return int(value) if value else 0
        except Exception as e:
            self._logger.warning("redis_get_usage_error", error=str(e))
            return await self._local_store.get_usage(entity_id, resource, period)

    async def increment_usage(
        self,
        entity_id: str,
        resource: str,
        period: QuotaPeriod,
        amount: int = 1,
    ) -> int:
        """Increment usage and return new value"""
        if self._redis is None:
            return await self._local_store.increment_usage(
                entity_id, resource, period, amount
            )

        try:
            key = self._make_usage_key(entity_id, resource, period)
            new_value = await self._redis.incrby(key, amount)

            # Set expiry based on period
            if period != QuotaPeriod.LIFETIME:
                ttl = self._get_period_ttl(period)
                await self._redis.expire(key, ttl)

            return new_value
        except Exception as e:
            self._logger.warning("redis_increment_usage_error", error=str(e))
            return await self._local_store.increment_usage(
                entity_id, resource, period, amount
            )

    async def reset_usage(
        self,
        entity_id: str,
        resource: str,
        period: QuotaPeriod,
    ) -> None:
        """Reset usage for a period"""
        if self._redis is None:
            return await self._local_store.reset_usage(entity_id, resource, period)

        try:
            key = self._make_usage_key(entity_id, resource, period)
            await self._redis.delete(key)
        except Exception as e:
            self._logger.warning("redis_reset_usage_error", error=str(e))
            await self._local_store.reset_usage(entity_id, resource, period)

    def _make_usage_key(
        self,
        entity_id: str,
        resource: str,
        period: QuotaPeriod,
    ) -> str:
        """Create usage key with period bucket"""
        now = datetime.utcnow()
        bucket = self._get_period_bucket(now, period)
        return f"{self._key_prefix}usage:{entity_id}:{resource}:{period.value}:{bucket}"

    def _get_period_bucket(self, dt: datetime, period: QuotaPeriod) -> str:
        """Get period bucket identifier"""
        if period == QuotaPeriod.MINUTE:
            return dt.strftime("%Y%m%d%H%M")
        elif period == QuotaPeriod.HOUR:
            return dt.strftime("%Y%m%d%H")
        elif period == QuotaPeriod.DAY:
            return dt.strftime("%Y%m%d")
        elif period == QuotaPeriod.WEEK:
            return dt.strftime("%Y%W")
        elif period == QuotaPeriod.MONTH:
            return dt.strftime("%Y%m")
        elif period == QuotaPeriod.YEAR:
            return dt.strftime("%Y")
        return "lifetime"

    def _get_period_ttl(self, period: QuotaPeriod) -> int:
        """Get TTL for period"""
        # Add buffer to TTL
        return period.seconds + 300

    def _serialize_quota(self, quota: Quota) -> Dict[str, Any]:
        """Serialize quota to dict"""
        return {
            "entity_id": quota.entity_id,
            "entity_type": quota.entity_type,
            "limits": [
                {
                    "resource": l.resource,
                    "limit": l.limit,
                    "period": l.period.value,
                    "description": l.description,
                    "soft_limit": l.soft_limit,
                    "overage_allowed": l.overage_allowed,
                    "overage_rate": l.overage_rate,
                }
                for l in quota.limits
            ],
            "metadata": quota.metadata,
            "created_at": quota.created_at.isoformat(),
            "updated_at": quota.updated_at.isoformat(),
        }

    def _deserialize_quota(self, data: Dict[str, Any]) -> Quota:
        """Deserialize quota from dict"""
        limits = [
            QuotaLimit(
                resource=l["resource"],
                limit=l["limit"],
                period=QuotaPeriod(l["period"]),
                description=l.get("description", ""),
                soft_limit=l.get("soft_limit"),
                overage_allowed=l.get("overage_allowed", False),
                overage_rate=l.get("overage_rate", 0.0),
            )
            for l in data.get("limits", [])
        ]

        return Quota(
            entity_id=data["entity_id"],
            entity_type=data.get("entity_type", "organization"),
            limits=limits,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class QuotaManager:
    """
    Manages quota tracking and enforcement.

    Usage:
        manager = QuotaManager(config, store)
        result = await manager.check("org_123", "api_calls")
        if not result.allowed:
            raise QuotaExceeded(...)
        await manager.increment("org_123", "api_calls")
    """

    def __init__(
        self,
        config: Optional[QuotaConfig] = None,
        store: Optional[QuotaStore] = None,
    ):
        self.config = config or QuotaConfig()
        self._store = store or InMemoryQuotaStore()
        self._logger = structlog.get_logger("quota_manager")
        self._alert_callbacks: List[Callable[[str, str, QuotaUsage], None]] = []

    async def get_quota(self, entity_id: str) -> Quota:
        """Get or create quota for an entity"""
        quota = await self._store.get_quota(entity_id)
        if quota is None:
            quota = Quota(
                entity_id=entity_id,
                limits=list(self.config.default_limits),
            )
            await self._store.set_quota(quota)
        return quota

    async def set_quota(self, quota: Quota) -> None:
        """Set quota for an entity"""
        quota.updated_at = datetime.utcnow()
        await self._store.set_quota(quota)

    async def check(
        self,
        entity_id: str,
        resource: str,
        amount: int = 1,
    ) -> QuotaUsage:
        """Check if quota allows the operation"""
        quota = await self.get_quota(entity_id)
        limit = quota.get_limit(resource)

        if limit is None:
            # No limit defined - allow
            return QuotaUsage(
                resource=resource,
                used=0,
                limit=0,
                period=QuotaPeriod.LIFETIME,
                period_start=datetime.utcnow(),
                period_end=datetime.utcnow(),
            )

        current_usage = await self._store.get_usage(
            entity_id, resource, limit.period
        )

        period_start, period_end = self._get_period_bounds(limit.period)

        usage = QuotaUsage(
            resource=resource,
            used=current_usage,
            limit=limit.limit,
            period=limit.period,
            period_start=period_start,
            period_end=period_end,
        )

        # Check soft limit
        if limit.soft_limit and current_usage >= limit.soft_limit:
            usage.soft_limit_reached = True
            if self.config.alert_on_soft_limit:
                await self._trigger_alert(entity_id, resource, usage)

        # Check hard limit
        if current_usage + amount > limit.limit:
            usage.hard_limit_reached = True

            if not limit.overage_allowed:
                raise QuotaExceeded(
                    message=f"Quota exceeded for {resource}",
                    resource=resource,
                    limit=limit.limit,
                    used=current_usage,
                    reset_at=period_end,
                )

        return usage

    async def increment(
        self,
        entity_id: str,
        resource: str,
        amount: int = 1,
    ) -> QuotaUsage:
        """Increment usage and return updated usage"""
        quota = await self.get_quota(entity_id)
        limit = quota.get_limit(resource)

        if limit is None:
            return QuotaUsage(
                resource=resource,
                used=amount,
                limit=0,
                period=QuotaPeriod.LIFETIME,
                period_start=datetime.utcnow(),
                period_end=datetime.utcnow(),
            )

        new_usage = await self._store.increment_usage(
            entity_id, resource, limit.period, amount
        )

        period_start, period_end = self._get_period_bounds(limit.period)

        usage = QuotaUsage(
            resource=resource,
            used=new_usage,
            limit=limit.limit,
            period=limit.period,
            period_start=period_start,
            period_end=period_end,
        )

        # Check limits
        if limit.soft_limit and new_usage >= limit.soft_limit:
            usage.soft_limit_reached = True
        if new_usage >= limit.limit:
            usage.hard_limit_reached = True

        return usage

    async def get_usage(
        self,
        entity_id: str,
        resource: Optional[str] = None,
    ) -> List[QuotaUsage]:
        """Get current usage for all or specific resources"""
        quota = await self.get_quota(entity_id)
        usages = []

        for limit in quota.limits:
            if resource and limit.resource != resource:
                continue

            current = await self._store.get_usage(
                entity_id, limit.resource, limit.period
            )
            period_start, period_end = self._get_period_bounds(limit.period)

            usage = QuotaUsage(
                resource=limit.resource,
                used=current,
                limit=limit.limit,
                period=limit.period,
                period_start=period_start,
                period_end=period_end,
            )

            if limit.soft_limit and current >= limit.soft_limit:
                usage.soft_limit_reached = True
            if current >= limit.limit:
                usage.hard_limit_reached = True

            usages.append(usage)

        return usages

    async def reset(
        self,
        entity_id: str,
        resource: str,
        period: Optional[QuotaPeriod] = None,
    ) -> None:
        """Reset usage for a resource"""
        quota = await self.get_quota(entity_id)
        limit = quota.get_limit(resource)

        if limit is None:
            return

        await self._store.reset_usage(
            entity_id, resource, period or limit.period
        )

    def on_soft_limit_alert(
        self,
        callback: Callable[[str, str, QuotaUsage], None],
    ) -> None:
        """Register callback for soft limit alerts"""
        self._alert_callbacks.append(callback)

    async def _trigger_alert(
        self,
        entity_id: str,
        resource: str,
        usage: QuotaUsage,
    ) -> None:
        """Trigger soft limit alert"""
        for callback in self._alert_callbacks:
            try:
                callback(entity_id, resource, usage)
            except Exception as e:
                self._logger.error(
                    "alert_callback_error",
                    error=str(e),
                )

    def _get_period_bounds(
        self,
        period: QuotaPeriod,
    ) -> Tuple[datetime, datetime]:
        """Get period start and end times"""
        now = datetime.utcnow()

        if period == QuotaPeriod.MINUTE:
            start = now.replace(second=0, microsecond=0)
            end = start + timedelta(minutes=1)
        elif period == QuotaPeriod.HOUR:
            start = now.replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
        elif period == QuotaPeriod.DAY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == QuotaPeriod.WEEK:
            days_since_monday = now.weekday()
            start = (now - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end = start + timedelta(weeks=1)
        elif period == QuotaPeriod.MONTH:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)
        elif period == QuotaPeriod.YEAR:
            start = now.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            end = start.replace(year=now.year + 1)
        else:
            start = datetime.min
            end = datetime.max

        return start, end


# Import for Tuple
from typing import Tuple
