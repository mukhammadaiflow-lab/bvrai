"""
Developer Platform Service Module

This module provides comprehensive developer platform services including
API key management, rate limiting, usage tracking, and webhook management.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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

from .base import (
    # Enums
    APIKeyStatus,
    APIKeyType,
    QuotaPeriod,
    RateLimitScope,
    RateLimitStrategy,
    SDKLanguage,
    WebhookEventType,
    # Types
    APIKey,
    APIKeyScope,
    APIUsageEvent,
    RateLimitConfig,
    RateLimitResult,
    RateLimitState,
    UsageQuota,
    WebhookEndpoint,
    # Exceptions
    APIKeyError,
    DeveloperPlatformError,
    QuotaExceededError,
    RateLimitError,
    WebhookError,
)


logger = logging.getLogger(__name__)


# =============================================================================
# API Key Manager
# =============================================================================


class APIKeyManager:
    """
    Manages API keys for organizations.

    Features:
    - Key creation with secure generation
    - Key rotation and revocation
    - Scope-based permissions
    - Usage tracking
    """

    def __init__(self):
        self._keys: Dict[str, APIKey] = {}
        self._keys_by_org: Dict[str, Set[str]] = defaultdict(set)
        self._key_prefix_index: Dict[str, str] = {}  # prefix -> key_id
        self._key_hash_index: Dict[str, str] = {}  # hash -> key_id

    async def create_key(
        self,
        organization_id: str,
        name: str,
        key_type: APIKeyType = APIKeyType.LIVE,
        scope: Optional[APIKeyScope] = None,
        description: str = "",
        created_by: Optional[str] = None,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[APIKey, str]:
        """
        Create a new API key.

        Returns:
            Tuple of (APIKey, full_key_string)
            Note: Full key is only returned once, store securely
        """
        # Generate key
        full_key, key_prefix, key_hash = APIKey.generate_key(key_type)

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create key record
        api_key = APIKey(
            id=f"key_{uuid.uuid4().hex[:24]}",
            organization_id=organization_id,
            name=name,
            key_prefix=key_prefix,
            key_hash=key_hash,
            key_type=key_type,
            scope=scope or APIKeyScope(),
            description=description,
            created_by=created_by,
            expires_at=expires_at,
        )

        # Store
        self._keys[api_key.id] = api_key
        self._keys_by_org[organization_id].add(api_key.id)
        self._key_prefix_index[key_prefix] = api_key.id
        self._key_hash_index[key_hash] = api_key.id

        logger.info(f"Created API key {api_key.id} for organization {organization_id}")

        return api_key, full_key

    async def get_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        return self._keys.get(key_id)

    async def get_key_by_value(self, key_value: str) -> Optional[APIKey]:
        """Get API key by its actual value."""
        key_hash = hashlib.sha256(key_value.encode()).hexdigest()
        key_id = self._key_hash_index.get(key_hash)
        if key_id:
            return self._keys.get(key_id)
        return None

    async def validate_key(
        self,
        key_value: str,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[bool, Optional[APIKey], Optional[str]]:
        """
        Validate an API key.

        Returns:
            Tuple of (is_valid, api_key, error_message)
        """
        api_key = await self.get_key_by_value(key_value)

        if not api_key:
            return False, None, "Invalid API key"

        if not api_key.is_valid():
            if api_key.status == APIKeyStatus.REVOKED:
                return False, api_key, "API key has been revoked"
            if api_key.status == APIKeyStatus.EXPIRED:
                return False, api_key, "API key has expired"
            return False, api_key, "API key is not active"

        # Check resource permission
        if resource and not api_key.scope.allows_resource(resource):
            return False, api_key, f"API key does not have access to {resource}"

        # Check action permission
        if action and not api_key.scope.allows_action(action):
            return False, api_key, f"API key cannot perform {action}"

        # Check IP restriction
        if ip_address and not api_key.scope.allows_ip(ip_address):
            return False, api_key, "IP address not allowed"

        # Update last used
        api_key.last_used_at = datetime.utcnow()
        api_key.last_used_ip = ip_address
        api_key.request_count += 1

        return True, api_key, None

    async def list_keys(
        self,
        organization_id: str,
        key_type: Optional[APIKeyType] = None,
        status: Optional[APIKeyStatus] = None,
        include_expired: bool = False,
    ) -> List[APIKey]:
        """List API keys for organization."""
        key_ids = self._keys_by_org.get(organization_id, set())
        keys = []

        for key_id in key_ids:
            key = self._keys.get(key_id)
            if not key:
                continue

            # Filter by type
            if key_type and key.key_type != key_type:
                continue

            # Filter by status
            if status and key.status != status:
                continue

            # Filter expired
            if not include_expired and key.expires_at:
                if key.expires_at < datetime.utcnow():
                    continue

            keys.append(key)

        return sorted(keys, key=lambda k: k.created_at, reverse=True)

    async def update_key(
        self,
        key_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scope: Optional[APIKeyScope] = None,
    ) -> APIKey:
        """Update API key metadata."""
        key = await self.get_key(key_id)
        if not key:
            raise APIKeyError(f"API key {key_id} not found")

        if name:
            key.name = name
        if description is not None:
            key.description = description
        if scope:
            key.scope = scope

        key.updated_at = datetime.utcnow()

        logger.info(f"Updated API key {key_id}")

        return key

    async def rotate_key(
        self,
        key_id: str,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[APIKey, str]:
        """
        Rotate an API key (generate new key value).

        Returns:
            Tuple of (updated_key, new_full_key_string)
        """
        old_key = await self.get_key(key_id)
        if not old_key:
            raise APIKeyError(f"API key {key_id} not found")

        # Generate new key value
        full_key, key_prefix, key_hash = APIKey.generate_key(old_key.key_type)

        # Remove old indexes
        old_hash_key = None
        for h, kid in self._key_hash_index.items():
            if kid == key_id:
                old_hash_key = h
                break
        if old_hash_key:
            del self._key_hash_index[old_hash_key]

        if old_key.key_prefix in self._key_prefix_index:
            del self._key_prefix_index[old_key.key_prefix]

        # Update key
        old_key.key_prefix = key_prefix
        old_key.key_hash = key_hash
        old_key.updated_at = datetime.utcnow()

        if expires_in_days:
            old_key.expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Add new indexes
        self._key_prefix_index[key_prefix] = key_id
        self._key_hash_index[key_hash] = key_id

        logger.info(f"Rotated API key {key_id}")

        return old_key, full_key

    async def revoke_key(self, key_id: str, reason: str = "") -> APIKey:
        """Revoke an API key."""
        key = await self.get_key(key_id)
        if not key:
            raise APIKeyError(f"API key {key_id} not found")

        key.status = APIKeyStatus.REVOKED
        key.revoked_at = datetime.utcnow()
        key.updated_at = datetime.utcnow()
        if reason:
            key.description = f"{key.description} | Revoked: {reason}"

        logger.info(f"Revoked API key {key_id}: {reason}")

        return key

    async def delete_key(self, key_id: str) -> bool:
        """Permanently delete an API key."""
        key = self._keys.get(key_id)
        if not key:
            return False

        # Remove from indexes
        del self._keys[key_id]
        self._keys_by_org[key.organization_id].discard(key_id)

        if key.key_prefix in self._key_prefix_index:
            del self._key_prefix_index[key.key_prefix]

        # Find and remove hash index
        for h, kid in list(self._key_hash_index.items()):
            if kid == key_id:
                del self._key_hash_index[h]
                break

        logger.info(f"Deleted API key {key_id}")

        return True

    async def record_error(self, key_id: str) -> None:
        """Record an error for the API key."""
        key = await self.get_key(key_id)
        if key:
            key.error_count += 1
            key.updated_at = datetime.utcnow()

    async def get_key_statistics(self, organization_id: str) -> Dict[str, Any]:
        """Get API key statistics for organization."""
        keys = await self.list_keys(organization_id, include_expired=True)

        active_count = sum(1 for k in keys if k.status == APIKeyStatus.ACTIVE)
        revoked_count = sum(1 for k in keys if k.status == APIKeyStatus.REVOKED)
        expired_count = sum(
            1 for k in keys
            if k.expires_at and k.expires_at < datetime.utcnow()
        )

        total_requests = sum(k.request_count for k in keys)
        total_errors = sum(k.error_count for k in keys)

        return {
            "total_keys": len(keys),
            "active_keys": active_count,
            "revoked_keys": revoked_count,
            "expired_keys": expired_count,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": total_errors / total_requests if total_requests > 0 else 0,
        }


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def check(
        self,
        key: str,
        config: RateLimitConfig,
    ) -> RateLimitResult:
        """Check if request is allowed."""
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for key."""
        pass


class FixedWindowRateLimiter(RateLimiter):
    """Fixed window rate limiter."""

    def __init__(self):
        self._windows: Dict[str, Dict[str, Any]] = {}

    async def check(
        self,
        key: str,
        config: RateLimitConfig,
    ) -> RateLimitResult:
        """Check rate limit using fixed window algorithm."""
        now = datetime.utcnow()
        window_key = f"{key}:minute"

        # Get or create window
        if window_key not in self._windows:
            self._windows[window_key] = {
                "count": 0,
                "window_start": now,
            }

        window = self._windows[window_key]

        # Check if window has expired
        window_duration = timedelta(minutes=1)
        if now - window["window_start"] >= window_duration:
            window["count"] = 0
            window["window_start"] = now

        # Check limit
        limit = config.requests_per_minute
        if window["count"] >= limit:
            reset_at = window["window_start"] + window_duration
            retry_after = int((reset_at - now).total_seconds())
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=limit,
                reset_at=reset_at,
                retry_after_seconds=max(1, retry_after),
            )

        # Increment counter
        window["count"] += 1
        remaining = limit - window["count"]
        reset_at = window["window_start"] + window_duration

        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            limit=limit,
            reset_at=reset_at,
        )

    async def reset(self, key: str) -> None:
        """Reset rate limit for key."""
        window_key = f"{key}:minute"
        if window_key in self._windows:
            del self._windows[window_key]


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter with sub-window precision."""

    def __init__(self, precision_seconds: int = 6):
        self._windows: Dict[str, List[Tuple[datetime, int]]] = defaultdict(list)
        self._precision = timedelta(seconds=precision_seconds)

    async def check(
        self,
        key: str,
        config: RateLimitConfig,
    ) -> RateLimitResult:
        """Check rate limit using sliding window algorithm."""
        now = datetime.utcnow()
        window_duration = timedelta(minutes=1)
        cutoff = now - window_duration

        # Clean old entries
        self._windows[key] = [
            (ts, count) for ts, count in self._windows[key]
            if ts > cutoff
        ]

        # Count total requests in window
        total_count = sum(count for _, count in self._windows[key])

        limit = config.requests_per_minute
        if total_count >= limit:
            # Find oldest entry to calculate reset time
            oldest = min((ts for ts, _ in self._windows[key]), default=now)
            reset_at = oldest + window_duration
            retry_after = int((reset_at - now).total_seconds())

            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=limit,
                reset_at=reset_at,
                retry_after_seconds=max(1, retry_after),
            )

        # Add current request
        current_bucket = now.replace(
            microsecond=0,
            second=(now.second // self._precision.seconds) * self._precision.seconds,
        )

        # Check if we can add to existing bucket
        bucket_found = False
        for i, (ts, count) in enumerate(self._windows[key]):
            if abs((ts - current_bucket).total_seconds()) < self._precision.seconds:
                self._windows[key][i] = (ts, count + 1)
                bucket_found = True
                break

        if not bucket_found:
            self._windows[key].append((current_bucket, 1))

        remaining = limit - total_count - 1
        reset_at = now + window_duration

        return RateLimitResult(
            allowed=True,
            remaining=max(0, remaining),
            limit=limit,
            reset_at=reset_at,
        )

    async def reset(self, key: str) -> None:
        """Reset rate limit for key."""
        if key in self._windows:
            del self._windows[key]


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter for burst handling."""

    def __init__(self):
        self._buckets: Dict[str, Dict[str, Any]] = {}

    async def check(
        self,
        key: str,
        config: RateLimitConfig,
    ) -> RateLimitResult:
        """Check rate limit using token bucket algorithm."""
        now = datetime.utcnow()

        # Get or create bucket
        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": float(config.burst_size),
                "last_update": now,
            }

        bucket = self._buckets[key]

        # Calculate token refill
        elapsed = (now - bucket["last_update"]).total_seconds()
        tokens_to_add = elapsed * config.burst_recovery_rate
        bucket["tokens"] = min(
            float(config.burst_size),
            bucket["tokens"] + tokens_to_add,
        )
        bucket["last_update"] = now

        # Check if we have tokens
        if bucket["tokens"] < 1.0:
            # Calculate when we'll have a token
            time_until_token = (1.0 - bucket["tokens"]) / config.burst_recovery_rate
            retry_after = int(time_until_token) + 1

            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=config.burst_size,
                retry_after_seconds=retry_after,
            )

        # Consume token
        bucket["tokens"] -= 1.0

        return RateLimitResult(
            allowed=True,
            remaining=int(bucket["tokens"]),
            limit=config.burst_size,
        )

    async def reset(self, key: str) -> None:
        """Reset rate limit for key."""
        if key in self._buckets:
            del self._buckets[key]


class RateLimitManager:
    """
    Manages rate limiting across different scopes and strategies.

    Features:
    - Multiple rate limiting strategies
    - Scope-based limiting (global, endpoint, IP, user)
    - Configurable limits per organization
    """

    def __init__(self):
        self._limiters: Dict[RateLimitStrategy, RateLimiter] = {
            RateLimitStrategy.FIXED_WINDOW: FixedWindowRateLimiter(),
            RateLimitStrategy.SLIDING_WINDOW: SlidingWindowRateLimiter(),
            RateLimitStrategy.TOKEN_BUCKET: TokenBucketRateLimiter(),
        }
        self._configs: Dict[str, RateLimitConfig] = {}
        self._default_config = RateLimitConfig()

    def set_config(
        self,
        organization_id: str,
        config: RateLimitConfig,
    ) -> None:
        """Set rate limit config for organization."""
        self._configs[organization_id] = config

    def get_config(self, organization_id: str) -> RateLimitConfig:
        """Get rate limit config for organization."""
        return self._configs.get(organization_id, self._default_config)

    def _get_rate_limit_key(
        self,
        api_key_id: str,
        scope: RateLimitScope,
        endpoint: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Generate rate limit key based on scope."""
        if scope == RateLimitScope.GLOBAL:
            return f"global:{api_key_id}"
        elif scope == RateLimitScope.ENDPOINT:
            return f"endpoint:{api_key_id}:{endpoint or 'unknown'}"
        elif scope == RateLimitScope.IP:
            return f"ip:{ip_address or 'unknown'}"
        elif scope == RateLimitScope.USER:
            return f"user:{user_id or 'unknown'}"
        return f"global:{api_key_id}"

    async def check_rate_limit(
        self,
        organization_id: str,
        api_key_id: str,
        scope: RateLimitScope = RateLimitScope.GLOBAL,
        endpoint: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_id: Optional[str] = None,
        custom_config: Optional[RateLimitConfig] = None,
    ) -> RateLimitResult:
        """
        Check if request is within rate limits.

        Args:
            organization_id: Organization ID
            api_key_id: API key ID
            scope: Rate limit scope
            endpoint: Endpoint being accessed
            ip_address: Client IP address
            user_id: User ID if authenticated
            custom_config: Override config for this check

        Returns:
            RateLimitResult with allow status and headers
        """
        config = custom_config or self.get_config(organization_id)
        key = self._get_rate_limit_key(
            api_key_id, scope, endpoint, ip_address, user_id
        )

        limiter = self._limiters.get(config.strategy)
        if not limiter:
            limiter = self._limiters[RateLimitStrategy.SLIDING_WINDOW]

        result = await limiter.check(key, config)

        if not result.allowed:
            logger.warning(
                f"Rate limit exceeded for {key}: "
                f"{result.remaining}/{result.limit}"
            )

        return result

    async def reset_rate_limit(
        self,
        api_key_id: str,
        scope: RateLimitScope = RateLimitScope.GLOBAL,
        endpoint: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Reset rate limit for a specific key."""
        key = self._get_rate_limit_key(
            api_key_id, scope, endpoint, ip_address, user_id
        )

        for limiter in self._limiters.values():
            await limiter.reset(key)

        logger.info(f"Reset rate limit for {key}")


# =============================================================================
# Usage Tracker
# =============================================================================


class UsageTracker:
    """
    Tracks API usage and manages quotas.

    Features:
    - Usage event logging
    - Quota management
    - Usage analytics
    - Alert triggering
    """

    def __init__(self):
        self._quotas: Dict[str, UsageQuota] = {}
        self._events: Dict[str, List[APIUsageEvent]] = defaultdict(list)
        self._alert_callbacks: List[Callable[[UsageQuota], None]] = []

    async def create_quota(
        self,
        organization_id: str,
        name: str,
        api_calls_limit: int = 100000,
        minutes_limit: int = 10000,
        storage_mb_limit: int = 10000,
        period: QuotaPeriod = QuotaPeriod.MONTH,
        alert_threshold_percent: int = 80,
    ) -> UsageQuota:
        """Create a usage quota for organization."""
        quota = UsageQuota(
            id=f"quota_{uuid.uuid4().hex[:24]}",
            organization_id=organization_id,
            name=name,
            api_calls_limit=api_calls_limit,
            minutes_limit=minutes_limit,
            storage_mb_limit=storage_mb_limit,
            period=period,
            alert_threshold_percent=alert_threshold_percent,
        )

        self._quotas[organization_id] = quota

        logger.info(f"Created quota {quota.id} for organization {organization_id}")

        return quota

    async def get_quota(self, organization_id: str) -> Optional[UsageQuota]:
        """Get quota for organization."""
        return self._quotas.get(organization_id)

    async def update_quota(
        self,
        organization_id: str,
        api_calls_limit: Optional[int] = None,
        minutes_limit: Optional[int] = None,
        storage_mb_limit: Optional[int] = None,
        alert_threshold_percent: Optional[int] = None,
    ) -> UsageQuota:
        """Update quota limits."""
        quota = self._quotas.get(organization_id)
        if not quota:
            raise QuotaExceededError(f"No quota found for {organization_id}")

        if api_calls_limit is not None:
            quota.api_calls_limit = api_calls_limit
        if minutes_limit is not None:
            quota.minutes_limit = minutes_limit
        if storage_mb_limit is not None:
            quota.storage_mb_limit = storage_mb_limit
        if alert_threshold_percent is not None:
            quota.alert_threshold_percent = alert_threshold_percent

        return quota

    async def track_api_call(
        self,
        organization_id: str,
        api_key_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float = 0.0,
        tokens_used: int = 0,
        bytes_sent: int = 0,
        bytes_received: int = 0,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> APIUsageEvent:
        """
        Track an API call.

        Returns:
            The created usage event

        Raises:
            QuotaExceededError if quota is exceeded
        """
        # Create event
        event = APIUsageEvent(
            id=f"usage_{uuid.uuid4().hex[:24]}",
            api_key_id=api_key_id,
            organization_id=organization_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
        )

        # Store event
        self._events[organization_id].append(event)

        # Update quota
        quota = self._quotas.get(organization_id)
        if quota:
            quota.api_calls_used += 1

            # Check quota
            if quota.api_calls_used > quota.api_calls_limit:
                raise QuotaExceededError(
                    f"API call quota exceeded for {organization_id}: "
                    f"{quota.api_calls_used}/{quota.api_calls_limit}"
                )

            # Check alert threshold
            if quota.should_alert():
                quota.alert_sent = True
                await self._trigger_alerts(quota)

        return event

    async def track_minutes(
        self,
        organization_id: str,
        minutes: float,
    ) -> None:
        """Track minutes usage."""
        quota = self._quotas.get(organization_id)
        if quota:
            quota.minutes_used += minutes

            if quota.minutes_used > quota.minutes_limit:
                raise QuotaExceededError(
                    f"Minutes quota exceeded for {organization_id}: "
                    f"{quota.minutes_used}/{quota.minutes_limit}"
                )

    async def track_storage(
        self,
        organization_id: str,
        storage_mb: float,
    ) -> None:
        """Track storage usage."""
        quota = self._quotas.get(organization_id)
        if quota:
            quota.storage_mb_used += storage_mb

            if quota.storage_mb_used > quota.storage_mb_limit:
                raise QuotaExceededError(
                    f"Storage quota exceeded for {organization_id}: "
                    f"{quota.storage_mb_used}/{quota.storage_mb_limit}"
                )

    async def check_quota(
        self,
        organization_id: str,
        api_calls: int = 1,
        minutes: float = 0.0,
        storage_mb: float = 0.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if usage would exceed quota.

        Returns:
            Tuple of (is_within_quota, error_message)
        """
        quota = self._quotas.get(organization_id)
        if not quota:
            return True, None

        if quota.api_calls_used + api_calls > quota.api_calls_limit:
            return False, "API call quota would be exceeded"

        if quota.minutes_used + minutes > quota.minutes_limit:
            return False, "Minutes quota would be exceeded"

        if quota.storage_mb_used + storage_mb > quota.storage_mb_limit:
            return False, "Storage quota would be exceeded"

        return True, None

    async def reset_quota(self, organization_id: str) -> Optional[UsageQuota]:
        """Reset quota for new period."""
        quota = self._quotas.get(organization_id)
        if quota:
            quota.api_calls_used = 0
            quota.minutes_used = 0.0
            quota.storage_mb_used = 0.0
            quota.period_start = datetime.utcnow()
            quota._set_period_end()
            quota.alert_sent = False

            # Clear old events
            self._events[organization_id] = []

            logger.info(f"Reset quota for {organization_id}")

        return quota

    def add_alert_callback(
        self,
        callback: Callable[[UsageQuota], None],
    ) -> None:
        """Add callback for quota alerts."""
        self._alert_callbacks.append(callback)

    async def _trigger_alerts(self, quota: UsageQuota) -> None:
        """Trigger alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(quota)
                else:
                    callback(quota)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    async def get_usage_summary(
        self,
        organization_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get usage summary for organization."""
        events = self._events.get(organization_id, [])

        # Filter by date
        if start_date:
            events = [e for e in events if e.request_time >= start_date]
        if end_date:
            events = [e for e in events if e.request_time <= end_date]

        if not events:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "total_bytes_sent": 0,
                "total_bytes_received": 0,
                "avg_response_time_ms": 0.0,
                "endpoints": {},
            }

        # Aggregate
        successful = [e for e in events if 200 <= e.status_code < 400]
        failed = [e for e in events if e.status_code >= 400]

        endpoint_counts: Dict[str, int] = defaultdict(int)
        for event in events:
            endpoint_counts[event.endpoint] += 1

        total_response_time = sum(e.response_time_ms for e in events)

        return {
            "total_requests": len(events),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / len(events) if events else 0,
            "total_tokens": sum(e.tokens_used for e in events),
            "total_bytes_sent": sum(e.bytes_sent for e in events),
            "total_bytes_received": sum(e.bytes_received for e in events),
            "avg_response_time_ms": total_response_time / len(events) if events else 0,
            "endpoints": dict(endpoint_counts),
        }

    async def get_usage_by_endpoint(
        self,
        organization_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get usage breakdown by endpoint."""
        events = self._events.get(organization_id, [])

        endpoint_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "success": 0,
            "failed": 0,
            "total_time_ms": 0.0,
        })

        for event in events:
            data = endpoint_data[event.endpoint]
            data["count"] += 1
            if 200 <= event.status_code < 400:
                data["success"] += 1
            else:
                data["failed"] += 1
            data["total_time_ms"] += event.response_time_ms

        result = []
        for endpoint, data in endpoint_data.items():
            result.append({
                "endpoint": endpoint,
                "count": data["count"],
                "success": data["success"],
                "failed": data["failed"],
                "avg_time_ms": data["total_time_ms"] / data["count"] if data["count"] > 0 else 0,
            })

        return sorted(result, key=lambda x: x["count"], reverse=True)[:limit]


# =============================================================================
# Webhook Manager
# =============================================================================


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""

    id: str
    webhook_id: str
    event_type: WebhookEventType
    payload: Dict[str, Any]

    # Delivery info
    url: str
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    response_time_ms: float = 0.0

    # Status
    success: bool = False
    error_message: Optional[str] = None
    attempt_number: int = 1

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    delivered_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"whd_{uuid.uuid4().hex[:24]}"


class WebhookManager:
    """
    Manages webhook endpoints and delivery.

    Features:
    - Endpoint registration and management
    - Event filtering
    - Secure payload signing
    - Retry with exponential backoff
    - Delivery tracking
    """

    def __init__(self):
        self._endpoints: Dict[str, WebhookEndpoint] = {}
        self._endpoints_by_org: Dict[str, Set[str]] = defaultdict(set)
        self._deliveries: Dict[str, List[WebhookDelivery]] = defaultdict(list)
        self._delivery_queue: List[Tuple[str, WebhookEventType, Dict[str, Any]]] = []
        self._http_client: Optional[Any] = None  # Will be set externally

    async def create_endpoint(
        self,
        organization_id: str,
        url: str,
        events: Optional[List[WebhookEventType]] = None,
        description: str = "",
        retry_count: int = 3,
        timeout_seconds: int = 30,
    ) -> WebhookEndpoint:
        """Create a webhook endpoint."""
        endpoint = WebhookEndpoint(
            id=f"wh_{uuid.uuid4().hex[:24]}",
            organization_id=organization_id,
            url=url,
            events=events or [],
            description=description,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
        )

        self._endpoints[endpoint.id] = endpoint
        self._endpoints_by_org[organization_id].add(endpoint.id)

        logger.info(f"Created webhook endpoint {endpoint.id} for {organization_id}")

        return endpoint

    async def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get webhook endpoint by ID."""
        return self._endpoints.get(endpoint_id)

    async def list_endpoints(
        self,
        organization_id: str,
        event_type: Optional[WebhookEventType] = None,
        enabled_only: bool = True,
    ) -> List[WebhookEndpoint]:
        """List webhook endpoints for organization."""
        endpoint_ids = self._endpoints_by_org.get(organization_id, set())
        endpoints = []

        for endpoint_id in endpoint_ids:
            endpoint = self._endpoints.get(endpoint_id)
            if not endpoint:
                continue

            if enabled_only and not endpoint.enabled:
                continue

            if event_type and not endpoint.should_receive(event_type):
                continue

            endpoints.append(endpoint)

        return endpoints

    async def update_endpoint(
        self,
        endpoint_id: str,
        url: Optional[str] = None,
        events: Optional[List[WebhookEventType]] = None,
        enabled: Optional[bool] = None,
        retry_count: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        description: Optional[str] = None,
    ) -> WebhookEndpoint:
        """Update webhook endpoint."""
        endpoint = await self.get_endpoint(endpoint_id)
        if not endpoint:
            raise WebhookError(f"Webhook endpoint {endpoint_id} not found")

        if url:
            endpoint.url = url
        if events is not None:
            endpoint.events = events
        if enabled is not None:
            endpoint.enabled = enabled
        if retry_count is not None:
            endpoint.retry_count = retry_count
        if timeout_seconds is not None:
            endpoint.timeout_seconds = timeout_seconds
        if description is not None:
            endpoint.description = description

        endpoint.updated_at = datetime.utcnow()

        return endpoint

    async def delete_endpoint(self, endpoint_id: str) -> bool:
        """Delete webhook endpoint."""
        endpoint = self._endpoints.get(endpoint_id)
        if not endpoint:
            return False

        del self._endpoints[endpoint_id]
        self._endpoints_by_org[endpoint.organization_id].discard(endpoint_id)

        logger.info(f"Deleted webhook endpoint {endpoint_id}")

        return True

    async def rotate_secret(self, endpoint_id: str) -> str:
        """Rotate webhook secret and return new secret."""
        endpoint = await self.get_endpoint(endpoint_id)
        if not endpoint:
            raise WebhookError(f"Webhook endpoint {endpoint_id} not found")

        new_secret = secrets.token_urlsafe(32)
        endpoint.secret = new_secret
        endpoint.updated_at = datetime.utcnow()

        logger.info(f"Rotated secret for webhook {endpoint_id}")

        return new_secret

    async def dispatch_event(
        self,
        organization_id: str,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
    ) -> List[WebhookDelivery]:
        """
        Dispatch event to all matching webhooks.

        Returns:
            List of delivery records
        """
        endpoints = await self.list_endpoints(
            organization_id,
            event_type=event_type,
            enabled_only=True,
        )

        deliveries = []
        for endpoint in endpoints:
            delivery = await self._deliver_webhook(endpoint, event_type, payload)
            deliveries.append(delivery)

        return deliveries

    async def _deliver_webhook(
        self,
        endpoint: WebhookEndpoint,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
        attempt: int = 1,
    ) -> WebhookDelivery:
        """Deliver webhook to endpoint with retry logic."""
        # Add event metadata
        full_payload = {
            "id": f"evt_{uuid.uuid4().hex[:24]}",
            "type": event_type.value,
            "created_at": datetime.utcnow().isoformat(),
            "data": payload,
        }

        payload_str = json.dumps(full_payload)
        timestamp = int(time.time())
        signature = endpoint.sign_payload(payload_str, timestamp)

        delivery = WebhookDelivery(
            id=f"whd_{uuid.uuid4().hex[:24]}",
            webhook_id=endpoint.id,
            event_type=event_type,
            payload=full_payload,
            url=endpoint.url,
            attempt_number=attempt,
        )

        # Simulate delivery (in real implementation, use HTTP client)
        start_time = time.time()

        try:
            # In real implementation:
            # response = await self._http_client.post(
            #     endpoint.url,
            #     json=full_payload,
            #     headers={
            #         endpoint.signature_header: signature,
            #         "Content-Type": "application/json",
            #     },
            #     timeout=endpoint.timeout_seconds,
            # )

            # Simulate success
            delivery.success = True
            delivery.status_code = 200
            delivery.delivered_at = datetime.utcnow()

            # Update endpoint status
            endpoint.last_delivery_at = datetime.utcnow()
            endpoint.last_delivery_status = "success"
            endpoint.consecutive_failures = 0

            logger.info(
                f"Webhook delivered to {endpoint.url} for event {event_type.value}"
            )

        except Exception as e:
            delivery.success = False
            delivery.error_message = str(e)

            endpoint.consecutive_failures += 1
            endpoint.last_delivery_status = "failed"

            # Retry if within limits
            if attempt < endpoint.retry_count:
                delay = endpoint.retry_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    f"Webhook delivery failed, retry {attempt + 1} in {delay}s: {e}"
                )
                await asyncio.sleep(delay)
                return await self._deliver_webhook(
                    endpoint, event_type, payload, attempt + 1
                )

            logger.error(f"Webhook delivery failed after {attempt} attempts: {e}")

        delivery.response_time_ms = (time.time() - start_time) * 1000

        # Store delivery record
        self._deliveries[endpoint.id].append(delivery)

        return delivery

    async def get_delivery_history(
        self,
        endpoint_id: str,
        limit: int = 50,
        success_only: bool = False,
    ) -> List[WebhookDelivery]:
        """Get delivery history for endpoint."""
        deliveries = self._deliveries.get(endpoint_id, [])

        if success_only:
            deliveries = [d for d in deliveries if d.success]

        return sorted(
            deliveries,
            key=lambda d: d.created_at,
            reverse=True,
        )[:limit]

    async def test_endpoint(
        self,
        endpoint_id: str,
    ) -> WebhookDelivery:
        """Send test event to endpoint."""
        endpoint = await self.get_endpoint(endpoint_id)
        if not endpoint:
            raise WebhookError(f"Webhook endpoint {endpoint_id} not found")

        test_payload = {
            "message": "This is a test webhook",
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Temporarily enable endpoint for test
        was_enabled = endpoint.enabled
        endpoint.enabled = True

        try:
            delivery = await self._deliver_webhook(
                endpoint,
                WebhookEventType.CALL_STARTED,  # Use generic event for test
                test_payload,
            )
        finally:
            endpoint.enabled = was_enabled

        return delivery

    async def get_webhook_statistics(
        self,
        organization_id: str,
    ) -> Dict[str, Any]:
        """Get webhook statistics for organization."""
        endpoints = await self.list_endpoints(organization_id, enabled_only=False)

        total_deliveries = 0
        successful_deliveries = 0
        failed_deliveries = 0

        for endpoint in endpoints:
            deliveries = self._deliveries.get(endpoint.id, [])
            total_deliveries += len(deliveries)
            successful_deliveries += sum(1 for d in deliveries if d.success)
            failed_deliveries += sum(1 for d in deliveries if not d.success)

        return {
            "total_endpoints": len(endpoints),
            "enabled_endpoints": sum(1 for e in endpoints if e.enabled),
            "total_deliveries": total_deliveries,
            "successful_deliveries": successful_deliveries,
            "failed_deliveries": failed_deliveries,
            "success_rate": successful_deliveries / total_deliveries if total_deliveries > 0 else 0,
            "endpoints_with_failures": sum(
                1 for e in endpoints if e.consecutive_failures > 0
            ),
        }


# =============================================================================
# SDK Code Generator
# =============================================================================


class SDKCodeGenerator:
    """
    Generates SDK code snippets for various languages.

    Features:
    - Multi-language support
    - API endpoint code generation
    - Authentication examples
    - Error handling patterns
    """

    def __init__(self, base_url: str = "https://api.voiceagent.ai/v1"):
        self.base_url = base_url

    def generate_auth_snippet(
        self,
        language: SDKLanguage,
        api_key: str = "YOUR_API_KEY",
    ) -> str:
        """Generate authentication code snippet."""
        snippets = {
            SDKLanguage.PYTHON: f'''
import voiceagent

client = voiceagent.Client(api_key="{api_key}")
''',
            SDKLanguage.JAVASCRIPT: f'''
const VoiceAgent = require('voiceagent');

const client = new VoiceAgent('{api_key}');
''',
            SDKLanguage.TYPESCRIPT: f'''
import VoiceAgent from 'voiceagent';

const client = new VoiceAgent('{api_key}');
''',
            SDKLanguage.GO: f'''
package main

import "github.com/voiceagent/go-sdk"

func main() {{
    client := voiceagent.NewClient("{api_key}")
}}
''',
            SDKLanguage.JAVA: f'''
import com.voiceagent.VoiceAgentClient;

VoiceAgentClient client = new VoiceAgentClient("{api_key}");
''',
            SDKLanguage.RUBY: f'''
require 'voiceagent'

client = VoiceAgent::Client.new(api_key: '{api_key}')
''',
            SDKLanguage.PHP: f'''
<?php
require 'vendor/autoload.php';

$client = new VoiceAgent\\Client('{api_key}');
''',
            SDKLanguage.CSHARP: f'''
using VoiceAgent;

var client = new VoiceAgentClient("{api_key}");
''',
        }

        return snippets.get(language, "// Language not supported").strip()

    def generate_create_call_snippet(
        self,
        language: SDKLanguage,
    ) -> str:
        """Generate create call code snippet."""
        snippets = {
            SDKLanguage.PYTHON: '''
# Create an outbound call
call = client.calls.create(
    to="+15551234567",
    agent_id="agent_xxx",
    context={
        "customer_name": "John Doe",
        "appointment_date": "2024-01-15"
    }
)

print(f"Call started: {call.id}")
''',
            SDKLanguage.JAVASCRIPT: '''
// Create an outbound call
const call = await client.calls.create({
  to: '+15551234567',
  agentId: 'agent_xxx',
  context: {
    customerName: 'John Doe',
    appointmentDate: '2024-01-15'
  }
});

console.log(`Call started: ${call.id}`);
''',
            SDKLanguage.TYPESCRIPT: '''
// Create an outbound call
const call = await client.calls.create({
  to: '+15551234567',
  agentId: 'agent_xxx',
  context: {
    customerName: 'John Doe',
    appointmentDate: '2024-01-15'
  }
});

console.log(`Call started: ${call.id}`);
''',
            SDKLanguage.GO: '''
// Create an outbound call
call, err := client.Calls.Create(&voiceagent.CreateCallParams{
    To:      "+15551234567",
    AgentID: "agent_xxx",
    Context: map[string]interface{}{
        "customer_name":    "John Doe",
        "appointment_date": "2024-01-15",
    },
})
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Call started: %s\\n", call.ID)
''',
            SDKLanguage.JAVA: '''
// Create an outbound call
Map<String, Object> context = new HashMap<>();
context.put("customer_name", "John Doe");
context.put("appointment_date", "2024-01-15");

Call call = client.calls().create(
    CreateCallParams.builder()
        .to("+15551234567")
        .agentId("agent_xxx")
        .context(context)
        .build()
);

System.out.println("Call started: " + call.getId());
''',
        }

        return snippets.get(language, "// See documentation for this language").strip()

    def generate_webhook_handler_snippet(
        self,
        language: SDKLanguage,
    ) -> str:
        """Generate webhook handler code snippet."""
        snippets = {
            SDKLanguage.PYTHON: '''
from flask import Flask, request
import voiceagent

app = Flask(__name__)
webhook_secret = "your_webhook_secret"

@app.route("/webhooks", methods=["POST"])
def handle_webhook():
    payload = request.data
    signature = request.headers.get("X-Webhook-Signature")

    # Verify webhook signature
    if not voiceagent.Webhook.verify(payload, signature, webhook_secret):
        return "Invalid signature", 401

    event = request.json

    if event["type"] == "call.ended":
        call_data = event["data"]
        print(f"Call ended: {call_data['id']}, duration: {call_data['duration']}")

    return "OK", 200
''',
            SDKLanguage.JAVASCRIPT: '''
const express = require('express');
const VoiceAgent = require('voiceagent');

const app = express();
const webhookSecret = 'your_webhook_secret';

app.post('/webhooks', express.raw({type: 'application/json'}), (req, res) => {
  const signature = req.headers['x-webhook-signature'];

  // Verify webhook signature
  if (!VoiceAgent.Webhook.verify(req.body, signature, webhookSecret)) {
    return res.status(401).send('Invalid signature');
  }

  const event = JSON.parse(req.body);

  if (event.type === 'call.ended') {
    const callData = event.data;
    console.log(`Call ended: ${callData.id}, duration: ${callData.duration}`);
  }

  res.send('OK');
});
''',
            SDKLanguage.GO: '''
package main

import (
    "encoding/json"
    "net/http"
    "github.com/voiceagent/go-sdk"
)

func webhookHandler(w http.ResponseWriter, r *http.Request) {
    payload, _ := ioutil.ReadAll(r.Body)
    signature := r.Header.Get("X-Webhook-Signature")
    webhookSecret := "your_webhook_secret"

    // Verify webhook signature
    if !voiceagent.VerifyWebhook(payload, signature, webhookSecret) {
        http.Error(w, "Invalid signature", http.StatusUnauthorized)
        return
    }

    var event map[string]interface{}
    json.Unmarshal(payload, &event)

    if event["type"] == "call.ended" {
        callData := event["data"].(map[string]interface{})
        fmt.Printf("Call ended: %s, duration: %v\\n", callData["id"], callData["duration"])
    }

    w.WriteHeader(http.StatusOK)
}
''',
        }

        return snippets.get(language, "// See documentation for this language").strip()

    def generate_curl_example(
        self,
        endpoint: str,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate cURL example."""
        curl = f"curl -X {method} \\\n"
        curl += f"  {self.base_url}{endpoint} \\\n"
        curl += '  -H "Authorization: Bearer YOUR_API_KEY" \\\n'
        curl += '  -H "Content-Type: application/json"'

        if body:
            curl += f" \\\n  -d '{json.dumps(body, indent=2)}'"

        return curl


# =============================================================================
# Developer Platform Service
# =============================================================================


class DeveloperPlatformService:
    """
    Unified service for developer platform features.

    Provides:
    - API key management
    - Rate limiting
    - Usage tracking
    - Webhook management
    - SDK code generation
    """

    def __init__(self, base_url: str = "https://api.voiceagent.ai/v1"):
        self.api_keys = APIKeyManager()
        self.rate_limiter = RateLimitManager()
        self.usage = UsageTracker()
        self.webhooks = WebhookManager()
        self.sdk_generator = SDKCodeGenerator(base_url)

    async def authenticate_request(
        self,
        api_key_value: str,
        resource: str,
        action: str,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Tuple[bool, Optional[APIKey], Optional[str]]:
        """
        Authenticate and authorize an API request.

        Performs:
        1. API key validation
        2. Rate limit check
        3. Quota check

        Returns:
            Tuple of (is_allowed, api_key, error_message)
        """
        # Validate API key
        valid, api_key, error = await self.api_keys.validate_key(
            api_key_value,
            resource=resource,
            action=action,
            ip_address=ip_address,
        )

        if not valid:
            return False, api_key, error

        # Check rate limit
        rate_result = await self.rate_limiter.check_rate_limit(
            api_key.organization_id,
            api_key.id,
            endpoint=endpoint,
            ip_address=ip_address,
        )

        if not rate_result.allowed:
            return False, api_key, f"Rate limit exceeded. Retry after {rate_result.retry_after_seconds}s"

        # Check quota
        within_quota, quota_error = await self.usage.check_quota(
            api_key.organization_id,
            api_calls=1,
        )

        if not within_quota:
            return False, api_key, quota_error

        return True, api_key, None

    async def record_request(
        self,
        api_key: APIKey,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float = 0.0,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> APIUsageEvent:
        """Record an API request."""
        return await self.usage.track_api_call(
            organization_id=api_key.organization_id,
            api_key_id=api_key.id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            ip_address=ip_address,
            user_agent=user_agent,
        )

    async def dispatch_webhook_event(
        self,
        organization_id: str,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
    ) -> List[WebhookDelivery]:
        """Dispatch webhook event to organization's endpoints."""
        return await self.webhooks.dispatch_event(
            organization_id,
            event_type,
            payload,
        )

    async def get_developer_dashboard(
        self,
        organization_id: str,
    ) -> Dict[str, Any]:
        """Get comprehensive developer dashboard data."""
        api_key_stats = await self.api_keys.get_key_statistics(organization_id)
        usage_summary = await self.usage.get_usage_summary(organization_id)
        webhook_stats = await self.webhooks.get_webhook_statistics(organization_id)
        quota = await self.usage.get_quota(organization_id)

        return {
            "api_keys": api_key_stats,
            "usage": usage_summary,
            "webhooks": webhook_stats,
            "quota": quota.to_dict() if quota else None,
            "rate_limit_config": self.rate_limiter.get_config(organization_id).to_dict(),
        }

    async def setup_organization(
        self,
        organization_id: str,
        plan_name: str = "starter",
    ) -> Dict[str, Any]:
        """
        Set up developer platform for a new organization.

        Creates default quota, rate limits, and initial API key.
        """
        # Define plan limits
        plans = {
            "starter": {
                "api_calls_limit": 10000,
                "minutes_limit": 1000,
                "storage_mb_limit": 1000,
                "rate_limit_per_minute": 60,
            },
            "professional": {
                "api_calls_limit": 100000,
                "minutes_limit": 10000,
                "storage_mb_limit": 10000,
                "rate_limit_per_minute": 300,
            },
            "enterprise": {
                "api_calls_limit": 1000000,
                "minutes_limit": 100000,
                "storage_mb_limit": 100000,
                "rate_limit_per_minute": 1000,
            },
        }

        plan = plans.get(plan_name, plans["starter"])

        # Create quota
        quota = await self.usage.create_quota(
            organization_id=organization_id,
            name=f"{plan_name.title()} Plan",
            api_calls_limit=plan["api_calls_limit"],
            minutes_limit=plan["minutes_limit"],
            storage_mb_limit=plan["storage_mb_limit"],
        )

        # Set rate limits
        self.rate_limiter.set_config(
            organization_id,
            RateLimitConfig(
                requests_per_minute=plan["rate_limit_per_minute"],
                requests_per_hour=plan["rate_limit_per_minute"] * 60,
                requests_per_day=plan["rate_limit_per_minute"] * 60 * 24,
            ),
        )

        # Create initial API key
        api_key, full_key = await self.api_keys.create_key(
            organization_id=organization_id,
            name="Default API Key",
            key_type=APIKeyType.LIVE,
            description="Auto-generated key",
        )

        logger.info(f"Set up developer platform for organization {organization_id}")

        return {
            "quota": quota.to_dict(),
            "api_key": api_key.to_dict(),
            "full_key": full_key,  # Only returned once!
            "plan": plan_name,
        }

    async def upgrade_plan(
        self,
        organization_id: str,
        new_plan: str,
    ) -> Dict[str, Any]:
        """Upgrade organization to new plan."""
        plans = {
            "starter": {
                "api_calls_limit": 10000,
                "minutes_limit": 1000,
                "storage_mb_limit": 1000,
                "rate_limit_per_minute": 60,
            },
            "professional": {
                "api_calls_limit": 100000,
                "minutes_limit": 10000,
                "storage_mb_limit": 10000,
                "rate_limit_per_minute": 300,
            },
            "enterprise": {
                "api_calls_limit": 1000000,
                "minutes_limit": 100000,
                "storage_mb_limit": 100000,
                "rate_limit_per_minute": 1000,
            },
        }

        plan = plans.get(new_plan)
        if not plan:
            raise DeveloperPlatformError(f"Invalid plan: {new_plan}")

        # Update quota
        quota = await self.usage.update_quota(
            organization_id,
            api_calls_limit=plan["api_calls_limit"],
            minutes_limit=plan["minutes_limit"],
            storage_mb_limit=plan["storage_mb_limit"],
        )

        # Update rate limits
        self.rate_limiter.set_config(
            organization_id,
            RateLimitConfig(
                requests_per_minute=plan["rate_limit_per_minute"],
                requests_per_hour=plan["rate_limit_per_minute"] * 60,
                requests_per_day=plan["rate_limit_per_minute"] * 60 * 24,
            ),
        )

        logger.info(f"Upgraded {organization_id} to {new_plan} plan")

        return {
            "plan": new_plan,
            "quota": quota.to_dict(),
            "rate_limit": self.rate_limiter.get_config(organization_id).to_dict(),
        }


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # API Key Manager
    "APIKeyManager",
    # Rate Limiters
    "RateLimiter",
    "FixedWindowRateLimiter",
    "SlidingWindowRateLimiter",
    "TokenBucketRateLimiter",
    "RateLimitManager",
    # Usage Tracker
    "UsageTracker",
    # Webhook Manager
    "WebhookDelivery",
    "WebhookManager",
    # SDK Generator
    "SDKCodeGenerator",
    # Main Service
    "DeveloperPlatformService",
]
