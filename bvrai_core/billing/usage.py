"""
Usage Tracking System

Real-time usage metering and tracking for billing.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .base import (
    BillingAlert,
    BillingError,
    PlanFeatures,
    Subscription,
    UsageLimitError,
    UsageRecord,
    UsageStore,
    UsageSummary,
    UsageType,
)


logger = logging.getLogger(__name__)


@dataclass
class UsageQuota:
    """Usage quota configuration."""

    usage_type: UsageType
    limit: int
    period_type: str = "billing_cycle"  # billing_cycle, daily, monthly
    soft_limit_percent: int = 80  # Alert at 80%
    hard_limit_action: str = "block"  # block, throttle, allow


@dataclass
class UsageThreshold:
    """Usage threshold for alerts."""

    usage_type: UsageType
    threshold_percent: int
    triggered: bool = False
    triggered_at: Optional[datetime] = None


class InMemoryUsageStore(UsageStore):
    """In-memory usage store implementation."""

    def __init__(self):
        """Initialize in-memory store."""
        self._records: Dict[str, List[UsageRecord]] = defaultdict(list)
        self._idempotency_keys: Set[str] = set()

    async def record_usage(self, record: UsageRecord) -> None:
        """Record usage event."""
        # Check idempotency
        if record.idempotency_key:
            if record.idempotency_key in self._idempotency_keys:
                logger.debug(f"Duplicate usage record: {record.idempotency_key}")
                return
            self._idempotency_keys.add(record.idempotency_key)

        self._records[record.organization_id].append(record)

    async def get_usage(
        self,
        organization_id: str,
        usage_type: UsageType,
        start_time: datetime,
        end_time: datetime,
    ) -> List[UsageRecord]:
        """Get usage records for a period."""
        records = self._records.get(organization_id, [])

        return [
            r for r in records
            if r.usage_type == usage_type
            and start_time <= r.timestamp <= end_time
        ]

    async def get_usage_summary(
        self,
        organization_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> UsageSummary:
        """Get aggregated usage summary."""
        records = self._records.get(organization_id, [])

        usage_by_type: Dict[UsageType, Decimal] = defaultdict(Decimal)
        cost_by_type: Dict[UsageType, int] = defaultdict(int)

        for record in records:
            if start_time <= record.timestamp <= end_time:
                usage_by_type[record.usage_type] += record.quantity
                if record.total_cents:
                    cost_by_type[record.usage_type] += record.total_cents

        return UsageSummary(
            organization_id=organization_id,
            period_start=start_time,
            period_end=end_time,
            usage_by_type=dict(usage_by_type),
            cost_by_type=dict(cost_by_type),
            total_cost_cents=sum(cost_by_type.values()),
        )

    async def mark_as_billed(
        self,
        organization_id: str,
        invoice_id: str,
        end_time: datetime,
    ) -> int:
        """Mark usage records as billed."""
        records = self._records.get(organization_id, [])
        count = 0

        for record in records:
            if not record.billed and record.timestamp <= end_time:
                record.billed = True
                record.invoice_id = invoice_id
                count += 1

        return count

    async def get_unbilled_usage(
        self,
        organization_id: str,
    ) -> List[UsageRecord]:
        """Get unbilled usage records."""
        records = self._records.get(organization_id, [])
        return [r for r in records if not r.billed]


class UsageMeter:
    """
    Real-time usage metering with buffering.

    Buffers usage events and flushes them periodically for efficiency.
    """

    def __init__(
        self,
        store: UsageStore,
        flush_interval_seconds: float = 10.0,
        max_buffer_size: int = 1000,
    ):
        """Initialize usage meter."""
        self._store = store
        self._flush_interval = flush_interval_seconds
        self._max_buffer_size = max_buffer_size

        self._buffer: List[UsageRecord] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

        # Current period usage cache
        self._period_usage: Dict[str, Dict[UsageType, Decimal]] = defaultdict(
            lambda: defaultdict(Decimal)
        )
        self._usage_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the usage meter."""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Usage meter started")

    async def stop(self) -> None:
        """Stop the usage meter."""
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_buffer()
        logger.info("Usage meter stopped")

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")

    async def _flush_buffer(self) -> None:
        """Flush buffered records to store."""
        async with self._buffer_lock:
            if not self._buffer:
                return

            records = self._buffer
            self._buffer = []

        for record in records:
            try:
                await self._store.record_usage(record)
            except Exception as e:
                logger.error(f"Error storing usage record: {e}")

    async def record(
        self,
        organization_id: str,
        usage_type: UsageType,
        quantity: Decimal,
        call_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """Record a usage event."""
        import uuid

        record = UsageRecord(
            id=f"usage_{uuid.uuid4().hex[:16]}",
            organization_id=organization_id,
            usage_type=usage_type,
            quantity=quantity,
            timestamp=datetime.utcnow(),
            idempotency_key=idempotency_key,
            call_id=call_id,
            agent_id=agent_id,
            metadata=metadata or {},
        )

        # Update period cache
        async with self._usage_lock:
            self._period_usage[organization_id][usage_type] += quantity

        # Add to buffer
        async with self._buffer_lock:
            self._buffer.append(record)

            # Flush if buffer is full
            if len(self._buffer) >= self._max_buffer_size:
                await self._flush_buffer()

        return record

    async def get_current_usage(
        self,
        organization_id: str,
        usage_type: UsageType,
    ) -> Decimal:
        """Get current usage from cache."""
        async with self._usage_lock:
            return self._period_usage[organization_id][usage_type]

    async def get_all_current_usage(
        self,
        organization_id: str,
    ) -> Dict[UsageType, Decimal]:
        """Get all current usage for organization."""
        async with self._usage_lock:
            return dict(self._period_usage[organization_id])

    async def reset_period(self, organization_id: str) -> None:
        """Reset usage for a new billing period."""
        async with self._usage_lock:
            self._period_usage[organization_id] = defaultdict(Decimal)


class UsageTracker:
    """
    Complete usage tracking with quota enforcement.

    Tracks usage, enforces quotas, and generates alerts.
    """

    def __init__(
        self,
        meter: UsageMeter,
        store: UsageStore,
    ):
        """Initialize usage tracker."""
        self._meter = meter
        self._store = store

        # Quotas by organization
        self._quotas: Dict[str, Dict[UsageType, UsageQuota]] = {}

        # Thresholds for alerts
        self._thresholds: Dict[str, Dict[UsageType, UsageThreshold]] = {}

        # Alert callbacks
        self._alert_callbacks: List[Callable] = []

        # Current limits from subscriptions
        self._limits: Dict[str, PlanFeatures] = {}

    def set_organization_limits(
        self,
        organization_id: str,
        features: PlanFeatures,
    ) -> None:
        """Set usage limits for an organization based on plan features."""
        self._limits[organization_id] = features

        # Create quotas from features
        quotas = {
            UsageType.AGENTS: UsageQuota(
                usage_type=UsageType.AGENTS,
                limit=features.max_agents,
            ),
            UsageType.CONCURRENT_CALLS: UsageQuota(
                usage_type=UsageType.CONCURRENT_CALLS,
                limit=features.max_concurrent_calls,
            ),
            UsageType.PHONE_NUMBERS: UsageQuota(
                usage_type=UsageType.PHONE_NUMBERS,
                limit=features.max_phone_numbers,
            ),
            UsageType.CALL_MINUTES: UsageQuota(
                usage_type=UsageType.CALL_MINUTES,
                limit=features.included_call_minutes,
                hard_limit_action="allow",  # Overage billing
            ),
            UsageType.SMS_MESSAGES: UsageQuota(
                usage_type=UsageType.SMS_MESSAGES,
                limit=features.included_sms_messages,
                hard_limit_action="allow",
            ),
            UsageType.API_REQUESTS: UsageQuota(
                usage_type=UsageType.API_REQUESTS,
                limit=features.included_api_requests,
                hard_limit_action="allow",
            ),
        }

        self._quotas[organization_id] = quotas

        # Initialize thresholds
        self._thresholds[organization_id] = {
            usage_type: UsageThreshold(
                usage_type=usage_type,
                threshold_percent=quota.soft_limit_percent,
            )
            for usage_type, quota in quotas.items()
        }

    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback for usage alerts."""
        self._alert_callbacks.append(callback)

    async def _trigger_alert(
        self,
        organization_id: str,
        usage_type: UsageType,
        current: Decimal,
        limit: int,
        threshold_percent: int,
    ) -> None:
        """Trigger usage alert."""
        alert = BillingAlert(
            id=f"alert_{datetime.utcnow().timestamp()}",
            organization_id=organization_id,
            alert_type="usage_threshold",
            threshold_value=threshold_percent,
            threshold_type=usage_type,
        )

        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert, current, limit)
                else:
                    callback(alert, current, limit)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    async def check_quota(
        self,
        organization_id: str,
        usage_type: UsageType,
        requested_quantity: Decimal = Decimal("1"),
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if usage is within quota.

        Returns:
            Tuple of (allowed, error_message)
        """
        quotas = self._quotas.get(organization_id)
        if not quotas:
            return (True, None)

        quota = quotas.get(usage_type)
        if not quota:
            return (True, None)

        current = await self._meter.get_current_usage(organization_id, usage_type)
        projected = current + requested_quantity

        # Check hard limit
        if projected > quota.limit:
            if quota.hard_limit_action == "block":
                return (
                    False,
                    f"Usage limit exceeded for {usage_type.value}: {int(projected)}/{quota.limit}"
                )

        # Check soft limit for alerts
        current_percent = int((current / quota.limit) * 100) if quota.limit > 0 else 0
        threshold = self._thresholds.get(organization_id, {}).get(usage_type)

        if threshold and not threshold.triggered:
            if current_percent >= threshold.threshold_percent:
                threshold.triggered = True
                threshold.triggered_at = datetime.utcnow()
                await self._trigger_alert(
                    organization_id,
                    usage_type,
                    current,
                    quota.limit,
                    threshold.threshold_percent,
                )

        return (True, None)

    async def record_usage(
        self,
        organization_id: str,
        usage_type: UsageType,
        quantity: Decimal,
        call_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        enforce_quota: bool = True,
    ) -> UsageRecord:
        """Record usage with quota check."""
        if enforce_quota:
            allowed, error = await self.check_quota(organization_id, usage_type, quantity)
            if not allowed:
                raise UsageLimitError(
                    usage_type=usage_type,
                    limit=self._quotas.get(organization_id, {}).get(usage_type, UsageQuota(usage_type, 0)).limit,
                    current=int(await self._meter.get_current_usage(organization_id, usage_type)),
                )

        return await self._meter.record(
            organization_id=organization_id,
            usage_type=usage_type,
            quantity=quantity,
            call_id=call_id,
            agent_id=agent_id,
            idempotency_key=idempotency_key,
            metadata=metadata,
        )

    async def get_usage_summary(
        self,
        organization_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> UsageSummary:
        """Get usage summary for organization."""
        if not start_time:
            start_time = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if not end_time:
            end_time = datetime.utcnow()

        return await self._store.get_usage_summary(organization_id, start_time, end_time)

    async def get_quota_status(
        self,
        organization_id: str,
    ) -> Dict[UsageType, Dict[str, Any]]:
        """Get current quota status for all usage types."""
        quotas = self._quotas.get(organization_id, {})
        status = {}

        for usage_type, quota in quotas.items():
            current = await self._meter.get_current_usage(organization_id, usage_type)
            status[usage_type] = {
                "current": int(current),
                "limit": quota.limit,
                "percent_used": int((current / quota.limit) * 100) if quota.limit > 0 else 0,
                "remaining": max(0, quota.limit - int(current)),
                "soft_limit_percent": quota.soft_limit_percent,
                "hard_limit_action": quota.hard_limit_action,
            }

        return status


class CallUsageTracker:
    """Specialized tracker for call-related usage."""

    def __init__(self, tracker: UsageTracker):
        """Initialize call usage tracker."""
        self._tracker = tracker
        self._active_calls: Dict[str, Dict[str, Any]] = {}

    async def start_call(
        self,
        call_id: str,
        organization_id: str,
        agent_id: str,
        direction: str = "inbound",
    ) -> None:
        """Record call start."""
        # Check concurrent call limit
        allowed, error = await self._tracker.check_quota(
            organization_id,
            UsageType.CONCURRENT_CALLS,
        )

        if not allowed:
            raise UsageLimitError(
                usage_type=UsageType.CONCURRENT_CALLS,
                limit=0,
                current=0,
            )

        self._active_calls[call_id] = {
            "organization_id": organization_id,
            "agent_id": agent_id,
            "direction": direction,
            "started_at": datetime.utcnow(),
        }

        # Increment concurrent calls
        await self._tracker.record_usage(
            organization_id=organization_id,
            usage_type=UsageType.CONCURRENT_CALLS,
            quantity=Decimal("1"),
            call_id=call_id,
            agent_id=agent_id,
            idempotency_key=f"concurrent_start_{call_id}",
        )

    async def end_call(
        self,
        call_id: str,
        duration_seconds: float,
    ) -> UsageRecord:
        """Record call end and bill duration."""
        call_info = self._active_calls.pop(call_id, None)

        if not call_info:
            raise BillingError(f"Unknown call: {call_id}")

        organization_id = call_info["organization_id"]
        agent_id = call_info["agent_id"]
        direction = call_info["direction"]

        # Calculate minutes (rounded up)
        minutes = Decimal(str(duration_seconds)) / Decimal("60")
        rounded_minutes = minutes.quantize(Decimal("1"), rounding="ROUND_UP")

        # Record call minutes
        record = await self._tracker.record_usage(
            organization_id=organization_id,
            usage_type=UsageType.CALL_MINUTES,
            quantity=rounded_minutes,
            call_id=call_id,
            agent_id=agent_id,
            idempotency_key=f"call_minutes_{call_id}",
            metadata={
                "direction": direction,
                "duration_seconds": duration_seconds,
            },
            enforce_quota=False,  # Allow overage
        )

        # Also record the call count
        call_type = UsageType.INBOUND_CALLS if direction == "inbound" else UsageType.OUTBOUND_CALLS
        await self._tracker.record_usage(
            organization_id=organization_id,
            usage_type=call_type,
            quantity=Decimal("1"),
            call_id=call_id,
            agent_id=agent_id,
            idempotency_key=f"call_count_{call_id}",
            enforce_quota=False,
        )

        return record

    async def record_transcription(
        self,
        call_id: str,
        duration_seconds: float,
    ) -> UsageRecord:
        """Record transcription usage."""
        call_info = self._active_calls.get(call_id)
        if not call_info:
            raise BillingError(f"Unknown call: {call_id}")

        minutes = Decimal(str(duration_seconds)) / Decimal("60")
        rounded_minutes = minutes.quantize(Decimal("0.01"), rounding="ROUND_UP")

        return await self._tracker.record_usage(
            organization_id=call_info["organization_id"],
            usage_type=UsageType.TRANSCRIPTION_MINUTES,
            quantity=rounded_minutes,
            call_id=call_id,
            agent_id=call_info["agent_id"],
            idempotency_key=f"transcription_{call_id}_{datetime.utcnow().timestamp()}",
            enforce_quota=False,
        )

    async def get_active_calls(self, organization_id: str) -> int:
        """Get count of active calls for organization."""
        return sum(
            1 for call in self._active_calls.values()
            if call["organization_id"] == organization_id
        )


class ResourceUsageTracker:
    """Tracker for resource-based usage (agents, phone numbers)."""

    def __init__(self, tracker: UsageTracker):
        """Initialize resource tracker."""
        self._tracker = tracker
        self._resources: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )  # org_id -> resource_type -> resource_ids

    async def allocate_resource(
        self,
        organization_id: str,
        resource_type: UsageType,
        resource_id: str,
    ) -> None:
        """Allocate a resource."""
        current_count = len(self._resources[organization_id][resource_type.value])

        allowed, error = await self._tracker.check_quota(
            organization_id,
            resource_type,
            Decimal("1"),
        )

        if not allowed:
            raise UsageLimitError(
                usage_type=resource_type,
                limit=0,
                current=current_count,
            )

        self._resources[organization_id][resource_type.value].add(resource_id)

    async def deallocate_resource(
        self,
        organization_id: str,
        resource_type: UsageType,
        resource_id: str,
    ) -> None:
        """Deallocate a resource."""
        self._resources[organization_id][resource_type.value].discard(resource_id)

    async def get_resource_count(
        self,
        organization_id: str,
        resource_type: UsageType,
    ) -> int:
        """Get count of allocated resources."""
        return len(self._resources[organization_id][resource_type.value])

    async def get_allocated_resources(
        self,
        organization_id: str,
        resource_type: UsageType,
    ) -> Set[str]:
        """Get set of allocated resource IDs."""
        return self._resources[organization_id][resource_type.value].copy()


class APIUsageTracker:
    """Tracker for API request usage with rate limiting."""

    def __init__(
        self,
        tracker: UsageTracker,
        rate_limit_window_seconds: int = 60,
        rate_limit_requests: int = 100,
    ):
        """Initialize API tracker."""
        self._tracker = tracker
        self._rate_limit_window = rate_limit_window_seconds
        self._rate_limit_requests = rate_limit_requests

        # Sliding window for rate limiting
        self._request_times: Dict[str, List[datetime]] = defaultdict(list)

    async def record_request(
        self,
        organization_id: str,
        endpoint: str,
        api_key_id: Optional[str] = None,
    ) -> UsageRecord:
        """Record an API request."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self._rate_limit_window)

        # Clean old entries and check rate limit
        self._request_times[organization_id] = [
            t for t in self._request_times[organization_id]
            if t > window_start
        ]

        if len(self._request_times[organization_id]) >= self._rate_limit_requests:
            raise BillingError(
                f"Rate limit exceeded: {self._rate_limit_requests} requests per {self._rate_limit_window}s",
                code="rate_limit_exceeded"
            )

        self._request_times[organization_id].append(now)

        return await self._tracker.record_usage(
            organization_id=organization_id,
            usage_type=UsageType.API_REQUESTS,
            quantity=Decimal("1"),
            metadata={"endpoint": endpoint, "api_key_id": api_key_id},
            enforce_quota=False,
        )

    def get_rate_limit_remaining(self, organization_id: str) -> int:
        """Get remaining requests in current window."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self._rate_limit_window)

        recent = [
            t for t in self._request_times[organization_id]
            if t > window_start
        ]

        return max(0, self._rate_limit_requests - len(recent))

    def get_rate_limit_reset(self, organization_id: str) -> Optional[datetime]:
        """Get time when rate limit resets."""
        if not self._request_times[organization_id]:
            return None

        oldest = min(self._request_times[organization_id])
        return oldest + timedelta(seconds=self._rate_limit_window)
