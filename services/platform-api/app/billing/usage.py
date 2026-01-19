"""Usage tracking for billing."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from collections import defaultdict
import asyncio
import logging

logger = logging.getLogger(__name__)


class UsageType(str, Enum):
    """Types of billable usage."""
    # Call usage
    CALL_MINUTES = "call_minutes"
    CALL_COUNT = "call_count"

    # ASR usage
    ASR_MINUTES = "asr_minutes"
    ASR_CHARACTERS = "asr_characters"

    # TTS usage
    TTS_CHARACTERS = "tts_characters"
    TTS_MINUTES = "tts_minutes"

    # LLM usage
    LLM_INPUT_TOKENS = "llm_input_tokens"
    LLM_OUTPUT_TOKENS = "llm_output_tokens"

    # Storage usage
    STORAGE_GB = "storage_gb"
    RECORDING_MINUTES = "recording_minutes"

    # Phone numbers
    PHONE_NUMBERS = "phone_numbers"
    SMS_SENT = "sms_sent"
    SMS_RECEIVED = "sms_received"

    # API usage
    API_CALLS = "api_calls"
    WEBHOOK_DELIVERIES = "webhook_deliveries"

    # Transcription
    TRANSCRIPTION_MINUTES = "transcription_minutes"

    # Function calls
    FUNCTION_CALLS = "function_calls"


@dataclass
class UsageRecord:
    """Record of usage."""
    user_id: str
    usage_type: UsageType
    quantity: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resource_id: Optional[str] = None
    call_id: Optional[str] = None
    agent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "usage_type": self.usage_type.value,
            "quantity": self.quantity,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "resource_id": self.resource_id,
            "call_id": self.call_id,
            "agent_id": self.agent_id,
        }


@dataclass
class UsageSummary:
    """Summary of usage for a period."""
    user_id: str
    period_start: datetime
    period_end: datetime
    usage: Dict[str, float] = field(default_factory=dict)
    costs: Dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "usage": self.usage,
            "costs": self.costs,
            "total_cost": self.total_cost,
        }


class UsageTracker:
    """
    Tracks usage for billing.

    Usage:
        tracker = UsageTracker()

        # Record usage
        await tracker.record(
            user_id="user-123",
            usage_type=UsageType.CALL_MINUTES,
            quantity=5.5,
            call_id="call-456",
        )

        # Get usage summary
        summary = await tracker.get_summary(user_id, start_date, end_date)
    """

    def __init__(self):
        self._records: List[UsageRecord] = []
        self._lock = asyncio.Lock()
        self._callbacks: List[callable] = []

    def on_usage(self, callback: callable) -> None:
        """Register callback for usage events."""
        self._callbacks.append(callback)

    async def record(
        self,
        user_id: str,
        usage_type: UsageType,
        quantity: float,
        **kwargs,
    ) -> UsageRecord:
        """Record a usage event."""
        record = UsageRecord(
            user_id=user_id,
            usage_type=usage_type,
            quantity=quantity,
            **kwargs,
        )

        async with self._lock:
            self._records.append(record)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(record)
                else:
                    callback(record)
            except Exception as e:
                logger.error(f"Usage callback error: {e}")

        return record

    async def record_call(
        self,
        user_id: str,
        call_id: str,
        duration_seconds: float,
        agent_id: Optional[str] = None,
    ) -> List[UsageRecord]:
        """Record call usage."""
        records = []

        # Call minutes
        minutes = duration_seconds / 60
        records.append(await self.record(
            user_id=user_id,
            usage_type=UsageType.CALL_MINUTES,
            quantity=minutes,
            call_id=call_id,
            agent_id=agent_id,
        ))

        # Call count
        records.append(await self.record(
            user_id=user_id,
            usage_type=UsageType.CALL_COUNT,
            quantity=1,
            call_id=call_id,
            agent_id=agent_id,
        ))

        return records

    async def record_llm(
        self,
        user_id: str,
        input_tokens: int,
        output_tokens: int,
        call_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[UsageRecord]:
        """Record LLM token usage."""
        records = []

        records.append(await self.record(
            user_id=user_id,
            usage_type=UsageType.LLM_INPUT_TOKENS,
            quantity=input_tokens,
            call_id=call_id,
            metadata={"model": model} if model else {},
        ))

        records.append(await self.record(
            user_id=user_id,
            usage_type=UsageType.LLM_OUTPUT_TOKENS,
            quantity=output_tokens,
            call_id=call_id,
            metadata={"model": model} if model else {},
        ))

        return records

    async def record_tts(
        self,
        user_id: str,
        characters: int,
        call_id: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> UsageRecord:
        """Record TTS usage."""
        return await self.record(
            user_id=user_id,
            usage_type=UsageType.TTS_CHARACTERS,
            quantity=characters,
            call_id=call_id,
            metadata={"provider": provider} if provider else {},
        )

    async def record_asr(
        self,
        user_id: str,
        duration_seconds: float,
        call_id: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> UsageRecord:
        """Record ASR usage."""
        return await self.record(
            user_id=user_id,
            usage_type=UsageType.ASR_MINUTES,
            quantity=duration_seconds / 60,
            call_id=call_id,
            metadata={"provider": provider} if provider else {},
        )

    async def get_records(
        self,
        user_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        usage_type: Optional[UsageType] = None,
    ) -> List[UsageRecord]:
        """Get usage records."""
        async with self._lock:
            records = [
                r for r in self._records
                if r.user_id == user_id
            ]

        if start:
            records = [r for r in records if r.timestamp >= start]
        if end:
            records = [r for r in records if r.timestamp <= end]
        if usage_type:
            records = [r for r in records if r.usage_type == usage_type]

        return records

    async def get_totals(
        self,
        user_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get total usage by type."""
        records = await self.get_records(user_id, start, end)

        totals = defaultdict(float)
        for record in records:
            totals[record.usage_type.value] += record.quantity

        return dict(totals)


class UsageAggregator:
    """
    Aggregates usage data for reporting.

    Usage:
        aggregator = UsageAggregator(tracker)
        daily = await aggregator.aggregate_daily(user_id, start, end)
        monthly = await aggregator.aggregate_monthly(user_id, year, month)
    """

    def __init__(self, tracker: UsageTracker):
        self.tracker = tracker

    async def aggregate_daily(
        self,
        user_id: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """Aggregate usage by day."""
        results = []
        current = start_date

        while current <= end_date:
            start = datetime.combine(current, datetime.min.time())
            end = datetime.combine(current, datetime.max.time())

            totals = await self.tracker.get_totals(user_id, start, end)

            results.append({
                "date": current.isoformat(),
                "usage": totals,
            })

            current += timedelta(days=1)

        return results

    async def aggregate_monthly(
        self,
        user_id: str,
        year: int,
        month: int,
    ) -> Dict[str, Any]:
        """Aggregate usage for a month."""
        from calendar import monthrange

        _, last_day = monthrange(year, month)
        start = datetime(year, month, 1)
        end = datetime(year, month, last_day, 23, 59, 59)

        totals = await self.tracker.get_totals(user_id, start, end)

        return {
            "year": year,
            "month": month,
            "usage": totals,
        }

    async def aggregate_by_agent(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate usage by agent."""
        records = await self.tracker.get_records(user_id, start, end)

        by_agent: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for record in records:
            agent_id = record.agent_id or "unknown"
            by_agent[agent_id][record.usage_type.value] += record.quantity

        return {k: dict(v) for k, v in by_agent.items()}

    async def aggregate_by_call(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate usage by call."""
        records = await self.tracker.get_records(user_id, start, end)

        by_call: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for record in records:
            if record.call_id:
                by_call[record.call_id][record.usage_type.value] += record.quantity

        return {k: dict(v) for k, v in by_call.items()}


class UsageLimiter:
    """
    Enforces usage limits.

    Usage:
        limiter = UsageLimiter(tracker)
        limiter.set_limit(user_id, UsageType.CALL_MINUTES, 1000)

        can_use = await limiter.check_limit(user_id, UsageType.CALL_MINUTES, 5)
    """

    def __init__(self, tracker: UsageTracker):
        self.tracker = tracker
        self._limits: Dict[str, Dict[str, float]] = {}
        self._reset_day = 1  # Day of month to reset limits

    def set_limit(
        self,
        user_id: str,
        usage_type: UsageType,
        limit: float,
    ) -> None:
        """Set usage limit."""
        if user_id not in self._limits:
            self._limits[user_id] = {}
        self._limits[user_id][usage_type.value] = limit

    def get_limit(
        self,
        user_id: str,
        usage_type: UsageType,
    ) -> Optional[float]:
        """Get usage limit."""
        return self._limits.get(user_id, {}).get(usage_type.value)

    async def check_limit(
        self,
        user_id: str,
        usage_type: UsageType,
        quantity: float,
    ) -> bool:
        """Check if usage would exceed limit."""
        limit = self.get_limit(user_id, usage_type)
        if limit is None:
            return True  # No limit set

        # Get current period's usage
        now = datetime.utcnow()
        period_start = datetime(now.year, now.month, self._reset_day)
        if now.day < self._reset_day:
            # Before reset day, use previous month
            if now.month == 1:
                period_start = datetime(now.year - 1, 12, self._reset_day)
            else:
                period_start = datetime(now.year, now.month - 1, self._reset_day)

        totals = await self.tracker.get_totals(user_id, period_start, now)
        current = totals.get(usage_type.value, 0)

        return (current + quantity) <= limit

    async def get_remaining(
        self,
        user_id: str,
        usage_type: UsageType,
    ) -> Optional[float]:
        """Get remaining usage."""
        limit = self.get_limit(user_id, usage_type)
        if limit is None:
            return None

        now = datetime.utcnow()
        period_start = datetime(now.year, now.month, self._reset_day)

        totals = await self.tracker.get_totals(user_id, period_start, now)
        current = totals.get(usage_type.value, 0)

        return max(0, limit - current)


# Global usage tracker
_usage_tracker: Optional[UsageTracker] = None


def get_usage_tracker() -> UsageTracker:
    """Get or create the global usage tracker."""
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker()
    return _usage_tracker
