"""Metering service for real-time usage tracking."""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class MeterType(str, Enum):
    """Types of meters."""
    COUNTER = "counter"  # Only increments
    GAUGE = "gauge"  # Can go up or down
    HISTOGRAM = "histogram"  # Distribution of values


@dataclass
class MeterReading:
    """A single meter reading."""
    meter_id: str
    user_id: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "meter_id": self.meter_id,
            "user_id": self.user_id,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Meter:
    """A usage meter."""
    meter_id: str
    name: str
    meter_type: MeterType
    unit: str
    description: str = ""
    aggregation_window_seconds: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "meter_id": self.meter_id,
            "name": self.name,
            "meter_type": self.meter_type.value,
            "unit": self.unit,
            "description": self.description,
            "aggregation_window_seconds": self.aggregation_window_seconds,
        }


class MeteringService:
    """
    Real-time metering service.

    Tracks usage in real-time with aggregation and event emission.

    Usage:
        service = MeteringService()

        # Define meters
        service.create_meter("call_minutes", "Call Minutes", MeterType.COUNTER, "minutes")

        # Record usage
        await service.record("call_minutes", user_id, 5.0)

        # Get current usage
        usage = await service.get_current_usage(user_id, "call_minutes")
    """

    def __init__(
        self,
        aggregation_interval: float = 60.0,
        flush_interval: float = 300.0,
    ):
        self.aggregation_interval = aggregation_interval
        self.flush_interval = flush_interval

        self._meters: Dict[str, Meter] = {}
        self._readings: Dict[str, List[MeterReading]] = defaultdict(list)
        self._aggregated: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._callbacks: List[Callable[[MeterReading], Awaitable[None]]] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start the metering service."""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._aggregation_loop()),
            asyncio.create_task(self._flush_loop()),
        ]
        logger.info("Metering service started")

    async def stop(self) -> None:
        """Stop the metering service."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("Metering service stopped")

    def create_meter(
        self,
        meter_id: str,
        name: str,
        meter_type: MeterType,
        unit: str,
        description: str = "",
    ) -> Meter:
        """Create a new meter."""
        meter = Meter(
            meter_id=meter_id,
            name=name,
            meter_type=meter_type,
            unit=unit,
            description=description,
        )
        self._meters[meter_id] = meter
        return meter

    def get_meter(self, meter_id: str) -> Optional[Meter]:
        """Get a meter by ID."""
        return self._meters.get(meter_id)

    def on_reading(self, callback: Callable[[MeterReading], Awaitable[None]]) -> None:
        """Register callback for new readings."""
        self._callbacks.append(callback)

    async def record(
        self,
        meter_id: str,
        user_id: str,
        value: float,
        **metadata,
    ) -> MeterReading:
        """Record a meter reading."""
        meter = self._meters.get(meter_id)
        if not meter:
            raise ValueError(f"Unknown meter: {meter_id}")

        reading = MeterReading(
            meter_id=meter_id,
            user_id=user_id,
            value=value,
            metadata=metadata,
        )

        async with self._lock:
            self._readings[f"{user_id}:{meter_id}"].append(reading)

            # Update aggregated value
            if meter.meter_type == MeterType.COUNTER:
                self._aggregated[user_id][meter_id] += value
            elif meter.meter_type == MeterType.GAUGE:
                self._aggregated[user_id][meter_id] = value

        # Notify callbacks
        for callback in self._callbacks:
            try:
                await callback(reading)
            except Exception as e:
                logger.error(f"Metering callback error: {e}")

        return reading

    async def increment(
        self,
        meter_id: str,
        user_id: str,
        value: float = 1.0,
        **metadata,
    ) -> MeterReading:
        """Increment a counter meter."""
        meter = self._meters.get(meter_id)
        if meter and meter.meter_type != MeterType.COUNTER:
            raise ValueError(f"Meter {meter_id} is not a counter")
        return await self.record(meter_id, user_id, value, **metadata)

    async def set_gauge(
        self,
        meter_id: str,
        user_id: str,
        value: float,
        **metadata,
    ) -> MeterReading:
        """Set a gauge meter value."""
        meter = self._meters.get(meter_id)
        if meter and meter.meter_type != MeterType.GAUGE:
            raise ValueError(f"Meter {meter_id} is not a gauge")
        return await self.record(meter_id, user_id, value, **metadata)

    async def get_current_usage(
        self,
        user_id: str,
        meter_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get current aggregated usage."""
        async with self._lock:
            if meter_id:
                return {meter_id: self._aggregated[user_id].get(meter_id, 0.0)}
            return dict(self._aggregated[user_id])

    async def get_readings(
        self,
        user_id: str,
        meter_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[MeterReading]:
        """Get meter readings."""
        key = f"{user_id}:{meter_id}"

        async with self._lock:
            readings = self._readings.get(key, [])

        if start:
            readings = [r for r in readings if r.timestamp >= start]
        if end:
            readings = [r for r in readings if r.timestamp <= end]

        return readings

    async def get_rate(
        self,
        user_id: str,
        meter_id: str,
        window_seconds: int = 60,
    ) -> float:
        """Get rate of change over a window."""
        end = datetime.utcnow()
        start = end - timedelta(seconds=window_seconds)

        readings = await self.get_readings(user_id, meter_id, start, end)

        if len(readings) < 2:
            return 0.0

        total = sum(r.value for r in readings)
        return total / window_seconds

    async def reset(self, user_id: str, meter_id: Optional[str] = None) -> None:
        """Reset meter values."""
        async with self._lock:
            if meter_id:
                self._aggregated[user_id][meter_id] = 0.0
                key = f"{user_id}:{meter_id}"
                self._readings[key] = []
            else:
                self._aggregated[user_id] = defaultdict(float)
                for key in list(self._readings.keys()):
                    if key.startswith(f"{user_id}:"):
                        del self._readings[key]

    async def _aggregation_loop(self) -> None:
        """Periodic aggregation of readings."""
        while self._running:
            await asyncio.sleep(self.aggregation_interval)
            # Aggregation is done in real-time, this is for cleanup
            await self._cleanup_old_readings()

    async def _flush_loop(self) -> None:
        """Periodic flush of readings to persistent storage."""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self._flush_readings()

    async def _cleanup_old_readings(self) -> None:
        """Clean up old readings."""
        cutoff = datetime.utcnow() - timedelta(hours=1)

        async with self._lock:
            for key in self._readings:
                self._readings[key] = [
                    r for r in self._readings[key]
                    if r.timestamp > cutoff
                ]

    async def _flush_readings(self) -> None:
        """Flush readings to persistent storage."""
        # In a real implementation, this would write to a database
        async with self._lock:
            total_readings = sum(len(r) for r in self._readings.values())
            if total_readings > 0:
                logger.debug(f"Flushing {total_readings} meter readings")


class RealTimeUsageMonitor:
    """
    Real-time usage monitoring with alerts.

    Usage:
        monitor = RealTimeUsageMonitor(metering_service)

        # Set alert threshold
        monitor.set_threshold("call_minutes", user_id, 900, callback)

        # Check usage
        status = await monitor.check_usage(user_id)
    """

    def __init__(self, metering: MeteringService):
        self.metering = metering
        self._thresholds: Dict[str, Dict[str, tuple]] = defaultdict(dict)
        self._alerts_triggered: Dict[str, set] = defaultdict(set)

    def set_threshold(
        self,
        meter_id: str,
        user_id: str,
        threshold: float,
        callback: Optional[Callable] = None,
        alert_at_percent: float = 80,
    ) -> None:
        """Set usage threshold for alerts."""
        self._thresholds[user_id][meter_id] = (threshold, callback, alert_at_percent)

    async def check_usage(self, user_id: str) -> Dict[str, Any]:
        """Check usage against thresholds."""
        current = await self.metering.get_current_usage(user_id)
        status = {}

        for meter_id, (threshold, callback, alert_percent) in self._thresholds.get(user_id, {}).items():
            usage = current.get(meter_id, 0)
            percent = (usage / threshold * 100) if threshold > 0 else 0

            status[meter_id] = {
                "usage": usage,
                "threshold": threshold,
                "percent": percent,
                "remaining": max(0, threshold - usage),
                "exceeded": usage >= threshold,
            }

            # Check for alerts
            alert_key = f"{user_id}:{meter_id}"
            if percent >= alert_percent and alert_key not in self._alerts_triggered[user_id]:
                self._alerts_triggered[user_id].add(alert_key)
                if callback:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(user_id, meter_id, usage, threshold)
                        else:
                            callback(user_id, meter_id, usage, threshold)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")

        return status

    def reset_alerts(self, user_id: str) -> None:
        """Reset triggered alerts for user."""
        self._alerts_triggered[user_id] = set()


class UsageQuotaManager:
    """
    Manages usage quotas.

    Usage:
        quota = UsageQuotaManager(metering_service)
        quota.set_quota(user_id, "call_minutes", 1000)

        can_use = await quota.check_and_reserve(user_id, "call_minutes", 5)
    """

    def __init__(self, metering: MeteringService):
        self.metering = metering
        self._quotas: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._reserved: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._lock = asyncio.Lock()

    def set_quota(self, user_id: str, meter_id: str, quota: float) -> None:
        """Set quota for user."""
        self._quotas[user_id][meter_id] = quota

    def get_quota(self, user_id: str, meter_id: str) -> Optional[float]:
        """Get quota for user."""
        return self._quotas.get(user_id, {}).get(meter_id)

    async def get_available(self, user_id: str, meter_id: str) -> float:
        """Get available quota."""
        quota = self.get_quota(user_id, meter_id)
        if quota is None:
            return float("inf")

        current = await self.metering.get_current_usage(user_id, meter_id)
        used = current.get(meter_id, 0)

        async with self._lock:
            reserved = self._reserved[user_id][meter_id]

        return max(0, quota - used - reserved)

    async def check_and_reserve(
        self,
        user_id: str,
        meter_id: str,
        amount: float,
    ) -> bool:
        """Check quota and reserve amount."""
        available = await self.get_available(user_id, meter_id)

        if amount > available:
            return False

        async with self._lock:
            self._reserved[user_id][meter_id] += amount

        return True

    async def release_reservation(
        self,
        user_id: str,
        meter_id: str,
        amount: float,
    ) -> None:
        """Release a reservation."""
        async with self._lock:
            self._reserved[user_id][meter_id] = max(
                0, self._reserved[user_id][meter_id] - amount
            )

    async def consume(
        self,
        user_id: str,
        meter_id: str,
        amount: float,
    ) -> bool:
        """Consume quota (record usage and release reservation)."""
        await self.metering.record(meter_id, user_id, amount)
        await self.release_reservation(user_id, meter_id, amount)
        return True


# Default meters
DEFAULT_METERS = [
    Meter("call_minutes", "Call Minutes", MeterType.COUNTER, "minutes", "Voice call duration"),
    Meter("call_count", "Call Count", MeterType.COUNTER, "calls", "Number of calls"),
    Meter("llm_tokens", "LLM Tokens", MeterType.COUNTER, "tokens", "Language model tokens"),
    Meter("tts_characters", "TTS Characters", MeterType.COUNTER, "characters", "Text-to-speech characters"),
    Meter("asr_minutes", "ASR Minutes", MeterType.COUNTER, "minutes", "Speech recognition duration"),
    Meter("active_calls", "Active Calls", MeterType.GAUGE, "calls", "Currently active calls"),
    Meter("storage_bytes", "Storage", MeterType.GAUGE, "bytes", "Storage used"),
]


# Global metering service
_metering_service: Optional[MeteringService] = None


def get_metering_service() -> MeteringService:
    """Get or create the global metering service."""
    global _metering_service
    if _metering_service is None:
        _metering_service = MeteringService()
        for meter in DEFAULT_METERS:
            _metering_service._meters[meter.meter_id] = meter
    return _metering_service
