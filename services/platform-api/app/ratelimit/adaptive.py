"""
Adaptive Rate Limiting

Dynamic rate limiters that adjust based on:
- System load and resource utilization
- Request latency and error rates
- Concurrent connection counts
- AI/ML model inference capacity
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
import asyncio
import time
import logging
import math
import statistics

from app.ratelimit.algorithms import RateLimiter, RateLimitResult

logger = logging.getLogger(__name__)


class LoadLevel(str, Enum):
    """System load levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """Current system metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    active_connections: int = 0
    request_latency_ms: float = 0.0
    error_rate: float = 0.0
    queue_depth: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def get_load_level(self, thresholds: "LoadThresholds") -> LoadLevel:
        """Determine load level from metrics."""
        # Critical if any metric exceeds critical threshold
        if (
            self.cpu_percent >= thresholds.cpu_critical or
            self.memory_percent >= thresholds.memory_critical or
            self.error_rate >= thresholds.error_rate_critical or
            self.request_latency_ms >= thresholds.latency_critical_ms
        ):
            return LoadLevel.CRITICAL

        # High if any metric exceeds high threshold
        if (
            self.cpu_percent >= thresholds.cpu_high or
            self.memory_percent >= thresholds.memory_high or
            self.error_rate >= thresholds.error_rate_high or
            self.request_latency_ms >= thresholds.latency_high_ms
        ):
            return LoadLevel.HIGH

        # Normal if any metric exceeds normal threshold
        if (
            self.cpu_percent >= thresholds.cpu_normal or
            self.memory_percent >= thresholds.memory_normal
        ):
            return LoadLevel.NORMAL

        return LoadLevel.LOW


@dataclass
class LoadThresholds:
    """Thresholds for load levels."""
    # CPU thresholds (percent)
    cpu_normal: float = 50.0
    cpu_high: float = 70.0
    cpu_critical: float = 90.0

    # Memory thresholds (percent)
    memory_normal: float = 60.0
    memory_high: float = 80.0
    memory_critical: float = 95.0

    # Error rate thresholds (percent)
    error_rate_high: float = 5.0
    error_rate_critical: float = 10.0

    # Latency thresholds (ms)
    latency_high_ms: float = 500.0
    latency_critical_ms: float = 1000.0


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive rate limiting."""
    base_rate: float = 100.0  # Base requests per second
    min_rate: float = 10.0  # Minimum rate even under load
    max_rate: float = 500.0  # Maximum rate when system is idle

    # Multipliers for each load level
    normal_multiplier: float = 1.0
    high_multiplier: float = 0.5
    critical_multiplier: float = 0.1

    # Adjustment parameters
    adjustment_interval_seconds: float = 5.0
    smoothing_factor: float = 0.3  # Exponential smoothing
    burst_allowance: float = 1.5  # Allow burst up to this multiple

    # Thresholds
    thresholds: LoadThresholds = field(default_factory=LoadThresholds)


class MetricsCollector(ABC):
    """Abstract base for metrics collection."""

    @abstractmethod
    async def collect(self) -> SystemMetrics:
        """Collect current system metrics."""
        pass


class DefaultMetricsCollector(MetricsCollector):
    """Default metrics collector using psutil."""

    def __init__(self):
        self._latency_window: deque = deque(maxlen=100)
        self._error_window: deque = deque(maxlen=100)

    async def collect(self) -> SystemMetrics:
        """Collect system metrics."""
        try:
            import psutil

            # Get CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # Calculate average latency from window
            avg_latency = (
                statistics.mean(self._latency_window)
                if self._latency_window
                else 0.0
            )

            # Calculate error rate from window
            error_rate = (
                sum(1 for e in self._error_window if e) / len(self._error_window) * 100
                if self._error_window
                else 0.0
            )

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                active_connections=0,  # Would need app-specific tracking
                request_latency_ms=avg_latency,
                error_rate=error_rate,
            )
        except ImportError:
            logger.warning("psutil not installed, using default metrics")
            return SystemMetrics()

    def record_latency(self, latency_ms: float) -> None:
        """Record a request latency."""
        self._latency_window.append(latency_ms)

    def record_request(self, success: bool) -> None:
        """Record a request outcome."""
        self._error_window.append(not success)


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts based on system load.

    Monitors system metrics and dynamically adjusts rate limits
    to protect system stability.
    """

    def __init__(
        self,
        config: Optional[AdaptiveConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        base_limiter_factory: Optional[Callable[[float, int], RateLimiter]] = None,
    ):
        """
        Args:
            config: Adaptive configuration
            metrics_collector: Metrics collector
            base_limiter_factory: Factory to create base limiters
        """
        self.config = config or AdaptiveConfig()
        self.metrics_collector = metrics_collector or DefaultMetricsCollector()

        # Import here to avoid circular dependency
        from app.ratelimit.algorithms import TokenBucket

        if base_limiter_factory:
            self._limiter_factory = base_limiter_factory
        else:
            self._limiter_factory = lambda rate, cap: TokenBucket(rate=rate, capacity=cap)

        # Current state
        self._current_rate = self.config.base_rate
        self._current_capacity = int(self.config.base_rate * self.config.burst_allowance)
        self._current_limiter: RateLimiter = self._create_limiter()
        self._current_load_level = LoadLevel.NORMAL

        # Metrics history
        self._metrics_history: deque = deque(maxlen=60)
        self._rate_history: deque = deque(maxlen=60)

        # Background task
        self._adjustment_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()

    def _create_limiter(self) -> RateLimiter:
        """Create limiter with current rate."""
        return self._limiter_factory(
            self._current_rate,
            self._current_capacity,
        )

    async def start(self) -> None:
        """Start adaptive adjustment."""
        self._running = True
        self._adjustment_task = asyncio.create_task(self._adjustment_loop())
        logger.info(f"Adaptive rate limiter started with base rate: {self.config.base_rate}")

    async def stop(self) -> None:
        """Stop adaptive adjustment."""
        self._running = False
        if self._adjustment_task:
            self._adjustment_task.cancel()
            try:
                await self._adjustment_task
            except asyncio.CancelledError:
                pass
        logger.info("Adaptive rate limiter stopped")

    async def _adjustment_loop(self) -> None:
        """Background loop to adjust rates."""
        while self._running:
            try:
                await self._adjust_rate()
            except Exception as e:
                logger.error(f"Rate adjustment error: {e}")

            await asyncio.sleep(self.config.adjustment_interval_seconds)

    async def _adjust_rate(self) -> None:
        """Adjust rate based on current metrics."""
        metrics = await self.metrics_collector.collect()
        self._metrics_history.append(metrics)

        load_level = metrics.get_load_level(self.config.thresholds)
        self._current_load_level = load_level

        # Calculate target rate based on load level
        if load_level == LoadLevel.LOW:
            target_rate = self.config.max_rate
        elif load_level == LoadLevel.NORMAL:
            target_rate = self.config.base_rate * self.config.normal_multiplier
        elif load_level == LoadLevel.HIGH:
            target_rate = self.config.base_rate * self.config.high_multiplier
        else:  # CRITICAL
            target_rate = self.config.base_rate * self.config.critical_multiplier

        # Clamp to min/max
        target_rate = max(self.config.min_rate, min(self.config.max_rate, target_rate))

        # Apply exponential smoothing
        new_rate = (
            self.config.smoothing_factor * target_rate +
            (1 - self.config.smoothing_factor) * self._current_rate
        )

        # Update if significantly different
        if abs(new_rate - self._current_rate) / self._current_rate > 0.05:
            async with self._lock:
                self._current_rate = new_rate
                self._current_capacity = int(new_rate * self.config.burst_allowance)
                self._current_limiter = self._create_limiter()

            self._rate_history.append({
                "timestamp": datetime.utcnow(),
                "rate": new_rate,
                "load_level": load_level.value,
            })

            logger.info(
                f"Adjusted rate: {new_rate:.2f} req/s "
                f"(load: {load_level.value}, cpu: {metrics.cpu_percent:.1f}%)"
            )

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check rate limit with current adaptive rate."""
        async with self._lock:
            result = await self._current_limiter.check(key, cost)

        # Add load level info to result
        result_dict = result.to_dict()
        result_dict["load_level"] = self._current_load_level.value
        result_dict["current_rate"] = self._current_rate

        return result

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        async with self._lock:
            await self._current_limiter.reset(key)

    def get_status(self) -> Dict[str, Any]:
        """Get current adaptive status."""
        return {
            "current_rate": self._current_rate,
            "current_capacity": self._current_capacity,
            "load_level": self._current_load_level.value,
            "base_rate": self.config.base_rate,
            "min_rate": self.config.min_rate,
            "max_rate": self.config.max_rate,
            "rate_history": list(self._rate_history)[-10:],
        }


class LoadBasedLimiter(RateLimiter):
    """
    Rate limiter that adjusts based on external load signal.

    Useful when you have a load balancer or orchestrator
    that can provide load information.
    """

    def __init__(
        self,
        base_rate: float = 100.0,
        load_provider: Optional[Callable[[], Awaitable[float]]] = None,
    ):
        """
        Args:
            base_rate: Base requests per second
            load_provider: Async function returning load 0.0-1.0
        """
        self.base_rate = base_rate
        self.load_provider = load_provider

        from app.ratelimit.algorithms import TokenBucket
        self._limiter = TokenBucket(rate=base_rate, capacity=int(base_rate * 2))
        self._lock = asyncio.Lock()

    async def _get_adjusted_rate(self) -> float:
        """Get rate adjusted for current load."""
        if self.load_provider:
            load = await self.load_provider()
            # Reduce rate linearly with load
            return self.base_rate * (1.0 - load * 0.9)
        return self.base_rate

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check with load-adjusted rate."""
        adjusted_rate = await self._get_adjusted_rate()

        # Recreate limiter if rate changed significantly
        if abs(adjusted_rate - self._limiter.rate) / self._limiter.rate > 0.1:
            async with self._lock:
                from app.ratelimit.algorithms import TokenBucket
                self._limiter = TokenBucket(
                    rate=adjusted_rate,
                    capacity=int(adjusted_rate * 2)
                )

        return await self._limiter.check(key, cost)

    async def reset(self, key: str) -> None:
        """Reset rate limit."""
        await self._limiter.reset(key)


class ConcurrencyLimiter:
    """
    Limits concurrent operations rather than rate.

    Useful for protecting resources with limited capacity
    like database connections or model inference.
    """

    def __init__(
        self,
        max_concurrent: int = 100,
        max_queue: int = 500,
        timeout_seconds: float = 30.0,
        per_key_max: Optional[int] = None,
    ):
        """
        Args:
            max_concurrent: Maximum concurrent operations
            max_queue: Maximum queued operations
            timeout_seconds: Timeout waiting for slot
            per_key_max: Maximum concurrent per key
        """
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.timeout_seconds = timeout_seconds
        self.per_key_max = per_key_max

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_count = 0
        self._key_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()

        # Metrics
        self._total_acquired = 0
        self._total_rejected = 0
        self._total_timeouts = 0

    async def acquire(self, key: str = "default") -> bool:
        """
        Try to acquire a concurrency slot.

        Returns True if acquired, False if rejected.
        """
        async with self._lock:
            # Check queue limit
            if self._queue_count >= self.max_queue:
                self._total_rejected += 1
                return False

            # Check per-key limit
            if self.per_key_max:
                current = self._key_counts.get(key, 0)
                if current >= self.per_key_max:
                    self._total_rejected += 1
                    return False

            self._queue_count += 1

        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.timeout_seconds
            )

            if acquired:
                async with self._lock:
                    self._queue_count -= 1
                    self._key_counts[key] = self._key_counts.get(key, 0) + 1
                    self._total_acquired += 1
                return True

        except asyncio.TimeoutError:
            async with self._lock:
                self._queue_count -= 1
                self._total_timeouts += 1
            return False

        return False

    async def release(self, key: str = "default") -> None:
        """Release a concurrency slot."""
        self._semaphore.release()

        async with self._lock:
            if key in self._key_counts:
                self._key_counts[key] -= 1
                if self._key_counts[key] <= 0:
                    del self._key_counts[key]

    async def __aenter__(self):
        """Context manager entry."""
        if not await self.acquire():
            raise ConcurrencyLimitExceeded("Concurrency limit exceeded")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.release()

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "max_concurrent": self.max_concurrent,
            "current_concurrent": self.max_concurrent - self._semaphore._value,
            "queue_count": self._queue_count,
            "max_queue": self.max_queue,
            "per_key_counts": dict(self._key_counts),
            "metrics": {
                "total_acquired": self._total_acquired,
                "total_rejected": self._total_rejected,
                "total_timeouts": self._total_timeouts,
            },
        }


class ConcurrencyLimitExceeded(Exception):
    """Raised when concurrency limit is exceeded."""
    pass


class AIMLLimiter:
    """
    Specialized rate limiter for AI/ML inference.

    Considers:
    - Model capacity (tokens/second)
    - GPU memory constraints
    - Batch processing optimization
    - Priority queuing
    """

    def __init__(
        self,
        model_tokens_per_second: float = 1000.0,
        max_batch_size: int = 8,
        max_input_tokens: int = 4096,
        max_concurrent_batches: int = 4,
        priority_levels: int = 3,
    ):
        """
        Args:
            model_tokens_per_second: Model throughput
            max_batch_size: Maximum batch size
            max_input_tokens: Maximum input tokens per request
            max_concurrent_batches: Maximum concurrent batches
            priority_levels: Number of priority levels (higher = more urgent)
        """
        self.model_tps = model_tokens_per_second
        self.max_batch_size = max_batch_size
        self.max_input_tokens = max_input_tokens
        self.max_concurrent_batches = max_concurrent_batches
        self.priority_levels = priority_levels

        # Token bucket for overall throughput
        from app.ratelimit.algorithms import TokenBucket
        self._token_limiter = TokenBucket(
            rate=model_tokens_per_second,
            capacity=int(model_tokens_per_second * 2)
        )

        # Concurrency for batches
        self._batch_limiter = ConcurrencyLimiter(
            max_concurrent=max_concurrent_batches,
            max_queue=max_concurrent_batches * 10,
        )

        # Priority queues
        self._priority_queues: List[asyncio.Queue] = [
            asyncio.Queue() for _ in range(priority_levels)
        ]

        self._lock = asyncio.Lock()
        self._current_batch: List[Dict[str, Any]] = []
        self._batch_event = asyncio.Event()

        # Metrics
        self._total_tokens_processed = 0
        self._total_requests = 0
        self._batches_processed = 0

    async def check_capacity(
        self,
        estimated_tokens: int,
        priority: int = 1,
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if there's capacity for a request.

        Args:
            estimated_tokens: Estimated tokens for this request
            priority: Priority level (0 = lowest)

        Returns:
            Tuple of (allowed, estimated_wait_seconds)
        """
        # Validate input
        if estimated_tokens > self.max_input_tokens:
            return False, None

        # Check token budget
        result = await self._token_limiter.check("global", estimated_tokens)

        if result.allowed:
            return True, 0.0

        # Calculate wait time based on token debt
        wait_time = result.retry_after or (estimated_tokens / self.model_tps)

        # Adjust wait time based on priority
        priority_multiplier = 1.0 - (priority / self.priority_levels) * 0.5
        adjusted_wait = wait_time * priority_multiplier

        return False, adjusted_wait

    async def acquire_batch_slot(self) -> bool:
        """Acquire a slot for batch processing."""
        return await self._batch_limiter.acquire()

    async def release_batch_slot(self) -> None:
        """Release a batch processing slot."""
        await self._batch_limiter.release()

    async def record_completion(
        self,
        tokens_processed: int,
        latency_ms: float,
    ) -> None:
        """Record completion of a request."""
        async with self._lock:
            self._total_tokens_processed += tokens_processed
            self._total_requests += 1

    def get_status(self) -> Dict[str, Any]:
        """Get limiter status."""
        return {
            "model_tps": self.model_tps,
            "max_batch_size": self.max_batch_size,
            "batch_status": self._batch_limiter.get_status(),
            "metrics": {
                "total_tokens": self._total_tokens_processed,
                "total_requests": self._total_requests,
                "batches_processed": self._batches_processed,
                "avg_tokens_per_request": (
                    self._total_tokens_processed / self._total_requests
                    if self._total_requests > 0 else 0
                ),
            },
        }


class GradualDegradationLimiter(RateLimiter):
    """
    Rate limiter with gradual degradation.

    Instead of hard cutoffs, gradually reduces service quality
    as limits are approached.
    """

    def __init__(
        self,
        soft_limit: int = 80,
        hard_limit: int = 100,
        window_seconds: int = 60,
        degradation_levels: int = 5,
    ):
        """
        Args:
            soft_limit: Soft limit before degradation starts
            hard_limit: Hard limit (complete denial)
            window_seconds: Window duration
            degradation_levels: Number of degradation levels
        """
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self.window_seconds = window_seconds
        self.degradation_levels = degradation_levels

        from app.ratelimit.algorithms import SlidingWindow
        self._limiter = SlidingWindow(
            limit=hard_limit,
            window_seconds=window_seconds
        )

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check with degradation level."""
        result = await self._limiter.check(key, cost)

        # Calculate degradation level
        current = result.current
        degradation_level = 0

        if current >= self.soft_limit:
            # Calculate degradation between soft and hard limit
            range_size = self.hard_limit - self.soft_limit
            position = current - self.soft_limit
            degradation_level = min(
                self.degradation_levels,
                int((position / range_size) * self.degradation_levels) + 1
            )

        # Enhance result with degradation info
        result_dict = result.to_dict()
        result_dict["degradation_level"] = degradation_level
        result_dict["max_degradation"] = self.degradation_levels
        result_dict["soft_limit"] = self.soft_limit
        result_dict["quality_factor"] = 1.0 - (degradation_level / self.degradation_levels)

        return result

    async def reset(self, key: str) -> None:
        """Reset counter."""
        await self._limiter.reset(key)

    def get_degradation_action(self, level: int) -> Dict[str, Any]:
        """Get suggested actions for degradation level."""
        actions = {
            0: {"action": "full_service", "description": "Full service quality"},
            1: {"action": "reduce_features", "description": "Disable non-essential features"},
            2: {"action": "simplify_response", "description": "Use simplified responses"},
            3: {"action": "increase_latency", "description": "Add artificial delay"},
            4: {"action": "reduce_accuracy", "description": "Use faster, less accurate models"},
            5: {"action": "minimal_service", "description": "Minimal functionality only"},
        }
        return actions.get(level, actions[5])


class FairShareLimiter(RateLimiter):
    """
    Fair share rate limiter.

    Ensures fair distribution of capacity among all users,
    preventing any single user from monopolizing resources.
    """

    def __init__(
        self,
        total_rate: float = 1000.0,  # Total system rate
        max_users: int = 100,
        min_share: float = 1.0,  # Minimum rate per user
        rebalance_interval: float = 10.0,
    ):
        """
        Args:
            total_rate: Total system capacity
            max_users: Maximum expected concurrent users
            min_share: Minimum guaranteed rate per user
            rebalance_interval: Interval to rebalance shares
        """
        self.total_rate = total_rate
        self.max_users = max_users
        self.min_share = min_share
        self.rebalance_interval = rebalance_interval

        self._user_activity: Dict[str, datetime] = {}
        self._user_limiters: Dict[str, RateLimiter] = {}
        self._lock = asyncio.Lock()

        # Calculate initial share
        self._current_share = total_rate / max_users

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check with fair share limit."""
        async with self._lock:
            # Update activity
            self._user_activity[key] = datetime.utcnow()

            # Get or create limiter
            if key not in self._user_limiters:
                from app.ratelimit.algorithms import TokenBucket
                self._user_limiters[key] = TokenBucket(
                    rate=max(self._current_share, self.min_share),
                    capacity=int(max(self._current_share, self.min_share) * 2)
                )

        return await self._user_limiters[key].check(key, cost)

    async def reset(self, key: str) -> None:
        """Reset user's rate limit."""
        async with self._lock:
            if key in self._user_limiters:
                await self._user_limiters[key].reset(key)

    async def rebalance(self) -> None:
        """Rebalance shares among active users."""
        async with self._lock:
            # Remove inactive users (inactive for more than 5 minutes)
            cutoff = datetime.utcnow() - timedelta(minutes=5)
            inactive = [
                key for key, last_active in self._user_activity.items()
                if last_active < cutoff
            ]
            for key in inactive:
                del self._user_activity[key]
                if key in self._user_limiters:
                    del self._user_limiters[key]

            # Calculate new share
            active_users = len(self._user_activity)
            if active_users > 0:
                self._current_share = self.total_rate / active_users
                self._current_share = max(self._current_share, self.min_share)

                # Update all limiters
                from app.ratelimit.algorithms import TokenBucket
                for key in self._user_limiters:
                    self._user_limiters[key] = TokenBucket(
                        rate=self._current_share,
                        capacity=int(self._current_share * 2)
                    )

            logger.info(
                f"Rebalanced fair share: {self._current_share:.2f} req/s "
                f"for {active_users} active users"
            )

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "total_rate": self.total_rate,
            "current_share": self._current_share,
            "active_users": len(self._user_activity),
            "max_users": self.max_users,
            "min_share": self.min_share,
        }


# Type alias for typing
from typing import Tuple
