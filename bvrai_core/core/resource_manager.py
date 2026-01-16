"""
Resource Manager
================

Enterprise-grade resource management system for platform resource
allocation, pooling, scaling, and optimization.

Features:
- Resource pooling with configurable limits
- Dynamic scaling based on demand
- Resource quotas and rate limiting
- Priority-based allocation
- Resource health monitoring
- Cost tracking and optimization

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from heapq import heappush, heappop
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

T = TypeVar("T")
ResourceT = TypeVar("ResourceT", bound="Resource")


# =============================================================================
# ENUMS
# =============================================================================


class ResourceType(str, Enum):
    """Types of platform resources"""

    # Compute
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"

    # Network
    CONNECTION = "connection"
    BANDWIDTH = "bandwidth"
    SOCKET = "socket"

    # Storage
    DISK = "disk"
    CACHE = "cache"

    # Application
    WORKER = "worker"
    THREAD = "thread"
    PROCESS = "process"

    # Service
    API_CALL = "api_call"
    LLM_TOKEN = "llm_token"
    STT_MINUTE = "stt_minute"
    TTS_CHARACTER = "tts_character"
    CALL_MINUTE = "call_minute"

    # Custom
    CUSTOM = "custom"


class ResourceState(str, Enum):
    """Resource state"""

    AVAILABLE = "available"
    ALLOCATED = "allocated"
    IN_USE = "in_use"
    RESERVED = "reserved"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"


class AllocationStrategy(str, Enum):
    """Resource allocation strategies"""

    FIRST_AVAILABLE = "first_available"
    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"
    RANDOM = "random"
    PRIORITY_BASED = "priority_based"
    WEIGHTED = "weighted"
    BEST_FIT = "best_fit"


class ScalingDirection(str, Enum):
    """Scaling direction"""

    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingTrigger(str, Enum):
    """What triggers scaling"""

    UTILIZATION = "utilization"
    QUEUE_LENGTH = "queue_length"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    SCHEDULE = "schedule"
    MANUAL = "manual"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class ResourceLimits(BaseModel):
    """Resource limits configuration"""

    max_total: int = 1000
    max_per_user: int = 100
    max_per_organization: int = 500
    min_available: int = 10
    burst_limit: int = 150
    burst_window_seconds: float = 60.0


class ScalingPolicy(BaseModel):
    """Scaling policy configuration"""

    enabled: bool = True
    min_instances: int = 1
    max_instances: int = 100
    target_utilization: float = 0.7
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_up_cooldown_seconds: float = 60.0
    scale_down_cooldown_seconds: float = 300.0
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    trigger: ScalingTrigger = ScalingTrigger.UTILIZATION

    # Predictive scaling
    enable_predictive: bool = False
    prediction_window_minutes: int = 30
    prediction_buffer_percent: float = 0.2


class QuotaConfig(BaseModel):
    """Quota configuration"""

    # Rate limits
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    requests_per_day: int = 10000

    # Resource limits
    max_concurrent: int = 10
    max_tokens_per_request: int = 4000
    max_call_duration_seconds: int = 3600

    # Cost limits
    daily_cost_limit: float = 100.0
    monthly_cost_limit: float = 1000.0


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Resource:
    """
    Base resource class.

    Represents a platform resource that can be allocated and managed.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    type: ResourceType = ResourceType.CUSTOM
    name: str = ""
    capacity: float = 1.0
    used: float = 0.0
    state: ResourceState = ResourceState.AVAILABLE
    owner_id: Optional[str] = None
    organization_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    cost_per_unit: float = 0.0
    priority: int = 0

    @property
    def available(self) -> float:
        return max(0, self.capacity - self.used)

    @property
    def utilization(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return self.used / self.capacity

    @property
    def is_available(self) -> bool:
        return self.state == ResourceState.AVAILABLE and self.available > 0

    @property
    def is_expired(self) -> bool:
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False

    def allocate(self, amount: float = 1.0, owner_id: Optional[str] = None) -> bool:
        """Allocate resource"""
        if amount > self.available:
            return False

        self.used += amount
        self.owner_id = owner_id
        self.last_used_at = datetime.utcnow()

        if self.available <= 0:
            self.state = ResourceState.ALLOCATED

        return True

    def release(self, amount: Optional[float] = None) -> None:
        """Release resource"""
        if amount is None:
            amount = self.used

        self.used = max(0, self.used - amount)

        if self.used == 0:
            self.owner_id = None
            self.state = ResourceState.AVAILABLE


@dataclass
class ResourceAllocation:
    """Represents a resource allocation"""

    id: str = field(default_factory=lambda: str(uuid4()))
    resource_id: str = ""
    resource_type: ResourceType = ResourceType.CUSTOM
    amount: float = 1.0
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    purpose: str = ""
    priority: int = 0
    allocated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    released_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.released_at is None

    @property
    def duration_seconds(self) -> float:
        end_time = self.released_at or datetime.utcnow()
        return (end_time - self.allocated_at).total_seconds()


@dataclass
class ResourcePool:
    """
    Pool of resources of the same type.

    Manages a collection of resources with pooling, allocation,
    and scaling capabilities.
    """

    name: str
    resource_type: ResourceType
    resources: Dict[str, Resource] = field(default_factory=dict)
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    scaling_policy: ScalingPolicy = field(default_factory=ScalingPolicy)
    allocation_strategy: AllocationStrategy = AllocationStrategy.LEAST_USED
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Allocation tracking
    allocations: Dict[str, ResourceAllocation] = field(default_factory=dict)
    _round_robin_index: int = field(default=0, repr=False)

    @property
    def total_capacity(self) -> float:
        return sum(r.capacity for r in self.resources.values())

    @property
    def total_used(self) -> float:
        return sum(r.used for r in self.resources.values())

    @property
    def total_available(self) -> float:
        return self.total_capacity - self.total_used

    @property
    def utilization(self) -> float:
        if self.total_capacity <= 0:
            return 0.0
        return self.total_used / self.total_capacity

    @property
    def available_resources(self) -> List[Resource]:
        return [r for r in self.resources.values() if r.is_available]

    def add_resource(self, resource: Resource) -> None:
        """Add a resource to the pool"""
        resource.type = self.resource_type
        self.resources[resource.id] = resource

    def remove_resource(self, resource_id: str) -> Optional[Resource]:
        """Remove a resource from the pool"""
        return self.resources.pop(resource_id, None)

    def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a resource by ID"""
        return self.resources.get(resource_id)


@dataclass
class QuotaUsage:
    """Quota usage tracking"""

    user_id: str
    organization_id: str
    resource_type: ResourceType
    minute_count: int = 0
    hour_count: int = 0
    day_count: int = 0
    month_count: int = 0
    minute_reset: datetime = field(default_factory=datetime.utcnow)
    hour_reset: datetime = field(default_factory=datetime.utcnow)
    day_reset: datetime = field(default_factory=datetime.utcnow)
    month_reset: datetime = field(default_factory=datetime.utcnow)
    total_cost: float = 0.0
    daily_cost: float = 0.0
    monthly_cost: float = 0.0

    def increment(self, amount: int = 1, cost: float = 0.0) -> None:
        """Increment usage counts"""
        now = datetime.utcnow()

        # Reset if needed
        if now - self.minute_reset > timedelta(minutes=1):
            self.minute_count = 0
            self.minute_reset = now

        if now - self.hour_reset > timedelta(hours=1):
            self.hour_count = 0
            self.hour_reset = now

        if now - self.day_reset > timedelta(days=1):
            self.day_count = 0
            self.daily_cost = 0.0
            self.day_reset = now

        if now - self.month_reset > timedelta(days=30):
            self.month_count = 0
            self.monthly_cost = 0.0
            self.month_reset = now

        self.minute_count += amount
        self.hour_count += amount
        self.day_count += amount
        self.month_count += amount
        self.total_cost += cost
        self.daily_cost += cost
        self.monthly_cost += cost


@dataclass
class ResourceMetrics:
    """Resource pool metrics"""

    pool_name: str
    resource_type: ResourceType
    total_resources: int = 0
    available_resources: int = 0
    total_capacity: float = 0.0
    used_capacity: float = 0.0
    utilization: float = 0.0
    allocations_total: int = 0
    allocations_active: int = 0
    allocations_per_minute: float = 0.0
    avg_allocation_duration_ms: float = 0.0
    scale_events: int = 0
    last_scale_time: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# RESOURCE FACTORY
# =============================================================================


class ResourceFactory:
    """Factory for creating resources"""

    @staticmethod
    def create_worker(
        capacity: int = 1,
        **kwargs
    ) -> Resource:
        """Create a worker resource"""
        return Resource(
            type=ResourceType.WORKER,
            name=kwargs.get("name", f"worker-{uuid4().hex[:8]}"),
            capacity=capacity,
            **kwargs
        )

    @staticmethod
    def create_connection(
        max_connections: int = 100,
        **kwargs
    ) -> Resource:
        """Create a connection pool resource"""
        return Resource(
            type=ResourceType.CONNECTION,
            name=kwargs.get("name", f"conn-pool-{uuid4().hex[:8]}"),
            capacity=max_connections,
            **kwargs
        )

    @staticmethod
    def create_api_quota(
        calls_per_minute: int = 100,
        cost_per_call: float = 0.001,
        **kwargs
    ) -> Resource:
        """Create an API quota resource"""
        return Resource(
            type=ResourceType.API_CALL,
            name=kwargs.get("name", f"api-quota-{uuid4().hex[:8]}"),
            capacity=calls_per_minute,
            cost_per_unit=cost_per_call,
            **kwargs
        )

    @staticmethod
    def create_token_quota(
        tokens_per_request: int = 4000,
        cost_per_token: float = 0.00001,
        **kwargs
    ) -> Resource:
        """Create an LLM token quota resource"""
        return Resource(
            type=ResourceType.LLM_TOKEN,
            name=kwargs.get("name", f"token-quota-{uuid4().hex[:8]}"),
            capacity=tokens_per_request,
            cost_per_unit=cost_per_token,
            **kwargs
        )


# =============================================================================
# RESOURCE ALLOCATOR
# =============================================================================


class ResourceAllocator:
    """
    Allocates resources from pools based on strategy.
    """

    def __init__(self, pool: ResourcePool):
        self.pool = pool
        self._logger = structlog.get_logger(f"allocator.{pool.name}")

    def allocate(
        self,
        amount: float = 1.0,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        priority: int = 0,
        purpose: str = ""
    ) -> Optional[ResourceAllocation]:
        """Allocate a resource based on pool strategy"""
        resource = self._select_resource(amount, priority)

        if not resource:
            self._logger.warning(
                "allocation_failed_no_resource",
                amount=amount,
                available=self.pool.total_available
            )
            return None

        if not resource.allocate(amount, user_id):
            self._logger.warning(
                "allocation_failed",
                resource_id=resource.id,
                amount=amount
            )
            return None

        allocation = ResourceAllocation(
            resource_id=resource.id,
            resource_type=self.pool.resource_type,
            amount=amount,
            user_id=user_id,
            organization_id=organization_id,
            purpose=purpose,
            priority=priority
        )

        self.pool.allocations[allocation.id] = allocation

        self._logger.info(
            "resource_allocated",
            allocation_id=allocation.id,
            resource_id=resource.id,
            amount=amount
        )

        return allocation

    def release(self, allocation_id: str) -> bool:
        """Release an allocation"""
        allocation = self.pool.allocations.get(allocation_id)
        if not allocation or not allocation.is_active:
            return False

        resource = self.pool.get_resource(allocation.resource_id)
        if resource:
            resource.release(allocation.amount)

        allocation.released_at = datetime.utcnow()

        self._logger.info(
            "resource_released",
            allocation_id=allocation_id,
            duration_seconds=allocation.duration_seconds
        )

        return True

    def _select_resource(
        self,
        amount: float,
        priority: int
    ) -> Optional[Resource]:
        """Select a resource based on allocation strategy"""
        available = [
            r for r in self.pool.resources.values()
            if r.is_available and r.available >= amount
        ]

        if not available:
            return None

        strategy = self.pool.allocation_strategy

        if strategy == AllocationStrategy.FIRST_AVAILABLE:
            return available[0]

        elif strategy == AllocationStrategy.ROUND_ROBIN:
            index = self.pool._round_robin_index % len(available)
            self.pool._round_robin_index += 1
            return available[index]

        elif strategy == AllocationStrategy.LEAST_USED:
            return min(available, key=lambda r: r.utilization)

        elif strategy == AllocationStrategy.RANDOM:
            return random.choice(available)

        elif strategy == AllocationStrategy.PRIORITY_BASED:
            return max(available, key=lambda r: r.priority)

        elif strategy == AllocationStrategy.BEST_FIT:
            # Find resource with smallest available capacity that fits
            fitting = [r for r in available if r.available >= amount]
            if fitting:
                return min(fitting, key=lambda r: r.available)
            return None

        return available[0]


# =============================================================================
# AUTO SCALER
# =============================================================================


class AutoScaler:
    """
    Auto-scales resource pools based on policies.
    """

    def __init__(
        self,
        pool: ResourcePool,
        resource_factory: Callable[[], Resource]
    ):
        self.pool = pool
        self.resource_factory = resource_factory
        self._last_scale_up: Optional[datetime] = None
        self._last_scale_down: Optional[datetime] = None
        self._scale_history: List[Dict[str, Any]] = []
        self._logger = structlog.get_logger(f"autoscaler.{pool.name}")

    def evaluate(self) -> ScalingDirection:
        """Evaluate if scaling is needed"""
        policy = self.pool.scaling_policy

        if not policy.enabled:
            return ScalingDirection.NONE

        utilization = self.pool.utilization
        current_count = len(self.pool.resources)

        # Check if we should scale up
        if utilization >= policy.scale_up_threshold:
            if current_count < policy.max_instances:
                if self._can_scale_up():
                    return ScalingDirection.UP

        # Check if we should scale down
        elif utilization <= policy.scale_down_threshold:
            if current_count > policy.min_instances:
                if self._can_scale_down():
                    return ScalingDirection.DOWN

        return ScalingDirection.NONE

    def scale(self, direction: ScalingDirection) -> int:
        """Execute scaling operation"""
        policy = self.pool.scaling_policy

        if direction == ScalingDirection.UP:
            return self._scale_up(policy.scale_up_increment)
        elif direction == ScalingDirection.DOWN:
            return self._scale_down(policy.scale_down_increment)

        return 0

    def _can_scale_up(self) -> bool:
        """Check if scale up cooldown has passed"""
        if not self._last_scale_up:
            return True

        cooldown = self.pool.scaling_policy.scale_up_cooldown_seconds
        elapsed = (datetime.utcnow() - self._last_scale_up).total_seconds()
        return elapsed >= cooldown

    def _can_scale_down(self) -> bool:
        """Check if scale down cooldown has passed"""
        if not self._last_scale_down:
            return True

        cooldown = self.pool.scaling_policy.scale_down_cooldown_seconds
        elapsed = (datetime.utcnow() - self._last_scale_down).total_seconds()
        return elapsed >= cooldown

    def _scale_up(self, count: int) -> int:
        """Scale up by adding resources"""
        added = 0
        policy = self.pool.scaling_policy

        for _ in range(count):
            if len(self.pool.resources) >= policy.max_instances:
                break

            resource = self.resource_factory()
            self.pool.add_resource(resource)
            added += 1

        if added > 0:
            self._last_scale_up = datetime.utcnow()
            self._record_scale_event(ScalingDirection.UP, added)
            self._logger.info(
                "scaled_up",
                count=added,
                total=len(self.pool.resources)
            )

        return added

    def _scale_down(self, count: int) -> int:
        """Scale down by removing resources"""
        removed = 0
        policy = self.pool.scaling_policy

        # Find resources to remove (prefer unused)
        available = sorted(
            self.pool.available_resources,
            key=lambda r: r.utilization
        )

        for resource in available[:count]:
            if len(self.pool.resources) <= policy.min_instances:
                break

            if resource.available == resource.capacity:
                self.pool.remove_resource(resource.id)
                removed += 1

        if removed > 0:
            self._last_scale_down = datetime.utcnow()
            self._record_scale_event(ScalingDirection.DOWN, removed)
            self._logger.info(
                "scaled_down",
                count=removed,
                total=len(self.pool.resources)
            )

        return removed

    def _record_scale_event(
        self,
        direction: ScalingDirection,
        count: int
    ) -> None:
        """Record a scale event"""
        self._scale_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "direction": direction.value,
            "count": count,
            "utilization": self.pool.utilization,
            "total_resources": len(self.pool.resources)
        })

        # Keep last 100 events
        if len(self._scale_history) > 100:
            self._scale_history = self._scale_history[-100:]


# =============================================================================
# QUOTA MANAGER
# =============================================================================


class QuotaManager:
    """
    Manages resource quotas and rate limiting.
    """

    def __init__(self):
        self._quotas: Dict[str, QuotaConfig] = {}
        self._usage: Dict[Tuple[str, str, ResourceType], QuotaUsage] = {}
        self._logger = structlog.get_logger("quota_manager")

    def set_quota(
        self,
        organization_id: str,
        quota: QuotaConfig
    ) -> None:
        """Set quota for an organization"""
        self._quotas[organization_id] = quota

    def get_quota(self, organization_id: str) -> QuotaConfig:
        """Get quota for an organization"""
        return self._quotas.get(organization_id, QuotaConfig())

    def check_quota(
        self,
        user_id: str,
        organization_id: str,
        resource_type: ResourceType,
        amount: int = 1
    ) -> Tuple[bool, str]:
        """Check if quota allows the operation"""
        quota = self.get_quota(organization_id)
        usage = self._get_or_create_usage(user_id, organization_id, resource_type)

        # Check rate limits
        if usage.minute_count + amount > quota.requests_per_minute:
            return False, "Rate limit exceeded (per minute)"

        if usage.hour_count + amount > quota.requests_per_hour:
            return False, "Rate limit exceeded (per hour)"

        if usage.day_count + amount > quota.requests_per_day:
            return False, "Rate limit exceeded (per day)"

        # Check cost limits
        if usage.daily_cost >= quota.daily_cost_limit:
            return False, "Daily cost limit exceeded"

        if usage.monthly_cost >= quota.monthly_cost_limit:
            return False, "Monthly cost limit exceeded"

        return True, "OK"

    def record_usage(
        self,
        user_id: str,
        organization_id: str,
        resource_type: ResourceType,
        amount: int = 1,
        cost: float = 0.0
    ) -> None:
        """Record resource usage"""
        usage = self._get_or_create_usage(user_id, organization_id, resource_type)
        usage.increment(amount, cost)

    def get_usage(
        self,
        user_id: str,
        organization_id: str,
        resource_type: ResourceType
    ) -> QuotaUsage:
        """Get current usage"""
        return self._get_or_create_usage(user_id, organization_id, resource_type)

    def _get_or_create_usage(
        self,
        user_id: str,
        organization_id: str,
        resource_type: ResourceType
    ) -> QuotaUsage:
        """Get or create usage tracker"""
        key = (user_id, organization_id, resource_type)
        if key not in self._usage:
            self._usage[key] = QuotaUsage(
                user_id=user_id,
                organization_id=organization_id,
                resource_type=resource_type
            )
        return self._usage[key]


# =============================================================================
# RESOURCE MANAGER
# =============================================================================


class ResourceManager:
    """
    Central resource management system.

    Manages all platform resources including allocation, pooling,
    scaling, quotas, and monitoring.

    Usage:
        manager = ResourceManager()
        await manager.start()

        # Create and add a pool
        pool = manager.create_pool("workers", ResourceType.WORKER)

        # Allocate resources
        allocation = await manager.allocate("workers", amount=1)

        # Release resources
        await manager.release(allocation.id)

        await manager.stop()
    """

    def __init__(
        self,
        auto_scale_interval: float = 30.0,
        metrics_interval: float = 60.0
    ):
        self._pools: Dict[str, ResourcePool] = {}
        self._allocators: Dict[str, ResourceAllocator] = {}
        self._scalers: Dict[str, AutoScaler] = {}
        self._quota_manager = QuotaManager()
        self._running = False
        self._auto_scale_interval = auto_scale_interval
        self._metrics_interval = metrics_interval
        self._auto_scale_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._metrics_history: Dict[str, List[ResourceMetrics]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger("resource_manager")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the resource manager"""
        self._running = True

        # Start auto-scaling loop
        self._auto_scale_task = asyncio.create_task(self._auto_scale_loop())

        # Start metrics collection loop
        self._metrics_task = asyncio.create_task(self._metrics_loop())

        self._logger.info("resource_manager_started")

    async def stop(self) -> None:
        """Stop the resource manager"""
        self._running = False

        if self._auto_scale_task:
            self._auto_scale_task.cancel()
            try:
                await self._auto_scale_task
            except asyncio.CancelledError:
                pass

        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass

        self._logger.info("resource_manager_stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # -------------------------------------------------------------------------
    # Pool Management
    # -------------------------------------------------------------------------

    def create_pool(
        self,
        name: str,
        resource_type: ResourceType,
        limits: Optional[ResourceLimits] = None,
        scaling_policy: Optional[ScalingPolicy] = None,
        allocation_strategy: AllocationStrategy = AllocationStrategy.LEAST_USED,
        resource_factory: Optional[Callable[[], Resource]] = None
    ) -> ResourcePool:
        """Create a new resource pool"""
        pool = ResourcePool(
            name=name,
            resource_type=resource_type,
            limits=limits or ResourceLimits(),
            scaling_policy=scaling_policy or ScalingPolicy(),
            allocation_strategy=allocation_strategy
        )

        self._pools[name] = pool
        self._allocators[name] = ResourceAllocator(pool)

        # Set up auto-scaler if factory provided
        if resource_factory:
            self._scalers[name] = AutoScaler(pool, resource_factory)

        self._logger.info(
            "pool_created",
            name=name,
            type=resource_type.value
        )

        return pool

    def get_pool(self, name: str) -> Optional[ResourcePool]:
        """Get a resource pool by name"""
        return self._pools.get(name)

    def remove_pool(self, name: str) -> bool:
        """Remove a resource pool"""
        if name in self._pools:
            del self._pools[name]
            self._allocators.pop(name, None)
            self._scalers.pop(name, None)
            self._logger.info("pool_removed", name=name)
            return True
        return False

    def add_resource_to_pool(
        self,
        pool_name: str,
        resource: Resource
    ) -> bool:
        """Add a resource to a pool"""
        pool = self._pools.get(pool_name)
        if pool:
            pool.add_resource(resource)
            return True
        return False

    # -------------------------------------------------------------------------
    # Allocation
    # -------------------------------------------------------------------------

    async def allocate(
        self,
        pool_name: str,
        amount: float = 1.0,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        priority: int = 0,
        purpose: str = "",
        timeout: float = 30.0
    ) -> Optional[ResourceAllocation]:
        """
        Allocate resources from a pool.

        Args:
            pool_name: Name of the pool to allocate from
            amount: Amount of resource to allocate
            user_id: User requesting the resource
            organization_id: Organization requesting the resource
            priority: Allocation priority
            purpose: Purpose of allocation
            timeout: Maximum wait time for allocation

        Returns:
            ResourceAllocation if successful, None otherwise
        """
        async with self._lock:
            pool = self._pools.get(pool_name)
            if not pool:
                self._logger.error("pool_not_found", pool_name=pool_name)
                return None

            # Check quota if organization provided
            if organization_id:
                allowed, reason = self._quota_manager.check_quota(
                    user_id or "anonymous",
                    organization_id,
                    pool.resource_type
                )
                if not allowed:
                    self._logger.warning(
                        "quota_exceeded",
                        organization_id=organization_id,
                        reason=reason
                    )
                    return None

            allocator = self._allocators.get(pool_name)
            if not allocator:
                return None

            allocation = allocator.allocate(
                amount=amount,
                user_id=user_id,
                organization_id=organization_id,
                priority=priority,
                purpose=purpose
            )

            if allocation and organization_id:
                # Record usage
                resource = pool.get_resource(allocation.resource_id)
                cost = resource.cost_per_unit * amount if resource else 0.0

                self._quota_manager.record_usage(
                    user_id or "anonymous",
                    organization_id,
                    pool.resource_type,
                    amount=int(amount),
                    cost=cost
                )

            return allocation

    async def release(
        self,
        allocation_id: str,
        pool_name: Optional[str] = None
    ) -> bool:
        """Release an allocation"""
        async with self._lock:
            # Find the allocation in pools
            if pool_name:
                pools_to_check = [pool_name]
            else:
                pools_to_check = list(self._pools.keys())

            for name in pools_to_check:
                allocator = self._allocators.get(name)
                if allocator and allocator.release(allocation_id):
                    return True

            return False

    @asynccontextmanager
    async def acquire(
        self,
        pool_name: str,
        amount: float = 1.0,
        **kwargs
    ):
        """
        Context manager for resource acquisition.

        Usage:
            async with manager.acquire("workers") as allocation:
                # Use the resource
                pass
            # Resource is automatically released
        """
        allocation = await self.allocate(pool_name, amount, **kwargs)
        if not allocation:
            raise RuntimeError(f"Failed to allocate from pool {pool_name}")

        try:
            yield allocation
        finally:
            await self.release(allocation.id, pool_name)

    # -------------------------------------------------------------------------
    # Quota Management
    # -------------------------------------------------------------------------

    def set_quota(
        self,
        organization_id: str,
        quota: QuotaConfig
    ) -> None:
        """Set quota for an organization"""
        self._quota_manager.set_quota(organization_id, quota)

    def get_quota(self, organization_id: str) -> QuotaConfig:
        """Get quota for an organization"""
        return self._quota_manager.get_quota(organization_id)

    def get_usage(
        self,
        user_id: str,
        organization_id: str,
        resource_type: ResourceType
    ) -> QuotaUsage:
        """Get usage for a user/organization"""
        return self._quota_manager.get_usage(
            user_id, organization_id, resource_type
        )

    # -------------------------------------------------------------------------
    # Auto Scaling
    # -------------------------------------------------------------------------

    async def _auto_scale_loop(self) -> None:
        """Background loop for auto-scaling"""
        while self._running:
            try:
                await asyncio.sleep(self._auto_scale_interval)
                await self._evaluate_scaling()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("auto_scale_error", error=str(e))

    async def _evaluate_scaling(self) -> None:
        """Evaluate all pools for scaling"""
        for name, scaler in self._scalers.items():
            direction = scaler.evaluate()
            if direction != ScalingDirection.NONE:
                scaler.scale(direction)

    def trigger_scale(
        self,
        pool_name: str,
        direction: ScalingDirection,
        count: int = 1
    ) -> int:
        """Manually trigger scaling"""
        scaler = self._scalers.get(pool_name)
        if not scaler:
            return 0

        if direction == ScalingDirection.UP:
            return scaler._scale_up(count)
        elif direction == ScalingDirection.DOWN:
            return scaler._scale_down(count)

        return 0

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    async def _metrics_loop(self) -> None:
        """Background loop for metrics collection"""
        while self._running:
            try:
                await asyncio.sleep(self._metrics_interval)
                self._collect_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("metrics_error", error=str(e))

    def _collect_metrics(self) -> None:
        """Collect metrics from all pools"""
        for name, pool in self._pools.items():
            metrics = ResourceMetrics(
                pool_name=name,
                resource_type=pool.resource_type,
                total_resources=len(pool.resources),
                available_resources=len(pool.available_resources),
                total_capacity=pool.total_capacity,
                used_capacity=pool.total_used,
                utilization=pool.utilization,
                allocations_total=len(pool.allocations),
                allocations_active=sum(
                    1 for a in pool.allocations.values() if a.is_active
                )
            )

            self._metrics_history[name].append(metrics)

            # Keep last 1000 metrics per pool
            if len(self._metrics_history[name]) > 1000:
                self._metrics_history[name] = self._metrics_history[name][-1000:]

    def get_metrics(self, pool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current metrics"""
        if pool_name:
            pool = self._pools.get(pool_name)
            if not pool:
                return {}

            return {
                "pool_name": pool_name,
                "resource_type": pool.resource_type.value,
                "total_resources": len(pool.resources),
                "available_resources": len(pool.available_resources),
                "total_capacity": pool.total_capacity,
                "used_capacity": pool.total_used,
                "utilization": pool.utilization,
                "allocations_active": sum(
                    1 for a in pool.allocations.values() if a.is_active
                )
            }

        return {
            name: self.get_metrics(name)
            for name in self._pools
        }

    def get_metrics_history(
        self,
        pool_name: str,
        limit: int = 100
    ) -> List[ResourceMetrics]:
        """Get metrics history for a pool"""
        return self._metrics_history.get(pool_name, [])[-limit:]

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get overall resource manager status"""
        return {
            "running": self._running,
            "pools": {
                name: {
                    "type": pool.resource_type.value,
                    "resources": len(pool.resources),
                    "utilization": pool.utilization,
                    "allocations_active": sum(
                        1 for a in pool.allocations.values() if a.is_active
                    ),
                    "scaling_enabled": pool.scaling_policy.enabled
                }
                for name, pool in self._pools.items()
            },
            "total_resources": sum(
                len(p.resources) for p in self._pools.values()
            ),
            "total_allocations": sum(
                len(p.allocations) for p in self._pools.values()
            )
        }
