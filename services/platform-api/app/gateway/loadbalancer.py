"""
Load Balancer

Traffic distribution strategies:
- Round robin
- Weighted distribution
- Least connections
- Consistent hashing
- Health-aware routing
"""

from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import hashlib
import random
import logging

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    IP_HASH = "ip_hash"
    CONSISTENT_HASH = "consistent_hash"
    LEAST_RESPONSE_TIME = "least_response_time"


class InstanceHealth(str, Enum):
    """Instance health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DRAINING = "draining"


@dataclass
class ServiceInstance:
    """Service instance representation."""
    id: str
    host: str
    port: int
    weight: int = 1
    health: InstanceHealth = InstanceHealth.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Connection tracking
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0

    # Response time tracking
    avg_response_time_ms: float = 0.0
    last_response_time_ms: float = 0.0

    # Health check
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0

    @property
    def url(self) -> str:
        """Get instance URL."""
        scheme = self.metadata.get("scheme", "http")
        return f"{scheme}://{self.host}:{self.port}"

    @property
    def is_healthy(self) -> bool:
        """Check if instance is healthy."""
        return self.health == InstanceHealth.HEALTHY

    def record_request(self, success: bool, response_time_ms: float) -> None:
        """Record request result."""
        self.total_requests += 1

        if success:
            self.consecutive_failures = 0
            # Update response time with exponential moving average
            alpha = 0.3
            self.avg_response_time_ms = (
                alpha * response_time_ms +
                (1 - alpha) * self.avg_response_time_ms
            )
        else:
            self.failed_requests += 1
            self.consecutive_failures += 1

        self.last_response_time_ms = response_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "url": self.url,
            "weight": self.weight,
            "health": self.health.value,
            "active_connections": self.active_connections,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
        }


@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    health_check_enabled: bool = True
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 5
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    connection_draining_seconds: int = 30
    max_connections_per_instance: int = 1000
    retry_failed_instances: bool = True
    retry_delay_seconds: float = 1.0


class LoadBalancer(ABC):
    """Abstract base load balancer."""

    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self._instances: Dict[str, ServiceInstance] = {}
        self._lock = asyncio.Lock()

    @abstractmethod
    async def select(self, key: Optional[str] = None) -> Optional[ServiceInstance]:
        """Select an instance for the request."""
        pass

    def add_instance(self, instance: ServiceInstance) -> None:
        """Add instance to pool."""
        self._instances[instance.id] = instance
        logger.info(f"Added instance: {instance.id} ({instance.url})")

    def remove_instance(self, instance_id: str) -> bool:
        """Remove instance from pool."""
        if instance_id in self._instances:
            del self._instances[instance_id]
            logger.info(f"Removed instance: {instance_id}")
            return True
        return False

    def get_instance(self, instance_id: str) -> Optional[ServiceInstance]:
        """Get instance by ID."""
        return self._instances.get(instance_id)

    def get_healthy_instances(self) -> List[ServiceInstance]:
        """Get all healthy instances."""
        return [
            inst for inst in self._instances.values()
            if inst.is_healthy and inst.health != InstanceHealth.DRAINING
        ]

    def update_health(
        self,
        instance_id: str,
        healthy: bool,
    ) -> None:
        """Update instance health status."""
        instance = self._instances.get(instance_id)
        if not instance:
            return

        if healthy:
            if instance.health != InstanceHealth.HEALTHY:
                instance.consecutive_failures = 0
                instance.health = InstanceHealth.HEALTHY
                logger.info(f"Instance {instance_id} is now healthy")
        else:
            instance.consecutive_failures += 1
            if instance.consecutive_failures >= self.config.unhealthy_threshold:
                instance.health = InstanceHealth.UNHEALTHY
                logger.warning(f"Instance {instance_id} marked unhealthy")

        instance.last_health_check = datetime.utcnow()

    async def acquire_connection(self, instance: ServiceInstance) -> bool:
        """Acquire connection slot."""
        async with self._lock:
            if instance.active_connections >= self.config.max_connections_per_instance:
                return False
            instance.active_connections += 1
            return True

    async def release_connection(self, instance: ServiceInstance) -> None:
        """Release connection slot."""
        async with self._lock:
            instance.active_connections = max(0, instance.active_connections - 1)

    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        healthy = sum(1 for i in self._instances.values() if i.is_healthy)
        total = len(self._instances)
        total_connections = sum(i.active_connections for i in self._instances.values())
        total_requests = sum(i.total_requests for i in self._instances.values())

        return {
            "total_instances": total,
            "healthy_instances": healthy,
            "unhealthy_instances": total - healthy,
            "total_active_connections": total_connections,
            "total_requests": total_requests,
            "strategy": self.config.strategy.value,
            "instances": [i.to_dict() for i in self._instances.values()],
        }


class RoundRobinBalancer(LoadBalancer):
    """
    Round-robin load balancer.

    Distributes requests evenly across instances.
    """

    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        super().__init__(config)
        self._index = 0

    async def select(self, key: Optional[str] = None) -> Optional[ServiceInstance]:
        """Select next instance in rotation."""
        async with self._lock:
            healthy = self.get_healthy_instances()
            if not healthy:
                return None

            instance = healthy[self._index % len(healthy)]
            self._index = (self._index + 1) % len(healthy)
            return instance


class WeightedBalancer(LoadBalancer):
    """
    Weighted load balancer.

    Distributes based on instance weights.
    """

    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        cfg = config or LoadBalancerConfig(strategy=LoadBalancingStrategy.WEIGHTED)
        super().__init__(cfg)
        self._weighted_pool: List[str] = []
        self._pool_index = 0

    def add_instance(self, instance: ServiceInstance) -> None:
        """Add instance with weight consideration."""
        super().add_instance(instance)
        self._rebuild_pool()

    def remove_instance(self, instance_id: str) -> bool:
        """Remove instance and rebuild pool."""
        result = super().remove_instance(instance_id)
        if result:
            self._rebuild_pool()
        return result

    def _rebuild_pool(self) -> None:
        """Rebuild weighted pool."""
        self._weighted_pool = []
        for instance in self._instances.values():
            if instance.is_healthy:
                self._weighted_pool.extend([instance.id] * instance.weight)
        random.shuffle(self._weighted_pool)

    async def select(self, key: Optional[str] = None) -> Optional[ServiceInstance]:
        """Select instance based on weight."""
        async with self._lock:
            if not self._weighted_pool:
                self._rebuild_pool()

            if not self._weighted_pool:
                return None

            instance_id = self._weighted_pool[self._pool_index % len(self._weighted_pool)]
            self._pool_index = (self._pool_index + 1) % len(self._weighted_pool)

            return self._instances.get(instance_id)


class LeastConnectionsBalancer(LoadBalancer):
    """
    Least connections load balancer.

    Routes to instance with fewest active connections.
    """

    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        cfg = config or LoadBalancerConfig(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
        super().__init__(cfg)

    async def select(self, key: Optional[str] = None) -> Optional[ServiceInstance]:
        """Select instance with least connections."""
        async with self._lock:
            healthy = self.get_healthy_instances()
            if not healthy:
                return None

            # Sort by active connections
            sorted_instances = sorted(
                healthy,
                key=lambda i: (i.active_connections, i.avg_response_time_ms)
            )

            return sorted_instances[0]


class ConsistentHashBalancer(LoadBalancer):
    """
    Consistent hash load balancer.

    Routes based on hash ring for session affinity.
    """

    def __init__(
        self,
        config: Optional[LoadBalancerConfig] = None,
        virtual_nodes: int = 150,
    ):
        cfg = config or LoadBalancerConfig(strategy=LoadBalancingStrategy.CONSISTENT_HASH)
        super().__init__(cfg)
        self.virtual_nodes = virtual_nodes
        self._hash_ring: List[tuple] = []  # (hash_value, instance_id)

    def add_instance(self, instance: ServiceInstance) -> None:
        """Add instance to hash ring."""
        super().add_instance(instance)
        self._rebuild_ring()

    def remove_instance(self, instance_id: str) -> bool:
        """Remove instance from ring."""
        result = super().remove_instance(instance_id)
        if result:
            self._rebuild_ring()
        return result

    def _rebuild_ring(self) -> None:
        """Rebuild consistent hash ring."""
        self._hash_ring = []

        for instance in self._instances.values():
            if instance.is_healthy:
                for i in range(self.virtual_nodes):
                    key = f"{instance.id}:{i}"
                    hash_val = self._hash(key)
                    self._hash_ring.append((hash_val, instance.id))

        self._hash_ring.sort(key=lambda x: x[0])

    def _hash(self, key: str) -> int:
        """Generate hash value."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    async def select(self, key: Optional[str] = None) -> Optional[ServiceInstance]:
        """Select instance based on consistent hash."""
        if not self._hash_ring:
            return None

        if not key:
            # Fall back to random if no key
            key = str(random.random())

        hash_val = self._hash(key)

        # Binary search for position in ring
        left, right = 0, len(self._hash_ring)
        while left < right:
            mid = (left + right) // 2
            if self._hash_ring[mid][0] < hash_val:
                left = mid + 1
            else:
                right = mid

        # Wrap around
        idx = left % len(self._hash_ring)
        instance_id = self._hash_ring[idx][1]

        return self._instances.get(instance_id)


class LeastResponseTimeBalancer(LoadBalancer):
    """
    Least response time load balancer.

    Routes to instance with fastest average response time.
    """

    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        cfg = config or LoadBalancerConfig(strategy=LoadBalancingStrategy.LEAST_RESPONSE_TIME)
        super().__init__(cfg)

    async def select(self, key: Optional[str] = None) -> Optional[ServiceInstance]:
        """Select instance with best response time."""
        async with self._lock:
            healthy = self.get_healthy_instances()
            if not healthy:
                return None

            # Sort by response time (new instances get priority)
            sorted_instances = sorted(
                healthy,
                key=lambda i: i.avg_response_time_ms if i.total_requests > 0 else 0
            )

            return sorted_instances[0]


class RandomBalancer(LoadBalancer):
    """
    Random load balancer.

    Randomly selects from healthy instances.
    """

    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        cfg = config or LoadBalancerConfig(strategy=LoadBalancingStrategy.RANDOM)
        super().__init__(cfg)

    async def select(self, key: Optional[str] = None) -> Optional[ServiceInstance]:
        """Randomly select an instance."""
        healthy = self.get_healthy_instances()
        if not healthy:
            return None
        return random.choice(healthy)


class AdaptiveBalancer(LoadBalancer):
    """
    Adaptive load balancer.

    Dynamically adjusts strategy based on metrics.
    """

    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        super().__init__(config)
        self._strategies: Dict[LoadBalancingStrategy, LoadBalancer] = {
            LoadBalancingStrategy.ROUND_ROBIN: RoundRobinBalancer(config),
            LoadBalancingStrategy.LEAST_CONNECTIONS: LeastConnectionsBalancer(config),
            LoadBalancingStrategy.LEAST_RESPONSE_TIME: LeastResponseTimeBalancer(config),
        }
        self._active_strategy = LoadBalancingStrategy.ROUND_ROBIN
        self._metrics_window: List[Dict[str, Any]] = []

    def add_instance(self, instance: ServiceInstance) -> None:
        """Add to all strategies."""
        super().add_instance(instance)
        for strategy in self._strategies.values():
            strategy.add_instance(instance)

    def remove_instance(self, instance_id: str) -> bool:
        """Remove from all strategies."""
        result = super().remove_instance(instance_id)
        for strategy in self._strategies.values():
            strategy.remove_instance(instance_id)
        return result

    async def select(self, key: Optional[str] = None) -> Optional[ServiceInstance]:
        """Select using current strategy."""
        strategy = self._strategies[self._active_strategy]
        return await strategy.select(key)

    def adapt(self) -> None:
        """Adapt strategy based on metrics."""
        # Calculate variance in response times
        instances = list(self._instances.values())
        if len(instances) < 2:
            return

        response_times = [i.avg_response_time_ms for i in instances if i.total_requests > 0]
        if not response_times:
            return

        avg_time = sum(response_times) / len(response_times)
        variance = sum((t - avg_time) ** 2 for t in response_times) / len(response_times)

        # High variance: use least response time
        if variance > 1000:  # High variance threshold
            self._active_strategy = LoadBalancingStrategy.LEAST_RESPONSE_TIME

        # High connections variance: use least connections
        connections = [i.active_connections for i in instances]
        conn_variance = sum((c - sum(connections) / len(connections)) ** 2 for c in connections) / len(connections)
        if conn_variance > 100:
            self._active_strategy = LoadBalancingStrategy.LEAST_CONNECTIONS

        # Default to round robin for even distribution
        else:
            self._active_strategy = LoadBalancingStrategy.ROUND_ROBIN


class ServiceLoadBalancer:
    """
    Service-aware load balancer.

    Manages load balancing for multiple services.
    """

    def __init__(
        self,
        default_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
    ):
        self.default_strategy = default_strategy
        self._balancers: Dict[str, LoadBalancer] = {}
        self._service_strategies: Dict[str, LoadBalancingStrategy] = {}

    def _create_balancer(self, strategy: LoadBalancingStrategy) -> LoadBalancer:
        """Create balancer for strategy."""
        balancers = {
            LoadBalancingStrategy.ROUND_ROBIN: RoundRobinBalancer,
            LoadBalancingStrategy.WEIGHTED: WeightedBalancer,
            LoadBalancingStrategy.LEAST_CONNECTIONS: LeastConnectionsBalancer,
            LoadBalancingStrategy.CONSISTENT_HASH: ConsistentHashBalancer,
            LoadBalancingStrategy.LEAST_RESPONSE_TIME: LeastResponseTimeBalancer,
            LoadBalancingStrategy.RANDOM: RandomBalancer,
        }
        return balancers.get(strategy, RoundRobinBalancer)()

    def get_balancer(self, service_name: str) -> LoadBalancer:
        """Get or create balancer for service."""
        if service_name not in self._balancers:
            strategy = self._service_strategies.get(service_name, self.default_strategy)
            self._balancers[service_name] = self._create_balancer(strategy)
        return self._balancers[service_name]

    def set_strategy(
        self,
        service_name: str,
        strategy: LoadBalancingStrategy,
    ) -> None:
        """Set strategy for service."""
        self._service_strategies[service_name] = strategy

        # Recreate balancer with new strategy
        if service_name in self._balancers:
            old_balancer = self._balancers[service_name]
            new_balancer = self._create_balancer(strategy)

            # Transfer instances
            for instance in old_balancer._instances.values():
                new_balancer.add_instance(instance)

            self._balancers[service_name] = new_balancer

    def add_instance(
        self,
        service_name: str,
        instance: ServiceInstance,
    ) -> None:
        """Add instance to service."""
        balancer = self.get_balancer(service_name)
        balancer.add_instance(instance)

    def remove_instance(
        self,
        service_name: str,
        instance_id: str,
    ) -> bool:
        """Remove instance from service."""
        balancer = self.get_balancer(service_name)
        return balancer.remove_instance(instance_id)

    async def select(
        self,
        service_name: str,
        key: Optional[str] = None,
    ) -> Optional[ServiceInstance]:
        """Select instance for service."""
        balancer = self.get_balancer(service_name)
        return await balancer.select(key)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all services."""
        return {
            service: balancer.get_stats()
            for service, balancer in self._balancers.items()
        }
