"""
Service Discovery and Registry
==============================

Enterprise-grade service registry for dynamic service discovery,
load balancing, and health-aware routing.

Features:
- Service registration and discovery
- Health monitoring
- Multiple load balancing strategies
- Circuit breaker integration
- Service versioning
- Metadata and tagging
- Cluster awareness

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
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
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# ENUMS
# =============================================================================


class ServiceHealth(str, Enum):
    """Service health status"""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"


class LoadBalancerStrategy(str, Enum):
    """Load balancing strategies"""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    IP_HASH = "ip_hash"
    LEAST_LATENCY = "least_latency"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"


class HealthCheckType(str, Enum):
    """Types of health checks"""

    HTTP = "http"
    TCP = "tcp"
    GRPC = "grpc"
    CUSTOM = "custom"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class HealthCheck:
    """Health check configuration"""

    type: HealthCheckType = HealthCheckType.HTTP
    endpoint: str = "/health"
    interval_seconds: float = 10.0
    timeout_seconds: float = 5.0
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    expected_status: int = 200


@dataclass
class ServiceInstance:
    """
    Represents a single instance of a service.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    service_name: str = ""
    host: str = "localhost"
    port: int = 8080
    scheme: str = "http"
    version: str = "1.0.0"

    # Health
    health: ServiceHealth = ServiceHealth.UNKNOWN
    health_check: HealthCheck = field(default_factory=HealthCheck)
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    # Load balancing
    weight: int = 100
    active_connections: int = 0
    total_requests: int = 0
    total_failures: int = 0
    avg_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    zone: str = "default"
    region: str = "default"

    # Timestamps
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_seen_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    @property
    def is_healthy(self) -> bool:
        return self.health in (ServiceHealth.HEALTHY, ServiceHealth.DEGRADED)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.total_failures) / self.total_requests

    def record_request(self, latency_ms: float, success: bool = True) -> None:
        """Record a request for metrics"""
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.total_requests
        self.last_seen_at = datetime.utcnow()

        if not success:
            self.total_failures += 1


@dataclass
class ServiceDefinition:
    """
    Definition of a service in the registry.
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    instances: Dict[str, ServiceInstance] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    load_balancer: LoadBalancerStrategy = LoadBalancerStrategy.ROUND_ROBIN
    health_check: HealthCheck = field(default_factory=HealthCheck)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def healthy_instances(self) -> List[ServiceInstance]:
        return [i for i in self.instances.values() if i.is_healthy]

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    @property
    def healthy_count(self) -> int:
        return len(self.healthy_instances)


# =============================================================================
# LOAD BALANCERS
# =============================================================================


class LoadBalancer(ABC):
    """Abstract base class for load balancers"""

    @abstractmethod
    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        pass


class RoundRobinBalancer(LoadBalancer):
    """Round-robin load balancer"""

    def __init__(self):
        self._index: Dict[str, int] = defaultdict(int)

    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        if not instances:
            return None

        service_name = instances[0].service_name
        index = self._index[service_name] % len(instances)
        self._index[service_name] += 1

        return instances[index]


class RandomBalancer(LoadBalancer):
    """Random load balancer"""

    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        if not instances:
            return None
        return random.choice(instances)


class LeastConnectionsBalancer(LoadBalancer):
    """Least connections load balancer"""

    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        if not instances:
            return None
        return min(instances, key=lambda i: i.active_connections)


class WeightedBalancer(LoadBalancer):
    """Weighted load balancer"""

    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        if not instances:
            return None

        total_weight = sum(i.weight for i in instances)
        if total_weight == 0:
            return random.choice(instances)

        r = random.randint(1, total_weight)
        cumulative = 0

        for instance in instances:
            cumulative += instance.weight
            if r <= cumulative:
                return instance

        return instances[-1]


class LeastLatencyBalancer(LoadBalancer):
    """Least latency load balancer"""

    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        if not instances:
            return None

        # Prefer instances with known latency
        measured = [i for i in instances if i.total_requests > 0]
        if measured:
            return min(measured, key=lambda i: i.avg_latency_ms)

        return random.choice(instances)


class IPHashBalancer(LoadBalancer):
    """IP hash load balancer for sticky sessions"""

    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        if not instances:
            return None

        context = context or {}
        client_ip = context.get("client_ip", str(uuid4()))

        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        index = hash_value % len(instances)

        return instances[index]


# =============================================================================
# HEALTH CHECKER
# =============================================================================


class HealthChecker:
    """
    Performs health checks on service instances.
    """

    def __init__(self):
        self._logger = structlog.get_logger("health_checker")
        self._http_client: Optional[Any] = None

    async def check(self, instance: ServiceInstance) -> ServiceHealth:
        """Perform health check on an instance"""
        config = instance.health_check
        start_time = time.time()

        try:
            if config.type == HealthCheckType.HTTP:
                return await self._check_http(instance, config)
            elif config.type == HealthCheckType.TCP:
                return await self._check_tcp(instance, config)
            else:
                return ServiceHealth.UNKNOWN

        except asyncio.TimeoutError:
            self._logger.warning(
                "health_check_timeout",
                instance_id=instance.id,
                service=instance.service_name
            )
            return ServiceHealth.UNHEALTHY

        except Exception as e:
            self._logger.error(
                "health_check_error",
                instance_id=instance.id,
                error=str(e)
            )
            return ServiceHealth.UNHEALTHY

        finally:
            latency_ms = (time.time() - start_time) * 1000
            instance.last_health_check = datetime.utcnow()

    async def _check_http(
        self,
        instance: ServiceInstance,
        config: HealthCheck
    ) -> ServiceHealth:
        """HTTP health check"""
        import aiohttp

        url = f"{instance.url}{config.endpoint}"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=config.timeout_seconds)
            ) as response:
                if response.status == config.expected_status:
                    return ServiceHealth.HEALTHY
                elif response.status < 500:
                    return ServiceHealth.DEGRADED
                else:
                    return ServiceHealth.UNHEALTHY

    async def _check_tcp(
        self,
        instance: ServiceInstance,
        config: HealthCheck
    ) -> ServiceHealth:
        """TCP health check"""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(instance.host, instance.port),
                timeout=config.timeout_seconds
            )
            writer.close()
            await writer.wait_closed()
            return ServiceHealth.HEALTHY
        except Exception:
            return ServiceHealth.UNHEALTHY


# =============================================================================
# SERVICE REGISTRY
# =============================================================================


class ServiceRegistry:
    """
    Central service registry for service discovery.

    Manages service registration, discovery, health checking,
    and load balancing.

    Usage:
        registry = ServiceRegistry()
        await registry.start()

        # Register a service instance
        await registry.register(
            service_name="user-service",
            host="10.0.0.1",
            port=8080
        )

        # Discover a service
        instance = await registry.discover("user-service")

        await registry.stop()
    """

    def __init__(
        self,
        health_check_interval: float = 10.0,
        instance_ttl_seconds: float = 60.0,
        enable_health_checks: bool = True
    ):
        self._services: Dict[str, ServiceDefinition] = {}
        self._health_checker = HealthChecker()
        self._load_balancers: Dict[LoadBalancerStrategy, LoadBalancer] = {
            LoadBalancerStrategy.ROUND_ROBIN: RoundRobinBalancer(),
            LoadBalancerStrategy.RANDOM: RandomBalancer(),
            LoadBalancerStrategy.LEAST_CONNECTIONS: LeastConnectionsBalancer(),
            LoadBalancerStrategy.WEIGHTED: WeightedBalancer(),
            LoadBalancerStrategy.LEAST_LATENCY: LeastLatencyBalancer(),
            LoadBalancerStrategy.IP_HASH: IPHashBalancer(),
        }
        self._health_check_interval = health_check_interval
        self._instance_ttl = instance_ttl_seconds
        self._enable_health_checks = enable_health_checks
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger("service_registry")

        # Event handlers
        self._on_register_handlers: List[Callable] = []
        self._on_deregister_handlers: List[Callable] = []
        self._on_health_change_handlers: List[Callable] = []

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the registry"""
        self._running = True

        if self._enable_health_checks:
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )

        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self._logger.info("service_registry_started")

    async def stop(self) -> None:
        """Stop the registry"""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._logger.info("service_registry_stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    async def register(
        self,
        service_name: str,
        host: str,
        port: int,
        scheme: str = "http",
        version: str = "1.0.0",
        weight: int = 100,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        health_check: Optional[HealthCheck] = None,
        zone: str = "default",
        region: str = "default"
    ) -> str:
        """
        Register a service instance.

        Returns:
            Instance ID
        """
        instance = ServiceInstance(
            service_name=service_name,
            host=host,
            port=port,
            scheme=scheme,
            version=version,
            weight=weight,
            tags=tags or set(),
            metadata=metadata or {},
            health_check=health_check or HealthCheck(),
            zone=zone,
            region=region,
            health=ServiceHealth.STARTING
        )

        async with self._lock:
            # Create service definition if needed
            if service_name not in self._services:
                self._services[service_name] = ServiceDefinition(
                    name=service_name,
                    version=version
                )

            service = self._services[service_name]
            service.instances[instance.id] = instance
            service.updated_at = datetime.utcnow()

        # Initial health check
        if self._enable_health_checks:
            health = await self._health_checker.check(instance)
            instance.health = health

        self._logger.info(
            "service_registered",
            service=service_name,
            instance_id=instance.id,
            address=instance.address
        )

        # Notify handlers
        for handler in self._on_register_handlers:
            await handler(instance)

        return instance.id

    async def deregister(
        self,
        service_name: str,
        instance_id: str
    ) -> bool:
        """Deregister a service instance"""
        async with self._lock:
            service = self._services.get(service_name)
            if not service:
                return False

            instance = service.instances.pop(instance_id, None)
            if not instance:
                return False

            self._logger.info(
                "service_deregistered",
                service=service_name,
                instance_id=instance_id
            )

            # Notify handlers
            for handler in self._on_deregister_handlers:
                await handler(instance)

            return True

    async def heartbeat(
        self,
        service_name: str,
        instance_id: str
    ) -> bool:
        """Send a heartbeat for an instance"""
        service = self._services.get(service_name)
        if not service:
            return False

        instance = service.instances.get(instance_id)
        if not instance:
            return False

        instance.last_seen_at = datetime.utcnow()
        return True

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    async def discover(
        self,
        service_name: str,
        version: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        zone: Optional[str] = None,
        strategy: Optional[LoadBalancerStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServiceInstance]:
        """
        Discover a healthy service instance.

        Args:
            service_name: Name of the service
            version: Optional version filter
            tags: Optional tag filter
            zone: Optional zone filter
            strategy: Load balancing strategy override
            context: Context for load balancing (e.g., client_ip)

        Returns:
            Selected service instance or None
        """
        service = self._services.get(service_name)
        if not service:
            self._logger.warning(
                "service_not_found",
                service=service_name
            )
            return None

        # Get healthy instances
        instances = service.healthy_instances

        # Apply filters
        if version:
            instances = [i for i in instances if i.version == version]

        if tags:
            instances = [i for i in instances if tags.issubset(i.tags)]

        if zone:
            instances = [i for i in instances if i.zone == zone]

        if not instances:
            self._logger.warning(
                "no_healthy_instances",
                service=service_name
            )
            return None

        # Select instance using load balancer
        lb_strategy = strategy or service.load_balancer
        balancer = self._load_balancers.get(lb_strategy)

        if not balancer:
            balancer = self._load_balancers[LoadBalancerStrategy.ROUND_ROBIN]

        return balancer.select(instances, context)

    async def discover_all(
        self,
        service_name: str,
        healthy_only: bool = True
    ) -> List[ServiceInstance]:
        """Discover all instances of a service"""
        service = self._services.get(service_name)
        if not service:
            return []

        if healthy_only:
            return service.healthy_instances
        return list(service.instances.values())

    def list_services(self) -> List[str]:
        """List all registered services"""
        return list(self._services.keys())

    def get_service(self, service_name: str) -> Optional[ServiceDefinition]:
        """Get service definition"""
        return self._services.get(service_name)

    # -------------------------------------------------------------------------
    # Health Monitoring
    # -------------------------------------------------------------------------

    async def _health_check_loop(self) -> None:
        """Background loop for health checks"""
        while self._running:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("health_check_loop_error", error=str(e))

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all instances"""
        tasks = []

        for service in self._services.values():
            for instance in service.instances.values():
                tasks.append(self._check_instance_health(instance))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_instance_health(self, instance: ServiceInstance) -> None:
        """Check health of a single instance"""
        old_health = instance.health
        new_health = await self._health_checker.check(instance)

        # Update consecutive counters
        if new_health == ServiceHealth.HEALTHY:
            instance.consecutive_failures = 0
            instance.consecutive_successes += 1
        else:
            instance.consecutive_successes = 0
            instance.consecutive_failures += 1

        # Update health based on thresholds
        config = instance.health_check

        if new_health == ServiceHealth.HEALTHY:
            if instance.consecutive_successes >= config.healthy_threshold:
                instance.health = ServiceHealth.HEALTHY
        else:
            if instance.consecutive_failures >= config.unhealthy_threshold:
                instance.health = ServiceHealth.UNHEALTHY

        # Notify on health change
        if old_health != instance.health:
            self._logger.info(
                "instance_health_changed",
                instance_id=instance.id,
                service=instance.service_name,
                old_health=old_health.value,
                new_health=instance.health.value
            )

            for handler in self._on_health_change_handlers:
                await handler(instance, old_health, instance.health)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        """Background loop for cleaning up stale instances"""
        while self._running:
            try:
                await asyncio.sleep(self._instance_ttl / 2)
                await self._cleanup_stale_instances()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("cleanup_loop_error", error=str(e))

    async def _cleanup_stale_instances(self) -> None:
        """Remove instances that haven't been seen recently"""
        now = datetime.utcnow()
        ttl = timedelta(seconds=self._instance_ttl)

        async with self._lock:
            for service in self._services.values():
                stale_ids = [
                    instance_id
                    for instance_id, instance in service.instances.items()
                    if now - instance.last_seen_at > ttl
                ]

                for instance_id in stale_ids:
                    instance = service.instances.pop(instance_id)
                    self._logger.info(
                        "instance_expired",
                        service=service.name,
                        instance_id=instance_id
                    )

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def on_register(
        self,
        handler: Callable[[ServiceInstance], Awaitable[None]]
    ) -> None:
        """Register a handler for service registration events"""
        self._on_register_handlers.append(handler)

    def on_deregister(
        self,
        handler: Callable[[ServiceInstance], Awaitable[None]]
    ) -> None:
        """Register a handler for service deregistration events"""
        self._on_deregister_handlers.append(handler)

    def on_health_change(
        self,
        handler: Callable[[ServiceInstance, ServiceHealth, ServiceHealth], Awaitable[None]]
    ) -> None:
        """Register a handler for health change events"""
        self._on_health_change_handlers.append(handler)

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get registry status"""
        return {
            "running": self._running,
            "services": {
                name: {
                    "instance_count": service.instance_count,
                    "healthy_count": service.healthy_count,
                    "load_balancer": service.load_balancer.value,
                    "instances": {
                        inst_id: {
                            "address": inst.address,
                            "health": inst.health.value,
                            "weight": inst.weight,
                            "total_requests": inst.total_requests,
                            "avg_latency_ms": inst.avg_latency_ms,
                            "success_rate": inst.success_rate
                        }
                        for inst_id, inst in service.instances.items()
                    }
                }
                for name, service in self._services.items()
            }
        }
