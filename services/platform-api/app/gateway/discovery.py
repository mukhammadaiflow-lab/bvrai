"""
Service Discovery

Service registration and discovery:
- Static configuration
- Dynamic discovery (Consul, etc.)
- Health checking
- Service metadata
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging

from app.gateway.loadbalancer import ServiceInstance, InstanceHealth

logger = logging.getLogger(__name__)


class ServiceHealth(str, Enum):
    """Service overall health."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceDefinition:
    """Service definition."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Health thresholds
    min_healthy_instances: int = 1
    healthy_percentage: float = 0.5

    # Discovery
    health_check_path: str = "/health"
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class ServiceStatus:
    """Service status information."""
    service_name: str
    health: ServiceHealth = ServiceHealth.UNKNOWN
    total_instances: int = 0
    healthy_instances: int = 0
    unhealthy_instances: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "health": self.health.value,
            "total_instances": self.total_instances,
            "healthy_instances": self.healthy_instances,
            "unhealthy_instances": self.unhealthy_instances,
            "last_updated": self.last_updated.isoformat(),
        }


class ServiceRegistry(ABC):
    """Abstract service registry."""

    @abstractmethod
    async def register(
        self,
        service: ServiceDefinition,
        instance: ServiceInstance,
    ) -> bool:
        """Register service instance."""
        pass

    @abstractmethod
    async def deregister(
        self,
        service_name: str,
        instance_id: str,
    ) -> bool:
        """Deregister service instance."""
        pass

    @abstractmethod
    async def get_instances(
        self,
        service_name: str,
        healthy_only: bool = True,
    ) -> List[ServiceInstance]:
        """Get service instances."""
        pass

    @abstractmethod
    async def get_service_status(
        self,
        service_name: str,
    ) -> ServiceStatus:
        """Get service status."""
        pass


class StaticServiceDiscovery(ServiceRegistry):
    """
    Static service discovery.

    Uses configuration-based service definitions.
    """

    def __init__(self):
        self._services: Dict[str, ServiceDefinition] = {}
        self._instances: Dict[str, Dict[str, ServiceInstance]] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        service: ServiceDefinition,
        instance: ServiceInstance,
    ) -> bool:
        """Register service instance."""
        async with self._lock:
            if service.name not in self._services:
                self._services[service.name] = service
                self._instances[service.name] = {}

            self._instances[service.name][instance.id] = instance
            instance.health = InstanceHealth.HEALTHY

            logger.info(f"Registered instance {instance.id} for service {service.name}")
            return True

    async def deregister(
        self,
        service_name: str,
        instance_id: str,
    ) -> bool:
        """Deregister service instance."""
        async with self._lock:
            if service_name not in self._instances:
                return False

            if instance_id in self._instances[service_name]:
                del self._instances[service_name][instance_id]
                logger.info(f"Deregistered instance {instance_id} from service {service_name}")
                return True

            return False

    async def get_instances(
        self,
        service_name: str,
        healthy_only: bool = True,
    ) -> List[ServiceInstance]:
        """Get service instances."""
        if service_name not in self._instances:
            return []

        instances = list(self._instances[service_name].values())

        if healthy_only:
            instances = [i for i in instances if i.is_healthy]

        return instances

    async def get_service_status(
        self,
        service_name: str,
    ) -> ServiceStatus:
        """Get service status."""
        instances = await self.get_instances(service_name, healthy_only=False)
        healthy = sum(1 for i in instances if i.is_healthy)
        total = len(instances)

        # Determine health
        service = self._services.get(service_name)
        if not service:
            health = ServiceHealth.UNKNOWN
        elif healthy == 0:
            health = ServiceHealth.UNHEALTHY
        elif healthy < service.min_healthy_instances:
            health = ServiceHealth.DEGRADED
        elif healthy / total < service.healthy_percentage:
            health = ServiceHealth.DEGRADED
        else:
            health = ServiceHealth.HEALTHY

        return ServiceStatus(
            service_name=service_name,
            health=health,
            total_instances=total,
            healthy_instances=healthy,
            unhealthy_instances=total - healthy,
        )

    async def get_all_services(self) -> List[str]:
        """Get all service names."""
        return list(self._services.keys())

    async def get_service_definition(
        self,
        service_name: str,
    ) -> Optional[ServiceDefinition]:
        """Get service definition."""
        return self._services.get(service_name)

    async def update_instance_health(
        self,
        service_name: str,
        instance_id: str,
        healthy: bool,
    ) -> None:
        """Update instance health."""
        if service_name in self._instances:
            instance = self._instances[service_name].get(instance_id)
            if instance:
                instance.health = InstanceHealth.HEALTHY if healthy else InstanceHealth.UNHEALTHY
                instance.last_health_check = datetime.utcnow()


class ConsulServiceDiscovery(ServiceRegistry):
    """
    Consul-based service discovery.

    Integrates with HashiCorp Consul.
    """

    def __init__(
        self,
        consul_url: str = "http://localhost:8500",
        token: Optional[str] = None,
        datacenter: str = "dc1",
    ):
        self.consul_url = consul_url
        self.token = token
        self.datacenter = datacenter
        self._client = None

    async def connect(self) -> None:
        """Connect to Consul."""
        # In production, use aiohttp or httpx
        logger.info(f"Connecting to Consul at {self.consul_url}")

    async def register(
        self,
        service: ServiceDefinition,
        instance: ServiceInstance,
    ) -> bool:
        """Register service with Consul."""
        registration = {
            "ID": instance.id,
            "Name": service.name,
            "Tags": service.tags,
            "Address": instance.host,
            "Port": instance.port,
            "Meta": {
                **service.metadata,
                **instance.metadata,
                "version": service.version,
            },
            "Check": {
                "HTTP": f"{instance.url}{service.health_check_path}",
                "Interval": f"{service.health_check_interval_seconds}s",
                "Timeout": f"{service.health_check_timeout_seconds}s",
            },
        }

        # In production: await self._put(f"/v1/agent/service/register", registration)
        logger.info(f"Would register with Consul: {registration}")
        return True

    async def deregister(
        self,
        service_name: str,
        instance_id: str,
    ) -> bool:
        """Deregister service from Consul."""
        # In production: await self._put(f"/v1/agent/service/deregister/{instance_id}")
        logger.info(f"Would deregister {instance_id} from Consul")
        return True

    async def get_instances(
        self,
        service_name: str,
        healthy_only: bool = True,
    ) -> List[ServiceInstance]:
        """Get service instances from Consul."""
        # In production: query Consul catalog
        # endpoint = f"/v1/health/service/{service_name}"
        # if healthy_only:
        #     endpoint += "?passing"
        return []

    async def get_service_status(
        self,
        service_name: str,
    ) -> ServiceStatus:
        """Get service status from Consul."""
        instances = await self.get_instances(service_name, healthy_only=False)
        healthy = sum(1 for i in instances if i.is_healthy)

        return ServiceStatus(
            service_name=service_name,
            health=ServiceHealth.HEALTHY if healthy > 0 else ServiceHealth.UNHEALTHY,
            total_instances=len(instances),
            healthy_instances=healthy,
            unhealthy_instances=len(instances) - healthy,
        )


class HealthChecker:
    """
    Service health checker.

    Performs periodic health checks on service instances.
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        check_interval_seconds: int = 30,
        check_timeout_seconds: int = 5,
    ):
        self.registry = registry
        self.check_interval = check_interval_seconds
        self.check_timeout = check_timeout_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[str, str, bool], Awaitable[None]]] = []

    async def start(self) -> None:
        """Start health checking."""
        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info("Health checker started")

    async def stop(self) -> None:
        """Stop health checking."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")

    def on_health_change(
        self,
        callback: Callable[[str, str, bool], Awaitable[None]],
    ) -> None:
        """Register health change callback."""
        self._callbacks.append(callback)

    async def _check_loop(self) -> None:
        """Health check loop."""
        while self._running:
            try:
                await self._check_all_services()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_all_services(self) -> None:
        """Check all registered services."""
        if isinstance(self.registry, StaticServiceDiscovery):
            services = await self.registry.get_all_services()

            for service_name in services:
                instances = await self.registry.get_instances(
                    service_name, healthy_only=False
                )

                for instance in instances:
                    healthy = await self._check_instance(instance)
                    await self.registry.update_instance_health(
                        service_name, instance.id, healthy
                    )

                    for callback in self._callbacks:
                        await callback(service_name, instance.id, healthy)

    async def _check_instance(self, instance: ServiceInstance) -> bool:
        """Check single instance health."""
        health_url = f"{instance.url}/health"

        try:
            # In production, use aiohttp
            # async with aiohttp.ClientSession() as session:
            #     async with session.get(health_url, timeout=self.check_timeout) as resp:
            #         return resp.status == 200

            # Simulate health check
            return instance.health == InstanceHealth.HEALTHY

        except Exception as e:
            logger.debug(f"Health check failed for {instance.id}: {e}")
            return False


class ServiceDiscoveryManager:
    """
    Manages service discovery across multiple backends.
    """

    def __init__(
        self,
        primary_registry: Optional[ServiceRegistry] = None,
    ):
        self.primary = primary_registry or StaticServiceDiscovery()
        self._health_checker: Optional[HealthChecker] = None
        self._watchers: Dict[str, List[Callable]] = {}

    async def start(self) -> None:
        """Start discovery manager."""
        self._health_checker = HealthChecker(self.primary)
        self._health_checker.on_health_change(self._on_health_change)
        await self._health_checker.start()

    async def stop(self) -> None:
        """Stop discovery manager."""
        if self._health_checker:
            await self._health_checker.stop()

    async def register_service(
        self,
        name: str,
        host: str,
        port: int,
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a service instance."""
        import uuid

        service = ServiceDefinition(
            name=name,
            version=version,
            tags=tags or [],
            metadata=metadata or {},
        )

        instance = ServiceInstance(
            id=str(uuid.uuid4()),
            host=host,
            port=port,
            metadata=metadata or {},
        )

        await self.primary.register(service, instance)
        return instance.id

    async def deregister_service(
        self,
        service_name: str,
        instance_id: str,
    ) -> bool:
        """Deregister a service instance."""
        return await self.primary.deregister(service_name, instance_id)

    async def discover(
        self,
        service_name: str,
        healthy_only: bool = True,
    ) -> List[ServiceInstance]:
        """Discover service instances."""
        return await self.primary.get_instances(service_name, healthy_only)

    async def get_status(self, service_name: str) -> ServiceStatus:
        """Get service status."""
        return await self.primary.get_service_status(service_name)

    def watch(
        self,
        service_name: str,
        callback: Callable[[List[ServiceInstance]], None],
    ) -> None:
        """Watch service for changes."""
        if service_name not in self._watchers:
            self._watchers[service_name] = []
        self._watchers[service_name].append(callback)

    async def _on_health_change(
        self,
        service_name: str,
        instance_id: str,
        healthy: bool,
    ) -> None:
        """Handle health change."""
        if service_name in self._watchers:
            instances = await self.discover(service_name)
            for callback in self._watchers[service_name]:
                try:
                    callback(instances)
                except Exception as e:
                    logger.error(f"Watch callback error: {e}")
