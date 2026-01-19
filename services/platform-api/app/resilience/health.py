"""
Health Checking System

Comprehensive health checking for:
- Component health monitoring
- Dependency health tracking
- Aggregated health status
- Liveness and readiness probes
"""

from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ComponentHealth:
    """Health status of a component."""
    name: str
    status: HealthStatus
    last_check: datetime
    consecutive_failures: int = 0
    last_healthy: Optional[datetime] = None
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "last_healthy": self.last_healthy.isoformat() if self.last_healthy else None,
            "last_error": self.last_error,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


@dataclass
class DependencyHealth:
    """Health of an external dependency."""
    name: str
    type: str  # "database", "cache", "api", "queue", etc.
    status: HealthStatus
    endpoint: Optional[str] = None
    latency_ms: Optional[float] = None
    version: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "status": self.status.value,
            "endpoint": self.endpoint,
            "latency_ms": self.latency_ms,
            "version": self.version,
            "last_check": self.last_check.isoformat(),
            "error": self.error,
        }


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    def __init__(
        self,
        name: str,
        timeout_seconds: float = 5.0,
        critical: bool = True,
    ):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.critical = critical

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform health check."""
        pass

    async def execute(self) -> HealthCheckResult:
        """Execute health check with timeout."""
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self.check(),
                timeout=self.timeout_seconds,
            )
            result.latency_ms = (time.time() - start_time) * 1000
            return result

        except asyncio.TimeoutError:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout_seconds}s",
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connections."""

    def __init__(
        self,
        name: str,
        connection_func: Callable,
        query: str = "SELECT 1",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.connection_func = connection_func
        self.query = query

    async def check(self) -> HealthCheckResult:
        """Check database connectivity."""
        try:
            conn = await self.connection_func()
            result = await conn.execute(self.query)

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                details={"query": self.query},
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {e}",
            )


class RedisHealthCheck(HealthCheck):
    """Health check for Redis."""

    def __init__(
        self,
        name: str,
        redis_client: Any,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.redis = redis_client

    async def check(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        try:
            await self.redis.ping()
            info = await self.redis.info()

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Redis connection successful",
                details={
                    "redis_version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                },
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Redis check failed: {e}",
            )


class HTTPHealthCheck(HealthCheck):
    """Health check for HTTP endpoints."""

    def __init__(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        expected_body: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.url = url
        self.expected_status = expected_status
        self.expected_body = expected_body
        self.headers = headers or {}

    async def check(self) -> HealthCheckResult:
        """Check HTTP endpoint."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.url,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                ) as response:
                    body = await response.text()

                    if response.status != self.expected_status:
                        return HealthCheckResult(
                            status=HealthStatus.UNHEALTHY,
                            message=f"Unexpected status: {response.status}",
                            details={"expected": self.expected_status},
                        )

                    if self.expected_body and self.expected_body not in body:
                        return HealthCheckResult(
                            status=HealthStatus.DEGRADED,
                            message="Response body mismatch",
                        )

                    return HealthCheckResult(
                        status=HealthStatus.HEALTHY,
                        message="HTTP endpoint healthy",
                        details={"status_code": response.status},
                    )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"HTTP check failed: {e}",
            )


class CallbackHealthCheck(HealthCheck):
    """Health check using custom callback."""

    def __init__(
        self,
        name: str,
        callback: Callable[[], HealthCheckResult],
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.callback = callback

    async def check(self) -> HealthCheckResult:
        """Execute callback."""
        if asyncio.iscoroutinefunction(self.callback):
            return await self.callback()
        return self.callback()


class CompositeHealthCheck(HealthCheck):
    """Combines multiple health checks."""

    def __init__(
        self,
        name: str,
        checks: List[HealthCheck],
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.checks = checks

    async def check(self) -> HealthCheckResult:
        """Run all checks and aggregate results."""
        results = await asyncio.gather(
            *[c.execute() for c in self.checks],
            return_exceptions=True,
        )

        statuses = []
        details = {}

        for check, result in zip(self.checks, results):
            if isinstance(result, Exception):
                statuses.append(HealthStatus.UNHEALTHY)
                details[check.name] = {"status": "error", "error": str(result)}
            else:
                statuses.append(result.status)
                details[check.name] = result.to_dict()

        # Aggregate status
        if all(s == HealthStatus.HEALTHY for s in statuses):
            aggregate_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            # Check if any critical check failed
            critical_failed = any(
                check.critical and isinstance(results[i], HealthCheckResult) and results[i].status == HealthStatus.UNHEALTHY
                for i, check in enumerate(self.checks)
            )
            aggregate_status = HealthStatus.UNHEALTHY if critical_failed else HealthStatus.DEGRADED
        else:
            aggregate_status = HealthStatus.DEGRADED

        return HealthCheckResult(
            status=aggregate_status,
            message=f"{sum(1 for s in statuses if s == HealthStatus.HEALTHY)}/{len(statuses)} checks healthy",
            details=details,
        )


class HealthChecker:
    """
    Central health checking service.

    Manages multiple health checks and provides aggregated status.
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        failure_threshold: int = 3,
    ):
        """
        Args:
            check_interval: Seconds between automatic checks
            failure_threshold: Consecutive failures before marking unhealthy
        """
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold

        self._checks: Dict[str, HealthCheck] = {}
        self._component_health: Dict[str, ComponentHealth] = {}
        self._dependency_health: Dict[str, DependencyHealth] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._check_task: Optional[asyncio.Task] = None

    def register_check(self, check: HealthCheck) -> None:
        """Register a health check."""
        self._checks[check.name] = check
        self._component_health[check.name] = ComponentHealth(
            name=check.name,
            status=HealthStatus.UNKNOWN,
            last_check=datetime.utcnow(),
        )

    def register_dependency(
        self,
        name: str,
        type: str,
        check: HealthCheck,
        endpoint: Optional[str] = None,
    ) -> None:
        """Register a dependency health check."""
        self._checks[name] = check
        self._dependency_health[name] = DependencyHealth(
            name=name,
            type=type,
            status=HealthStatus.UNKNOWN,
            endpoint=endpoint,
        )

    async def start(self) -> None:
        """Start automatic health checking."""
        self._running = True
        self._check_task = asyncio.create_task(self._check_loop())
        logger.info("Health checker started")

    async def stop(self) -> None:
        """Stop automatic health checking."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")

    async def _check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                await self.check_all()
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

            await asyncio.sleep(self.check_interval)

    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}

        for name, check in self._checks.items():
            try:
                result = await check.execute()
                results[name] = result
                await self._update_health(name, result)
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                )
                await self._update_health(name, results[name])

        return results

    async def check_one(self, name: str) -> HealthCheckResult:
        """Run a single health check."""
        if name not in self._checks:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Unknown health check: {name}",
            )

        result = await self._checks[name].execute()
        await self._update_health(name, result)
        return result

    async def _update_health(
        self,
        name: str,
        result: HealthCheckResult,
    ) -> None:
        """Update component/dependency health based on result."""
        async with self._lock:
            if name in self._component_health:
                component = self._component_health[name]
                component.status = result.status
                component.last_check = datetime.utcnow()
                component.latency_ms = result.latency_ms
                component.details = result.details

                if result.status == HealthStatus.HEALTHY:
                    component.consecutive_failures = 0
                    component.last_healthy = datetime.utcnow()
                else:
                    component.consecutive_failures += 1
                    component.last_error = result.message

            if name in self._dependency_health:
                dependency = self._dependency_health[name]
                dependency.status = result.status
                dependency.last_check = datetime.utcnow()
                dependency.latency_ms = result.latency_ms
                dependency.error = result.message if result.status != HealthStatus.HEALTHY else None

    async def get_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        async with self._lock:
            components = {
                name: health.to_dict()
                for name, health in self._component_health.items()
            }
            dependencies = {
                name: health.to_dict()
                for name, health in self._dependency_health.items()
            }

            # Calculate overall status
            all_statuses = [h.status for h in self._component_health.values()]
            all_statuses.extend(h.status for h in self._dependency_health.values())

            if all(s == HealthStatus.HEALTHY for s in all_statuses):
                overall = HealthStatus.HEALTHY
            elif any(s == HealthStatus.UNHEALTHY for s in all_statuses):
                overall = HealthStatus.UNHEALTHY
            elif any(s == HealthStatus.DEGRADED for s in all_statuses):
                overall = HealthStatus.DEGRADED
            else:
                overall = HealthStatus.UNKNOWN

            return {
                "status": overall.value,
                "timestamp": datetime.utcnow().isoformat(),
                "components": components,
                "dependencies": dependencies,
            }

    async def is_healthy(self) -> bool:
        """Quick check if system is healthy."""
        status = await self.get_status()
        return status["status"] == HealthStatus.HEALTHY.value

    async def is_ready(self) -> bool:
        """Check if system is ready to serve requests."""
        # Ready if not unhealthy (healthy or degraded is OK)
        status = await self.get_status()
        return status["status"] != HealthStatus.UNHEALTHY.value

    async def is_live(self) -> bool:
        """Check if system is alive (for Kubernetes liveness probes)."""
        # Always return True if the check runs
        # Could add more sophisticated checks here
        return True


class HealthRegistry:
    """
    Global health check registry.

    Provides singleton access to health checking.
    """

    _instance: Optional["HealthRegistry"] = None
    _checker: Optional[HealthChecker] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_checker(cls) -> HealthChecker:
        """Get or create health checker."""
        if cls._checker is None:
            cls._checker = HealthChecker()
        return cls._checker

    @classmethod
    def register(cls, check: HealthCheck) -> None:
        """Register a health check."""
        cls.get_checker().register_check(check)

    @classmethod
    def register_dependency(
        cls,
        name: str,
        type: str,
        check: HealthCheck,
        endpoint: Optional[str] = None,
    ) -> None:
        """Register a dependency."""
        cls.get_checker().register_dependency(name, type, check, endpoint)

    @classmethod
    async def check(cls, name: Optional[str] = None) -> Union[HealthCheckResult, Dict[str, HealthCheckResult]]:
        """Run health check(s)."""
        checker = cls.get_checker()
        if name:
            return await checker.check_one(name)
        return await checker.check_all()

    @classmethod
    async def status(cls) -> Dict[str, Any]:
        """Get health status."""
        return await cls.get_checker().get_status()


# Health check decorators
def health_check(
    name: str,
    timeout: float = 5.0,
    critical: bool = True,
) -> Callable:
    """
    Decorator to create a health check from a function.

    Usage:
        @health_check("my-service")
        async def check_my_service() -> HealthCheckResult:
            ...
    """
    def decorator(func: Callable) -> HealthCheck:
        check = CallbackHealthCheck(
            name=name,
            callback=func,
            timeout_seconds=timeout,
            critical=critical,
        )
        HealthRegistry.register(check)
        return check
    return decorator


# Memory health check
class MemoryHealthCheck(HealthCheck):
    """Check memory usage."""

    def __init__(
        self,
        name: str = "memory",
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    async def check(self) -> HealthCheckResult:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()

            usage = memory.percent / 100

            if usage >= self.critical_threshold:
                status = HealthStatus.UNHEALTHY
            elif usage >= self.warning_threshold:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            return HealthCheckResult(
                status=status,
                message=f"Memory usage: {memory.percent:.1f}%",
                details={
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                },
            )
        except ImportError:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message="psutil not installed",
            )


# Disk health check
class DiskHealthCheck(HealthCheck):
    """Check disk usage."""

    def __init__(
        self,
        name: str = "disk",
        path: str = "/",
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.path = path
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    async def check(self) -> HealthCheckResult:
        """Check disk usage."""
        try:
            import psutil
            disk = psutil.disk_usage(self.path)

            usage = disk.percent / 100

            if usage >= self.critical_threshold:
                status = HealthStatus.UNHEALTHY
            elif usage >= self.warning_threshold:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            return HealthCheckResult(
                status=status,
                message=f"Disk usage ({self.path}): {disk.percent:.1f}%",
                details={
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent,
                },
            )
        except ImportError:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message="psutil not installed",
            )
