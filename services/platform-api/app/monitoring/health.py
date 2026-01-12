"""Health checks for service monitoring."""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class CheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    duration_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HealthReport:
    """Overall health report."""
    status: HealthStatus
    checks: List[CheckResult]
    version: str = "unknown"
    uptime_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "timestamp": self.timestamp.isoformat(),
            "checks": [check.to_dict() for check in self.checks],
        }


HealthCheckFunc = Callable[[], Awaitable[CheckResult]]


class HealthCheck:
    """
    Base class for health checks.

    Usage:
        class DatabaseCheck(HealthCheck):
            async def check(self) -> CheckResult:
                try:
                    await db.execute("SELECT 1")
                    return self.healthy("Database connection OK")
                except Exception as e:
                    return self.unhealthy(f"Database error: {e}")
    """

    def __init__(
        self,
        name: str,
        critical: bool = True,
        timeout: float = 5.0,
    ):
        self.name = name
        self.critical = critical
        self.timeout = timeout

    async def check(self) -> CheckResult:
        """Perform the health check."""
        raise NotImplementedError

    def healthy(
        self,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> CheckResult:
        """Create a healthy result."""
        return CheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message=message,
            details=details or {},
        )

    def degraded(
        self,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> CheckResult:
        """Create a degraded result."""
        return CheckResult(
            name=self.name,
            status=HealthStatus.DEGRADED,
            message=message,
            details=details or {},
        )

    def unhealthy(
        self,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> CheckResult:
        """Create an unhealthy result."""
        return CheckResult(
            name=self.name,
            status=HealthStatus.UNHEALTHY,
            message=message,
            details=details or {},
        )

    async def run(self) -> CheckResult:
        """Run the check with timeout."""
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.check(),
                timeout=self.timeout,
            )
            result.duration_ms = (time.perf_counter() - start) * 1000
            return result
        except asyncio.TimeoutError:
            return CheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {self.timeout}s",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return CheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )


class FunctionHealthCheck(HealthCheck):
    """Health check from a function."""

    def __init__(
        self,
        name: str,
        check_func: HealthCheckFunc,
        critical: bool = True,
        timeout: float = 5.0,
    ):
        super().__init__(name, critical, timeout)
        self.check_func = check_func

    async def check(self) -> CheckResult:
        """Run the check function."""
        return await self.check_func()


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connection."""

    def __init__(
        self,
        name: str = "database",
        get_connection: Optional[Callable] = None,
        critical: bool = True,
    ):
        super().__init__(name, critical)
        self.get_connection = get_connection

    async def check(self) -> CheckResult:
        """Check database connectivity."""
        if not self.get_connection:
            return self.degraded("No database connection configured")

        try:
            conn = await self.get_connection()
            # Execute a simple query
            await conn.execute("SELECT 1")
            return self.healthy("Database connection OK")
        except Exception as e:
            return self.unhealthy(f"Database error: {str(e)}")


class RedisHealthCheck(HealthCheck):
    """Health check for Redis connection."""

    def __init__(
        self,
        name: str = "redis",
        get_redis: Optional[Callable] = None,
        critical: bool = False,
    ):
        super().__init__(name, critical)
        self.get_redis = get_redis

    async def check(self) -> CheckResult:
        """Check Redis connectivity."""
        if not self.get_redis:
            return self.degraded("No Redis client configured")

        try:
            redis = await self.get_redis()
            await redis.ping()
            info = await redis.info("memory")
            return self.healthy(
                "Redis connection OK",
                details={
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                },
            )
        except Exception as e:
            return self.unhealthy(f"Redis error: {str(e)}")


class HTTPHealthCheck(HealthCheck):
    """Health check for HTTP endpoints."""

    def __init__(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        timeout: float = 5.0,
        critical: bool = False,
    ):
        super().__init__(name, critical, timeout)
        self.url = url
        self.expected_status = expected_status

    async def check(self) -> CheckResult:
        """Check HTTP endpoint."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.url,
                    timeout=self.timeout,
                )
                if response.status_code == self.expected_status:
                    return self.healthy(
                        f"HTTP {response.status_code}",
                        details={"url": self.url},
                    )
                else:
                    return self.unhealthy(
                        f"Unexpected status: {response.status_code}",
                        details={"url": self.url, "status": response.status_code},
                    )
        except Exception as e:
            return self.unhealthy(f"HTTP error: {str(e)}")


class DiskSpaceHealthCheck(HealthCheck):
    """Health check for disk space."""

    def __init__(
        self,
        name: str = "disk",
        path: str = "/",
        min_free_gb: float = 1.0,
        warn_free_gb: float = 5.0,
        critical: bool = True,
    ):
        super().__init__(name, critical)
        self.path = path
        self.min_free_gb = min_free_gb
        self.warn_free_gb = warn_free_gb

    async def check(self) -> CheckResult:
        """Check disk space."""
        import shutil

        try:
            usage = shutil.disk_usage(self.path)
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            used_percent = (usage.used / usage.total) * 100

            details = {
                "path": self.path,
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_percent": round(used_percent, 1),
            }

            if free_gb < self.min_free_gb:
                return self.unhealthy(
                    f"Low disk space: {free_gb:.1f}GB free",
                    details=details,
                )
            elif free_gb < self.warn_free_gb:
                return self.degraded(
                    f"Disk space warning: {free_gb:.1f}GB free",
                    details=details,
                )
            else:
                return self.healthy(
                    f"Disk space OK: {free_gb:.1f}GB free",
                    details=details,
                )
        except Exception as e:
            return self.unhealthy(f"Disk check error: {str(e)}")


class MemoryHealthCheck(HealthCheck):
    """Health check for memory usage."""

    def __init__(
        self,
        name: str = "memory",
        max_percent: float = 90.0,
        warn_percent: float = 80.0,
        critical: bool = True,
    ):
        super().__init__(name, critical)
        self.max_percent = max_percent
        self.warn_percent = warn_percent

    async def check(self) -> CheckResult:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            available_gb = memory.available / (1024 ** 3)

            details = {
                "used_percent": round(used_percent, 1),
                "available_gb": round(available_gb, 2),
                "total_gb": round(memory.total / (1024 ** 3), 2),
            }

            if used_percent > self.max_percent:
                return self.unhealthy(
                    f"High memory usage: {used_percent:.1f}%",
                    details=details,
                )
            elif used_percent > self.warn_percent:
                return self.degraded(
                    f"Memory usage warning: {used_percent:.1f}%",
                    details=details,
                )
            else:
                return self.healthy(
                    f"Memory OK: {used_percent:.1f}% used",
                    details=details,
                )
        except ImportError:
            return self.degraded("psutil not installed")
        except Exception as e:
            return self.unhealthy(f"Memory check error: {str(e)}")


class CPUHealthCheck(HealthCheck):
    """Health check for CPU usage."""

    def __init__(
        self,
        name: str = "cpu",
        max_percent: float = 90.0,
        warn_percent: float = 80.0,
        critical: bool = False,
    ):
        super().__init__(name, critical)
        self.max_percent = max_percent
        self.warn_percent = warn_percent

    async def check(self) -> CheckResult:
        """Check CPU usage."""
        try:
            import psutil
            # Get CPU percent over 1 second interval
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            details = {
                "used_percent": round(cpu_percent, 1),
                "cpu_count": cpu_count,
            }

            if cpu_percent > self.max_percent:
                return self.unhealthy(
                    f"High CPU usage: {cpu_percent:.1f}%",
                    details=details,
                )
            elif cpu_percent > self.warn_percent:
                return self.degraded(
                    f"CPU usage warning: {cpu_percent:.1f}%",
                    details=details,
                )
            else:
                return self.healthy(
                    f"CPU OK: {cpu_percent:.1f}% used",
                    details=details,
                )
        except ImportError:
            return self.degraded("psutil not installed")
        except Exception as e:
            return self.unhealthy(f"CPU check error: {str(e)}")


class HealthChecker:
    """
    Health checker that runs multiple health checks.

    Usage:
        checker = HealthChecker(version="1.0.0")
        checker.add_check(DatabaseHealthCheck(get_connection=get_db))
        checker.add_check(RedisHealthCheck(get_redis=get_redis))

        # Run all checks
        report = await checker.run_checks()

        # Run only liveness checks
        report = await checker.run_liveness()

        # Run only readiness checks
        report = await checker.run_readiness()
    """

    def __init__(
        self,
        version: str = "unknown",
        start_time: Optional[datetime] = None,
    ):
        self.version = version
        self.start_time = start_time or datetime.utcnow()
        self._checks: List[HealthCheck] = []
        self._liveness_checks: List[HealthCheck] = []
        self._readiness_checks: List[HealthCheck] = []

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.utcnow() - self.start_time).total_seconds()

    def add_check(
        self,
        check: HealthCheck,
        liveness: bool = True,
        readiness: bool = True,
    ) -> None:
        """Add a health check."""
        self._checks.append(check)
        if liveness:
            self._liveness_checks.append(check)
        if readiness:
            self._readiness_checks.append(check)

    def add_function_check(
        self,
        name: str,
        check_func: HealthCheckFunc,
        critical: bool = True,
        liveness: bool = True,
        readiness: bool = True,
    ) -> None:
        """Add a function-based health check."""
        check = FunctionHealthCheck(name, check_func, critical)
        self.add_check(check, liveness, readiness)

    async def _run_checks(self, checks: List[HealthCheck]) -> HealthReport:
        """Run a list of checks."""
        results = await asyncio.gather(
            *[check.run() for check in checks],
            return_exceptions=True,
        )

        check_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                check_results.append(CheckResult(
                    name=checks[i].name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check exception: {str(result)}",
                ))
            else:
                check_results.append(result)

        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        for i, result in enumerate(check_results):
            if result.status == HealthStatus.UNHEALTHY:
                if checks[i].critical:
                    overall_status = HealthStatus.UNHEALTHY
                    break
                else:
                    overall_status = HealthStatus.DEGRADED
            elif result.status == HealthStatus.DEGRADED:
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED

        return HealthReport(
            status=overall_status,
            checks=check_results,
            version=self.version,
            uptime_seconds=self.uptime_seconds,
        )

    async def run_checks(self) -> HealthReport:
        """Run all health checks."""
        return await self._run_checks(self._checks)

    async def run_liveness(self) -> HealthReport:
        """Run liveness checks (is the service alive?)."""
        return await self._run_checks(self._liveness_checks)

    async def run_readiness(self) -> HealthReport:
        """Run readiness checks (is the service ready to serve traffic?)."""
        return await self._run_checks(self._readiness_checks)


class CachedHealthChecker:
    """
    Health checker with caching to prevent check storms.

    Caches health check results for a configurable duration.
    """

    def __init__(
        self,
        checker: HealthChecker,
        cache_ttl: float = 5.0,
    ):
        self.checker = checker
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}  # key -> (report, timestamp)
        self._lock = asyncio.Lock()

    async def _get_cached_or_run(
        self,
        key: str,
        run_func: Callable,
    ) -> HealthReport:
        """Get cached result or run checks."""
        async with self._lock:
            now = time.time()

            if key in self._cache:
                report, timestamp = self._cache[key]
                if now - timestamp < self.cache_ttl:
                    return report

            report = await run_func()
            self._cache[key] = (report, now)
            return report

    async def run_checks(self) -> HealthReport:
        """Run all checks with caching."""
        return await self._get_cached_or_run(
            "all",
            self.checker.run_checks,
        )

    async def run_liveness(self) -> HealthReport:
        """Run liveness checks with caching."""
        return await self._get_cached_or_run(
            "liveness",
            self.checker.run_liveness,
        )

    async def run_readiness(self) -> HealthReport:
        """Run readiness checks with caching."""
        return await self._get_cached_or_run(
            "readiness",
            self.checker.run_readiness,
        )


# Global health checker
_health_checker: Optional[HealthChecker] = None


def get_health_checker(version: str = "unknown") -> HealthChecker:
    """Get or create the global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(version)
    return _health_checker


def setup_default_health_checks(
    checker: Optional[HealthChecker] = None,
    include_disk: bool = True,
    include_memory: bool = True,
    include_cpu: bool = False,
) -> HealthChecker:
    """Setup default system health checks."""
    checker = checker or get_health_checker()

    if include_disk:
        checker.add_check(DiskSpaceHealthCheck())

    if include_memory:
        checker.add_check(MemoryHealthCheck())

    if include_cpu:
        checker.add_check(CPUHealthCheck())

    return checker


class ComponentHealth:
    """
    Health tracking for a single component.

    Tracks health state with history and automatic recovery detection.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_threshold: int = 2,
        window_size: int = 10,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold
        self.window_size = window_size

        self._history: List[bool] = []
        self._status = HealthStatus.HEALTHY
        self._last_failure: Optional[datetime] = None
        self._failure_count = 0
        self._lock = asyncio.Lock()

    @property
    def status(self) -> HealthStatus:
        """Get current status."""
        return self._status

    async def record_success(self) -> None:
        """Record a successful operation."""
        async with self._lock:
            self._history.append(True)
            if len(self._history) > self.window_size:
                self._history = self._history[-self.window_size:]

            # Check for recovery
            if self._status == HealthStatus.UNHEALTHY:
                recent_successes = sum(1 for x in self._history[-self.recovery_threshold:] if x)
                if recent_successes >= self.recovery_threshold:
                    logger.info(f"Component '{self.name}' recovered")
                    self._status = HealthStatus.HEALTHY
                    self._failure_count = 0

    async def record_failure(self, error: Optional[str] = None) -> None:
        """Record a failed operation."""
        async with self._lock:
            self._history.append(False)
            if len(self._history) > self.window_size:
                self._history = self._history[-self.window_size:]

            self._failure_count += 1
            self._last_failure = datetime.utcnow()

            # Check for unhealthy threshold
            recent_failures = sum(1 for x in self._history[-self.failure_threshold:] if not x)
            if recent_failures >= self.failure_threshold:
                if self._status != HealthStatus.UNHEALTHY:
                    logger.warning(f"Component '{self.name}' is unhealthy: {error}")
                self._status = HealthStatus.UNHEALTHY

    def get_health_result(self) -> CheckResult:
        """Get health check result for this component."""
        success_rate = (
            sum(1 for x in self._history if x) / len(self._history) * 100
            if self._history else 100.0
        )

        return CheckResult(
            name=self.name,
            status=self._status,
            message=f"Success rate: {success_rate:.1f}%",
            details={
                "failure_count": self._failure_count,
                "last_failure": self._last_failure.isoformat() if self._last_failure else None,
                "success_rate": success_rate,
            },
        )


class HealthRegistry:
    """Registry for component health tracking."""

    def __init__(self):
        self._components: Dict[str, ComponentHealth] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_threshold: int = 2,
    ) -> ComponentHealth:
        """Get or create a component health tracker."""
        async with self._lock:
            if name not in self._components:
                self._components[name] = ComponentHealth(
                    name,
                    failure_threshold=failure_threshold,
                    recovery_threshold=recovery_threshold,
                )
            return self._components[name]

    def get_all_health(self) -> List[CheckResult]:
        """Get health results for all components."""
        return [comp.get_health_result() for comp in self._components.values()]


# Global health registry
_health_registry = HealthRegistry()


async def get_component_health(name: str) -> ComponentHealth:
    """Get or create a component health tracker."""
    return await _health_registry.get_or_create(name)


def track_health(component_name: str):
    """
    Decorator to track function health.

    Usage:
        @track_health("database")
        async def query_database():
            ...
    """
    def decorator(func: Callable) -> Callable:
        import functools

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                component = await get_component_health(component_name)
                try:
                    result = await func(*args, **kwargs)
                    await component.record_success()
                    return result
                except Exception as e:
                    await component.record_failure(str(e))
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, we need to run in async context
                component = asyncio.get_event_loop().run_until_complete(
                    get_component_health(component_name)
                )
                try:
                    result = func(*args, **kwargs)
                    asyncio.get_event_loop().run_until_complete(
                        component.record_success()
                    )
                    return result
                except Exception as e:
                    asyncio.get_event_loop().run_until_complete(
                        component.record_failure(str(e))
                    )
                    raise
            return sync_wrapper
    return decorator
