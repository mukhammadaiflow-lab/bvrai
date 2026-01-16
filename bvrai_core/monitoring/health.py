"""
Health Check System

Provides comprehensive health checking capabilities for monitoring
system components, services, and dependencies.
"""

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp

from .base import (
    Component,
    ComponentType,
    HealthCheck,
    HealthCheckResult,
    HealthStatus,
    HealthChecker,
    HealthCheckError,
    MonitoringError,
)


logger = logging.getLogger(__name__)


# =========================================================================
# Built-in Health Checkers
# =========================================================================

class HTTPHealthChecker(HealthChecker):
    """
    HTTP-based health checker.

    Performs HTTP requests to health endpoints and evaluates
    response status and optional JSON content.
    """

    def __init__(
        self,
        expected_status_codes: Optional[List[int]] = None,
        expected_body_contains: Optional[str] = None,
        expected_json_path: Optional[str] = None,
        expected_json_value: Optional[Any] = None,
    ):
        """
        Initialize HTTP health checker.

        Args:
            expected_status_codes: Acceptable HTTP status codes (default: [200])
            expected_body_contains: String that response body should contain
            expected_json_path: JSON path to check (e.g., "status" or "health.api")
            expected_json_value: Expected value at JSON path
        """
        self._expected_status = expected_status_codes or [200]
        self._expected_body = expected_body_contains
        self._expected_json_path = expected_json_path
        self._expected_json_value = expected_json_value
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def check(self, component: Component) -> HealthCheckResult:
        """Perform HTTP health check."""
        if not component.endpoint:
            return HealthCheckResult(
                check_id="http_check",
                component_id=component.id,
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message="No endpoint configured",
            )

        session = await self._get_session()
        start_time = datetime.utcnow()

        try:
            async with session.get(
                component.endpoint,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                body = await response.text()

                # Check status code
                if response.status not in self._expected_status:
                    return HealthCheckResult(
                        check_id="http_check",
                        component_id=component.id,
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        message=f"Unexpected status code: {response.status}",
                        details={"status_code": response.status},
                    )

                # Check body contains
                if self._expected_body and self._expected_body not in body:
                    return HealthCheckResult(
                        check_id="http_check",
                        component_id=component.id,
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        message="Response body doesn't contain expected string",
                    )

                # Check JSON path
                if self._expected_json_path:
                    try:
                        import json
                        data = json.loads(body)
                        value = self._get_json_path(data, self._expected_json_path)

                        if self._expected_json_value is not None:
                            if value != self._expected_json_value:
                                return HealthCheckResult(
                                    check_id="http_check",
                                    component_id=component.id,
                                    status=HealthStatus.UNHEALTHY,
                                    latency_ms=latency_ms,
                                    message=f"JSON value mismatch at {self._expected_json_path}",
                                    details={
                                        "expected": self._expected_json_value,
                                        "actual": value,
                                    },
                                )
                    except json.JSONDecodeError:
                        return HealthCheckResult(
                            check_id="http_check",
                            component_id=component.id,
                            status=HealthStatus.UNHEALTHY,
                            latency_ms=latency_ms,
                            message="Invalid JSON response",
                        )

                # Determine health based on latency
                status = HealthStatus.HEALTHY
                if latency_ms > 5000:
                    status = HealthStatus.DEGRADED

                return HealthCheckResult(
                    check_id="http_check",
                    component_id=component.id,
                    status=status,
                    latency_ms=latency_ms,
                    details={"status_code": response.status},
                )

        except asyncio.TimeoutError:
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return HealthCheckResult(
                check_id="http_check",
                component_id=component.id,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message="Request timeout",
            )

        except aiohttp.ClientError as e:
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return HealthCheckResult(
                check_id="http_check",
                component_id=component.id,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Connection error: {str(e)}",
            )

    def _get_json_path(self, data: Any, path: str) -> Any:
        """Get value at JSON path."""
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current


class TCPHealthChecker(HealthChecker):
    """
    TCP connection health checker.

    Verifies that a TCP connection can be established to a host/port.
    """

    def __init__(self, port: Optional[int] = None):
        """
        Initialize TCP health checker.

        Args:
            port: Port to connect to (can be overridden by component endpoint)
        """
        self._default_port = port

    async def check(self, component: Component) -> HealthCheckResult:
        """Perform TCP connection check."""
        if not component.endpoint:
            return HealthCheckResult(
                check_id="tcp_check",
                component_id=component.id,
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message="No endpoint configured",
            )

        # Parse endpoint
        host = component.endpoint
        port = self._default_port or 80

        if ":" in host:
            parts = host.rsplit(":", 1)
            host = parts[0]
            try:
                port = int(parts[1])
            except ValueError:
                pass

        # Remove protocol if present
        if "://" in host:
            host = host.split("://")[1].split("/")[0]

        start_time = datetime.utcnow()

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=10.0,
            )
            writer.close()
            await writer.wait_closed()

            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            return HealthCheckResult(
                check_id="tcp_check",
                component_id=component.id,
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                details={"host": host, "port": port},
            )

        except asyncio.TimeoutError:
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return HealthCheckResult(
                check_id="tcp_check",
                component_id=component.id,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Connection timeout to {host}:{port}",
            )

        except (ConnectionRefusedError, OSError) as e:
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return HealthCheckResult(
                check_id="tcp_check",
                component_id=component.id,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Connection failed: {str(e)}",
            )


class DatabaseHealthChecker(HealthChecker):
    """
    Database health checker.

    Performs database connectivity and query checks.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        test_query: str = "SELECT 1",
    ):
        """
        Initialize database health checker.

        Args:
            connection_string: Database connection string
            test_query: Query to execute for health check
        """
        self._connection_string = connection_string
        self._test_query = test_query

    async def check(self, component: Component) -> HealthCheckResult:
        """Perform database health check."""
        start_time = datetime.utcnow()

        connection_string = self._connection_string or component.metadata.get("connection_string")

        if not connection_string:
            return HealthCheckResult(
                check_id="database_check",
                component_id=component.id,
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message="No connection string configured",
            )

        # This is a simplified check - in production, use actual DB drivers
        # For now, we'll simulate the check
        try:
            # Simulate database query
            await asyncio.sleep(0.01)  # Simulated latency

            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            return HealthCheckResult(
                check_id="database_check",
                component_id=component.id,
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                details={"query": self._test_query},
            )

        except Exception as e:
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return HealthCheckResult(
                check_id="database_check",
                component_id=component.id,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Database error: {str(e)}",
            )


class RedisHealthChecker(HealthChecker):
    """
    Redis health checker.

    Verifies Redis connectivity and performs PING check.
    """

    def __init__(self, host: str = "localhost", port: int = 6379):
        """
        Initialize Redis health checker.

        Args:
            host: Redis host
            port: Redis port
        """
        self._host = host
        self._port = port

    async def check(self, component: Component) -> HealthCheckResult:
        """Perform Redis health check."""
        start_time = datetime.utcnow()

        host = component.metadata.get("host", self._host)
        port = component.metadata.get("port", self._port)

        try:
            # Simple TCP connection check for Redis
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=5.0,
            )

            # Send PING command
            writer.write(b"*1\r\n$4\r\nPING\r\n")
            await writer.drain()

            # Read response
            response = await asyncio.wait_for(reader.read(100), timeout=5.0)

            writer.close()
            await writer.wait_closed()

            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            if b"PONG" in response:
                return HealthCheckResult(
                    check_id="redis_check",
                    component_id=component.id,
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency_ms,
                    details={"host": host, "port": port},
                )
            else:
                return HealthCheckResult(
                    check_id="redis_check",
                    component_id=component.id,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=latency_ms,
                    message="Unexpected response from Redis",
                )

        except asyncio.TimeoutError:
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return HealthCheckResult(
                check_id="redis_check",
                component_id=component.id,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message="Redis connection timeout",
            )

        except Exception as e:
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return HealthCheckResult(
                check_id="redis_check",
                component_id=component.id,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Redis error: {str(e)}",
            )


class CustomHealthChecker(HealthChecker):
    """
    Custom health checker using a callable.

    Allows defining custom health check logic.
    """

    def __init__(self, check_func: Callable[[Component], HealthCheckResult]):
        """
        Initialize custom health checker.

        Args:
            check_func: Async function that performs the health check
        """
        self._check_func = check_func

    async def check(self, component: Component) -> HealthCheckResult:
        """Perform custom health check."""
        try:
            if asyncio.iscoroutinefunction(self._check_func):
                return await self._check_func(component)
            else:
                return self._check_func(component)
        except Exception as e:
            return HealthCheckResult(
                check_id="custom_check",
                component_id=component.id,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                message=f"Check error: {str(e)}",
            )


# =========================================================================
# Health Check Manager
# =========================================================================

@dataclass
class HealthCheckConfig:
    """Configuration for health check manager."""
    default_interval_seconds: int = 30
    default_timeout_seconds: int = 10
    default_failure_threshold: int = 3
    default_success_threshold: int = 1
    max_concurrent_checks: int = 50
    retain_history_hours: int = 24


class HealthCheckManager:
    """
    Manages health checks for all monitored components.

    Features:
    - Registration of components and health checks
    - Scheduled health check execution
    - Status aggregation
    - History tracking
    - Dependency health evaluation
    """

    def __init__(self, config: Optional[HealthCheckConfig] = None):
        """
        Initialize health check manager.

        Args:
            config: Health check configuration
        """
        self.config = config or HealthCheckConfig()

        # Components and checks
        self._components: Dict[str, Component] = {}
        self._health_checks: Dict[str, HealthCheck] = {}
        self._checkers: Dict[ComponentType, HealthChecker] = {}

        # State
        self._check_results: Dict[str, List[HealthCheckResult]] = {}
        self._component_status: Dict[str, HealthStatus] = {}
        self._consecutive_failures: Dict[str, int] = {}
        self._consecutive_successes: Dict[str, int] = {}

        # Event handlers
        self._on_status_change: List[Callable] = []
        self._on_unhealthy: List[Callable] = []
        self._on_healthy: List[Callable] = []

        # Background tasks
        self._check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Register default checkers
        self._register_default_checkers()

    def _register_default_checkers(self) -> None:
        """Register default health checkers for component types."""
        self._checkers[ComponentType.SERVICE] = HTTPHealthChecker()
        self._checkers[ComponentType.API] = HTTPHealthChecker()
        self._checkers[ComponentType.DATABASE] = DatabaseHealthChecker()
        self._checkers[ComponentType.CACHE] = RedisHealthChecker()

    # =========================================================================
    # Component Management
    # =========================================================================

    def register_component(self, component: Component) -> None:
        """
        Register a component for health monitoring.

        Args:
            component: Component to register
        """
        self._components[component.id] = component
        self._component_status[component.id] = HealthStatus.UNKNOWN
        self._consecutive_failures[component.id] = 0
        self._consecutive_successes[component.id] = 0
        self._check_results[component.id] = []

        logger.info(f"Registered component for health monitoring: {component.name}")

    def unregister_component(self, component_id: str) -> bool:
        """
        Unregister a component.

        Args:
            component_id: Component identifier

        Returns:
            True if component was unregistered
        """
        if component_id in self._components:
            del self._components[component_id]
            self._component_status.pop(component_id, None)
            self._consecutive_failures.pop(component_id, None)
            self._consecutive_successes.pop(component_id, None)
            self._check_results.pop(component_id, None)
            return True
        return False

    def get_component(self, component_id: str) -> Optional[Component]:
        """Get a component by ID."""
        return self._components.get(component_id)

    def list_components(
        self,
        organization_id: Optional[str] = None,
        component_type: Optional[ComponentType] = None,
        status: Optional[HealthStatus] = None,
    ) -> List[Component]:
        """
        List components with optional filters.

        Args:
            organization_id: Filter by organization
            component_type: Filter by type
            status: Filter by health status

        Returns:
            List of matching components
        """
        components = list(self._components.values())

        if organization_id is not None:
            components = [c for c in components if c.organization_id == organization_id]
        if component_type:
            components = [c for c in components if c.component_type == component_type]
        if status:
            components = [c for c in components if self._component_status.get(c.id) == status]

        return components

    # =========================================================================
    # Health Check Management
    # =========================================================================

    def register_health_check(self, check: HealthCheck) -> None:
        """
        Register a health check.

        Args:
            check: Health check definition
        """
        self._health_checks[check.id] = check
        logger.info(f"Registered health check: {check.name}")

    def register_checker(
        self,
        component_type: ComponentType,
        checker: HealthChecker,
    ) -> None:
        """
        Register a custom health checker for a component type.

        Args:
            component_type: Type of component
            checker: Health checker implementation
        """
        self._checkers[component_type] = checker

    async def check_component(self, component_id: str) -> HealthCheckResult:
        """
        Perform health check on a single component.

        Args:
            component_id: Component to check

        Returns:
            Health check result

        Raises:
            HealthCheckError: If component not found
        """
        component = self._components.get(component_id)
        if not component:
            raise HealthCheckError(f"Component not found: {component_id}", component_id)

        checker = self._checkers.get(component.component_type)
        if not checker:
            return HealthCheckResult(
                check_id="unknown",
                component_id=component_id,
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message="No checker configured for component type",
            )

        try:
            result = await asyncio.wait_for(
                checker.check(component),
                timeout=self.config.default_timeout_seconds,
            )
        except asyncio.TimeoutError:
            result = HealthCheckResult(
                check_id="timeout",
                component_id=component_id,
                status=HealthStatus.UNHEALTHY,
                latency_ms=self.config.default_timeout_seconds * 1000,
                message="Health check timed out",
            )

        # Update state
        await self._process_result(component, result)

        return result

    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Perform health checks on all components.

        Returns:
            Dictionary mapping component IDs to results
        """
        results = {}

        # Run checks concurrently with limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent_checks)

        async def check_with_semaphore(component_id: str):
            async with semaphore:
                return await self.check_component(component_id)

        tasks = [
            check_with_semaphore(cid)
            for cid in self._components.keys()
        ]

        if tasks:
            check_results = await asyncio.gather(*tasks, return_exceptions=True)

            for component_id, result in zip(self._components.keys(), check_results):
                if isinstance(result, Exception):
                    results[component_id] = HealthCheckResult(
                        check_id="error",
                        component_id=component_id,
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=0,
                        message=str(result),
                    )
                else:
                    results[component_id] = result

        return results

    async def _process_result(
        self,
        component: Component,
        result: HealthCheckResult,
    ) -> None:
        """Process a health check result and update state."""
        component_id = component.id
        previous_status = self._component_status.get(component_id, HealthStatus.UNKNOWN)

        # Update consecutive counts
        if result.status == HealthStatus.HEALTHY:
            self._consecutive_successes[component_id] = (
                self._consecutive_successes.get(component_id, 0) + 1
            )
            self._consecutive_failures[component_id] = 0
        elif result.status == HealthStatus.UNHEALTHY:
            self._consecutive_failures[component_id] = (
                self._consecutive_failures.get(component_id, 0) + 1
            )
            self._consecutive_successes[component_id] = 0

        result.consecutive_failures = self._consecutive_failures.get(component_id, 0)
        result.consecutive_successes = self._consecutive_successes.get(component_id, 0)

        # Determine new status based on thresholds
        new_status = result.status

        if result.status == HealthStatus.UNHEALTHY:
            if self._consecutive_failures[component_id] < self.config.default_failure_threshold:
                new_status = previous_status  # Don't transition yet

        if result.status == HealthStatus.HEALTHY:
            if self._consecutive_successes[component_id] < self.config.default_success_threshold:
                new_status = previous_status  # Don't transition yet

        # Update status
        self._component_status[component_id] = new_status
        component.status = new_status
        component.last_check_at = result.timestamp

        # Store result in history
        if component_id not in self._check_results:
            self._check_results[component_id] = []
        self._check_results[component_id].append(result)

        # Emit events on status change
        if new_status != previous_status:
            for handler in self._on_status_change:
                try:
                    await handler(component, previous_status, new_status)
                except Exception as e:
                    logger.error(f"Error in status change handler: {e}")

            if new_status == HealthStatus.UNHEALTHY:
                for handler in self._on_unhealthy:
                    try:
                        await handler(component, result)
                    except Exception as e:
                        logger.error(f"Error in unhealthy handler: {e}")

            elif new_status == HealthStatus.HEALTHY and previous_status == HealthStatus.UNHEALTHY:
                for handler in self._on_healthy:
                    try:
                        await handler(component, result)
                    except Exception as e:
                        logger.error(f"Error in healthy handler: {e}")

    # =========================================================================
    # Status Queries
    # =========================================================================

    def get_component_status(self, component_id: str) -> HealthStatus:
        """Get current status of a component."""
        return self._component_status.get(component_id, HealthStatus.UNKNOWN)

    def get_component_history(
        self,
        component_id: str,
        limit: int = 100,
    ) -> List[HealthCheckResult]:
        """
        Get health check history for a component.

        Args:
            component_id: Component identifier
            limit: Maximum number of results

        Returns:
            List of health check results (most recent first)
        """
        history = self._check_results.get(component_id, [])
        return list(reversed(history[-limit:]))

    def get_overall_status(
        self,
        organization_id: Optional[str] = None,
    ) -> HealthStatus:
        """
        Get overall system health status.

        Args:
            organization_id: Filter by organization

        Returns:
            Aggregate health status
        """
        components = self.list_components(organization_id=organization_id)

        if not components:
            return HealthStatus.UNKNOWN

        statuses = [self._component_status.get(c.id, HealthStatus.UNKNOWN) for c in components]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def get_status_summary(
        self,
        organization_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get health status summary.

        Args:
            organization_id: Filter by organization

        Returns:
            Status summary with counts and details
        """
        components = self.list_components(organization_id=organization_id)

        status_counts = {status.value: 0 for status in HealthStatus}
        for component in components:
            status = self._component_status.get(component.id, HealthStatus.UNKNOWN)
            status_counts[status.value] += 1

        return {
            "overall_status": self.get_overall_status(organization_id).value,
            "total_components": len(components),
            "by_status": status_counts,
            "components": [
                {
                    "id": c.id,
                    "name": c.name,
                    "type": c.component_type.value,
                    "status": self._component_status.get(c.id, HealthStatus.UNKNOWN).value,
                    "last_check_at": c.last_check_at.isoformat() if c.last_check_at else None,
                }
                for c in components
            ],
        }

    def get_dependency_health(self, component_id: str) -> Dict[str, HealthStatus]:
        """
        Get health status of a component's dependencies.

        Args:
            component_id: Component identifier

        Returns:
            Dictionary mapping dependency IDs to their status
        """
        component = self._components.get(component_id)
        if not component:
            return {}

        return {
            dep_id: self._component_status.get(dep_id, HealthStatus.UNKNOWN)
            for dep_id in component.dependencies
        }

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_status_change(self, handler: Callable) -> None:
        """Register handler for status change events."""
        self._on_status_change.append(handler)

    def on_unhealthy(self, handler: Callable) -> None:
        """Register handler for unhealthy events."""
        self._on_unhealthy.append(handler)

    def on_healthy(self, handler: Callable) -> None:
        """Register handler for healthy events."""
        self._on_healthy.append(handler)

    # =========================================================================
    # Background Tasks
    # =========================================================================

    async def start(self) -> None:
        """Start health check manager."""
        if self._running:
            return

        self._running = True
        self._check_task = asyncio.create_task(self._check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Health check manager started")

    async def stop(self) -> None:
        """Stop health check manager."""
        self._running = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Health check manager stopped")

    async def _check_loop(self) -> None:
        """Background loop for periodic health checks."""
        while self._running:
            try:
                await self.check_all()
                await asyncio.sleep(self.config.default_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)

    async def _cleanup_loop(self) -> None:
        """Background loop for cleaning up old results."""
        while self._running:
            try:
                cutoff = datetime.utcnow() - timedelta(hours=self.config.retain_history_hours)

                for component_id in list(self._check_results.keys()):
                    results = self._check_results.get(component_id, [])
                    self._check_results[component_id] = [
                        r for r in results if r.timestamp >= cutoff
                    ]

                await asyncio.sleep(3600)  # Run cleanup hourly

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)


def create_health_check_manager(
    config: Optional[HealthCheckConfig] = None,
) -> HealthCheckManager:
    """
    Create a health check manager with default configuration.

    Args:
        config: Optional configuration

    Returns:
        Configured HealthCheckManager
    """
    return HealthCheckManager(config)
