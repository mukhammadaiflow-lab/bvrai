"""
Comprehensive Health Check System

This module provides health check endpoints for Kubernetes probes
and service monitoring with detailed component status reporting.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# =============================================================================
# Health Status Types
# =============================================================================


class HealthStatus(str, Enum):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentType(str, Enum):
    """Types of components to check."""

    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    STORAGE = "storage"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL_SERVICE = "internal_service"


# =============================================================================
# Health Check Models
# =============================================================================


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    name: str
    status: HealthStatus
    component_type: ComponentType
    response_time_ms: Optional[float] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    last_check: datetime = Field(default_factory=datetime.utcnow)


class SystemHealth(BaseModel):
    """Overall system health status."""

    status: HealthStatus
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime_seconds: float
    components: Dict[str, ComponentHealth] = {}

    # Summary stats
    total_components: int = 0
    healthy_components: int = 0
    degraded_components: int = 0
    unhealthy_components: int = 0


# =============================================================================
# Health Check Functions
# =============================================================================


async def check_postgres(
    connection_string: Optional[str] = None,
    timeout: float = 5.0,
) -> ComponentHealth:
    """
    Check PostgreSQL database health.

    Args:
        connection_string: Database connection string
        timeout: Timeout in seconds

    Returns:
        ComponentHealth status
    """
    start_time = time.time()

    try:
        # Try to import and check database
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text

        db_url = connection_string or os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/bvrai"
        )

        # Ensure async driver
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")

        engine = create_async_engine(db_url, pool_pre_ping=True)

        async with engine.connect() as conn:
            result = await asyncio.wait_for(
                conn.execute(text("SELECT 1 as health_check")),
                timeout=timeout,
            )
            row = result.fetchone()

            if row and row[0] == 1:
                response_time = (time.time() - start_time) * 1000
                await engine.dispose()

                return ComponentHealth(
                    name="PostgreSQL",
                    status=HealthStatus.HEALTHY,
                    component_type=ComponentType.DATABASE,
                    response_time_ms=response_time,
                    message="Database connection successful",
                    details={"connected": True},
                )

        await engine.dispose()
        raise Exception("Unexpected query result")

    except asyncio.TimeoutError:
        return ComponentHealth(
            name="PostgreSQL",
            status=HealthStatus.UNHEALTHY,
            component_type=ComponentType.DATABASE,
            response_time_ms=(time.time() - start_time) * 1000,
            message=f"Database connection timeout after {timeout}s",
            details={"error": "timeout"},
        )
    except Exception as e:
        return ComponentHealth(
            name="PostgreSQL",
            status=HealthStatus.UNHEALTHY,
            component_type=ComponentType.DATABASE,
            response_time_ms=(time.time() - start_time) * 1000,
            message=f"Database connection failed: {str(e)}",
            details={"error": str(e)},
        )


async def check_redis(
    redis_url: Optional[str] = None,
    timeout: float = 5.0,
) -> ComponentHealth:
    """
    Check Redis cache health.

    Args:
        redis_url: Redis connection URL
        timeout: Timeout in seconds

    Returns:
        ComponentHealth status
    """
    start_time = time.time()

    try:
        import redis.asyncio as redis

        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        client = redis.from_url(url)

        # Ping Redis
        pong = await asyncio.wait_for(
            client.ping(),
            timeout=timeout,
        )

        if pong:
            # Get additional info
            info = await client.info("server")
            response_time = (time.time() - start_time) * 1000

            await client.close()

            return ComponentHealth(
                name="Redis",
                status=HealthStatus.HEALTHY,
                component_type=ComponentType.CACHE,
                response_time_ms=response_time,
                message="Redis connection successful",
                details={
                    "connected": True,
                    "version": info.get("redis_version", "unknown"),
                },
            )

        await client.close()
        raise Exception("Redis ping failed")

    except asyncio.TimeoutError:
        return ComponentHealth(
            name="Redis",
            status=HealthStatus.DEGRADED,
            component_type=ComponentType.CACHE,
            response_time_ms=(time.time() - start_time) * 1000,
            message=f"Redis connection timeout after {timeout}s",
            details={"error": "timeout"},
        )
    except ImportError:
        return ComponentHealth(
            name="Redis",
            status=HealthStatus.DEGRADED,
            component_type=ComponentType.CACHE,
            response_time_ms=(time.time() - start_time) * 1000,
            message="Redis client not installed",
            details={"error": "redis package not installed"},
        )
    except Exception as e:
        return ComponentHealth(
            name="Redis",
            status=HealthStatus.DEGRADED,
            component_type=ComponentType.CACHE,
            response_time_ms=(time.time() - start_time) * 1000,
            message=f"Redis connection failed: {str(e)}",
            details={"error": str(e)},
        )


async def check_rabbitmq(
    amqp_url: Optional[str] = None,
    timeout: float = 5.0,
) -> ComponentHealth:
    """
    Check RabbitMQ message queue health.

    Args:
        amqp_url: RabbitMQ connection URL
        timeout: Timeout in seconds

    Returns:
        ComponentHealth status
    """
    start_time = time.time()

    try:
        import aio_pika

        url = amqp_url or os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

        connection = await asyncio.wait_for(
            aio_pika.connect_robust(url),
            timeout=timeout,
        )

        # Create a channel to verify connection works
        channel = await connection.channel()
        await channel.close()
        await connection.close()

        response_time = (time.time() - start_time) * 1000

        return ComponentHealth(
            name="RabbitMQ",
            status=HealthStatus.HEALTHY,
            component_type=ComponentType.QUEUE,
            response_time_ms=response_time,
            message="RabbitMQ connection successful",
            details={"connected": True},
        )

    except asyncio.TimeoutError:
        return ComponentHealth(
            name="RabbitMQ",
            status=HealthStatus.DEGRADED,
            component_type=ComponentType.QUEUE,
            response_time_ms=(time.time() - start_time) * 1000,
            message=f"RabbitMQ connection timeout after {timeout}s",
            details={"error": "timeout"},
        )
    except ImportError:
        return ComponentHealth(
            name="RabbitMQ",
            status=HealthStatus.DEGRADED,
            component_type=ComponentType.QUEUE,
            response_time_ms=(time.time() - start_time) * 1000,
            message="aio_pika not installed",
            details={"error": "aio_pika package not installed"},
        )
    except Exception as e:
        return ComponentHealth(
            name="RabbitMQ",
            status=HealthStatus.DEGRADED,
            component_type=ComponentType.QUEUE,
            response_time_ms=(time.time() - start_time) * 1000,
            message=f"RabbitMQ connection failed: {str(e)}",
            details={"error": str(e)},
        )


async def check_external_api(
    name: str,
    url: str,
    timeout: float = 10.0,
    expected_status: int = 200,
) -> ComponentHealth:
    """
    Check an external API endpoint health.

    Args:
        name: Name of the service
        url: URL to check
        timeout: Timeout in seconds
        expected_status: Expected HTTP status code

    Returns:
        ComponentHealth status
    """
    start_time = time.time()

    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response_time = (time.time() - start_time) * 1000

            if response.status_code == expected_status:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    component_type=ComponentType.EXTERNAL_SERVICE,
                    response_time_ms=response_time,
                    message=f"{name} is reachable",
                    details={
                        "status_code": response.status_code,
                        "url": url,
                    },
                )
            else:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.DEGRADED,
                    component_type=ComponentType.EXTERNAL_SERVICE,
                    response_time_ms=response_time,
                    message=f"Unexpected status code: {response.status_code}",
                    details={
                        "status_code": response.status_code,
                        "expected": expected_status,
                    },
                )

    except asyncio.TimeoutError:
        return ComponentHealth(
            name=name,
            status=HealthStatus.DEGRADED,
            component_type=ComponentType.EXTERNAL_SERVICE,
            response_time_ms=(time.time() - start_time) * 1000,
            message=f"Request timeout after {timeout}s",
            details={"error": "timeout"},
        )
    except Exception as e:
        return ComponentHealth(
            name=name,
            status=HealthStatus.DEGRADED,
            component_type=ComponentType.EXTERNAL_SERVICE,
            response_time_ms=(time.time() - start_time) * 1000,
            message=f"Request failed: {str(e)}",
            details={"error": str(e)},
        )


# =============================================================================
# Health Check Service
# =============================================================================


@dataclass
class HealthCheckService:
    """
    Service for running health checks on all system components.

    Usage:
        service = HealthCheckService(version="1.0.0")
        service.register_check("database", check_postgres)
        health = await service.run_all_checks()
    """

    version: str = "1.0.0"
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    start_time: float = field(default_factory=time.time)

    _checks: Dict[str, Callable] = field(default_factory=dict)

    def register_check(
        self,
        name: str,
        check_func: Callable[[], ComponentHealth],
    ) -> None:
        """Register a health check function."""
        self._checks[name] = check_func

    def register_default_checks(self) -> None:
        """Register default infrastructure checks."""
        self._checks["postgres"] = check_postgres
        self._checks["redis"] = check_redis
        self._checks["rabbitmq"] = check_rabbitmq

    async def run_check(self, name: str) -> Optional[ComponentHealth]:
        """Run a single health check."""
        check_func = self._checks.get(name)
        if not check_func:
            return None

        try:
            if asyncio.iscoroutinefunction(check_func):
                return await check_func()
            else:
                return check_func()
        except Exception as e:
            logger.exception(f"Health check '{name}' failed with exception")
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                component_type=ComponentType.INTERNAL_SERVICE,
                message=f"Check failed with exception: {str(e)}",
            )

    async def run_all_checks(
        self,
        parallel: bool = True,
    ) -> SystemHealth:
        """
        Run all registered health checks.

        Args:
            parallel: Run checks in parallel

        Returns:
            SystemHealth with all component statuses
        """
        components: Dict[str, ComponentHealth] = {}

        if parallel:
            # Run checks in parallel
            tasks = {
                name: asyncio.create_task(self.run_check(name))
                for name in self._checks
            }

            for name, task in tasks.items():
                try:
                    result = await task
                    if result:
                        components[name] = result
                except Exception as e:
                    components[name] = ComponentHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        component_type=ComponentType.INTERNAL_SERVICE,
                        message=f"Check failed: {str(e)}",
                    )
        else:
            # Run checks sequentially
            for name in self._checks:
                result = await self.run_check(name)
                if result:
                    components[name] = result

        # Calculate overall status
        status = self._calculate_overall_status(components)

        # Count by status
        healthy = sum(1 for c in components.values() if c.status == HealthStatus.HEALTHY)
        degraded = sum(1 for c in components.values() if c.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for c in components.values() if c.status == HealthStatus.UNHEALTHY)

        return SystemHealth(
            status=status,
            version=self.version,
            environment=self.environment,
            uptime_seconds=time.time() - self.start_time,
            components=components,
            total_components=len(components),
            healthy_components=healthy,
            degraded_components=degraded,
            unhealthy_components=unhealthy,
        )

    def _calculate_overall_status(
        self,
        components: Dict[str, ComponentHealth],
    ) -> HealthStatus:
        """Calculate overall system status from component statuses."""
        if not components:
            return HealthStatus.HEALTHY

        statuses = [c.status for c in components.values()]

        # If any critical component is unhealthy, system is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY

        # If any component is degraded, system is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    async def liveness_check(self) -> bool:
        """
        Simple liveness check for Kubernetes.
        Returns True if the service is running.
        """
        return True

    async def readiness_check(self) -> bool:
        """
        Readiness check for Kubernetes.
        Returns True if the service is ready to accept traffic.
        """
        # Check critical components
        postgres_health = await check_postgres(timeout=2.0)

        # Service is ready if database is accessible
        return postgres_health.status != HealthStatus.UNHEALTHY


# =============================================================================
# Singleton Instance
# =============================================================================


_health_service: Optional[HealthCheckService] = None


def get_health_service(version: str = "1.0.0") -> HealthCheckService:
    """Get or create the health check service singleton."""
    global _health_service

    if _health_service is None:
        _health_service = HealthCheckService(version=version)
        _health_service.register_default_checks()

    return _health_service


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "HealthStatus",
    "ComponentType",
    "ComponentHealth",
    "SystemHealth",
    "HealthCheckService",
    "get_health_service",
    "check_postgres",
    "check_redis",
    "check_rabbitmq",
    "check_external_api",
]
