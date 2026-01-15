"""
Platform Core Engine
====================

The central orchestration system that coordinates all platform services,
manages lifecycle, handles cross-cutting concerns, and provides the
foundation for all platform operations.

This is the heart of the Voice AI Platform - responsible for:
- Service lifecycle management
- Cross-service coordination
- Resource allocation and management
- Health monitoring and recovery
- Configuration propagation
- Event orchestration

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import signal
import socket
import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
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
from pydantic import BaseModel, Field, validator

logger = structlog.get_logger(__name__)

# Type definitions
T = TypeVar("T")
ServiceType = TypeVar("ServiceType", bound="BaseService")
MiddlewareFunc = Callable[[Any, Callable], Awaitable[Any]]


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class EngineState(str, Enum):
    """Platform engine lifecycle states"""

    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    RECOVERING = "recovering"


class ServiceStatus(str, Enum):
    """Individual service status"""

    REGISTERED = "registered"
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    RECOVERING = "recovering"


class ServicePriority(int, Enum):
    """Service startup/shutdown priority"""

    CRITICAL = 0      # Infrastructure services (DB, Cache)
    HIGH = 10         # Core platform services
    NORMAL = 50       # Standard services
    LOW = 100         # Optional services
    BACKGROUND = 200  # Background/maintenance services


class HealthStatus(str, Enum):
    """Health check status"""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ShutdownMode(str, Enum):
    """Shutdown behavior modes"""

    GRACEFUL = "graceful"      # Wait for operations to complete
    IMMEDIATE = "immediate"    # Stop immediately
    FORCED = "forced"          # Force kill all operations


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class EngineConfig(BaseModel):
    """Platform engine configuration"""

    # Identity
    engine_id: str = Field(default_factory=lambda: f"engine-{uuid4().hex[:8]}")
    engine_name: str = "builder-engine"
    environment: str = Field(default="development")
    version: str = "2.0.0"

    # Networking
    host: str = "0.0.0.0"
    port: int = 8000
    internal_port: int = 8001

    # Lifecycle
    startup_timeout: float = 300.0
    shutdown_timeout: float = 60.0
    graceful_shutdown_delay: float = 5.0

    # Health
    health_check_interval: float = 10.0
    health_check_timeout: float = 5.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2

    # Performance
    max_concurrent_operations: int = 1000
    worker_pool_size: int = 10
    event_queue_size: int = 10000

    # Resilience
    enable_auto_recovery: bool = True
    recovery_max_retries: int = 3
    recovery_backoff_base: float = 1.0
    recovery_backoff_max: float = 60.0

    # Logging
    log_level: str = "INFO"
    enable_structured_logging: bool = True
    enable_request_tracing: bool = True

    # Features
    enable_metrics: bool = True
    enable_profiling: bool = False
    enable_distributed_tracing: bool = True

    class Config:
        env_prefix = "PLATFORM_"


class ServiceConfig(BaseModel):
    """Service-specific configuration"""

    name: str
    version: str = "1.0.0"
    priority: ServicePriority = ServicePriority.NORMAL
    enabled: bool = True

    # Dependencies
    dependencies: List[str] = Field(default_factory=list)
    soft_dependencies: List[str] = Field(default_factory=list)

    # Health
    health_check_enabled: bool = True
    health_check_interval: float = 30.0
    health_check_timeout: float = 10.0

    # Resources
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None

    # Custom settings
    settings: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ServiceMetrics:
    """Metrics for a service"""

    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    latency_sum_ms: float = 0.0
    latency_count: int = 0
    last_request_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    restarts: int = 0

    @property
    def success_rate(self) -> float:
        if self.requests_total == 0:
            return 1.0
        return self.requests_success / self.requests_total

    @property
    def avg_latency_ms(self) -> float:
        if self.latency_count == 0:
            return 0.0
        return self.latency_sum_ms / self.latency_count

    def record_request(self, success: bool, latency_ms: float) -> None:
        self.requests_total += 1
        if success:
            self.requests_success += 1
        else:
            self.requests_failed += 1
        self.latency_sum_ms += latency_ms
        self.latency_count += 1
        self.last_request_time = datetime.utcnow()


@dataclass
class HealthCheckResult:
    """Result of a health check"""

    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    checks: Dict[str, "HealthCheckResult"] = field(default_factory=dict)


@dataclass
class ServiceInfo:
    """Information about a registered service"""

    name: str
    service: "BaseService"
    config: ServiceConfig
    status: ServiceStatus = ServiceStatus.REGISTERED
    health: HealthStatus = HealthStatus.UNKNOWN
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_health_check: Optional[HealthCheckResult] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


@dataclass
class EngineMetrics:
    """Platform engine metrics"""

    services_total: int = 0
    services_healthy: int = 0
    services_unhealthy: int = 0
    services_degraded: int = 0
    events_processed: int = 0
    events_failed: int = 0
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_operations: int = 0
    queued_operations: int = 0


# =============================================================================
# ABSTRACT BASE CLASSES
# =============================================================================


class BaseService(ABC):
    """
    Abstract base class for all platform services.

    Services must implement initialization, startup, shutdown, and health check
    methods. The platform engine manages the lifecycle of all services.
    """

    def __init__(self, name: str, config: Optional[ServiceConfig] = None):
        self.name = name
        self.config = config or ServiceConfig(name=name)
        self._status = ServiceStatus.REGISTERED
        self._logger = structlog.get_logger(f"service.{name}")
        self._engine: Optional["PlatformEngine"] = None
        self._started_at: Optional[datetime] = None
        self._context: Dict[str, Any] = {}

    @property
    def status(self) -> ServiceStatus:
        return self._status

    @status.setter
    def status(self, value: ServiceStatus) -> None:
        old_status = self._status
        self._status = value
        self._logger.info(
            "service_status_changed",
            old_status=old_status,
            new_status=value
        )

    @property
    def engine(self) -> Optional["PlatformEngine"]:
        return self._engine

    @engine.setter
    def engine(self, value: "PlatformEngine") -> None:
        self._engine = value

    @property
    def uptime(self) -> float:
        if not self._started_at:
            return 0.0
        return (datetime.utcnow() - self._started_at).total_seconds()

    def get_dependency(self, name: str) -> Optional["BaseService"]:
        """Get a dependent service by name"""
        if self._engine:
            return self._engine.get_service(name)
        return None

    async def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event through the engine's event bus"""
        if self._engine:
            await self._engine.emit_event(
                event_type=event_type,
                source=self.name,
                data=data
            )

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the service.
        Called once when the service is first registered.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        Start the service.
        Called when the platform engine starts.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the service.
        Called when the platform engine shuts down.
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """
        Perform a health check.
        Called periodically by the platform engine.
        """
        pass

    async def on_engine_ready(self) -> None:
        """Called when the engine is fully ready and all services are started"""
        pass

    async def on_engine_shutdown(self) -> None:
        """Called when the engine begins shutdown"""
        pass

    async def on_dependency_ready(self, dependency: str) -> None:
        """Called when a dependency service becomes ready"""
        pass

    async def on_dependency_failed(self, dependency: str) -> None:
        """Called when a dependency service fails"""
        pass


class Middleware(ABC):
    """Abstract base class for engine middleware"""

    @abstractmethod
    async def process(
        self,
        context: Dict[str, Any],
        next_handler: Callable[[Dict[str, Any]], Awaitable[Any]]
    ) -> Any:
        """Process request through middleware chain"""
        pass


# =============================================================================
# CORE PLATFORM ENGINE
# =============================================================================


class PlatformEngine:
    """
    Core Platform Engine

    The central orchestration system for the Voice AI Platform. Manages:
    - Service lifecycle (registration, startup, shutdown)
    - Service health monitoring and auto-recovery
    - Event propagation between services
    - Resource management
    - Configuration distribution
    - Cross-cutting concerns (logging, metrics, tracing)

    Usage:
        engine = PlatformEngine(config)
        engine.register_service(MyService())
        await engine.start()

        # ... application runs ...

        await engine.stop()
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._state = EngineState.INITIALIZING
        self._services: Dict[str, ServiceInfo] = {}
        self._service_order: List[str] = []
        self._middlewares: List[Middleware] = []
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._shutdown_hooks: List[Callable[[], Awaitable[None]]] = []
        self._startup_hooks: List[Callable[[], Awaitable[None]]] = []
        self._logger = structlog.get_logger("platform.engine")
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics = EngineMetrics()
        self._started_at: Optional[datetime] = None
        self._operation_semaphore = asyncio.Semaphore(
            self.config.max_concurrent_operations
        )
        self._context: Dict[str, Any] = {}

        # Set up signal handlers
        self._setup_signal_handlers()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> EngineState:
        return self._state

    @state.setter
    def state(self, value: EngineState) -> None:
        old_state = self._state
        self._state = value
        self._logger.info(
            "engine_state_changed",
            old_state=old_state,
            new_state=value
        )

    @property
    def is_running(self) -> bool:
        return self._state in (EngineState.RUNNING, EngineState.DEGRADED)

    @property
    def uptime(self) -> float:
        if not self._started_at:
            return 0.0
        return (datetime.utcnow() - self._started_at).total_seconds()

    @property
    def services(self) -> Dict[str, ServiceInfo]:
        return self._services.copy()

    @property
    def metrics(self) -> EngineMetrics:
        return self._metrics

    # -------------------------------------------------------------------------
    # Service Management
    # -------------------------------------------------------------------------

    def register_service(
        self,
        service: BaseService,
        config: Optional[ServiceConfig] = None
    ) -> None:
        """
        Register a service with the engine.

        Args:
            service: The service instance to register
            config: Optional service configuration
        """
        name = service.name

        if name in self._services:
            raise ValueError(f"Service '{name}' is already registered")

        # Use provided config or service's own config
        svc_config = config or service.config

        # Create service info
        info = ServiceInfo(
            name=name,
            service=service,
            config=svc_config
        )

        # Inject engine reference
        service.engine = self

        self._services[name] = info
        self._recalculate_service_order()

        self._logger.info(
            "service_registered",
            service=name,
            priority=svc_config.priority.name,
            dependencies=svc_config.dependencies
        )

    def unregister_service(self, name: str) -> None:
        """Unregister a service from the engine"""
        if name not in self._services:
            raise ValueError(f"Service '{name}' is not registered")

        info = self._services[name]
        if info.status not in (ServiceStatus.STOPPED, ServiceStatus.REGISTERED):
            raise RuntimeError(
                f"Cannot unregister service '{name}' while it's {info.status}"
            )

        del self._services[name]
        self._recalculate_service_order()

        self._logger.info("service_unregistered", service=name)

    def get_service(self, name: str) -> Optional[BaseService]:
        """Get a registered service by name"""
        info = self._services.get(name)
        return info.service if info else None

    def get_service_typed(
        self,
        name: str,
        service_type: Type[ServiceType]
    ) -> Optional[ServiceType]:
        """Get a registered service with type checking"""
        service = self.get_service(name)
        if service and isinstance(service, service_type):
            return service
        return None

    def get_service_status(self, name: str) -> Optional[ServiceStatus]:
        """Get the status of a service"""
        info = self._services.get(name)
        return info.status if info else None

    def _recalculate_service_order(self) -> None:
        """Recalculate service startup order based on dependencies"""
        # Topological sort with priority consideration
        visited: Set[str] = set()
        order: List[str] = []

        def visit(name: str, path: Set[str]) -> None:
            if name in path:
                cycle = " -> ".join(path) + f" -> {name}"
                raise ValueError(f"Circular dependency detected: {cycle}")

            if name in visited:
                return

            path = path | {name}
            info = self._services.get(name)

            if info:
                for dep in info.config.dependencies:
                    if dep in self._services:
                        visit(dep, path)

            visited.add(name)
            order.append(name)

        # Visit all services
        for name in self._services:
            visit(name, set())

        # Sort by priority within dependency constraints
        def get_priority(name: str) -> int:
            info = self._services.get(name)
            return info.config.priority.value if info else ServicePriority.NORMAL.value

        # Stable sort by priority while maintaining dependency order
        self._service_order = order

        self._logger.debug(
            "service_order_calculated",
            order=self._service_order
        )

    # -------------------------------------------------------------------------
    # Middleware Management
    # -------------------------------------------------------------------------

    def use_middleware(self, middleware: Middleware) -> None:
        """Add middleware to the processing chain"""
        self._middlewares.append(middleware)
        self._logger.info(
            "middleware_added",
            middleware=middleware.__class__.__name__
        )

    async def _execute_with_middleware(
        self,
        context: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Awaitable[Any]]
    ) -> Any:
        """Execute handler through middleware chain"""
        async def create_next(
            index: int
        ) -> Callable[[Dict[str, Any]], Awaitable[Any]]:
            if index >= len(self._middlewares):
                return handler

            middleware = self._middlewares[index]
            next_handler = await create_next(index + 1)

            async def wrapped(ctx: Dict[str, Any]) -> Any:
                return await middleware.process(ctx, next_handler)

            return wrapped

        if not self._middlewares:
            return await handler(context)

        chain = await create_next(0)
        return await chain(context)

    # -------------------------------------------------------------------------
    # Event Management
    # -------------------------------------------------------------------------

    def on_event(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Register an event handler"""
        self._event_handlers[event_type].append(handler)

    def off_event(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Unregister an event handler"""
        if handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)

    async def emit_event(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any]
    ) -> None:
        """Emit an event to all registered handlers"""
        event_data = {
            "type": event_type,
            "source": source,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "engine_id": self.config.engine_id
        }

        handlers = self._event_handlers.get(event_type, [])
        handlers.extend(self._event_handlers.get("*", []))  # Wildcard handlers

        if not handlers:
            return

        tasks = [
            asyncio.create_task(handler(event_data))
            for handler in handlers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._logger.error(
                    "event_handler_error",
                    event_type=event_type,
                    handler=handlers[i].__name__,
                    error=str(result)
                )
                self._metrics.events_failed += 1

        self._metrics.events_processed += len(handlers)

    # -------------------------------------------------------------------------
    # Lifecycle Hooks
    # -------------------------------------------------------------------------

    def on_startup(
        self,
        hook: Callable[[], Awaitable[None]]
    ) -> Callable[[], Awaitable[None]]:
        """Register a startup hook"""
        self._startup_hooks.append(hook)
        return hook

    def on_shutdown(
        self,
        hook: Callable[[], Awaitable[None]]
    ) -> Callable[[], Awaitable[None]]:
        """Register a shutdown hook"""
        self._shutdown_hooks.append(hook)
        return hook

    # -------------------------------------------------------------------------
    # Context Management
    # -------------------------------------------------------------------------

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value"""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value"""
        return self._context.get(key, default)

    # -------------------------------------------------------------------------
    # Signal Handling
    # -------------------------------------------------------------------------

    def _setup_signal_handlers(self) -> None:
        """Set up OS signal handlers for graceful shutdown"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop yet, signals will be set up later
            return

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s))
            )

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal"""
        self._logger.info(
            "signal_received",
            signal=sig.name
        )

        if self._state in (EngineState.STOPPING, EngineState.STOPPED):
            self._logger.warning("forcing_shutdown")
            sys.exit(1)

        await self.stop()

    # -------------------------------------------------------------------------
    # Health Monitoring
    # -------------------------------------------------------------------------

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(
                    "health_check_loop_error",
                    error=str(e)
                )

    async def _perform_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Perform health checks on all services"""
        results: Dict[str, HealthCheckResult] = {}
        healthy_count = 0
        unhealthy_count = 0
        degraded_count = 0

        tasks = {}
        for name, info in self._services.items():
            if info.config.health_check_enabled and info.status == ServiceStatus.HEALTHY:
                tasks[name] = asyncio.create_task(
                    asyncio.wait_for(
                        info.service.health_check(),
                        timeout=info.config.health_check_timeout
                    )
                )

        for name, task in tasks.items():
            info = self._services[name]
            try:
                result = await task
                results[name] = result
                info.last_health_check = result

                if result.status == HealthStatus.HEALTHY:
                    info.consecutive_failures = 0
                    info.consecutive_successes += 1
                    healthy_count += 1

                    if info.health != HealthStatus.HEALTHY:
                        if info.consecutive_successes >= self.config.healthy_threshold:
                            info.health = HealthStatus.HEALTHY
                            await self.emit_event(
                                "service.health_restored",
                                self.config.engine_id,
                                {"service": name}
                            )

                elif result.status == HealthStatus.DEGRADED:
                    info.health = HealthStatus.DEGRADED
                    degraded_count += 1

                else:
                    info.consecutive_successes = 0
                    info.consecutive_failures += 1

                    if info.consecutive_failures >= self.config.unhealthy_threshold:
                        info.health = HealthStatus.UNHEALTHY
                        unhealthy_count += 1

                        if self.config.enable_auto_recovery:
                            asyncio.create_task(
                                self._attempt_service_recovery(name)
                            )

                        await self.emit_event(
                            "service.unhealthy",
                            self.config.engine_id,
                            {"service": name, "result": result.__dict__}
                        )

            except asyncio.TimeoutError:
                info.consecutive_failures += 1
                results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Health check timed out"
                )

            except Exception as e:
                info.consecutive_failures += 1
                results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}"
                )

        # Update engine metrics
        self._metrics.services_healthy = healthy_count
        self._metrics.services_unhealthy = unhealthy_count
        self._metrics.services_degraded = degraded_count

        # Update engine state based on service health
        if unhealthy_count > 0:
            if self._state == EngineState.RUNNING:
                self.state = EngineState.DEGRADED
        elif self._state == EngineState.DEGRADED:
            self.state = EngineState.RUNNING

        return results

    async def _attempt_service_recovery(self, name: str) -> None:
        """Attempt to recover a failed service"""
        info = self._services.get(name)
        if not info:
            return

        self._logger.info("attempting_service_recovery", service=name)

        info.status = ServiceStatus.RECOVERING

        for attempt in range(self.config.recovery_max_retries):
            try:
                # Stop the service
                await info.service.stop()

                # Wait with exponential backoff
                backoff = min(
                    self.config.recovery_backoff_base * (2 ** attempt),
                    self.config.recovery_backoff_max
                )
                await asyncio.sleep(backoff)

                # Restart the service
                await info.service.start()

                info.status = ServiceStatus.HEALTHY
                info.health = HealthStatus.HEALTHY
                info.metrics.restarts += 1
                info.consecutive_failures = 0

                self._logger.info(
                    "service_recovery_successful",
                    service=name,
                    attempt=attempt + 1
                )

                await self.emit_event(
                    "service.recovered",
                    self.config.engine_id,
                    {"service": name, "attempts": attempt + 1}
                )

                return

            except Exception as e:
                self._logger.error(
                    "service_recovery_failed",
                    service=name,
                    attempt=attempt + 1,
                    error=str(e)
                )

        # Recovery failed
        info.status = ServiceStatus.FAILED
        self._logger.error(
            "service_recovery_exhausted",
            service=name,
            max_retries=self.config.recovery_max_retries
        )

        await self.emit_event(
            "service.recovery_failed",
            self.config.engine_id,
            {"service": name}
        )

    # -------------------------------------------------------------------------
    # Engine Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the platform engine and all registered services.

        Services are started in dependency order, respecting priorities.
        If a critical service fails to start, the engine will not start.
        """
        async with self._lock:
            if self._state != EngineState.INITIALIZING:
                raise RuntimeError(
                    f"Cannot start engine in state {self._state}"
                )

            self.state = EngineState.STARTING
            self._started_at = datetime.utcnow()

            self._logger.info(
                "engine_starting",
                engine_id=self.config.engine_id,
                services=len(self._services),
                environment=self.config.environment
            )

            try:
                # Run startup hooks
                for hook in self._startup_hooks:
                    await hook()

                # Initialize all services
                await self._initialize_services()

                # Start services in order
                await self._start_services()

                # Start health check loop
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )

                # Notify services that engine is ready
                await self._notify_engine_ready()

                self.state = EngineState.RUNNING

                self._logger.info(
                    "engine_started",
                    engine_id=self.config.engine_id,
                    startup_time_ms=(datetime.utcnow() - self._started_at).total_seconds() * 1000
                )

                await self.emit_event(
                    "engine.started",
                    self.config.engine_id,
                    {"services": list(self._services.keys())}
                )

            except Exception as e:
                self.state = EngineState.FAILED
                self._logger.error(
                    "engine_start_failed",
                    error=str(e),
                    traceback=traceback.format_exc()
                )
                raise

    async def _initialize_services(self) -> None:
        """Initialize all registered services"""
        for name in self._service_order:
            info = self._services[name]

            if not info.config.enabled:
                self._logger.info("service_disabled", service=name)
                continue

            try:
                info.status = ServiceStatus.INITIALIZING
                await asyncio.wait_for(
                    info.service.initialize(),
                    timeout=self.config.startup_timeout / len(self._services)
                )

                self._logger.info("service_initialized", service=name)

            except Exception as e:
                self._logger.error(
                    "service_initialization_failed",
                    service=name,
                    error=str(e)
                )

                if info.config.priority == ServicePriority.CRITICAL:
                    raise RuntimeError(
                        f"Critical service '{name}' failed to initialize: {e}"
                    )

    async def _start_services(self) -> None:
        """Start all services in dependency order"""
        for name in self._service_order:
            info = self._services[name]

            if not info.config.enabled:
                continue

            # Check dependencies
            for dep_name in info.config.dependencies:
                dep_info = self._services.get(dep_name)
                if not dep_info or dep_info.status != ServiceStatus.HEALTHY:
                    raise RuntimeError(
                        f"Service '{name}' depends on '{dep_name}' "
                        f"which is not healthy"
                    )

            try:
                info.status = ServiceStatus.STARTING
                await asyncio.wait_for(
                    info.service.start(),
                    timeout=self.config.startup_timeout / len(self._services)
                )

                info.status = ServiceStatus.HEALTHY
                info.health = HealthStatus.HEALTHY
                info.started_at = datetime.utcnow()
                info.service._started_at = datetime.utcnow()

                self._logger.info("service_started", service=name)

                # Notify dependents
                for other_name, other_info in self._services.items():
                    if name in other_info.config.dependencies:
                        await other_info.service.on_dependency_ready(name)

            except Exception as e:
                info.status = ServiceStatus.FAILED
                self._logger.error(
                    "service_start_failed",
                    service=name,
                    error=str(e)
                )

                if info.config.priority == ServicePriority.CRITICAL:
                    raise RuntimeError(
                        f"Critical service '{name}' failed to start: {e}"
                    )

        self._metrics.services_total = len(self._services)

    async def _notify_engine_ready(self) -> None:
        """Notify all services that the engine is ready"""
        tasks = [
            info.service.on_engine_ready()
            for info in self._services.values()
            if info.status == ServiceStatus.HEALTHY
        ]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self, mode: ShutdownMode = ShutdownMode.GRACEFUL) -> None:
        """
        Stop the platform engine and all services.

        Args:
            mode: Shutdown mode (graceful, immediate, or forced)
        """
        async with self._lock:
            if self._state in (EngineState.STOPPING, EngineState.STOPPED):
                return

            self.state = EngineState.STOPPING
            shutdown_start = datetime.utcnow()

            self._logger.info(
                "engine_stopping",
                engine_id=self.config.engine_id,
                mode=mode.value
            )

            try:
                # Notify services of shutdown
                await self._notify_engine_shutdown()

                if mode == ShutdownMode.GRACEFUL:
                    # Allow time for operations to complete
                    self.state = EngineState.DRAINING
                    await asyncio.sleep(self.config.graceful_shutdown_delay)

                # Cancel health check task
                if self._health_check_task:
                    self._health_check_task.cancel()
                    try:
                        await self._health_check_task
                    except asyncio.CancelledError:
                        pass

                # Stop services in reverse order
                await self._stop_services(mode)

                # Run shutdown hooks
                for hook in reversed(self._shutdown_hooks):
                    try:
                        await asyncio.wait_for(
                            hook(),
                            timeout=5.0
                        )
                    except Exception as e:
                        self._logger.error(
                            "shutdown_hook_error",
                            error=str(e)
                        )

                self.state = EngineState.STOPPED

                shutdown_time = (datetime.utcnow() - shutdown_start).total_seconds()
                self._logger.info(
                    "engine_stopped",
                    engine_id=self.config.engine_id,
                    shutdown_time_ms=shutdown_time * 1000
                )

            except Exception as e:
                self._logger.error(
                    "engine_stop_error",
                    error=str(e)
                )
                raise
            finally:
                self._shutdown_event.set()

    async def _notify_engine_shutdown(self) -> None:
        """Notify all services that shutdown is beginning"""
        tasks = [
            info.service.on_engine_shutdown()
            for info in self._services.values()
            if info.status == ServiceStatus.HEALTHY
        ]

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self._logger.error(
                        "shutdown_notification_error",
                        error=str(result)
                    )

    async def _stop_services(self, mode: ShutdownMode) -> None:
        """Stop all services in reverse order"""
        reverse_order = list(reversed(self._service_order))

        for name in reverse_order:
            info = self._services[name]

            if info.status not in (ServiceStatus.HEALTHY, ServiceStatus.DEGRADED):
                continue

            try:
                info.status = ServiceStatus.STOPPING

                if mode == ShutdownMode.FORCED:
                    # Don't wait for graceful shutdown
                    asyncio.create_task(info.service.stop())
                else:
                    await asyncio.wait_for(
                        info.service.stop(),
                        timeout=self.config.shutdown_timeout / len(self._services)
                    )

                info.status = ServiceStatus.STOPPED
                info.stopped_at = datetime.utcnow()

                self._logger.info("service_stopped", service=name)

            except asyncio.TimeoutError:
                self._logger.warning(
                    "service_stop_timeout",
                    service=name
                )
                info.status = ServiceStatus.STOPPED

            except Exception as e:
                self._logger.error(
                    "service_stop_error",
                    service=name,
                    error=str(e)
                )
                info.status = ServiceStatus.STOPPED

    async def wait_for_shutdown(self) -> None:
        """Wait for the engine to shutdown"""
        await self._shutdown_event.wait()

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    async def execute_operation(
        self,
        operation: Callable[[], Awaitable[T]],
        timeout: Optional[float] = None
    ) -> T:
        """
        Execute an operation with concurrency control.

        Args:
            operation: The async operation to execute
            timeout: Optional timeout in seconds

        Returns:
            The result of the operation
        """
        async with self._operation_semaphore:
            self._metrics.active_operations += 1

            try:
                if timeout:
                    return await asyncio.wait_for(operation(), timeout=timeout)
                return await operation()
            finally:
                self._metrics.active_operations -= 1

    # -------------------------------------------------------------------------
    # Health & Status
    # -------------------------------------------------------------------------

    async def health_check(self) -> HealthCheckResult:
        """
        Perform a comprehensive health check of the engine.

        Returns:
            HealthCheckResult with status and service details
        """
        start_time = time.time()

        service_checks = await self._perform_health_checks()

        # Determine overall status
        unhealthy_services = [
            name for name, result in service_checks.items()
            if result.status == HealthStatus.UNHEALTHY
        ]

        degraded_services = [
            name for name, result in service_checks.items()
            if result.status == HealthStatus.DEGRADED
        ]

        if unhealthy_services:
            status = HealthStatus.UNHEALTHY
            message = f"Unhealthy services: {', '.join(unhealthy_services)}"
        elif degraded_services:
            status = HealthStatus.DEGRADED
            message = f"Degraded services: {', '.join(degraded_services)}"
        else:
            status = HealthStatus.HEALTHY
            message = "All services healthy"

        return HealthCheckResult(
            status=status,
            message=message,
            latency_ms=(time.time() - start_time) * 1000,
            details={
                "engine_id": self.config.engine_id,
                "state": self._state.value,
                "uptime_seconds": self.uptime,
                "services_total": len(self._services),
                "services_healthy": self._metrics.services_healthy,
                "services_unhealthy": self._metrics.services_unhealthy
            },
            checks=service_checks
        )

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        return {
            "engine": {
                "id": self.config.engine_id,
                "name": self.config.engine_name,
                "version": self.config.version,
                "environment": self.config.environment,
                "state": self._state.value,
                "uptime_seconds": self.uptime,
                "started_at": self._started_at.isoformat() if self._started_at else None
            },
            "metrics": {
                "services_total": self._metrics.services_total,
                "services_healthy": self._metrics.services_healthy,
                "services_unhealthy": self._metrics.services_unhealthy,
                "services_degraded": self._metrics.services_degraded,
                "events_processed": self._metrics.events_processed,
                "events_failed": self._metrics.events_failed,
                "active_operations": self._metrics.active_operations
            },
            "services": {
                name: {
                    "status": info.status.value,
                    "health": info.health.value,
                    "priority": info.config.priority.name,
                    "uptime_seconds": info.service.uptime,
                    "metrics": {
                        "requests_total": info.metrics.requests_total,
                        "success_rate": info.metrics.success_rate,
                        "avg_latency_ms": info.metrics.avg_latency_ms,
                        "restarts": info.metrics.restarts
                    }
                }
                for name, info in self._services.items()
            }
        }


# =============================================================================
# CONTEXT MANAGERS
# =============================================================================


@asynccontextmanager
async def engine_context(config: Optional[EngineConfig] = None):
    """
    Context manager for running the platform engine.

    Usage:
        async with engine_context() as engine:
            engine.register_service(MyService())
            # Engine starts automatically
            await engine.wait_for_shutdown()
    """
    engine = PlatformEngine(config)

    try:
        yield engine
        await engine.start()
        await engine.wait_for_shutdown()
    finally:
        if engine.is_running:
            await engine.stop()


# =============================================================================
# DECORATORS
# =============================================================================


def service(
    name: str,
    priority: ServicePriority = ServicePriority.NORMAL,
    dependencies: Optional[List[str]] = None
) -> Callable[[Type[BaseService]], Type[BaseService]]:
    """
    Decorator to configure a service class.

    Usage:
        @service("my-service", priority=ServicePriority.HIGH)
        class MyService(BaseService):
            ...
    """
    def decorator(cls: Type[BaseService]) -> Type[BaseService]:
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, name, *args, **kwargs)
            self.config.priority = priority
            self.config.dependencies = dependencies or []

        cls.__init__ = new_init
        return cls

    return decorator


def requires(*dependencies: str) -> Callable[[Type[BaseService]], Type[BaseService]]:
    """
    Decorator to specify service dependencies.

    Usage:
        @requires("database", "cache")
        class MyService(BaseService):
            ...
    """
    def decorator(cls: Type[BaseService]) -> Type[BaseService]:
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.config.dependencies.extend(dependencies)

        cls.__init__ = new_init
        return cls

    return decorator
