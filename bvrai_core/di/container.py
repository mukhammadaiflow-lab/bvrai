"""
Dependency Injection Container
==============================

A lightweight, production-grade dependency injection container for managing
service lifetimes and dependencies in the BVRAI platform.

Features:
    - Singleton, scoped, and transient service lifetimes
    - Type-safe service registration and resolution
    - Async context manager support for scoped services
    - Factory-based lazy initialization
    - Thread-safe singleton management

Usage:
    # Register services
    container = Container()
    container.register_singleton(DatabaseManager, factory=create_database)
    container.register_scoped(UnitOfWork)
    container.register_transient(Logger)

    # Resolve services
    db = await container.resolve(DatabaseManager)

    # Scoped services (e.g., per-request)
    async with container.create_scope() as scope:
        uow = await scope.resolve(UnitOfWork)
        ...

Author: Platform Architecture Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    Union,
    List,
    Awaitable,
)
from weakref import WeakValueDictionary
import threading

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class ServiceLifetime(str, Enum):
    """Service lifetime definitions."""

    SINGLETON = "singleton"  # Single instance for application lifetime
    SCOPED = "scoped"  # Single instance per scope (e.g., request)
    TRANSIENT = "transient"  # New instance every time


@dataclass
class ServiceDescriptor(Generic[T]):
    """Describes how to create and manage a service."""

    service_type: Type[T]
    lifetime: ServiceLifetime
    factory: Optional[Callable[..., Union[T, Awaitable[T]]]] = None
    implementation_type: Optional[Type[T]] = None
    instance: Optional[T] = None  # For singleton instances

    def __post_init__(self):
        """Validate descriptor configuration."""
        if self.factory is None and self.implementation_type is None:
            raise ValueError(
                f"Service {self.service_type.__name__} must have either "
                "a factory or implementation_type"
            )


class ContainerError(Exception):
    """Base exception for container errors."""
    pass


class ServiceNotFoundError(ContainerError):
    """Service not registered in container."""
    pass


class CircularDependencyError(ContainerError):
    """Circular dependency detected during resolution."""
    pass


class Scope:
    """A dependency injection scope for scoped service management.

    Scopes are typically created per-request in web applications.
    Scoped services are instantiated once per scope and disposed
    when the scope ends.
    """

    def __init__(self, container: "Container"):
        """Initialize scope with parent container.

        Args:
            container: Parent container with service registrations
        """
        self._container = container
        self._scoped_instances: Dict[Type, Any] = {}
        self._resolution_stack: List[Type] = []

    async def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service within this scope.

        Args:
            service_type: The type of service to resolve

        Returns:
            Instance of the requested service

        Raises:
            ServiceNotFoundError: If service is not registered
            CircularDependencyError: If circular dependency detected
        """
        return await self._container._resolve_internal(
            service_type,
            self._scoped_instances,
            self._resolution_stack,
        )

    async def dispose(self):
        """Dispose all scoped instances that implement cleanup."""
        for instance in self._scoped_instances.values():
            if hasattr(instance, "dispose"):
                try:
                    if asyncio.iscoroutinefunction(instance.dispose):
                        await instance.dispose()
                    else:
                        instance.dispose()
                except Exception as e:
                    logger.error(f"Error disposing service: {e}")
            elif hasattr(instance, "close"):
                try:
                    if asyncio.iscoroutinefunction(instance.close):
                        await instance.close()
                    else:
                        instance.close()
                except Exception as e:
                    logger.error(f"Error closing service: {e}")

        self._scoped_instances.clear()


class Container:
    """Dependency injection container.

    Thread-safe container for managing service registrations and resolution.

    Example:
        container = Container()

        # Register singleton database
        container.register_singleton(
            DatabaseManager,
            factory=lambda: DatabaseManager(config.database_url)
        )

        # Register scoped unit of work
        container.register_scoped(
            UnitOfWork,
            factory=lambda db: UnitOfWork(db),
            dependencies=[DatabaseManager]
        )

        # Resolve service
        db = await container.resolve(DatabaseManager)

        # Use scoped services
        async with container.create_scope() as scope:
            uow = await scope.resolve(UnitOfWork)
    """

    def __init__(self):
        """Initialize empty container."""
        self._descriptors: Dict[Type, ServiceDescriptor] = {}
        self._singleton_instances: Dict[Type, Any] = {}
        self._lock = threading.Lock()

    def register_singleton(
        self,
        service_type: Type[T],
        factory: Optional[Callable[..., Union[T, Awaitable[T]]]] = None,
        implementation_type: Optional[Type[T]] = None,
        instance: Optional[T] = None,
    ) -> "Container":
        """Register a singleton service.

        Args:
            service_type: The service interface/type
            factory: Optional factory function to create instance
            implementation_type: Optional concrete implementation type
            instance: Optional pre-created instance

        Returns:
            Self for method chaining
        """
        with self._lock:
            if instance is not None:
                # Pre-created instance
                self._descriptors[service_type] = ServiceDescriptor(
                    service_type=service_type,
                    lifetime=ServiceLifetime.SINGLETON,
                    factory=lambda: instance,
                    instance=instance,
                )
                self._singleton_instances[service_type] = instance
            else:
                self._descriptors[service_type] = ServiceDescriptor(
                    service_type=service_type,
                    lifetime=ServiceLifetime.SINGLETON,
                    factory=factory,
                    implementation_type=implementation_type or service_type,
                )
        return self

    def register_scoped(
        self,
        service_type: Type[T],
        factory: Optional[Callable[..., Union[T, Awaitable[T]]]] = None,
        implementation_type: Optional[Type[T]] = None,
    ) -> "Container":
        """Register a scoped service (one instance per scope).

        Args:
            service_type: The service interface/type
            factory: Optional factory function
            implementation_type: Optional concrete type

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._descriptors[service_type] = ServiceDescriptor(
                service_type=service_type,
                lifetime=ServiceLifetime.SCOPED,
                factory=factory,
                implementation_type=implementation_type or service_type,
            )
        return self

    def register_transient(
        self,
        service_type: Type[T],
        factory: Optional[Callable[..., Union[T, Awaitable[T]]]] = None,
        implementation_type: Optional[Type[T]] = None,
    ) -> "Container":
        """Register a transient service (new instance every resolution).

        Args:
            service_type: The service interface/type
            factory: Optional factory function
            implementation_type: Optional concrete type

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._descriptors[service_type] = ServiceDescriptor(
                service_type=service_type,
                lifetime=ServiceLifetime.TRANSIENT,
                factory=factory,
                implementation_type=implementation_type or service_type,
            )
        return self

    @asynccontextmanager
    async def create_scope(self):
        """Create a new dependency injection scope.

        Usage:
            async with container.create_scope() as scope:
                service = await scope.resolve(MyService)
                ...
        """
        scope = Scope(self)
        try:
            yield scope
        finally:
            await scope.dispose()

    async def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service from the container.

        Note: For scoped services, use create_scope() instead.

        Args:
            service_type: The type to resolve

        Returns:
            Service instance
        """
        return await self._resolve_internal(service_type, {}, [])

    async def _resolve_internal(
        self,
        service_type: Type[T],
        scoped_instances: Dict[Type, Any],
        resolution_stack: List[Type],
    ) -> T:
        """Internal resolution with scope and circular dependency tracking.

        Args:
            service_type: Type to resolve
            scoped_instances: Dict for caching scoped instances
            resolution_stack: Stack for detecting circular dependencies

        Returns:
            Resolved service instance
        """
        # Check for circular dependency
        if service_type in resolution_stack:
            cycle = " -> ".join(t.__name__ for t in resolution_stack)
            raise CircularDependencyError(
                f"Circular dependency detected: {cycle} -> {service_type.__name__}"
            )

        # Get descriptor
        descriptor = self._descriptors.get(service_type)
        if descriptor is None:
            raise ServiceNotFoundError(
                f"Service {service_type.__name__} is not registered. "
                f"Available services: {list(self._descriptors.keys())}"
            )

        # Check for existing instance based on lifetime
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if service_type in self._singleton_instances:
                return self._singleton_instances[service_type]

        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if service_type in scoped_instances:
                return scoped_instances[service_type]

        # Create new instance
        resolution_stack.append(service_type)
        try:
            instance = await self._create_instance(
                descriptor, scoped_instances, resolution_stack
            )
        finally:
            resolution_stack.pop()

        # Cache based on lifetime
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            with self._lock:
                self._singleton_instances[service_type] = instance

        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            scoped_instances[service_type] = instance

        return instance

    async def _create_instance(
        self,
        descriptor: ServiceDescriptor[T],
        scoped_instances: Dict[Type, Any],
        resolution_stack: List[Type],
    ) -> T:
        """Create a service instance using factory or constructor.

        Args:
            descriptor: Service descriptor
            scoped_instances: Scoped instance cache
            resolution_stack: Circular dependency tracker

        Returns:
            New service instance
        """
        if descriptor.factory:
            # Use factory
            factory = descriptor.factory

            # Get factory parameters for dependency resolution
            sig = inspect.signature(factory)
            kwargs = {}

            for param_name, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    # Resolve dependency
                    dep_type = param.annotation
                    if dep_type in self._descriptors:
                        kwargs[param_name] = await self._resolve_internal(
                            dep_type, scoped_instances, resolution_stack
                        )

            # Call factory
            result = factory(**kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result

        elif descriptor.implementation_type:
            # Use constructor
            impl_type = descriptor.implementation_type

            # Get constructor parameters
            sig = inspect.signature(impl_type.__init__)
            kwargs = {}

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param.annotation != inspect.Parameter.empty:
                    dep_type = param.annotation
                    if dep_type in self._descriptors:
                        kwargs[param_name] = await self._resolve_internal(
                            dep_type, scoped_instances, resolution_stack
                        )

            return impl_type(**kwargs)

        raise ContainerError(
            f"Cannot create instance of {descriptor.service_type.__name__}"
        )

    async def dispose(self):
        """Dispose all singleton instances."""
        for instance in self._singleton_instances.values():
            if hasattr(instance, "dispose"):
                try:
                    if asyncio.iscoroutinefunction(instance.dispose):
                        await instance.dispose()
                    else:
                        instance.dispose()
                except Exception as e:
                    logger.error(f"Error disposing singleton: {e}")

        self._singleton_instances.clear()


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance.

    Returns:
        The global Container instance
    """
    global _container
    if _container is None:
        _container = Container()
    return _container


def configure_services(container: Container) -> Container:
    """Configure default services in the container.

    Override this function to customize service registrations.

    Args:
        container: Container to configure

    Returns:
        Configured container
    """
    # Example registrations - override in application code
    # container.register_singleton(DatabaseManager, factory=create_db)
    # container.register_scoped(UnitOfWork)
    return container


__all__ = [
    "Container",
    "Scope",
    "ServiceLifetime",
    "ServiceDescriptor",
    "ContainerError",
    "ServiceNotFoundError",
    "CircularDependencyError",
    "get_container",
    "configure_services",
]
