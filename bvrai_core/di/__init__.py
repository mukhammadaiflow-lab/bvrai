"""
Dependency Injection Module
===========================

Provides dependency injection infrastructure for the BVRAI platform.

Usage:
    from bvrai_core.di import Container, get_container

    # Get global container
    container = get_container()

    # Register services
    container.register_singleton(DatabaseManager, factory=create_db)
    container.register_scoped(UnitOfWork)

    # Resolve services
    db = await container.resolve(DatabaseManager)

    # Use scoped services
    async with container.create_scope() as scope:
        uow = await scope.resolve(UnitOfWork)
"""

from .container import (
    Container,
    Scope,
    ServiceLifetime,
    ServiceDescriptor,
    ContainerError,
    ServiceNotFoundError,
    CircularDependencyError,
    get_container,
    configure_services,
)

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
