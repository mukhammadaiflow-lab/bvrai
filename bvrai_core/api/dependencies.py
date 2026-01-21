"""
API Dependencies

FastAPI dependency injection providers for database sessions,
authentication, and other common dependencies.
"""

import logging
import os
from typing import AsyncGenerator, Optional

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.base import DatabaseManager
from ..api.middleware.org_context import async_set_org_context
from .auth import AuthContext


logger = logging.getLogger(__name__)

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get or create the global database manager.

    Returns:
        DatabaseManager instance
    """
    global _db_manager

    if _db_manager is None:
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://localhost/bvrai"
        )
        _db_manager = DatabaseManager(database_url)

    return _db_manager


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    db_manager = get_db_manager()
    async with db_manager.session() as session:
        yield session


async def get_db_with_org_context(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AsyncSession:
    """
    Get database session with organization context set for RLS.

    This dependency:
    1. Gets a database session
    2. Sets the app.current_org_id PostgreSQL variable
    3. Returns the session for use in the handler

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db_with_org_context)):
            # Queries are now filtered by organization
            ...
    """
    # Get auth context from request state
    auth_context: Optional[AuthContext] = getattr(
        request.state, "auth_context", None
    )

    # Set organization context for RLS
    if auth_context and auth_context.organization_id:
        await async_set_org_context(db, auth_context.organization_id)

    return db


def init_db(database_url: str) -> DatabaseManager:
    """
    Initialize the database manager with a specific URL.

    Call this at application startup.

    Args:
        database_url: Database connection URL

    Returns:
        Configured DatabaseManager
    """
    global _db_manager
    _db_manager = DatabaseManager(database_url)
    return _db_manager


async def close_db() -> None:
    """
    Close database connections.

    Call this at application shutdown.
    """
    global _db_manager
    if _db_manager:
        await _db_manager.close()
        _db_manager = None


__all__ = [
    "get_db_manager",
    "get_db",
    "get_db_with_org_context",
    "init_db",
    "close_db",
]
