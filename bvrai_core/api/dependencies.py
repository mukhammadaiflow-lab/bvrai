"""
API Dependencies

This module provides FastAPI dependencies for database sessions,
authentication, and other shared resources.
"""

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.base import get_session


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Usage in routes:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db_session)):
            repo = ItemRepository(db)
            return await repo.get_all()
    """
    async with get_session() as session:
        yield session


# Alias for shorter import
get_db = get_db_session
