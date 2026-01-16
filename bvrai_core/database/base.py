"""
Database Base Module

Provides the base classes, mixins, and database management
infrastructure for the platform.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
)

from sqlalchemy import (
    Column,
    DateTime,
    String,
    Boolean,
    Text,
    Integer,
    Float,
    ForeignKey,
    JSON,
    Index,
    event,
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


# =============================================================================
# Base Declarative Class
# =============================================================================


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    # Common columns for all models
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Base":
        """Create model from dictionary."""
        return cls(**{
            k: v for k, v in data.items()
            if k in [c.name for c in cls.__table__.columns]
        })


# =============================================================================
# Mixins
# =============================================================================


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
    )

    def soft_delete(self) -> None:
        """Mark as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()

    def restore(self) -> None:
        """Restore from deletion."""
        self.is_deleted = False
        self.deleted_at = None


class AuditMixin:
    """Mixin for audit trail."""

    created_by: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
    )
    updated_by: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
    )


# =============================================================================
# Database Manager
# =============================================================================


class DatabaseManager:
    """
    Database connection and session management.

    Handles connection pooling, session creation, and lifecycle.
    """

    _instance: Optional["DatabaseManager"] = None

    def __init__(
        self,
        database_url: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 1800,
        echo: bool = False,
    ):
        """
        Initialize database manager.

        Args:
            database_url: Database connection URL
            pool_size: Size of connection pool
            max_overflow: Max connections beyond pool size
            pool_timeout: Timeout for acquiring connection
            pool_recycle: Recycle connections after this many seconds
            echo: Echo SQL statements (for debugging)
        """
        # Convert sync URL to async if needed
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://"
            )
        elif database_url.startswith("mysql://"):
            database_url = database_url.replace(
                "mysql://", "mysql+aiomysql://"
            )
        elif database_url.startswith("sqlite://"):
            database_url = database_url.replace(
                "sqlite://", "sqlite+aiosqlite://"
            )

        self._database_url = database_url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle
        self._echo = echo

        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None

    @property
    def engine(self) -> AsyncEngine:
        """Get or create the database engine."""
        if self._engine is None:
            self._engine = create_async_engine(
                self._database_url,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=self._pool_timeout,
                pool_recycle=self._pool_recycle,
                echo=self._echo,
            )
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._session_factory

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session.

        Usage:
            async with db.session() as session:
                # use session
        """
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def create_all(self) -> None:
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_all(self) -> None:
        """Drop all tables (use with caution!)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the database connection."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None


# =============================================================================
# Global Instance Management
# =============================================================================


_database: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    global _database
    if _database is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _database


def init_database(
    database_url: str,
    **kwargs,
) -> DatabaseManager:
    """
    Initialize the global database manager.

    Args:
        database_url: Database connection URL
        **kwargs: Additional arguments for DatabaseManager
    """
    global _database
    _database = DatabaseManager(database_url, **kwargs)
    return _database


async def close_database() -> None:
    """Close the global database connection."""
    global _database
    if _database:
        await _database.close()
        _database = None


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session from the global manager."""
    db = get_database()
    async with db.session() as session:
        yield session


# Type alias for session factory
AsyncSessionFactory = async_sessionmaker[AsyncSession]


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Base classes
    "Base",
    "TimestampMixin",
    "SoftDeleteMixin",
    "AuditMixin",
    # Database manager
    "DatabaseManager",
    "get_database",
    "init_database",
    "close_database",
    # Session management
    "get_session",
    "AsyncSessionFactory",
]
