"""
Advanced Connection Pool Manager

Production-ready database connection management with:
- Connection pooling with configurable limits
- Health checks and automatic reconnection
- Read replica routing
- Connection metrics and monitoring
- Graceful shutdown handling
"""

from typing import Optional, Dict, Any, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager
import asyncio
import logging
import hashlib
import time

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import QueuePool, AsyncAdaptedQueuePool
from sqlalchemy import text, event

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConnectionState(str, Enum):
    """Connection state."""
    IDLE = "idle"
    ACTIVE = "active"
    FAILED = "failed"
    CLOSED = "closed"


class DatabaseRole(str, Enum):
    """Database role for routing."""
    PRIMARY = "primary"
    REPLICA = "replica"
    ANALYTICS = "analytics"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    database: str
    username: str
    password: str
    driver: str = "postgresql+asyncpg"
    role: DatabaseRole = DatabaseRole.PRIMARY

    # Pool settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 1800
    pool_pre_ping: bool = True

    # Connection settings
    connect_timeout: int = 10
    command_timeout: int = 30
    ssl_mode: str = "prefer"

    # Additional options
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str:
        """Get connection URL."""
        return f"{self.driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def safe_url(self) -> str:
        """Get URL without password for logging."""
        return f"{self.driver}://{self.username}:***@{self.host}:{self.port}/{self.database}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "role": self.role.value,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
        }


@dataclass
class ConnectionMetrics:
    """Connection pool metrics."""
    pool_size: int = 0
    checked_out: int = 0
    checked_in: int = 0
    overflow: int = 0
    invalidated: int = 0
    total_connections: int = 0
    total_queries: int = 0
    failed_queries: int = 0
    avg_query_time_ms: float = 0.0
    last_health_check: Optional[datetime] = None
    is_healthy: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pool_size": self.pool_size,
            "checked_out": self.checked_out,
            "checked_in": self.checked_in,
            "overflow": self.overflow,
            "invalidated": self.invalidated,
            "total_connections": self.total_connections,
            "total_queries": self.total_queries,
            "failed_queries": self.failed_queries,
            "avg_query_time_ms": self.avg_query_time_ms,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "is_healthy": self.is_healthy,
        }


class ConnectionPool:
    """
    Advanced connection pool with health monitoring.

    Features:
    - Configurable pool size and overflow
    - Automatic health checks
    - Query timing and metrics
    - Event hooks for monitoring
    """

    def __init__(
        self,
        config: DatabaseConfig,
        name: str = "default",
    ):
        self.config = config
        self.name = name
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._metrics = ConnectionMetrics()
        self._query_times: List[float] = []
        self._max_query_samples = 1000
        self._health_check_interval = 30
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._is_initialized:
            return

        async with self._lock:
            if self._is_initialized:
                return

            logger.info(f"Initializing connection pool '{self.name}': {self.config.safe_url}")

            # Create engine with pool settings
            self._engine = create_async_engine(
                self.config.url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=False,
                connect_args={
                    "timeout": self.config.connect_timeout,
                    "command_timeout": self.config.command_timeout,
                },
            )

            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )

            # Set up event listeners
            self._setup_event_listeners()

            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            self._is_initialized = True
            logger.info(f"Connection pool '{self.name}' initialized successfully")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._engine:
            await self._engine.dispose()
            self._engine = None

        self._is_initialized = False
        logger.info(f"Connection pool '{self.name}' closed")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a database session from the pool."""
        if not self._is_initialized:
            await self.initialize()

        async with self._session_factory() as session:
            self._metrics.checked_out += 1
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self._metrics.failed_queries += 1
                raise
            finally:
                self._metrics.checked_in += 1

    @asynccontextmanager
    async def acquire_readonly(self):
        """Acquire a read-only session."""
        async with self.acquire() as session:
            # Set transaction to read-only
            await session.execute(text("SET TRANSACTION READ ONLY"))
            yield session

    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a query and return results."""
        start_time = time.time()

        async with self.acquire() as session:
            result = await session.execute(text(query), params or {})
            self._record_query_time(time.time() - start_time)
            return result

    async def fetch_one(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single row."""
        result = await self.execute(query, params)
        row = result.fetchone()
        return dict(row._mapping) if row else None

    async def fetch_all(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all rows."""
        result = await self.execute(query, params)
        return [dict(row._mapping) for row in result.fetchall()]

    async def health_check(self) -> bool:
        """Perform health check on the connection pool."""
        try:
            async with self.acquire() as session:
                await session.execute(text("SELECT 1"))
            self._metrics.is_healthy = True
            self._metrics.last_health_check = datetime.utcnow()
            return True
        except Exception as e:
            logger.error(f"Health check failed for pool '{self.name}': {e}")
            self._metrics.is_healthy = False
            self._metrics.last_health_check = datetime.utcnow()
            return False

    def get_metrics(self) -> ConnectionMetrics:
        """Get current pool metrics."""
        if self._engine:
            pool = self._engine.pool
            self._metrics.pool_size = pool.size() if hasattr(pool, 'size') else 0
            self._metrics.overflow = pool.overflow() if hasattr(pool, 'overflow') else 0

        if self._query_times:
            self._metrics.avg_query_time_ms = (sum(self._query_times) / len(self._query_times)) * 1000

        return self._metrics

    def _record_query_time(self, duration: float) -> None:
        """Record query execution time."""
        self._query_times.append(duration)
        if len(self._query_times) > self._max_query_samples:
            self._query_times = self._query_times[-self._max_query_samples:]
        self._metrics.total_queries += 1

    def _setup_event_listeners(self) -> None:
        """Set up SQLAlchemy event listeners."""
        if not self._engine:
            return

        @event.listens_for(self._engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            self._metrics.total_connections += 1

        @event.listens_for(self._engine.sync_engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            connection_record.info["checkout_time"] = time.time()

        @event.listens_for(self._engine.sync_engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            checkout_time = connection_record.info.get("checkout_time")
            if checkout_time:
                duration = time.time() - checkout_time
                self._record_query_time(duration)

        @event.listens_for(self._engine.sync_engine, "invalidate")
        def on_invalidate(dbapi_connection, connection_record, exception):
            self._metrics.invalidated += 1
            logger.warning(f"Connection invalidated in pool '{self.name}': {exception}")

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")


class DatabaseConnection:
    """
    Single database connection wrapper with query tracking.
    """

    def __init__(
        self,
        session: AsyncSession,
        pool: ConnectionPool,
    ):
        self.session = session
        self.pool = pool
        self._query_count = 0
        self._start_time = time.time()

    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query."""
        self._query_count += 1
        return await self.session.execute(text(query), params or {})

    async def commit(self) -> None:
        """Commit the transaction."""
        await self.session.commit()

    async def rollback(self) -> None:
        """Rollback the transaction."""
        await self.session.rollback()

    @property
    def duration(self) -> float:
        """Get connection duration in seconds."""
        return time.time() - self._start_time

    @property
    def query_count(self) -> int:
        """Get number of queries executed."""
        return self._query_count


class ConnectionManager:
    """
    Manages multiple database connections with routing.

    Features:
    - Primary/replica routing
    - Automatic failover
    - Connection affinity
    - Load balancing
    """

    def __init__(self):
        self._pools: Dict[str, ConnectionPool] = {}
        self._primary: Optional[ConnectionPool] = None
        self._replicas: List[ConnectionPool] = []
        self._replica_index = 0
        self._lock = asyncio.Lock()

    async def add_pool(
        self,
        name: str,
        config: DatabaseConfig,
    ) -> ConnectionPool:
        """Add a connection pool."""
        pool = ConnectionPool(config, name)
        await pool.initialize()

        self._pools[name] = pool

        if config.role == DatabaseRole.PRIMARY:
            self._primary = pool
        elif config.role == DatabaseRole.REPLICA:
            self._replicas.append(pool)

        logger.info(f"Added connection pool '{name}' with role {config.role.value}")
        return pool

    def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get a pool by name."""
        return self._pools.get(name)

    def get_primary(self) -> Optional[ConnectionPool]:
        """Get the primary pool."""
        return self._primary

    def get_replica(self) -> Optional[ConnectionPool]:
        """Get a replica pool using round-robin."""
        if not self._replicas:
            return self._primary

        # Round-robin selection
        pool = self._replicas[self._replica_index % len(self._replicas)]
        self._replica_index += 1
        return pool

    @asynccontextmanager
    async def acquire_primary(self):
        """Acquire a session from the primary pool."""
        if not self._primary:
            raise RuntimeError("No primary database configured")

        async with self._primary.acquire() as session:
            yield session

    @asynccontextmanager
    async def acquire_replica(self):
        """Acquire a session from a replica pool."""
        pool = self.get_replica()
        if not pool:
            raise RuntimeError("No database pool available")

        async with pool.acquire_readonly() as session:
            yield session

    @asynccontextmanager
    async def acquire(self, readonly: bool = False):
        """Acquire a session with automatic routing."""
        if readonly and self._replicas:
            async with self.acquire_replica() as session:
                yield session
        else:
            async with self.acquire_primary() as session:
                yield session

    async def health_check_all(self) -> Dict[str, bool]:
        """Run health checks on all pools."""
        results = {}
        for name, pool in self._pools.items():
            results[name] = await pool.health_check()
        return results

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all pools."""
        return {
            name: pool.get_metrics().to_dict()
            for name, pool in self._pools.items()
        }

    async def close_all(self) -> None:
        """Close all connection pools."""
        for pool in self._pools.values():
            await pool.close()
        self._pools.clear()
        self._primary = None
        self._replicas.clear()


# Global connection manager
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get or create the global connection manager."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


class TransactionContext:
    """
    Transaction context manager with savepoint support.

    Usage:
        async with TransactionContext(session) as tx:
            await tx.execute("INSERT ...")
            async with tx.savepoint() as sp:
                await tx.execute("UPDATE ...")
                # Rollback to savepoint
                await sp.rollback()
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self._savepoint_counter = 0

    async def __aenter__(self) -> "TransactionContext":
        await self.session.begin()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.session.rollback()
        else:
            await self.session.commit()

    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query within the transaction."""
        return await self.session.execute(text(query), params or {})

    @asynccontextmanager
    async def savepoint(self):
        """Create a savepoint."""
        self._savepoint_counter += 1
        savepoint_name = f"sp_{self._savepoint_counter}"

        await self.session.execute(text(f"SAVEPOINT {savepoint_name}"))

        class Savepoint:
            def __init__(self, session: AsyncSession, name: str):
                self._session = session
                self._name = name

            async def rollback(self):
                await self._session.execute(text(f"ROLLBACK TO SAVEPOINT {self._name}"))

            async def release(self):
                await self._session.execute(text(f"RELEASE SAVEPOINT {self._name}"))

        sp = Savepoint(self.session, savepoint_name)
        try:
            yield sp
            await sp.release()
        except Exception:
            await sp.rollback()
            raise
