"""
Enterprise Database Layer

Production-ready database abstraction with:
- Connection pooling with health monitoring
- Query builder with SQL injection prevention
- Migration system with rollback support
- Repository pattern with Unit of Work
- Read replica routing
- Query caching and optimization
- Sharding support for horizontal scaling
"""

from app.database.session import get_db, engine, AsyncSessionLocal, init_db, close_db
from app.database.models import Base, Agent, Call, CallLog, KnowledgeBase, User, APIKey

from app.database.connection import (
    DatabaseConfig,
    ConnectionPool,
    DatabaseConnection,
    ConnectionManager,
    ConnectionMetrics,
    DatabaseRole,
    TransactionContext,
    get_connection_manager,
)

from app.database.query import (
    QueryBuilder,
    SelectQuery,
    InsertQuery,
    UpdateQuery,
    DeleteQuery,
    RawQuery,
    QueryResult,
    Expression,
    Condition,
    Operator,
    JoinType,
    OrderDirection,
)

from app.database.repository import (
    Repository,
    BaseRepository,
    CRUDRepository,
    UnitOfWork,
    TransactionManager,
    Specification,
    QueryOptions,
    OptimisticLockError,
)

from app.database.migration import (
    Migration,
    MigrationRunner,
    MigrationHistory,
    MigrationStatus,
    SchemaBuilder,
    TableBuilder,
    Column,
    ColumnType,
    Index,
    ForeignKey,
)

from app.database.sharding import (
    ShardingStrategy,
    HashSharding,
    RangeSharding,
    DirectorySharding,
    ShardRouter,
    ShardManager,
    ShardConfig,
    ShardKey,
    ShardState,
)

from app.database.optimization import (
    QueryAnalyzer,
    QueryCache,
    IndexAdvisor,
    SlowQueryLog,
    QueryPlan,
    SlowQuery,
    IndexRecommendation,
)

__all__ = [
    # Session
    "get_db",
    "engine",
    "AsyncSessionLocal",
    "init_db",
    "close_db",
    # Models
    "Base",
    "Agent",
    "Call",
    "CallLog",
    "KnowledgeBase",
    "User",
    "APIKey",
    # Connection
    "DatabaseConfig",
    "ConnectionPool",
    "DatabaseConnection",
    "ConnectionManager",
    "ConnectionMetrics",
    "DatabaseRole",
    "TransactionContext",
    "get_connection_manager",
    # Query
    "QueryBuilder",
    "SelectQuery",
    "InsertQuery",
    "UpdateQuery",
    "DeleteQuery",
    "RawQuery",
    "QueryResult",
    "Expression",
    "Condition",
    "Operator",
    "JoinType",
    "OrderDirection",
    # Repository
    "Repository",
    "BaseRepository",
    "CRUDRepository",
    "UnitOfWork",
    "TransactionManager",
    "Specification",
    "QueryOptions",
    "OptimisticLockError",
    # Migration
    "Migration",
    "MigrationRunner",
    "MigrationHistory",
    "MigrationStatus",
    "SchemaBuilder",
    "TableBuilder",
    "Column",
    "ColumnType",
    "Index",
    "ForeignKey",
    # Sharding
    "ShardingStrategy",
    "HashSharding",
    "RangeSharding",
    "DirectorySharding",
    "ShardRouter",
    "ShardManager",
    "ShardConfig",
    "ShardKey",
    "ShardState",
    # Optimization
    "QueryAnalyzer",
    "QueryCache",
    "IndexAdvisor",
    "SlowQueryLog",
    "QueryPlan",
    "SlowQuery",
    "IndexRecommendation",
]
