"""
Data Isolation Strategies

Multi-tenant data isolation with:
- Row-level isolation
- Schema-based isolation
- Database-level isolation
- Hybrid approaches
"""

from typing import Optional, Dict, Any, List, Type, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class IsolationLevel(str, Enum):
    """Data isolation levels."""
    ROW = "row"  # Shared tables with tenant_id column
    SCHEMA = "schema"  # Separate schema per tenant
    DATABASE = "database"  # Separate database per tenant
    HYBRID = "hybrid"  # Mixed based on data sensitivity


@dataclass
class IsolationConfig:
    """Configuration for isolation strategy."""
    level: IsolationLevel = IsolationLevel.ROW
    tenant_id_column: str = "tenant_id"
    schema_prefix: str = "tenant_"
    database_prefix: str = "tenant_"
    auto_create_schema: bool = True
    auto_create_database: bool = False
    enable_cross_tenant_queries: bool = False
    audit_access: bool = True


class IsolationStrategy(ABC):
    """Abstract base for isolation strategies."""

    def __init__(self, config: Optional[IsolationConfig] = None):
        self.config = config or IsolationConfig()

    @abstractmethod
    async def apply_tenant_filter(
        self,
        query: Any,
        tenant_id: str,
    ) -> Any:
        """Apply tenant filter to query."""
        pass

    @abstractmethod
    async def get_tenant_connection(
        self,
        tenant_id: str,
    ) -> Any:
        """Get database connection for tenant."""
        pass

    @abstractmethod
    async def ensure_tenant_isolated(
        self,
        tenant_id: str,
    ) -> bool:
        """Ensure tenant isolation is set up."""
        pass

    @abstractmethod
    async def validate_tenant_access(
        self,
        tenant_id: str,
        resource_tenant_id: str,
    ) -> bool:
        """Validate tenant can access resource."""
        pass


class RowLevelIsolation(IsolationStrategy):
    """
    Row-level tenant isolation.

    Uses tenant_id column in all tables.
    Most efficient for small to medium tenants.
    """

    def __init__(
        self,
        config: Optional[IsolationConfig] = None,
    ):
        super().__init__(config)
        self._tenant_filters: Dict[str, str] = {}

    async def apply_tenant_filter(
        self,
        query: Any,
        tenant_id: str,
    ) -> Any:
        """Add tenant_id filter to query."""
        # For SQLAlchemy-style queries
        if hasattr(query, 'filter'):
            return query.filter_by(**{self.config.tenant_id_column: tenant_id})

        # For raw SQL queries
        if isinstance(query, str):
            if 'WHERE' in query.upper():
                return f"{query} AND {self.config.tenant_id_column} = '{tenant_id}'"
            else:
                return f"{query} WHERE {self.config.tenant_id_column} = '{tenant_id}'"

        return query

    async def get_tenant_connection(
        self,
        tenant_id: str,
    ) -> Any:
        """Return shared connection with tenant context."""
        # Row-level uses shared connection
        # Tenant filtering is applied at query level
        return {
            "tenant_id": tenant_id,
            "isolation_level": IsolationLevel.ROW.value,
        }

    async def ensure_tenant_isolated(
        self,
        tenant_id: str,
    ) -> bool:
        """Row-level isolation is always ready."""
        return True

    async def validate_tenant_access(
        self,
        tenant_id: str,
        resource_tenant_id: str,
    ) -> bool:
        """Validate tenant owns resource."""
        if self.config.enable_cross_tenant_queries:
            return True
        return tenant_id == resource_tenant_id

    def build_insert_values(
        self,
        tenant_id: str,
        values: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add tenant_id to insert values."""
        return {**values, self.config.tenant_id_column: tenant_id}

    def get_filter_clause(self, tenant_id: str) -> str:
        """Get SQL filter clause."""
        return f"{self.config.tenant_id_column} = '{tenant_id}'"


class SchemaIsolation(IsolationStrategy):
    """
    Schema-level tenant isolation.

    Creates separate database schema per tenant.
    Better isolation than row-level.
    """

    def __init__(
        self,
        config: Optional[IsolationConfig] = None,
        db_connection: Any = None,
    ):
        super().__init__(config)
        self._db = db_connection
        self._schema_cache: Dict[str, bool] = {}

    def _get_schema_name(self, tenant_id: str) -> str:
        """Get schema name for tenant."""
        return f"{self.config.schema_prefix}{tenant_id}"

    async def apply_tenant_filter(
        self,
        query: Any,
        tenant_id: str,
    ) -> Any:
        """Set schema search path for query."""
        schema_name = self._get_schema_name(tenant_id)

        if isinstance(query, str):
            # Prepend SET search_path for raw SQL
            return f"SET search_path TO {schema_name}; {query}"

        return query

    async def get_tenant_connection(
        self,
        tenant_id: str,
    ) -> Any:
        """Get connection with tenant schema."""
        schema_name = self._get_schema_name(tenant_id)

        # Ensure schema exists
        await self.ensure_tenant_isolated(tenant_id)

        return {
            "tenant_id": tenant_id,
            "schema": schema_name,
            "isolation_level": IsolationLevel.SCHEMA.value,
        }

    async def ensure_tenant_isolated(
        self,
        tenant_id: str,
    ) -> bool:
        """Create schema if needed."""
        schema_name = self._get_schema_name(tenant_id)

        if schema_name in self._schema_cache:
            return True

        if self.config.auto_create_schema:
            # Create schema (PostgreSQL example)
            create_sql = f"CREATE SCHEMA IF NOT EXISTS {schema_name}"
            logger.info(f"Creating schema: {schema_name}")

            # In production, execute: await self._db.execute(create_sql)
            self._schema_cache[schema_name] = True

        return schema_name in self._schema_cache

    async def validate_tenant_access(
        self,
        tenant_id: str,
        resource_tenant_id: str,
    ) -> bool:
        """Schema isolation prevents cross-tenant access by design."""
        if self.config.enable_cross_tenant_queries:
            return True
        return tenant_id == resource_tenant_id

    async def migrate_schema(
        self,
        tenant_id: str,
        migrations: List[str],
    ) -> bool:
        """Apply migrations to tenant schema."""
        schema_name = self._get_schema_name(tenant_id)

        for migration in migrations:
            schema_migration = f"SET search_path TO {schema_name}; {migration}"
            logger.info(f"Applying migration to {schema_name}")
            # In production: await self._db.execute(schema_migration)

        return True

    async def drop_schema(self, tenant_id: str) -> bool:
        """Drop tenant schema (for cleanup)."""
        schema_name = self._get_schema_name(tenant_id)
        drop_sql = f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"

        logger.warning(f"Dropping schema: {schema_name}")
        # In production: await self._db.execute(drop_sql)

        if schema_name in self._schema_cache:
            del self._schema_cache[schema_name]

        return True


class DatabaseIsolation(IsolationStrategy):
    """
    Database-level tenant isolation.

    Creates separate database per tenant.
    Maximum isolation and security.
    """

    def __init__(
        self,
        config: Optional[IsolationConfig] = None,
        db_factory: Any = None,
    ):
        super().__init__(config)
        self._db_factory = db_factory
        self._connections: Dict[str, Any] = {}

    def _get_database_name(self, tenant_id: str) -> str:
        """Get database name for tenant."""
        return f"{self.config.database_prefix}{tenant_id}"

    async def apply_tenant_filter(
        self,
        query: Any,
        tenant_id: str,
    ) -> Any:
        """No filter needed - database is already isolated."""
        return query

    async def get_tenant_connection(
        self,
        tenant_id: str,
    ) -> Any:
        """Get dedicated connection to tenant database."""
        db_name = self._get_database_name(tenant_id)

        if tenant_id not in self._connections:
            # Ensure database exists
            await self.ensure_tenant_isolated(tenant_id)

            # Create connection
            self._connections[tenant_id] = {
                "database": db_name,
                "tenant_id": tenant_id,
                "isolation_level": IsolationLevel.DATABASE.value,
            }

        return self._connections[tenant_id]

    async def ensure_tenant_isolated(
        self,
        tenant_id: str,
    ) -> bool:
        """Create database if needed."""
        db_name = self._get_database_name(tenant_id)

        if self.config.auto_create_database:
            # Create database (PostgreSQL example)
            create_sql = f"CREATE DATABASE {db_name}"
            logger.info(f"Creating database: {db_name}")
            # In production: execute on master connection

        return True

    async def validate_tenant_access(
        self,
        tenant_id: str,
        resource_tenant_id: str,
    ) -> bool:
        """Database isolation prevents cross-tenant access by design."""
        # Cross-tenant queries not possible with separate databases
        return tenant_id == resource_tenant_id

    async def close_connection(self, tenant_id: str) -> None:
        """Close tenant database connection."""
        if tenant_id in self._connections:
            # In production: close actual connection
            del self._connections[tenant_id]

    async def drop_database(self, tenant_id: str) -> bool:
        """Drop tenant database (for cleanup)."""
        db_name = self._get_database_name(tenant_id)

        await self.close_connection(tenant_id)

        drop_sql = f"DROP DATABASE IF EXISTS {db_name}"
        logger.warning(f"Dropping database: {db_name}")
        # In production: execute on master connection

        return True


class HybridIsolation(IsolationStrategy):
    """
    Hybrid isolation strategy.

    Uses different isolation levels based on data sensitivity.
    - Sensitive data: Schema or database isolation
    - Regular data: Row-level isolation
    """

    def __init__(
        self,
        config: Optional[IsolationConfig] = None,
        sensitive_tables: Optional[List[str]] = None,
    ):
        super().__init__(config)
        self.sensitive_tables = set(sensitive_tables or [
            "payments", "financial_data", "pii_data",
            "audit_logs", "credentials", "api_keys",
        ])
        self._row_isolation = RowLevelIsolation(config)
        self._schema_isolation = SchemaIsolation(config)

    def _is_sensitive(self, table_name: str) -> bool:
        """Check if table contains sensitive data."""
        return table_name.lower() in self.sensitive_tables

    async def apply_tenant_filter(
        self,
        query: Any,
        tenant_id: str,
        table_name: Optional[str] = None,
    ) -> Any:
        """Apply appropriate isolation based on table."""
        if table_name and self._is_sensitive(table_name):
            return await self._schema_isolation.apply_tenant_filter(query, tenant_id)
        return await self._row_isolation.apply_tenant_filter(query, tenant_id)

    async def get_tenant_connection(
        self,
        tenant_id: str,
        table_name: Optional[str] = None,
    ) -> Any:
        """Get appropriate connection based on sensitivity."""
        if table_name and self._is_sensitive(table_name):
            return await self._schema_isolation.get_tenant_connection(tenant_id)
        return await self._row_isolation.get_tenant_connection(tenant_id)

    async def ensure_tenant_isolated(
        self,
        tenant_id: str,
    ) -> bool:
        """Ensure both isolation strategies are ready."""
        row_ready = await self._row_isolation.ensure_tenant_isolated(tenant_id)
        schema_ready = await self._schema_isolation.ensure_tenant_isolated(tenant_id)
        return row_ready and schema_ready

    async def validate_tenant_access(
        self,
        tenant_id: str,
        resource_tenant_id: str,
    ) -> bool:
        """Validate access across isolation strategies."""
        return tenant_id == resource_tenant_id


class IsolationManager:
    """
    Central isolation management.

    Coordinates tenant isolation across the application.
    """

    def __init__(
        self,
        default_level: IsolationLevel = IsolationLevel.ROW,
        config: Optional[IsolationConfig] = None,
    ):
        self.default_level = default_level
        self.config = config or IsolationConfig(level=default_level)
        self._strategies: Dict[IsolationLevel, IsolationStrategy] = {}
        self._tenant_overrides: Dict[str, IsolationLevel] = {}
        self._initialize_strategies()

    def _initialize_strategies(self) -> None:
        """Initialize isolation strategies."""
        self._strategies = {
            IsolationLevel.ROW: RowLevelIsolation(self.config),
            IsolationLevel.SCHEMA: SchemaIsolation(self.config),
            IsolationLevel.DATABASE: DatabaseIsolation(self.config),
            IsolationLevel.HYBRID: HybridIsolation(self.config),
        }

    def get_strategy(
        self,
        tenant_id: Optional[str] = None,
        level: Optional[IsolationLevel] = None,
    ) -> IsolationStrategy:
        """Get isolation strategy for tenant."""
        # Check for tenant-specific override
        if tenant_id and tenant_id in self._tenant_overrides:
            level = self._tenant_overrides[tenant_id]

        level = level or self.default_level
        return self._strategies.get(level, self._strategies[IsolationLevel.ROW])

    def set_tenant_override(
        self,
        tenant_id: str,
        level: IsolationLevel,
    ) -> None:
        """Set isolation level override for specific tenant."""
        self._tenant_overrides[tenant_id] = level
        logger.info(f"Set isolation override for {tenant_id}: {level}")

    async def provision_tenant(
        self,
        tenant_id: str,
        level: Optional[IsolationLevel] = None,
    ) -> bool:
        """Provision isolation for new tenant."""
        strategy = self.get_strategy(tenant_id, level)
        return await strategy.ensure_tenant_isolated(tenant_id)

    async def validate_access(
        self,
        tenant_id: str,
        resource_tenant_id: str,
        level: Optional[IsolationLevel] = None,
    ) -> bool:
        """Validate tenant access to resource."""
        strategy = self.get_strategy(tenant_id, level)
        return await strategy.validate_tenant_access(tenant_id, resource_tenant_id)

    @asynccontextmanager
    async def tenant_scope(self, tenant_id: str):
        """Context manager for tenant-scoped operations."""
        strategy = self.get_strategy(tenant_id)
        connection = await strategy.get_tenant_connection(tenant_id)

        try:
            yield connection
        finally:
            # Cleanup if needed
            pass


@dataclass
class TenantAwareModel:
    """
    Base for tenant-aware data models.

    Automatically includes tenant_id in operations.
    """
    tenant_id: Optional[str] = None

    @classmethod
    def with_tenant(cls: Type[T], tenant_id: str, **kwargs) -> T:
        """Create instance with tenant_id."""
        return cls(tenant_id=tenant_id, **kwargs)


class TenantAwareRepository:
    """
    Base repository with tenant isolation.

    Automatically applies tenant filters to all operations.
    """

    def __init__(
        self,
        isolation_manager: IsolationManager,
        table_name: str,
    ):
        self._isolation = isolation_manager
        self._table_name = table_name

    async def get_connection(self, tenant_id: str) -> Any:
        """Get tenant-scoped connection."""
        strategy = self._isolation.get_strategy(tenant_id)
        return await strategy.get_tenant_connection(tenant_id)

    async def filter_query(
        self,
        tenant_id: str,
        query: Any,
    ) -> Any:
        """Apply tenant filter to query."""
        strategy = self._isolation.get_strategy(tenant_id)
        return await strategy.apply_tenant_filter(query, tenant_id)

    async def validate_ownership(
        self,
        tenant_id: str,
        resource_tenant_id: str,
    ) -> bool:
        """Validate tenant owns the resource."""
        return await self._isolation.validate_access(tenant_id, resource_tenant_id)

    def require_tenant(self, tenant_id: Optional[str]) -> str:
        """Require tenant_id to be present."""
        if not tenant_id:
            raise ValueError("tenant_id is required for this operation")
        return tenant_id


class CrossTenantQueryGuard:
    """
    Guard against cross-tenant data access.

    Validates all queries for proper tenant isolation.
    """

    def __init__(self, isolation_manager: IsolationManager):
        self._isolation = isolation_manager
        self._violations: List[Dict[str, Any]] = []

    async def check_query(
        self,
        tenant_id: str,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check query for isolation violations."""
        # Check for missing tenant filter
        if "tenant_id" not in query.lower() and "search_path" not in query.lower():
            self._record_violation(tenant_id, query, "Missing tenant filter")
            return False

        # Check for dangerous patterns
        dangerous_patterns = [
            "OR 1=1",
            "tenant_id != ",
            "tenant_id <>",
            "ALL TENANTS",
            "CROSS JOIN",
        ]

        for pattern in dangerous_patterns:
            if pattern.lower() in query.lower():
                self._record_violation(tenant_id, query, f"Dangerous pattern: {pattern}")
                return False

        return True

    def _record_violation(
        self,
        tenant_id: str,
        query: str,
        reason: str,
    ) -> None:
        """Record isolation violation."""
        violation = {
            "tenant_id": tenant_id,
            "query": query[:200],
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._violations.append(violation)
        logger.warning(f"Isolation violation: {reason} for tenant {tenant_id}")

    def get_violations(self) -> List[Dict[str, Any]]:
        """Get recorded violations."""
        return self._violations.copy()

    def clear_violations(self) -> None:
        """Clear violation records."""
        self._violations.clear()
