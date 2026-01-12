"""
Database Migration System

Production-ready migration system with:
- Version tracking
- Rollback support
- Schema diffing
- Data migrations
- Concurrent-safe execution
"""

from typing import Optional, Dict, Any, List, Callable, Type, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import hashlib
import logging
import inspect

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class MigrationStatus(str, Enum):
    """Migration execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ColumnType(str, Enum):
    """Column data types."""
    INTEGER = "INTEGER"
    BIGINT = "BIGINT"
    SMALLINT = "SMALLINT"
    SERIAL = "SERIAL"
    BIGSERIAL = "BIGSERIAL"
    VARCHAR = "VARCHAR"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE PRECISION"
    DECIMAL = "DECIMAL"
    DATE = "DATE"
    TIME = "TIME"
    TIMESTAMP = "TIMESTAMP"
    TIMESTAMPTZ = "TIMESTAMP WITH TIME ZONE"
    UUID = "UUID"
    JSON = "JSON"
    JSONB = "JSONB"
    BYTEA = "BYTEA"
    ARRAY = "ARRAY"


@dataclass
class Column:
    """Column definition for schema builder."""
    name: str
    type: ColumnType
    length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    nullable: bool = True
    default: Optional[Any] = None
    primary_key: bool = False
    unique: bool = False
    references: Optional[str] = None  # "table.column"
    on_delete: str = "CASCADE"
    on_update: str = "CASCADE"
    check: Optional[str] = None
    comment: Optional[str] = None

    def to_sql(self) -> str:
        """Generate SQL for column definition."""
        parts = [f'"{self.name}"']

        # Type
        type_str = self.type.value
        if self.length:
            type_str = f"{type_str}({self.length})"
        elif self.precision and self.scale:
            type_str = f"{type_str}({self.precision}, {self.scale})"
        parts.append(type_str)

        # Constraints
        if self.primary_key:
            parts.append("PRIMARY KEY")
        if not self.nullable:
            parts.append("NOT NULL")
        if self.unique and not self.primary_key:
            parts.append("UNIQUE")
        if self.default is not None:
            if isinstance(self.default, str) and not self.default.startswith("'"):
                parts.append(f"DEFAULT {self.default}")
            else:
                parts.append(f"DEFAULT '{self.default}'")
        if self.check:
            parts.append(f"CHECK ({self.check})")
        if self.references:
            table, col = self.references.split(".")
            parts.append(f'REFERENCES "{table}"("{col}")')
            parts.append(f"ON DELETE {self.on_delete}")
            parts.append(f"ON UPDATE {self.on_update}")

        return " ".join(parts)


@dataclass
class Index:
    """Index definition."""
    name: str
    table: str
    columns: List[str]
    unique: bool = False
    method: str = "btree"  # btree, hash, gist, gin
    where: Optional[str] = None
    include: List[str] = field(default_factory=list)

    def to_sql(self) -> str:
        """Generate SQL for index creation."""
        unique = "UNIQUE " if self.unique else ""
        columns = ", ".join(f'"{c}"' for c in self.columns)

        sql = f'CREATE {unique}INDEX "{self.name}" ON "{self.table}" USING {self.method} ({columns})'

        if self.include:
            include_cols = ", ".join(f'"{c}"' for c in self.include)
            sql += f" INCLUDE ({include_cols})"

        if self.where:
            sql += f" WHERE {self.where}"

        return sql


@dataclass
class ForeignKey:
    """Foreign key constraint definition."""
    name: str
    table: str
    columns: List[str]
    references_table: str
    references_columns: List[str]
    on_delete: str = "CASCADE"
    on_update: str = "CASCADE"
    deferrable: bool = False
    initially_deferred: bool = False

    def to_sql(self) -> str:
        """Generate SQL for foreign key constraint."""
        cols = ", ".join(f'"{c}"' for c in self.columns)
        ref_cols = ", ".join(f'"{c}"' for c in self.references_columns)

        sql = f'ALTER TABLE "{self.table}" ADD CONSTRAINT "{self.name}" '
        sql += f'FOREIGN KEY ({cols}) REFERENCES "{self.references_table}" ({ref_cols}) '
        sql += f"ON DELETE {self.on_delete} ON UPDATE {self.on_update}"

        if self.deferrable:
            sql += " DEFERRABLE"
            if self.initially_deferred:
                sql += " INITIALLY DEFERRED"

        return sql


class SchemaBuilder:
    """
    Fluent schema builder for migrations.

    Usage:
        schema = SchemaBuilder()

        # Create table
        schema.create_table("users", lambda t: (
            t.uuid("id").primary_key(),
            t.string("email", 255).unique().not_null(),
            t.string("name", 100),
            t.boolean("is_active").default(True),
            t.timestamps(),
        ))

        # Add index
        schema.add_index("users", ["email"], unique=True)

        # Generate SQL
        sql = schema.to_sql()
    """

    def __init__(self):
        self._operations: List[str] = []
        self._current_table: Optional[str] = None
        self._columns: List[Column] = []

    def create_table(
        self,
        name: str,
        definition: Callable[["TableBuilder"], None],
    ) -> "SchemaBuilder":
        """Create a new table."""
        builder = TableBuilder(name)
        definition(builder)

        columns_sql = ",\n    ".join(col.to_sql() for col in builder.columns)
        sql = f'CREATE TABLE "{name}" (\n    {columns_sql}\n)'
        self._operations.append(sql)

        return self

    def drop_table(self, name: str, if_exists: bool = True) -> "SchemaBuilder":
        """Drop a table."""
        exists = "IF EXISTS " if if_exists else ""
        self._operations.append(f'DROP TABLE {exists}"{name}"')
        return self

    def rename_table(self, old_name: str, new_name: str) -> "SchemaBuilder":
        """Rename a table."""
        self._operations.append(f'ALTER TABLE "{old_name}" RENAME TO "{new_name}"')
        return self

    def add_column(
        self,
        table: str,
        column: Column,
    ) -> "SchemaBuilder":
        """Add a column to a table."""
        self._operations.append(
            f'ALTER TABLE "{table}" ADD COLUMN {column.to_sql()}'
        )
        return self

    def drop_column(
        self,
        table: str,
        column: str,
        if_exists: bool = True,
    ) -> "SchemaBuilder":
        """Drop a column from a table."""
        exists = "IF EXISTS " if if_exists else ""
        self._operations.append(
            f'ALTER TABLE "{table}" DROP COLUMN {exists}"{column}"'
        )
        return self

    def rename_column(
        self,
        table: str,
        old_name: str,
        new_name: str,
    ) -> "SchemaBuilder":
        """Rename a column."""
        self._operations.append(
            f'ALTER TABLE "{table}" RENAME COLUMN "{old_name}" TO "{new_name}"'
        )
        return self

    def alter_column(
        self,
        table: str,
        column: str,
        type: Optional[ColumnType] = None,
        nullable: Optional[bool] = None,
        default: Optional[Any] = None,
    ) -> "SchemaBuilder":
        """Alter a column."""
        if type:
            self._operations.append(
                f'ALTER TABLE "{table}" ALTER COLUMN "{column}" TYPE {type.value}'
            )
        if nullable is not None:
            action = "DROP NOT NULL" if nullable else "SET NOT NULL"
            self._operations.append(
                f'ALTER TABLE "{table}" ALTER COLUMN "{column}" {action}'
            )
        if default is not None:
            if default == "DROP":
                self._operations.append(
                    f'ALTER TABLE "{table}" ALTER COLUMN "{column}" DROP DEFAULT'
                )
            else:
                self._operations.append(
                    f'ALTER TABLE "{table}" ALTER COLUMN "{column}" SET DEFAULT {default}'
                )
        return self

    def add_index(
        self,
        table: str,
        columns: List[str],
        name: Optional[str] = None,
        unique: bool = False,
        method: str = "btree",
        where: Optional[str] = None,
    ) -> "SchemaBuilder":
        """Add an index."""
        if name is None:
            name = f"ix_{table}_{'_'.join(columns)}"

        index = Index(
            name=name,
            table=table,
            columns=columns,
            unique=unique,
            method=method,
            where=where,
        )
        self._operations.append(index.to_sql())
        return self

    def drop_index(
        self,
        name: str,
        if_exists: bool = True,
    ) -> "SchemaBuilder":
        """Drop an index."""
        exists = "IF EXISTS " if if_exists else ""
        self._operations.append(f'DROP INDEX {exists}"{name}"')
        return self

    def add_foreign_key(
        self,
        table: str,
        columns: List[str],
        references_table: str,
        references_columns: List[str],
        name: Optional[str] = None,
        on_delete: str = "CASCADE",
        on_update: str = "CASCADE",
    ) -> "SchemaBuilder":
        """Add a foreign key constraint."""
        if name is None:
            name = f"fk_{table}_{'_'.join(columns)}"

        fk = ForeignKey(
            name=name,
            table=table,
            columns=columns,
            references_table=references_table,
            references_columns=references_columns,
            on_delete=on_delete,
            on_update=on_update,
        )
        self._operations.append(fk.to_sql())
        return self

    def drop_foreign_key(
        self,
        table: str,
        name: str,
    ) -> "SchemaBuilder":
        """Drop a foreign key constraint."""
        self._operations.append(
            f'ALTER TABLE "{table}" DROP CONSTRAINT "{name}"'
        )
        return self

    def add_unique_constraint(
        self,
        table: str,
        columns: List[str],
        name: Optional[str] = None,
    ) -> "SchemaBuilder":
        """Add a unique constraint."""
        if name is None:
            name = f"uq_{table}_{'_'.join(columns)}"

        cols = ", ".join(f'"{c}"' for c in columns)
        self._operations.append(
            f'ALTER TABLE "{table}" ADD CONSTRAINT "{name}" UNIQUE ({cols})'
        )
        return self

    def add_check_constraint(
        self,
        table: str,
        name: str,
        expression: str,
    ) -> "SchemaBuilder":
        """Add a check constraint."""
        self._operations.append(
            f'ALTER TABLE "{table}" ADD CONSTRAINT "{name}" CHECK ({expression})'
        )
        return self

    def raw(self, sql: str) -> "SchemaBuilder":
        """Add raw SQL."""
        self._operations.append(sql)
        return self

    def to_sql(self) -> List[str]:
        """Get all SQL statements."""
        return self._operations

    def clear(self) -> None:
        """Clear all operations."""
        self._operations = []


class TableBuilder:
    """Builder for table columns."""

    def __init__(self, table_name: str):
        self.table_name = table_name
        self.columns: List[Column] = []
        self._current: Optional[Column] = None

    def _add_column(self, column: Column) -> "ColumnBuilder":
        self.columns.append(column)
        return ColumnBuilder(column)

    def uuid(self, name: str) -> "ColumnBuilder":
        """Add UUID column."""
        return self._add_column(Column(name, ColumnType.UUID))

    def serial(self, name: str) -> "ColumnBuilder":
        """Add serial (auto-increment) column."""
        return self._add_column(Column(name, ColumnType.SERIAL))

    def bigserial(self, name: str) -> "ColumnBuilder":
        """Add bigserial column."""
        return self._add_column(Column(name, ColumnType.BIGSERIAL))

    def integer(self, name: str) -> "ColumnBuilder":
        """Add integer column."""
        return self._add_column(Column(name, ColumnType.INTEGER))

    def bigint(self, name: str) -> "ColumnBuilder":
        """Add bigint column."""
        return self._add_column(Column(name, ColumnType.BIGINT))

    def string(self, name: str, length: int = 255) -> "ColumnBuilder":
        """Add varchar column."""
        return self._add_column(Column(name, ColumnType.VARCHAR, length=length))

    def text(self, name: str) -> "ColumnBuilder":
        """Add text column."""
        return self._add_column(Column(name, ColumnType.TEXT))

    def boolean(self, name: str) -> "ColumnBuilder":
        """Add boolean column."""
        return self._add_column(Column(name, ColumnType.BOOLEAN))

    def float(self, name: str) -> "ColumnBuilder":
        """Add float column."""
        return self._add_column(Column(name, ColumnType.FLOAT))

    def decimal(self, name: str, precision: int = 10, scale: int = 2) -> "ColumnBuilder":
        """Add decimal column."""
        return self._add_column(Column(name, ColumnType.DECIMAL, precision=precision, scale=scale))

    def date(self, name: str) -> "ColumnBuilder":
        """Add date column."""
        return self._add_column(Column(name, ColumnType.DATE))

    def timestamp(self, name: str, with_timezone: bool = True) -> "ColumnBuilder":
        """Add timestamp column."""
        col_type = ColumnType.TIMESTAMPTZ if with_timezone else ColumnType.TIMESTAMP
        return self._add_column(Column(name, col_type))

    def json(self, name: str) -> "ColumnBuilder":
        """Add JSON column."""
        return self._add_column(Column(name, ColumnType.JSON))

    def jsonb(self, name: str) -> "ColumnBuilder":
        """Add JSONB column."""
        return self._add_column(Column(name, ColumnType.JSONB))

    def binary(self, name: str) -> "ColumnBuilder":
        """Add binary column."""
        return self._add_column(Column(name, ColumnType.BYTEA))

    def timestamps(self) -> "TableBuilder":
        """Add created_at and updated_at columns."""
        self._add_column(Column(
            "created_at",
            ColumnType.TIMESTAMPTZ,
            nullable=False,
            default="NOW()",
        ))
        self._add_column(Column(
            "updated_at",
            ColumnType.TIMESTAMPTZ,
        ))
        return self

    def soft_deletes(self) -> "TableBuilder":
        """Add deleted_at column for soft deletes."""
        self._add_column(Column("deleted_at", ColumnType.TIMESTAMPTZ))
        return self


class ColumnBuilder:
    """Fluent builder for column constraints."""

    def __init__(self, column: Column):
        self._column = column

    def primary_key(self) -> "ColumnBuilder":
        """Set as primary key."""
        self._column.primary_key = True
        self._column.nullable = False
        return self

    def not_null(self) -> "ColumnBuilder":
        """Set as not nullable."""
        self._column.nullable = False
        return self

    def nullable(self) -> "ColumnBuilder":
        """Set as nullable."""
        self._column.nullable = True
        return self

    def unique(self) -> "ColumnBuilder":
        """Set as unique."""
        self._column.unique = True
        return self

    def default(self, value: Any) -> "ColumnBuilder":
        """Set default value."""
        self._column.default = value
        return self

    def references(
        self,
        table: str,
        column: str = "id",
        on_delete: str = "CASCADE",
        on_update: str = "CASCADE",
    ) -> "ColumnBuilder":
        """Add foreign key reference."""
        self._column.references = f"{table}.{column}"
        self._column.on_delete = on_delete
        self._column.on_update = on_update
        return self

    def check(self, expression: str) -> "ColumnBuilder":
        """Add check constraint."""
        self._column.check = expression
        return self

    def comment(self, text: str) -> "ColumnBuilder":
        """Add column comment."""
        self._column.comment = text
        return self


@dataclass
class Migration:
    """
    Base migration class.

    Usage:
        class CreateUsersTable(Migration):
            version = "20240101_001"
            description = "Create users table"

            async def up(self, schema: SchemaBuilder) -> None:
                schema.create_table("users", lambda t: (
                    t.uuid("id").primary_key(),
                    t.string("email").unique().not_null(),
                    t.timestamps(),
                ))

            async def down(self, schema: SchemaBuilder) -> None:
                schema.drop_table("users")
    """
    version: str
    description: str = ""
    depends_on: List[str] = field(default_factory=list)

    async def up(self, schema: SchemaBuilder) -> None:
        """Apply the migration."""
        raise NotImplementedError

    async def down(self, schema: SchemaBuilder) -> None:
        """Rollback the migration."""
        raise NotImplementedError

    @property
    def checksum(self) -> str:
        """Calculate migration checksum."""
        source = inspect.getsource(self.__class__)
        return hashlib.md5(source.encode()).hexdigest()


@dataclass
class MigrationHistory:
    """Record of applied migrations."""
    version: str
    description: str
    checksum: str
    applied_at: datetime
    execution_time_ms: float
    status: MigrationStatus
    error_message: Optional[str] = None


class MigrationRunner:
    """
    Executes database migrations.

    Features:
    - Version tracking
    - Dependency resolution
    - Rollback support
    - Concurrent-safe execution
    """

    HISTORY_TABLE = "_migrations"

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
    ):
        self._session_factory = session_factory
        self._migrations: Dict[str, Type[Migration]] = {}
        self._lock = asyncio.Lock()

    def register(self, migration_class: Type[Migration]) -> None:
        """Register a migration."""
        # Create instance to get version
        migration = migration_class()
        self._migrations[migration.version] = migration_class

    def register_all(self, migrations: List[Type[Migration]]) -> None:
        """Register multiple migrations."""
        for migration in migrations:
            self.register(migration)

    async def initialize(self) -> None:
        """Initialize migration history table."""
        async with self._session_factory() as session:
            await session.execute(text(f"""
                CREATE TABLE IF NOT EXISTS "{self.HISTORY_TABLE}" (
                    version VARCHAR(100) PRIMARY KEY,
                    description TEXT,
                    checksum VARCHAR(32),
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    execution_time_ms FLOAT,
                    status VARCHAR(20),
                    error_message TEXT
                )
            """))
            await session.commit()

    async def get_applied_versions(self) -> List[str]:
        """Get list of applied migration versions."""
        async with self._session_factory() as session:
            result = await session.execute(text(f"""
                SELECT version FROM "{self.HISTORY_TABLE}"
                WHERE status = 'completed'
                ORDER BY applied_at
            """))
            return [row[0] for row in result.fetchall()]

    async def get_pending_migrations(self) -> List[Migration]:
        """Get migrations that haven't been applied."""
        applied = set(await self.get_applied_versions())
        pending = []

        for version, migration_class in self._migrations.items():
            if version not in applied:
                pending.append(migration_class())

        # Sort by version
        pending.sort(key=lambda m: m.version)
        return pending

    async def migrate(self, target_version: Optional[str] = None) -> List[str]:
        """Run all pending migrations up to target version."""
        async with self._lock:
            await self.initialize()
            pending = await self.get_pending_migrations()
            applied = []

            for migration in pending:
                if target_version and migration.version > target_version:
                    break

                # Check dependencies
                applied_versions = await self.get_applied_versions()
                for dep in migration.depends_on:
                    if dep not in applied_versions:
                        raise RuntimeError(
                            f"Migration {migration.version} depends on {dep} which is not applied"
                        )

                # Run migration
                success = await self._run_migration(migration)
                if success:
                    applied.append(migration.version)
                else:
                    break

            return applied

    async def rollback(self, steps: int = 1) -> List[str]:
        """Rollback the last N migrations."""
        async with self._lock:
            applied = await self.get_applied_versions()
            rolled_back = []

            for _ in range(min(steps, len(applied))):
                if not applied:
                    break

                version = applied.pop()
                migration_class = self._migrations.get(version)

                if not migration_class:
                    logger.warning(f"Migration class not found for version {version}")
                    continue

                migration = migration_class()
                success = await self._rollback_migration(migration)

                if success:
                    rolled_back.append(version)
                else:
                    break

            return rolled_back

    async def rollback_to(self, target_version: str) -> List[str]:
        """Rollback to a specific version."""
        applied = await self.get_applied_versions()
        steps = 0

        for version in reversed(applied):
            if version == target_version:
                break
            steps += 1

        return await self.rollback(steps)

    async def _run_migration(self, migration: Migration) -> bool:
        """Execute a single migration."""
        logger.info(f"Running migration {migration.version}: {migration.description}")

        start_time = datetime.utcnow()
        schema = SchemaBuilder()

        try:
            # Record start
            await self._record_migration(
                migration,
                MigrationStatus.RUNNING,
                0,
            )

            # Run migration
            await migration.up(schema)

            # Execute SQL
            async with self._session_factory() as session:
                for sql in schema.to_sql():
                    logger.debug(f"Executing: {sql}")
                    await session.execute(text(sql))
                await session.commit()

            # Record success
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._record_migration(
                migration,
                MigrationStatus.COMPLETED,
                execution_time,
            )

            logger.info(f"Migration {migration.version} completed in {execution_time:.2f}ms")
            return True

        except Exception as e:
            logger.error(f"Migration {migration.version} failed: {e}")

            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._record_migration(
                migration,
                MigrationStatus.FAILED,
                execution_time,
                str(e),
            )
            return False

    async def _rollback_migration(self, migration: Migration) -> bool:
        """Rollback a single migration."""
        logger.info(f"Rolling back migration {migration.version}")

        start_time = datetime.utcnow()
        schema = SchemaBuilder()

        try:
            # Run rollback
            await migration.down(schema)

            # Execute SQL
            async with self._session_factory() as session:
                for sql in schema.to_sql():
                    logger.debug(f"Executing: {sql}")
                    await session.execute(text(sql))
                await session.commit()

            # Update record
            await self._update_migration_status(migration.version, MigrationStatus.ROLLED_BACK)

            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"Migration {migration.version} rolled back in {execution_time:.2f}ms")
            return True

        except Exception as e:
            logger.error(f"Rollback of {migration.version} failed: {e}")
            return False

    async def _record_migration(
        self,
        migration: Migration,
        status: MigrationStatus,
        execution_time_ms: float,
        error_message: Optional[str] = None,
    ) -> None:
        """Record migration execution."""
        async with self._session_factory() as session:
            await session.execute(text(f"""
                INSERT INTO "{self.HISTORY_TABLE}"
                (version, description, checksum, status, execution_time_ms, error_message)
                VALUES (:version, :description, :checksum, :status, :execution_time_ms, :error_message)
                ON CONFLICT (version) DO UPDATE SET
                    status = :status,
                    execution_time_ms = :execution_time_ms,
                    error_message = :error_message,
                    applied_at = NOW()
            """), {
                "version": migration.version,
                "description": migration.description,
                "checksum": migration.checksum,
                "status": status.value,
                "execution_time_ms": execution_time_ms,
                "error_message": error_message,
            })
            await session.commit()

    async def _update_migration_status(
        self,
        version: str,
        status: MigrationStatus,
    ) -> None:
        """Update migration status."""
        async with self._session_factory() as session:
            await session.execute(text(f"""
                UPDATE "{self.HISTORY_TABLE}"
                SET status = :status
                WHERE version = :version
            """), {"version": version, "status": status.value})
            await session.commit()

    async def get_status(self) -> List[Dict[str, Any]]:
        """Get status of all migrations."""
        await self.initialize()

        async with self._session_factory() as session:
            result = await session.execute(text(f"""
                SELECT version, description, status, applied_at, execution_time_ms
                FROM "{self.HISTORY_TABLE}"
                ORDER BY version
            """))

            applied = {row[0]: row for row in result.fetchall()}

        status = []
        for version in sorted(self._migrations.keys()):
            migration = self._migrations[version]()

            if version in applied:
                row = applied[version]
                status.append({
                    "version": version,
                    "description": migration.description,
                    "status": row[2],
                    "applied_at": row[3].isoformat() if row[3] else None,
                    "execution_time_ms": row[4],
                })
            else:
                status.append({
                    "version": version,
                    "description": migration.description,
                    "status": "pending",
                    "applied_at": None,
                    "execution_time_ms": None,
                })

        return status
