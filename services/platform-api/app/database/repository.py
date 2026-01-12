"""
Repository Pattern Implementation

Enterprise repository pattern with:
- Generic CRUD operations
- Query specifications
- Unit of Work pattern
- Optimistic locking
- Soft deletes
- Audit trails
"""

from typing import (
    Optional, Dict, Any, List, Union, Tuple, TypeVar, Generic,
    Type, Callable, Awaitable, Protocol, runtime_checkable,
)
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import asyncio
import logging
import uuid

from sqlalchemy import select, update, delete, func, and_, or_, inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload, contains_eager

logger = logging.getLogger(__name__)

T = TypeVar("T")
ID = TypeVar("ID")


@runtime_checkable
class Entity(Protocol):
    """Protocol for entity classes."""
    id: Any


class Specification(Generic[T]):
    """
    Specification pattern for query composition.

    Usage:
        spec = (
            ActiveUsersSpec() &
            CreatedAfterSpec(date(2024, 1, 1)) |
            PremiumUsersSpec()
        )
        users = await repo.find_by_spec(spec)
    """

    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity satisfies the specification."""
        raise NotImplementedError

    def to_expression(self, model: Type[T]) -> Any:
        """Convert to SQLAlchemy expression."""
        raise NotImplementedError

    def __and__(self, other: "Specification[T]") -> "AndSpecification[T]":
        return AndSpecification(self, other)

    def __or__(self, other: "Specification[T]") -> "OrSpecification[T]":
        return OrSpecification(self, other)

    def __invert__(self) -> "NotSpecification[T]":
        return NotSpecification(self)


class AndSpecification(Specification[T]):
    """AND combination of specifications."""

    def __init__(self, left: Specification[T], right: Specification[T]):
        self.left = left
        self.right = right

    def is_satisfied_by(self, entity: T) -> bool:
        return self.left.is_satisfied_by(entity) and self.right.is_satisfied_by(entity)

    def to_expression(self, model: Type[T]) -> Any:
        return and_(
            self.left.to_expression(model),
            self.right.to_expression(model),
        )


class OrSpecification(Specification[T]):
    """OR combination of specifications."""

    def __init__(self, left: Specification[T], right: Specification[T]):
        self.left = left
        self.right = right

    def is_satisfied_by(self, entity: T) -> bool:
        return self.left.is_satisfied_by(entity) or self.right.is_satisfied_by(entity)

    def to_expression(self, model: Type[T]) -> Any:
        return or_(
            self.left.to_expression(model),
            self.right.to_expression(model),
        )


class NotSpecification(Specification[T]):
    """NOT specification."""

    def __init__(self, spec: Specification[T]):
        self.spec = spec

    def is_satisfied_by(self, entity: T) -> bool:
        return not self.spec.is_satisfied_by(entity)

    def to_expression(self, model: Type[T]) -> Any:
        return ~self.spec.to_expression(model)


@dataclass
class QueryOptions:
    """Options for repository queries."""
    # Pagination
    limit: Optional[int] = None
    offset: Optional[int] = None

    # Ordering
    order_by: Optional[str] = None
    order_desc: bool = False

    # Loading
    eager_load: List[str] = field(default_factory=list)
    select_columns: Optional[List[str]] = None

    # Filtering
    filters: Dict[str, Any] = field(default_factory=dict)

    # Locking
    for_update: bool = False
    skip_locked: bool = False

    # Soft delete
    include_deleted: bool = False


class Repository(ABC, Generic[T, ID]):
    """
    Abstract repository interface.

    Defines the contract for all repositories.
    """

    @abstractmethod
    async def get(self, id: ID) -> Optional[T]:
        """Get entity by ID."""
        pass

    @abstractmethod
    async def get_all(self, options: Optional[QueryOptions] = None) -> List[T]:
        """Get all entities with optional filtering."""
        pass

    @abstractmethod
    async def find_by(self, **criteria) -> List[T]:
        """Find entities by criteria."""
        pass

    @abstractmethod
    async def find_one_by(self, **criteria) -> Optional[T]:
        """Find single entity by criteria."""
        pass

    @abstractmethod
    async def add(self, entity: T) -> T:
        """Add a new entity."""
        pass

    @abstractmethod
    async def add_many(self, entities: List[T]) -> List[T]:
        """Add multiple entities."""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update an entity."""
        pass

    @abstractmethod
    async def delete(self, entity: T) -> None:
        """Delete an entity."""
        pass

    @abstractmethod
    async def delete_by_id(self, id: ID) -> bool:
        """Delete entity by ID."""
        pass

    @abstractmethod
    async def exists(self, id: ID) -> bool:
        """Check if entity exists."""
        pass

    @abstractmethod
    async def count(self, **criteria) -> int:
        """Count entities."""
        pass


class BaseRepository(Repository[T, ID]):
    """
    Base repository implementation with SQLAlchemy.

    Provides common CRUD operations for all entities.
    """

    def __init__(
        self,
        session: AsyncSession,
        model: Type[T],
        id_field: str = "id",
    ):
        self._session = session
        self._model = model
        self._id_field = id_field

    @property
    def session(self) -> AsyncSession:
        """Get the current session."""
        return self._session

    async def get(self, id: ID) -> Optional[T]:
        """Get entity by ID."""
        return await self._session.get(self._model, id)

    async def get_or_fail(self, id: ID) -> T:
        """Get entity by ID or raise exception."""
        entity = await self.get(id)
        if entity is None:
            raise ValueError(f"{self._model.__name__} with id {id} not found")
        return entity

    async def get_all(self, options: Optional[QueryOptions] = None) -> List[T]:
        """Get all entities with optional filtering."""
        options = options or QueryOptions()
        query = select(self._model)

        # Apply filters
        for field, value in options.filters.items():
            if hasattr(self._model, field):
                column = getattr(self._model, field)
                if isinstance(value, (list, tuple)):
                    query = query.where(column.in_(value))
                else:
                    query = query.where(column == value)

        # Apply soft delete filter
        if not options.include_deleted and hasattr(self._model, "deleted_at"):
            query = query.where(getattr(self._model, "deleted_at").is_(None))

        # Apply ordering
        if options.order_by and hasattr(self._model, options.order_by):
            column = getattr(self._model, options.order_by)
            if options.order_desc:
                column = column.desc()
            query = query.order_by(column)

        # Apply eager loading
        for relation in options.eager_load:
            if hasattr(self._model, relation):
                query = query.options(selectinload(getattr(self._model, relation)))

        # Apply pagination
        if options.limit:
            query = query.limit(options.limit)
        if options.offset:
            query = query.offset(options.offset)

        # Apply locking
        if options.for_update:
            query = query.with_for_update(skip_locked=options.skip_locked)

        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def find_by(self, **criteria) -> List[T]:
        """Find entities by criteria."""
        query = select(self._model)

        for field, value in criteria.items():
            if hasattr(self._model, field):
                column = getattr(self._model, field)
                if isinstance(value, (list, tuple)):
                    query = query.where(column.in_(value))
                elif value is None:
                    query = query.where(column.is_(None))
                else:
                    query = query.where(column == value)

        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def find_one_by(self, **criteria) -> Optional[T]:
        """Find single entity by criteria."""
        entities = await self.find_by(**criteria)
        return entities[0] if entities else None

    async def find_by_spec(self, spec: Specification[T]) -> List[T]:
        """Find entities by specification."""
        query = select(self._model).where(spec.to_expression(self._model))
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def add(self, entity: T) -> T:
        """Add a new entity."""
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return entity

    async def add_many(self, entities: List[T]) -> List[T]:
        """Add multiple entities."""
        self._session.add_all(entities)
        await self._session.flush()
        for entity in entities:
            await self._session.refresh(entity)
        return entities

    async def update(self, entity: T) -> T:
        """Update an entity."""
        merged = await self._session.merge(entity)
        await self._session.flush()
        return merged

    async def update_by_id(self, id: ID, **fields) -> Optional[T]:
        """Update entity by ID with specific fields."""
        entity = await self.get(id)
        if entity is None:
            return None

        for field, value in fields.items():
            if hasattr(entity, field):
                setattr(entity, field, value)

        return await self.update(entity)

    async def delete(self, entity: T) -> None:
        """Delete an entity (hard delete)."""
        await self._session.delete(entity)
        await self._session.flush()

    async def soft_delete(self, entity: T) -> T:
        """Soft delete an entity."""
        if hasattr(entity, "deleted_at"):
            setattr(entity, "deleted_at", datetime.utcnow())
            return await self.update(entity)
        raise ValueError(f"{self._model.__name__} does not support soft delete")

    async def restore(self, entity: T) -> T:
        """Restore a soft-deleted entity."""
        if hasattr(entity, "deleted_at"):
            setattr(entity, "deleted_at", None)
            return await self.update(entity)
        raise ValueError(f"{self._model.__name__} does not support soft delete")

    async def delete_by_id(self, id: ID) -> bool:
        """Delete entity by ID."""
        entity = await self.get(id)
        if entity is None:
            return False
        await self.delete(entity)
        return True

    async def exists(self, id: ID) -> bool:
        """Check if entity exists."""
        query = select(func.count()).select_from(self._model).where(
            getattr(self._model, self._id_field) == id
        )
        result = await self._session.execute(query)
        return result.scalar() > 0

    async def count(self, **criteria) -> int:
        """Count entities."""
        query = select(func.count()).select_from(self._model)

        for field, value in criteria.items():
            if hasattr(self._model, field):
                column = getattr(self._model, field)
                query = query.where(column == value)

        result = await self._session.execute(query)
        return result.scalar() or 0


class CRUDRepository(BaseRepository[T, ID]):
    """
    Extended CRUD repository with additional features.

    Includes:
    - Bulk operations
    - Optimistic locking
    - Audit trails
    - Full-text search
    """

    def __init__(
        self,
        session: AsyncSession,
        model: Type[T],
        id_field: str = "id",
        version_field: Optional[str] = None,
    ):
        super().__init__(session, model, id_field)
        self._version_field = version_field

    async def bulk_create(self, entities: List[T], batch_size: int = 100) -> List[T]:
        """Create entities in batches."""
        results = []
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            created = await self.add_many(batch)
            results.extend(created)
        return results

    async def bulk_update(
        self,
        criteria: Dict[str, Any],
        values: Dict[str, Any],
    ) -> int:
        """Update multiple entities matching criteria."""
        query = update(self._model).values(**values)

        for field, value in criteria.items():
            if hasattr(self._model, field):
                query = query.where(getattr(self._model, field) == value)

        result = await self._session.execute(query)
        await self._session.flush()
        return result.rowcount

    async def bulk_delete(self, criteria: Dict[str, Any]) -> int:
        """Delete multiple entities matching criteria."""
        query = delete(self._model)

        for field, value in criteria.items():
            if hasattr(self._model, field):
                query = query.where(getattr(self._model, field) == value)

        result = await self._session.execute(query)
        await self._session.flush()
        return result.rowcount

    async def update_with_optimistic_lock(self, entity: T) -> T:
        """Update with optimistic locking."""
        if not self._version_field:
            return await self.update(entity)

        current_version = getattr(entity, self._version_field)
        new_version = current_version + 1

        # Build update query with version check
        id_value = getattr(entity, self._id_field)
        query = (
            update(self._model)
            .where(getattr(self._model, self._id_field) == id_value)
            .where(getattr(self._model, self._version_field) == current_version)
            .values(**{self._version_field: new_version})
        )

        # Add other updated fields
        mapper = inspect(self._model)
        for column in mapper.columns:
            if column.key not in (self._id_field, self._version_field):
                value = getattr(entity, column.key, None)
                if value is not None:
                    query = query.values(**{column.key: value})

        result = await self._session.execute(query)
        await self._session.flush()

        if result.rowcount == 0:
            raise OptimisticLockError(
                f"Entity {self._model.__name__} with id {id_value} was modified by another transaction"
            )

        setattr(entity, self._version_field, new_version)
        return entity

    async def find_with_pagination(
        self,
        page: int = 1,
        per_page: int = 20,
        order_by: Optional[str] = None,
        order_desc: bool = False,
        **criteria,
    ) -> Tuple[List[T], int]:
        """Find with pagination, returns (entities, total_count)."""
        # Get total count
        total = await self.count(**criteria)

        # Get paginated results
        options = QueryOptions(
            limit=per_page,
            offset=(page - 1) * per_page,
            order_by=order_by,
            order_desc=order_desc,
            filters=criteria,
        )
        entities = await self.get_all(options)

        return entities, total

    async def search(
        self,
        query_string: str,
        search_fields: List[str],
        limit: int = 20,
    ) -> List[T]:
        """Full-text search across multiple fields."""
        conditions = []
        for field in search_fields:
            if hasattr(self._model, field):
                column = getattr(self._model, field)
                conditions.append(column.ilike(f"%{query_string}%"))

        if not conditions:
            return []

        query = select(self._model).where(or_(*conditions)).limit(limit)
        result = await self._session.execute(query)
        return list(result.scalars().all())


class OptimisticLockError(Exception):
    """Raised when optimistic locking fails."""
    pass


class UnitOfWork:
    """
    Unit of Work pattern implementation.

    Manages transactions across multiple repositories.

    Usage:
        async with UnitOfWork(session_factory) as uow:
            user = await uow.users.add(User(...))
            profile = await uow.profiles.add(Profile(user_id=user.id, ...))
            await uow.commit()
    """

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
    ):
        self._session_factory = session_factory
        self._session: Optional[AsyncSession] = None
        self._repositories: Dict[str, Repository] = {}

    async def __aenter__(self) -> "UnitOfWork":
        self._session = self._session_factory()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.rollback()
        await self._session.close()

    def register_repository(
        self,
        name: str,
        repository_class: Type[Repository],
        model: Type,
        **kwargs,
    ) -> None:
        """Register a repository."""
        self._repositories[name] = (repository_class, model, kwargs)

    def __getattr__(self, name: str) -> Repository:
        """Get a registered repository."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        if name not in self._repositories:
            raise AttributeError(f"Repository '{name}' not registered")

        repo_class, model, kwargs = self._repositories[name]
        return repo_class(self._session, model, **kwargs)

    async def commit(self) -> None:
        """Commit the transaction."""
        await self._session.commit()

    async def rollback(self) -> None:
        """Rollback the transaction."""
        await self._session.rollback()

    async def flush(self) -> None:
        """Flush pending changes."""
        await self._session.flush()

    @asynccontextmanager
    async def transaction(self):
        """Create a nested transaction (savepoint)."""
        async with self._session.begin_nested():
            yield


class TransactionManager:
    """
    Transaction manager for complex operations.

    Provides:
    - Automatic retry on transient failures
    - Distributed transaction support
    - Transaction hooks
    """

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ):
        self._session_factory = session_factory
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._before_commit_hooks: List[Callable] = []
        self._after_commit_hooks: List[Callable] = []
        self._after_rollback_hooks: List[Callable] = []

    def before_commit(self, hook: Callable) -> None:
        """Register a before-commit hook."""
        self._before_commit_hooks.append(hook)

    def after_commit(self, hook: Callable) -> None:
        """Register an after-commit hook."""
        self._after_commit_hooks.append(hook)

    def after_rollback(self, hook: Callable) -> None:
        """Register an after-rollback hook."""
        self._after_rollback_hooks.append(hook)

    @asynccontextmanager
    async def transaction(self):
        """Execute a transaction with retry support."""
        last_error = None

        for attempt in range(self._max_retries):
            session = self._session_factory()
            try:
                yield session

                # Run before-commit hooks
                for hook in self._before_commit_hooks:
                    if asyncio.iscoroutinefunction(hook):
                        await hook(session)
                    else:
                        hook(session)

                await session.commit()

                # Run after-commit hooks
                for hook in self._after_commit_hooks:
                    try:
                        if asyncio.iscoroutinefunction(hook):
                            await hook(session)
                        else:
                            hook(session)
                    except Exception as e:
                        logger.error(f"After-commit hook error: {e}")

                return

            except Exception as e:
                await session.rollback()
                last_error = e

                # Run after-rollback hooks
                for hook in self._after_rollback_hooks:
                    try:
                        if asyncio.iscoroutinefunction(hook):
                            await hook(session, e)
                        else:
                            hook(session, e)
                    except Exception as hook_error:
                        logger.error(f"After-rollback hook error: {hook_error}")

                # Check if error is retryable
                if self._is_retryable_error(e) and attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                    continue

                raise

            finally:
                await session.close()

        if last_error:
            raise last_error

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable."""
        retryable_errors = (
            "deadlock detected",
            "could not serialize access",
            "connection reset",
            "connection refused",
        )
        error_str = str(error).lower()
        return any(msg in error_str for msg in retryable_errors)


# Common specifications
class IsActiveSpec(Specification[T]):
    """Specification for active entities."""

    def is_satisfied_by(self, entity: T) -> bool:
        return getattr(entity, "is_active", True)

    def to_expression(self, model: Type[T]) -> Any:
        if hasattr(model, "is_active"):
            return getattr(model, "is_active") == True
        return True


class IsNotDeletedSpec(Specification[T]):
    """Specification for non-deleted entities."""

    def is_satisfied_by(self, entity: T) -> bool:
        return getattr(entity, "deleted_at", None) is None

    def to_expression(self, model: Type[T]) -> Any:
        if hasattr(model, "deleted_at"):
            return getattr(model, "deleted_at").is_(None)
        return True


class CreatedAfterSpec(Specification[T]):
    """Specification for entities created after a date."""

    def __init__(self, date: datetime):
        self.date = date

    def is_satisfied_by(self, entity: T) -> bool:
        created_at = getattr(entity, "created_at", None)
        return created_at is not None and created_at > self.date

    def to_expression(self, model: Type[T]) -> Any:
        if hasattr(model, "created_at"):
            return getattr(model, "created_at") > self.date
        return True


class BelongsToUserSpec(Specification[T]):
    """Specification for entities belonging to a user."""

    def __init__(self, user_id: Any, field: str = "user_id"):
        self.user_id = user_id
        self.field = field

    def is_satisfied_by(self, entity: T) -> bool:
        return getattr(entity, self.field, None) == self.user_id

    def to_expression(self, model: Type[T]) -> Any:
        if hasattr(model, self.field):
            return getattr(model, self.field) == self.user_id
        return True
