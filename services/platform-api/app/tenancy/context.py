"""
Tenant Context Management

Context propagation for multi-tenant applications:
- Thread-safe context variables
- Async-safe context handling
- Context propagation across tasks
- Request-scoped tenant context
"""

from typing import Optional, Dict, Any, TypeVar, Generic, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextvars import ContextVar, Token
from contextlib import contextmanager, asynccontextmanager
import asyncio
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Context variables for tenant information
_tenant_context: ContextVar[Optional["TenantContext"]] = ContextVar(
    "tenant_context", default=None
)


@dataclass
class TenantContext:
    """
    Tenant context holding tenant information.

    Propagated through async tasks and request handling.
    """
    tenant_id: str
    tenant_slug: Optional[str] = None
    tenant_name: Optional[str] = None
    tier: Optional[str] = None

    # User context
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_roles: list = field(default_factory=list)

    # Request context
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Permissions
    permissions: set = field(default_factory=set)
    is_admin: bool = False
    is_super_admin: bool = False

    def has_permission(self, permission: str) -> bool:
        """Check if context has permission."""
        if self.is_super_admin:
            return True
        return permission in self.permissions

    def has_role(self, role: str) -> bool:
        """Check if user has role."""
        return role in self.user_roles

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "tenant_slug": self.tenant_slug,
            "tenant_name": self.tenant_name,
            "tier": self.tier,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "user_roles": self.user_roles,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "permissions": list(self.permissions),
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat(),
        }

    def copy(self) -> "TenantContext":
        """Create a copy of the context."""
        return TenantContext(
            tenant_id=self.tenant_id,
            tenant_slug=self.tenant_slug,
            tenant_name=self.tenant_name,
            tier=self.tier,
            user_id=self.user_id,
            user_email=self.user_email,
            user_roles=list(self.user_roles),
            request_id=self.request_id,
            correlation_id=self.correlation_id,
            metadata=dict(self.metadata),
            permissions=set(self.permissions),
            is_admin=self.is_admin,
            is_super_admin=self.is_super_admin,
        )


class TenantContextVar:
    """
    Enhanced context variable for tenant context.

    Provides type-safe access to tenant context.
    """

    def __init__(self):
        self._var: ContextVar[Optional[TenantContext]] = ContextVar(
            "tenant_ctx", default=None
        )
        self._token_stack: list = []

    def get(self) -> Optional[TenantContext]:
        """Get current tenant context."""
        return self._var.get()

    def get_or_raise(self) -> TenantContext:
        """Get current context or raise if not set."""
        ctx = self._var.get()
        if ctx is None:
            raise RuntimeError("No tenant context available")
        return ctx

    def set(self, context: TenantContext) -> Token:
        """Set tenant context."""
        return self._var.set(context)

    def reset(self, token: Token) -> None:
        """Reset to previous context."""
        self._var.reset(token)

    @contextmanager
    def __call__(self, context: TenantContext):
        """Context manager for setting tenant context."""
        token = self.set(context)
        try:
            yield context
        finally:
            self.reset(token)


# Global context variable instance
_context_var = TenantContextVar()


def get_current_tenant() -> Optional[TenantContext]:
    """Get the current tenant context."""
    return _tenant_context.get()


def get_current_tenant_id() -> Optional[str]:
    """Get current tenant ID or None."""
    ctx = _tenant_context.get()
    return ctx.tenant_id if ctx else None


def require_current_tenant() -> TenantContext:
    """Get current tenant context or raise."""
    ctx = _tenant_context.get()
    if ctx is None:
        raise RuntimeError("No tenant context available. Ensure request is authenticated.")
    return ctx


def require_tenant_id() -> str:
    """Get current tenant ID or raise."""
    ctx = require_current_tenant()
    return ctx.tenant_id


def set_current_tenant(context: TenantContext) -> Token:
    """Set the current tenant context."""
    return _tenant_context.set(context)


def clear_tenant_context(token: Optional[Token] = None) -> None:
    """Clear the current tenant context."""
    if token:
        _tenant_context.reset(token)
    else:
        _tenant_context.set(None)


@contextmanager
def tenant_context(
    tenant_id: str,
    **kwargs,
):
    """
    Context manager for tenant scope.

    Usage:
        with tenant_context("tenant-123") as ctx:
            # All operations in this block use tenant-123
            pass
    """
    ctx = TenantContext(tenant_id=tenant_id, **kwargs)
    token = set_current_tenant(ctx)
    try:
        yield ctx
    finally:
        clear_tenant_context(token)


@asynccontextmanager
async def async_tenant_context(
    tenant_id: str,
    **kwargs,
):
    """
    Async context manager for tenant scope.

    Usage:
        async with async_tenant_context("tenant-123") as ctx:
            await some_async_operation()
    """
    ctx = TenantContext(tenant_id=tenant_id, **kwargs)
    token = set_current_tenant(ctx)
    try:
        yield ctx
    finally:
        clear_tenant_context(token)


class TenantContextManager:
    """
    Advanced tenant context manager.

    Provides utilities for context propagation and management.
    """

    def __init__(self):
        self._context_history: list = []
        self._max_history = 100

    def create_context(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> TenantContext:
        """Create a new tenant context."""
        import uuid

        ctx = TenantContext(
            tenant_id=tenant_id,
            user_id=user_id,
            request_id=request_id or str(uuid.uuid4()),
            **kwargs,
        )

        self._record_context(ctx)
        return ctx

    def _record_context(self, ctx: TenantContext) -> None:
        """Record context in history."""
        self._context_history.append({
            "tenant_id": ctx.tenant_id,
            "user_id": ctx.user_id,
            "request_id": ctx.request_id,
            "created_at": ctx.created_at.isoformat(),
        })

        # Trim history
        if len(self._context_history) > self._max_history:
            self._context_history = self._context_history[-self._max_history:]

    def get_current(self) -> Optional[TenantContext]:
        """Get current context."""
        return get_current_tenant()

    def require_current(self) -> TenantContext:
        """Get current context or raise."""
        return require_current_tenant()

    @contextmanager
    def scoped(
        self,
        tenant_id: str,
        **kwargs,
    ):
        """Create scoped tenant context."""
        ctx = self.create_context(tenant_id, **kwargs)
        token = set_current_tenant(ctx)
        try:
            yield ctx
        finally:
            clear_tenant_context(token)

    @asynccontextmanager
    async def async_scoped(
        self,
        tenant_id: str,
        **kwargs,
    ):
        """Create async scoped tenant context."""
        ctx = self.create_context(tenant_id, **kwargs)
        token = set_current_tenant(ctx)
        try:
            yield ctx
        finally:
            clear_tenant_context(token)

    def propagate_to_task(
        self,
        coro,
        context: Optional[TenantContext] = None,
    ):
        """
        Wrap coroutine to propagate tenant context.

        Usage:
            task = asyncio.create_task(
                manager.propagate_to_task(my_coroutine())
            )
        """
        ctx = context or get_current_tenant()

        async def wrapper():
            if ctx:
                token = set_current_tenant(ctx)
                try:
                    return await coro
                finally:
                    clear_tenant_context(token)
            return await coro

        return wrapper()


class TenantContextPropagator:
    """
    Propagates tenant context across async boundaries.

    Ensures context is maintained in:
    - Background tasks
    - Thread pools
    - External calls
    """

    def __init__(self, manager: Optional[TenantContextManager] = None):
        self.manager = manager or TenantContextManager()

    def copy_context(self) -> Optional[TenantContext]:
        """Copy current context for propagation."""
        ctx = get_current_tenant()
        return ctx.copy() if ctx else None

    def wrap_callback(
        self,
        callback: Callable,
        context: Optional[TenantContext] = None,
    ) -> Callable:
        """Wrap callback to include tenant context."""
        ctx = context or self.copy_context()

        def wrapped(*args, **kwargs):
            if ctx:
                with tenant_context(ctx.tenant_id):
                    return callback(*args, **kwargs)
            return callback(*args, **kwargs)

        return wrapped

    def wrap_async_callback(
        self,
        callback: Callable,
        context: Optional[TenantContext] = None,
    ) -> Callable:
        """Wrap async callback to include tenant context."""
        ctx = context or self.copy_context()

        async def wrapped(*args, **kwargs):
            if ctx:
                async with async_tenant_context(ctx.tenant_id):
                    return await callback(*args, **kwargs)
            return await callback(*args, **kwargs)

        return wrapped

    async def run_in_context(
        self,
        coro,
        context: Optional[TenantContext] = None,
    ):
        """Run coroutine in specific tenant context."""
        ctx = context or self.copy_context()

        if ctx:
            async with async_tenant_context(ctx.tenant_id):
                return await coro
        return await coro

    def create_task_with_context(
        self,
        coro,
        context: Optional[TenantContext] = None,
    ) -> asyncio.Task:
        """Create async task with propagated context."""
        return asyncio.create_task(
            self.run_in_context(coro, context)
        )


class TenantContextFilter:
    """
    Filter for validating tenant context operations.
    """

    def __init__(
        self,
        require_tenant: bool = True,
        require_user: bool = False,
        require_request_id: bool = False,
    ):
        self.require_tenant = require_tenant
        self.require_user = require_user
        self.require_request_id = require_request_id

    def validate(self, context: Optional[TenantContext] = None) -> bool:
        """Validate context meets requirements."""
        ctx = context or get_current_tenant()

        if self.require_tenant and (not ctx or not ctx.tenant_id):
            return False

        if self.require_user and (not ctx or not ctx.user_id):
            return False

        if self.require_request_id and (not ctx or not ctx.request_id):
            return False

        return True

    def require(self, context: Optional[TenantContext] = None) -> TenantContext:
        """Validate and return context or raise."""
        if not self.validate(context):
            raise RuntimeError("Tenant context validation failed")
        return context or require_current_tenant()


def tenant_required(func: Callable) -> Callable:
    """
    Decorator requiring tenant context.

    Usage:
        @tenant_required
        async def my_endpoint():
            ctx = get_current_tenant()
            ...
    """
    from functools import wraps

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        if get_current_tenant() is None:
            raise RuntimeError("Tenant context required")
        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        if get_current_tenant() is None:
            raise RuntimeError("Tenant context required")
        return func(*args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def with_tenant_context(tenant_id: str, **kwargs):
    """
    Decorator that sets tenant context for function execution.

    Usage:
        @with_tenant_context("tenant-123")
        async def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        from functools import wraps

        @wraps(func)
        async def async_wrapper(*args, **fkwargs):
            async with async_tenant_context(tenant_id, **kwargs):
                return await func(*args, **fkwargs)

        @wraps(func)
        def sync_wrapper(*args, **fkwargs):
            with tenant_context(tenant_id, **kwargs):
                return func(*args, **fkwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Create global instances
context_manager = TenantContextManager()
context_propagator = TenantContextPropagator(context_manager)
