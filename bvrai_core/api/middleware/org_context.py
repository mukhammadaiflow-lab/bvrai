"""
Organization Context Middleware

This middleware sets the PostgreSQL session variable for Row-Level Security (RLS)
based on the authenticated user's organization. This is critical for multi-tenant
data isolation.

Usage:
    app.add_middleware(OrgContextMiddleware, db_manager=db)
"""

import logging
from typing import Optional, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from ..auth import AuthContext


logger = logging.getLogger(__name__)


class OrgContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware that sets the organization context for database queries.

    This is required for PostgreSQL Row-Level Security (RLS) to work.
    It sets the `app.current_org_id` session variable based on the
    authenticated user's organization.
    """

    def __init__(
        self,
        app,
        get_db_session: Optional[Callable] = None,
    ):
        """
        Initialize the middleware.

        Args:
            app: The FastAPI/Starlette application
            get_db_session: Optional callable to get database session
        """
        super().__init__(app)
        self._get_db_session = get_db_session

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """
        Set organization context before processing request.

        Args:
            request: The incoming request
            call_next: The next middleware/handler

        Returns:
            The response
        """
        # Get auth context if available
        auth_context: Optional[AuthContext] = getattr(
            request.state, "auth_context", None
        )

        # Store org_id in request state for use in dependencies
        org_id = ""
        if auth_context and auth_context.organization_id:
            org_id = auth_context.organization_id
            request.state.org_id = org_id

        # If we have a database session callback, set the context
        if self._get_db_session and org_id:
            try:
                async with self._get_db_session() as session:
                    # Set PostgreSQL session variable for RLS
                    await session.execute(
                        f"SET app.current_org_id = '{org_id}'"
                    )
            except Exception as e:
                logger.warning(f"Failed to set org context: {e}")

        # Process request
        response = await call_next(request)

        return response


def set_org_context_for_session(session, organization_id: str) -> None:
    """
    Helper function to set organization context on a database session.

    Use this in your request handlers when you have a session already:

        async def my_handler(db: AsyncSession = Depends(get_db)):
            await set_org_context_for_session(db, org_id)
            # Now RLS policies will filter by org_id

    Args:
        session: SQLAlchemy async session
        organization_id: The organization ID to set
    """
    if organization_id:
        session.execute(f"SET app.current_org_id = '{organization_id}'")


async def async_set_org_context(session, organization_id: str) -> None:
    """
    Async helper to set organization context on a database session.

    Args:
        session: SQLAlchemy async session
        organization_id: The organization ID to set
    """
    if organization_id:
        await session.execute(
            f"SET app.current_org_id = '{organization_id}'"
        )


def get_org_context_dependency(auth_context: AuthContext):
    """
    FastAPI dependency that returns a function to set org context.

    Usage:
        @router.get("/items")
        async def get_items(
            set_org: Callable = Depends(get_org_context_dependency),
            db: AsyncSession = Depends(get_db)
        ):
            await set_org(db)
            # Query now filtered by RLS
    """
    async def _set_org(session):
        if auth_context and auth_context.organization_id:
            await async_set_org_context(session, auth_context.organization_id)

    return _set_org


__all__ = [
    "OrgContextMiddleware",
    "set_org_context_for_session",
    "async_set_org_context",
    "get_org_context_dependency",
]
