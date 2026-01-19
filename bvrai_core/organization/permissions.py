"""
Permission System

Role-based access control and permission checking.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

from .base import (
    APIKey,
    APIKeyScope,
    AuthorizationError,
    MemberRole,
    Organization,
    OrganizationMember,
    Permission,
    ROLE_PERMISSIONS,
    Session,
    User,
)


logger = logging.getLogger(__name__)


# Scope to permissions mapping
SCOPE_PERMISSIONS: Dict[APIKeyScope, Set[Permission]] = {
    APIKeyScope.ALL: set(Permission),
    APIKeyScope.AGENTS: {
        Permission.AGENT_CREATE,
        Permission.AGENT_READ,
        Permission.AGENT_UPDATE,
        Permission.AGENT_DELETE,
        Permission.AGENT_DEPLOY,
    },
    APIKeyScope.CALLS: {
        Permission.CALL_INITIATE,
        Permission.CALL_READ,
        Permission.CALL_MANAGE,
    },
    APIKeyScope.ANALYTICS: {
        Permission.ANALYTICS_READ,
        Permission.ANALYTICS_EXPORT,
    },
    APIKeyScope.WEBHOOKS: {
        Permission.WEBHOOK_CREATE,
        Permission.WEBHOOK_READ,
        Permission.WEBHOOK_UPDATE,
        Permission.WEBHOOK_DELETE,
    },
}


@dataclass
class AuthContext:
    """
    Authentication and authorization context.

    Carries information about the current request's authentication.
    """

    # User info
    user: Optional[User] = None
    user_id: Optional[str] = None

    # Organization context
    organization: Optional[Organization] = None
    organization_id: Optional[str] = None

    # Membership
    membership: Optional[OrganizationMember] = None
    role: Optional[MemberRole] = None

    # Authentication method
    session: Optional[Session] = None
    api_key: Optional[APIKey] = None

    # Request context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def is_authenticated(self) -> bool:
        """Check if context is authenticated."""
        return self.user_id is not None

    def is_api_key_auth(self) -> bool:
        """Check if authenticated via API key."""
        return self.api_key is not None

    def is_session_auth(self) -> bool:
        """Check if authenticated via session."""
        return self.session is not None

    def get_effective_permissions(self) -> Set[Permission]:
        """Get effective permissions for this context."""
        permissions: Set[Permission] = set()

        # From membership role
        if self.membership:
            permissions = self.membership.get_permissions()

        # API key scope restrictions
        if self.api_key:
            scope_permissions: Set[Permission] = set()
            for scope in self.api_key.scopes:
                scope_permissions |= SCOPE_PERMISSIONS.get(scope, set())

            # Intersect with role permissions
            permissions = permissions & scope_permissions

        return permissions

    def has_permission(self, permission: Permission) -> bool:
        """Check if context has permission."""
        return permission in self.get_effective_permissions()

    def has_any_permission(self, permissions: Set[Permission]) -> bool:
        """Check if context has any of the permissions."""
        return bool(self.get_effective_permissions() & permissions)

    def has_all_permissions(self, permissions: Set[Permission]) -> bool:
        """Check if context has all of the permissions."""
        return permissions <= self.get_effective_permissions()


class PermissionChecker:
    """
    Permission checker for authorization.

    Provides methods to check permissions in various contexts.
    """

    def __init__(self):
        """Initialize permission checker."""
        self._resource_permissions: Dict[str, Dict[str, Permission]] = {}

        # Setup default resource permissions
        self._setup_resource_permissions()

    def _setup_resource_permissions(self) -> None:
        """Setup default resource permission mappings."""
        self._resource_permissions = {
            "agent": {
                "create": Permission.AGENT_CREATE,
                "read": Permission.AGENT_READ,
                "update": Permission.AGENT_UPDATE,
                "delete": Permission.AGENT_DELETE,
                "deploy": Permission.AGENT_DEPLOY,
            },
            "phone_number": {
                "create": Permission.PHONE_CREATE,
                "read": Permission.PHONE_READ,
                "update": Permission.PHONE_UPDATE,
                "delete": Permission.PHONE_DELETE,
            },
            "call": {
                "create": Permission.CALL_INITIATE,
                "read": Permission.CALL_READ,
                "manage": Permission.CALL_MANAGE,
            },
            "knowledge_base": {
                "create": Permission.KNOWLEDGE_CREATE,
                "read": Permission.KNOWLEDGE_READ,
                "update": Permission.KNOWLEDGE_UPDATE,
                "delete": Permission.KNOWLEDGE_DELETE,
            },
            "webhook": {
                "create": Permission.WEBHOOK_CREATE,
                "read": Permission.WEBHOOK_READ,
                "update": Permission.WEBHOOK_UPDATE,
                "delete": Permission.WEBHOOK_DELETE,
            },
            "campaign": {
                "create": Permission.CAMPAIGN_CREATE,
                "read": Permission.CAMPAIGN_READ,
                "update": Permission.CAMPAIGN_UPDATE,
                "delete": Permission.CAMPAIGN_DELETE,
                "execute": Permission.CAMPAIGN_EXECUTE,
            },
            "api_key": {
                "create": Permission.API_KEY_CREATE,
                "read": Permission.API_KEY_READ,
                "delete": Permission.API_KEY_DELETE,
            },
            "organization": {
                "read": Permission.ORG_READ,
                "update": Permission.ORG_UPDATE,
                "delete": Permission.ORG_DELETE,
                "manage_members": Permission.ORG_MANAGE_MEMBERS,
                "manage_billing": Permission.ORG_MANAGE_BILLING,
            },
        }

    def check(
        self,
        context: AuthContext,
        permission: Permission,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if context has permission.

        Returns:
            Tuple of (allowed, error_message)
        """
        if not context.is_authenticated():
            return (False, "Authentication required")

        if not context.organization_id:
            return (False, "Organization context required")

        if context.has_permission(permission):
            return (True, None)

        return (False, f"Permission denied: {permission.value}")

    def check_resource(
        self,
        context: AuthContext,
        resource_type: str,
        action: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check permission for resource action.

        Args:
            context: Auth context
            resource_type: Type of resource (e.g., "agent", "call")
            action: Action to perform (e.g., "create", "read")

        Returns:
            Tuple of (allowed, error_message)
        """
        resource_perms = self._resource_permissions.get(resource_type)
        if not resource_perms:
            logger.warning(f"Unknown resource type: {resource_type}")
            return (False, f"Unknown resource type: {resource_type}")

        permission = resource_perms.get(action)
        if not permission:
            logger.warning(f"Unknown action {action} for resource {resource_type}")
            return (False, f"Unknown action: {action}")

        return self.check(context, permission)

    def require(
        self,
        context: AuthContext,
        permission: Permission,
    ) -> None:
        """
        Require permission or raise error.

        Raises:
            AuthorizationError: If permission check fails
        """
        allowed, error = self.check(context, permission)
        if not allowed:
            raise AuthorizationError(error or "Permission denied")

    def require_resource(
        self,
        context: AuthContext,
        resource_type: str,
        action: str,
    ) -> None:
        """
        Require resource permission or raise error.

        Raises:
            AuthorizationError: If permission check fails
        """
        allowed, error = self.check_resource(context, resource_type, action)
        if not allowed:
            raise AuthorizationError(error or "Permission denied")

    def require_any(
        self,
        context: AuthContext,
        permissions: Set[Permission],
    ) -> None:
        """
        Require any of the permissions or raise error.

        Raises:
            AuthorizationError: If none of the permissions are granted
        """
        if not context.is_authenticated():
            raise AuthorizationError("Authentication required")

        if not context.has_any_permission(permissions):
            raise AuthorizationError("Permission denied")

    def require_all(
        self,
        context: AuthContext,
        permissions: Set[Permission],
    ) -> None:
        """
        Require all permissions or raise error.

        Raises:
            AuthorizationError: If any permission is missing
        """
        if not context.is_authenticated():
            raise AuthorizationError("Authentication required")

        if not context.has_all_permissions(permissions):
            missing = permissions - context.get_effective_permissions()
            raise AuthorizationError(
                f"Missing permissions: {', '.join(p.value for p in missing)}"
            )

    def can_manage_member(
        self,
        context: AuthContext,
        target_role: MemberRole,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if context can manage member with target role.

        Rules:
        - Only owners can manage other owners
        - Admins can manage members and viewers
        - Members cannot manage anyone
        """
        if not context.has_permission(Permission.ORG_MANAGE_MEMBERS):
            return (False, "Cannot manage members")

        if not context.role:
            return (False, "No role assigned")

        # Role hierarchy
        role_hierarchy = {
            MemberRole.OWNER: 4,
            MemberRole.ADMIN: 3,
            MemberRole.MEMBER: 2,
            MemberRole.VIEWER: 1,
        }

        context_level = role_hierarchy.get(context.role, 0)
        target_level = role_hierarchy.get(target_role, 0)

        # Special case: only owners can manage owners
        if target_role == MemberRole.OWNER and context.role != MemberRole.OWNER:
            return (False, "Only owners can manage owners")

        # Can only manage roles below your level
        if context_level <= target_level:
            return (False, f"Cannot manage {target_role.value} members")

        return (True, None)


# Permission decorator
F = TypeVar('F', bound=Callable[..., Any])


def require_permission(*permissions: Permission) -> Callable[[F], F]:
    """
    Decorator to require permissions on a function.

    Usage:
        @require_permission(Permission.AGENT_CREATE)
        async def create_agent(self, context: AuthContext, ...):
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find AuthContext in args or kwargs
            context = None
            for arg in args:
                if isinstance(arg, AuthContext):
                    context = arg
                    break
            if not context:
                context = kwargs.get('context')

            if not context:
                raise AuthorizationError("No auth context provided")

            # Check permissions
            for permission in permissions:
                allowed, error = PermissionChecker().check(context, permission)
                if not allowed:
                    raise AuthorizationError(error or "Permission denied")

            return await func(*args, **kwargs)

        return wrapper  # type: ignore
    return decorator


def require_resource_permission(
    resource_type: str,
    action: str,
) -> Callable[[F], F]:
    """
    Decorator to require resource permission.

    Usage:
        @require_resource_permission("agent", "create")
        async def create_agent(self, context: AuthContext, ...):
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find AuthContext in args or kwargs
            context = None
            for arg in args:
                if isinstance(arg, AuthContext):
                    context = arg
                    break
            if not context:
                context = kwargs.get('context')

            if not context:
                raise AuthorizationError("No auth context provided")

            # Check permission
            PermissionChecker().require_resource(context, resource_type, action)

            return await func(*args, **kwargs)

        return wrapper  # type: ignore
    return decorator


class ResourceAccessControl:
    """
    Resource-level access control.

    Handles checking access to specific resource instances.
    """

    def __init__(self):
        """Initialize resource access control."""
        self._resource_owners: Dict[str, Dict[str, str]] = {}  # resource_type -> resource_id -> org_id

    def register_resource(
        self,
        resource_type: str,
        resource_id: str,
        organization_id: str,
    ) -> None:
        """Register a resource's ownership."""
        if resource_type not in self._resource_owners:
            self._resource_owners[resource_type] = {}
        self._resource_owners[resource_type][resource_id] = organization_id

    def unregister_resource(
        self,
        resource_type: str,
        resource_id: str,
    ) -> None:
        """Unregister a resource."""
        if resource_type in self._resource_owners:
            self._resource_owners[resource_type].pop(resource_id, None)

    def get_resource_organization(
        self,
        resource_type: str,
        resource_id: str,
    ) -> Optional[str]:
        """Get organization that owns a resource."""
        return self._resource_owners.get(resource_type, {}).get(resource_id)

    def check_access(
        self,
        context: AuthContext,
        resource_type: str,
        resource_id: str,
        action: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check access to a specific resource instance.

        Returns:
            Tuple of (allowed, error_message)
        """
        # First check permission
        checker = PermissionChecker()
        allowed, error = checker.check_resource(context, resource_type, action)
        if not allowed:
            return (allowed, error)

        # Check resource ownership
        owner_org = self.get_resource_organization(resource_type, resource_id)

        if owner_org is None:
            # Resource not registered - allow (might be new)
            return (True, None)

        if owner_org != context.organization_id:
            return (False, "Access denied: resource belongs to another organization")

        return (True, None)

    def require_access(
        self,
        context: AuthContext,
        resource_type: str,
        resource_id: str,
        action: str,
    ) -> None:
        """
        Require access to resource or raise error.

        Raises:
            AuthorizationError: If access check fails
        """
        allowed, error = self.check_access(context, resource_type, resource_id, action)
        if not allowed:
            raise AuthorizationError(error or "Access denied")


class PolicyEngine:
    """
    Policy engine for complex authorization rules.

    Supports defining and evaluating custom policies.
    """

    def __init__(self):
        """Initialize policy engine."""
        self._policies: Dict[str, Callable[[AuthContext, Dict[str, Any]], bool]] = {}

    def register_policy(
        self,
        name: str,
        policy_func: Callable[[AuthContext, Dict[str, Any]], bool],
    ) -> None:
        """Register a policy function."""
        self._policies[name] = policy_func

    def evaluate(
        self,
        policy_name: str,
        context: AuthContext,
        resource: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate a policy.

        Args:
            policy_name: Name of the policy to evaluate
            context: Auth context
            resource: Optional resource data for policy evaluation

        Returns:
            True if policy allows, False otherwise
        """
        policy_func = self._policies.get(policy_name)
        if not policy_func:
            logger.warning(f"Unknown policy: {policy_name}")
            return False

        try:
            return policy_func(context, resource or {})
        except Exception as e:
            logger.error(f"Error evaluating policy {policy_name}: {e}")
            return False

    def require(
        self,
        policy_name: str,
        context: AuthContext,
        resource: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Require policy to pass or raise error.

        Raises:
            AuthorizationError: If policy denies
        """
        if not self.evaluate(policy_name, context, resource):
            raise AuthorizationError(f"Policy denied: {policy_name}")


# Default policy engine with common policies
def create_default_policy_engine() -> PolicyEngine:
    """Create policy engine with default policies."""
    engine = PolicyEngine()

    # Only owners can delete organization
    def can_delete_organization(context: AuthContext, resource: Dict) -> bool:
        return context.role == MemberRole.OWNER

    # Only admins+ can manage billing
    def can_manage_billing(context: AuthContext, resource: Dict) -> bool:
        return context.role in [MemberRole.OWNER, MemberRole.ADMIN]

    # Only owners can transfer ownership
    def can_transfer_ownership(context: AuthContext, resource: Dict) -> bool:
        return context.role == MemberRole.OWNER

    # Rate limit check
    def within_rate_limit(context: AuthContext, resource: Dict) -> bool:
        # This would check against actual rate limit data
        return True

    # Subscription feature check
    def has_feature(context: AuthContext, resource: Dict) -> bool:
        feature = resource.get("feature")
        # This would check against subscription features
        return True

    engine.register_policy("can_delete_organization", can_delete_organization)
    engine.register_policy("can_manage_billing", can_manage_billing)
    engine.register_policy("can_transfer_ownership", can_transfer_ownership)
    engine.register_policy("within_rate_limit", within_rate_limit)
    engine.register_policy("has_feature", has_feature)

    return engine
