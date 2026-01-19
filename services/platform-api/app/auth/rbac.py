"""Role-Based Access Control (RBAC) system."""

from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import logging

from fastapi import HTTPException, status, Request

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """
    System permissions.

    Permissions follow the format: resource:action
    """
    # Agent permissions
    AGENTS_READ = "agents:read"
    AGENTS_CREATE = "agents:create"
    AGENTS_UPDATE = "agents:update"
    AGENTS_DELETE = "agents:delete"
    AGENTS_DEPLOY = "agents:deploy"

    # Call permissions
    CALLS_READ = "calls:read"
    CALLS_CREATE = "calls:create"
    CALLS_HANGUP = "calls:hangup"
    CALLS_TRANSFER = "calls:transfer"

    # Knowledge permissions
    KNOWLEDGE_READ = "knowledge:read"
    KNOWLEDGE_CREATE = "knowledge:create"
    KNOWLEDGE_UPDATE = "knowledge:update"
    KNOWLEDGE_DELETE = "knowledge:delete"

    # Phone number permissions
    PHONE_READ = "phone:read"
    PHONE_CREATE = "phone:create"
    PHONE_DELETE = "phone:delete"
    PHONE_CONFIGURE = "phone:configure"

    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_EXPORT = "analytics:export"

    # Webhook permissions
    WEBHOOKS_READ = "webhooks:read"
    WEBHOOKS_CREATE = "webhooks:create"
    WEBHOOKS_UPDATE = "webhooks:update"
    WEBHOOKS_DELETE = "webhooks:delete"

    # API key permissions
    API_KEYS_READ = "api_keys:read"
    API_KEYS_CREATE = "api_keys:create"
    API_KEYS_REVOKE = "api_keys:revoke"

    # User/team permissions
    USERS_READ = "users:read"
    USERS_INVITE = "users:invite"
    USERS_REMOVE = "users:remove"
    USERS_MANAGE_ROLES = "users:manage_roles"

    # Billing permissions
    BILLING_READ = "billing:read"
    BILLING_UPDATE = "billing:update"

    # Admin permissions
    ADMIN_ACCESS = "admin:access"
    ADMIN_IMPERSONATE = "admin:impersonate"


class Role(str, Enum):
    """System roles."""
    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    VIEWER = "viewer"


@dataclass
class RoleDefinition:
    """Definition of a role with its permissions."""
    name: str
    description: str
    permissions: Set[Permission]
    inherits_from: Optional[str] = None


# Define role permissions
ROLE_PERMISSIONS: Dict[Role, RoleDefinition] = {
    Role.VIEWER: RoleDefinition(
        name="Viewer",
        description="Read-only access to resources",
        permissions={
            Permission.AGENTS_READ,
            Permission.CALLS_READ,
            Permission.KNOWLEDGE_READ,
            Permission.PHONE_READ,
            Permission.ANALYTICS_READ,
            Permission.WEBHOOKS_READ,
        },
    ),
    Role.ANALYST: RoleDefinition(
        name="Analyst",
        description="Access to analytics and reporting",
        permissions={
            Permission.AGENTS_READ,
            Permission.CALLS_READ,
            Permission.KNOWLEDGE_READ,
            Permission.PHONE_READ,
            Permission.ANALYTICS_READ,
            Permission.ANALYTICS_EXPORT,
            Permission.WEBHOOKS_READ,
        },
    ),
    Role.DEVELOPER: RoleDefinition(
        name="Developer",
        description="Full access to development resources",
        permissions={
            Permission.AGENTS_READ,
            Permission.AGENTS_CREATE,
            Permission.AGENTS_UPDATE,
            Permission.AGENTS_DELETE,
            Permission.AGENTS_DEPLOY,
            Permission.CALLS_READ,
            Permission.CALLS_CREATE,
            Permission.CALLS_HANGUP,
            Permission.CALLS_TRANSFER,
            Permission.KNOWLEDGE_READ,
            Permission.KNOWLEDGE_CREATE,
            Permission.KNOWLEDGE_UPDATE,
            Permission.KNOWLEDGE_DELETE,
            Permission.PHONE_READ,
            Permission.PHONE_CONFIGURE,
            Permission.ANALYTICS_READ,
            Permission.ANALYTICS_EXPORT,
            Permission.WEBHOOKS_READ,
            Permission.WEBHOOKS_CREATE,
            Permission.WEBHOOKS_UPDATE,
            Permission.WEBHOOKS_DELETE,
            Permission.API_KEYS_READ,
            Permission.API_KEYS_CREATE,
        },
    ),
    Role.ADMIN: RoleDefinition(
        name="Admin",
        description="Administrative access to organization",
        permissions={
            Permission.AGENTS_READ,
            Permission.AGENTS_CREATE,
            Permission.AGENTS_UPDATE,
            Permission.AGENTS_DELETE,
            Permission.AGENTS_DEPLOY,
            Permission.CALLS_READ,
            Permission.CALLS_CREATE,
            Permission.CALLS_HANGUP,
            Permission.CALLS_TRANSFER,
            Permission.KNOWLEDGE_READ,
            Permission.KNOWLEDGE_CREATE,
            Permission.KNOWLEDGE_UPDATE,
            Permission.KNOWLEDGE_DELETE,
            Permission.PHONE_READ,
            Permission.PHONE_CREATE,
            Permission.PHONE_DELETE,
            Permission.PHONE_CONFIGURE,
            Permission.ANALYTICS_READ,
            Permission.ANALYTICS_EXPORT,
            Permission.WEBHOOKS_READ,
            Permission.WEBHOOKS_CREATE,
            Permission.WEBHOOKS_UPDATE,
            Permission.WEBHOOKS_DELETE,
            Permission.API_KEYS_READ,
            Permission.API_KEYS_CREATE,
            Permission.API_KEYS_REVOKE,
            Permission.USERS_READ,
            Permission.USERS_INVITE,
            Permission.USERS_REMOVE,
            Permission.BILLING_READ,
        },
    ),
    Role.OWNER: RoleDefinition(
        name="Owner",
        description="Full access to organization",
        permissions={p for p in Permission if not p.value.startswith("admin:")},
    ),
}


class RBACManager:
    """
    Role-Based Access Control manager.

    Usage:
        rbac = RBACManager()

        # Check permission
        if rbac.has_permission(user.role, Permission.AGENTS_CREATE):
            create_agent()

        # Get role permissions
        permissions = rbac.get_permissions(Role.DEVELOPER)
    """

    def __init__(self):
        self._role_permissions = ROLE_PERMISSIONS.copy()
        self._custom_permissions: Dict[str, Set[Permission]] = {}

    def get_permissions(self, role: Role) -> Set[Permission]:
        """Get all permissions for a role."""
        role_def = self._role_permissions.get(role)
        if not role_def:
            return set()
        return role_def.permissions.copy()

    def has_permission(self, role: Role, permission: Permission) -> bool:
        """Check if a role has a specific permission."""
        permissions = self.get_permissions(role)
        return permission in permissions

    def has_any_permission(
        self,
        role: Role,
        permissions: List[Permission],
    ) -> bool:
        """Check if a role has any of the specified permissions."""
        role_permissions = self.get_permissions(role)
        return any(p in role_permissions for p in permissions)

    def has_all_permissions(
        self,
        role: Role,
        permissions: List[Permission],
    ) -> bool:
        """Check if a role has all of the specified permissions."""
        role_permissions = self.get_permissions(role)
        return all(p in role_permissions for p in permissions)

    def add_custom_permission(
        self,
        user_id: str,
        permission: Permission,
    ) -> None:
        """Add a custom permission to a user."""
        if user_id not in self._custom_permissions:
            self._custom_permissions[user_id] = set()
        self._custom_permissions[user_id].add(permission)

    def remove_custom_permission(
        self,
        user_id: str,
        permission: Permission,
    ) -> None:
        """Remove a custom permission from a user."""
        if user_id in self._custom_permissions:
            self._custom_permissions[user_id].discard(permission)

    def get_user_permissions(
        self,
        role: Role,
        user_id: Optional[str] = None,
    ) -> Set[Permission]:
        """Get all permissions for a user including custom permissions."""
        permissions = self.get_permissions(role)
        if user_id and user_id in self._custom_permissions:
            permissions = permissions.union(self._custom_permissions[user_id])
        return permissions

    def check_permission_for_user(
        self,
        role: Role,
        permission: Permission,
        user_id: Optional[str] = None,
    ) -> bool:
        """Check if a user has a specific permission."""
        permissions = self.get_user_permissions(role, user_id)
        return permission in permissions


# Global RBAC manager
_rbac_manager: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get or create the global RBAC manager."""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager


@dataclass
class ResourcePolicy:
    """Policy for resource access."""
    resource_type: str
    owner_only: bool = False
    required_permission: Optional[Permission] = None
    custom_check: Optional[callable] = None


class PolicyEnforcer:
    """
    Policy enforcer for resource-level access control.

    Usage:
        enforcer = PolicyEnforcer()
        enforcer.register_policy("agents", ResourcePolicy(
            resource_type="agents",
            required_permission=Permission.AGENTS_READ,
        ))

        # Check access
        enforcer.check_access(user, "agents", agent_id)
    """

    def __init__(self, rbac: Optional[RBACManager] = None):
        self.rbac = rbac or get_rbac_manager()
        self._policies: Dict[str, ResourcePolicy] = {}

    def register_policy(self, resource_type: str, policy: ResourcePolicy) -> None:
        """Register a resource policy."""
        self._policies[resource_type] = policy

    async def check_access(
        self,
        user_role: Role,
        user_id: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        action: str = "read",
        resource_owner_id: Optional[str] = None,
    ) -> bool:
        """
        Check if user has access to a resource.

        Args:
            user_role: User's role
            user_id: User's ID
            resource_type: Type of resource (e.g., "agents")
            resource_id: ID of specific resource (optional)
            action: Action being performed (read, create, update, delete)
            resource_owner_id: Owner of the resource (for owner-only policies)

        Returns:
            True if access is allowed, False otherwise
        """
        policy = self._policies.get(resource_type)
        if not policy:
            # No policy defined, allow by default
            return True

        # Check owner-only policy
        if policy.owner_only and resource_owner_id:
            if user_id != resource_owner_id:
                return False

        # Check required permission
        if policy.required_permission:
            if not self.rbac.check_permission_for_user(
                user_role,
                policy.required_permission,
                user_id,
            ):
                return False

        # Run custom check if defined
        if policy.custom_check:
            return await policy.custom_check(
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
            )

        return True

    def enforce_access(
        self,
        user_role: Role,
        user_id: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        action: str = "read",
        resource_owner_id: Optional[str] = None,
    ) -> None:
        """Enforce access policy, raising exception if denied."""
        import asyncio
        loop = asyncio.get_event_loop()
        allowed = loop.run_until_complete(
            self.check_access(
                user_role,
                user_id,
                resource_type,
                resource_id,
                action,
                resource_owner_id,
            )
        )
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied to {resource_type}",
            )


def require_permission(*permissions: Permission):
    """
    FastAPI dependency for requiring specific permissions.

    Usage:
        @app.post("/agents")
        async def create_agent(
            _: None = Depends(require_permission(Permission.AGENTS_CREATE))
        ):
            ...
    """
    def dependency(request: Request) -> None:
        from app.auth.middleware import get_current_user

        user = get_current_user(request)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        # Get user role (would normally come from user object/database)
        user_role = getattr(user, "role", Role.VIEWER)
        if isinstance(user_role, str):
            user_role = Role(user_role)

        rbac = get_rbac_manager()
        if not rbac.has_any_permission(user_role, list(permissions)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {[p.value for p in permissions]}",
            )

    return dependency


def check_permission(permission: Permission):
    """
    Decorator for checking permission on endpoint.

    Usage:
        @check_permission(Permission.AGENTS_CREATE)
        async def create_agent():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            if not request:
                # Try to find request in args
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request:
                from app.auth.middleware import get_current_user
                user = get_current_user(request)
                if user:
                    user_role = getattr(user, "role", Role.VIEWER)
                    if isinstance(user_role, str):
                        user_role = Role(user_role)

                    rbac = get_rbac_manager()
                    if not rbac.has_permission(user_role, permission):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Missing required permission: {permission.value}",
                        )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Scope to permission mapping
SCOPE_TO_PERMISSIONS: Dict[str, List[Permission]] = {
    "agents:read": [Permission.AGENTS_READ],
    "agents:write": [Permission.AGENTS_CREATE, Permission.AGENTS_UPDATE, Permission.AGENTS_DELETE],
    "calls:read": [Permission.CALLS_READ],
    "calls:write": [Permission.CALLS_CREATE, Permission.CALLS_HANGUP],
    "knowledge:read": [Permission.KNOWLEDGE_READ],
    "knowledge:write": [Permission.KNOWLEDGE_CREATE, Permission.KNOWLEDGE_UPDATE, Permission.KNOWLEDGE_DELETE],
    "analytics:read": [Permission.ANALYTICS_READ],
    "webhooks:read": [Permission.WEBHOOKS_READ],
    "webhooks:write": [Permission.WEBHOOKS_CREATE, Permission.WEBHOOKS_UPDATE, Permission.WEBHOOKS_DELETE],
    "*": list(Permission),  # All permissions
}


def scopes_to_permissions(scopes: List[str]) -> Set[Permission]:
    """Convert API scopes to permissions."""
    permissions = set()
    for scope in scopes:
        if scope in SCOPE_TO_PERMISSIONS:
            permissions.update(SCOPE_TO_PERMISSIONS[scope])
    return permissions


def permissions_to_scopes(permissions: Set[Permission]) -> List[str]:
    """Convert permissions to API scopes."""
    scopes = []
    for scope, perms in SCOPE_TO_PERMISSIONS.items():
        if scope != "*" and all(p in permissions for p in perms):
            scopes.append(scope)
    return scopes
