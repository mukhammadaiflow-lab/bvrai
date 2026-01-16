"""
Access Control System
=====================

Role-Based (RBAC) and Attribute-Based (ABAC) access control
for fine-grained authorization.

Author: Platform Security Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


class PolicyEffect(str, Enum):
    """Effect of a policy."""

    ALLOW = "allow"
    DENY = "deny"


class AccessDecision(str, Enum):
    """Result of an access check."""

    ALLOWED = "allowed"
    DENIED = "denied"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class Permission:
    """A permission grants access to perform an action on a resource."""

    id: str = field(default_factory=lambda: f"perm_{uuid4().hex[:12]}")
    name: str = ""
    description: str = ""
    resource: str = "*"  # Resource type or pattern
    actions: List[str] = field(default_factory=lambda: ["*"])  # Actions allowed
    conditions: Dict[str, Any] = field(default_factory=dict)  # Additional conditions
    created_at: datetime = field(default_factory=datetime.utcnow)

    def matches_resource(self, resource: str) -> bool:
        """Check if permission matches resource."""
        if self.resource == "*":
            return True
        return fnmatch.fnmatch(resource, self.resource)

    def matches_action(self, action: str) -> bool:
        """Check if permission allows action."""
        if "*" in self.actions:
            return True
        return action in self.actions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "resource": self.resource,
            "actions": self.actions,
            "conditions": self.conditions,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Role:
    """A role is a collection of permissions."""

    id: str = field(default_factory=lambda: f"role_{uuid4().hex[:12]}")
    name: str = ""
    description: str = ""
    permissions: List[str] = field(default_factory=list)  # Permission IDs
    parent_roles: List[str] = field(default_factory=list)  # Inherited roles
    organization_id: Optional[str] = None
    is_system: bool = False  # System roles can't be modified
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "permissions": self.permissions,
            "parent_roles": self.parent_roles,
            "organization_id": self.organization_id,
            "is_system": self.is_system,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Policy:
    """
    An access control policy.

    Policies define rules for granting or denying access based on
    various conditions including resource, action, subject attributes,
    and environmental factors.
    """

    id: str = field(default_factory=lambda: f"policy_{uuid4().hex[:12]}")
    name: str = ""
    description: str = ""
    effect: PolicyEffect = PolicyEffect.ALLOW
    priority: int = 0  # Higher priority policies are evaluated first

    # Subject conditions (who)
    subjects: List[str] = field(default_factory=list)  # User/role IDs or patterns
    subject_attributes: Dict[str, Any] = field(default_factory=dict)

    # Resource conditions (what)
    resources: List[str] = field(default_factory=list)  # Resource patterns
    resource_attributes: Dict[str, Any] = field(default_factory=dict)

    # Action conditions (how)
    actions: List[str] = field(default_factory=list)  # Action patterns

    # Environmental conditions (when/where)
    conditions: Dict[str, Any] = field(default_factory=dict)
    time_restrictions: Dict[str, Any] = field(default_factory=dict)
    ip_restrictions: List[str] = field(default_factory=list)

    # Status
    enabled: bool = True
    organization_id: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def matches(
        self,
        subject_id: str,
        resource: str,
        action: str,
        context: Dict[str, Any],
    ) -> bool:
        """Check if policy matches the access request."""
        if not self.enabled:
            return False

        # Check subject
        if self.subjects:
            subject_match = any(
                self._pattern_matches(subject_id, pattern)
                for pattern in self.subjects
            )
            if not subject_match:
                return False

        # Check subject attributes
        if self.subject_attributes:
            for attr, expected in self.subject_attributes.items():
                actual = context.get(f"subject.{attr}")
                if not self._value_matches(actual, expected):
                    return False

        # Check resource
        if self.resources:
            resource_match = any(
                self._pattern_matches(resource, pattern)
                for pattern in self.resources
            )
            if not resource_match:
                return False

        # Check resource attributes
        if self.resource_attributes:
            for attr, expected in self.resource_attributes.items():
                actual = context.get(f"resource.{attr}")
                if not self._value_matches(actual, expected):
                    return False

        # Check action
        if self.actions:
            action_match = any(
                self._pattern_matches(action, pattern)
                for pattern in self.actions
            )
            if not action_match:
                return False

        # Check conditions
        if self.conditions:
            for key, expected in self.conditions.items():
                actual = context.get(key)
                if not self._value_matches(actual, expected):
                    return False

        # Check time restrictions
        if self.time_restrictions:
            if not self._check_time_restrictions(context):
                return False

        # Check IP restrictions
        if self.ip_restrictions:
            client_ip = context.get("client_ip")
            if client_ip and not self._check_ip_restrictions(client_ip):
                return False

        return True

    def _pattern_matches(self, value: str, pattern: str) -> bool:
        """Check if value matches pattern (supports wildcards)."""
        if pattern == "*":
            return True
        if "*" in pattern:
            return fnmatch.fnmatch(value, pattern)
        return value == pattern

    def _value_matches(self, actual: Any, expected: Any) -> bool:
        """Check if actual value matches expected."""
        if expected is None:
            return True
        if isinstance(expected, dict):
            # Comparison operators
            for op, val in expected.items():
                if op == "$eq":
                    if actual != val:
                        return False
                elif op == "$ne":
                    if actual == val:
                        return False
                elif op == "$gt":
                    if not (actual and actual > val):
                        return False
                elif op == "$gte":
                    if not (actual and actual >= val):
                        return False
                elif op == "$lt":
                    if not (actual and actual < val):
                        return False
                elif op == "$lte":
                    if not (actual and actual <= val):
                        return False
                elif op == "$in":
                    if actual not in val:
                        return False
                elif op == "$nin":
                    if actual in val:
                        return False
                elif op == "$exists":
                    if (actual is not None) != val:
                        return False
                elif op == "$regex":
                    if not (actual and re.match(val, str(actual))):
                        return False
            return True
        if isinstance(expected, list):
            return actual in expected
        return actual == expected

    def _check_time_restrictions(self, context: Dict[str, Any]) -> bool:
        """Check time-based restrictions."""
        now = context.get("current_time", datetime.utcnow())

        # Check day of week
        allowed_days = self.time_restrictions.get("days_of_week")
        if allowed_days:
            if now.weekday() not in allowed_days:
                return False

        # Check hours
        start_hour = self.time_restrictions.get("start_hour")
        end_hour = self.time_restrictions.get("end_hour")
        if start_hour is not None and end_hour is not None:
            if not (start_hour <= now.hour < end_hour):
                return False

        return True

    def _check_ip_restrictions(self, client_ip: str) -> bool:
        """Check IP-based restrictions."""
        for pattern in self.ip_restrictions:
            if self._ip_matches(client_ip, pattern):
                return True
        return False

    def _ip_matches(self, ip: str, pattern: str) -> bool:
        """Check if IP matches pattern or CIDR."""
        if pattern == "*":
            return True
        if "/" in pattern:
            # CIDR notation
            try:
                import ipaddress
                network = ipaddress.ip_network(pattern, strict=False)
                return ipaddress.ip_address(ip) in network
            except (ValueError, ImportError):
                return ip.startswith(pattern.split("/")[0])
        return ip == pattern or fnmatch.fnmatch(ip, pattern)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "effect": self.effect.value,
            "priority": self.priority,
            "subjects": self.subjects,
            "subject_attributes": self.subject_attributes,
            "resources": self.resources,
            "resource_attributes": self.resource_attributes,
            "actions": self.actions,
            "conditions": self.conditions,
            "time_restrictions": self.time_restrictions,
            "ip_restrictions": self.ip_restrictions,
            "enabled": self.enabled,
            "organization_id": self.organization_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ResourceMatcher:
    """Utility for matching resources against patterns."""

    @staticmethod
    def matches(resource: str, pattern: str) -> bool:
        """Check if resource matches pattern."""
        if pattern == "*":
            return True
        if pattern.endswith("/*"):
            prefix = pattern[:-2]
            return resource.startswith(prefix)
        if "*" in pattern:
            return fnmatch.fnmatch(resource, pattern)
        return resource == pattern

    @staticmethod
    def parse_resource(resource: str) -> Dict[str, str]:
        """
        Parse resource string into components.

        Format: type:org_id/resource_id
        """
        result = {"type": "", "organization_id": "", "resource_id": ""}

        if ":" in resource:
            result["type"], rest = resource.split(":", 1)
        else:
            rest = resource

        if "/" in rest:
            result["organization_id"], result["resource_id"] = rest.split("/", 1)
        else:
            result["resource_id"] = rest

        return result


@dataclass
class AccessRequest:
    """An access control request."""

    subject_id: str
    resource: str
    action: str
    context: Dict[str, Any] = field(default_factory=dict)
    organization_id: Optional[str] = None


@dataclass
class AccessResult:
    """Result of an access check."""

    decision: AccessDecision
    reason: str = ""
    policy_id: Optional[str] = None
    evaluated_policies: int = 0
    duration_ms: float = 0.0


class RBACManager:
    """
    Role-Based Access Control Manager.

    Manages roles, permissions, and role assignments for users.
    """

    def __init__(self):
        self._roles: Dict[str, Role] = {}
        self._permissions: Dict[str, Permission] = {}
        self._user_roles: Dict[str, Set[str]] = defaultdict(set)  # user_id -> role_ids
        self._role_permissions: Dict[str, Set[str]] = defaultdict(set)  # role_id -> permission_ids
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger("rbac_manager")
        self._init_system_roles()

    def _init_system_roles(self) -> None:
        """Initialize system roles and permissions."""
        # System permissions
        system_permissions = [
            Permission(id="perm_all", name="All Access", resource="*", actions=["*"]),
            Permission(id="perm_read_all", name="Read All", resource="*", actions=["read", "list"]),
            Permission(id="perm_users_manage", name="Manage Users", resource="users/*", actions=["*"]),
            Permission(id="perm_calls_read", name="Read Calls", resource="calls/*", actions=["read", "list"]),
            Permission(id="perm_calls_manage", name="Manage Calls", resource="calls/*", actions=["*"]),
            Permission(id="perm_recordings_read", name="Read Recordings", resource="recordings/*", actions=["read", "list"]),
            Permission(id="perm_recordings_manage", name="Manage Recordings", resource="recordings/*", actions=["*"]),
            Permission(id="perm_agents_read", name="Read Agents", resource="agents/*", actions=["read", "list"]),
            Permission(id="perm_agents_manage", name="Manage Agents", resource="agents/*", actions=["*"]),
            Permission(id="perm_analytics_read", name="Read Analytics", resource="analytics/*", actions=["read", "list"]),
            Permission(id="perm_settings_read", name="Read Settings", resource="settings/*", actions=["read"]),
            Permission(id="perm_settings_manage", name="Manage Settings", resource="settings/*", actions=["*"]),
            Permission(id="perm_integrations_manage", name="Manage Integrations", resource="integrations/*", actions=["*"]),
            Permission(id="perm_billing_read", name="Read Billing", resource="billing/*", actions=["read", "list"]),
            Permission(id="perm_billing_manage", name="Manage Billing", resource="billing/*", actions=["*"]),
        ]

        for perm in system_permissions:
            self._permissions[perm.id] = perm

        # System roles
        system_roles = [
            Role(
                id="role_super_admin",
                name="Super Admin",
                description="Full system access",
                permissions=["perm_all"],
                is_system=True,
            ),
            Role(
                id="role_admin",
                name="Admin",
                description="Organization admin",
                permissions=[
                    "perm_users_manage", "perm_calls_manage", "perm_recordings_manage",
                    "perm_agents_manage", "perm_analytics_read", "perm_settings_manage",
                    "perm_integrations_manage", "perm_billing_read",
                ],
                is_system=True,
            ),
            Role(
                id="role_manager",
                name="Manager",
                description="Team manager",
                permissions=[
                    "perm_calls_manage", "perm_recordings_read", "perm_agents_read",
                    "perm_analytics_read", "perm_settings_read",
                ],
                is_system=True,
            ),
            Role(
                id="role_agent",
                name="Agent",
                description="Call center agent",
                permissions=[
                    "perm_calls_read", "perm_recordings_read", "perm_agents_read",
                ],
                is_system=True,
            ),
            Role(
                id="role_viewer",
                name="Viewer",
                description="Read-only access",
                permissions=["perm_read_all"],
                is_system=True,
            ),
        ]

        for role in system_roles:
            self._roles[role.id] = role
            self._role_permissions[role.id] = set(role.permissions)

    async def create_role(
        self,
        name: str,
        description: str = "",
        permissions: Optional[List[str]] = None,
        parent_roles: Optional[List[str]] = None,
        organization_id: Optional[str] = None,
    ) -> Role:
        """Create a new role."""
        role = Role(
            name=name,
            description=description,
            permissions=permissions or [],
            parent_roles=parent_roles or [],
            organization_id=organization_id,
        )

        async with self._lock:
            self._roles[role.id] = role
            self._role_permissions[role.id] = set(role.permissions)

        self._logger.info(f"Created role: {role.name}")
        return role

    async def get_role(self, role_id: str) -> Optional[Role]:
        """Get a role by ID."""
        return self._roles.get(role_id)

    async def update_role(
        self,
        role_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        permissions: Optional[List[str]] = None,
    ) -> Optional[Role]:
        """Update a role."""
        role = self._roles.get(role_id)
        if not role:
            return None
        if role.is_system:
            raise ValueError("Cannot modify system roles")

        async with self._lock:
            if name:
                role.name = name
            if description:
                role.description = description
            if permissions is not None:
                role.permissions = permissions
                self._role_permissions[role_id] = set(permissions)
            role.updated_at = datetime.utcnow()

        return role

    async def delete_role(self, role_id: str) -> bool:
        """Delete a role."""
        role = self._roles.get(role_id)
        if not role:
            return False
        if role.is_system:
            raise ValueError("Cannot delete system roles")

        async with self._lock:
            del self._roles[role_id]
            del self._role_permissions[role_id]

            # Remove role from users
            for user_id in list(self._user_roles.keys()):
                self._user_roles[user_id].discard(role_id)

        return True

    async def assign_role(self, user_id: str, role_id: str) -> None:
        """Assign a role to a user."""
        if role_id not in self._roles:
            raise ValueError(f"Role not found: {role_id}")

        async with self._lock:
            self._user_roles[user_id].add(role_id)

        self._logger.info(f"Assigned role {role_id} to user {user_id}")

    async def revoke_role(self, user_id: str, role_id: str) -> None:
        """Revoke a role from a user."""
        async with self._lock:
            self._user_roles[user_id].discard(role_id)

        self._logger.info(f"Revoked role {role_id} from user {user_id}")

    async def get_user_roles(self, user_id: str) -> List[Role]:
        """Get all roles for a user."""
        role_ids = self._user_roles.get(user_id, set())
        return [self._roles[rid] for rid in role_ids if rid in self._roles]

    async def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user (including inherited)."""
        permissions: Set[str] = set()

        def collect_permissions(role_id: str, visited: Set[str]) -> None:
            if role_id in visited:
                return
            visited.add(role_id)

            role = self._roles.get(role_id)
            if not role:
                return

            permissions.update(role.permissions)

            # Collect from parent roles
            for parent_id in role.parent_roles:
                collect_permissions(parent_id, visited)

        visited: Set[str] = set()
        for role_id in self._user_roles.get(user_id, set()):
            collect_permissions(role_id, visited)

        return permissions

    async def check_permission(
        self,
        user_id: str,
        resource: str,
        action: str,
    ) -> bool:
        """Check if user has permission for resource and action."""
        permission_ids = await self.get_user_permissions(user_id)

        for perm_id in permission_ids:
            perm = self._permissions.get(perm_id)
            if perm and perm.matches_resource(resource) and perm.matches_action(action):
                return True

        return False


class ABACManager:
    """
    Attribute-Based Access Control Manager.

    Evaluates policies based on attributes of subjects, resources,
    actions, and environment.
    """

    def __init__(self, rbac: Optional[RBACManager] = None):
        self._policies: Dict[str, Policy] = {}
        self._rbac = rbac
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger("abac_manager")
        self._decision_cache: Dict[str, Tuple[AccessResult, datetime]] = {}
        self._cache_ttl_seconds = 60

    async def add_policy(self, policy: Policy) -> None:
        """Add a policy."""
        async with self._lock:
            self._policies[policy.id] = policy
        self._logger.info(f"Added policy: {policy.name}")

    async def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy."""
        async with self._lock:
            if policy_id in self._policies:
                del self._policies[policy_id]
                return True
        return False

    async def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    async def list_policies(
        self,
        organization_id: Optional[str] = None,
        enabled_only: bool = True,
    ) -> List[Policy]:
        """List policies."""
        policies = list(self._policies.values())

        if organization_id:
            policies = [p for p in policies if p.organization_id == organization_id or p.organization_id is None]

        if enabled_only:
            policies = [p for p in policies if p.enabled]

        return sorted(policies, key=lambda p: p.priority, reverse=True)

    async def check_access(
        self,
        request: AccessRequest,
    ) -> AccessResult:
        """
        Check if access should be granted.

        Uses deny-by-default: access is denied unless explicitly allowed.
        Deny policies always take precedence over allow policies.
        """
        import time
        start_time = time.time()

        # Check cache
        cache_key = f"{request.subject_id}:{request.resource}:{request.action}"
        if cache_key in self._decision_cache:
            result, cached_at = self._decision_cache[cache_key]
            if (datetime.utcnow() - cached_at).total_seconds() < self._cache_ttl_seconds:
                return result

        # Build context
        context = {**request.context}
        context["subject.id"] = request.subject_id
        context["resource"] = request.resource
        context["action"] = request.action
        if request.organization_id:
            context["organization_id"] = request.organization_id

        # Check RBAC first if available
        if self._rbac:
            has_permission = await self._rbac.check_permission(
                request.subject_id,
                request.resource,
                request.action,
            )
            if has_permission:
                context["rbac_allowed"] = True

        # Get applicable policies
        policies = await self.list_policies(
            organization_id=request.organization_id,
            enabled_only=True,
        )

        # Evaluate policies (deny takes precedence)
        allow_policy: Optional[Policy] = None
        evaluated = 0

        for policy in policies:
            evaluated += 1
            if policy.matches(request.subject_id, request.resource, request.action, context):
                if policy.effect == PolicyEffect.DENY:
                    # Deny immediately
                    result = AccessResult(
                        decision=AccessDecision.DENIED,
                        reason=f"Denied by policy: {policy.name}",
                        policy_id=policy.id,
                        evaluated_policies=evaluated,
                        duration_ms=(time.time() - start_time) * 1000,
                    )
                    self._decision_cache[cache_key] = (result, datetime.utcnow())
                    return result

                if not allow_policy:
                    allow_policy = policy

        # Check if allowed
        if allow_policy:
            result = AccessResult(
                decision=AccessDecision.ALLOWED,
                reason=f"Allowed by policy: {allow_policy.name}",
                policy_id=allow_policy.id,
                evaluated_policies=evaluated,
                duration_ms=(time.time() - start_time) * 1000,
            )
        elif context.get("rbac_allowed"):
            result = AccessResult(
                decision=AccessDecision.ALLOWED,
                reason="Allowed by RBAC permission",
                evaluated_policies=evaluated,
                duration_ms=(time.time() - start_time) * 1000,
            )
        else:
            result = AccessResult(
                decision=AccessDecision.DENIED,
                reason="No matching allow policy",
                evaluated_policies=evaluated,
                duration_ms=(time.time() - start_time) * 1000,
            )

        self._decision_cache[cache_key] = (result, datetime.utcnow())
        return result

    def clear_cache(self, subject_id: Optional[str] = None) -> None:
        """Clear decision cache."""
        if subject_id:
            keys_to_remove = [k for k in self._decision_cache if k.startswith(f"{subject_id}:")]
            for key in keys_to_remove:
                del self._decision_cache[key]
        else:
            self._decision_cache.clear()


class AccessControl:
    """
    Unified Access Control System.

    Combines RBAC and ABAC for comprehensive access control.
    """

    def __init__(self):
        self.rbac = RBACManager()
        self.abac = ABACManager(self.rbac)
        self._logger = structlog.get_logger("access_control")

    async def check(
        self,
        subject_id: str,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        organization_id: Optional[str] = None,
    ) -> AccessResult:
        """
        Check if access should be granted.

        Args:
            subject_id: ID of the subject (user, service, etc.)
            resource: Resource being accessed
            action: Action being performed
            context: Additional context for evaluation
            organization_id: Organization context

        Returns:
            AccessResult with decision and details
        """
        request = AccessRequest(
            subject_id=subject_id,
            resource=resource,
            action=action,
            context=context or {},
            organization_id=organization_id,
        )

        result = await self.abac.check_access(request)

        self._logger.debug(
            "Access check",
            subject=subject_id,
            resource=resource,
            action=action,
            decision=result.decision.value,
            reason=result.reason,
        )

        return result

    async def require(
        self,
        subject_id: str,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        organization_id: Optional[str] = None,
    ) -> None:
        """
        Require access, raising exception if denied.

        Raises:
            PermissionError: If access is denied
        """
        result = await self.check(
            subject_id, resource, action, context, organization_id
        )

        if result.decision != AccessDecision.ALLOWED:
            raise PermissionError(f"Access denied: {result.reason}")

    def create_context(
        self,
        request: Any = None,
        user: Any = None,
        organization: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create context dictionary for access checks."""
        context: Dict[str, Any] = {}

        if request:
            context["client_ip"] = getattr(request, "client_ip", None) or getattr(request, "remote_addr", None)
            context["request_method"] = getattr(request, "method", None)
            context["request_path"] = getattr(request, "path", None)

        if user:
            context["subject.id"] = getattr(user, "id", None)
            context["subject.email"] = getattr(user, "email", None)
            context["subject.roles"] = getattr(user, "roles", [])

        if organization:
            context["organization_id"] = getattr(organization, "id", None)
            context["organization.plan"] = getattr(organization, "plan", None)

        context["current_time"] = datetime.utcnow()
        context.update(kwargs)

        return context
