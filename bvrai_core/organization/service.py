"""
Organization Service

Main organization management service.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .auth import (
    APIKeyManager,
    AuthConfig,
    AuthenticationService,
    SessionManager,
    generate_token,
    hash_password,
)
from .base import (
    APIKey,
    APIKeyScope,
    AuditLogEntry,
    AuthorizationError,
    MemberRole,
    Organization,
    OrganizationError,
    OrganizationMember,
    OrganizationNotFoundError,
    OrganizationStatus,
    Permission,
    Session,
    Team,
    User,
    UserNotFoundError,
    UserStatus,
)
from .invitations import BulkInvitationManager, InvitationConfig, InvitationManager
from .models import (
    InMemoryAPIKeyStore,
    InMemoryAuditLogStore,
    InMemoryInvitationStore,
    InMemoryMemberStore,
    InMemoryOrganizationStore,
    InMemorySessionStore,
    InMemoryTeamStore,
    InMemoryUserStore,
)
from .permissions import (
    AuthContext,
    PermissionChecker,
    PolicyEngine,
    ResourceAccessControl,
    create_default_policy_engine,
)


logger = logging.getLogger(__name__)


@dataclass
class OrganizationServiceConfig:
    """Configuration for organization service."""

    # Organization settings
    max_organizations_per_user: int = 10
    default_max_members: int = 10
    slug_min_length: int = 3
    slug_max_length: int = 50

    # Auth config
    auth_config: AuthConfig = field(default_factory=AuthConfig)

    # Invitation config
    invitation_config: InvitationConfig = field(default_factory=InvitationConfig)

    # Audit logging
    enable_audit_logging: bool = True


def generate_slug(name: str) -> str:
    """Generate a URL-safe slug from a name."""
    # Convert to lowercase
    slug = name.lower()
    # Replace spaces and underscores with hyphens
    slug = re.sub(r'[\s_]+', '-', slug)
    # Remove non-alphanumeric characters except hyphens
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    # Remove consecutive hyphens
    slug = re.sub(r'-+', '-', slug)
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    return slug


class OrganizationService:
    """
    Main organization management service.

    Provides high-level operations for managing organizations,
    users, memberships, and authentication.
    """

    def __init__(self, config: Optional[OrganizationServiceConfig] = None):
        """Initialize organization service."""
        self._config = config or OrganizationServiceConfig()
        self._running = False

        # Initialize stores
        self._init_stores()

        # Initialize managers
        self._init_managers()

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

    def _init_stores(self) -> None:
        """Initialize data stores."""
        self._user_store = InMemoryUserStore()
        self._member_store = InMemoryMemberStore()
        self._org_store = InMemoryOrganizationStore(self._member_store)
        self._invitation_store = InMemoryInvitationStore()
        self._team_store = InMemoryTeamStore()
        self._api_key_store = InMemoryAPIKeyStore()
        self._session_store = InMemorySessionStore()
        self._audit_store = InMemoryAuditLogStore()

    def _init_managers(self) -> None:
        """Initialize managers."""
        # Session manager
        self._session_manager = SessionManager(
            self._session_store,
            self._user_store,
            self._config.auth_config,
        )

        # API key manager
        self._api_key_manager = APIKeyManager(
            self._api_key_store,
            self._config.auth_config,
        )

        # Authentication service
        self._auth_service = AuthenticationService(
            self._user_store,
            self._org_store,
            self._member_store,
            self._session_manager,
            self._api_key_manager,
            self._config.auth_config,
        )

        # Invitation manager
        self._invitation_manager = InvitationManager(
            self._invitation_store,
            self._user_store,
            self._org_store,
            self._member_store,
            self._config.invitation_config,
        )

        # Bulk invitation manager
        self._bulk_invitation_manager = BulkInvitationManager(self._invitation_manager)

        # Permission checker
        self._permission_checker = PermissionChecker()

        # Resource access control
        self._resource_access = ResourceAccessControl()

        # Policy engine
        self._policy_engine = create_default_policy_engine()

    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def _emit_event(self, event_type: str, data: Any) -> None:
        """Emit event to handlers."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    async def _audit_log(
        self,
        organization_id: str,
        user_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        description: str = "",
        changes: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Create audit log entry."""
        if not self._config.enable_audit_logging:
            return

        import uuid

        entry = AuditLogEntry(
            id=f"audit_{uuid.uuid4().hex[:16]}",
            organization_id=organization_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            description=description,
            changes=changes,
            ip_address=ip_address,
            success=success,
            error_message=error_message,
        )

        await self._audit_store.create(entry)

    # User Management

    async def register_user(
        self,
        email: str,
        password: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> User:
        """Register a new user."""
        user = await self._auth_service.register_user(
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
        )

        await self._emit_event("user.registered", user)
        return user

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return await self._user_store.get(user_id)

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return await self._user_store.get_by_email(email)

    async def update_user(
        self,
        context: AuthContext,
        user_id: str,
        updates: Dict[str, Any],
    ) -> User:
        """Update user profile."""
        # Users can only update themselves
        if context.user_id != user_id:
            raise AuthorizationError("Cannot update other users")

        user = await self._user_store.get(user_id)
        if not user:
            raise UserNotFoundError(user_id)

        # Apply updates
        allowed_fields = {
            "first_name", "last_name", "display_name",
            "phone_number", "timezone", "locale",
            "notification_preferences", "avatar_url",
        }

        for field, value in updates.items():
            if field in allowed_fields:
                setattr(user, field, value)

        user.updated_at = datetime.utcnow()
        await self._user_store.update(user)

        await self._emit_event("user.updated", user)
        return user

    async def deactivate_user(
        self,
        context: AuthContext,
        user_id: str,
    ) -> User:
        """Deactivate a user account."""
        if context.user_id != user_id:
            raise AuthorizationError("Cannot deactivate other users")

        user = await self._user_store.get(user_id)
        if not user:
            raise UserNotFoundError(user_id)

        user.status = UserStatus.DEACTIVATED
        user.updated_at = datetime.utcnow()
        await self._user_store.update(user)

        # Revoke all sessions
        await self._session_manager.revoke_all_user_sessions(user_id)

        await self._emit_event("user.deactivated", user)
        return user

    # Authentication

    async def login(
        self,
        email: str,
        password: str,
        organization_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[Session, str, User]:
        """Login with email/password."""
        session, token, user = await self._auth_service.authenticate_password(
            email=email,
            password=password,
            organization_id=organization_id,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        await self._emit_event("user.login", {"user": user, "session": session})
        return (session, token, user)

    async def logout(self, session_id: str) -> bool:
        """Logout (revoke session)."""
        return await self._session_manager.revoke_session(session_id)

    async def authenticate_token(self, token: str) -> Optional[AuthContext]:
        """Authenticate with session token."""
        return await self._auth_service.authenticate_token(token)

    async def authenticate_api_key(
        self,
        api_key: str,
        ip_address: Optional[str] = None,
    ) -> Optional[AuthContext]:
        """Authenticate with API key."""
        return await self._auth_service.authenticate_api_key(api_key, ip_address)

    async def switch_organization(
        self,
        context: AuthContext,
        organization_id: str,
    ) -> Optional[Session]:
        """Switch to a different organization."""
        if not context.session:
            raise OrganizationError("Session required")

        return await self._auth_service.switch_organization(
            session_id=context.session.id,
            organization_id=organization_id,
            user_id=context.user_id,
        )

    # Organization Management

    async def create_organization(
        self,
        context: AuthContext,
        name: str,
        slug: Optional[str] = None,
        **kwargs,
    ) -> Organization:
        """Create a new organization."""
        import uuid

        if not context.is_authenticated():
            raise AuthorizationError("Authentication required")

        # Check limit
        user_orgs = await self._org_store.list_for_user(context.user_id)
        if len(user_orgs) >= self._config.max_organizations_per_user:
            raise OrganizationError(
                f"Maximum organizations ({self._config.max_organizations_per_user}) reached"
            )

        # Generate slug
        if not slug:
            slug = generate_slug(name)

        # Validate slug
        if len(slug) < self._config.slug_min_length:
            raise OrganizationError(
                f"Slug must be at least {self._config.slug_min_length} characters"
            )
        if len(slug) > self._config.slug_max_length:
            raise OrganizationError(
                f"Slug must be at most {self._config.slug_max_length} characters"
            )

        # Check slug uniqueness
        existing = await self._org_store.get_by_slug(slug)
        if existing:
            # Append random suffix
            slug = f"{slug}-{uuid.uuid4().hex[:6]}"

        org = Organization(
            id=f"org_{uuid.uuid4().hex[:16]}",
            name=name,
            slug=slug,
            status=OrganizationStatus.ACTIVE,
            owner_id=context.user_id,
            max_members=self._config.default_max_members,
            **kwargs,
        )

        await self._org_store.create(org)

        # Create owner membership
        member = OrganizationMember(
            id=f"member_{uuid.uuid4().hex[:16]}",
            organization_id=org.id,
            user_id=context.user_id,
            role=MemberRole.OWNER,
            accepted=True,
            accepted_at=datetime.utcnow(),
        )
        await self._member_store.create(member)

        await self._audit_log(
            organization_id=org.id,
            user_id=context.user_id,
            action="create",
            resource_type="organization",
            resource_id=org.id,
            description=f"Created organization {name}",
        )

        await self._emit_event("organization.created", org)
        return org

    async def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID."""
        return await self._org_store.get(org_id)

    async def get_organization_by_slug(self, slug: str) -> Optional[Organization]:
        """Get organization by slug."""
        return await self._org_store.get_by_slug(slug)

    async def update_organization(
        self,
        context: AuthContext,
        org_id: str,
        updates: Dict[str, Any],
    ) -> Organization:
        """Update organization."""
        self._permission_checker.require(context, Permission.ORG_UPDATE)

        org = await self._org_store.get(org_id)
        if not org:
            raise OrganizationNotFoundError(org_id)

        old_values = {}
        allowed_fields = {
            "name", "email", "phone", "website",
            "address_line1", "address_line2", "city", "state",
            "postal_code", "country", "logo_url", "primary_color",
            "timezone", "locale", "default_voice", "default_language",
        }

        for field, value in updates.items():
            if field in allowed_fields:
                old_values[field] = getattr(org, field, None)
                setattr(org, field, value)

        # Handle slug separately
        if "slug" in updates and updates["slug"] != org.slug:
            existing = await self._org_store.get_by_slug(updates["slug"])
            if existing and existing.id != org_id:
                raise OrganizationError("Slug already in use")
            old_values["slug"] = org.slug
            org.slug = updates["slug"]

        org.updated_at = datetime.utcnow()
        await self._org_store.update(org)

        await self._audit_log(
            organization_id=org_id,
            user_id=context.user_id,
            action="update",
            resource_type="organization",
            resource_id=org_id,
            description=f"Updated organization {org.name}",
            changes=updates,
            previous_values=old_values,
        )

        await self._emit_event("organization.updated", org)
        return org

    async def delete_organization(
        self,
        context: AuthContext,
        org_id: str,
    ) -> bool:
        """Delete organization (soft delete)."""
        self._permission_checker.require(context, Permission.ORG_DELETE)
        self._policy_engine.require("can_delete_organization", context)

        org = await self._org_store.get(org_id)
        if not org:
            raise OrganizationNotFoundError(org_id)

        org.status = OrganizationStatus.DELETED
        org.updated_at = datetime.utcnow()
        await self._org_store.update(org)

        await self._audit_log(
            organization_id=org_id,
            user_id=context.user_id,
            action="delete",
            resource_type="organization",
            resource_id=org_id,
            description=f"Deleted organization {org.name}",
        )

        await self._emit_event("organization.deleted", org)
        return True

    async def list_user_organizations(
        self,
        user_id: str,
    ) -> List[Organization]:
        """List organizations for a user."""
        return await self._org_store.list_for_user(user_id)

    # Member Management

    async def add_member(
        self,
        context: AuthContext,
        org_id: str,
        user_id: str,
        role: MemberRole = MemberRole.MEMBER,
    ) -> OrganizationMember:
        """Add a member to organization directly."""
        import uuid

        self._permission_checker.require(context, Permission.ORG_MANAGE_MEMBERS)

        # Check role management
        allowed, error = self._permission_checker.can_manage_member(context, role)
        if not allowed:
            raise AuthorizationError(error)

        org = await self._org_store.get(org_id)
        if not org:
            raise OrganizationNotFoundError(org_id)

        user = await self._user_store.get(user_id)
        if not user:
            raise UserNotFoundError(user_id)

        # Check if already member
        existing = await self._member_store.get_by_user_and_org(user_id, org_id)
        if existing:
            raise OrganizationError("User is already a member")

        # Check member limit
        members = await self._member_store.list_for_organization(org_id)
        if len(members) >= org.max_members:
            raise OrganizationError("Maximum members reached")

        member = OrganizationMember(
            id=f"member_{uuid.uuid4().hex[:16]}",
            organization_id=org_id,
            user_id=user_id,
            role=role,
            accepted=True,
            accepted_at=datetime.utcnow(),
        )

        await self._member_store.create(member)

        await self._audit_log(
            organization_id=org_id,
            user_id=context.user_id,
            action="add_member",
            resource_type="member",
            resource_id=member.id,
            description=f"Added {user.email} as {role.value}",
        )

        await self._emit_event("member.added", {"member": member, "organization": org})
        return member

    async def update_member_role(
        self,
        context: AuthContext,
        member_id: str,
        new_role: MemberRole,
    ) -> OrganizationMember:
        """Update member's role."""
        self._permission_checker.require(context, Permission.ORG_MANAGE_MEMBERS)

        member = await self._member_store.get(member_id)
        if not member:
            raise OrganizationError("Member not found")

        # Check if can manage current role
        allowed, error = self._permission_checker.can_manage_member(context, member.role)
        if not allowed:
            raise AuthorizationError(error)

        # Check if can assign new role
        allowed, error = self._permission_checker.can_manage_member(context, new_role)
        if not allowed:
            raise AuthorizationError(error)

        old_role = member.role
        member.role = new_role
        member.updated_at = datetime.utcnow()
        await self._member_store.update(member)

        await self._audit_log(
            organization_id=member.organization_id,
            user_id=context.user_id,
            action="update_member_role",
            resource_type="member",
            resource_id=member_id,
            description=f"Changed role from {old_role.value} to {new_role.value}",
            changes={"role": new_role.value},
            previous_values={"role": old_role.value},
        )

        await self._emit_event("member.role_changed", member)
        return member

    async def remove_member(
        self,
        context: AuthContext,
        member_id: str,
    ) -> bool:
        """Remove member from organization."""
        self._permission_checker.require(context, Permission.ORG_MANAGE_MEMBERS)

        member = await self._member_store.get(member_id)
        if not member:
            return False

        # Cannot remove owner
        if member.role == MemberRole.OWNER:
            raise AuthorizationError("Cannot remove organization owner")

        # Check if can manage this role
        allowed, error = self._permission_checker.can_manage_member(context, member.role)
        if not allowed:
            raise AuthorizationError(error)

        await self._member_store.delete(member_id)

        await self._audit_log(
            organization_id=member.organization_id,
            user_id=context.user_id,
            action="remove_member",
            resource_type="member",
            resource_id=member_id,
            description="Removed member from organization",
        )

        await self._emit_event("member.removed", member)
        return True

    async def leave_organization(
        self,
        context: AuthContext,
        org_id: str,
    ) -> bool:
        """Leave an organization."""
        member = await self._member_store.get_by_user_and_org(context.user_id, org_id)
        if not member:
            raise OrganizationError("Not a member of this organization")

        # Cannot leave if owner
        if member.role == MemberRole.OWNER:
            raise OrganizationError("Owner cannot leave. Transfer ownership first.")

        await self._member_store.delete(member.id)

        await self._audit_log(
            organization_id=org_id,
            user_id=context.user_id,
            action="leave",
            resource_type="member",
            resource_id=member.id,
            description="Left organization",
        )

        await self._emit_event("member.left", member)
        return True

    async def transfer_ownership(
        self,
        context: AuthContext,
        org_id: str,
        new_owner_id: str,
    ) -> bool:
        """Transfer organization ownership."""
        self._policy_engine.require("can_transfer_ownership", context)

        org = await self._org_store.get(org_id)
        if not org:
            raise OrganizationNotFoundError(org_id)

        # Get new owner's membership
        new_owner_member = await self._member_store.get_by_user_and_org(new_owner_id, org_id)
        if not new_owner_member:
            raise OrganizationError("New owner must be an existing member")

        # Get current owner's membership
        current_owner_member = await self._member_store.get_by_user_and_org(context.user_id, org_id)

        # Update roles
        new_owner_member.role = MemberRole.OWNER
        new_owner_member.updated_at = datetime.utcnow()
        await self._member_store.update(new_owner_member)

        current_owner_member.role = MemberRole.ADMIN
        current_owner_member.updated_at = datetime.utcnow()
        await self._member_store.update(current_owner_member)

        # Update organization owner
        org.owner_id = new_owner_id
        org.updated_at = datetime.utcnow()
        await self._org_store.update(org)

        await self._audit_log(
            organization_id=org_id,
            user_id=context.user_id,
            action="transfer_ownership",
            resource_type="organization",
            resource_id=org_id,
            description=f"Transferred ownership to {new_owner_id}",
        )

        await self._emit_event("organization.ownership_transferred", {
            "organization": org,
            "new_owner_id": new_owner_id,
        })

        return True

    async def list_organization_members(
        self,
        context: AuthContext,
        org_id: str,
    ) -> List[Dict[str, Any]]:
        """List organization members with user details."""
        self._permission_checker.require(context, Permission.ORG_READ)

        members = await self._member_store.list_for_organization(org_id)

        result = []
        for member in members:
            user = await self._user_store.get(member.user_id)
            result.append({
                "member_id": member.id,
                "user_id": member.user_id,
                "role": member.role.value,
                "email": user.email if user else None,
                "name": user.full_name if user else None,
                "avatar_url": user.avatar_url if user else None,
                "joined_at": member.joined_at.isoformat(),
            })

        return result

    # Invitations

    async def invite_member(
        self,
        context: AuthContext,
        org_id: str,
        email: str,
        role: MemberRole = MemberRole.MEMBER,
        team_id: Optional[str] = None,
    ):
        """Invite a user to the organization."""
        self._permission_checker.require(context, Permission.ORG_MANAGE_MEMBERS)

        # Check if can assign role
        allowed, error = self._permission_checker.can_manage_member(context, role)
        if not allowed:
            raise AuthorizationError(error)

        return await self._invitation_manager.create_invitation(
            organization_id=org_id,
            email=email,
            role=role,
            invited_by_user_id=context.user_id,
            team_id=team_id,
        )

    async def accept_invitation(
        self,
        token: str,
        user: Optional[User] = None,
    ):
        """Accept an invitation."""
        return await self._invitation_manager.accept_invitation(
            token=token,
            user=user,
            create_user=False,
        )

    # API Keys

    async def create_api_key(
        self,
        context: AuthContext,
        name: str,
        scopes: Optional[List[APIKeyScope]] = None,
        allowed_ips: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[APIKey, str]:
        """Create an API key."""
        self._permission_checker.require(context, Permission.API_KEY_CREATE)

        api_key, key_value = await self._api_key_manager.create_api_key(
            organization_id=context.organization_id,
            name=name,
            created_by_user_id=context.user_id,
            scopes=scopes,
            allowed_ips=allowed_ips,
            expires_in_days=expires_in_days,
        )

        await self._audit_log(
            organization_id=context.organization_id,
            user_id=context.user_id,
            action="create",
            resource_type="api_key",
            resource_id=api_key.id,
            description=f"Created API key: {name}",
        )

        return (api_key, key_value)

    async def list_api_keys(
        self,
        context: AuthContext,
    ) -> List[APIKey]:
        """List API keys for organization."""
        self._permission_checker.require(context, Permission.API_KEY_READ)

        return await self._api_key_manager.list_organization_keys(
            context.organization_id,
            include_revoked=False,
        )

    async def revoke_api_key(
        self,
        context: AuthContext,
        key_id: str,
    ) -> bool:
        """Revoke an API key."""
        self._permission_checker.require(context, Permission.API_KEY_DELETE)

        result = await self._api_key_manager.revoke_api_key(key_id)

        if result:
            await self._audit_log(
                organization_id=context.organization_id,
                user_id=context.user_id,
                action="revoke",
                resource_type="api_key",
                resource_id=key_id,
                description="Revoked API key",
            )

        return result

    # Audit Logs

    async def list_audit_logs(
        self,
        context: AuthContext,
        limit: int = 100,
        offset: int = 0,
        resource_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[AuditLogEntry]:
        """List audit logs for organization."""
        self._permission_checker.require(context, Permission.ORG_READ)

        return await self._audit_store.list_for_organization(
            org_id=context.organization_id,
            limit=limit,
            offset=offset,
            resource_type=resource_type,
            user_id=user_id,
        )


def create_organization_service(
    **kwargs,
) -> OrganizationService:
    """Create organization service with configuration."""
    config = OrganizationServiceConfig(**kwargs)
    return OrganizationService(config)
