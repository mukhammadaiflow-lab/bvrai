"""
Organization Package

This package provides comprehensive multi-tenant organization management
for the voice agent platform, including:

- User account management
- Organization and team management
- Role-based access control (RBAC)
- Authentication (sessions and API keys)
- Member invitations
- Audit logging

Example usage:

    from bvrai_core.organization import (
        OrganizationService,
        OrganizationServiceConfig,
        MemberRole,
        Permission,
    )

    # Create organization service
    service = OrganizationService()

    # Register user
    user = await service.register_user(
        email="user@example.com",
        password="SecurePass123",
        first_name="John",
        last_name="Doe",
    )

    # Login
    session, token, user = await service.login(
        email="user@example.com",
        password="SecurePass123",
    )

    # Authenticate with token
    context = await service.authenticate_token(token)

    # Create organization
    org = await service.create_organization(
        context=context,
        name="Acme Corp",
    )

    # Invite member
    invitation = await service.invite_member(
        context=context,
        org_id=org.id,
        email="colleague@example.com",
        role=MemberRole.MEMBER,
    )

    # Create API key
    api_key, key_value = await service.create_api_key(
        context=context,
        name="Production API Key",
    )
"""

# Base types
from .base import (
    UserStatus,
    OrganizationStatus,
    InvitationStatus,
    MemberRole,
    Permission,
    ROLE_PERMISSIONS,
    APIKeyScope,
    User,
    Organization,
    OrganizationMember,
    Team,
    Invitation,
    APIKey,
    Session,
    AuditLogEntry,
    UserStore,
    OrganizationStore,
    MemberStore,
    InvitationStore,
    OrganizationError,
    AuthenticationError,
    AuthorizationError,
    UserNotFoundError,
    OrganizationNotFoundError,
    InvitationError,
)

# Models (in-memory stores)
from .models import (
    InMemoryUserStore,
    InMemoryOrganizationStore,
    InMemoryMemberStore,
    InMemoryInvitationStore,
    InMemoryTeamStore,
    InMemoryAPIKeyStore,
    InMemorySessionStore,
    InMemoryAuditLogStore,
)

# Permissions
from .permissions import (
    SCOPE_PERMISSIONS,
    AuthContext,
    PermissionChecker,
    require_permission,
    require_resource_permission,
    ResourceAccessControl,
    PolicyEngine,
    create_default_policy_engine,
)

# Authentication
from .auth import (
    hash_password,
    verify_password,
    generate_token,
    hash_token,
    generate_api_key,
    AuthConfig,
    PasswordValidator,
    SessionManager,
    APIKeyManager,
    AuthenticationService,
)

# Invitations
from .invitations import (
    InvitationConfig,
    InvitationManager,
    BulkInvitationManager,
)

# Service
from .service import (
    OrganizationServiceConfig,
    generate_slug,
    OrganizationService,
    create_organization_service,
)


__all__ = [
    # Base types
    "UserStatus",
    "OrganizationStatus",
    "InvitationStatus",
    "MemberRole",
    "Permission",
    "ROLE_PERMISSIONS",
    "APIKeyScope",
    "User",
    "Organization",
    "OrganizationMember",
    "Team",
    "Invitation",
    "APIKey",
    "Session",
    "AuditLogEntry",
    "UserStore",
    "OrganizationStore",
    "MemberStore",
    "InvitationStore",
    "OrganizationError",
    "AuthenticationError",
    "AuthorizationError",
    "UserNotFoundError",
    "OrganizationNotFoundError",
    "InvitationError",
    # Models
    "InMemoryUserStore",
    "InMemoryOrganizationStore",
    "InMemoryMemberStore",
    "InMemoryInvitationStore",
    "InMemoryTeamStore",
    "InMemoryAPIKeyStore",
    "InMemorySessionStore",
    "InMemoryAuditLogStore",
    # Permissions
    "SCOPE_PERMISSIONS",
    "AuthContext",
    "PermissionChecker",
    "require_permission",
    "require_resource_permission",
    "ResourceAccessControl",
    "PolicyEngine",
    "create_default_policy_engine",
    # Authentication
    "hash_password",
    "verify_password",
    "generate_token",
    "hash_token",
    "generate_api_key",
    "AuthConfig",
    "PasswordValidator",
    "SessionManager",
    "APIKeyManager",
    "AuthenticationService",
    # Invitations
    "InvitationConfig",
    "InvitationManager",
    "BulkInvitationManager",
    # Service
    "OrganizationServiceConfig",
    "generate_slug",
    "OrganizationService",
    "create_organization_service",
]
