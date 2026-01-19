"""
Organization Base Types

Core types and data structures for multi-tenant organization management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class UserStatus(str, Enum):
    """User account status."""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"


class OrganizationStatus(str, Enum):
    """Organization status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class InvitationStatus(str, Enum):
    """Invitation status."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    EXPIRED = "expired"
    REVOKED = "revoked"


class MemberRole(str, Enum):
    """Organization member roles."""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class Permission(str, Enum):
    """System permissions."""
    # Organization management
    ORG_READ = "org:read"
    ORG_UPDATE = "org:update"
    ORG_DELETE = "org:delete"
    ORG_MANAGE_MEMBERS = "org:manage_members"
    ORG_MANAGE_BILLING = "org:manage_billing"

    # Agent management
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    AGENT_DEPLOY = "agent:deploy"

    # Phone number management
    PHONE_CREATE = "phone:create"
    PHONE_READ = "phone:read"
    PHONE_UPDATE = "phone:update"
    PHONE_DELETE = "phone:delete"

    # Call management
    CALL_INITIATE = "call:initiate"
    CALL_READ = "call:read"
    CALL_MANAGE = "call:manage"

    # Knowledge base
    KNOWLEDGE_CREATE = "knowledge:create"
    KNOWLEDGE_READ = "knowledge:read"
    KNOWLEDGE_UPDATE = "knowledge:update"
    KNOWLEDGE_DELETE = "knowledge:delete"

    # Analytics
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_EXPORT = "analytics:export"

    # API keys
    API_KEY_CREATE = "api_key:create"
    API_KEY_READ = "api_key:read"
    API_KEY_DELETE = "api_key:delete"

    # Webhooks
    WEBHOOK_CREATE = "webhook:create"
    WEBHOOK_READ = "webhook:read"
    WEBHOOK_UPDATE = "webhook:update"
    WEBHOOK_DELETE = "webhook:delete"

    # Campaigns
    CAMPAIGN_CREATE = "campaign:create"
    CAMPAIGN_READ = "campaign:read"
    CAMPAIGN_UPDATE = "campaign:update"
    CAMPAIGN_DELETE = "campaign:delete"
    CAMPAIGN_EXECUTE = "campaign:execute"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[MemberRole, Set[Permission]] = {
    MemberRole.OWNER: set(Permission),  # All permissions
    MemberRole.ADMIN: {
        Permission.ORG_READ,
        Permission.ORG_UPDATE,
        Permission.ORG_MANAGE_MEMBERS,
        Permission.ORG_MANAGE_BILLING,
        Permission.AGENT_CREATE,
        Permission.AGENT_READ,
        Permission.AGENT_UPDATE,
        Permission.AGENT_DELETE,
        Permission.AGENT_DEPLOY,
        Permission.PHONE_CREATE,
        Permission.PHONE_READ,
        Permission.PHONE_UPDATE,
        Permission.PHONE_DELETE,
        Permission.CALL_INITIATE,
        Permission.CALL_READ,
        Permission.CALL_MANAGE,
        Permission.KNOWLEDGE_CREATE,
        Permission.KNOWLEDGE_READ,
        Permission.KNOWLEDGE_UPDATE,
        Permission.KNOWLEDGE_DELETE,
        Permission.ANALYTICS_READ,
        Permission.ANALYTICS_EXPORT,
        Permission.API_KEY_CREATE,
        Permission.API_KEY_READ,
        Permission.API_KEY_DELETE,
        Permission.WEBHOOK_CREATE,
        Permission.WEBHOOK_READ,
        Permission.WEBHOOK_UPDATE,
        Permission.WEBHOOK_DELETE,
        Permission.CAMPAIGN_CREATE,
        Permission.CAMPAIGN_READ,
        Permission.CAMPAIGN_UPDATE,
        Permission.CAMPAIGN_DELETE,
        Permission.CAMPAIGN_EXECUTE,
    },
    MemberRole.MEMBER: {
        Permission.ORG_READ,
        Permission.AGENT_CREATE,
        Permission.AGENT_READ,
        Permission.AGENT_UPDATE,
        Permission.PHONE_READ,
        Permission.CALL_INITIATE,
        Permission.CALL_READ,
        Permission.KNOWLEDGE_READ,
        Permission.KNOWLEDGE_UPDATE,
        Permission.ANALYTICS_READ,
        Permission.API_KEY_READ,
        Permission.WEBHOOK_READ,
        Permission.CAMPAIGN_READ,
        Permission.CAMPAIGN_UPDATE,
    },
    MemberRole.VIEWER: {
        Permission.ORG_READ,
        Permission.AGENT_READ,
        Permission.PHONE_READ,
        Permission.CALL_READ,
        Permission.KNOWLEDGE_READ,
        Permission.ANALYTICS_READ,
        Permission.WEBHOOK_READ,
        Permission.CAMPAIGN_READ,
    },
}


class APIKeyScope(str, Enum):
    """API key scopes."""
    ALL = "all"
    AGENTS = "agents"
    CALLS = "calls"
    ANALYTICS = "analytics"
    WEBHOOKS = "webhooks"


@dataclass
class User:
    """User account."""

    id: str
    email: str
    status: UserStatus

    # Profile
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    phone_number: Optional[str] = None

    # Authentication
    password_hash: Optional[str] = None
    email_verified: bool = False
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None

    # External auth
    google_id: Optional[str] = None
    github_id: Optional[str] = None

    # Settings
    timezone: str = "UTC"
    locale: str = "en"
    notification_preferences: Dict[str, bool] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = None
    email_verified_at: Optional[datetime] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.display_name or self.email.split("@")[0]

    def is_active(self) -> bool:
        """Check if user is active."""
        return self.status == UserStatus.ACTIVE


@dataclass
class Organization:
    """Organization/tenant."""

    id: str
    name: str
    slug: str
    status: OrganizationStatus

    # Contact
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None

    # Address
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: str = "US"

    # Branding
    logo_url: Optional[str] = None
    primary_color: Optional[str] = None

    # Settings
    timezone: str = "UTC"
    locale: str = "en"
    default_voice: Optional[str] = None
    default_language: str = "en-US"

    # Limits (can be overridden by subscription)
    max_members: int = 10
    max_agents: int = 5
    max_phone_numbers: int = 3

    # Owner
    owner_id: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if organization is active."""
        return self.status == OrganizationStatus.ACTIVE


@dataclass
class OrganizationMember:
    """Organization membership."""

    id: str
    organization_id: str
    user_id: str
    role: MemberRole

    # Permissions (override role defaults)
    additional_permissions: Set[Permission] = field(default_factory=set)
    revoked_permissions: Set[Permission] = field(default_factory=set)

    # Status
    accepted: bool = True
    accepted_at: Optional[datetime] = None

    # Timestamps
    joined_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_permissions(self) -> Set[Permission]:
        """Get effective permissions for member."""
        base_permissions = ROLE_PERMISSIONS.get(self.role, set())
        return (base_permissions | self.additional_permissions) - self.revoked_permissions

    def has_permission(self, permission: Permission) -> bool:
        """Check if member has permission."""
        return permission in self.get_permissions()


@dataclass
class Team:
    """Team within an organization."""

    id: str
    organization_id: str
    name: str
    description: Optional[str] = None

    # Members
    member_ids: Set[str] = field(default_factory=set)

    # Lead
    lead_user_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Invitation:
    """Organization invitation."""

    id: str
    organization_id: str
    email: str
    role: MemberRole
    status: InvitationStatus

    # Inviter
    invited_by_user_id: str

    # Token for accepting
    token: str
    expires_at: datetime

    # Team assignment
    team_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    accepted_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if invitation is still valid."""
        if self.status != InvitationStatus.PENDING:
            return False
        return datetime.utcnow() < self.expires_at


@dataclass
class APIKey:
    """API key for programmatic access."""

    id: str
    organization_id: str
    name: str

    # Key (only shown once on creation)
    key_prefix: str  # First 8 chars for display
    key_hash: str  # Hashed full key

    # Scope
    scopes: Set[APIKeyScope] = field(default_factory=lambda: {APIKeyScope.ALL})

    # Restrictions
    allowed_ips: Optional[List[str]] = None
    rate_limit_per_minute: Optional[int] = None

    # Status
    active: bool = True
    last_used_at: Optional[datetime] = None
    last_used_ip: Optional[str] = None

    # Expiration
    expires_at: Optional[datetime] = None

    # Creator
    created_by_user_id: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if API key is valid."""
        if not self.active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


@dataclass
class Session:
    """User session."""

    id: str
    user_id: str
    organization_id: Optional[str]

    # Token
    token_hash: str
    refresh_token_hash: Optional[str] = None

    # Device info
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    device_type: Optional[str] = None

    # Location
    country: Optional[str] = None
    city: Optional[str] = None

    # Status
    active: bool = True
    revoked_at: Optional[datetime] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))

    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if session is valid."""
        if not self.active:
            return False
        return datetime.utcnow() < self.expires_at


@dataclass
class AuditLogEntry:
    """Audit log entry for tracking actions."""

    id: str
    organization_id: str
    user_id: Optional[str]

    # Action
    action: str
    resource_type: str
    resource_id: Optional[str]

    # Details
    description: str
    changes: Optional[Dict[str, Any]] = None
    previous_values: Optional[Dict[str, Any]] = None

    # Context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    api_key_id: Optional[str] = None

    # Status
    success: bool = True
    error_message: Optional[str] = None

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    metadata: Dict[str, Any] = field(default_factory=dict)


# Abstract interfaces

class UserStore(ABC):
    """Abstract user storage interface."""

    @abstractmethod
    async def create(self, user: User) -> None:
        """Create user."""
        pass

    @abstractmethod
    async def get(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        pass

    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        pass

    @abstractmethod
    async def update(self, user: User) -> None:
        """Update user."""
        pass

    @abstractmethod
    async def delete(self, user_id: str) -> None:
        """Delete user."""
        pass


class OrganizationStore(ABC):
    """Abstract organization storage interface."""

    @abstractmethod
    async def create(self, org: Organization) -> None:
        """Create organization."""
        pass

    @abstractmethod
    async def get(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID."""
        pass

    @abstractmethod
    async def get_by_slug(self, slug: str) -> Optional[Organization]:
        """Get organization by slug."""
        pass

    @abstractmethod
    async def update(self, org: Organization) -> None:
        """Update organization."""
        pass

    @abstractmethod
    async def delete(self, org_id: str) -> None:
        """Delete organization."""
        pass

    @abstractmethod
    async def list_for_user(self, user_id: str) -> List[Organization]:
        """List organizations for user."""
        pass


class MemberStore(ABC):
    """Abstract membership storage interface."""

    @abstractmethod
    async def create(self, member: OrganizationMember) -> None:
        """Create membership."""
        pass

    @abstractmethod
    async def get(self, member_id: str) -> Optional[OrganizationMember]:
        """Get membership by ID."""
        pass

    @abstractmethod
    async def get_by_user_and_org(
        self,
        user_id: str,
        org_id: str,
    ) -> Optional[OrganizationMember]:
        """Get membership by user and organization."""
        pass

    @abstractmethod
    async def update(self, member: OrganizationMember) -> None:
        """Update membership."""
        pass

    @abstractmethod
    async def delete(self, member_id: str) -> None:
        """Delete membership."""
        pass

    @abstractmethod
    async def list_for_organization(self, org_id: str) -> List[OrganizationMember]:
        """List members for organization."""
        pass


class InvitationStore(ABC):
    """Abstract invitation storage interface."""

    @abstractmethod
    async def create(self, invitation: Invitation) -> None:
        """Create invitation."""
        pass

    @abstractmethod
    async def get(self, invitation_id: str) -> Optional[Invitation]:
        """Get invitation by ID."""
        pass

    @abstractmethod
    async def get_by_token(self, token: str) -> Optional[Invitation]:
        """Get invitation by token."""
        pass

    @abstractmethod
    async def update(self, invitation: Invitation) -> None:
        """Update invitation."""
        pass

    @abstractmethod
    async def list_for_organization(
        self,
        org_id: str,
        status: Optional[InvitationStatus] = None,
    ) -> List[Invitation]:
        """List invitations for organization."""
        pass


# Errors

class OrganizationError(Exception):
    """Base organization error."""

    def __init__(self, message: str, code: str = "organization_error"):
        self.message = message
        self.code = code
        super().__init__(message)


class AuthenticationError(OrganizationError):
    """Authentication error."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "authentication_error")


class AuthorizationError(OrganizationError):
    """Authorization error."""

    def __init__(self, message: str = "Permission denied"):
        super().__init__(message, "authorization_error")


class UserNotFoundError(OrganizationError):
    """User not found error."""

    def __init__(self, user_id: str):
        super().__init__(f"User not found: {user_id}", "user_not_found")


class OrganizationNotFoundError(OrganizationError):
    """Organization not found error."""

    def __init__(self, org_id: str):
        super().__init__(f"Organization not found: {org_id}", "organization_not_found")


class InvitationError(OrganizationError):
    """Invitation error."""

    def __init__(self, message: str):
        super().__init__(message, "invitation_error")
