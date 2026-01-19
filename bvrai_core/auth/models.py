"""Authentication and authorization data models."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


class IdentityProviderType(str, Enum):
    """Types of identity providers."""
    SAML = "saml"
    OIDC = "oidc"
    OAUTH2 = "oauth2"
    LDAP = "ldap"


class MemberRole(str, Enum):
    """Organization member roles."""
    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    VIEWER = "viewer"


class InvitationStatus(str, Enum):
    """Invitation status."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    EXPIRED = "expired"
    REVOKED = "revoked"


class AuditAction(str, Enum):
    """Types of audit actions."""
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_LOGIN_FAILED = "user.login_failed"
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_PASSWORD_CHANGED = "user.password_changed"
    USER_PASSWORD_RESET = "user.password_reset"
    USER_MFA_ENABLED = "user.mfa_enabled"
    USER_MFA_DISABLED = "user.mfa_disabled"

    ORG_CREATED = "org.created"
    ORG_UPDATED = "org.updated"
    ORG_DELETED = "org.deleted"

    MEMBER_ADDED = "member.added"
    MEMBER_REMOVED = "member.removed"
    MEMBER_ROLE_CHANGED = "member.role_changed"

    IDP_CREATED = "idp.created"
    IDP_UPDATED = "idp.updated"
    IDP_DELETED = "idp.deleted"

    API_KEY_CREATED = "api_key.created"
    API_KEY_REVOKED = "api_key.revoked"

    SESSION_CREATED = "session.created"
    SESSION_REVOKED = "session.revoked"

    SSO_LOGIN = "sso.login"
    SSO_LOGOUT = "sso.logout"


@dataclass
class SAMLConfig:
    """SAML identity provider configuration."""
    entity_id: str  # IdP entity ID
    sso_url: str  # Single Sign-On URL
    slo_url: str = ""  # Single Logout URL (optional)
    certificate: str = ""  # IdP X.509 certificate

    # SP configuration
    sp_entity_id: str = ""  # Service Provider entity ID
    acs_url: str = ""  # Assertion Consumer Service URL

    # Attribute mapping
    email_attribute: str = "email"
    first_name_attribute: str = "firstName"
    last_name_attribute: str = "lastName"
    groups_attribute: str = "groups"

    # Options
    sign_requests: bool = True
    want_assertions_signed: bool = True
    want_assertions_encrypted: bool = False
    signature_algorithm: str = "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OIDCConfig:
    """OpenID Connect identity provider configuration."""
    issuer: str  # IdP issuer URL
    client_id: str
    client_secret: str = ""  # Encrypted

    # Endpoints (auto-discovered if issuer supports .well-known)
    authorization_endpoint: str = ""
    token_endpoint: str = ""
    userinfo_endpoint: str = ""
    jwks_uri: str = ""
    end_session_endpoint: str = ""

    # Scopes
    scopes: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])

    # Attribute mapping
    email_claim: str = "email"
    name_claim: str = "name"
    groups_claim: str = "groups"

    # Options
    use_pkce: bool = True
    response_type: str = "code"

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "client_secret": "***" if self.client_secret else "",
        }


@dataclass
class IdentityProvider:
    """Identity provider configuration."""
    id: str
    organization_id: str
    name: str
    provider_type: IdentityProviderType

    # Configuration (one of these based on provider_type)
    saml_config: Optional[SAMLConfig] = None
    oidc_config: Optional[OIDCConfig] = None

    # Settings
    is_active: bool = True
    is_primary: bool = False
    auto_provision_users: bool = True
    default_role: MemberRole = MemberRole.VIEWER

    # Domain restrictions
    allowed_domains: List[str] = field(default_factory=list)  # e.g., ["company.com"]

    # Group to role mapping
    group_role_mapping: Dict[str, MemberRole] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "provider_type": self.provider_type.value,
            "saml_config": self.saml_config.to_dict() if self.saml_config else None,
            "oidc_config": self.oidc_config.to_dict() if self.oidc_config else None,
            "is_active": self.is_active,
            "is_primary": self.is_primary,
            "auto_provision_users": self.auto_provision_users,
            "default_role": self.default_role.value,
            "allowed_domains": self.allowed_domains,
            "group_role_mapping": {k: v.value for k, v in self.group_role_mapping.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "metadata": self.metadata,
        }


@dataclass
class User:
    """User account."""
    id: str
    email: str

    # Profile
    first_name: str = ""
    last_name: str = ""
    display_name: str = ""
    avatar_url: str = ""
    phone: str = ""
    timezone: str = "UTC"
    locale: str = "en"

    # Authentication
    password_hash: str = ""  # Empty for SSO-only users
    mfa_enabled: bool = False
    mfa_secret: str = ""  # Encrypted TOTP secret
    mfa_backup_codes: List[str] = field(default_factory=list)  # Encrypted

    # SSO
    external_id: str = ""  # ID from identity provider
    identity_provider_id: str = ""  # Which IdP authenticated this user

    # Status
    is_active: bool = True
    is_verified: bool = False
    is_system_admin: bool = False

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = None
    password_changed_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "display_name": self.display_name or f"{self.first_name} {self.last_name}".strip(),
            "avatar_url": self.avatar_url,
            "phone": self.phone,
            "timezone": self.timezone,
            "locale": self.locale,
            "mfa_enabled": self.mfa_enabled,
            "external_id": self.external_id,
            "identity_provider_id": self.identity_provider_id,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "is_system_admin": self.is_system_admin,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "metadata": self.metadata,
        }

        if include_sensitive:
            data["password_hash"] = self.password_hash
            data["mfa_secret"] = self.mfa_secret
            data["mfa_backup_codes"] = self.mfa_backup_codes

        return data

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


@dataclass
class Organization:
    """Organization/tenant."""
    id: str
    name: str
    slug: str  # URL-safe identifier

    # Settings
    plan: str = "free"  # free, starter, professional, enterprise
    max_seats: int = 5

    # SSO settings
    require_sso: bool = False  # Require SSO for all members
    allowed_email_domains: List[str] = field(default_factory=list)

    # Security settings
    enforce_mfa: bool = False
    session_timeout_hours: int = 24
    ip_allowlist: List[str] = field(default_factory=list)

    # Branding
    logo_url: str = ""
    primary_color: str = ""

    # Status
    is_active: bool = True

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Billing
    billing_email: str = ""
    stripe_customer_id: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "plan": self.plan,
            "max_seats": self.max_seats,
            "require_sso": self.require_sso,
            "allowed_email_domains": self.allowed_email_domains,
            "enforce_mfa": self.enforce_mfa,
            "session_timeout_hours": self.session_timeout_hours,
            "ip_allowlist": self.ip_allowlist,
            "logo_url": self.logo_url,
            "primary_color": self.primary_color,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "billing_email": self.billing_email,
            "metadata": self.metadata,
        }


@dataclass
class OrganizationMember:
    """Organization membership."""
    id: str
    organization_id: str
    user_id: str

    role: MemberRole = MemberRole.VIEWER

    # Permissions (for fine-grained access control)
    permissions: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True

    # Timestamps
    joined_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Who added this member
    invited_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "role": self.role.value,
            "permissions": self.permissions,
            "is_active": self.is_active,
            "joined_at": self.joined_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "invited_by": self.invited_by,
        }

    def has_permission(self, permission: str) -> bool:
        """Check if member has a specific permission."""
        # Owners and admins have all permissions
        if self.role in (MemberRole.OWNER, MemberRole.ADMIN):
            return True
        return permission in self.permissions


@dataclass
class Invitation:
    """Organization invitation."""
    id: str
    organization_id: str
    email: str
    role: MemberRole = MemberRole.VIEWER

    status: InvitationStatus = InvitationStatus.PENDING

    # Token for accepting invitation
    token: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=7))
    accepted_at: Optional[datetime] = None

    # Who sent this invitation
    invited_by: str = ""

    # Message to include in invitation email
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "email": self.email,
            "role": self.role.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "invited_by": self.invited_by,
            "message": self.message,
        }

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


@dataclass
class Session:
    """User session."""
    id: str
    user_id: str
    organization_id: str

    # Session token (hashed)
    token_hash: str = ""

    # Refresh token (hashed)
    refresh_token_hash: str = ""

    # Client info
    ip_address: str = ""
    user_agent: str = ""
    device_id: str = ""
    device_name: str = ""

    # Location (from IP)
    country: str = ""
    city: str = ""

    # SSO info
    identity_provider_id: str = ""
    sso_session_id: str = ""  # IdP session ID for SLO

    # Status
    is_active: bool = True

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))
    last_activity_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "country": self.country,
            "city": self.city,
            "identity_provider_id": self.identity_provider_id,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
        }

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


@dataclass
class AuthToken:
    """Authentication token (JWT or API key)."""
    id: str
    user_id: str
    organization_id: str

    # Token type
    token_type: str = "access"  # access, refresh, api_key

    # For API keys
    name: str = ""  # User-defined name
    prefix: str = ""  # First few chars for identification
    token_hash: str = ""  # Hashed token

    # Scopes/permissions
    scopes: List[str] = field(default_factory=list)

    # Limits (for API keys)
    rate_limit: int = 0  # Requests per minute, 0 = default
    allowed_ips: List[str] = field(default_factory=list)

    # Status
    is_active: bool = True

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "token_type": self.token_type,
            "name": self.name,
            "prefix": self.prefix,
            "scopes": self.scopes,
            "rate_limit": self.rate_limit,
            "allowed_ips": self.allowed_ips,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
        }


@dataclass
class SSOLogin:
    """SSO login attempt/session."""
    id: str
    organization_id: str
    identity_provider_id: str

    # Request state
    state: str = ""  # CSRF token
    nonce: str = ""  # For OIDC
    redirect_uri: str = ""

    # SAML-specific
    saml_request_id: str = ""

    # Status
    status: str = "pending"  # pending, completed, failed
    error_message: str = ""

    # Result
    user_id: str = ""
    session_id: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=10))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "identity_provider_id": self.identity_provider_id,
            "state": self.state,
            "redirect_uri": self.redirect_uri,
            "status": self.status,
            "error_message": self.error_message,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "expires_at": self.expires_at.isoformat(),
        }


@dataclass
class AuditLog:
    """Audit log entry."""
    id: str
    organization_id: str

    # Actor
    user_id: str = ""
    user_email: str = ""

    # Action
    action: AuditAction = AuditAction.USER_LOGIN

    # Target
    resource_type: str = ""  # user, org, agent, etc.
    resource_id: str = ""

    # Details
    details: Dict[str, Any] = field(default_factory=dict)

    # Changes (for updates)
    old_values: Dict[str, Any] = field(default_factory=dict)
    new_values: Dict[str, Any] = field(default_factory=dict)

    # Request info
    ip_address: str = ""
    user_agent: str = ""
    request_id: str = ""

    # Result
    success: bool = True
    error_message: str = ""

    # Timestamp
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "action": self.action.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "old_values": self.old_values,
            "new_values": self.new_values,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "success": self.success,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
        }
