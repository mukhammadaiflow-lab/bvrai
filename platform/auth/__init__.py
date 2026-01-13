"""Enterprise SSO/SAML authentication system."""

from .models import (
    User,
    Organization,
    OrganizationMember,
    IdentityProvider,
    IdentityProviderType,
    SAMLConfig,
    OIDCConfig,
    Session,
    AuthToken,
    SSOLogin,
    AuditLog,
    AuditAction,
    MemberRole,
    InvitationStatus,
    Invitation,
)
from .saml import SAMLProvider, SAMLResponse, SAMLAssertion
from .oidc import OIDCProvider, OIDCTokens
from .manager import AuthManager
from .session import SessionManager
from .routes import router as auth_router, init_routes

__all__ = [
    # Models
    "User",
    "Organization",
    "OrganizationMember",
    "IdentityProvider",
    "IdentityProviderType",
    "SAMLConfig",
    "OIDCConfig",
    "Session",
    "AuthToken",
    "SSOLogin",
    "AuditLog",
    "AuditAction",
    "MemberRole",
    "InvitationStatus",
    "Invitation",
    # SAML
    "SAMLProvider",
    "SAMLResponse",
    "SAMLAssertion",
    # OIDC
    "OIDCProvider",
    "OIDCTokens",
    # Services
    "AuthManager",
    "SessionManager",
    # Routes
    "auth_router",
    "init_routes",
]
