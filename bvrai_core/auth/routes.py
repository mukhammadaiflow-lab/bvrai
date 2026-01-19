"""API routes for authentication."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, BackgroundTasks
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, EmailStr

from .models import (
    User,
    Organization,
    OrganizationMember,
    IdentityProvider,
    IdentityProviderType,
    SAMLConfig,
    OIDCConfig,
    MemberRole,
    AuditAction,
)
from .manager import AuthManager
from .session import SessionManager


router = APIRouter(prefix="/auth", tags=["Authentication"])

# Dependency injection
_auth_manager: Optional[AuthManager] = None
_session_manager: Optional[SessionManager] = None


def get_auth_manager() -> AuthManager:
    if not _auth_manager:
        raise HTTPException(status_code=503, detail="Auth manager not initialized")
    return _auth_manager


def get_session_manager() -> SessionManager:
    if not _session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    return _session_manager


def init_routes(
    auth_manager: AuthManager,
    session_manager: SessionManager,
) -> None:
    """Initialize route dependencies."""
    global _auth_manager, _session_manager
    _auth_manager = auth_manager
    _session_manager = session_manager


# Request/Response Models

class LoginRequest(BaseModel):
    """Password login request."""
    email: EmailStr
    password: str = Field(..., min_length=1)
    remember_me: bool = False


class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: str = ""
    last_name: str = ""
    organization_name: str = ""


class PasswordChangeRequest(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    first_name: str
    last_name: str
    display_name: str
    avatar_url: str
    mfa_enabled: bool
    is_active: bool
    is_verified: bool
    created_at: str
    last_login_at: Optional[str]

    @classmethod
    def from_user(cls, user: User) -> "UserResponse":
        return cls(
            id=user.id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            display_name=user.display_name or user.full_name,
            avatar_url=user.avatar_url,
            mfa_enabled=user.mfa_enabled,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at.isoformat(),
            last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
        )


class SessionResponse(BaseModel):
    """Session response model."""
    id: str
    ip_address: str
    user_agent: str
    device_name: str
    country: str
    city: str
    is_current: bool
    created_at: str
    last_activity_at: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: UserResponse


class OrganizationCreate(BaseModel):
    """Organization creation request."""
    name: str = Field(..., min_length=1, max_length=100)
    slug: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-z0-9-]+$")


class OrganizationResponse(BaseModel):
    """Organization response model."""
    id: str
    name: str
    slug: str
    plan: str
    max_seats: int
    require_sso: bool
    enforce_mfa: bool
    is_active: bool
    created_at: str

    @classmethod
    def from_organization(cls, org: Organization) -> "OrganizationResponse":
        return cls(
            id=org.id,
            name=org.name,
            slug=org.slug,
            plan=org.plan,
            max_seats=org.max_seats,
            require_sso=org.require_sso,
            enforce_mfa=org.enforce_mfa,
            is_active=org.is_active,
            created_at=org.created_at.isoformat(),
        )


class MemberResponse(BaseModel):
    """Member response model."""
    id: str
    user_id: str
    role: str
    joined_at: str
    user: Optional[UserResponse] = None


class InviteMemberRequest(BaseModel):
    """Member invitation request."""
    email: EmailStr
    role: str = "viewer"
    message: str = ""


class SAMLConfigCreate(BaseModel):
    """SAML configuration request."""
    entity_id: str = Field(..., description="IdP Entity ID")
    sso_url: str = Field(..., description="Single Sign-On URL")
    slo_url: str = ""
    certificate: str = Field(..., description="IdP X.509 Certificate (PEM format)")
    email_attribute: str = "email"
    first_name_attribute: str = "firstName"
    last_name_attribute: str = "lastName"
    groups_attribute: str = "groups"
    sign_requests: bool = True
    want_assertions_signed: bool = True


class OIDCConfigCreate(BaseModel):
    """OIDC configuration request."""
    issuer: str = Field(..., description="IdP Issuer URL")
    client_id: str
    client_secret: str = ""
    scopes: List[str] = Field(default=["openid", "profile", "email"])
    email_claim: str = "email"
    name_claim: str = "name"
    groups_claim: str = "groups"
    use_pkce: bool = True


class IdentityProviderCreate(BaseModel):
    """Identity provider creation request."""
    name: str = Field(..., min_length=1, max_length=100)
    provider_type: str = Field(..., description="saml or oidc")
    saml_config: Optional[SAMLConfigCreate] = None
    oidc_config: Optional[OIDCConfigCreate] = None
    is_primary: bool = False
    auto_provision_users: bool = True
    default_role: str = "viewer"
    allowed_domains: List[str] = Field(default_factory=list)
    group_role_mapping: Dict[str, str] = Field(default_factory=dict)


class IdentityProviderResponse(BaseModel):
    """Identity provider response model."""
    id: str
    name: str
    provider_type: str
    is_active: bool
    is_primary: bool
    auto_provision_users: bool
    default_role: str
    allowed_domains: List[str]
    created_at: str
    last_login_at: Optional[str]

    @classmethod
    def from_idp(cls, idp: IdentityProvider) -> "IdentityProviderResponse":
        return cls(
            id=idp.id,
            name=idp.name,
            provider_type=idp.provider_type.value,
            is_active=idp.is_active,
            is_primary=idp.is_primary,
            auto_provision_users=idp.auto_provision_users,
            default_role=idp.default_role.value,
            allowed_domains=idp.allowed_domains,
            created_at=idp.created_at.isoformat(),
            last_login_at=idp.last_login_at.isoformat() if idp.last_login_at else None,
        )


class SSOInitiateRequest(BaseModel):
    """SSO initiation request."""
    redirect_uri: str = ""


class APIKeyCreate(BaseModel):
    """API key creation request."""
    name: str = Field(..., min_length=1, max_length=100)
    scopes: List[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = None
    rate_limit: int = 0
    allowed_ips: List[str] = Field(default_factory=list)


class APIKeyResponse(BaseModel):
    """API key response model."""
    id: str
    name: str
    prefix: str
    scopes: List[str]
    rate_limit: int
    is_active: bool
    created_at: str
    expires_at: Optional[str]
    last_used_at: Optional[str]
    key: Optional[str] = None  # Only included on creation


class AuditLogResponse(BaseModel):
    """Audit log response model."""
    id: str
    user_id: str
    user_email: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: str
    success: bool
    created_at: str


# Authentication Routes

@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    req: Request,
    auth_manager: AuthManager = Depends(get_auth_manager),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TokenResponse:
    """Authenticate with email and password."""
    ip_address = req.client.host if req.client else ""
    user_agent = req.headers.get("user-agent", "")

    user, error = await auth_manager.authenticate_with_password(
        email=request.email,
        password=request.password,
        ip_address=ip_address,
        user_agent=user_agent,
    )

    if not user:
        raise HTTPException(status_code=401, detail=error or "Authentication failed")

    # Get user's default organization
    # For simplicity, use first org membership
    memberships = []  # Would need to implement get_user_memberships

    org_id = memberships[0].organization_id if memberships else ""

    # Create session
    session, session_token, refresh_token = await session_manager.create_session(
        user=user,
        organization_id=org_id,
        ip_address=ip_address,
        user_agent=user_agent,
        session_timeout_hours=24 * 30 if request.remember_me else 24,
    )

    # Generate access token
    access_token = session_manager.generate_access_token(
        user_id=user.id,
        organization_id=org_id,
        session_id=session.id,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=session_manager.access_token_ttl_minutes * 60,
        user=UserResponse.from_user(user),
    )


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    request: RegisterRequest,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> UserResponse:
    """Register a new user."""
    try:
        user = await auth_manager.create_user(
            email=request.email,
            password=request.password,
            first_name=request.first_name,
            last_name=request.last_name,
        )

        # Create organization if name provided
        if request.organization_name:
            slug = request.organization_name.lower().replace(" ", "-")
            await auth_manager.create_organization(
                name=request.organization_name,
                slug=slug,
                owner_id=user.id,
            )

        return UserResponse.from_user(user)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_token: str,
    session_manager: SessionManager = Depends(get_session_manager),
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> TokenResponse:
    """Refresh access token."""
    session, new_session_token, new_refresh_token = await session_manager.refresh_session(
        refresh_token=refresh_token,
    )

    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    user = await auth_manager.get_user(session.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    access_token = session_manager.generate_access_token(
        user_id=user.id,
        organization_id=session.organization_id,
        session_id=session.id,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=session_manager.access_token_ttl_minutes * 60,
        user=UserResponse.from_user(user),
    )


@router.post("/logout")
async def logout(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> Dict[str, str]:
    """Logout and revoke session."""
    await session_manager.revoke_session(session_id)
    return {"status": "logged out"}


@router.post("/logout/all")
async def logout_all(
    user_id: str,
    except_current: str = "",
    session_manager: SessionManager = Depends(get_session_manager),
) -> Dict[str, int]:
    """Logout from all sessions."""
    count = await session_manager.revoke_all_sessions(user_id, except_session_id=except_current)
    return {"revoked_sessions": count}


# Password Routes

@router.post("/password/change")
async def change_password(
    user_id: str,
    request: PasswordChangeRequest,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> Dict[str, str]:
    """Change password."""
    try:
        success = await auth_manager.change_password(
            user_id=user_id,
            current_password=request.current_password,
            new_password=request.new_password,
        )

        if not success:
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        return {"status": "password changed"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# SSO Routes

@router.post("/sso/{organization_id}/initiate")
async def initiate_sso(
    organization_id: str,
    identity_provider_id: str,
    request: SSOInitiateRequest,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> Dict[str, str]:
    """Initiate SSO login flow."""
    try:
        redirect_url, sso_login = await auth_manager.initiate_sso_login(
            organization_id=organization_id,
            identity_provider_id=identity_provider_id,
            redirect_uri=request.redirect_uri,
        )

        return {
            "redirect_url": redirect_url,
            "state": sso_login.state,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/saml/callback")
async def saml_callback(
    SAMLResponse: str,
    RelayState: str = "",
    req: Request = None,
    auth_manager: AuthManager = Depends(get_auth_manager),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TokenResponse:
    """Handle SAML assertion callback."""
    ip_address = req.client.host if req and req.client else ""
    user_agent = req.headers.get("user-agent", "") if req else ""

    user, membership, error = await auth_manager.complete_saml_login(
        saml_response=SAMLResponse,
        relay_state=RelayState,
        ip_address=ip_address,
        user_agent=user_agent,
    )

    if not user:
        raise HTTPException(status_code=401, detail=error or "SAML authentication failed")

    # Create session
    session, session_token, refresh_token = await session_manager.create_session(
        user=user,
        organization_id=membership.organization_id if membership else "",
        ip_address=ip_address,
        user_agent=user_agent,
    )

    access_token = session_manager.generate_access_token(
        user_id=user.id,
        organization_id=membership.organization_id if membership else "",
        session_id=session.id,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=session_manager.access_token_ttl_minutes * 60,
        user=UserResponse.from_user(user),
    )


@router.get("/oidc/callback")
async def oidc_callback(
    code: str,
    state: str,
    req: Request,
    auth_manager: AuthManager = Depends(get_auth_manager),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TokenResponse:
    """Handle OIDC callback."""
    ip_address = req.client.host if req.client else ""
    user_agent = req.headers.get("user-agent", "")

    user, membership, error = await auth_manager.complete_oidc_login(
        code=code,
        state=state,
        ip_address=ip_address,
        user_agent=user_agent,
    )

    if not user:
        raise HTTPException(status_code=401, detail=error or "OIDC authentication failed")

    # Create session
    session, session_token, refresh_token = await session_manager.create_session(
        user=user,
        organization_id=membership.organization_id if membership else "",
        ip_address=ip_address,
        user_agent=user_agent,
    )

    access_token = session_manager.generate_access_token(
        user_id=user.id,
        organization_id=membership.organization_id if membership else "",
        session_id=session.id,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=session_manager.access_token_ttl_minutes * 60,
        user=UserResponse.from_user(user),
    )


# User Routes

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    user_id: str,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> UserResponse:
    """Get current user profile."""
    user = await auth_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse.from_user(user)


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    user_id: str,
    updates: Dict[str, Any],
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> UserResponse:
    """Update current user profile."""
    # Only allow certain fields to be updated
    allowed_fields = {"first_name", "last_name", "display_name", "avatar_url", "phone", "timezone", "locale"}
    filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}

    user = await auth_manager.update_user(user_id, filtered_updates)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse.from_user(user)


@router.get("/me/sessions", response_model=List[SessionResponse])
async def list_user_sessions(
    user_id: str,
    current_session_id: str = "",
    session_manager: SessionManager = Depends(get_session_manager),
) -> List[SessionResponse]:
    """List user's active sessions."""
    sessions = await session_manager.get_user_sessions(user_id)

    return [
        SessionResponse(
            id=s.id,
            ip_address=s.ip_address,
            user_agent=s.user_agent,
            device_name=s.device_name,
            country=s.country,
            city=s.city,
            is_current=s.id == current_session_id,
            created_at=s.created_at.isoformat(),
            last_activity_at=s.last_activity_at.isoformat(),
        )
        for s in sessions
    ]


# Organization Routes

@router.post("/organizations", response_model=OrganizationResponse, status_code=201)
async def create_organization(
    user_id: str,
    request: OrganizationCreate,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> OrganizationResponse:
    """Create a new organization."""
    try:
        org = await auth_manager.create_organization(
            name=request.name,
            slug=request.slug,
            owner_id=user_id,
        )
        return OrganizationResponse.from_organization(org)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/organizations/{organization_id}", response_model=OrganizationResponse)
async def get_organization(
    organization_id: str,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> OrganizationResponse:
    """Get organization details."""
    org = await auth_manager.get_organization(organization_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return OrganizationResponse.from_organization(org)


@router.get("/organizations/{organization_id}/members", response_model=List[MemberResponse])
async def list_organization_members(
    organization_id: str,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> List[MemberResponse]:
    """List organization members."""
    members = await auth_manager.list_members(organization_id)

    responses = []
    for member in members:
        user = await auth_manager.get_user(member.user_id)
        responses.append(MemberResponse(
            id=member.id,
            user_id=member.user_id,
            role=member.role.value,
            joined_at=member.joined_at.isoformat(),
            user=UserResponse.from_user(user) if user else None,
        ))

    return responses


@router.post("/organizations/{organization_id}/members", response_model=MemberResponse, status_code=201)
async def add_organization_member(
    organization_id: str,
    request: InviteMemberRequest,
    invited_by: str = "",
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> MemberResponse:
    """Invite a member to organization."""
    # Get or create user
    user = await auth_manager.get_user_by_email(request.email)

    if not user:
        # Create user without password (will need to set via invite)
        user = await auth_manager.create_user(
            email=request.email,
            is_verified=False,
        )

    try:
        role = MemberRole(request.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}")

    member = await auth_manager.add_member(
        organization_id=organization_id,
        user_id=user.id,
        role=role,
        invited_by=invited_by,
    )

    return MemberResponse(
        id=member.id,
        user_id=member.user_id,
        role=member.role.value,
        joined_at=member.joined_at.isoformat(),
        user=UserResponse.from_user(user),
    )


@router.patch("/organizations/{organization_id}/members/{user_id}")
async def update_member_role(
    organization_id: str,
    user_id: str,
    role: str,
    updated_by: str = "",
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> MemberResponse:
    """Update member role."""
    try:
        new_role = MemberRole(role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {role}")

    member = await auth_manager.update_member_role(
        organization_id=organization_id,
        user_id=user_id,
        new_role=new_role,
        updated_by=updated_by,
    )

    if not member:
        raise HTTPException(status_code=404, detail="Member not found")

    user = await auth_manager.get_user(user_id)

    return MemberResponse(
        id=member.id,
        user_id=member.user_id,
        role=member.role.value,
        joined_at=member.joined_at.isoformat(),
        user=UserResponse.from_user(user) if user else None,
    )


@router.delete("/organizations/{organization_id}/members/{user_id}", status_code=204)
async def remove_organization_member(
    organization_id: str,
    user_id: str,
    removed_by: str = "",
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> None:
    """Remove a member from organization."""
    removed = await auth_manager.remove_member(
        organization_id=organization_id,
        user_id=user_id,
        removed_by=removed_by,
    )

    if not removed:
        raise HTTPException(status_code=404, detail="Member not found")


# Identity Provider Routes

@router.get("/organizations/{organization_id}/identity-providers", response_model=List[IdentityProviderResponse])
async def list_identity_providers(
    organization_id: str,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> List[IdentityProviderResponse]:
    """List identity providers for organization."""
    idps = await auth_manager.list_identity_providers(organization_id)
    return [IdentityProviderResponse.from_idp(idp) for idp in idps]


@router.post("/organizations/{organization_id}/identity-providers", response_model=IdentityProviderResponse, status_code=201)
async def create_identity_provider(
    organization_id: str,
    request: IdentityProviderCreate,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> IdentityProviderResponse:
    """Create an identity provider."""
    try:
        provider_type = IdentityProviderType(request.provider_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid provider type: {request.provider_type}")

    saml_config = None
    if request.saml_config:
        saml_config = SAMLConfig(
            entity_id=request.saml_config.entity_id,
            sso_url=request.saml_config.sso_url,
            slo_url=request.saml_config.slo_url,
            certificate=request.saml_config.certificate,
            email_attribute=request.saml_config.email_attribute,
            first_name_attribute=request.saml_config.first_name_attribute,
            last_name_attribute=request.saml_config.last_name_attribute,
            groups_attribute=request.saml_config.groups_attribute,
            sign_requests=request.saml_config.sign_requests,
            want_assertions_signed=request.saml_config.want_assertions_signed,
        )

    oidc_config = None
    if request.oidc_config:
        oidc_config = OIDCConfig(
            issuer=request.oidc_config.issuer,
            client_id=request.oidc_config.client_id,
            client_secret=request.oidc_config.client_secret,
            scopes=request.oidc_config.scopes,
            email_claim=request.oidc_config.email_claim,
            name_claim=request.oidc_config.name_claim,
            groups_claim=request.oidc_config.groups_claim,
            use_pkce=request.oidc_config.use_pkce,
        )

    try:
        default_role = MemberRole(request.default_role)
    except ValueError:
        default_role = MemberRole.VIEWER

    group_role_mapping = {}
    for group, role in request.group_role_mapping.items():
        try:
            group_role_mapping[group] = MemberRole(role)
        except ValueError:
            continue

    idp = await auth_manager.create_identity_provider(
        organization_id=organization_id,
        name=request.name,
        provider_type=provider_type,
        saml_config=saml_config,
        oidc_config=oidc_config,
        is_primary=request.is_primary,
        auto_provision_users=request.auto_provision_users,
        default_role=default_role,
        allowed_domains=request.allowed_domains,
        group_role_mapping=group_role_mapping,
    )

    return IdentityProviderResponse.from_idp(idp)


@router.get("/organizations/{organization_id}/identity-providers/{idp_id}", response_model=IdentityProviderResponse)
async def get_identity_provider(
    organization_id: str,
    idp_id: str,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> IdentityProviderResponse:
    """Get identity provider details."""
    idp = await auth_manager.get_identity_provider(idp_id)
    if not idp or idp.organization_id != organization_id:
        raise HTTPException(status_code=404, detail="Identity provider not found")
    return IdentityProviderResponse.from_idp(idp)


@router.delete("/organizations/{organization_id}/identity-providers/{idp_id}", status_code=204)
async def delete_identity_provider(
    organization_id: str,
    idp_id: str,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> None:
    """Delete an identity provider."""
    idp = await auth_manager.get_identity_provider(idp_id)
    if not idp or idp.organization_id != organization_id:
        raise HTTPException(status_code=404, detail="Identity provider not found")

    await auth_manager.delete_identity_provider(idp_id)


# API Key Routes

@router.get("/organizations/{organization_id}/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    organization_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> List[APIKeyResponse]:
    """List API keys for organization."""
    keys = await session_manager.list_api_keys(organization_id)

    return [
        APIKeyResponse(
            id=k.id,
            name=k.name,
            prefix=k.prefix,
            scopes=k.scopes,
            rate_limit=k.rate_limit,
            is_active=k.is_active,
            created_at=k.created_at.isoformat(),
            expires_at=k.expires_at.isoformat() if k.expires_at else None,
            last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
        )
        for k in keys
    ]


@router.post("/organizations/{organization_id}/api-keys", response_model=APIKeyResponse, status_code=201)
async def create_api_key(
    organization_id: str,
    user_id: str,
    request: APIKeyCreate,
    session_manager: SessionManager = Depends(get_session_manager),
) -> APIKeyResponse:
    """Create an API key."""
    auth_token, raw_key = await session_manager.create_api_key(
        user_id=user_id,
        organization_id=organization_id,
        name=request.name,
        scopes=request.scopes,
        expires_in_days=request.expires_in_days,
        rate_limit=request.rate_limit,
        allowed_ips=request.allowed_ips,
    )

    return APIKeyResponse(
        id=auth_token.id,
        name=auth_token.name,
        prefix=auth_token.prefix,
        scopes=auth_token.scopes,
        rate_limit=auth_token.rate_limit,
        is_active=auth_token.is_active,
        created_at=auth_token.created_at.isoformat(),
        expires_at=auth_token.expires_at.isoformat() if auth_token.expires_at else None,
        last_used_at=None,
        key=raw_key,  # Only returned on creation
    )


@router.delete("/organizations/{organization_id}/api-keys/{key_id}", status_code=204)
async def revoke_api_key(
    organization_id: str,
    key_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> None:
    """Revoke an API key."""
    revoked = await session_manager.revoke_api_key(organization_id, key_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="API key not found")


# Audit Log Routes

@router.get("/organizations/{organization_id}/audit-logs", response_model=List[AuditLogResponse])
async def get_audit_logs(
    organization_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    action: Optional[str] = None,
    user_id: Optional[str] = None,
    auth_manager: AuthManager = Depends(get_auth_manager),
) -> List[AuditLogResponse]:
    """Get audit logs for organization."""
    audit_action = None
    if action:
        try:
            audit_action = AuditAction(action)
        except ValueError:
            pass

    logs = await auth_manager.get_audit_logs(
        organization_id=organization_id,
        limit=limit,
        offset=offset,
        action=audit_action,
        user_id=user_id or "",
    )

    return [
        AuditLogResponse(
            id=log.id,
            user_id=log.user_id,
            user_email=log.user_email,
            action=log.action.value,
            resource_type=log.resource_type,
            resource_id=log.resource_id,
            details=log.details,
            ip_address=log.ip_address,
            success=log.success,
            created_at=log.created_at.isoformat(),
        )
        for log in logs
    ]
