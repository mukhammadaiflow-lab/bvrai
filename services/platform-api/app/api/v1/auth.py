"""
Authentication API Routes

Handles:
- User registration and login
- Token management
- API key management
- Password reset
- OAuth flows
"""

from typing import Optional, List
from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_db_session,
    get_current_user,
    get_redis,
    UserContext,
    RateLimitDep,
)

router = APIRouter(prefix="/auth")


# ============================================================================
# Schemas
# ============================================================================

class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    name: str = Field(..., min_length=1, max_length=100)
    company: Optional[str] = None


class LoginRequest(BaseModel):
    """Login request."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


class UserResponse(BaseModel):
    """User response."""
    id: str
    email: str
    name: str
    tenant_id: str
    roles: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation."""
    token: str
    new_password: str = Field(..., min_length=8)


class ChangePasswordRequest(BaseModel):
    """Change password request."""
    current_password: str
    new_password: str = Field(..., min_length=8)


class APIKeyCreate(BaseModel):
    """API key creation request."""
    name: str = Field(..., min_length=1, max_length=100)
    scopes: List[str] = []
    expires_at: Optional[datetime] = None


class APIKeyResponse(BaseModel):
    """API key response."""
    id: str
    name: str
    key_prefix: str
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]


class APIKeyCreatedResponse(APIKeyResponse):
    """API key created response (includes full key)."""
    key: str  # Only shown once at creation


# ============================================================================
# Rate Limiters
# ============================================================================

login_rate_limit = RateLimitDep(requests=10, window=60, scope="login")
register_rate_limit = RateLimitDep(requests=5, window=300, scope="register")


# ============================================================================
# Registration & Login
# ============================================================================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    data: RegisterRequest,
    db: AsyncSession = Depends(get_db_session),
    _rate_limit: None = Depends(register_rate_limit),
):
    """
    Register a new user.

    Creates a new user account and tenant.
    """
    from app.auth.service import AuthService

    service = AuthService(db)

    # Check if email already exists
    existing = await service.get_user_by_email(data.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = await service.register(
        email=data.email,
        password=data.password,
        name=data.name,
        company=data.company,
    )

    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        tenant_id=str(user.tenant_id),
        roles=user.roles or [],
        created_at=user.created_at,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    data: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    _rate_limit: None = Depends(login_rate_limit),
):
    """
    Authenticate and get access tokens.

    Returns JWT access and refresh tokens.
    """
    from app.auth.service import AuthService

    service = AuthService(db)

    user = await service.authenticate(data.email, data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Generate tokens
    tokens = await service.create_tokens(user)

    # Update last login
    await service.update_last_login(user.id, request.client.host)

    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=tokens["expires_in"],
    )


@router.post("/login/oauth2", response_model=TokenResponse)
async def login_oauth2(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """OAuth2 compatible login endpoint."""
    from app.auth.service import AuthService

    service = AuthService(db)

    user = await service.authenticate(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    tokens = await service.create_tokens(user)

    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=tokens["expires_in"],
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Refresh access token.

    Exchange a valid refresh token for new access and refresh tokens.
    """
    from app.auth.service import AuthService

    service = AuthService(db)

    try:
        tokens = await service.refresh_tokens(data.refresh_token)
        return TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            expires_in=tokens["expires_in"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    user: UserContext = Depends(get_current_user),
    cache = Depends(get_redis),
):
    """
    Logout current session.

    Invalidates the current access token.
    """
    # Add token to blacklist
    await cache.setex(
        f"token:blacklist:{user.user_id}",
        3600,  # 1 hour
        "1",
    )


# ============================================================================
# Password Management
# ============================================================================

@router.post("/password/reset", status_code=status.HTTP_204_NO_CONTENT)
async def request_password_reset(
    data: PasswordResetRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Request password reset.

    Sends a password reset email if the email exists.
    Always returns success to prevent email enumeration.
    """
    from app.auth.service import AuthService

    service = AuthService(db)
    await service.request_password_reset(data.email)

    # Always return success to prevent email enumeration


@router.post("/password/reset/confirm", status_code=status.HTTP_204_NO_CONTENT)
async def confirm_password_reset(
    data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Confirm password reset.

    Reset password using the token from the reset email.
    """
    from app.auth.service import AuthService

    service = AuthService(db)

    success = await service.reset_password(data.token, data.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )


@router.post("/password/change", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    data: ChangePasswordRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Change password.

    Requires current password for verification.
    """
    from app.auth.service import AuthService

    service = AuthService(db)

    success = await service.change_password(
        user_id=user.user_id,
        current_password=data.current_password,
        new_password=data.new_password,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )


# ============================================================================
# User Profile
# ============================================================================

@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Get current user profile."""
    from app.auth.service import AuthService

    service = AuthService(db)
    user_data = await service.get_user(user.user_id)

    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserResponse(
        id=str(user_data.id),
        email=user_data.email,
        name=user_data.name,
        tenant_id=str(user_data.tenant_id),
        roles=user_data.roles or [],
        created_at=user_data.created_at,
        last_login=user_data.last_login,
    )


# ============================================================================
# API Keys
# ============================================================================

@router.post("/api-keys", response_model=APIKeyCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    data: APIKeyCreate,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Create a new API key.

    The full API key is only shown once at creation.
    Store it securely as it cannot be retrieved later.
    """
    from app.auth.service import AuthService

    service = AuthService(db)

    key_data = await service.create_api_key(
        user_id=user.user_id,
        tenant_id=user.tenant_id,
        name=data.name,
        scopes=data.scopes,
        expires_at=data.expires_at,
    )

    return APIKeyCreatedResponse(
        id=key_data["id"],
        name=key_data["name"],
        key=key_data["key"],  # Full key only shown once
        key_prefix=key_data["key_prefix"],
        scopes=key_data["scopes"],
        created_at=key_data["created_at"],
        expires_at=key_data.get("expires_at"),
        last_used_at=None,
    )


@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """List all API keys for the current user."""
    from app.auth.service import AuthService

    service = AuthService(db)
    keys = await service.list_api_keys(user.user_id)

    return [
        APIKeyResponse(
            id=str(k.id),
            name=k.name,
            key_prefix=k.key_prefix,
            scopes=k.scopes or [],
            created_at=k.created_at,
            expires_at=k.expires_at,
            last_used_at=k.last_used_at,
        )
        for k in keys
    ]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: UUID,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete an API key."""
    from app.auth.service import AuthService

    service = AuthService(db)
    deleted = await service.delete_api_key(str(key_id), user.user_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )


# ============================================================================
# OAuth Providers
# ============================================================================

@router.get("/oauth/{provider}")
async def oauth_redirect(
    provider: str,
    request: Request,
):
    """
    Initiate OAuth flow.

    Redirects to the OAuth provider's authorization page.
    """
    from app.auth.oauth import get_oauth_provider

    oauth = get_oauth_provider(provider)
    if not oauth:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown OAuth provider: {provider}",
        )

    redirect_uri = str(request.url_for("oauth_callback", provider=provider))
    return await oauth.get_authorization_redirect(redirect_uri)


@router.get("/oauth/{provider}/callback", response_model=TokenResponse)
async def oauth_callback(
    provider: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """
    OAuth callback handler.

    Handles the OAuth provider's callback and creates/links the user.
    """
    from app.auth.oauth import get_oauth_provider
    from app.auth.service import AuthService

    oauth = get_oauth_provider(provider)
    if not oauth:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown OAuth provider: {provider}",
        )

    # Exchange code for user info
    user_info = await oauth.handle_callback(request)

    # Create or link user
    service = AuthService(db)
    user = await service.oauth_login(provider, user_info)

    # Generate tokens
    tokens = await service.create_tokens(user)

    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=tokens["expires_in"],
    )
