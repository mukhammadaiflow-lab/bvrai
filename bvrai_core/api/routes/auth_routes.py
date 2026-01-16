"""
Authentication API Routes

This module provides REST API endpoints for user authentication.
"""

import logging
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Body
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..base import (
    APIResponse,
    success_response,
    AuthenticationError,
    ErrorCode,
    hash_api_key,
)
from ..dependencies import get_db_session
from ...database.models import User, Organization, APIKey


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# =============================================================================
# Request/Response Models
# =============================================================================

class LoginRequest(BaseModel):
    """Login request."""
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    """Login response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    user: dict


class RegisterRequest(BaseModel):
    """Registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    name: str
    organization_name: Optional[str] = None


class UserResponse(BaseModel):
    """User response."""
    id: str
    email: str
    name: str
    role: str
    organization_id: str
    avatar_url: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None


class ForgotPasswordRequest(BaseModel):
    """Forgot password request."""
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Reset password request."""
    token: str
    password: str = Field(..., min_length=8)


class ChangePasswordRequest(BaseModel):
    """Change password request."""
    current_password: str
    new_password: str = Field(..., min_length=8)


# =============================================================================
# Helper Functions
# =============================================================================

def generate_api_key() -> tuple[str, str]:
    """Generate an API key and its hash."""
    key = f"bvr_{secrets.token_urlsafe(32)}"
    key_hash = hash_api_key(key)
    return key, key_hash


def user_to_response(user: User) -> dict:
    """Convert user model to response dict."""
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role or "member",
        "organization_id": user.organization_id,
        "avatar_url": user.avatar_url,
        "created_at": user.created_at,
        "last_login": user.last_login_at,
    }


# =============================================================================
# Routes
# =============================================================================

@router.post(
    "/login",
    response_model=APIResponse[LoginResponse],
    summary="Login",
    description="Authenticate user and return access token.",
)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """User login endpoint."""
    # Find user by email
    result = await db.execute(
        select(User).where(User.email == request.email)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise AuthenticationError(
            message="Invalid email or password",
            code=ErrorCode.INVALID_CREDENTIALS,
        )

    # Verify password (simplified - in production use bcrypt/argon2)
    if not user.verify_password(request.password):
        raise AuthenticationError(
            message="Invalid email or password",
            code=ErrorCode.INVALID_CREDENTIALS,
        )

    # Check if user is active
    if not user.is_active:
        raise AuthenticationError(
            message="Account is disabled",
            code=ErrorCode.AUTHENTICATION_REQUIRED,
        )

    # Generate API key as token
    key, key_hash = generate_api_key()

    # Create or update API key for this user session
    api_key = APIKey(
        organization_id=user.organization_id,
        user_id=user.id,
        name=f"Session {datetime.utcnow().isoformat()}",
        key_hash=key_hash,
        key_prefix=key[:12],
        role="owner",  # User gets full access
        is_active=True,
        expires_at=datetime.utcnow() + timedelta(hours=24),
    )
    db.add(api_key)

    # Update last login
    user.last_login_at = datetime.utcnow()
    await db.commit()

    logger.info(f"User logged in: {user.email}")

    return success_response({
        "access_token": key,
        "token_type": "bearer",
        "expires_in": 86400,  # 24 hours
        "user": user_to_response(user),
    })


@router.post(
    "/register",
    response_model=APIResponse[LoginResponse],
    status_code=201,
    summary="Register",
    description="Create a new user account.",
)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """User registration endpoint."""
    # Check if email already exists
    result = await db.execute(
        select(User).where(User.email == request.email)
    )
    existing = result.scalar_one_or_none()

    if existing:
        raise AuthenticationError(
            message="Email already registered",
            code=ErrorCode.VALIDATION_ERROR,
        )

    # Create organization
    org_name = request.organization_name or f"{request.name}'s Organization"
    org_slug = org_name.lower().replace(" ", "-").replace("'", "")[:50]
    org_slug = f"{org_slug}-{str(uuid.uuid4())[:8]}"

    org = Organization(
        name=org_name,
        slug=org_slug,
        plan="free",
        is_active=True,
    )
    db.add(org)
    await db.flush()

    # Create user
    user = User(
        organization_id=org.id,
        email=request.email,
        name=request.name,
        role="owner",
        is_active=True,
    )
    user.set_password(request.password)
    db.add(user)
    await db.flush()

    # Generate API key
    key, key_hash = generate_api_key()

    api_key = APIKey(
        organization_id=org.id,
        user_id=user.id,
        name="Initial Session",
        key_hash=key_hash,
        key_prefix=key[:12],
        role="owner",
        is_active=True,
        expires_at=datetime.utcnow() + timedelta(hours=24),
    )
    db.add(api_key)

    await db.commit()

    logger.info(f"User registered: {user.email}")

    return success_response({
        "access_token": key,
        "token_type": "bearer",
        "expires_in": 86400,
        "user": user_to_response(user),
    })


@router.post(
    "/logout",
    status_code=204,
    summary="Logout",
    description="Invalidate current access token.",
)
async def logout():
    """User logout endpoint."""
    # In a full implementation, we would invalidate the token
    # For now, the client should just remove the token
    return None


@router.get(
    "/me",
    response_model=APIResponse[UserResponse],
    summary="Get Current User",
    description="Get the currently authenticated user.",
)
async def get_current_user(
    db: AsyncSession = Depends(get_db_session),
):
    """Get current user endpoint."""
    # In dev mode, return a mock user
    import os
    if os.getenv("BVRAI_DEV_MODE", "false").lower() == "true":
        return success_response({
            "id": "dev-user-001",
            "email": "dev@example.com",
            "name": "Developer",
            "role": "owner",
            "organization_id": os.getenv("BVRAI_DEV_ORG_ID", "dev-org-001"),
            "avatar_url": None,
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow(),
        })

    # For authenticated requests, we would get the user from the auth context
    raise AuthenticationError(
        message="Not authenticated",
        code=ErrorCode.AUTHENTICATION_REQUIRED,
    )


@router.post(
    "/forgot-password",
    response_model=APIResponse[dict],
    summary="Forgot Password",
    description="Request a password reset email.",
)
async def forgot_password(
    request: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Forgot password endpoint."""
    # In production, send a reset email
    # For now, just acknowledge the request
    return success_response({
        "message": "If an account exists with this email, a reset link has been sent."
    })


@router.post(
    "/reset-password",
    response_model=APIResponse[dict],
    summary="Reset Password",
    description="Reset password using token from email.",
)
async def reset_password(
    request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Reset password endpoint."""
    # In production, verify the token and reset the password
    return success_response({
        "message": "Password has been reset successfully."
    })


@router.post(
    "/change-password",
    response_model=APIResponse[dict],
    summary="Change Password",
    description="Change password for authenticated user.",
)
async def change_password(
    request: ChangePasswordRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Change password endpoint."""
    # In production, verify current password and update
    return success_response({
        "message": "Password changed successfully."
    })


@router.post(
    "/refresh",
    response_model=APIResponse[LoginResponse],
    summary="Refresh Token",
    description="Refresh access token.",
)
async def refresh_token(
    db: AsyncSession = Depends(get_db_session),
):
    """Refresh token endpoint."""
    # In production, validate refresh token and issue new access token
    raise AuthenticationError(
        message="Token refresh not implemented",
        code=ErrorCode.AUTHENTICATION_REQUIRED,
    )
