"""
Authentication System

User authentication, session management, and API key handling.
"""

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    APIKey,
    APIKeyScope,
    AuthenticationError,
    MemberRole,
    Organization,
    OrganizationMember,
    Session,
    User,
    UserStatus,
)
from .models import (
    InMemoryAPIKeyStore,
    InMemoryMemberStore,
    InMemoryOrganizationStore,
    InMemorySessionStore,
    InMemoryUserStore,
)
from .permissions import AuthContext


logger = logging.getLogger(__name__)


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash a password using PBKDF2.

    Returns:
        Tuple of (hash, salt)
    """
    if salt is None:
        salt = secrets.token_hex(32)

    hash_bytes = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000,  # iterations
    )

    return hash_bytes.hex(), salt


def verify_password(password: str, hash_value: str, salt: str) -> bool:
    """Verify a password against its hash."""
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, hash_value)


def generate_token(length: int = 32) -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(length)


def hash_token(token: str) -> str:
    """Hash a token for storage."""
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def generate_api_key() -> Tuple[str, str, str]:
    """
    Generate an API key.

    Returns:
        Tuple of (full_key, prefix, hash)
    """
    key = f"bvr_{secrets.token_urlsafe(32)}"
    prefix = key[:12]
    key_hash = hash_token(key)
    return key, prefix, key_hash


@dataclass
class AuthConfig:
    """Authentication configuration."""

    # Session settings
    session_expiry_days: int = 30
    session_refresh_threshold_days: int = 7

    # Token settings
    access_token_expiry_minutes: int = 60
    refresh_token_expiry_days: int = 30

    # API key settings
    api_key_expiry_days: Optional[int] = None

    # Password requirements
    min_password_length: int = 8
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digit: bool = True
    require_special: bool = False

    # MFA settings
    mfa_issuer: str = "BuilderEngine"

    # Rate limiting
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15


class PasswordValidator:
    """Password validation."""

    def __init__(self, config: AuthConfig):
        """Initialize validator."""
        self._config = config

    def validate(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate a password.

        Returns:
            Tuple of (valid, list of error messages)
        """
        errors = []

        if len(password) < self._config.min_password_length:
            errors.append(f"Password must be at least {self._config.min_password_length} characters")

        if self._config.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain an uppercase letter")

        if self._config.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain a lowercase letter")

        if self._config.require_digit and not any(c.isdigit() for c in password):
            errors.append("Password must contain a digit")

        if self._config.require_special:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                errors.append("Password must contain a special character")

        return (len(errors) == 0, errors)


class SessionManager:
    """
    Manager for user sessions.

    Handles session creation, validation, and revocation.
    """

    def __init__(
        self,
        session_store: InMemorySessionStore,
        user_store: InMemoryUserStore,
        config: AuthConfig,
    ):
        """Initialize session manager."""
        self._sessions = session_store
        self._users = user_store
        self._config = config

    async def create_session(
        self,
        user: User,
        organization_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[Session, str]:
        """
        Create a new session for user.

        Returns:
            Tuple of (session, token)
        """
        import uuid

        token = generate_token()
        refresh_token = generate_token()

        session = Session(
            id=f"sess_{uuid.uuid4().hex[:16]}",
            user_id=user.id,
            organization_id=organization_id,
            token_hash=hash_token(token),
            refresh_token_hash=hash_token(refresh_token),
            user_agent=user_agent,
            ip_address=ip_address,
            expires_at=datetime.utcnow() + timedelta(days=self._config.session_expiry_days),
        )

        await self._sessions.create(session)

        # Update user last login
        user.last_login_at = datetime.utcnow()
        await self._users.update(user)

        return session, token

    async def validate_session(self, token: str) -> Optional[Session]:
        """Validate a session token."""
        token_hash = hash_token(token)
        session = await self._sessions.get_by_token(token_hash)

        if not session or not session.is_valid():
            return None

        # Update last active
        session.last_active_at = datetime.utcnow()
        await self._sessions.update(session)

        return session

    async def refresh_session(
        self,
        session_id: str,
        refresh_token: str,
    ) -> Tuple[Optional[Session], Optional[str]]:
        """
        Refresh a session with new token.

        Returns:
            Tuple of (updated session, new token) or (None, None)
        """
        session = await self._sessions.get(session_id)
        if not session:
            return (None, None)

        # Verify refresh token
        if session.refresh_token_hash != hash_token(refresh_token):
            return (None, None)

        # Generate new tokens
        new_token = generate_token()
        new_refresh_token = generate_token()

        session.token_hash = hash_token(new_token)
        session.refresh_token_hash = hash_token(new_refresh_token)
        session.expires_at = datetime.utcnow() + timedelta(days=self._config.session_expiry_days)
        session.last_active_at = datetime.utcnow()

        await self._sessions.update(session)

        return (session, new_token)

    async def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        session = await self._sessions.get(session_id)
        if not session:
            return False

        session.active = False
        session.revoked_at = datetime.utcnow()
        await self._sessions.update(session)

        return True

    async def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user."""
        return await self._sessions.revoke_all_for_user(user_id)

    async def list_user_sessions(self, user_id: str) -> List[Session]:
        """List active sessions for a user."""
        sessions = await self._sessions.list_for_user(user_id)
        return [s for s in sessions if s.is_valid()]


class APIKeyManager:
    """
    Manager for API keys.

    Handles API key creation, validation, and revocation.
    """

    def __init__(
        self,
        api_key_store: InMemoryAPIKeyStore,
        config: AuthConfig,
    ):
        """Initialize API key manager."""
        self._keys = api_key_store
        self._config = config

    async def create_api_key(
        self,
        organization_id: str,
        name: str,
        created_by_user_id: str,
        scopes: Optional[List[APIKeyScope]] = None,
        allowed_ips: Optional[List[str]] = None,
        rate_limit_per_minute: Optional[int] = None,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[APIKey, str]:
        """
        Create a new API key.

        Returns:
            Tuple of (api_key, full_key_value)
            Note: full_key_value is only returned once!
        """
        import uuid

        key, prefix, key_hash = generate_api_key()

        expires_at = None
        if expires_in_days or self._config.api_key_expiry_days:
            days = expires_in_days or self._config.api_key_expiry_days
            expires_at = datetime.utcnow() + timedelta(days=days)

        api_key = APIKey(
            id=f"key_{uuid.uuid4().hex[:12]}",
            organization_id=organization_id,
            name=name,
            key_prefix=prefix,
            key_hash=key_hash,
            scopes=set(scopes) if scopes else {APIKeyScope.ALL},
            allowed_ips=allowed_ips,
            rate_limit_per_minute=rate_limit_per_minute,
            expires_at=expires_at,
            created_by_user_id=created_by_user_id,
        )

        await self._keys.create(api_key)

        return (api_key, key)

    async def validate_api_key(
        self,
        key: str,
        ip_address: Optional[str] = None,
    ) -> Optional[APIKey]:
        """
        Validate an API key.

        Returns:
            APIKey if valid, None otherwise
        """
        key_hash = hash_token(key)
        api_key = await self._keys.get_by_hash(key_hash)

        if not api_key or not api_key.is_valid():
            return None

        # Check IP restriction
        if api_key.allowed_ips and ip_address:
            if ip_address not in api_key.allowed_ips:
                logger.warning(f"API key {api_key.id} used from unauthorized IP: {ip_address}")
                return None

        # Update last used
        api_key.last_used_at = datetime.utcnow()
        api_key.last_used_ip = ip_address
        await self._keys.update(api_key)

        return api_key

    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        api_key = await self._keys.get(key_id)
        if not api_key:
            return False

        api_key.active = False
        await self._keys.update(api_key)

        return True

    async def list_organization_keys(
        self,
        organization_id: str,
        include_revoked: bool = False,
    ) -> List[APIKey]:
        """List API keys for organization."""
        keys = await self._keys.list_for_organization(organization_id)

        if not include_revoked:
            keys = [k for k in keys if k.active]

        return keys


class AuthenticationService:
    """
    Main authentication service.

    Handles user authentication and context building.
    """

    def __init__(
        self,
        user_store: InMemoryUserStore,
        org_store: InMemoryOrganizationStore,
        member_store: InMemoryMemberStore,
        session_manager: SessionManager,
        api_key_manager: APIKeyManager,
        config: Optional[AuthConfig] = None,
    ):
        """Initialize authentication service."""
        self._users = user_store
        self._orgs = org_store
        self._members = member_store
        self._sessions = session_manager
        self._api_keys = api_key_manager
        self._config = config or AuthConfig()

        self._password_validator = PasswordValidator(self._config)

        # Failed login tracking
        self._failed_attempts: Dict[str, List[datetime]] = {}

    def _check_lockout(self, email: str) -> bool:
        """Check if account is locked out."""
        attempts = self._failed_attempts.get(email, [])

        # Remove old attempts
        cutoff = datetime.utcnow() - timedelta(minutes=self._config.lockout_duration_minutes)
        attempts = [a for a in attempts if a > cutoff]
        self._failed_attempts[email] = attempts

        return len(attempts) >= self._config.max_login_attempts

    def _record_failed_attempt(self, email: str) -> None:
        """Record a failed login attempt."""
        if email not in self._failed_attempts:
            self._failed_attempts[email] = []
        self._failed_attempts[email].append(datetime.utcnow())

    def _clear_failed_attempts(self, email: str) -> None:
        """Clear failed attempts after successful login."""
        self._failed_attempts.pop(email, None)

    async def authenticate_password(
        self,
        email: str,
        password: str,
        organization_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[Session, str, User]:
        """
        Authenticate user with email/password.

        Returns:
            Tuple of (session, token, user)

        Raises:
            AuthenticationError: If authentication fails
        """
        # Check lockout
        if self._check_lockout(email):
            raise AuthenticationError("Account temporarily locked due to too many failed attempts")

        # Find user
        user = await self._users.get_by_email(email)
        if not user:
            self._record_failed_attempt(email)
            raise AuthenticationError("Invalid email or password")

        # Check status
        if not user.is_active():
            raise AuthenticationError("Account is not active")

        # Verify password
        if not user.password_hash:
            raise AuthenticationError("Password login not enabled")

        # Password format: hash:salt
        parts = user.password_hash.split(":")
        if len(parts) != 2:
            raise AuthenticationError("Invalid password format")

        if not verify_password(password, parts[0], parts[1]):
            self._record_failed_attempt(email)
            raise AuthenticationError("Invalid email or password")

        self._clear_failed_attempts(email)

        # Check email verification
        if not user.email_verified:
            raise AuthenticationError("Email not verified")

        # Create session
        session, token = await self._sessions.create_session(
            user=user,
            organization_id=organization_id,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        return (session, token, user)

    async def authenticate_token(
        self,
        token: str,
    ) -> Optional[AuthContext]:
        """
        Authenticate with session token.

        Returns:
            AuthContext if valid, None otherwise
        """
        session = await self._sessions.validate_session(token)
        if not session:
            return None

        return await self._build_context_from_session(session)

    async def authenticate_api_key(
        self,
        api_key: str,
        ip_address: Optional[str] = None,
    ) -> Optional[AuthContext]:
        """
        Authenticate with API key.

        Returns:
            AuthContext if valid, None otherwise
        """
        key = await self._api_keys.validate_api_key(api_key, ip_address)
        if not key:
            return None

        return await self._build_context_from_api_key(key, ip_address)

    async def _build_context_from_session(
        self,
        session: Session,
    ) -> AuthContext:
        """Build auth context from session."""
        user = await self._users.get(session.user_id)

        context = AuthContext(
            user=user,
            user_id=session.user_id,
            session=session,
            ip_address=session.ip_address,
            user_agent=session.user_agent,
        )

        # Add organization context if set
        if session.organization_id:
            org = await self._orgs.get(session.organization_id)
            if org:
                context.organization = org
                context.organization_id = org.id

                # Get membership
                member = await self._members.get_by_user_and_org(session.user_id, org.id)
                if member:
                    context.membership = member
                    context.role = member.role

        return context

    async def _build_context_from_api_key(
        self,
        api_key: APIKey,
        ip_address: Optional[str] = None,
    ) -> AuthContext:
        """Build auth context from API key."""
        org = await self._orgs.get(api_key.organization_id)

        # Get the creator user for the context
        user = await self._users.get(api_key.created_by_user_id)

        context = AuthContext(
            user=user,
            user_id=api_key.created_by_user_id,
            organization=org,
            organization_id=api_key.organization_id,
            api_key=api_key,
            ip_address=ip_address,
        )

        # Get membership for role
        if user:
            member = await self._members.get_by_user_and_org(user.id, api_key.organization_id)
            if member:
                context.membership = member
                context.role = member.role

        return context

    async def switch_organization(
        self,
        session_id: str,
        organization_id: str,
        user_id: str,
    ) -> Optional[Session]:
        """Switch session to different organization."""
        session = await self._sessions._sessions.get(session_id)
        if not session or session.user_id != user_id:
            return None

        # Verify membership
        member = await self._members.get_by_user_and_org(user_id, organization_id)
        if not member:
            return None

        session.organization_id = organization_id
        await self._sessions._sessions.update(session)

        return session

    async def register_user(
        self,
        email: str,
        password: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> User:
        """
        Register a new user.

        Raises:
            AuthenticationError: If registration fails
        """
        import uuid

        # Check if email exists
        existing = await self._users.get_by_email(email)
        if existing:
            raise AuthenticationError("Email already registered")

        # Validate password
        valid, errors = self._password_validator.validate(password)
        if not valid:
            raise AuthenticationError("; ".join(errors))

        # Hash password
        password_hash, salt = hash_password(password)

        user = User(
            id=f"user_{uuid.uuid4().hex[:16]}",
            email=email,
            status=UserStatus.PENDING,
            first_name=first_name,
            last_name=last_name,
            password_hash=f"{password_hash}:{salt}",
        )

        await self._users.create(user)

        return user

    async def verify_email(self, user_id: str) -> bool:
        """Mark user's email as verified."""
        user = await self._users.get(user_id)
        if not user:
            return False

        user.email_verified = True
        user.email_verified_at = datetime.utcnow()
        user.status = UserStatus.ACTIVE
        await self._users.update(user)

        return True

    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> bool:
        """
        Change user's password.

        Raises:
            AuthenticationError: If change fails
        """
        user = await self._users.get(user_id)
        if not user or not user.password_hash:
            raise AuthenticationError("User not found")

        # Verify current password
        parts = user.password_hash.split(":")
        if len(parts) != 2 or not verify_password(current_password, parts[0], parts[1]):
            raise AuthenticationError("Current password is incorrect")

        # Validate new password
        valid, errors = self._password_validator.validate(new_password)
        if not valid:
            raise AuthenticationError("; ".join(errors))

        # Hash new password
        password_hash, salt = hash_password(new_password)
        user.password_hash = f"{password_hash}:{salt}"
        await self._users.update(user)

        # Revoke all sessions
        await self._sessions.revoke_all_user_sessions(user_id)

        return True

    async def request_password_reset(self, email: str) -> Optional[str]:
        """
        Request password reset.

        Returns:
            Reset token if user exists, None otherwise
        """
        user = await self._users.get_by_email(email)
        if not user:
            return None

        # Generate reset token
        token = generate_token()
        token_hash = hash_token(token)

        # Store token (in real impl, would store in DB with expiry)
        user.metadata["reset_token"] = token_hash
        user.metadata["reset_token_expires"] = (
            datetime.utcnow() + timedelta(hours=1)
        ).isoformat()
        await self._users.update(user)

        return token

    async def reset_password(
        self,
        token: str,
        new_password: str,
    ) -> bool:
        """
        Reset password with token.

        Raises:
            AuthenticationError: If reset fails
        """
        token_hash = hash_token(token)

        # Find user with this token
        users = await self._users.list_all()
        user = None
        for u in users:
            if u.metadata.get("reset_token") == token_hash:
                expires = u.metadata.get("reset_token_expires")
                if expires and datetime.fromisoformat(expires) > datetime.utcnow():
                    user = u
                    break

        if not user:
            raise AuthenticationError("Invalid or expired reset token")

        # Validate new password
        valid, errors = self._password_validator.validate(new_password)
        if not valid:
            raise AuthenticationError("; ".join(errors))

        # Hash new password
        password_hash, salt = hash_password(new_password)
        user.password_hash = f"{password_hash}:{salt}"

        # Clear reset token
        user.metadata.pop("reset_token", None)
        user.metadata.pop("reset_token_expires", None)

        await self._users.update(user)

        # Revoke all sessions
        await self._sessions.revoke_all_user_sessions(user.id)

        return True
