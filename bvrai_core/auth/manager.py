"""Authentication and authorization manager."""

import asyncio
import hashlib
import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import redis.asyncio as redis

from .models import (
    User,
    Organization,
    OrganizationMember,
    IdentityProvider,
    IdentityProviderType,
    SAMLConfig,
    OIDCConfig,
    SSOLogin,
    AuditLog,
    AuditAction,
    MemberRole,
    Invitation,
    InvitationStatus,
)
from .saml import SAMLProvider, SAMLResponse, extract_user_attributes
from .oidc import OIDCProvider, OIDCTokens, OIDCUserInfo
from .session import SessionManager

logger = logging.getLogger(__name__)


class AuthManager:
    """
    Central authentication and authorization manager.

    Handles:
    - User authentication (password, SSO)
    - Organization management
    - Identity provider configuration
    - User provisioning from SSO
    - Role-based access control
    - Audit logging
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        session_manager: Optional[SessionManager] = None,
        base_url: str = "http://localhost:8000",
        password_min_length: int = 8,
        password_require_special: bool = True,
        password_require_number: bool = True,
        password_require_uppercase: bool = True,
    ):
        self.redis_url = redis_url
        self.base_url = base_url.rstrip("/")
        self.password_min_length = password_min_length
        self.password_require_special = password_require_special
        self.password_require_number = password_require_number
        self.password_require_uppercase = password_require_uppercase

        self.redis: Optional[redis.Redis] = None
        self.session_manager = session_manager

        # In-memory caches
        self._users: Dict[str, User] = {}
        self._organizations: Dict[str, Organization] = {}
        self._identity_providers: Dict[str, IdentityProvider] = {}
        self._lock = asyncio.Lock()

        # SSO providers (lazy-loaded)
        self._saml_providers: Dict[str, SAMLProvider] = {}
        self._oidc_providers: Dict[str, OIDCProvider] = {}

    async def start(self) -> None:
        """Start the auth manager."""
        self.redis = redis.from_url(self.redis_url, decode_responses=True)

        if not self.session_manager:
            self.session_manager = SessionManager(redis_url=self.redis_url)

        await self.session_manager.start()
        await self._load_data()

        logger.info("Auth manager started")

    async def stop(self) -> None:
        """Stop the auth manager."""
        if self.session_manager:
            await self.session_manager.stop()
        if self.redis:
            await self.redis.close()

        logger.info("Auth manager stopped")

    # User Management

    async def create_user(
        self,
        email: str,
        password: str = "",
        first_name: str = "",
        last_name: str = "",
        external_id: str = "",
        identity_provider_id: str = "",
        is_verified: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> User:
        """Create a new user."""
        # Check if email exists
        existing = await self.get_user_by_email(email)
        if existing:
            raise ValueError(f"User with email {email} already exists")

        # Validate password if provided
        if password:
            errors = self._validate_password(password)
            if errors:
                raise ValueError(f"Invalid password: {'; '.join(errors)}")

        user_id = f"usr_{uuid4().hex[:12]}"

        user = User(
            id=user_id,
            email=email.lower(),
            first_name=first_name,
            last_name=last_name,
            password_hash=self._hash_password(password) if password else "",
            external_id=external_id,
            identity_provider_id=identity_provider_id,
            is_verified=is_verified,
            metadata=metadata or {},
        )

        # Store user
        async with self._lock:
            self._users[user_id] = user

        await self._persist_user(user)

        await self._audit(
            organization_id="",
            user_id=user_id,
            action=AuditAction.USER_CREATED,
            resource_type="user",
            resource_id=user_id,
        )

        logger.info(f"Created user: {user_id} ({email})")
        return user

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        async with self._lock:
            if user_id in self._users:
                return self._users[user_id]

        # Load from Redis
        if self.redis:
            data = await self.redis.hget("users", user_id)
            if data:
                user = self._deserialize_user(json.loads(data))
                async with self._lock:
                    self._users[user_id] = user
                return user

        return None

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        email = email.lower()

        # Check cache
        async with self._lock:
            for user in self._users.values():
                if user.email == email:
                    return user

        # Check Redis index
        if self.redis:
            user_id = await self.redis.hget("user_emails", email)
            if user_id:
                return await self.get_user(user_id)

        return None

    async def update_user(
        self,
        user_id: str,
        updates: Dict[str, Any],
    ) -> Optional[User]:
        """Update a user."""
        user = await self.get_user(user_id)
        if not user:
            return None

        old_values = {}
        new_values = {}

        for key, value in updates.items():
            if hasattr(user, key) and key not in ("id", "email", "password_hash"):
                old_values[key] = getattr(user, key)
                setattr(user, key, value)
                new_values[key] = value

        user.updated_at = datetime.utcnow()

        async with self._lock:
            self._users[user_id] = user

        await self._persist_user(user)

        await self._audit(
            organization_id="",
            user_id=user_id,
            action=AuditAction.USER_UPDATED,
            resource_type="user",
            resource_id=user_id,
            old_values=old_values,
            new_values=new_values,
        )

        return user

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        user = await self.get_user(user_id)
        if not user:
            return False

        async with self._lock:
            self._users.pop(user_id, None)

        if self.redis:
            await self.redis.hdel("users", user_id)
            await self.redis.hdel("user_emails", user.email)

        # Revoke all sessions
        if self.session_manager:
            await self.session_manager.revoke_all_sessions(user_id)

        await self._audit(
            organization_id="",
            user_id=user_id,
            action=AuditAction.USER_DELETED,
            resource_type="user",
            resource_id=user_id,
        )

        logger.info(f"Deleted user: {user_id}")
        return True

    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> bool:
        """Change user password."""
        user = await self.get_user(user_id)
        if not user:
            return False

        # Verify current password
        if not self._verify_password(current_password, user.password_hash):
            return False

        # Validate new password
        errors = self._validate_password(new_password)
        if errors:
            raise ValueError(f"Invalid password: {'; '.join(errors)}")

        user.password_hash = self._hash_password(new_password)
        user.password_changed_at = datetime.utcnow()
        user.updated_at = datetime.utcnow()

        async with self._lock:
            self._users[user_id] = user

        await self._persist_user(user)

        await self._audit(
            organization_id="",
            user_id=user_id,
            action=AuditAction.USER_PASSWORD_CHANGED,
            resource_type="user",
            resource_id=user_id,
        )

        return True

    # Authentication

    async def authenticate_with_password(
        self,
        email: str,
        password: str,
        ip_address: str = "",
        user_agent: str = "",
    ) -> Tuple[Optional[User], str]:
        """
        Authenticate user with email and password.

        Returns:
            Tuple of (user, error_message)
        """
        user = await self.get_user_by_email(email)
        if not user:
            return None, "Invalid email or password"

        if not user.is_active:
            return None, "Account is disabled"

        if not user.password_hash:
            return None, "Password login not enabled for this account"

        if not self._verify_password(password, user.password_hash):
            await self._audit(
                organization_id="",
                user_id=user.id,
                action=AuditAction.USER_LOGIN_FAILED,
                resource_type="user",
                resource_id=user.id,
                ip_address=ip_address,
                success=False,
                error_message="Invalid password",
            )
            return None, "Invalid email or password"

        # Update last login
        user.last_login_at = datetime.utcnow()
        async with self._lock:
            self._users[user.id] = user
        await self._persist_user(user)

        return user, ""

    # SSO Authentication

    async def initiate_sso_login(
        self,
        organization_id: str,
        identity_provider_id: str,
        redirect_uri: str = "",
    ) -> Tuple[str, SSOLogin]:
        """
        Initiate SSO login flow.

        Returns:
            Tuple of (redirect_url, sso_login)
        """
        idp = await self.get_identity_provider(identity_provider_id)
        if not idp:
            raise ValueError("Identity provider not found")

        if not idp.is_active:
            raise ValueError("Identity provider is disabled")

        # Create SSO login record
        sso_login = SSOLogin(
            id=f"sso_{uuid4().hex[:12]}",
            organization_id=organization_id,
            identity_provider_id=identity_provider_id,
            state=secrets.token_urlsafe(32),
            nonce=secrets.token_urlsafe(32),
            redirect_uri=redirect_uri or f"{self.base_url}/auth/callback",
        )

        if idp.provider_type == IdentityProviderType.SAML:
            provider = await self._get_saml_provider(idp)
            request_id, redirect_url, _ = provider.create_authn_request(
                relay_state=sso_login.state,
            )
            sso_login.saml_request_id = request_id

        elif idp.provider_type == IdentityProviderType.OIDC:
            provider = await self._get_oidc_provider(idp)
            redirect_url, state, nonce, code_verifier = provider.create_authorization_url(
                state=sso_login.state,
                nonce=sso_login.nonce,
                redirect_uri=sso_login.redirect_uri,
            )
            # Store code verifier for PKCE
            sso_login.metadata = {"code_verifier": code_verifier}

        else:
            raise ValueError(f"Unsupported provider type: {idp.provider_type}")

        # Store SSO login
        if self.redis:
            await self.redis.setex(
                f"sso_login:{sso_login.state}",
                600,  # 10 minutes
                json.dumps(self._serialize_sso_login(sso_login)),
            )

        return redirect_url, sso_login

    async def complete_saml_login(
        self,
        saml_response: str,
        relay_state: str,
        ip_address: str = "",
        user_agent: str = "",
    ) -> Tuple[Optional[User], Optional[OrganizationMember], str]:
        """
        Complete SAML SSO login.

        Returns:
            Tuple of (user, membership, error_message)
        """
        # Get SSO login record
        sso_login = await self._get_sso_login(relay_state)
        if not sso_login:
            return None, None, "Invalid or expired SSO session"

        # Get identity provider
        idp = await self.get_identity_provider(sso_login.identity_provider_id)
        if not idp:
            return None, None, "Identity provider not found"

        # Get SAML provider
        provider = await self._get_saml_provider(idp)

        # Parse and validate response
        response = provider.parse_response(saml_response, sso_login.saml_request_id)

        is_valid, error = provider.validate_response(response, idp.saml_config.sp_entity_id if idp.saml_config else "")
        if not is_valid:
            sso_login.status = "failed"
            sso_login.error_message = error
            return None, None, error

        # Extract user attributes
        assertion = response.first_assertion
        if not assertion:
            return None, None, "No assertion in SAML response"

        attrs = extract_user_attributes(
            assertion,
            idp.saml_config.email_attribute if idp.saml_config else "email",
            idp.saml_config.first_name_attribute if idp.saml_config else "firstName",
            idp.saml_config.last_name_attribute if idp.saml_config else "lastName",
            idp.saml_config.groups_attribute if idp.saml_config else "groups",
        )

        # Provision or get user
        user, membership, error = await self._provision_sso_user(
            idp=idp,
            email=attrs["email"],
            first_name=attrs.get("first_name", ""),
            last_name=attrs.get("last_name", ""),
            external_id=attrs["name_id"],
            groups=attrs.get("groups", []),
        )

        if error:
            return None, None, error

        # Update SSO login record
        sso_login.status = "completed"
        sso_login.user_id = user.id if user else ""
        sso_login.completed_at = datetime.utcnow()

        # Update IdP last login
        idp.last_login_at = datetime.utcnow()
        await self._persist_identity_provider(idp)

        await self._audit(
            organization_id=sso_login.organization_id,
            user_id=user.id if user else "",
            action=AuditAction.SSO_LOGIN,
            resource_type="identity_provider",
            resource_id=idp.id,
            ip_address=ip_address,
            details={"provider_type": "saml", "email": attrs["email"]},
        )

        return user, membership, ""

    async def complete_oidc_login(
        self,
        code: str,
        state: str,
        ip_address: str = "",
        user_agent: str = "",
    ) -> Tuple[Optional[User], Optional[OrganizationMember], str]:
        """
        Complete OIDC SSO login.

        Returns:
            Tuple of (user, membership, error_message)
        """
        # Get SSO login record
        sso_login = await self._get_sso_login(state)
        if not sso_login:
            return None, None, "Invalid or expired SSO session"

        # Get identity provider
        idp = await self.get_identity_provider(sso_login.identity_provider_id)
        if not idp:
            return None, None, "Identity provider not found"

        # Get OIDC provider
        provider = await self._get_oidc_provider(idp)

        try:
            # Exchange code for tokens
            code_verifier = sso_login.metadata.get("code_verifier", "") if sso_login.metadata else ""
            tokens = await provider.exchange_code(
                code=code,
                redirect_uri=sso_login.redirect_uri,
                code_verifier=code_verifier,
            )

            # Validate ID token
            is_valid, error, claims = await provider.validate_id_token(
                tokens.id_token,
                expected_nonce=sso_login.nonce,
            )

            if not is_valid:
                return None, None, error

            # Get user info
            userinfo = await provider.get_userinfo(tokens.access_token)

            # Provision or get user
            user, membership, error = await self._provision_sso_user(
                idp=idp,
                email=userinfo.email,
                first_name=userinfo.given_name,
                last_name=userinfo.family_name,
                external_id=userinfo.sub,
                groups=userinfo.groups,
            )

            if error:
                return None, None, error

            # Update SSO login record
            sso_login.status = "completed"
            sso_login.user_id = user.id if user else ""
            sso_login.completed_at = datetime.utcnow()

            # Update IdP last login
            idp.last_login_at = datetime.utcnow()
            await self._persist_identity_provider(idp)

            await self._audit(
                organization_id=sso_login.organization_id,
                user_id=user.id if user else "",
                action=AuditAction.SSO_LOGIN,
                resource_type="identity_provider",
                resource_id=idp.id,
                ip_address=ip_address,
                details={"provider_type": "oidc", "email": userinfo.email},
            )

            return user, membership, ""

        except Exception as e:
            logger.error(f"OIDC login failed: {e}")
            return None, None, str(e)

    async def _provision_sso_user(
        self,
        idp: IdentityProvider,
        email: str,
        first_name: str,
        last_name: str,
        external_id: str,
        groups: List[str],
    ) -> Tuple[Optional[User], Optional[OrganizationMember], str]:
        """Provision or update user from SSO."""
        if not email:
            return None, None, "Email not provided by identity provider"

        # Check domain restrictions
        if idp.allowed_domains:
            domain = email.split("@")[-1].lower()
            if domain not in [d.lower() for d in idp.allowed_domains]:
                return None, None, f"Email domain {domain} not allowed"

        # Find or create user
        user = await self.get_user_by_email(email)

        if not user:
            if not idp.auto_provision_users:
                return None, None, "User not found and auto-provisioning is disabled"

            user = await self.create_user(
                email=email,
                first_name=first_name,
                last_name=last_name,
                external_id=external_id,
                identity_provider_id=idp.id,
                is_verified=True,
            )
        else:
            # Update user info from IdP
            if external_id and user.external_id != external_id:
                user.external_id = external_id
            if idp.id and user.identity_provider_id != idp.id:
                user.identity_provider_id = idp.id

            user.last_login_at = datetime.utcnow()
            async with self._lock:
                self._users[user.id] = user
            await self._persist_user(user)

        # Get or create membership
        membership = await self.get_membership(idp.organization_id, user.id)

        if not membership:
            # Determine role from groups
            role = idp.default_role
            for group, mapped_role in idp.group_role_mapping.items():
                if group in groups:
                    role = mapped_role
                    break

            membership = await self.add_member(
                organization_id=idp.organization_id,
                user_id=user.id,
                role=role,
                invited_by="sso",
            )

        return user, membership, ""

    # Organization Management

    async def create_organization(
        self,
        name: str,
        slug: str,
        owner_id: str,
        plan: str = "free",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Organization:
        """Create a new organization."""
        # Check if slug is unique
        existing = await self.get_organization_by_slug(slug)
        if existing:
            raise ValueError(f"Organization with slug {slug} already exists")

        org_id = f"org_{uuid4().hex[:12]}"

        org = Organization(
            id=org_id,
            name=name,
            slug=slug.lower(),
            plan=plan,
            metadata=metadata or {},
        )

        async with self._lock:
            self._organizations[org_id] = org

        await self._persist_organization(org)

        # Add owner as member
        await self.add_member(org_id, owner_id, MemberRole.OWNER)

        await self._audit(
            organization_id=org_id,
            user_id=owner_id,
            action=AuditAction.ORG_CREATED,
            resource_type="organization",
            resource_id=org_id,
        )

        logger.info(f"Created organization: {org_id} ({name})")
        return org

    async def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get an organization by ID."""
        async with self._lock:
            if org_id in self._organizations:
                return self._organizations[org_id]

        if self.redis:
            data = await self.redis.hget("organizations", org_id)
            if data:
                org = self._deserialize_organization(json.loads(data))
                async with self._lock:
                    self._organizations[org_id] = org
                return org

        return None

    async def get_organization_by_slug(self, slug: str) -> Optional[Organization]:
        """Get an organization by slug."""
        slug = slug.lower()

        async with self._lock:
            for org in self._organizations.values():
                if org.slug == slug:
                    return org

        if self.redis:
            org_id = await self.redis.hget("org_slugs", slug)
            if org_id:
                return await self.get_organization(org_id)

        return None

    # Membership Management

    async def add_member(
        self,
        organization_id: str,
        user_id: str,
        role: MemberRole = MemberRole.VIEWER,
        invited_by: str = "",
    ) -> OrganizationMember:
        """Add a member to an organization."""
        member_id = f"mem_{uuid4().hex[:12]}"

        member = OrganizationMember(
            id=member_id,
            organization_id=organization_id,
            user_id=user_id,
            role=role,
            invited_by=invited_by,
        )

        if self.redis:
            await self.redis.hset(
                f"org_members:{organization_id}",
                user_id,
                json.dumps(self._serialize_member(member)),
            )

        await self._audit(
            organization_id=organization_id,
            user_id=invited_by,
            action=AuditAction.MEMBER_ADDED,
            resource_type="member",
            resource_id=member_id,
            details={"user_id": user_id, "role": role.value},
        )

        return member

    async def get_membership(
        self,
        organization_id: str,
        user_id: str,
    ) -> Optional[OrganizationMember]:
        """Get a user's membership in an organization."""
        if not self.redis:
            return None

        data = await self.redis.hget(f"org_members:{organization_id}", user_id)
        if data:
            return self._deserialize_member(json.loads(data))

        return None

    async def list_members(self, organization_id: str) -> List[OrganizationMember]:
        """List all members of an organization."""
        if not self.redis:
            return []

        members_data = await self.redis.hgetall(f"org_members:{organization_id}")
        members = []

        for data in members_data.values():
            member = self._deserialize_member(json.loads(data))
            members.append(member)

        return members

    async def update_member_role(
        self,
        organization_id: str,
        user_id: str,
        new_role: MemberRole,
        updated_by: str = "",
    ) -> Optional[OrganizationMember]:
        """Update a member's role."""
        member = await self.get_membership(organization_id, user_id)
        if not member:
            return None

        old_role = member.role
        member.role = new_role
        member.updated_at = datetime.utcnow()

        if self.redis:
            await self.redis.hset(
                f"org_members:{organization_id}",
                user_id,
                json.dumps(self._serialize_member(member)),
            )

        await self._audit(
            organization_id=organization_id,
            user_id=updated_by,
            action=AuditAction.MEMBER_ROLE_CHANGED,
            resource_type="member",
            resource_id=member.id,
            old_values={"role": old_role.value},
            new_values={"role": new_role.value},
        )

        return member

    async def remove_member(
        self,
        organization_id: str,
        user_id: str,
        removed_by: str = "",
    ) -> bool:
        """Remove a member from an organization."""
        member = await self.get_membership(organization_id, user_id)
        if not member:
            return False

        if self.redis:
            await self.redis.hdel(f"org_members:{organization_id}", user_id)

        # Revoke sessions for this org
        if self.session_manager:
            # Would need to implement org-specific session revocation
            pass

        await self._audit(
            organization_id=organization_id,
            user_id=removed_by,
            action=AuditAction.MEMBER_REMOVED,
            resource_type="member",
            resource_id=member.id,
            details={"user_id": user_id},
        )

        return True

    # Identity Provider Management

    async def create_identity_provider(
        self,
        organization_id: str,
        name: str,
        provider_type: IdentityProviderType,
        saml_config: Optional[SAMLConfig] = None,
        oidc_config: Optional[OIDCConfig] = None,
        is_primary: bool = False,
        auto_provision_users: bool = True,
        default_role: MemberRole = MemberRole.VIEWER,
        allowed_domains: Optional[List[str]] = None,
        group_role_mapping: Optional[Dict[str, MemberRole]] = None,
    ) -> IdentityProvider:
        """Create an identity provider."""
        idp_id = f"idp_{uuid4().hex[:12]}"

        # Set SP configuration for SAML
        if saml_config:
            saml_config.sp_entity_id = f"{self.base_url}/saml/{organization_id}"
            saml_config.acs_url = f"{self.base_url}/auth/saml/callback"

        idp = IdentityProvider(
            id=idp_id,
            organization_id=organization_id,
            name=name,
            provider_type=provider_type,
            saml_config=saml_config,
            oidc_config=oidc_config,
            is_primary=is_primary,
            auto_provision_users=auto_provision_users,
            default_role=default_role,
            allowed_domains=allowed_domains or [],
            group_role_mapping=group_role_mapping or {},
        )

        # If this is primary, unset other primary IdPs
        if is_primary:
            existing_idps = await self.list_identity_providers(organization_id)
            for existing in existing_idps:
                if existing.is_primary:
                    existing.is_primary = False
                    await self._persist_identity_provider(existing)

        async with self._lock:
            self._identity_providers[idp_id] = idp

        await self._persist_identity_provider(idp)

        await self._audit(
            organization_id=organization_id,
            action=AuditAction.IDP_CREATED,
            resource_type="identity_provider",
            resource_id=idp_id,
            details={"name": name, "type": provider_type.value},
        )

        logger.info(f"Created identity provider: {idp_id} ({name})")
        return idp

    async def get_identity_provider(self, idp_id: str) -> Optional[IdentityProvider]:
        """Get an identity provider by ID."""
        async with self._lock:
            if idp_id in self._identity_providers:
                return self._identity_providers[idp_id]

        if self.redis:
            data = await self.redis.hget("identity_providers", idp_id)
            if data:
                idp = self._deserialize_identity_provider(json.loads(data))
                async with self._lock:
                    self._identity_providers[idp_id] = idp
                return idp

        return None

    async def list_identity_providers(
        self,
        organization_id: str,
    ) -> List[IdentityProvider]:
        """List identity providers for an organization."""
        idps = []

        async with self._lock:
            for idp in self._identity_providers.values():
                if idp.organization_id == organization_id:
                    idps.append(idp)

        return idps

    async def delete_identity_provider(self, idp_id: str) -> bool:
        """Delete an identity provider."""
        idp = await self.get_identity_provider(idp_id)
        if not idp:
            return False

        async with self._lock:
            self._identity_providers.pop(idp_id, None)
            self._saml_providers.pop(idp_id, None)
            self._oidc_providers.pop(idp_id, None)

        if self.redis:
            await self.redis.hdel("identity_providers", idp_id)

        await self._audit(
            organization_id=idp.organization_id,
            action=AuditAction.IDP_DELETED,
            resource_type="identity_provider",
            resource_id=idp_id,
        )

        return True

    async def _get_saml_provider(self, idp: IdentityProvider) -> SAMLProvider:
        """Get or create SAML provider instance."""
        if idp.id in self._saml_providers:
            return self._saml_providers[idp.id]

        if not idp.saml_config:
            raise ValueError("SAML configuration not found")

        provider = SAMLProvider(
            entity_id=idp.saml_config.sp_entity_id,
            acs_url=idp.saml_config.acs_url,
            idp_entity_id=idp.saml_config.entity_id,
            idp_sso_url=idp.saml_config.sso_url,
            idp_certificate=idp.saml_config.certificate,
            idp_slo_url=idp.saml_config.slo_url,
            sign_requests=idp.saml_config.sign_requests,
            want_assertions_signed=idp.saml_config.want_assertions_signed,
            signature_algorithm=idp.saml_config.signature_algorithm,
        )

        self._saml_providers[idp.id] = provider
        return provider

    async def _get_oidc_provider(self, idp: IdentityProvider) -> OIDCProvider:
        """Get or create OIDC provider instance."""
        if idp.id in self._oidc_providers:
            return self._oidc_providers[idp.id]

        if not idp.oidc_config:
            raise ValueError("OIDC configuration not found")

        provider = OIDCProvider(
            issuer=idp.oidc_config.issuer,
            client_id=idp.oidc_config.client_id,
            client_secret=idp.oidc_config.client_secret,
            redirect_uri=f"{self.base_url}/auth/oidc/callback",
            scopes=idp.oidc_config.scopes,
            use_pkce=idp.oidc_config.use_pkce,
            authorization_endpoint=idp.oidc_config.authorization_endpoint,
            token_endpoint=idp.oidc_config.token_endpoint,
            userinfo_endpoint=idp.oidc_config.userinfo_endpoint,
            jwks_uri=idp.oidc_config.jwks_uri,
            end_session_endpoint=idp.oidc_config.end_session_endpoint,
            email_claim=idp.oidc_config.email_claim,
            name_claim=idp.oidc_config.name_claim,
            groups_claim=idp.oidc_config.groups_claim,
        )

        # Discover endpoints
        await provider.discover()

        self._oidc_providers[idp.id] = provider
        return provider

    async def _get_sso_login(self, state: str) -> Optional[SSOLogin]:
        """Get SSO login record by state."""
        if not self.redis:
            return None

        data = await self.redis.get(f"sso_login:{state}")
        if not data:
            return None

        return self._deserialize_sso_login(json.loads(data))

    # Audit Logging

    async def _audit(
        self,
        organization_id: str,
        action: AuditAction,
        resource_type: str = "",
        resource_id: str = "",
        user_id: str = "",
        user_email: str = "",
        details: Optional[Dict[str, Any]] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        ip_address: str = "",
        user_agent: str = "",
        success: bool = True,
        error_message: str = "",
    ) -> AuditLog:
        """Create an audit log entry."""
        log_id = f"log_{uuid4().hex[:16]}"

        audit_log = AuditLog(
            id=log_id,
            organization_id=organization_id,
            user_id=user_id,
            user_email=user_email,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            old_values=old_values or {},
            new_values=new_values or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
        )

        if self.redis:
            # Store in sorted set by timestamp
            await self.redis.zadd(
                f"audit_logs:{organization_id}",
                {json.dumps(audit_log.to_dict()): audit_log.created_at.timestamp()},
            )

            # Keep only last 10000 entries
            await self.redis.zremrangebyrank(f"audit_logs:{organization_id}", 0, -10001)

        return audit_log

    async def get_audit_logs(
        self,
        organization_id: str,
        limit: int = 100,
        offset: int = 0,
        action: Optional[AuditAction] = None,
        user_id: str = "",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[AuditLog]:
        """Get audit logs for an organization."""
        if not self.redis:
            return []

        # Get from sorted set (newest first)
        start_score = start_time.timestamp() if start_time else "-inf"
        end_score = end_time.timestamp() if end_time else "+inf"

        entries = await self.redis.zrevrangebyscore(
            f"audit_logs:{organization_id}",
            end_score,
            start_score,
            start=offset,
            num=limit,
        )

        logs = []
        for entry in entries:
            data = json.loads(entry)
            log = AuditLog(
                id=data["id"],
                organization_id=data["organization_id"],
                user_id=data.get("user_id", ""),
                user_email=data.get("user_email", ""),
                action=AuditAction(data["action"]),
                resource_type=data.get("resource_type", ""),
                resource_id=data.get("resource_id", ""),
                details=data.get("details", {}),
                old_values=data.get("old_values", {}),
                new_values=data.get("new_values", {}),
                ip_address=data.get("ip_address", ""),
                user_agent=data.get("user_agent", ""),
                success=data.get("success", True),
                error_message=data.get("error_message", ""),
                created_at=datetime.fromisoformat(data["created_at"]),
            )

            # Apply filters
            if action and log.action != action:
                continue
            if user_id and log.user_id != user_id:
                continue

            logs.append(log)

        return logs

    # Password helpers

    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt-like approach."""
        import hashlib
        import secrets

        salt = secrets.token_hex(16)
        hash_val = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000,
        ).hex()

        return f"{salt}${hash_val}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        import hashlib

        if "$" not in password_hash:
            return False

        salt, hash_val = password_hash.split("$", 1)

        computed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000,
        ).hex()

        return computed == hash_val

    def _validate_password(self, password: str) -> List[str]:
        """Validate password strength."""
        errors = []

        if len(password) < self.password_min_length:
            errors.append(f"Password must be at least {self.password_min_length} characters")

        if self.password_require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")

        if self.password_require_number and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")

        if self.password_require_special:
            special_chars = "!@#$%^&*()_+-=[]{}|;:',.<>?/`~"
            if not any(c in special_chars for c in password):
                errors.append("Password must contain at least one special character")

        return errors

    # Serialization helpers

    def _serialize_user(self, user: User) -> Dict[str, Any]:
        return user.to_dict(include_sensitive=True)

    def _deserialize_user(self, data: Dict[str, Any]) -> User:
        return User(
            id=data["id"],
            email=data["email"],
            first_name=data.get("first_name", ""),
            last_name=data.get("last_name", ""),
            display_name=data.get("display_name", ""),
            avatar_url=data.get("avatar_url", ""),
            phone=data.get("phone", ""),
            timezone=data.get("timezone", "UTC"),
            locale=data.get("locale", "en"),
            password_hash=data.get("password_hash", ""),
            mfa_enabled=data.get("mfa_enabled", False),
            mfa_secret=data.get("mfa_secret", ""),
            mfa_backup_codes=data.get("mfa_backup_codes", []),
            external_id=data.get("external_id", ""),
            identity_provider_id=data.get("identity_provider_id", ""),
            is_active=data.get("is_active", True),
            is_verified=data.get("is_verified", False),
            is_system_admin=data.get("is_system_admin", False),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            last_login_at=datetime.fromisoformat(data["last_login_at"]) if data.get("last_login_at") else None,
            metadata=data.get("metadata", {}),
        )

    def _serialize_organization(self, org: Organization) -> Dict[str, Any]:
        return org.to_dict()

    def _deserialize_organization(self, data: Dict[str, Any]) -> Organization:
        return Organization(
            id=data["id"],
            name=data["name"],
            slug=data["slug"],
            plan=data.get("plan", "free"),
            max_seats=data.get("max_seats", 5),
            require_sso=data.get("require_sso", False),
            allowed_email_domains=data.get("allowed_email_domains", []),
            enforce_mfa=data.get("enforce_mfa", False),
            session_timeout_hours=data.get("session_timeout_hours", 24),
            ip_allowlist=data.get("ip_allowlist", []),
            logo_url=data.get("logo_url", ""),
            primary_color=data.get("primary_color", ""),
            is_active=data.get("is_active", True),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            billing_email=data.get("billing_email", ""),
            stripe_customer_id=data.get("stripe_customer_id", ""),
            metadata=data.get("metadata", {}),
        )

    def _serialize_member(self, member: OrganizationMember) -> Dict[str, Any]:
        return member.to_dict()

    def _deserialize_member(self, data: Dict[str, Any]) -> OrganizationMember:
        return OrganizationMember(
            id=data["id"],
            organization_id=data["organization_id"],
            user_id=data["user_id"],
            role=MemberRole(data.get("role", "viewer")),
            permissions=data.get("permissions", []),
            is_active=data.get("is_active", True),
            joined_at=datetime.fromisoformat(data["joined_at"]) if data.get("joined_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            invited_by=data.get("invited_by", ""),
        )

    def _serialize_identity_provider(self, idp: IdentityProvider) -> Dict[str, Any]:
        return idp.to_dict()

    def _deserialize_identity_provider(self, data: Dict[str, Any]) -> IdentityProvider:
        saml_config = None
        if data.get("saml_config"):
            saml_config = SAMLConfig(**data["saml_config"])

        oidc_config = None
        if data.get("oidc_config"):
            oidc_config = OIDCConfig(**data["oidc_config"])

        group_role_mapping = {}
        for group, role in data.get("group_role_mapping", {}).items():
            group_role_mapping[group] = MemberRole(role)

        return IdentityProvider(
            id=data["id"],
            organization_id=data["organization_id"],
            name=data["name"],
            provider_type=IdentityProviderType(data["provider_type"]),
            saml_config=saml_config,
            oidc_config=oidc_config,
            is_active=data.get("is_active", True),
            is_primary=data.get("is_primary", False),
            auto_provision_users=data.get("auto_provision_users", True),
            default_role=MemberRole(data.get("default_role", "viewer")),
            allowed_domains=data.get("allowed_domains", []),
            group_role_mapping=group_role_mapping,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            last_login_at=datetime.fromisoformat(data["last_login_at"]) if data.get("last_login_at") else None,
            metadata=data.get("metadata", {}),
        )

    def _serialize_sso_login(self, sso_login: SSOLogin) -> Dict[str, Any]:
        return {
            **sso_login.to_dict(),
            "metadata": sso_login.metadata if hasattr(sso_login, 'metadata') else {},
        }

    def _deserialize_sso_login(self, data: Dict[str, Any]) -> SSOLogin:
        sso_login = SSOLogin(
            id=data["id"],
            organization_id=data["organization_id"],
            identity_provider_id=data["identity_provider_id"],
            state=data.get("state", ""),
            nonce=data.get("nonce", ""),
            redirect_uri=data.get("redirect_uri", ""),
            saml_request_id=data.get("saml_request_id", ""),
            status=data.get("status", "pending"),
            error_message=data.get("error_message", ""),
            user_id=data.get("user_id", ""),
            session_id=data.get("session_id", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else datetime.utcnow(),
        )
        sso_login.metadata = data.get("metadata", {})
        return sso_login

    # Persistence

    async def _persist_user(self, user: User) -> None:
        if not self.redis:
            return

        await self.redis.hset("users", user.id, json.dumps(self._serialize_user(user)))
        await self.redis.hset("user_emails", user.email, user.id)

    async def _persist_organization(self, org: Organization) -> None:
        if not self.redis:
            return

        await self.redis.hset("organizations", org.id, json.dumps(self._serialize_organization(org)))
        await self.redis.hset("org_slugs", org.slug, org.id)

    async def _persist_identity_provider(self, idp: IdentityProvider) -> None:
        if not self.redis:
            return

        await self.redis.hset("identity_providers", idp.id, json.dumps(self._serialize_identity_provider(idp)))

    async def _load_data(self) -> None:
        """Load data from Redis."""
        if not self.redis:
            return

        # Load organizations
        org_data = await self.redis.hgetall("organizations")
        for org_json in org_data.values():
            org = self._deserialize_organization(json.loads(org_json))
            self._organizations[org.id] = org

        # Load identity providers
        idp_data = await self.redis.hgetall("identity_providers")
        for idp_json in idp_data.values():
            idp = self._deserialize_identity_provider(json.loads(idp_json))
            self._identity_providers[idp.id] = idp

        logger.info(f"Loaded {len(self._organizations)} organizations, {len(self._identity_providers)} identity providers")
