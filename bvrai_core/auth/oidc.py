"""OpenID Connect authentication provider."""

import base64
import hashlib
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlencode, parse_qs, urlparse

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class OIDCTokens:
    """OIDC token response."""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: str = ""
    id_token: str = ""
    scope: str = ""

    # Parsed from id_token
    claims: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    issued_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "access_token": self.access_token[:20] + "..." if self.access_token else "",
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "has_refresh_token": bool(self.refresh_token),
            "has_id_token": bool(self.id_token),
            "scope": self.scope,
            "claims": self.claims,
            "issued_at": self.issued_at.isoformat(),
        }

    @property
    def expires_at(self) -> datetime:
        return self.issued_at + timedelta(seconds=self.expires_in)

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at


@dataclass
class OIDCUserInfo:
    """User information from OIDC provider."""
    sub: str  # Subject identifier
    email: str = ""
    email_verified: bool = False
    name: str = ""
    given_name: str = ""
    family_name: str = ""
    preferred_username: str = ""
    picture: str = ""
    locale: str = ""
    zoneinfo: str = ""
    groups: List[str] = field(default_factory=list)
    raw_claims: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub": self.sub,
            "email": self.email,
            "email_verified": self.email_verified,
            "name": self.name,
            "given_name": self.given_name,
            "family_name": self.family_name,
            "preferred_username": self.preferred_username,
            "picture": self.picture,
            "locale": self.locale,
            "zoneinfo": self.zoneinfo,
            "groups": self.groups,
        }


class OIDCProvider:
    """
    OpenID Connect authentication provider.

    Supports:
    - Authorization Code flow with PKCE
    - Token exchange
    - UserInfo endpoint
    - Discovery via .well-known/openid-configuration
    - Token refresh
    - Logout
    """

    def __init__(
        self,
        issuer: str,
        client_id: str,
        client_secret: str = "",
        redirect_uri: str = "",
        scopes: Optional[List[str]] = None,
        use_pkce: bool = True,
        response_type: str = "code",
        # Endpoints (auto-discovered if not provided)
        authorization_endpoint: str = "",
        token_endpoint: str = "",
        userinfo_endpoint: str = "",
        jwks_uri: str = "",
        end_session_endpoint: str = "",
        # Claim mapping
        email_claim: str = "email",
        name_claim: str = "name",
        groups_claim: str = "groups",
    ):
        self.issuer = issuer.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scopes = scopes or ["openid", "profile", "email"]
        self.use_pkce = use_pkce
        self.response_type = response_type

        self.authorization_endpoint = authorization_endpoint
        self.token_endpoint = token_endpoint
        self.userinfo_endpoint = userinfo_endpoint
        self.jwks_uri = jwks_uri
        self.end_session_endpoint = end_session_endpoint

        self.email_claim = email_claim
        self.name_claim = name_claim
        self.groups_claim = groups_claim

        self._discovery: Optional[Dict[str, Any]] = None
        self._jwks: Optional[Dict[str, Any]] = None

    async def discover(self) -> Dict[str, Any]:
        """
        Fetch OIDC discovery document.

        Returns provider configuration from .well-known/openid-configuration
        """
        if self._discovery:
            return self._discovery

        discovery_url = f"{self.issuer}/.well-known/openid-configuration"

        async with aiohttp.ClientSession() as session:
            async with session.get(discovery_url) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to fetch discovery document: {response.status}")

                self._discovery = await response.json()

                # Update endpoints from discovery
                if not self.authorization_endpoint:
                    self.authorization_endpoint = self._discovery.get("authorization_endpoint", "")
                if not self.token_endpoint:
                    self.token_endpoint = self._discovery.get("token_endpoint", "")
                if not self.userinfo_endpoint:
                    self.userinfo_endpoint = self._discovery.get("userinfo_endpoint", "")
                if not self.jwks_uri:
                    self.jwks_uri = self._discovery.get("jwks_uri", "")
                if not self.end_session_endpoint:
                    self.end_session_endpoint = self._discovery.get("end_session_endpoint", "")

                return self._discovery

    async def get_jwks(self) -> Dict[str, Any]:
        """Fetch JSON Web Key Set for token verification."""
        if self._jwks:
            return self._jwks

        if not self.jwks_uri:
            await self.discover()

        if not self.jwks_uri:
            raise ValueError("JWKS URI not available")

        async with aiohttp.ClientSession() as session:
            async with session.get(self.jwks_uri) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to fetch JWKS: {response.status}")

                self._jwks = await response.json()
                return self._jwks

    def create_authorization_url(
        self,
        state: str = "",
        nonce: str = "",
        redirect_uri: str = "",
        additional_params: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str, str, str]:
        """
        Create authorization URL for initiating OIDC flow.

        Returns:
            Tuple of (auth_url, state, nonce, code_verifier)
        """
        if not self.authorization_endpoint:
            raise ValueError("Authorization endpoint not configured")

        # Generate state and nonce if not provided
        state = state or secrets.token_urlsafe(32)
        nonce = nonce or secrets.token_urlsafe(32)

        params = {
            "response_type": self.response_type,
            "client_id": self.client_id,
            "redirect_uri": redirect_uri or self.redirect_uri,
            "scope": " ".join(self.scopes),
            "state": state,
            "nonce": nonce,
        }

        code_verifier = ""

        # PKCE
        if self.use_pkce:
            code_verifier = secrets.token_urlsafe(64)
            code_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode()).digest()
            ).decode().rstrip("=")

            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        if additional_params:
            params.update(additional_params)

        auth_url = f"{self.authorization_endpoint}?{urlencode(params)}"

        return auth_url, state, nonce, code_verifier

    async def exchange_code(
        self,
        code: str,
        redirect_uri: str = "",
        code_verifier: str = "",
    ) -> OIDCTokens:
        """
        Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback
            redirect_uri: Redirect URI used in authorization request
            code_verifier: PKCE code verifier (if used)

        Returns:
            OIDCTokens with access_token, id_token, etc.
        """
        if not self.token_endpoint:
            await self.discover()

        if not self.token_endpoint:
            raise ValueError("Token endpoint not configured")

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri or self.redirect_uri,
            "client_id": self.client_id,
        }

        if self.client_secret:
            data["client_secret"] = self.client_secret

        if code_verifier:
            data["code_verifier"] = code_verifier

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ) as response:
                if response.status != 200:
                    error_body = await response.text()
                    raise ValueError(f"Token exchange failed: {response.status} - {error_body}")

                token_response = await response.json()

                tokens = OIDCTokens(
                    access_token=token_response.get("access_token", ""),
                    token_type=token_response.get("token_type", "Bearer"),
                    expires_in=token_response.get("expires_in", 3600),
                    refresh_token=token_response.get("refresh_token", ""),
                    id_token=token_response.get("id_token", ""),
                    scope=token_response.get("scope", ""),
                )

                # Parse ID token claims
                if tokens.id_token:
                    tokens.claims = self._parse_jwt_claims(tokens.id_token)

                return tokens

    async def refresh_tokens(self, refresh_token: str) -> OIDCTokens:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token from previous exchange

        Returns:
            New OIDCTokens
        """
        if not self.token_endpoint:
            await self.discover()

        if not self.token_endpoint:
            raise ValueError("Token endpoint not configured")

        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
        }

        if self.client_secret:
            data["client_secret"] = self.client_secret

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ) as response:
                if response.status != 200:
                    error_body = await response.text()
                    raise ValueError(f"Token refresh failed: {response.status} - {error_body}")

                token_response = await response.json()

                return OIDCTokens(
                    access_token=token_response.get("access_token", ""),
                    token_type=token_response.get("token_type", "Bearer"),
                    expires_in=token_response.get("expires_in", 3600),
                    refresh_token=token_response.get("refresh_token", refresh_token),
                    id_token=token_response.get("id_token", ""),
                    scope=token_response.get("scope", ""),
                )

    async def get_userinfo(self, access_token: str) -> OIDCUserInfo:
        """
        Fetch user information from userinfo endpoint.

        Args:
            access_token: Valid access token

        Returns:
            OIDCUserInfo with user details
        """
        if not self.userinfo_endpoint:
            await self.discover()

        if not self.userinfo_endpoint:
            raise ValueError("UserInfo endpoint not configured")

        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.userinfo_endpoint,
                headers={"Authorization": f"Bearer {access_token}"},
            ) as response:
                if response.status != 200:
                    error_body = await response.text()
                    raise ValueError(f"UserInfo request failed: {response.status} - {error_body}")

                claims = await response.json()

                return self._parse_userinfo(claims)

    def _parse_userinfo(self, claims: Dict[str, Any]) -> OIDCUserInfo:
        """Parse userinfo response into OIDCUserInfo."""
        groups = claims.get(self.groups_claim, [])
        if isinstance(groups, str):
            groups = [groups]

        return OIDCUserInfo(
            sub=claims.get("sub", ""),
            email=claims.get(self.email_claim, ""),
            email_verified=claims.get("email_verified", False),
            name=claims.get(self.name_claim, ""),
            given_name=claims.get("given_name", ""),
            family_name=claims.get("family_name", ""),
            preferred_username=claims.get("preferred_username", ""),
            picture=claims.get("picture", ""),
            locale=claims.get("locale", ""),
            zoneinfo=claims.get("zoneinfo", ""),
            groups=groups,
            raw_claims=claims,
        )

    def _parse_jwt_claims(self, token: str) -> Dict[str, Any]:
        """
        Parse JWT claims without verification.

        Note: For production, verify the signature using JWKS.
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return {}

            # Decode payload (second part)
            payload = parts[1]
            # Add padding if needed
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding

            decoded = base64.urlsafe_b64decode(payload)
            return json.loads(decoded)

        except Exception as e:
            logger.error(f"Failed to parse JWT: {e}")
            return {}

    async def validate_id_token(
        self,
        id_token: str,
        expected_nonce: str = "",
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate ID token.

        Note: This is a simplified validation. For production,
        implement full JWT verification with JWKS.

        Returns:
            Tuple of (is_valid, error_message, claims)
        """
        claims = self._parse_jwt_claims(id_token)

        if not claims:
            return False, "Failed to parse ID token", {}

        # Check issuer
        if claims.get("iss") != self.issuer:
            return False, f"Invalid issuer: expected {self.issuer}, got {claims.get('iss')}", claims

        # Check audience
        aud = claims.get("aud")
        if isinstance(aud, list):
            if self.client_id not in aud:
                return False, f"Invalid audience: {self.client_id} not in {aud}", claims
        elif aud != self.client_id:
            return False, f"Invalid audience: expected {self.client_id}, got {aud}", claims

        # Check expiration
        exp = claims.get("exp")
        if exp and int(time.time()) >= exp:
            return False, "ID token expired", claims

        # Check nonce
        if expected_nonce:
            if claims.get("nonce") != expected_nonce:
                return False, "Invalid nonce", claims

        return True, "", claims

    def create_logout_url(
        self,
        id_token_hint: str = "",
        post_logout_redirect_uri: str = "",
        state: str = "",
    ) -> str:
        """
        Create logout URL for RP-initiated logout.

        Args:
            id_token_hint: ID token to hint which user to log out
            post_logout_redirect_uri: Where to redirect after logout
            state: State parameter for callback

        Returns:
            Logout URL
        """
        if not self.end_session_endpoint:
            raise ValueError("End session endpoint not configured")

        params = {}

        if id_token_hint:
            params["id_token_hint"] = id_token_hint

        if post_logout_redirect_uri:
            params["post_logout_redirect_uri"] = post_logout_redirect_uri

        if state:
            params["state"] = state

        if params:
            return f"{self.end_session_endpoint}?{urlencode(params)}"

        return self.end_session_endpoint

    async def introspect_token(self, token: str) -> Dict[str, Any]:
        """
        Introspect a token (if introspection endpoint is available).

        Returns token metadata including active status.
        """
        if not self._discovery:
            await self.discover()

        introspection_endpoint = self._discovery.get("introspection_endpoint") if self._discovery else None

        if not introspection_endpoint:
            raise ValueError("Introspection endpoint not available")

        data = {
            "token": token,
            "client_id": self.client_id,
        }

        if self.client_secret:
            data["client_secret"] = self.client_secret

        async with aiohttp.ClientSession() as session:
            async with session.post(
                introspection_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ) as response:
                if response.status != 200:
                    error_body = await response.text()
                    raise ValueError(f"Token introspection failed: {response.status} - {error_body}")

                return await response.json()


# Common OIDC providers configuration helpers

def create_google_provider(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> OIDCProvider:
    """Create an OIDC provider configured for Google."""
    return OIDCProvider(
        issuer="https://accounts.google.com",
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scopes=["openid", "profile", "email"],
        use_pkce=True,
    )


def create_microsoft_provider(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    tenant_id: str = "common",
) -> OIDCProvider:
    """Create an OIDC provider configured for Microsoft/Azure AD."""
    return OIDCProvider(
        issuer=f"https://login.microsoftonline.com/{tenant_id}/v2.0",
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scopes=["openid", "profile", "email"],
        use_pkce=True,
        groups_claim="groups",
    )


def create_okta_provider(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    domain: str,  # e.g., "company.okta.com"
) -> OIDCProvider:
    """Create an OIDC provider configured for Okta."""
    return OIDCProvider(
        issuer=f"https://{domain}",
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scopes=["openid", "profile", "email", "groups"],
        use_pkce=True,
        groups_claim="groups",
    )


def create_auth0_provider(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    domain: str,  # e.g., "company.auth0.com"
) -> OIDCProvider:
    """Create an OIDC provider configured for Auth0."""
    return OIDCProvider(
        issuer=f"https://{domain}/",
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scopes=["openid", "profile", "email"],
        use_pkce=True,
    )


def create_keycloak_provider(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    server_url: str,
    realm: str,
) -> OIDCProvider:
    """Create an OIDC provider configured for Keycloak."""
    return OIDCProvider(
        issuer=f"{server_url}/realms/{realm}",
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scopes=["openid", "profile", "email"],
        use_pkce=True,
        groups_claim="groups",
    )
