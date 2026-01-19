"""JWT authentication support."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import os
import secrets
import hashlib
import hmac
import base64
import json


class TokenType(str, Enum):
    """Types of JWT tokens."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


@dataclass
class TokenClaims:
    """JWT token claims."""
    sub: str  # Subject (user ID)
    iat: datetime  # Issued at
    exp: datetime  # Expiration
    jti: str  # JWT ID
    token_type: TokenType
    scopes: List[str] = field(default_factory=list)
    org_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for encoding."""
        data = {
            "sub": self.sub,
            "iat": int(self.iat.timestamp()),
            "exp": int(self.exp.timestamp()),
            "jti": self.jti,
            "type": self.token_type.value,
        }
        if self.scopes:
            data["scopes"] = self.scopes
        if self.org_id:
            data["org_id"] = self.org_id
        if self.extra:
            data.update(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenClaims":
        """Create from dictionary."""
        return cls(
            sub=data["sub"],
            iat=datetime.fromtimestamp(data["iat"]),
            exp=datetime.fromtimestamp(data["exp"]),
            jti=data["jti"],
            token_type=TokenType(data.get("type", "access")),
            scopes=data.get("scopes", []),
            org_id=data.get("org_id"),
            extra={k: v for k, v in data.items()
                   if k not in {"sub", "iat", "exp", "jti", "type", "scopes", "org_id"}},
        )


class JWTError(Exception):
    """Base JWT error."""
    pass


class TokenExpiredError(JWTError):
    """Token has expired."""
    pass


class InvalidTokenError(JWTError):
    """Token is invalid."""
    pass


class JWTManager:
    """
    JWT token manager.

    Handles token creation, validation, and parsing.
    Uses HS256 for signing (HMAC-SHA256).
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        issuer: str = "bvrai",
        audience: str = "bvrai-api",
    ):
        self.secret_key = secret_key or os.environ.get(
            "JWT_SECRET_KEY",
            secrets.token_hex(32),
        )
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.issuer = issuer
        self.audience = audience

    def _base64url_encode(self, data: bytes) -> str:
        """Base64 URL encode."""
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

    def _base64url_decode(self, data: str) -> bytes:
        """Base64 URL decode."""
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    def _sign(self, message: str) -> str:
        """Create HMAC-SHA256 signature."""
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return self._base64url_encode(signature)

    def _verify_signature(self, message: str, signature: str) -> bool:
        """Verify HMAC-SHA256 signature."""
        expected = self._sign(message)
        return hmac.compare_digest(expected, signature)

    def create_token(
        self,
        user_id: str,
        token_type: TokenType = TokenType.ACCESS,
        scopes: Optional[List[str]] = None,
        org_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a JWT token."""
        now = datetime.utcnow()

        if token_type == TokenType.ACCESS:
            expires = now + timedelta(minutes=self.access_token_expire_minutes)
        elif token_type == TokenType.REFRESH:
            expires = now + timedelta(days=self.refresh_token_expire_days)
        else:  # API_KEY
            expires = now + timedelta(days=365 * 10)  # 10 years

        claims = TokenClaims(
            sub=user_id,
            iat=now,
            exp=expires,
            jti=secrets.token_hex(16),
            token_type=token_type,
            scopes=scopes or [],
            org_id=org_id,
            extra=extra or {},
        )

        # Create header
        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = self._base64url_encode(
            json.dumps(header, separators=(",", ":")).encode("utf-8")
        )

        # Create payload
        payload = claims.to_dict()
        payload["iss"] = self.issuer
        payload["aud"] = self.audience
        payload_b64 = self._base64url_encode(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        )

        # Sign
        message = f"{header_b64}.{payload_b64}"
        signature = self._sign(message)

        return f"{message}.{signature}"

    def decode_token(
        self,
        token: str,
        verify_exp: bool = True,
    ) -> TokenClaims:
        """Decode and verify a JWT token."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                raise InvalidTokenError("Invalid token format")

            header_b64, payload_b64, signature = parts

            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            if not self._verify_signature(message, signature):
                raise InvalidTokenError("Invalid signature")

            # Decode header
            header = json.loads(self._base64url_decode(header_b64))
            if header.get("alg") != "HS256":
                raise InvalidTokenError(f"Unsupported algorithm: {header.get('alg')}")

            # Decode payload
            payload = json.loads(self._base64url_decode(payload_b64))

            # Verify issuer and audience
            if payload.get("iss") != self.issuer:
                raise InvalidTokenError("Invalid issuer")
            if payload.get("aud") != self.audience:
                raise InvalidTokenError("Invalid audience")

            # Verify expiration
            if verify_exp:
                exp = datetime.fromtimestamp(payload["exp"])
                if datetime.utcnow() > exp:
                    raise TokenExpiredError("Token has expired")

            return TokenClaims.from_dict(payload)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise InvalidTokenError(f"Invalid token: {str(e)}")

    def refresh_token(self, refresh_token: str) -> tuple[str, str]:
        """
        Refresh an access token using a refresh token.

        Returns (new_access_token, new_refresh_token).
        """
        claims = self.decode_token(refresh_token)

        if claims.token_type != TokenType.REFRESH:
            raise InvalidTokenError("Not a refresh token")

        # Create new tokens
        access_token = self.create_token(
            user_id=claims.sub,
            token_type=TokenType.ACCESS,
            scopes=claims.scopes,
            org_id=claims.org_id,
        )

        new_refresh_token = self.create_token(
            user_id=claims.sub,
            token_type=TokenType.REFRESH,
            scopes=claims.scopes,
            org_id=claims.org_id,
        )

        return access_token, new_refresh_token

    def create_access_token(
        self,
        user_id: str,
        scopes: Optional[List[str]] = None,
        org_id: Optional[str] = None,
    ) -> str:
        """Create an access token."""
        return self.create_token(
            user_id=user_id,
            token_type=TokenType.ACCESS,
            scopes=scopes,
            org_id=org_id,
        )

    def create_refresh_token(
        self,
        user_id: str,
        scopes: Optional[List[str]] = None,
        org_id: Optional[str] = None,
    ) -> str:
        """Create a refresh token."""
        return self.create_token(
            user_id=user_id,
            token_type=TokenType.REFRESH,
            scopes=scopes,
            org_id=org_id,
        )


class APIKeyGenerator:
    """
    Generates and validates API keys.

    API keys follow the format: bvr_live_xxxx or bvr_test_xxxx
    """

    PREFIX_LIVE = "bvr_live_"
    PREFIX_TEST = "bvr_test_"

    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or os.environ.get(
            "API_KEY_SECRET",
            secrets.token_hex(32),
        )

    def generate(self, is_test: bool = False) -> tuple[str, str]:
        """
        Generate a new API key.

        Returns (raw_key, key_hash).
        """
        prefix = self.PREFIX_TEST if is_test else self.PREFIX_LIVE
        key_body = secrets.token_urlsafe(32)
        raw_key = f"{prefix}{key_body}"
        key_hash = self.hash_key(raw_key)
        return raw_key, key_hash

    def hash_key(self, raw_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(
            f"{raw_key}{self.secret_key}".encode("utf-8")
        ).hexdigest()

    def verify_key(self, raw_key: str, key_hash: str) -> bool:
        """Verify an API key against its hash."""
        return hmac.compare_digest(self.hash_key(raw_key), key_hash)

    def is_valid_format(self, key: str) -> bool:
        """Check if key has valid format."""
        return key.startswith(self.PREFIX_LIVE) or key.startswith(self.PREFIX_TEST)

    def is_test_key(self, key: str) -> bool:
        """Check if key is a test key."""
        return key.startswith(self.PREFIX_TEST)


# Global instances
_jwt_manager: Optional[JWTManager] = None
_api_key_generator: Optional[APIKeyGenerator] = None


def get_jwt_manager() -> JWTManager:
    """Get or create the global JWT manager."""
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager()
    return _jwt_manager


def get_api_key_generator() -> APIKeyGenerator:
    """Get or create the global API key generator."""
    global _api_key_generator
    if _api_key_generator is None:
        _api_key_generator = APIKeyGenerator()
    return _api_key_generator
