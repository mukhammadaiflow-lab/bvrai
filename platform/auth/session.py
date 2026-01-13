"""Session management for authentication."""

import asyncio
import hashlib
import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import redis.asyncio as redis

from .models import Session, AuthToken, User, Organization

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages user sessions and authentication tokens.

    Features:
    - Session creation and validation
    - Token generation and verification
    - Session refresh and expiration
    - Multi-device session tracking
    - Session revocation
    - Redis-backed storage
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        session_ttl_hours: int = 24,
        refresh_token_ttl_days: int = 30,
        access_token_ttl_minutes: int = 15,
        max_sessions_per_user: int = 10,
        jwt_secret: str = "",
        jwt_algorithm: str = "HS256",
    ):
        self.redis_url = redis_url
        self.session_ttl_hours = session_ttl_hours
        self.refresh_token_ttl_days = refresh_token_ttl_days
        self.access_token_ttl_minutes = access_token_ttl_minutes
        self.max_sessions_per_user = max_sessions_per_user
        self.jwt_secret = jwt_secret or secrets.token_hex(32)
        self.jwt_algorithm = jwt_algorithm

        self.redis: Optional[redis.Redis] = None

        # In-memory cache for active sessions
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the session manager."""
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        logger.info("Session manager started")

    async def stop(self) -> None:
        """Stop the session manager."""
        if self.redis:
            await self.redis.close()
        logger.info("Session manager stopped")

    # Session Management

    async def create_session(
        self,
        user: User,
        organization_id: str,
        ip_address: str = "",
        user_agent: str = "",
        device_id: str = "",
        device_name: str = "",
        identity_provider_id: str = "",
        sso_session_id: str = "",
        session_timeout_hours: Optional[int] = None,
    ) -> Tuple[Session, str, str]:
        """
        Create a new session for a user.

        Returns:
            Tuple of (session, session_token, refresh_token)
        """
        session_id = f"sess_{uuid4().hex[:16]}"
        session_token = secrets.token_urlsafe(48)
        refresh_token = secrets.token_urlsafe(64)

        ttl = session_timeout_hours or self.session_ttl_hours

        session = Session(
            id=session_id,
            user_id=user.id,
            organization_id=organization_id,
            token_hash=self._hash_token(session_token),
            refresh_token_hash=self._hash_token(refresh_token),
            ip_address=ip_address,
            user_agent=self._truncate_user_agent(user_agent),
            device_id=device_id,
            device_name=device_name,
            identity_provider_id=identity_provider_id,
            sso_session_id=sso_session_id,
            expires_at=datetime.utcnow() + timedelta(hours=ttl),
        )

        # Detect location from IP (placeholder)
        if ip_address:
            session.country, session.city = await self._get_location_from_ip(ip_address)

        # Enforce max sessions limit
        await self._enforce_session_limit(user.id)

        # Store session
        await self._store_session(session)

        logger.info(f"Created session {session_id} for user {user.id}")

        return session, session_token, refresh_token

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        # Check cache first
        async with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                if not session.is_expired and session.is_active:
                    return session
                else:
                    del self._sessions[session_id]

        # Load from Redis
        if not self.redis:
            return None

        data = await self.redis.get(f"session:{session_id}")
        if not data:
            return None

        try:
            session = self._deserialize_session(json.loads(data))
            if session.is_expired or not session.is_active:
                await self._delete_session(session_id)
                return None

            # Cache it
            async with self._lock:
                self._sessions[session_id] = session

            return session

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    async def validate_session_token(
        self,
        session_token: str,
    ) -> Optional[Session]:
        """
        Validate a session token and return the session.

        Also updates last_activity_at.
        """
        if not self.redis:
            return None

        token_hash = self._hash_token(session_token)

        # Look up session by token hash
        session_id = await self.redis.get(f"session_token:{token_hash}")
        if not session_id:
            return None

        session = await self.get_session(session_id)
        if not session:
            return None

        # Verify token hash matches
        if session.token_hash != token_hash:
            return None

        # Update last activity
        await self._update_session_activity(session)

        return session

    async def refresh_session(
        self,
        refresh_token: str,
    ) -> Tuple[Optional[Session], str, str]:
        """
        Refresh a session using a refresh token.

        Returns new session token and optionally new refresh token.

        Returns:
            Tuple of (session, new_session_token, new_refresh_token)
        """
        if not self.redis:
            return None, "", ""

        token_hash = self._hash_token(refresh_token)

        # Look up session by refresh token
        session_id = await self.redis.get(f"refresh_token:{token_hash}")
        if not session_id:
            return None, "", ""

        session = await self.get_session(session_id)
        if not session:
            return None, "", ""

        # Verify refresh token hash
        if session.refresh_token_hash != token_hash:
            return None, "", ""

        # Generate new tokens
        new_session_token = secrets.token_urlsafe(48)
        new_refresh_token = secrets.token_urlsafe(64)

        # Update session
        old_token_hash = session.token_hash
        old_refresh_hash = session.refresh_token_hash

        session.token_hash = self._hash_token(new_session_token)
        session.refresh_token_hash = self._hash_token(new_refresh_token)
        session.expires_at = datetime.utcnow() + timedelta(hours=self.session_ttl_hours)
        session.last_activity_at = datetime.utcnow()

        # Store updated session
        await self._store_session(session)

        # Remove old token mappings
        await self.redis.delete(f"session_token:{old_token_hash}")
        await self.redis.delete(f"refresh_token:{old_refresh_hash}")

        logger.info(f"Refreshed session {session.id}")

        return session, new_session_token, new_refresh_token

    async def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        session = await self.get_session(session_id)
        if not session:
            return False

        session.is_active = False

        # Delete from Redis
        await self._delete_session(session_id)

        # Remove from cache
        async with self._lock:
            self._sessions.pop(session_id, None)

        logger.info(f"Revoked session {session_id}")
        return True

    async def revoke_all_sessions(
        self,
        user_id: str,
        except_session_id: str = "",
    ) -> int:
        """Revoke all sessions for a user."""
        if not self.redis:
            return 0

        # Get all session IDs for user
        session_ids = await self.redis.smembers(f"user_sessions:{user_id}")

        count = 0
        for session_id in session_ids:
            if session_id == except_session_id:
                continue

            if await self.revoke_session(session_id):
                count += 1

        logger.info(f"Revoked {count} sessions for user {user_id}")
        return count

    async def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user."""
        if not self.redis:
            return []

        session_ids = await self.redis.smembers(f"user_sessions:{user_id}")
        sessions = []

        for session_id in session_ids:
            session = await self.get_session(session_id)
            if session:
                sessions.append(session)

        # Sort by last activity
        sessions.sort(key=lambda s: s.last_activity_at, reverse=True)

        return sessions

    # API Key Management

    async def create_api_key(
        self,
        user_id: str,
        organization_id: str,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        rate_limit: int = 0,
        allowed_ips: Optional[List[str]] = None,
    ) -> Tuple[AuthToken, str]:
        """
        Create an API key.

        Returns:
            Tuple of (auth_token, raw_api_key)
        """
        key_id = f"key_{uuid4().hex[:12]}"
        raw_key = f"be_{secrets.token_urlsafe(32)}"  # be_ prefix for Builder Engine

        auth_token = AuthToken(
            id=key_id,
            user_id=user_id,
            organization_id=organization_id,
            token_type="api_key",
            name=name,
            prefix=raw_key[:10],
            token_hash=self._hash_token(raw_key),
            scopes=scopes or [],
            rate_limit=rate_limit,
            allowed_ips=allowed_ips or [],
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days) if expires_in_days else None,
        )

        # Store API key
        if self.redis:
            await self.redis.hset(
                f"api_keys:{organization_id}",
                key_id,
                json.dumps(self._serialize_auth_token(auth_token)),
            )

            # Index by token hash for lookup
            await self.redis.set(
                f"api_key:{auth_token.token_hash}",
                json.dumps({"id": key_id, "org_id": organization_id}),
            )

        logger.info(f"Created API key {key_id} for user {user_id}")

        return auth_token, raw_key

    async def validate_api_key(
        self,
        api_key: str,
        required_scope: str = "",
    ) -> Optional[AuthToken]:
        """
        Validate an API key.

        Returns AuthToken if valid, None otherwise.
        """
        if not self.redis:
            return None

        token_hash = self._hash_token(api_key)

        # Look up key
        key_ref = await self.redis.get(f"api_key:{token_hash}")
        if not key_ref:
            return None

        try:
            ref = json.loads(key_ref)
            key_data = await self.redis.hget(f"api_keys:{ref['org_id']}", ref['id'])
            if not key_data:
                return None

            auth_token = self._deserialize_auth_token(json.loads(key_data))

            # Check if active
            if not auth_token.is_active:
                return None

            # Check expiration
            if auth_token.expires_at and datetime.utcnow() >= auth_token.expires_at:
                return None

            # Check scope
            if required_scope and required_scope not in auth_token.scopes:
                # Check for wildcard scopes
                if "*" not in auth_token.scopes:
                    return None

            # Update last used
            auth_token.last_used_at = datetime.utcnow()
            await self.redis.hset(
                f"api_keys:{ref['org_id']}",
                ref['id'],
                json.dumps(self._serialize_auth_token(auth_token)),
            )

            return auth_token

        except Exception as e:
            logger.error(f"Failed to validate API key: {e}")
            return None

    async def revoke_api_key(
        self,
        organization_id: str,
        key_id: str,
    ) -> bool:
        """Revoke an API key."""
        if not self.redis:
            return False

        key_data = await self.redis.hget(f"api_keys:{organization_id}", key_id)
        if not key_data:
            return False

        try:
            auth_token = self._deserialize_auth_token(json.loads(key_data))

            # Remove key hash index
            await self.redis.delete(f"api_key:{auth_token.token_hash}")

            # Remove key
            await self.redis.hdel(f"api_keys:{organization_id}", key_id)

            logger.info(f"Revoked API key {key_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to revoke API key: {e}")
            return False

    async def list_api_keys(self, organization_id: str) -> List[AuthToken]:
        """List all API keys for an organization."""
        if not self.redis:
            return []

        keys_data = await self.redis.hgetall(f"api_keys:{organization_id}")
        keys = []

        for key_data in keys_data.values():
            try:
                auth_token = self._deserialize_auth_token(json.loads(key_data))
                keys.append(auth_token)
            except Exception:
                continue

        # Sort by creation date
        keys.sort(key=lambda k: k.created_at, reverse=True)

        return keys

    # JWT Token Generation (for short-lived access tokens)

    def generate_access_token(
        self,
        user_id: str,
        organization_id: str,
        session_id: str,
        scopes: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a short-lived JWT access token.

        Note: For production, use a proper JWT library like PyJWT.
        """
        import base64
        import hmac

        header = {
            "alg": self.jwt_algorithm,
            "typ": "JWT",
        }

        now = int(datetime.utcnow().timestamp())
        payload = {
            "sub": user_id,
            "org": organization_id,
            "sid": session_id,
            "iat": now,
            "exp": now + (self.access_token_ttl_minutes * 60),
            "scopes": scopes or [],
        }

        # Encode header and payload
        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header).encode()
        ).decode().rstrip("=")

        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")

        # Sign
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.jwt_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).digest()

        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")

        return f"{message}.{signature_b64}"

    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify a JWT access token.

        Returns payload if valid, None otherwise.
        """
        import base64
        import hmac

        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts

            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            expected_signature = hmac.new(
                self.jwt_secret.encode(),
                message.encode(),
                hashlib.sha256,
            ).digest()

            # Add padding for base64
            signature_b64_padded = signature_b64 + "=" * (4 - len(signature_b64) % 4)
            actual_signature = base64.urlsafe_b64decode(signature_b64_padded)

            if not hmac.compare_digest(expected_signature, actual_signature):
                return None

            # Decode payload
            payload_b64_padded = payload_b64 + "=" * (4 - len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64_padded))

            # Check expiration
            if payload.get("exp", 0) < int(datetime.utcnow().timestamp()):
                return None

            return payload

        except Exception as e:
            logger.debug(f"Token verification failed: {e}")
            return None

    # Helper methods

    def _hash_token(self, token: str) -> str:
        """Hash a token for secure storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    def _truncate_user_agent(self, user_agent: str, max_length: int = 256) -> str:
        """Truncate user agent string."""
        if len(user_agent) <= max_length:
            return user_agent
        return user_agent[:max_length - 3] + "..."

    async def _get_location_from_ip(self, ip_address: str) -> Tuple[str, str]:
        """Get country and city from IP address (placeholder)."""
        # In production, use a GeoIP service like MaxMind
        return "", ""

    async def _enforce_session_limit(self, user_id: str) -> None:
        """Enforce maximum sessions per user."""
        if not self.redis:
            return

        sessions = await self.get_user_sessions(user_id)
        if len(sessions) >= self.max_sessions_per_user:
            # Revoke oldest sessions
            sessions_to_revoke = sessions[self.max_sessions_per_user - 1:]
            for session in sessions_to_revoke:
                await self.revoke_session(session.id)

    async def _store_session(self, session: Session) -> None:
        """Store session in Redis."""
        if not self.redis:
            return

        session_data = self._serialize_session(session)
        ttl_seconds = int((session.expires_at - datetime.utcnow()).total_seconds())

        if ttl_seconds <= 0:
            return

        pipe = self.redis.pipeline()

        # Store session
        pipe.setex(
            f"session:{session.id}",
            ttl_seconds,
            json.dumps(session_data),
        )

        # Index by token hash
        pipe.setex(
            f"session_token:{session.token_hash}",
            ttl_seconds,
            session.id,
        )

        # Index by refresh token hash
        refresh_ttl = self.refresh_token_ttl_days * 86400
        pipe.setex(
            f"refresh_token:{session.refresh_token_hash}",
            refresh_ttl,
            session.id,
        )

        # Add to user's sessions
        pipe.sadd(f"user_sessions:{session.user_id}", session.id)

        await pipe.execute()

        # Cache
        async with self._lock:
            self._sessions[session.id] = session

    async def _delete_session(self, session_id: str) -> None:
        """Delete session from Redis."""
        if not self.redis:
            return

        session = await self.get_session(session_id)
        if not session:
            return

        pipe = self.redis.pipeline()

        pipe.delete(f"session:{session_id}")
        pipe.delete(f"session_token:{session.token_hash}")
        pipe.delete(f"refresh_token:{session.refresh_token_hash}")
        pipe.srem(f"user_sessions:{session.user_id}", session_id)

        await pipe.execute()

    async def _update_session_activity(self, session: Session) -> None:
        """Update session last activity timestamp."""
        session.last_activity_at = datetime.utcnow()

        if self.redis:
            ttl_seconds = int((session.expires_at - datetime.utcnow()).total_seconds())
            if ttl_seconds > 0:
                await self.redis.setex(
                    f"session:{session.id}",
                    ttl_seconds,
                    json.dumps(self._serialize_session(session)),
                )

        # Update cache
        async with self._lock:
            self._sessions[session.id] = session

    def _serialize_session(self, session: Session) -> Dict[str, Any]:
        """Serialize session for storage."""
        return {
            "id": session.id,
            "user_id": session.user_id,
            "organization_id": session.organization_id,
            "token_hash": session.token_hash,
            "refresh_token_hash": session.refresh_token_hash,
            "ip_address": session.ip_address,
            "user_agent": session.user_agent,
            "device_id": session.device_id,
            "device_name": session.device_name,
            "country": session.country,
            "city": session.city,
            "identity_provider_id": session.identity_provider_id,
            "sso_session_id": session.sso_session_id,
            "is_active": session.is_active,
            "created_at": session.created_at.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "last_activity_at": session.last_activity_at.isoformat(),
        }

    def _deserialize_session(self, data: Dict[str, Any]) -> Session:
        """Deserialize session from storage."""
        return Session(
            id=data["id"],
            user_id=data["user_id"],
            organization_id=data["organization_id"],
            token_hash=data["token_hash"],
            refresh_token_hash=data["refresh_token_hash"],
            ip_address=data.get("ip_address", ""),
            user_agent=data.get("user_agent", ""),
            device_id=data.get("device_id", ""),
            device_name=data.get("device_name", ""),
            country=data.get("country", ""),
            city=data.get("city", ""),
            identity_provider_id=data.get("identity_provider_id", ""),
            sso_session_id=data.get("sso_session_id", ""),
            is_active=data.get("is_active", True),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            last_activity_at=datetime.fromisoformat(data["last_activity_at"]),
        )

    def _serialize_auth_token(self, token: AuthToken) -> Dict[str, Any]:
        """Serialize auth token for storage."""
        return {
            **token.to_dict(),
            "created_at": token.created_at.isoformat(),
            "expires_at": token.expires_at.isoformat() if token.expires_at else None,
            "last_used_at": token.last_used_at.isoformat() if token.last_used_at else None,
        }

    def _deserialize_auth_token(self, data: Dict[str, Any]) -> AuthToken:
        """Deserialize auth token from storage."""
        return AuthToken(
            id=data["id"],
            user_id=data["user_id"],
            organization_id=data["organization_id"],
            token_type=data.get("token_type", "api_key"),
            name=data.get("name", ""),
            prefix=data.get("prefix", ""),
            token_hash=data.get("token_hash", ""),
            scopes=data.get("scopes", []),
            rate_limit=data.get("rate_limit", 0),
            allowed_ips=data.get("allowed_ips", []),
            is_active=data.get("is_active", True),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            last_used_at=datetime.fromisoformat(data["last_used_at"]) if data.get("last_used_at") else None,
        )
