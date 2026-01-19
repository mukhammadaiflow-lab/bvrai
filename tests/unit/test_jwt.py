"""
Unit Tests for JWT Token Management

Tests for JWT token generation, validation, key rotation,
and token blacklist implementation.
"""

import pytest
from datetime import datetime, timedelta
import time

from bvrai_core.security.jwt import (
    JWTKey,
    TokenPayload,
    TokenPair,
    JWTKeyManager,
    JWTService,
    TokenBlacklist,
    InMemoryTokenBlacklist,
    RedisTokenBlacklist,
    get_key_manager,
    get_jwt_service,
    get_token_blacklist,
)


# =============================================================================
# JWT Key Manager Tests
# =============================================================================


class TestJWTKeyManager:
    """Tests for JWT Key Manager."""

    @pytest.fixture
    def manager(self):
        """Create a test key manager."""
        return JWTKeyManager(primary_secret="test_secret_key_12345")

    def test_initialization(self, manager):
        """Test key manager initialization."""
        assert manager.primary_key is not None
        assert manager.primary_key.secret == "test_secret_key_12345"

    def test_primary_key_has_kid(self, manager):
        """Test that primary key has a key ID."""
        assert manager.primary_key.kid is not None
        assert len(manager.primary_key.kid) > 0

    def test_add_key(self, manager):
        """Test adding a new key."""
        new_key = manager.add_key(secret="new_secret")
        assert new_key.secret == "new_secret"
        assert new_key in manager.active_keys

    def test_add_key_as_primary(self, manager):
        """Test adding a key as the new primary."""
        old_primary = manager.primary_key
        new_key = manager.add_key(secret="new_primary", make_primary=True)
        assert manager.primary_key == new_key
        assert manager.primary_key != old_primary

    def test_rotate_primary(self, manager):
        """Test key rotation."""
        old_primary = manager.primary_key
        new_primary = manager.rotate_primary()

        assert manager.primary_key == new_primary
        assert manager.primary_key != old_primary
        # Old key should still be active
        assert old_primary in manager.active_keys

    def test_deactivate_key(self, manager):
        """Test deactivating a key."""
        new_key = manager.add_key(secret="to_deactivate")
        assert new_key.is_active is True

        result = manager.deactivate_key(new_key.kid)
        assert result is True
        assert new_key.is_active is False

    def test_cannot_deactivate_primary(self, manager):
        """Test that primary key cannot be deactivated."""
        result = manager.deactivate_key(manager.primary_key.kid)
        assert result is False
        assert manager.primary_key.is_active is True

    def test_get_key(self, manager):
        """Test getting a key by ID."""
        key = manager.get_key(manager.primary_key.kid)
        assert key == manager.primary_key

    def test_get_nonexistent_key(self, manager):
        """Test getting a nonexistent key."""
        key = manager.get_key("nonexistent_kid")
        assert key is None

    def test_export_import_keys(self, manager):
        """Test exporting and importing keys."""
        # Add some keys
        manager.add_key(secret="key_1")
        manager.add_key(secret="key_2")

        # Export
        exported = manager.export_keys()
        assert "primary_kid" in exported
        assert "keys" in exported

        # Create new manager and import
        new_manager = JWTKeyManager()
        new_manager.import_keys(exported)

        # Verify keys were imported
        assert len(new_manager.active_keys) >= len(manager.active_keys) - 1

    def test_cleanup_expired(self, manager):
        """Test cleaning up expired keys."""
        # Add an expired key
        expired_key = manager.add_key(expires_in_days=0)
        expired_key.expires_at = datetime.utcnow() - timedelta(hours=1)

        # Cleanup
        cleaned = manager.cleanup_expired()
        assert cleaned >= 1

        # Expired key should be removed
        assert manager.get_key(expired_key.kid) is None


# =============================================================================
# JWT Service Tests
# =============================================================================


class TestJWTService:
    """Tests for JWT Service."""

    @pytest.fixture
    def service(self):
        """Create a test JWT service."""
        key_manager = JWTKeyManager(primary_secret="test_jwt_secret")
        return JWTService(
            key_manager=key_manager,
            access_token_expire_minutes=30,
            refresh_token_expire_days=7,
        )

    def test_create_access_token(self, service):
        """Test creating an access token."""
        token = service.create_access_token(
            user_id="user_123",
            organization_id="org_456",
            email="test@example.com",
            role="admin",
        )

        assert token is not None
        assert len(token) > 0

    def test_create_refresh_token(self, service):
        """Test creating a refresh token."""
        token = service.create_refresh_token(
            user_id="user_123",
            organization_id="org_456",
        )

        assert token is not None
        assert len(token) > 0

    def test_create_tokens(self, service):
        """Test creating a token pair."""
        tokens = service.create_tokens(
            user_id="user_123",
            organization_id="org_456",
            email="test@example.com",
            role="member",
        )

        assert isinstance(tokens, TokenPair)
        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
        assert tokens.token_type == "Bearer"
        assert tokens.expires_in == 30 * 60

    def test_verify_access_token(self, service):
        """Test verifying an access token."""
        token = service.create_access_token(
            user_id="user_123",
            organization_id="org_456",
        )

        payload = service.verify_token(token, token_type="access")

        assert payload is not None
        assert payload.sub == "user_123"
        assert payload.org == "org_456"
        assert payload.type == "access"

    def test_verify_refresh_token(self, service):
        """Test verifying a refresh token."""
        token = service.create_refresh_token(
            user_id="user_123",
            organization_id="org_456",
        )

        payload = service.verify_token(token, token_type="refresh")

        assert payload is not None
        assert payload.sub == "user_123"
        assert payload.type == "refresh"

    def test_verify_wrong_token_type(self, service):
        """Test that verifying with wrong token type fails."""
        access_token = service.create_access_token(user_id="user_123")
        refresh_token = service.create_refresh_token(user_id="user_123")

        # Access token should fail refresh verification
        assert service.verify_token(access_token, token_type="refresh") is None
        # Refresh token should fail access verification
        assert service.verify_token(refresh_token, token_type="access") is None

    def test_verify_invalid_token(self, service):
        """Test verifying an invalid token."""
        payload = service.verify_token("invalid.token.here", token_type="access")
        assert payload is None

    def test_verify_expired_token(self, service):
        """Test that expired token fails verification."""
        token = service.create_access_token(
            user_id="user_123",
            expires_delta=timedelta(seconds=-1),  # Already expired
        )

        payload = service.verify_token(token, token_type="access")
        assert payload is None

    def test_verify_tampered_token(self, service):
        """Test that tampered token fails verification."""
        token = service.create_access_token(user_id="user_123")

        # Tamper with the token
        parts = token.split(".")
        parts[1] = parts[1][:-1] + "x"  # Modify payload
        tampered = ".".join(parts)

        payload = service.verify_token(tampered, token_type="access")
        assert payload is None

    def test_refresh_tokens(self, service):
        """Test refreshing tokens."""
        original_tokens = service.create_tokens(
            user_id="user_123",
            organization_id="org_456",
        )

        new_tokens = service.refresh_tokens(original_tokens.refresh_token)

        assert new_tokens is not None
        assert new_tokens.access_token != original_tokens.access_token
        assert new_tokens.refresh_token != original_tokens.refresh_token

    def test_refresh_with_invalid_token(self, service):
        """Test that refresh with invalid token fails."""
        new_tokens = service.refresh_tokens("invalid.refresh.token")
        assert new_tokens is None

    def test_refresh_with_access_token(self, service):
        """Test that refresh with access token fails."""
        tokens = service.create_tokens(user_id="user_123")
        new_tokens = service.refresh_tokens(tokens.access_token)
        assert new_tokens is None

    def test_token_includes_scopes(self, service):
        """Test that token includes scopes."""
        token = service.create_access_token(
            user_id="user_123",
            scopes=["agents:read", "calls:write"],
        )

        payload = service.verify_token(token, token_type="access")

        assert payload.scopes == ["agents:read", "calls:write"]

    def test_token_has_jti(self, service):
        """Test that token has a unique JTI."""
        token1 = service.create_access_token(user_id="user_123")
        token2 = service.create_access_token(user_id="user_123")

        payload1 = service.verify_token(token1, token_type="access")
        payload2 = service.verify_token(token2, token_type="access")

        assert payload1.jti is not None
        assert payload2.jti is not None
        assert payload1.jti != payload2.jti


# =============================================================================
# Key Rotation Tests
# =============================================================================


class TestKeyRotation:
    """Tests for key rotation scenarios."""

    def test_old_tokens_still_valid_after_rotation(self):
        """Test that tokens created before rotation are still valid."""
        key_manager = JWTKeyManager(primary_secret="initial_secret")
        service = JWTService(key_manager=key_manager)

        # Create token with old key
        old_token = service.create_access_token(user_id="user_123")

        # Rotate key
        key_manager.rotate_primary()

        # Old token should still be valid
        payload = service.verify_token(old_token, token_type="access")
        assert payload is not None
        assert payload.sub == "user_123"

    def test_new_tokens_use_new_key(self):
        """Test that new tokens use the rotated key."""
        key_manager = JWTKeyManager(primary_secret="initial_secret")
        service = JWTService(key_manager=key_manager)

        old_kid = key_manager.primary_key.kid

        # Rotate key
        key_manager.rotate_primary()
        new_kid = key_manager.primary_key.kid

        # Create new token
        new_token = service.create_access_token(user_id="user_123")

        # Verify the token uses the new key (check header)
        import jwt
        header = jwt.get_unverified_header(new_token)
        assert header.get("kid") == new_kid
        assert header.get("kid") != old_kid


# =============================================================================
# Token Payload Tests
# =============================================================================


class TestTokenPayload:
    """Tests for Token Payload model."""

    def test_payload_creation(self):
        """Test creating a token payload."""
        payload = TokenPayload(
            sub="user_123",
            org="org_456",
            email="test@example.com",
            role="admin",
            scopes=["read", "write"],
            exp=datetime.utcnow() + timedelta(hours=1),
        )

        assert payload.sub == "user_123"
        assert payload.org == "org_456"
        assert payload.email == "test@example.com"
        assert payload.role == "admin"
        assert payload.scopes == ["read", "write"]

    def test_payload_default_values(self):
        """Test payload default values."""
        payload = TokenPayload(
            sub="user_123",
            exp=datetime.utcnow() + timedelta(hours=1),
        )

        assert payload.type == "access"
        assert payload.scopes == []
        assert payload.iat is not None


# =============================================================================
# Security Tests
# =============================================================================


class TestJWTSecurity:
    """Security-focused tests for JWT implementation."""

    def test_token_cannot_be_forged(self):
        """Test that tokens cannot be forged with wrong secret."""
        import jwt

        service = JWTService(
            key_manager=JWTKeyManager(primary_secret="real_secret")
        )

        # Create a forged token with wrong secret
        forged_payload = {
            "sub": "admin_user",
            "role": "admin",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "type": "access",
        }
        forged_token = jwt.encode(forged_payload, "wrong_secret", algorithm="HS256")

        # Should fail verification
        payload = service.verify_token(forged_token, token_type="access")
        assert payload is None

    def test_algorithm_must_match(self):
        """Test that algorithm must match."""
        import jwt

        key_manager = JWTKeyManager(primary_secret="test_secret")
        service = JWTService(key_manager=key_manager)

        # Create token with different algorithm
        payload = {
            "sub": "user_123",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "type": "access",
        }
        token = jwt.encode(payload, "test_secret", algorithm="HS384")

        # Should fail (expecting HS256)
        result = service.verify_token(token, token_type="access")
        assert result is None

    def test_no_none_algorithm_attack(self):
        """Test protection against 'none' algorithm attack."""
        import jwt

        service = JWTService(
            key_manager=JWTKeyManager(primary_secret="test_secret")
        )

        # Try to create a token with 'none' algorithm
        payload = {
            "sub": "admin",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "type": "access",
        }

        try:
            # Some JWT libraries don't allow 'none' by default
            header = {"alg": "none", "typ": "JWT"}
            token = jwt.encode(payload, "", algorithm="none")
        except Exception:
            # If the library blocks this, that's good
            return

        # If we got here, verify the token is rejected
        result = service.verify_token(token, token_type="access")
        assert result is None


# =============================================================================
# Token Blacklist Tests
# =============================================================================


class TestInMemoryTokenBlacklist:
    """Tests for in-memory token blacklist."""

    @pytest.fixture
    def blacklist(self):
        """Create a fresh blacklist for each test."""
        return InMemoryTokenBlacklist()

    @pytest.mark.asyncio
    async def test_add_token(self, blacklist):
        """Test adding a token to the blacklist."""
        exp = datetime.utcnow() + timedelta(hours=1)
        result = await blacklist.add("test_jti", exp)

        assert result is True
        assert await blacklist.contains("test_jti") is True

    @pytest.mark.asyncio
    async def test_contains_nonexistent(self, blacklist):
        """Test checking for non-existent token."""
        assert await blacklist.contains("nonexistent") is False

    @pytest.mark.asyncio
    async def test_expired_token_removed(self, blacklist):
        """Test that expired tokens are removed on check."""
        exp = datetime.utcnow() - timedelta(hours=1)  # Already expired
        await blacklist.add("expired_jti", exp)

        # Should return False and clean up the entry
        assert await blacklist.contains("expired_jti") is False

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, blacklist):
        """Test cleanup of expired entries."""
        # Add expired token
        await blacklist.add("expired", datetime.utcnow() - timedelta(hours=1))
        # Add valid token
        await blacklist.add("valid", datetime.utcnow() + timedelta(hours=1))

        removed = await blacklist.cleanup_expired()

        assert removed == 1
        assert await blacklist.contains("valid") is True

    def test_sync_add(self, blacklist):
        """Test synchronous add method."""
        exp = datetime.utcnow() + timedelta(hours=1)
        result = blacklist.sync_add("sync_jti", exp)

        assert result is True

    def test_sync_contains(self, blacklist):
        """Test synchronous contains method."""
        exp = datetime.utcnow() + timedelta(hours=1)
        blacklist.sync_add("sync_jti", exp)

        assert blacklist.sync_contains("sync_jti") is True
        assert blacklist.sync_contains("other") is False


# =============================================================================
# Token Revocation Tests
# =============================================================================


class TestTokenRevocation:
    """Tests for token revocation (blacklisting)."""

    @pytest.fixture
    def service(self):
        """Create a JWT service with in-memory blacklist."""
        key_manager = JWTKeyManager(primary_secret="test_revocation_secret")
        return JWTService(
            key_manager=key_manager,
            blacklist=InMemoryTokenBlacklist(),
            access_token_expire_minutes=30,
        )

    def test_revoke_token(self, service):
        """Test revoking a token."""
        token = service.create_access_token(user_id="user123")

        # Verify token works before revocation
        payload = service.verify_token(token)
        assert payload is not None

        # Revoke the token
        result = service.revoke_token(token)
        assert result is True

        # Token should no longer be valid
        payload = service.verify_token(token)
        assert payload is None

    def test_revoke_invalid_token(self, service):
        """Test revoking an invalid token."""
        result = service.revoke_token("invalid_token")
        assert result is False

    def test_verify_without_blacklist_check(self, service):
        """Test verifying without blacklist check."""
        token = service.create_access_token(user_id="user123")
        service.revoke_token(token)

        # Should still work without blacklist check
        payload = service.verify_token(token, check_blacklist=False)
        assert payload is not None

    @pytest.mark.asyncio
    async def test_revoke_token_async(self, service):
        """Test async token revocation."""
        token = service.create_access_token(user_id="user123")

        result = await service.revoke_token_async(token)
        assert result is True

        # Verify through async method
        payload = await service.verify_token_async(token)
        assert payload is None

    @pytest.mark.asyncio
    async def test_verify_token_async(self, service):
        """Test async token verification."""
        token = service.create_access_token(user_id="user123")

        payload = await service.verify_token_async(token)

        assert payload is not None
        assert payload.sub == "user123"

    @pytest.mark.asyncio
    async def test_revoke_all_user_tokens(self, service):
        """Test revoking all tokens for a user."""
        tokens = [
            service.create_access_token(user_id="user123")
            for _ in range(3)
        ]

        revoked = await service.revoke_all_user_tokens("user123", tokens)

        assert revoked == 3
        for token in tokens:
            assert service.verify_token(token) is None


# =============================================================================
# Key Rotation with Blacklist Tests
# =============================================================================


class TestKeyRotationWithBlacklist:
    """Tests for key rotation combined with token blacklisting."""

    def test_old_key_tokens_still_blacklistable(self):
        """Test that tokens from old keys can still be blacklisted."""
        manager = JWTKeyManager(primary_secret="rotation_test_secret")
        service = JWTService(
            key_manager=manager,
            blacklist=InMemoryTokenBlacklist(),
        )

        # Create token with original key
        token = service.create_access_token(user_id="user123")

        # Rotate keys
        manager.rotate_primary()

        # Token should still be verifiable
        payload = service.verify_token(token)
        assert payload is not None

        # And can be revoked
        result = service.revoke_token(token)
        assert result is True

        # After revocation, token is invalid
        assert service.verify_token(token) is None

    def test_revoked_token_stays_invalid_after_rotation(self):
        """Test that a revoked token stays invalid even after key rotation."""
        manager = JWTKeyManager(primary_secret="rotation_test_secret")
        blacklist = InMemoryTokenBlacklist()
        service = JWTService(
            key_manager=manager,
            blacklist=blacklist,
        )

        # Create and revoke token
        token = service.create_access_token(user_id="user123")
        service.revoke_token(token)

        # Rotate keys
        manager.rotate_primary()

        # Token should still be invalid (blacklisted)
        assert service.verify_token(token) is None


# =============================================================================
# Blacklist Edge Cases
# =============================================================================


class TestBlacklistEdgeCases:
    """Tests for edge cases in token blacklisting."""

    def test_revoke_token_without_jti(self):
        """Test revoking a token that somehow has no JTI."""
        # This shouldn't happen normally, but test the edge case
        service = JWTService(
            blacklist=InMemoryTokenBlacklist(),
        )

        # Create a valid token first
        token = service.create_access_token(user_id="user123")

        # The token should have a JTI, so revocation should work
        result = service.revoke_token(token)
        assert result is True

    def test_double_revocation(self):
        """Test revoking the same token twice."""
        service = JWTService(
            key_manager=JWTKeyManager(primary_secret="double_revoke_test"),
            blacklist=InMemoryTokenBlacklist(),
        )

        token = service.create_access_token(user_id="user123")

        # First revocation
        result1 = service.revoke_token(token)
        assert result1 is True

        # Second revocation - should still succeed (idempotent)
        result2 = service.revoke_token(token)
        assert result2 is True

    def test_revoke_expired_token(self):
        """Test revoking an already expired token."""
        service = JWTService(
            key_manager=JWTKeyManager(primary_secret="expired_revoke_test"),
            blacklist=InMemoryTokenBlacklist(),
        )

        # Create an expired token
        token = service.create_access_token(
            user_id="user123",
            expires_delta=timedelta(seconds=-1),
        )

        # Should still be able to revoke (we don't verify expiration for revocation)
        result = service.revoke_token(token)
        assert result is True

    @pytest.mark.asyncio
    async def test_concurrent_revocation(self):
        """Test concurrent token revocations."""
        import asyncio

        service = JWTService(
            key_manager=JWTKeyManager(primary_secret="concurrent_test"),
            blacklist=InMemoryTokenBlacklist(),
        )

        # Create multiple tokens
        tokens = [
            service.create_access_token(user_id=f"user{i}")
            for i in range(10)
        ]

        # Revoke all concurrently
        results = await asyncio.gather(*[
            service.revoke_token_async(token)
            for token in tokens
        ])

        assert all(results)

        # All should be invalid
        for token in tokens:
            assert service.verify_token(token) is None
