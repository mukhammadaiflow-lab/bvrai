"""
Unit Tests for API Key Management

Tests for secure API key generation, hashing, and validation.
"""

import pytest
from datetime import datetime, timedelta

from bvrai_core.security.api_keys import (
    API_KEY_PREFIX,
    APIKeyMetadata,
    APIKeyData,
    generate_api_key,
    hash_api_key_pbkdf2,
    verify_api_key_pbkdf2,
    validate_api_key_format,
    extract_key_prefix,
    APIKeyService,
)


# =============================================================================
# Key Generation Tests
# =============================================================================


class TestGenerateApiKey:
    """Tests for API key generation."""

    def test_generate_key_returns_tuple(self):
        """Test that generate_api_key returns a tuple."""
        result = generate_api_key()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_generate_key_has_correct_prefix(self):
        """Test that generated key has the correct prefix."""
        full_key, key_prefix = generate_api_key()
        assert full_key.startswith(API_KEY_PREFIX)

    def test_generate_key_prefix_matches(self):
        """Test that key prefix matches the full key."""
        full_key, key_prefix = generate_api_key()
        key_body = full_key[len(API_KEY_PREFIX):]
        assert key_body.startswith(key_prefix)

    def test_generate_key_is_unique(self):
        """Test that generated keys are unique."""
        keys = [generate_api_key()[0] for _ in range(100)]
        assert len(set(keys)) == 100

    def test_generate_key_length(self):
        """Test that generated key has expected length."""
        full_key, _ = generate_api_key()
        # Prefix + base64 encoded 32 bytes (approximately 43 chars)
        assert len(full_key) >= len(API_KEY_PREFIX) + 40


# =============================================================================
# Key Hashing Tests (PBKDF2)
# =============================================================================


class TestHashApiKeyPbkdf2:
    """Tests for PBKDF2 key hashing."""

    def test_hash_returns_tuple(self):
        """Test that hash returns a tuple of hash and salt."""
        full_key, _ = generate_api_key()
        result = hash_api_key_pbkdf2(full_key)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_hash_is_deterministic_with_same_salt(self):
        """Test that same key + salt produces same hash."""
        full_key, _ = generate_api_key()
        hash1, salt = hash_api_key_pbkdf2(full_key)
        hash2, _ = hash_api_key_pbkdf2(full_key, bytes.fromhex(salt))
        assert hash1 == hash2

    def test_hash_is_different_with_different_salt(self):
        """Test that same key produces different hash with different salt."""
        full_key, _ = generate_api_key()
        hash1, _ = hash_api_key_pbkdf2(full_key)
        hash2, _ = hash_api_key_pbkdf2(full_key)
        # Different auto-generated salts should produce different hashes
        # Note: This might occasionally fail if salts happen to be identical
        # which is astronomically unlikely
        assert hash1 != hash2

    def test_different_keys_produce_different_hashes(self):
        """Test that different keys produce different hashes."""
        key1, _ = generate_api_key()
        key2, _ = generate_api_key()
        hash1, salt1 = hash_api_key_pbkdf2(key1)
        hash2, salt2 = hash_api_key_pbkdf2(key2)
        assert hash1 != hash2


# =============================================================================
# Key Verification Tests (PBKDF2)
# =============================================================================


class TestVerifyApiKeyPbkdf2:
    """Tests for PBKDF2 key verification."""

    def test_verify_valid_key(self):
        """Test that valid key verifies successfully."""
        full_key, _ = generate_api_key()
        key_hash, salt = hash_api_key_pbkdf2(full_key)
        assert verify_api_key_pbkdf2(full_key, key_hash, salt) is True

    def test_verify_invalid_key(self):
        """Test that invalid key fails verification."""
        full_key1, _ = generate_api_key()
        full_key2, _ = generate_api_key()
        key_hash, salt = hash_api_key_pbkdf2(full_key1)
        assert verify_api_key_pbkdf2(full_key2, key_hash, salt) is False

    def test_verify_wrong_hash(self):
        """Test that wrong hash fails verification."""
        full_key, _ = generate_api_key()
        _, salt = hash_api_key_pbkdf2(full_key)
        assert verify_api_key_pbkdf2(full_key, "wronghash", salt) is False

    def test_verify_wrong_salt(self):
        """Test that wrong salt fails verification."""
        full_key, _ = generate_api_key()
        key_hash, _ = hash_api_key_pbkdf2(full_key)
        wrong_salt = "00" * 16
        assert verify_api_key_pbkdf2(full_key, key_hash, wrong_salt) is False


# =============================================================================
# Key Format Validation Tests
# =============================================================================


class TestValidateApiKeyFormat:
    """Tests for API key format validation."""

    def test_valid_key_format(self):
        """Test that generated key has valid format."""
        full_key, _ = generate_api_key()
        assert validate_api_key_format(full_key) is True

    def test_empty_key_invalid(self):
        """Test that empty key is invalid."""
        assert validate_api_key_format("") is False

    def test_none_key_invalid(self):
        """Test that None key is invalid."""
        assert validate_api_key_format(None) is False

    def test_wrong_prefix_invalid(self):
        """Test that wrong prefix is invalid."""
        assert validate_api_key_format("wrong_prefix_abc123") is False

    def test_too_short_invalid(self):
        """Test that too short key is invalid."""
        assert validate_api_key_format(f"{API_KEY_PREFIX}abc") is False

    def test_invalid_characters(self):
        """Test that invalid characters make key invalid."""
        invalid_key = f"{API_KEY_PREFIX}abc!@#$%^&*()" + "a" * 40
        assert validate_api_key_format(invalid_key) is False


# =============================================================================
# Key Prefix Extraction Tests
# =============================================================================


class TestExtractKeyPrefix:
    """Tests for extracting key prefix."""

    def test_extract_prefix_from_valid_key(self):
        """Test extracting prefix from valid key."""
        full_key, expected_prefix = generate_api_key()
        extracted = extract_key_prefix(full_key)
        assert extracted == expected_prefix

    def test_extract_prefix_from_invalid_key(self):
        """Test that invalid key returns None."""
        assert extract_key_prefix("invalid_key") is None

    def test_extract_prefix_returns_8_chars(self):
        """Test that extracted prefix is 8 characters."""
        full_key, _ = generate_api_key()
        prefix = extract_key_prefix(full_key)
        assert len(prefix) == 8


# =============================================================================
# API Key Service Tests
# =============================================================================


class TestAPIKeyService:
    """Tests for the API Key Service."""

    @pytest.fixture
    def service(self):
        """Create a test service instance."""
        return APIKeyService(use_argon2=False)  # Use PBKDF2 for tests

    @pytest.fixture
    def metadata(self):
        """Create test metadata."""
        return APIKeyMetadata(
            name="Test Key",
            organization_id="org_123",
            scopes=["agents:read", "calls:read"],
        )

    def test_create_key(self, service, metadata):
        """Test creating a new API key."""
        full_key, key_data = service.create_key(metadata)

        assert full_key.startswith(API_KEY_PREFIX)
        assert key_data.key_prefix is not None
        assert key_data.key_hash is not None
        assert key_data.metadata.name == "Test Key"

    def test_verify_key_success(self, service, metadata):
        """Test verifying a valid key."""
        full_key, key_data = service.create_key(metadata)
        assert service.verify_key(full_key, key_data) is True

    def test_verify_key_wrong_key(self, service, metadata):
        """Test that wrong key fails verification."""
        _, key_data = service.create_key(metadata)
        wrong_key, _ = generate_api_key()
        assert service.verify_key(wrong_key, key_data) is False

    def test_verify_key_inactive(self, service, metadata):
        """Test that inactive key fails verification."""
        full_key, key_data = service.create_key(metadata)
        key_data.metadata.is_active = False
        assert service.verify_key(full_key, key_data) is False

    def test_verify_key_expired(self, service, metadata):
        """Test that expired key fails verification."""
        metadata.expires_at = datetime.utcnow() - timedelta(hours=1)
        full_key, key_data = service.create_key(metadata)
        assert service.verify_key(full_key, key_data) is False

    def test_create_key_with_expiration(self, service, metadata):
        """Test creating a key with expiration."""
        full_key, key_data = service.create_key(metadata, expires_in_days=30)
        assert key_data.metadata.expires_at is not None
        assert key_data.metadata.expires_at > datetime.utcnow()

    def test_has_scope_exact_match(self, service, metadata):
        """Test scope checking with exact match."""
        _, key_data = service.create_key(metadata)
        assert service.has_scope(key_data, "agents:read") is True
        assert service.has_scope(key_data, "agents:write") is False

    def test_has_scope_wildcard(self, service):
        """Test scope checking with wildcard."""
        metadata = APIKeyMetadata(
            name="Wildcard Key",
            organization_id="org_123",
            scopes=["agents:*"],
        )
        _, key_data = service.create_key(metadata)
        assert service.has_scope(key_data, "agents:read") is True
        assert service.has_scope(key_data, "agents:write") is True
        assert service.has_scope(key_data, "calls:read") is False

    def test_has_scope_global_wildcard(self, service):
        """Test scope checking with global wildcard."""
        metadata = APIKeyMetadata(
            name="Admin Key",
            organization_id="org_123",
            scopes=["*"],
        )
        _, key_data = service.create_key(metadata)
        assert service.has_scope(key_data, "anything:here") is True

    def test_has_scope_empty_scopes(self, service):
        """Test that empty scopes means full access."""
        metadata = APIKeyMetadata(
            name="Full Access Key",
            organization_id="org_123",
            scopes=[],
        )
        _, key_data = service.create_key(metadata)
        assert service.has_scope(key_data, "anything:here") is True


# =============================================================================
# Security Tests
# =============================================================================


class TestSecurityProperties:
    """Tests for security properties of the API key system."""

    def test_timing_attack_resistance(self):
        """Test that verification uses constant-time comparison."""
        full_key, _ = generate_api_key()
        key_hash, salt = hash_api_key_pbkdf2(full_key)

        # Generate keys with varying similarity to the original
        import time

        times = []
        for i in range(10):
            wrong_key = f"{API_KEY_PREFIX}{'x' * (40 + i)}"
            start = time.perf_counter_ns()
            verify_api_key_pbkdf2(wrong_key, key_hash, salt)
            elapsed = time.perf_counter_ns() - start
            times.append(elapsed)

        # Timing variance should be relatively small
        # (This is a basic check; real timing attacks need more sophisticated testing)
        variance = max(times) - min(times)
        # Allow for some variance due to system noise
        # The main point is that the time shouldn't scale with string similarity
        assert variance < 10_000_000  # 10ms variance is acceptable

    def test_key_entropy(self):
        """Test that generated keys have sufficient entropy."""
        import string

        keys = [generate_api_key()[0] for _ in range(100)]

        # Check that keys use a good character set
        valid_chars = set(string.ascii_letters + string.digits + "_-")
        for key in keys:
            key_body = key[len(API_KEY_PREFIX):]
            assert all(c in valid_chars for c in key_body)

        # Check that keys don't have obvious patterns
        key_bodies = [k[len(API_KEY_PREFIX):] for k in keys]

        # No two keys should share more than a few characters at the start
        for i, k1 in enumerate(key_bodies):
            for k2 in key_bodies[i + 1:]:
                common_prefix = 0
                for c1, c2 in zip(k1, k2):
                    if c1 == c2:
                        common_prefix += 1
                    else:
                        break
                # Allow up to 3 common prefix chars by chance
                assert common_prefix <= 3
