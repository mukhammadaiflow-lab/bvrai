"""
Unit Tests for Encryption Service

Tests for secure encryption, key derivation, hashing, and the
removal of insecure XOR fallback (critical security fix).
"""

import pytest
import base64
from datetime import datetime

from platform.security.encryption import (
    EncryptionAlgorithm,
    KeyType,
    HashAlgorithm,
    EncryptedData,
    KeyMetadata,
    DataEncryptor,
    FieldEncryptor,
    EncryptionService,
    HAS_CRYPTOGRAPHY,
)


# =============================================================================
# EncryptedData Tests
# =============================================================================


class TestEncryptedData:
    """Tests for EncryptedData serialization."""

    def test_to_bytes_and_from_bytes_roundtrip(self):
        """Test serialization roundtrip."""
        original = EncryptedData(
            ciphertext=b"test_ciphertext",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_id="test_key_123",
            iv=b"test_iv_bytes",
            tag=b"test_tag_bytes",
            version=1,
        )

        serialized = original.to_bytes()
        restored = EncryptedData.from_bytes(serialized)

        assert restored.ciphertext == original.ciphertext
        assert restored.algorithm == original.algorithm
        assert restored.key_id == original.key_id
        assert restored.iv == original.iv
        assert restored.tag == original.tag
        assert restored.version == original.version

    def test_to_base64_and_from_base64_roundtrip(self):
        """Test base64 serialization roundtrip."""
        original = EncryptedData(
            ciphertext=b"secret_data",
            algorithm=EncryptionAlgorithm.FERNET,
            key_id="key_456",
        )

        base64_str = original.to_base64()
        restored = EncryptedData.from_base64(base64_str)

        assert restored.ciphertext == original.ciphertext
        assert restored.algorithm == original.algorithm
        assert restored.key_id == original.key_id

    def test_encrypted_data_without_optional_fields(self):
        """Test EncryptedData without IV/tag."""
        data = EncryptedData(
            ciphertext=b"encrypted",
            algorithm=EncryptionAlgorithm.FERNET,
            key_id="key_789",
        )

        serialized = data.to_bytes()
        restored = EncryptedData.from_bytes(serialized)

        assert restored.iv is None
        assert restored.tag is None


# =============================================================================
# KeyMetadata Tests
# =============================================================================


class TestKeyMetadata:
    """Tests for key metadata."""

    def test_is_expired_with_no_expiry(self):
        """Test that key with no expiry is never expired."""
        metadata = KeyMetadata(
            key_id="key_1",
            key_type=KeyType.MASTER,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            expires_at=None,
        )
        assert metadata.is_expired is False

    def test_is_expired_with_future_expiry(self):
        """Test that key with future expiry is not expired."""
        from datetime import timedelta
        metadata = KeyMetadata(
            key_id="key_2",
            key_type=KeyType.DATA,
            algorithm=EncryptionAlgorithm.FERNET,
            expires_at=datetime.utcnow() + timedelta(days=1),
        )
        assert metadata.is_expired is False

    def test_is_expired_with_past_expiry(self):
        """Test that key with past expiry is expired."""
        from datetime import timedelta
        metadata = KeyMetadata(
            key_id="key_3",
            key_type=KeyType.SESSION,
            algorithm=EncryptionAlgorithm.AES_256_CBC,
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        assert metadata.is_expired is True


# =============================================================================
# DataEncryptor Tests
# =============================================================================


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography package required")
class TestDataEncryptor:
    """Tests for DataEncryptor with cryptography package."""

    @pytest.fixture
    def encryptor(self):
        """Create a test encryptor."""
        key = b"a" * 32  # 256-bit key
        return DataEncryptor(master_key=key)

    def test_encrypt_decrypt_roundtrip(self, encryptor):
        """Test basic encrypt/decrypt roundtrip."""
        plaintext = b"Hello, World!"
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == plaintext

    def test_encrypt_string(self, encryptor):
        """Test encrypting string input."""
        plaintext = "String message"
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == plaintext.encode()

    def test_encrypt_with_aad(self, encryptor):
        """Test encryption with additional authenticated data."""
        plaintext = b"Sensitive data"
        aad = b"context_info"
        encrypted = encryptor.encrypt(plaintext, aad=aad)
        decrypted = encryptor.decrypt(encrypted, aad=aad)
        assert decrypted == plaintext

    def test_decrypt_with_wrong_aad_fails(self, encryptor):
        """Test that decryption with wrong AAD fails."""
        plaintext = b"Sensitive data"
        aad = b"context_info"
        encrypted = encryptor.encrypt(plaintext, aad=aad)

        with pytest.raises(Exception):  # InvalidTag from cryptography
            encryptor.decrypt(encrypted, aad=b"wrong_context")

    def test_different_keys_produce_different_ciphertext(self):
        """Test that same plaintext with different keys produces different ciphertext."""
        plaintext = b"Same message"
        enc1 = DataEncryptor(master_key=b"key1" + b"x" * 28)
        enc2 = DataEncryptor(master_key=b"key2" + b"x" * 28)

        result1 = enc1.encrypt(plaintext)
        result2 = enc2.encrypt(plaintext)

        assert result1.ciphertext != result2.ciphertext

    def test_encrypt_aes_cbc(self):
        """Test AES-CBC encryption."""
        key = b"b" * 32
        encryptor = DataEncryptor(master_key=key, algorithm=EncryptionAlgorithm.AES_256_CBC)

        plaintext = b"Test AES-CBC"
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == plaintext
        assert encrypted.algorithm == EncryptionAlgorithm.AES_256_CBC

    def test_encrypt_fernet(self):
        """Test Fernet encryption."""
        key = b"c" * 32
        encryptor = DataEncryptor(master_key=key, algorithm=EncryptionAlgorithm.FERNET)

        plaintext = b"Test Fernet"
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == plaintext
        assert encrypted.algorithm == EncryptionAlgorithm.FERNET


# =============================================================================
# Security Critical: XOR Fallback Removal Tests
# =============================================================================


class TestXORFallbackRemoval:
    """
    SECURITY CRITICAL: Tests to verify XOR encryption fallback is removed.

    The XOR fallback was a security vulnerability that provided no real
    encryption. These tests ensure the fallback now raises RuntimeError.
    """

    def test_encrypt_fallback_raises_runtime_error(self):
        """Test that encryption fallback raises RuntimeError."""
        key = b"d" * 32
        encryptor = DataEncryptor(master_key=key)

        # Simulate missing cryptography by calling fallback directly
        with pytest.raises(RuntimeError) as exc_info:
            encryptor._encrypt_fernet_fallback(b"test data")

        assert "cryptography package is required" in str(exc_info.value)

    def test_decrypt_fallback_raises_runtime_error(self):
        """Test that decryption fallback raises RuntimeError."""
        key = b"e" * 32
        encryptor = DataEncryptor(master_key=key)

        # Create dummy encrypted data
        encrypted = EncryptedData(
            ciphertext=b"fake",
            algorithm=EncryptionAlgorithm.FERNET,
            key_id="test",
        )

        with pytest.raises(RuntimeError) as exc_info:
            encryptor._decrypt_fernet_fallback(encrypted)

        assert "cryptography package is required" in str(exc_info.value)

    def test_fallback_does_not_use_xor(self):
        """Test that fallback doesn't perform XOR (weak) encryption."""
        key = b"f" * 32
        encryptor = DataEncryptor(master_key=key)

        # If XOR was being used, calling the fallback would return encrypted data
        # instead of raising an error
        try:
            result = encryptor._encrypt_fernet_fallback(b"test")
            # If we get here without exception, XOR is still being used - FAIL
            assert False, "XOR fallback should have been removed!"
        except RuntimeError:
            # Expected behavior - fallback raises error
            pass


# =============================================================================
# FieldEncryptor Tests
# =============================================================================


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography package required")
class TestFieldEncryptor:
    """Tests for field-level encryption."""

    @pytest.fixture
    def field_encryptor(self):
        """Create a test field encryptor."""
        key = b"g" * 32
        encryptor = DataEncryptor(master_key=key)
        return FieldEncryptor(encryptor)

    def test_encrypt_sensitive_field(self, field_encryptor):
        """Test that sensitive fields are encrypted."""
        data = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30,
        }

        encrypted = field_encryptor.encrypt_fields(data)

        # Non-sensitive fields unchanged
        assert encrypted["name"] == "John Doe"
        assert encrypted["age"] == 30

        # Sensitive field is encrypted
        assert encrypted["email"]["__encrypted__"] is True
        assert "data" in encrypted["email"]

    def test_encrypt_decrypt_roundtrip(self, field_encryptor):
        """Test field encryption/decryption roundtrip."""
        original = {
            "user_id": "123",
            "phone": "+1234567890",
            "credit_card": "4111-1111-1111-1111",
        }

        encrypted = field_encryptor.encrypt_fields(original)
        decrypted = field_encryptor.decrypt_fields(encrypted)

        assert decrypted == original

    def test_mask_fields(self, field_encryptor):
        """Test field masking."""
        data = {
            "name": "John",
            "ssn": "123-45-6789",
            "email": "john@example.com",
        }

        masked = field_encryptor.mask_fields(data)

        assert masked["name"] == "John"  # Not sensitive
        assert "*" in masked["ssn"]  # Masked
        assert "*" in masked["email"]  # Masked
        assert masked["ssn"].endswith("6789")  # Last 4 visible

    def test_add_remove_sensitive_field(self, field_encryptor):
        """Test adding/removing sensitive fields."""
        # Add custom field
        field_encryptor.add_sensitive_field("custom_secret")

        data = {"custom_secret": "secret_value", "normal": "public"}
        encrypted = field_encryptor.encrypt_fields(data)

        assert encrypted["custom_secret"]["__encrypted__"] is True
        assert encrypted["normal"] == "public"

        # Remove field
        field_encryptor.remove_sensitive_field("custom_secret")
        encrypted2 = field_encryptor.encrypt_fields(data)

        assert encrypted2["custom_secret"] == "secret_value"


# =============================================================================
# EncryptionService Tests
# =============================================================================


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography package required")
class TestEncryptionService:
    """Tests for the unified encryption service."""

    @pytest.fixture
    def service(self):
        """Create a test encryption service."""
        key = b"h" * 32
        return EncryptionService(master_key=key)

    def test_encrypt_string(self, service):
        """Test string encryption."""
        plaintext = "Secret message"
        encrypted = service.encrypt_string(plaintext)
        decrypted = service.decrypt_string(encrypted)
        assert decrypted == plaintext

    def test_hash_password(self, service):
        """Test password hashing."""
        password = "my_secure_password"
        hashed = service.hash_password(password)

        assert hashed.startswith("$")  # PHC format
        assert password not in hashed

    def test_verify_password_correct(self, service):
        """Test password verification with correct password."""
        password = "test_password_123"
        hashed = service.hash_password(password)
        assert service.verify_password(password, hashed) is True

    def test_verify_password_incorrect(self, service):
        """Test password verification with incorrect password."""
        password = "correct_password"
        hashed = service.hash_password(password)
        assert service.verify_password("wrong_password", hashed) is False

    def test_hash_data_sha256(self, service):
        """Test SHA-256 hashing."""
        data = "test data"
        hash1 = service.hash_data(data, HashAlgorithm.SHA256)
        hash2 = service.hash_data(data, HashAlgorithm.SHA256)

        assert hash1 == hash2  # Deterministic
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars

    def test_hash_data_different_algorithms(self, service):
        """Test different hash algorithms produce different results."""
        data = "test data"
        sha256 = service.hash_data(data, HashAlgorithm.SHA256)
        sha512 = service.hash_data(data, HashAlgorithm.SHA512)

        assert sha256 != sha512
        assert len(sha512) > len(sha256)

    def test_generate_and_verify_hmac(self, service):
        """Test HMAC generation and verification."""
        data = "important data"
        signature = service.generate_hmac(data)

        assert service.verify_hmac(data, signature) is True
        assert service.verify_hmac("tampered data", signature) is False

    def test_derive_key(self, service):
        """Test key derivation from password."""
        password = "user_password"
        key1, salt1 = service.derive_key(password)
        key2, salt2 = service.derive_key(password)

        # Different salts should produce different keys
        assert salt1 != salt2
        assert key1 != key2

        # Same salt should produce same key
        key3, _ = service.derive_key(password, salt=salt1)
        assert key1 == key3

    def test_generate_api_key(self, service):
        """Test API key generation."""
        key = service.generate_api_key(prefix="test")

        assert key.startswith("test_")
        assert len(key) > 10

    def test_generate_token(self, service):
        """Test token generation."""
        token = service.generate_token(length=32)

        assert len(token) > 0
        # URL-safe means no + or /
        assert "+" not in token
        assert "/" not in token

    def test_constant_time_compare(self, service):
        """Test constant-time comparison."""
        a = "test_string"
        b = "test_string"
        c = "different"

        assert service.constant_time_compare(a, b) is True
        assert service.constant_time_compare(a, c) is False

        # Test with bytes
        assert service.constant_time_compare(a.encode(), b.encode()) is True


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.skipif(not HAS_CRYPTOGRAPHY, reason="cryptography package required")
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_encrypt_empty_data(self):
        """Test encrypting empty data."""
        encryptor = DataEncryptor(master_key=b"i" * 32)
        encrypted = encryptor.encrypt(b"")
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == b""

    def test_encrypt_large_data(self):
        """Test encrypting large data."""
        encryptor = DataEncryptor(master_key=b"j" * 32)
        large_data = b"x" * (1024 * 1024)  # 1 MB
        encrypted = encryptor.encrypt(large_data)
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == large_data

    def test_unicode_data(self):
        """Test encrypting unicode strings."""
        service = EncryptionService(master_key=b"k" * 32)
        plaintext = "Hello, World!"
        encrypted = service.encrypt_string(plaintext)
        decrypted = service.decrypt_string(encrypted)
        assert decrypted == plaintext

    def test_invalid_password_hash_format(self):
        """Test verifying password with invalid hash format."""
        service = EncryptionService(master_key=b"l" * 32)
        assert service.verify_password("password", "invalid_hash") is False
        assert service.verify_password("password", "") is False
        assert service.verify_password("password", "$invalid$") is False
