"""
Secure API Key Management

This module provides secure API key generation, hashing, and validation
using industry-standard cryptographic practices.
"""

import base64
import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple

from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================


API_KEY_PREFIX = "bvk_"  # Builder Voice Key
API_KEY_LENGTH = 32  # Bytes (256 bits)
HASH_ITERATIONS = 100000  # PBKDF2 iterations
SALT_LENGTH = 16  # Bytes


# =============================================================================
# Models
# =============================================================================


class APIKeyMetadata(BaseModel):
    """Metadata associated with an API key."""

    name: str
    organization_id: str
    created_by_user_id: Optional[str] = None
    scopes: list[str] = Field(default_factory=list)
    expires_at: Optional[datetime] = None
    rate_limit_per_minute: int = 60
    rate_limit_per_day: int = 10000
    ip_whitelist: list[str] = Field(default_factory=list)
    is_active: bool = True


class APIKeyData(BaseModel):
    """Full API key data for storage."""

    key_prefix: str  # First 8 characters for identification
    key_hash: str  # Argon2 or PBKDF2 hash of full key
    salt: str  # Salt used for hashing
    metadata: APIKeyMetadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    request_count: int = 0


# =============================================================================
# Key Generation
# =============================================================================


def generate_api_key() -> Tuple[str, str]:
    """
    Generate a new secure API key.

    Returns:
        Tuple of (full_key, key_prefix)

    The full key should be shown to the user once and never stored.
    Only the hash is stored in the database.
    """
    # Generate random bytes
    random_bytes = secrets.token_bytes(API_KEY_LENGTH)

    # Encode as URL-safe base64
    key_body = base64.urlsafe_b64encode(random_bytes).decode("utf-8").rstrip("=")

    # Add prefix
    full_key = f"{API_KEY_PREFIX}{key_body}"

    # Get prefix for identification (first 8 chars after prefix)
    key_prefix = key_body[:8]

    return full_key, key_prefix


def generate_salt() -> bytes:
    """Generate a cryptographic salt."""
    return secrets.token_bytes(SALT_LENGTH)


# =============================================================================
# Key Hashing (PBKDF2)
# =============================================================================


def hash_api_key_pbkdf2(api_key: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """
    Hash an API key using PBKDF2-HMAC-SHA256.

    Args:
        api_key: The API key to hash
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (hash_hex, salt_hex)
    """
    if salt is None:
        salt = generate_salt()

    # Remove prefix for hashing
    key_body = api_key.replace(API_KEY_PREFIX, "")

    # Hash using PBKDF2
    key_hash = hashlib.pbkdf2_hmac(
        "sha256",
        key_body.encode("utf-8"),
        salt,
        HASH_ITERATIONS,
    )

    return key_hash.hex(), salt.hex()


def verify_api_key_pbkdf2(api_key: str, stored_hash: str, stored_salt: str) -> bool:
    """
    Verify an API key against a stored hash.

    Args:
        api_key: The API key to verify
        stored_hash: The stored hash (hex)
        stored_salt: The stored salt (hex)

    Returns:
        True if the key is valid
    """
    try:
        # Hash the provided key with the stored salt
        salt = bytes.fromhex(stored_salt)
        computed_hash, _ = hash_api_key_pbkdf2(api_key, salt)

        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(computed_hash, stored_hash)
    except Exception:
        return False


# =============================================================================
# Key Hashing (Argon2) - Preferred if argon2-cffi is available
# =============================================================================


def hash_api_key_argon2(api_key: str) -> str:
    """
    Hash an API key using Argon2id.

    Args:
        api_key: The API key to hash

    Returns:
        The Argon2 hash string (includes salt)
    """
    try:
        from argon2 import PasswordHasher, Type

        hasher = PasswordHasher(
            time_cost=2,
            memory_cost=65536,  # 64 MB
            parallelism=4,
            hash_len=32,
            type=Type.ID,
        )

        # Remove prefix for hashing
        key_body = api_key.replace(API_KEY_PREFIX, "")

        return hasher.hash(key_body)
    except ImportError:
        raise RuntimeError("argon2-cffi package required for Argon2 hashing")


def verify_api_key_argon2(api_key: str, stored_hash: str) -> bool:
    """
    Verify an API key using Argon2.

    Args:
        api_key: The API key to verify
        stored_hash: The stored Argon2 hash

    Returns:
        True if the key is valid
    """
    try:
        from argon2 import PasswordHasher
        from argon2.exceptions import VerifyMismatchError

        hasher = PasswordHasher()
        key_body = api_key.replace(API_KEY_PREFIX, "")

        try:
            hasher.verify(stored_hash, key_body)
            return True
        except VerifyMismatchError:
            return False
    except ImportError:
        raise RuntimeError("argon2-cffi package required for Argon2 verification")


# =============================================================================
# Key Validation
# =============================================================================


def validate_api_key_format(api_key: str) -> bool:
    """
    Validate API key format without checking the database.

    Args:
        api_key: The API key to validate

    Returns:
        True if the format is valid
    """
    if not api_key:
        return False

    # Check prefix
    if not api_key.startswith(API_KEY_PREFIX):
        return False

    # Check length (prefix + base64 encoded 32 bytes ≈ 43 chars)
    if len(api_key) < len(API_KEY_PREFIX) + 40:
        return False

    # Check character set (URL-safe base64)
    key_body = api_key[len(API_KEY_PREFIX):]
    pattern = re.compile(r"^[A-Za-z0-9_-]+$")
    if not pattern.match(key_body):
        return False

    return True


def extract_key_prefix(api_key: str) -> Optional[str]:
    """
    Extract the prefix from an API key for database lookup.

    Args:
        api_key: The full API key

    Returns:
        The 8-character prefix or None if invalid
    """
    if not validate_api_key_format(api_key):
        return None

    key_body = api_key[len(API_KEY_PREFIX):]
    return key_body[:8]


# =============================================================================
# API Key Service
# =============================================================================


class APIKeyService:
    """
    Service for managing API keys.

    Usage:
        service = APIKeyService(use_argon2=True)
        key, data = service.create_key(metadata)
        is_valid = service.verify_key(key, data)
    """

    def __init__(self, use_argon2: bool = True):
        """
        Initialize the service.

        Args:
            use_argon2: Use Argon2 for hashing (falls back to PBKDF2 if not available)
        """
        self.use_argon2 = use_argon2

        # Check if Argon2 is available
        if use_argon2:
            try:
                import argon2
                self._hash_func = hash_api_key_argon2
                self._verify_func = verify_api_key_argon2
            except ImportError:
                self.use_argon2 = False
                self._hash_func = None
                self._verify_func = None

        if not self.use_argon2:
            self._hash_func = None
            self._verify_func = None

    def create_key(
        self,
        metadata: APIKeyMetadata,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, APIKeyData]:
        """
        Create a new API key.

        Args:
            metadata: Key metadata
            expires_in_days: Optional expiration in days

        Returns:
            Tuple of (full_key, key_data)
        """
        # Generate the key
        full_key, key_prefix = generate_api_key()

        # Set expiration if specified
        if expires_in_days:
            metadata.expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Hash the key
        if self.use_argon2:
            key_hash = hash_api_key_argon2(full_key)
            salt = ""  # Argon2 includes salt in hash
        else:
            key_hash, salt = hash_api_key_pbkdf2(full_key)

        # Create key data
        key_data = APIKeyData(
            key_prefix=key_prefix,
            key_hash=key_hash,
            salt=salt,
            metadata=metadata,
        )

        return full_key, key_data

    def verify_key(self, api_key: str, key_data: APIKeyData) -> bool:
        """
        Verify an API key against stored data.

        Args:
            api_key: The API key to verify
            key_data: The stored key data

        Returns:
            True if valid and not expired
        """
        # Check format
        if not validate_api_key_format(api_key):
            return False

        # Check if key is active
        if not key_data.metadata.is_active:
            return False

        # Check expiration
        if key_data.metadata.expires_at:
            if datetime.utcnow() > key_data.metadata.expires_at:
                return False

        # Verify hash
        if self.use_argon2:
            return verify_api_key_argon2(api_key, key_data.key_hash)
        else:
            return verify_api_key_pbkdf2(api_key, key_data.key_hash, key_data.salt)

    def has_scope(self, key_data: APIKeyData, required_scope: str) -> bool:
        """
        Check if a key has a required scope.

        Args:
            key_data: The key data
            required_scope: The scope to check

        Returns:
            True if the key has the scope
        """
        scopes = key_data.metadata.scopes

        # Empty scopes means all access
        if not scopes:
            return True

        # Check for exact match or wildcard
        if required_scope in scopes:
            return True

        # Check for wildcard patterns (e.g., "agents:*")
        scope_parts = required_scope.split(":")
        if len(scope_parts) == 2:
            wildcard = f"{scope_parts[0]}:*"
            if wildcard in scopes:
                return True

        # Check for global wildcard
        if "*" in scopes:
            return True

        return False


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "API_KEY_PREFIX",
    "APIKeyMetadata",
    "APIKeyData",
    "generate_api_key",
    "hash_api_key_pbkdf2",
    "verify_api_key_pbkdf2",
    "hash_api_key_argon2",
    "verify_api_key_argon2",
    "validate_api_key_format",
    "extract_key_prefix",
    "APIKeyService",
]
