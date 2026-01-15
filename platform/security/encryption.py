"""
Encryption Service
==================

Comprehensive encryption and cryptographic operations including
symmetric/asymmetric encryption, field-level encryption, and secure hashing.

Author: Platform Security Team
Version: 2.0.0
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import structlog

logger = structlog.get_logger(__name__)

# Cryptography imports with fallback
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
    from cryptography.hazmat.primitives import serialization
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    logger.warning("cryptography package not installed, using basic encryption")


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms."""

    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    FERNET = "fernet"
    RSA_OAEP = "rsa-oaep"


class KeyType(str, Enum):
    """Types of encryption keys."""

    MASTER = "master"
    DATA = "data"
    SESSION = "session"
    API = "api"
    SIGNING = "signing"


class HashAlgorithm(str, Enum):
    """Supported hash algorithms."""

    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    ARGON2 = "argon2"
    SCRYPT = "scrypt"


@dataclass
class EncryptedData:
    """Container for encrypted data."""

    ciphertext: bytes
    algorithm: EncryptionAlgorithm
    key_id: str
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    aad: Optional[bytes] = None  # Additional authenticated data
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serialize to bytes for storage."""
        import json
        header = {
            "algorithm": self.algorithm.value,
            "key_id": self.key_id,
            "iv": base64.b64encode(self.iv).decode() if self.iv else None,
            "tag": base64.b64encode(self.tag).decode() if self.tag else None,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
        }
        header_bytes = json.dumps(header).encode()
        header_len = len(header_bytes).to_bytes(4, "big")
        return header_len + header_bytes + self.ciphertext

    @classmethod
    def from_bytes(cls, data: bytes) -> "EncryptedData":
        """Deserialize from bytes."""
        import json
        header_len = int.from_bytes(data[:4], "big")
        header_bytes = data[4:4 + header_len]
        ciphertext = data[4 + header_len:]
        header = json.loads(header_bytes)
        return cls(
            ciphertext=ciphertext,
            algorithm=EncryptionAlgorithm(header["algorithm"]),
            key_id=header["key_id"],
            iv=base64.b64decode(header["iv"]) if header.get("iv") else None,
            tag=base64.b64decode(header["tag"]) if header.get("tag") else None,
            version=header.get("version", 1),
            created_at=datetime.fromisoformat(header["created_at"]) if header.get("created_at") else datetime.utcnow(),
        )

    def to_base64(self) -> str:
        """Encode as base64 string."""
        return base64.b64encode(self.to_bytes()).decode()

    @classmethod
    def from_base64(cls, data: str) -> "EncryptedData":
        """Decode from base64 string."""
        return cls.from_bytes(base64.b64decode(data))


@dataclass
class KeyMetadata:
    """Metadata about an encryption key."""

    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    rotated_at: Optional[datetime] = None
    rotation_count: int = 0
    enabled: bool = True
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at


class DataEncryptor:
    """
    High-level data encryption service.

    Provides symmetric encryption with automatic key management,
    authenticated encryption modes, and secure key derivation.
    """

    def __init__(
        self,
        master_key: Optional[bytes] = None,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
    ):
        self._master_key = master_key or self._generate_key()
        self._algorithm = algorithm
        self._key_id = self._derive_key_id(self._master_key)
        self._logger = structlog.get_logger("data_encryptor")

    @staticmethod
    def _generate_key() -> bytes:
        """Generate a new random key."""
        return secrets.token_bytes(32)  # 256 bits

    @staticmethod
    def _derive_key_id(key: bytes) -> str:
        """Derive key ID from key material."""
        return hashlib.sha256(key).hexdigest()[:16]

    def encrypt(
        self,
        plaintext: Union[bytes, str],
        aad: Optional[bytes] = None,
    ) -> EncryptedData:
        """
        Encrypt data.

        Args:
            plaintext: Data to encrypt
            aad: Additional authenticated data (for GCM/Poly1305 modes)

        Returns:
            EncryptedData containing ciphertext and metadata
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode()

        if self._algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._encrypt_aes_gcm(plaintext, aad)
        elif self._algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._encrypt_aes_cbc(plaintext)
        elif self._algorithm == EncryptionAlgorithm.FERNET:
            return self._encrypt_fernet(plaintext)
        else:
            raise ValueError(f"Unsupported algorithm: {self._algorithm}")

    def decrypt(
        self,
        encrypted: EncryptedData,
        aad: Optional[bytes] = None,
    ) -> bytes:
        """
        Decrypt data.

        Args:
            encrypted: EncryptedData to decrypt
            aad: Additional authenticated data (must match what was used for encryption)

        Returns:
            Decrypted plaintext bytes
        """
        if encrypted.algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._decrypt_aes_gcm(encrypted, aad)
        elif encrypted.algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._decrypt_aes_cbc(encrypted)
        elif encrypted.algorithm == EncryptionAlgorithm.FERNET:
            return self._decrypt_fernet(encrypted)
        else:
            raise ValueError(f"Unsupported algorithm: {encrypted.algorithm}")

    def _encrypt_aes_gcm(
        self,
        plaintext: bytes,
        aad: Optional[bytes] = None,
    ) -> EncryptedData:
        """Encrypt using AES-256-GCM."""
        if not HAS_CRYPTOGRAPHY:
            return self._encrypt_fernet_fallback(plaintext)

        iv = os.urandom(12)  # 96-bit IV for GCM
        cipher = Cipher(
            algorithms.AES(self._master_key),
            modes.GCM(iv),
            backend=default_backend(),
        )
        encryptor = cipher.encryptor()

        if aad:
            encryptor.authenticate_additional_data(aad)

        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_id=self._key_id,
            iv=iv,
            tag=encryptor.tag,
            aad=aad,
        )

    def _decrypt_aes_gcm(
        self,
        encrypted: EncryptedData,
        aad: Optional[bytes] = None,
    ) -> bytes:
        """Decrypt using AES-256-GCM."""
        if not HAS_CRYPTOGRAPHY:
            return self._decrypt_fernet_fallback(encrypted)

        cipher = Cipher(
            algorithms.AES(self._master_key),
            modes.GCM(encrypted.iv, encrypted.tag),
            backend=default_backend(),
        )
        decryptor = cipher.decryptor()

        if aad or encrypted.aad:
            decryptor.authenticate_additional_data(aad or encrypted.aad)

        return decryptor.update(encrypted.ciphertext) + decryptor.finalize()

    def _encrypt_aes_cbc(self, plaintext: bytes) -> EncryptedData:
        """Encrypt using AES-256-CBC."""
        if not HAS_CRYPTOGRAPHY:
            return self._encrypt_fernet_fallback(plaintext)

        iv = os.urandom(16)
        padder = padding.PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()

        cipher = Cipher(
            algorithms.AES(self._master_key),
            modes.CBC(iv),
            backend=default_backend(),
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded) + encryptor.finalize()

        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=EncryptionAlgorithm.AES_256_CBC,
            key_id=self._key_id,
            iv=iv,
        )

    def _decrypt_aes_cbc(self, encrypted: EncryptedData) -> bytes:
        """Decrypt using AES-256-CBC."""
        if not HAS_CRYPTOGRAPHY:
            return self._decrypt_fernet_fallback(encrypted)

        cipher = Cipher(
            algorithms.AES(self._master_key),
            modes.CBC(encrypted.iv),
            backend=default_backend(),
        )
        decryptor = cipher.decryptor()
        padded = decryptor.update(encrypted.ciphertext) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded) + unpadder.finalize()

    def _encrypt_fernet(self, plaintext: bytes) -> EncryptedData:
        """Encrypt using Fernet."""
        if not HAS_CRYPTOGRAPHY:
            return self._encrypt_fernet_fallback(plaintext)

        fernet_key = base64.urlsafe_b64encode(self._master_key)
        f = Fernet(fernet_key)
        ciphertext = f.encrypt(plaintext)

        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=EncryptionAlgorithm.FERNET,
            key_id=self._key_id,
        )

    def _decrypt_fernet(self, encrypted: EncryptedData) -> bytes:
        """Decrypt using Fernet."""
        if not HAS_CRYPTOGRAPHY:
            return self._decrypt_fernet_fallback(encrypted)

        fernet_key = base64.urlsafe_b64encode(self._master_key)
        f = Fernet(fernet_key)
        return f.decrypt(encrypted.ciphertext)

    def _encrypt_fernet_fallback(self, plaintext: bytes) -> EncryptedData:
        """Fallback encryption when cryptography is not available."""
        # Simple XOR-based encryption (NOT secure, just for demo)
        key_stream = self._master_key * (len(plaintext) // 32 + 1)
        ciphertext = bytes(p ^ k for p, k in zip(plaintext, key_stream))
        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=EncryptionAlgorithm.FERNET,
            key_id=self._key_id,
        )

    def _decrypt_fernet_fallback(self, encrypted: EncryptedData) -> bytes:
        """Fallback decryption when cryptography is not available."""
        key_stream = self._master_key * (len(encrypted.ciphertext) // 32 + 1)
        return bytes(c ^ k for c, k in zip(encrypted.ciphertext, key_stream))


class FieldEncryptor:
    """
    Field-level encryption for sensitive data fields.

    Encrypts specific fields in dictionaries/objects while leaving
    others unencrypted for queryability.
    """

    def __init__(
        self,
        encryptor: DataEncryptor,
        sensitive_fields: Optional[List[str]] = None,
    ):
        self._encryptor = encryptor
        self._sensitive_fields = set(sensitive_fields or [
            "ssn", "social_security_number",
            "credit_card", "card_number",
            "password", "secret", "api_key",
            "phone", "phone_number",
            "email", "address",
            "date_of_birth", "dob",
        ])
        self._logger = structlog.get_logger("field_encryptor")

    def add_sensitive_field(self, field_name: str) -> None:
        """Add a field to be encrypted."""
        self._sensitive_fields.add(field_name.lower())

    def remove_sensitive_field(self, field_name: str) -> None:
        """Remove a field from encryption."""
        self._sensitive_fields.discard(field_name.lower())

    def encrypt_fields(
        self,
        data: Dict[str, Any],
        additional_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Encrypt sensitive fields in a dictionary.

        Args:
            data: Dictionary with potentially sensitive fields
            additional_fields: Extra fields to encrypt for this call

        Returns:
            Dictionary with sensitive fields encrypted
        """
        result = {}
        fields_to_encrypt = self._sensitive_fields | set(additional_fields or [])

        for key, value in data.items():
            if key.lower() in fields_to_encrypt and value is not None:
                # Encrypt the value
                if isinstance(value, (dict, list)):
                    import json
                    value_bytes = json.dumps(value).encode()
                else:
                    value_bytes = str(value).encode()

                encrypted = self._encryptor.encrypt(value_bytes)
                result[key] = {
                    "__encrypted__": True,
                    "data": encrypted.to_base64(),
                }
            elif isinstance(value, dict):
                # Recursively encrypt nested dicts
                result[key] = self.encrypt_fields(value, additional_fields)
            else:
                result[key] = value

        return result

    def decrypt_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt encrypted fields in a dictionary.

        Args:
            data: Dictionary with encrypted fields

        Returns:
            Dictionary with fields decrypted
        """
        result = {}

        for key, value in data.items():
            if isinstance(value, dict):
                if value.get("__encrypted__"):
                    # Decrypt the value
                    encrypted = EncryptedData.from_base64(value["data"])
                    decrypted_bytes = self._encryptor.decrypt(encrypted)
                    # Try to parse as JSON
                    try:
                        import json
                        result[key] = json.loads(decrypted_bytes)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        result[key] = decrypted_bytes.decode()
                else:
                    # Recursively decrypt nested dicts
                    result[key] = self.decrypt_fields(value)
            else:
                result[key] = value

        return result

    def mask_fields(
        self,
        data: Dict[str, Any],
        mask_char: str = "*",
        visible_chars: int = 4,
    ) -> Dict[str, Any]:
        """
        Mask sensitive fields for display.

        Args:
            data: Dictionary with sensitive fields
            mask_char: Character to use for masking
            visible_chars: Number of characters to leave visible

        Returns:
            Dictionary with sensitive fields masked
        """
        result = {}

        for key, value in data.items():
            if key.lower() in self._sensitive_fields and value is not None:
                value_str = str(value)
                if len(value_str) <= visible_chars:
                    result[key] = mask_char * len(value_str)
                else:
                    result[key] = mask_char * (len(value_str) - visible_chars) + value_str[-visible_chars:]
            elif isinstance(value, dict):
                result[key] = self.mask_fields(value, mask_char, visible_chars)
            else:
                result[key] = value

        return result


class EncryptionService:
    """
    Unified encryption service.

    Provides a high-level interface for all encryption operations
    including symmetric, asymmetric, hashing, and key derivation.
    """

    def __init__(
        self,
        master_key: Optional[bytes] = None,
        default_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
    ):
        self._master_key = master_key or secrets.token_bytes(32)
        self._default_algorithm = default_algorithm
        self._encryptor = DataEncryptor(self._master_key, default_algorithm)
        self._field_encryptor = FieldEncryptor(self._encryptor)
        self._key_cache: Dict[str, bytes] = {}
        self._logger = structlog.get_logger("encryption_service")

    @property
    def field_encryptor(self) -> FieldEncryptor:
        """Get field-level encryptor."""
        return self._field_encryptor

    def encrypt(
        self,
        plaintext: Union[bytes, str],
        aad: Optional[bytes] = None,
    ) -> EncryptedData:
        """Encrypt data."""
        return self._encryptor.encrypt(plaintext, aad)

    def decrypt(
        self,
        encrypted: EncryptedData,
        aad: Optional[bytes] = None,
    ) -> bytes:
        """Decrypt data."""
        return self._encryptor.decrypt(encrypted, aad)

    def encrypt_string(self, plaintext: str, aad: Optional[bytes] = None) -> str:
        """Encrypt string and return base64-encoded result."""
        encrypted = self.encrypt(plaintext.encode(), aad)
        return encrypted.to_base64()

    def decrypt_string(self, ciphertext: str, aad: Optional[bytes] = None) -> str:
        """Decrypt base64-encoded ciphertext to string."""
        encrypted = EncryptedData.from_base64(ciphertext)
        return self.decrypt(encrypted, aad).decode()

    def hash_password(
        self,
        password: str,
        algorithm: HashAlgorithm = HashAlgorithm.SCRYPT,
    ) -> str:
        """
        Hash a password securely.

        Args:
            password: Password to hash
            algorithm: Hash algorithm to use

        Returns:
            Hashed password in PHC format
        """
        if algorithm == HashAlgorithm.SCRYPT and HAS_CRYPTOGRAPHY:
            salt = os.urandom(16)
            kdf = Scrypt(
                salt=salt,
                length=32,
                n=2**14,
                r=8,
                p=1,
                backend=default_backend(),
            )
            key = kdf.derive(password.encode())
            return f"$scrypt$n=16384,r=8,p=1${base64.b64encode(salt).decode()}${base64.b64encode(key).decode()}"
        else:
            # Fallback to PBKDF2
            salt = os.urandom(16)
            key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
            return f"$pbkdf2-sha256$100000${base64.b64encode(salt).decode()}${base64.b64encode(key).decode()}"

    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify a password against hash.

        Args:
            password: Password to verify
            hashed: Hashed password to compare

        Returns:
            True if password matches
        """
        try:
            parts = hashed.split("$")
            if len(parts) < 4:
                return False

            algorithm = parts[1]

            if algorithm.startswith("scrypt") and HAS_CRYPTOGRAPHY:
                # Parse scrypt parameters
                params = dict(p.split("=") for p in parts[2].split(","))
                salt = base64.b64decode(parts[3])
                expected_key = base64.b64decode(parts[4])

                kdf = Scrypt(
                    salt=salt,
                    length=32,
                    n=int(params.get("n", 16384)),
                    r=int(params.get("r", 8)),
                    p=int(params.get("p", 1)),
                    backend=default_backend(),
                )
                try:
                    kdf.verify(password.encode(), expected_key)
                    return True
                except Exception:
                    return False
            else:
                # PBKDF2
                iterations = int(parts[2])
                salt = base64.b64decode(parts[3])
                expected_key = base64.b64decode(parts[4])
                key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
                return hmac.compare_digest(key, expected_key)

        except Exception as e:
            self._logger.error(f"Password verification error: {e}")
            return False

    def hash_data(
        self,
        data: Union[bytes, str],
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ) -> str:
        """
        Hash data.

        Args:
            data: Data to hash
            algorithm: Hash algorithm

        Returns:
            Hex-encoded hash
        """
        if isinstance(data, str):
            data = data.encode()

        if algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).hexdigest()
        elif algorithm == HashAlgorithm.SHA384:
            return hashlib.sha384(data).hexdigest()
        elif algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data).hexdigest()
        elif algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(data).hexdigest()
        else:
            return hashlib.sha256(data).hexdigest()

    def generate_hmac(
        self,
        data: Union[bytes, str],
        key: Optional[bytes] = None,
    ) -> str:
        """Generate HMAC for data."""
        if isinstance(data, str):
            data = data.encode()
        key = key or self._master_key
        return hmac.new(key, data, hashlib.sha256).hexdigest()

    def verify_hmac(
        self,
        data: Union[bytes, str],
        signature: str,
        key: Optional[bytes] = None,
    ) -> bool:
        """Verify HMAC signature."""
        expected = self.generate_hmac(data, key)
        return hmac.compare_digest(expected, signature)

    def derive_key(
        self,
        password: str,
        salt: Optional[bytes] = None,
        length: int = 32,
    ) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password.

        Args:
            password: Password to derive from
            salt: Salt (generated if not provided)
            length: Key length in bytes

        Returns:
            Tuple of (derived_key, salt)
        """
        salt = salt or os.urandom(16)

        if HAS_CRYPTOGRAPHY:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=length,
                salt=salt,
                iterations=100000,
                backend=default_backend(),
            )
            key = kdf.derive(password.encode())
        else:
            key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)[:length]

        return key, salt

    def generate_key(self, length: int = 32) -> bytes:
        """Generate random encryption key."""
        return secrets.token_bytes(length)

    def generate_api_key(self, prefix: str = "sk") -> str:
        """Generate API key."""
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}_{random_part}"

    def generate_token(self, length: int = 32) -> str:
        """Generate URL-safe token."""
        return secrets.token_urlsafe(length)

    def constant_time_compare(self, a: Union[str, bytes], b: Union[str, bytes]) -> bool:
        """Constant-time comparison to prevent timing attacks."""
        if isinstance(a, str):
            a = a.encode()
        if isinstance(b, str):
            b = b.encode()
        return hmac.compare_digest(a, b)
