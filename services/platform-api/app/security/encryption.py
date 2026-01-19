"""
Encryption Services

Enterprise-grade encryption with:
- AES-256-GCM symmetric encryption
- RSA asymmetric encryption
- Hybrid encryption for large data
- Field-level encryption
- Key derivation (PBKDF2, Argon2)
"""

from typing import Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from functools import wraps
import base64
import secrets
import hashlib
import logging
import os

logger = logging.getLogger(__name__)

# Try to import cryptography library
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography library not installed, encryption will be limited")


@dataclass
class EncryptionConfig:
    """Configuration for encryption services."""
    algorithm: str = "AES-256-GCM"
    key_size: int = 256  # bits
    nonce_size: int = 12  # bytes for AES-GCM
    tag_size: int = 16  # bytes for authentication tag
    salt_size: int = 16  # bytes for key derivation
    iterations: int = 100000  # PBKDF2 iterations
    rsa_key_size: int = 4096  # bits for RSA


@dataclass
class EncryptedData:
    """Container for encrypted data."""
    ciphertext: bytes
    nonce: bytes
    tag: Optional[bytes] = None
    salt: Optional[bytes] = None
    algorithm: str = "AES-256-GCM"
    key_id: Optional[str] = None
    encrypted_at: datetime = field(default_factory=datetime.utcnow)

    def to_bytes(self) -> bytes:
        """Serialize to bytes for storage."""
        # Format: version(1) + nonce_len(1) + nonce + salt_len(1) + salt + ciphertext
        parts = [
            b'\x01',  # Version
            bytes([len(self.nonce)]),
            self.nonce,
            bytes([len(self.salt) if self.salt else 0]),
            self.salt or b'',
            self.ciphertext,
        ]
        return b''.join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "EncryptedData":
        """Deserialize from bytes."""
        version = data[0]
        if version != 1:
            raise ValueError(f"Unknown encryption format version: {version}")

        nonce_len = data[1]
        nonce = data[2:2+nonce_len]

        salt_len = data[2+nonce_len]
        salt_start = 3 + nonce_len
        salt = data[salt_start:salt_start+salt_len] if salt_len > 0 else None

        ciphertext = data[salt_start+salt_len:]

        return cls(
            ciphertext=ciphertext,
            nonce=nonce,
            salt=salt,
        )

    def to_base64(self) -> str:
        """Encode to base64 string."""
        return base64.b64encode(self.to_bytes()).decode('utf-8')

    @classmethod
    def from_base64(cls, data: str) -> "EncryptedData":
        """Decode from base64 string."""
        return cls.from_bytes(base64.b64decode(data))


class EncryptionProvider(ABC):
    """Abstract base for encryption providers."""

    @abstractmethod
    def encrypt(self, plaintext: bytes, key: bytes) -> EncryptedData:
        """Encrypt data."""
        pass

    @abstractmethod
    def decrypt(self, encrypted: EncryptedData, key: bytes) -> bytes:
        """Decrypt data."""
        pass


class AESEncryption(EncryptionProvider):
    """
    AES-256-GCM authenticated encryption.

    Provides confidentiality and authenticity with associated data support.
    """

    def __init__(self, config: Optional[EncryptionConfig] = None):
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for AES encryption")
        self.config = config or EncryptionConfig()

    def encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        associated_data: Optional[bytes] = None,
    ) -> EncryptedData:
        """
        Encrypt data with AES-256-GCM.

        Args:
            plaintext: Data to encrypt
            key: 32-byte encryption key
            associated_data: Optional authenticated but unencrypted data

        Returns:
            EncryptedData container
        """
        if len(key) != 32:
            raise ValueError("AES-256 requires 32-byte key")

        # Generate random nonce
        nonce = secrets.token_bytes(self.config.nonce_size)

        # Encrypt
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)

        return EncryptedData(
            ciphertext=ciphertext,
            nonce=nonce,
            algorithm="AES-256-GCM",
        )

    def decrypt(
        self,
        encrypted: EncryptedData,
        key: bytes,
        associated_data: Optional[bytes] = None,
    ) -> bytes:
        """
        Decrypt AES-256-GCM encrypted data.

        Args:
            encrypted: Encrypted data container
            key: 32-byte encryption key
            associated_data: Optional associated data (must match encryption)

        Returns:
            Decrypted plaintext
        """
        if len(key) != 32:
            raise ValueError("AES-256 requires 32-byte key")

        aesgcm = AESGCM(key)
        return aesgcm.decrypt(encrypted.nonce, encrypted.ciphertext, associated_data)


class RSAEncryption(EncryptionProvider):
    """
    RSA asymmetric encryption.

    Used for key exchange and encrypting small amounts of data.
    """

    def __init__(self, config: Optional[EncryptionConfig] = None):
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for RSA encryption")
        self.config = config or EncryptionConfig()
        self._private_key = None
        self._public_key = None

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate RSA key pair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.config.rsa_key_size,
            backend=default_backend(),
        )

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        return private_pem, public_pem

    def load_private_key(self, private_pem: bytes) -> None:
        """Load private key from PEM."""
        self._private_key = serialization.load_pem_private_key(
            private_pem,
            password=None,
            backend=default_backend(),
        )
        self._public_key = self._private_key.public_key()

    def load_public_key(self, public_pem: bytes) -> None:
        """Load public key from PEM."""
        self._public_key = serialization.load_pem_public_key(
            public_pem,
            backend=default_backend(),
        )

    def encrypt(self, plaintext: bytes, key: Optional[bytes] = None) -> EncryptedData:
        """
        Encrypt with RSA public key.

        Note: RSA can only encrypt small amounts of data.
        For larger data, use hybrid encryption.
        """
        if self._public_key is None:
            raise ValueError("Public key not loaded")

        # RSA-OAEP with SHA-256
        ciphertext = self._public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        return EncryptedData(
            ciphertext=ciphertext,
            nonce=b'',  # RSA doesn't use nonce
            algorithm="RSA-OAEP",
        )

    def decrypt(self, encrypted: EncryptedData, key: Optional[bytes] = None) -> bytes:
        """Decrypt with RSA private key."""
        if self._private_key is None:
            raise ValueError("Private key not loaded")

        return self._private_key.decrypt(
            encrypted.ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def sign(self, data: bytes) -> bytes:
        """Sign data with private key."""
        if self._private_key is None:
            raise ValueError("Private key not loaded")

        return self._private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify signature with public key."""
        if self._public_key is None:
            raise ValueError("Public key not loaded")

        try:
            self._public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False


class HybridEncryption:
    """
    Hybrid encryption combining RSA and AES.

    Uses RSA to encrypt a random AES key, then AES to encrypt the data.
    Best of both worlds: asymmetric key exchange with symmetric speed.
    """

    def __init__(self, config: Optional[EncryptionConfig] = None):
        self.config = config or EncryptionConfig()
        self.aes = AESEncryption(config)
        self.rsa = RSAEncryption(config)

    def encrypt(
        self,
        plaintext: bytes,
        public_key: bytes,
    ) -> Dict[str, Any]:
        """
        Encrypt data using hybrid encryption.

        Args:
            plaintext: Data to encrypt
            public_key: RSA public key PEM

        Returns:
            Dict containing encrypted_key and encrypted_data
        """
        # Generate random AES key
        aes_key = secrets.token_bytes(32)

        # Encrypt data with AES
        encrypted_data = self.aes.encrypt(plaintext, aes_key)

        # Encrypt AES key with RSA
        self.rsa.load_public_key(public_key)
        encrypted_key = self.rsa.encrypt(aes_key)

        return {
            "encrypted_key": base64.b64encode(encrypted_key.ciphertext).decode(),
            "encrypted_data": encrypted_data.to_base64(),
            "algorithm": "RSA-AES-256-GCM",
        }

    def decrypt(
        self,
        encrypted: Dict[str, Any],
        private_key: bytes,
    ) -> bytes:
        """
        Decrypt hybrid encrypted data.

        Args:
            encrypted: Dict from encrypt()
            private_key: RSA private key PEM

        Returns:
            Decrypted plaintext
        """
        # Decrypt AES key with RSA
        self.rsa.load_private_key(private_key)
        encrypted_key = EncryptedData(
            ciphertext=base64.b64decode(encrypted["encrypted_key"]),
            nonce=b'',
        )
        aes_key = self.rsa.decrypt(encrypted_key)

        # Decrypt data with AES
        encrypted_data = EncryptedData.from_base64(encrypted["encrypted_data"])
        return self.aes.decrypt(encrypted_data, aes_key)


class PasswordEncryption:
    """
    Password-based encryption using PBKDF2 key derivation.

    Suitable for encrypting data with user-provided passwords.
    """

    def __init__(self, config: Optional[EncryptionConfig] = None):
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required")
        self.config = config or EncryptionConfig()
        self.aes = AESEncryption(config)

    def derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.config.iterations,
            backend=default_backend(),
        )
        return kdf.derive(password.encode('utf-8'))

    def encrypt(self, plaintext: bytes, password: str) -> EncryptedData:
        """Encrypt data with password."""
        salt = secrets.token_bytes(self.config.salt_size)
        key = self.derive_key(password, salt)
        encrypted = self.aes.encrypt(plaintext, key)
        encrypted.salt = salt
        return encrypted

    def decrypt(self, encrypted: EncryptedData, password: str) -> bytes:
        """Decrypt data with password."""
        if not encrypted.salt:
            raise ValueError("Salt required for password decryption")
        key = self.derive_key(password, encrypted.salt)
        return self.aes.decrypt(encrypted, key)


class FieldEncryption:
    """
    Field-level encryption for database columns.

    Encrypts specific fields while keeping others searchable.
    """

    def __init__(
        self,
        key: bytes,
        fields: Optional[set] = None,
        config: Optional[EncryptionConfig] = None,
    ):
        """
        Args:
            key: 32-byte encryption key
            fields: Set of field names to encrypt
            config: Encryption configuration
        """
        self.key = key
        self.fields = fields or set()
        self.aes = AESEncryption(config)

    def encrypt_field(self, field_name: str, value: Any) -> str:
        """Encrypt a field value."""
        if field_name not in self.fields:
            return value

        if value is None:
            return None

        # Convert to bytes
        if isinstance(value, str):
            data = value.encode('utf-8')
        elif isinstance(value, bytes):
            data = value
        else:
            data = str(value).encode('utf-8')

        encrypted = self.aes.encrypt(data, self.key)
        return f"ENC:{encrypted.to_base64()}"

    def decrypt_field(self, field_name: str, value: Any) -> Any:
        """Decrypt a field value."""
        if value is None:
            return None

        if not isinstance(value, str) or not value.startswith("ENC:"):
            return value

        encrypted = EncryptedData.from_base64(value[4:])
        decrypted = self.aes.decrypt(encrypted, self.key)
        return decrypted.decode('utf-8')

    def encrypt_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt fields in a dictionary."""
        return {
            k: self.encrypt_field(k, v)
            for k, v in data.items()
        }

    def decrypt_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt fields in a dictionary."""
        return {
            k: self.decrypt_field(k, v)
            for k, v in data.items()
        }


class EncryptionService:
    """
    High-level encryption service.

    Provides a unified interface for various encryption operations.
    """

    def __init__(
        self,
        config: Optional[EncryptionConfig] = None,
        key: Optional[bytes] = None,
    ):
        self.config = config or EncryptionConfig()
        self._key = key or secrets.token_bytes(32)
        self._aes = AESEncryption(config)
        self._rsa = RSAEncryption(config)
        self._hybrid = HybridEncryption(config)
        self._password = PasswordEncryption(config)

    def encrypt(
        self,
        data: Union[str, bytes],
        key: Optional[bytes] = None,
    ) -> str:
        """Encrypt data and return base64 string."""
        if isinstance(data, str):
            data = data.encode('utf-8')

        encrypted = self._aes.encrypt(data, key or self._key)
        return encrypted.to_base64()

    def decrypt(
        self,
        encrypted_data: str,
        key: Optional[bytes] = None,
    ) -> bytes:
        """Decrypt base64 encrypted string."""
        encrypted = EncryptedData.from_base64(encrypted_data)
        return self._aes.decrypt(encrypted, key or self._key)

    def decrypt_string(
        self,
        encrypted_data: str,
        key: Optional[bytes] = None,
    ) -> str:
        """Decrypt and return as string."""
        return self.decrypt(encrypted_data, key).decode('utf-8')

    def encrypt_with_password(self, data: Union[str, bytes], password: str) -> str:
        """Encrypt data with a password."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        encrypted = self._password.encrypt(data, password)
        return encrypted.to_base64()

    def decrypt_with_password(self, encrypted_data: str, password: str) -> bytes:
        """Decrypt password-encrypted data."""
        encrypted = EncryptedData.from_base64(encrypted_data)
        return self._password.decrypt(encrypted, password)

    def generate_key(self) -> bytes:
        """Generate a new encryption key."""
        return secrets.token_bytes(32)

    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Hash a password securely."""
        salt = salt or secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.config.iterations,
            backend=default_backend(),
        )
        hash_bytes = kdf.derive(password.encode('utf-8'))
        return (
            base64.b64encode(hash_bytes).decode('utf-8'),
            base64.b64encode(salt).decode('utf-8'),
        )

    def verify_password(self, password: str, hash_b64: str, salt_b64: str) -> bool:
        """Verify a password against its hash."""
        expected_hash, _ = self.hash_password(password, base64.b64decode(salt_b64))
        return secrets.compare_digest(expected_hash, hash_b64)


# Convenience functions
def encrypt_field(value: Any, key: bytes, field_name: str = "data") -> str:
    """Encrypt a single field value."""
    encryptor = FieldEncryption(key, {field_name})
    return encryptor.encrypt_field(field_name, value)


def decrypt_field(value: str, key: bytes, field_name: str = "data") -> Any:
    """Decrypt a single field value."""
    encryptor = FieldEncryption(key, {field_name})
    return encryptor.decrypt_field(field_name, value)


# Decorator for encrypting function arguments
def encrypted_args(*field_names: str, key_param: str = "encryption_key"):
    """
    Decorator to automatically encrypt specified arguments.

    Usage:
        @encrypted_args("password", "ssn")
        def save_user(name, password, ssn, encryption_key):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = kwargs.get(key_param)
            if key and field_names:
                encryptor = FieldEncryption(key, set(field_names))
                for field in field_names:
                    if field in kwargs:
                        kwargs[field] = encryptor.encrypt_field(field, kwargs[field])
            return func(*args, **kwargs)
        return wrapper
    return decorator
