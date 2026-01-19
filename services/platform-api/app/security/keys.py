"""
Key Management System

Enterprise key management with:
- Key generation and storage
- Key rotation policies
- Key versioning
- Secure key access
- Integration with KMS (AWS, GCP, Azure, Vault)
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import secrets
import base64
import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)


class KeyType(str, Enum):
    """Types of encryption keys."""
    SYMMETRIC = "symmetric"  # AES keys
    ASYMMETRIC = "asymmetric"  # RSA keys
    HMAC = "hmac"  # HMAC signing keys
    API = "api"  # API keys
    SESSION = "session"  # Session keys


class KeyState(str, Enum):
    """Key lifecycle states."""
    PENDING = "pending"  # Created but not active
    ACTIVE = "active"  # Available for use
    ROTATED = "rotated"  # Replaced by new key, can still decrypt
    DISABLED = "disabled"  # Temporarily disabled
    DESTROYED = "destroyed"  # Permanently deleted


class KeyAlgorithm(str, Enum):
    """Supported key algorithms."""
    AES_256 = "AES-256"
    AES_128 = "AES-128"
    RSA_4096 = "RSA-4096"
    RSA_2048 = "RSA-2048"
    HMAC_SHA256 = "HMAC-SHA256"
    HMAC_SHA512 = "HMAC-SHA512"


@dataclass
class EncryptionKey:
    """Encryption key with metadata."""
    key_id: str
    key_type: KeyType
    algorithm: KeyAlgorithm
    state: KeyState = KeyState.ACTIVE
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    rotated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    # Key material (not serialized to external storage)
    key_material: Optional[bytes] = field(default=None, repr=False)
    public_key: Optional[bytes] = field(default=None, repr=False)
    private_key: Optional[bytes] = field(default=None, repr=False)

    @property
    def is_active(self) -> bool:
        """Check if key is active and not expired."""
        if self.state != KeyState.ACTIVE:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    @property
    def can_decrypt(self) -> bool:
        """Check if key can be used for decryption."""
        return self.state in (KeyState.ACTIVE, KeyState.ROTATED)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without key material)."""
        return {
            "key_id": self.key_id,
            "key_type": self.key_type.value,
            "algorithm": self.algorithm.value,
            "state": self.state.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "rotated_at": self.rotated_at.isoformat() if self.rotated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "description": self.description,
            "tags": self.tags,
        }


@dataclass
class KeyRotationPolicy:
    """Key rotation policy configuration."""
    rotation_interval_days: int = 90
    max_key_age_days: int = 365
    auto_rotate: bool = True
    retain_old_keys: int = 3  # Number of old keys to keep
    notify_before_days: int = 7


class KeyStore(ABC):
    """Abstract key storage backend."""

    @abstractmethod
    async def store(self, key: EncryptionKey, material: bytes) -> None:
        """Store key and its material."""
        pass

    @abstractmethod
    async def retrieve(self, key_id: str) -> Optional[EncryptionKey]:
        """Retrieve key by ID."""
        pass

    @abstractmethod
    async def retrieve_material(self, key_id: str) -> Optional[bytes]:
        """Retrieve key material."""
        pass

    @abstractmethod
    async def list_keys(
        self,
        key_type: Optional[KeyType] = None,
        state: Optional[KeyState] = None,
    ) -> List[EncryptionKey]:
        """List keys matching criteria."""
        pass

    @abstractmethod
    async def update_state(self, key_id: str, state: KeyState) -> None:
        """Update key state."""
        pass

    @abstractmethod
    async def delete(self, key_id: str) -> None:
        """Delete key (permanent)."""
        pass


class InMemoryKeyStore(KeyStore):
    """In-memory key store for development/testing."""

    def __init__(self):
        self._keys: Dict[str, EncryptionKey] = {}
        self._materials: Dict[str, bytes] = {}
        self._lock = asyncio.Lock()

    async def store(self, key: EncryptionKey, material: bytes) -> None:
        """Store key."""
        async with self._lock:
            self._keys[key.key_id] = key
            self._materials[key.key_id] = material

    async def retrieve(self, key_id: str) -> Optional[EncryptionKey]:
        """Retrieve key."""
        return self._keys.get(key_id)

    async def retrieve_material(self, key_id: str) -> Optional[bytes]:
        """Retrieve material."""
        return self._materials.get(key_id)

    async def list_keys(
        self,
        key_type: Optional[KeyType] = None,
        state: Optional[KeyState] = None,
    ) -> List[EncryptionKey]:
        """List keys."""
        keys = list(self._keys.values())
        if key_type:
            keys = [k for k in keys if k.key_type == key_type]
        if state:
            keys = [k for k in keys if k.state == state]
        return keys

    async def update_state(self, key_id: str, state: KeyState) -> None:
        """Update state."""
        if key_id in self._keys:
            self._keys[key_id].state = state

    async def delete(self, key_id: str) -> None:
        """Delete key."""
        async with self._lock:
            self._keys.pop(key_id, None)
            self._materials.pop(key_id, None)


class FileKeyStore(KeyStore):
    """File-based key store with encryption."""

    def __init__(self, directory: str, master_key: bytes):
        """
        Args:
            directory: Directory to store keys
            master_key: Key to encrypt stored keys
        """
        self.directory = directory
        self.master_key = master_key
        os.makedirs(directory, exist_ok=True)

    def _encrypt_material(self, material: bytes) -> bytes:
        """Encrypt key material for storage."""
        from app.security.encryption import AESEncryption
        aes = AESEncryption()
        encrypted = aes.encrypt(material, self.master_key)
        return encrypted.to_bytes()

    def _decrypt_material(self, encrypted: bytes) -> bytes:
        """Decrypt stored key material."""
        from app.security.encryption import AESEncryption, EncryptedData
        aes = AESEncryption()
        data = EncryptedData.from_bytes(encrypted)
        return aes.decrypt(data, self.master_key)

    async def store(self, key: EncryptionKey, material: bytes) -> None:
        """Store key to file."""
        key_file = os.path.join(self.directory, f"{key.key_id}.key")
        meta_file = os.path.join(self.directory, f"{key.key_id}.meta")

        # Encrypt and store material
        encrypted_material = self._encrypt_material(material)
        with open(key_file, 'wb') as f:
            f.write(encrypted_material)

        # Store metadata
        with open(meta_file, 'w') as f:
            json.dump(key.to_dict(), f)

    async def retrieve(self, key_id: str) -> Optional[EncryptionKey]:
        """Retrieve key from file."""
        meta_file = os.path.join(self.directory, f"{key_id}.meta")

        if not os.path.exists(meta_file):
            return None

        with open(meta_file, 'r') as f:
            data = json.load(f)

        return EncryptionKey(
            key_id=data["key_id"],
            key_type=KeyType(data["key_type"]),
            algorithm=KeyAlgorithm(data["algorithm"]),
            state=KeyState(data["state"]),
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            rotated_at=datetime.fromisoformat(data["rotated_at"]) if data.get("rotated_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            description=data.get("description"),
            tags=data.get("tags", {}),
        )

    async def retrieve_material(self, key_id: str) -> Optional[bytes]:
        """Retrieve key material from file."""
        key_file = os.path.join(self.directory, f"{key_id}.key")

        if not os.path.exists(key_file):
            return None

        with open(key_file, 'rb') as f:
            encrypted = f.read()

        return self._decrypt_material(encrypted)

    async def list_keys(
        self,
        key_type: Optional[KeyType] = None,
        state: Optional[KeyState] = None,
    ) -> List[EncryptionKey]:
        """List keys from directory."""
        keys = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.meta'):
                key_id = filename[:-5]
                key = await self.retrieve(key_id)
                if key:
                    if key_type and key.key_type != key_type:
                        continue
                    if state and key.state != state:
                        continue
                    keys.append(key)
        return keys

    async def update_state(self, key_id: str, state: KeyState) -> None:
        """Update key state."""
        key = await self.retrieve(key_id)
        if key:
            key.state = state
            meta_file = os.path.join(self.directory, f"{key_id}.meta")
            with open(meta_file, 'w') as f:
                json.dump(key.to_dict(), f)

    async def delete(self, key_id: str) -> None:
        """Delete key files."""
        key_file = os.path.join(self.directory, f"{key_id}.key")
        meta_file = os.path.join(self.directory, f"{key_id}.meta")

        if os.path.exists(key_file):
            os.remove(key_file)
        if os.path.exists(meta_file):
            os.remove(meta_file)


class VaultKeyStore(KeyStore):
    """HashiCorp Vault key store."""

    def __init__(
        self,
        vault_url: str,
        token: str,
        mount_path: str = "secret",
    ):
        self.vault_url = vault_url
        self.token = token
        self.mount_path = mount_path

    async def store(self, key: EncryptionKey, material: bytes) -> None:
        """Store key in Vault."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            url = f"{self.vault_url}/v1/{self.mount_path}/data/keys/{key.key_id}"
            headers = {"X-Vault-Token": self.token}
            data = {
                "data": {
                    "material": base64.b64encode(material).decode(),
                    "metadata": key.to_dict(),
                }
            }

            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    raise Exception(f"Vault store failed: {await response.text()}")

    async def retrieve(self, key_id: str) -> Optional[EncryptionKey]:
        """Retrieve key from Vault."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            url = f"{self.vault_url}/v1/{self.mount_path}/data/keys/{key_id}"
            headers = {"X-Vault-Token": self.token}

            async with session.get(url, headers=headers) as response:
                if response.status == 404:
                    return None
                if response.status != 200:
                    raise Exception(f"Vault retrieve failed: {await response.text()}")

                result = await response.json()
                data = result["data"]["data"]["metadata"]

                return EncryptionKey(
                    key_id=data["key_id"],
                    key_type=KeyType(data["key_type"]),
                    algorithm=KeyAlgorithm(data["algorithm"]),
                    state=KeyState(data["state"]),
                    version=data["version"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                )

    async def retrieve_material(self, key_id: str) -> Optional[bytes]:
        """Retrieve material from Vault."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            url = f"{self.vault_url}/v1/{self.mount_path}/data/keys/{key_id}"
            headers = {"X-Vault-Token": self.token}

            async with session.get(url, headers=headers) as response:
                if response.status == 404:
                    return None
                if response.status != 200:
                    return None

                result = await response.json()
                material_b64 = result["data"]["data"]["material"]
                return base64.b64decode(material_b64)

    async def list_keys(
        self,
        key_type: Optional[KeyType] = None,
        state: Optional[KeyState] = None,
    ) -> List[EncryptionKey]:
        """List keys from Vault."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            url = f"{self.vault_url}/v1/{self.mount_path}/metadata/keys"
            headers = {"X-Vault-Token": self.token}

            async with session.request("LIST", url, headers=headers) as response:
                if response.status != 200:
                    return []

                result = await response.json()
                key_ids = result.get("data", {}).get("keys", [])

                keys = []
                for key_id in key_ids:
                    key = await self.retrieve(key_id.rstrip('/'))
                    if key:
                        if key_type and key.key_type != key_type:
                            continue
                        if state and key.state != state:
                            continue
                        keys.append(key)

                return keys

    async def update_state(self, key_id: str, state: KeyState) -> None:
        """Update key state in Vault."""
        key = await self.retrieve(key_id)
        if key:
            key.state = state
            material = await self.retrieve_material(key_id)
            if material:
                await self.store(key, material)

    async def delete(self, key_id: str) -> None:
        """Delete key from Vault."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            url = f"{self.vault_url}/v1/{self.mount_path}/metadata/keys/{key_id}"
            headers = {"X-Vault-Token": self.token}

            async with session.delete(url, headers=headers) as response:
                if response.status not in (200, 204, 404):
                    raise Exception(f"Vault delete failed: {await response.text()}")


class KeyManager:
    """
    Enterprise key management service.

    Handles key lifecycle, rotation, and access.
    """

    def __init__(
        self,
        store: Optional[KeyStore] = None,
        rotation_policy: Optional[KeyRotationPolicy] = None,
    ):
        self.store = store or InMemoryKeyStore()
        self.rotation_policy = rotation_policy or KeyRotationPolicy()
        self._active_keys: Dict[KeyType, EncryptionKey] = {}
        self._lock = asyncio.Lock()

    async def generate_key(
        self,
        key_type: KeyType,
        algorithm: Optional[KeyAlgorithm] = None,
        expires_in_days: Optional[int] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> EncryptionKey:
        """Generate a new encryption key."""
        # Determine algorithm
        if algorithm is None:
            if key_type == KeyType.SYMMETRIC:
                algorithm = KeyAlgorithm.AES_256
            elif key_type == KeyType.ASYMMETRIC:
                algorithm = KeyAlgorithm.RSA_4096
            elif key_type == KeyType.HMAC:
                algorithm = KeyAlgorithm.HMAC_SHA256
            else:
                algorithm = KeyAlgorithm.AES_256

        # Generate key ID
        key_id = f"key-{secrets.token_hex(16)}"

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Generate key material
        if algorithm in (KeyAlgorithm.AES_256,):
            material = secrets.token_bytes(32)
        elif algorithm in (KeyAlgorithm.AES_128,):
            material = secrets.token_bytes(16)
        elif algorithm in (KeyAlgorithm.HMAC_SHA256,):
            material = secrets.token_bytes(32)
        elif algorithm in (KeyAlgorithm.HMAC_SHA512,):
            material = secrets.token_bytes(64)
        elif algorithm in (KeyAlgorithm.RSA_4096, KeyAlgorithm.RSA_2048):
            from app.security.encryption import RSAEncryption, EncryptionConfig
            rsa = RSAEncryption(EncryptionConfig(
                rsa_key_size=4096 if algorithm == KeyAlgorithm.RSA_4096 else 2048
            ))
            private_pem, public_pem = rsa.generate_keypair()
            material = private_pem  # Store private key as material
        else:
            material = secrets.token_bytes(32)

        # Create key object
        key = EncryptionKey(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            state=KeyState.ACTIVE,
            expires_at=expires_at,
            description=description,
            tags=tags or {},
        )

        # Store key
        await self.store.store(key, material)

        logger.info(f"Generated key {key_id} ({key_type.value}/{algorithm.value})")
        return key

    async def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get key by ID."""
        return await self.store.retrieve(key_id)

    async def get_key_material(self, key_id: str) -> Optional[bytes]:
        """Get key material (requires authorization)."""
        key = await self.store.retrieve(key_id)
        if not key or not key.can_decrypt:
            return None
        return await self.store.retrieve_material(key_id)

    async def get_active_key(self, key_type: KeyType) -> Optional[EncryptionKey]:
        """Get the active key for a type."""
        keys = await self.store.list_keys(key_type=key_type, state=KeyState.ACTIVE)
        if not keys:
            return None
        # Return the newest active key
        return max(keys, key=lambda k: k.created_at)

    async def rotate_key(self, key_id: str) -> EncryptionKey:
        """Rotate a key, creating a new version."""
        async with self._lock:
            old_key = await self.store.retrieve(key_id)
            if not old_key:
                raise ValueError(f"Key not found: {key_id}")

            # Mark old key as rotated
            old_key.state = KeyState.ROTATED
            old_key.rotated_at = datetime.utcnow()
            await self.store.update_state(key_id, KeyState.ROTATED)

            # Generate new key with incremented version
            new_key = await self.generate_key(
                key_type=old_key.key_type,
                algorithm=old_key.algorithm,
                description=old_key.description,
                tags=old_key.tags,
            )
            new_key.version = old_key.version + 1

            logger.info(f"Rotated key {key_id} to version {new_key.version}")
            return new_key

    async def disable_key(self, key_id: str) -> None:
        """Disable a key."""
        await self.store.update_state(key_id, KeyState.DISABLED)
        logger.warning(f"Disabled key {key_id}")

    async def destroy_key(self, key_id: str) -> None:
        """Permanently destroy a key."""
        await self.store.delete(key_id)
        logger.warning(f"Destroyed key {key_id}")

    async def check_rotation(self) -> List[str]:
        """Check for keys that need rotation."""
        keys_to_rotate = []

        all_keys = await self.store.list_keys(state=KeyState.ACTIVE)
        now = datetime.utcnow()

        for key in all_keys:
            age_days = (now - key.created_at).days

            if age_days >= self.rotation_policy.rotation_interval_days:
                keys_to_rotate.append(key.key_id)
            elif self.rotation_policy.notify_before_days:
                days_until_rotation = self.rotation_policy.rotation_interval_days - age_days
                if days_until_rotation <= self.rotation_policy.notify_before_days:
                    logger.warning(
                        f"Key {key.key_id} will need rotation in {days_until_rotation} days"
                    )

        return keys_to_rotate

    async def auto_rotate(self) -> List[EncryptionKey]:
        """Automatically rotate keys that need it."""
        if not self.rotation_policy.auto_rotate:
            return []

        keys_to_rotate = await self.check_rotation()
        rotated_keys = []

        for key_id in keys_to_rotate:
            try:
                new_key = await self.rotate_key(key_id)
                rotated_keys.append(new_key)
            except Exception as e:
                logger.error(f"Failed to rotate key {key_id}: {e}")

        # Clean up old rotated keys
        await self._cleanup_old_keys()

        return rotated_keys

    async def _cleanup_old_keys(self) -> None:
        """Remove old rotated keys beyond retention limit."""
        rotated_keys = await self.store.list_keys(state=KeyState.ROTATED)

        # Group by type
        by_type: Dict[KeyType, List[EncryptionKey]] = {}
        for key in rotated_keys:
            if key.key_type not in by_type:
                by_type[key.key_type] = []
            by_type[key.key_type].append(key)

        for key_type, keys in by_type.items():
            # Sort by rotation time, oldest first
            keys.sort(key=lambda k: k.rotated_at or k.created_at)

            # Remove oldest beyond retention limit
            while len(keys) > self.rotation_policy.retain_old_keys:
                old_key = keys.pop(0)
                await self.destroy_key(old_key.key_id)


class KeyRotation:
    """
    Key rotation scheduler.

    Runs periodic key rotation checks.
    """

    def __init__(
        self,
        key_manager: KeyManager,
        check_interval_hours: int = 24,
    ):
        self.key_manager = key_manager
        self.check_interval_hours = check_interval_hours
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start rotation scheduler."""
        self._running = True
        self._task = asyncio.create_task(self._rotation_loop())
        logger.info("Key rotation scheduler started")

    async def stop(self) -> None:
        """Stop rotation scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Key rotation scheduler stopped")

    async def _rotation_loop(self) -> None:
        """Background rotation loop."""
        while self._running:
            try:
                rotated = await self.key_manager.auto_rotate()
                if rotated:
                    logger.info(f"Auto-rotated {len(rotated)} keys")
            except Exception as e:
                logger.error(f"Key rotation error: {e}")

            await asyncio.sleep(self.check_interval_hours * 3600)
