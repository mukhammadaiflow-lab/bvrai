"""
Key Management System
=====================

Secure key storage, rotation, and lifecycle management for
encryption keys across the platform.

Author: Platform Security Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import secrets
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import structlog

from bvrai_core.security.encryption import (
    DataEncryptor,
    EncryptedData,
    EncryptionAlgorithm,
    KeyType,
    KeyMetadata,
)

logger = structlog.get_logger(__name__)


class KeyStatus(str, Enum):
    """Status of an encryption key."""

    ACTIVE = "active"
    PENDING_ROTATION = "pending_rotation"
    DEPRECATED = "deprecated"
    DESTROYED = "destroyed"
    SUSPENDED = "suspended"


@dataclass
class KeyRotationPolicy:
    """Policy for key rotation."""

    rotation_interval_days: int = 90
    auto_rotate: bool = True
    notify_before_days: int = 7
    max_versions: int = 5
    destroy_after_days: int = 30
    require_approval: bool = False

    def should_rotate(self, created_at: datetime, last_rotated: Optional[datetime] = None) -> bool:
        """Check if key should be rotated."""
        reference_time = last_rotated or created_at
        age_days = (datetime.utcnow() - reference_time).days
        return age_days >= self.rotation_interval_days


@dataclass
class MasterKey:
    """Master encryption key."""

    id: str = field(default_factory=lambda: f"mk_{uuid4().hex[:16]}")
    key_material: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    status: KeyStatus = KeyStatus.ACTIVE
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    rotated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.expires_at:
            self.expires_at = self.created_at + timedelta(days=365)

    @property
    def key_id_hash(self) -> str:
        """Get hash of key ID for logging."""
        return hashlib.sha256(self.id.encode()).hexdigest()[:8]

    @property
    def is_active(self) -> bool:
        """Check if key is active."""
        return self.status == KeyStatus.ACTIVE and not self.is_expired

    @property
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self, include_key: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "algorithm": self.algorithm.value,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "rotated_at": self.rotated_at.isoformat() if self.rotated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
        }
        if include_key:
            data["key_material"] = base64.b64encode(self.key_material).decode()
        return data


@dataclass
class DataKey:
    """Data encryption key (DEK) encrypted under a master key."""

    id: str = field(default_factory=lambda: f"dk_{uuid4().hex[:16]}")
    encrypted_key: bytes = b""  # Encrypted under master key
    master_key_id: str = ""
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    status: KeyStatus = KeyStatus.ACTIVE
    purpose: str = ""  # e.g., "recordings", "pii", "credentials"
    organization_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if key is active."""
        return self.status == KeyStatus.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "master_key_id": self.master_key_id,
            "algorithm": self.algorithm.value,
            "status": self.status.value,
            "purpose": self.purpose,
            "organization_id": self.organization_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class KeyStore(ABC):
    """Abstract base class for key storage."""

    @abstractmethod
    async def store_master_key(self, key: MasterKey) -> None:
        """Store a master key."""
        pass

    @abstractmethod
    async def get_master_key(self, key_id: str) -> Optional[MasterKey]:
        """Get a master key by ID."""
        pass

    @abstractmethod
    async def list_master_keys(self, status: Optional[KeyStatus] = None) -> List[MasterKey]:
        """List all master keys."""
        pass

    @abstractmethod
    async def delete_master_key(self, key_id: str) -> bool:
        """Delete a master key."""
        pass

    @abstractmethod
    async def store_data_key(self, key: DataKey) -> None:
        """Store a data key."""
        pass

    @abstractmethod
    async def get_data_key(self, key_id: str) -> Optional[DataKey]:
        """Get a data key by ID."""
        pass

    @abstractmethod
    async def list_data_keys(
        self,
        organization_id: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> List[DataKey]:
        """List data keys."""
        pass


class InMemoryKeyStore(KeyStore):
    """In-memory key store for development/testing."""

    def __init__(self):
        self._master_keys: Dict[str, MasterKey] = {}
        self._data_keys: Dict[str, DataKey] = {}
        self._lock = asyncio.Lock()

    async def store_master_key(self, key: MasterKey) -> None:
        """Store a master key."""
        async with self._lock:
            self._master_keys[key.id] = key

    async def get_master_key(self, key_id: str) -> Optional[MasterKey]:
        """Get a master key by ID."""
        return self._master_keys.get(key_id)

    async def list_master_keys(self, status: Optional[KeyStatus] = None) -> List[MasterKey]:
        """List all master keys."""
        keys = list(self._master_keys.values())
        if status:
            keys = [k for k in keys if k.status == status]
        return sorted(keys, key=lambda k: k.created_at, reverse=True)

    async def delete_master_key(self, key_id: str) -> bool:
        """Delete a master key."""
        async with self._lock:
            if key_id in self._master_keys:
                del self._master_keys[key_id]
                return True
            return False

    async def store_data_key(self, key: DataKey) -> None:
        """Store a data key."""
        async with self._lock:
            self._data_keys[key.id] = key

    async def get_data_key(self, key_id: str) -> Optional[DataKey]:
        """Get a data key by ID."""
        return self._data_keys.get(key_id)

    async def list_data_keys(
        self,
        organization_id: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> List[DataKey]:
        """List data keys."""
        keys = list(self._data_keys.values())
        if organization_id:
            keys = [k for k in keys if k.organization_id == organization_id]
        if purpose:
            keys = [k for k in keys if k.purpose == purpose]
        return keys


class EncryptedKeyStore(KeyStore):
    """File-based key store with encryption at rest."""

    def __init__(
        self,
        storage_path: str,
        encryption_key: bytes,
    ):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._encryptor = DataEncryptor(encryption_key)
        self._lock = asyncio.Lock()

    async def store_master_key(self, key: MasterKey) -> None:
        """Store a master key encrypted."""
        async with self._lock:
            key_data = key.to_dict(include_key=True)
            encrypted = self._encryptor.encrypt(json.dumps(key_data).encode())

            key_path = self._storage_path / f"master_{key.id}.key"
            with open(key_path, "wb") as f:
                f.write(encrypted.to_bytes())

    async def get_master_key(self, key_id: str) -> Optional[MasterKey]:
        """Get and decrypt a master key."""
        key_path = self._storage_path / f"master_{key_id}.key"
        if not key_path.exists():
            return None

        with open(key_path, "rb") as f:
            encrypted = EncryptedData.from_bytes(f.read())

        decrypted = self._encryptor.decrypt(encrypted)
        data = json.loads(decrypted)

        return MasterKey(
            id=data["id"],
            key_material=base64.b64decode(data["key_material"]),
            algorithm=EncryptionAlgorithm(data["algorithm"]),
            status=KeyStatus(data["status"]),
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            rotated_at=datetime.fromisoformat(data["rotated_at"]) if data.get("rotated_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            description=data.get("description", ""),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    async def list_master_keys(self, status: Optional[KeyStatus] = None) -> List[MasterKey]:
        """List all master keys."""
        keys = []
        for key_file in self._storage_path.glob("master_*.key"):
            key_id = key_file.stem.replace("master_", "")
            key = await self.get_master_key(key_id)
            if key:
                if status is None or key.status == status:
                    keys.append(key)
        return sorted(keys, key=lambda k: k.created_at, reverse=True)

    async def delete_master_key(self, key_id: str) -> bool:
        """Delete a master key."""
        key_path = self._storage_path / f"master_{key_id}.key"
        if key_path.exists():
            key_path.unlink()
            return True
        return False

    async def store_data_key(self, key: DataKey) -> None:
        """Store a data key encrypted."""
        async with self._lock:
            key_data = {
                "id": key.id,
                "encrypted_key": base64.b64encode(key.encrypted_key).decode(),
                "master_key_id": key.master_key_id,
                "algorithm": key.algorithm.value,
                "status": key.status.value,
                "purpose": key.purpose,
                "organization_id": key.organization_id,
                "created_at": key.created_at.isoformat(),
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "metadata": key.metadata,
            }
            encrypted = self._encryptor.encrypt(json.dumps(key_data).encode())

            key_path = self._storage_path / f"data_{key.id}.key"
            with open(key_path, "wb") as f:
                f.write(encrypted.to_bytes())

    async def get_data_key(self, key_id: str) -> Optional[DataKey]:
        """Get and decrypt a data key."""
        key_path = self._storage_path / f"data_{key_id}.key"
        if not key_path.exists():
            return None

        with open(key_path, "rb") as f:
            encrypted = EncryptedData.from_bytes(f.read())

        decrypted = self._encryptor.decrypt(encrypted)
        data = json.loads(decrypted)

        return DataKey(
            id=data["id"],
            encrypted_key=base64.b64decode(data["encrypted_key"]),
            master_key_id=data["master_key_id"],
            algorithm=EncryptionAlgorithm(data["algorithm"]),
            status=KeyStatus(data["status"]),
            purpose=data.get("purpose", ""),
            organization_id=data.get("organization_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
        )

    async def list_data_keys(
        self,
        organization_id: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> List[DataKey]:
        """List data keys."""
        keys = []
        for key_file in self._storage_path.glob("data_*.key"):
            key_id = key_file.stem.replace("data_", "")
            key = await self.get_data_key(key_id)
            if key:
                if organization_id and key.organization_id != organization_id:
                    continue
                if purpose and key.purpose != purpose:
                    continue
                keys.append(key)
        return keys


class KeyManager:
    """
    Key Management Service.

    Provides secure key generation, storage, rotation, and lifecycle
    management for encryption keys.
    """

    def __init__(
        self,
        store: KeyStore,
        rotation_policy: Optional[KeyRotationPolicy] = None,
    ):
        self._store = store
        self._rotation_policy = rotation_policy or KeyRotationPolicy()
        self._active_master_key: Optional[MasterKey] = None
        self._key_cache: Dict[str, bytes] = {}
        self._rotation_callbacks: List[Callable[[MasterKey, MasterKey], None]] = []
        self._logger = structlog.get_logger("key_manager")
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize key manager and ensure active master key exists."""
        # Load or create master key
        master_keys = await self._store.list_master_keys(status=KeyStatus.ACTIVE)

        if master_keys:
            self._active_master_key = master_keys[0]
            self._logger.info(f"Loaded active master key: {self._active_master_key.key_id_hash}")
        else:
            # Generate new master key
            self._active_master_key = await self.create_master_key(
                description="Auto-generated master key",
                tags=["auto-generated"],
            )
            self._logger.info(f"Created new master key: {self._active_master_key.key_id_hash}")

    @property
    def active_master_key(self) -> Optional[MasterKey]:
        """Get the active master key."""
        return self._active_master_key

    def add_rotation_callback(
        self,
        callback: Callable[[MasterKey, MasterKey], None],
    ) -> None:
        """Add callback to be called on key rotation."""
        self._rotation_callbacks.append(callback)

    async def create_master_key(
        self,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        description: str = "",
        tags: Optional[List[str]] = None,
        expires_in_days: int = 365,
    ) -> MasterKey:
        """Create a new master key."""
        key = MasterKey(
            algorithm=algorithm,
            description=description,
            tags=tags or [],
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days),
        )

        await self._store.store_master_key(key)
        self._logger.info(f"Created master key: {key.key_id_hash}")

        return key

    async def rotate_master_key(
        self,
        reason: str = "scheduled",
    ) -> Tuple[MasterKey, MasterKey]:
        """
        Rotate the active master key.

        Creates a new master key and deprecates the old one.

        Returns:
            Tuple of (new_key, old_key)
        """
        async with self._lock:
            if not self._active_master_key:
                raise ValueError("No active master key to rotate")

            old_key = self._active_master_key

            # Create new master key
            new_key = await self.create_master_key(
                algorithm=old_key.algorithm,
                description=f"Rotated from {old_key.id}: {reason}",
                tags=old_key.tags + ["rotated"],
            )
            new_key.version = old_key.version + 1

            # Update old key status
            old_key.status = KeyStatus.DEPRECATED
            old_key.rotated_at = datetime.utcnow()
            await self._store.store_master_key(old_key)

            # Set new active key
            self._active_master_key = new_key

            self._logger.info(
                f"Rotated master key: {old_key.key_id_hash} -> {new_key.key_id_hash}",
                reason=reason,
            )

            # Call rotation callbacks
            for callback in self._rotation_callbacks:
                try:
                    callback(old_key, new_key)
                except Exception as e:
                    self._logger.error(f"Rotation callback error: {e}")

            return new_key, old_key

    async def create_data_key(
        self,
        purpose: str,
        organization_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[DataKey, bytes]:
        """
        Create a new data encryption key.

        Returns:
            Tuple of (DataKey metadata, plaintext key material)
        """
        if not self._active_master_key:
            await self.initialize()

        # Generate random key
        plaintext_key = secrets.token_bytes(32)

        # Encrypt under master key
        encryptor = DataEncryptor(self._active_master_key.key_material)
        encrypted = encryptor.encrypt(plaintext_key)

        # Create data key record
        data_key = DataKey(
            encrypted_key=encrypted.to_bytes(),
            master_key_id=self._active_master_key.id,
            algorithm=self._active_master_key.algorithm,
            purpose=purpose,
            organization_id=organization_id,
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days) if expires_in_days else None,
        )

        await self._store.store_data_key(data_key)
        self._key_cache[data_key.id] = plaintext_key

        self._logger.debug(f"Created data key: {data_key.id} for purpose: {purpose}")

        return data_key, plaintext_key

    async def decrypt_data_key(self, key_id: str) -> Optional[bytes]:
        """
        Decrypt a data key.

        Returns:
            Plaintext key material
        """
        # Check cache
        if key_id in self._key_cache:
            return self._key_cache[key_id]

        # Get data key
        data_key = await self._store.get_data_key(key_id)
        if not data_key:
            return None

        # Get master key
        master_key = await self._store.get_master_key(data_key.master_key_id)
        if not master_key:
            self._logger.error(f"Master key not found: {data_key.master_key_id}")
            return None

        # Decrypt
        encrypted = EncryptedData.from_bytes(data_key.encrypted_key)
        encryptor = DataEncryptor(master_key.key_material)
        plaintext = encryptor.decrypt(encrypted)

        # Cache
        self._key_cache[key_id] = plaintext

        return plaintext

    async def get_or_create_data_key(
        self,
        purpose: str,
        organization_id: Optional[str] = None,
    ) -> Tuple[DataKey, bytes]:
        """
        Get existing data key or create new one.

        Returns:
            Tuple of (DataKey metadata, plaintext key material)
        """
        # Look for existing key
        existing_keys = await self._store.list_data_keys(
            organization_id=organization_id,
            purpose=purpose,
        )

        for key in existing_keys:
            if key.is_active:
                plaintext = await self.decrypt_data_key(key.id)
                if plaintext:
                    return key, plaintext

        # Create new key
        return await self.create_data_key(purpose, organization_id)

    async def destroy_key(self, key_id: str) -> bool:
        """
        Destroy a key (mark as destroyed and clear material).

        This is a permanent operation.
        """
        # Try master key first
        master_key = await self._store.get_master_key(key_id)
        if master_key:
            if master_key == self._active_master_key:
                raise ValueError("Cannot destroy active master key")

            master_key.status = KeyStatus.DESTROYED
            master_key.key_material = b""  # Clear material
            await self._store.store_master_key(master_key)
            self._logger.warning(f"Destroyed master key: {master_key.key_id_hash}")
            return True

        # Try data key
        data_key = await self._store.get_data_key(key_id)
        if data_key:
            data_key.status = KeyStatus.DESTROYED
            data_key.encrypted_key = b""
            await self._store.store_data_key(data_key)

            # Clear from cache
            self._key_cache.pop(key_id, None)

            self._logger.warning(f"Destroyed data key: {key_id}")
            return True

        return False

    async def check_rotation_needed(self) -> List[MasterKey]:
        """Check which keys need rotation."""
        keys_to_rotate = []
        master_keys = await self._store.list_master_keys(status=KeyStatus.ACTIVE)

        for key in master_keys:
            if self._rotation_policy.should_rotate(key.created_at, key.rotated_at):
                keys_to_rotate.append(key)

        return keys_to_rotate

    async def cleanup_old_keys(self) -> int:
        """Clean up old deprecated keys."""
        cleaned = 0
        cutoff = datetime.utcnow() - timedelta(days=self._rotation_policy.destroy_after_days)

        for status in [KeyStatus.DEPRECATED, KeyStatus.DESTROYED]:
            keys = await self._store.list_master_keys(status=status)
            for key in keys:
                if key.rotated_at and key.rotated_at < cutoff:
                    await self._store.delete_master_key(key.id)
                    cleaned += 1
                    self._logger.info(f"Cleaned up old key: {key.key_id_hash}")

        return cleaned

    async def get_key_statistics(self) -> Dict[str, Any]:
        """Get key management statistics."""
        master_keys = await self._store.list_master_keys()

        stats = {
            "total_master_keys": len(master_keys),
            "by_status": defaultdict(int),
            "active_key_id": self._active_master_key.key_id_hash if self._active_master_key else None,
            "active_key_version": self._active_master_key.version if self._active_master_key else 0,
            "keys_needing_rotation": 0,
            "cached_data_keys": len(self._key_cache),
        }

        for key in master_keys:
            stats["by_status"][key.status.value] += 1
            if self._rotation_policy.should_rotate(key.created_at, key.rotated_at):
                stats["keys_needing_rotation"] += 1

        return dict(stats)
