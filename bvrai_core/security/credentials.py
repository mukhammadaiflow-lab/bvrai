"""
Credential Store
================

Secure storage for sensitive credentials including API keys,
database passwords, OAuth tokens, and other secrets.

Author: Platform Security Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import structlog

from bvrai_core.security.encryption import DataEncryptor, EncryptedData, EncryptionAlgorithm

logger = structlog.get_logger(__name__)


class CredentialType(str, Enum):
    """Types of credentials."""

    API_KEY = "api_key"
    DATABASE = "database"
    OAUTH_TOKEN = "oauth_token"
    SSH_KEY = "ssh_key"
    TLS_CERTIFICATE = "tls_certificate"
    SECRET_KEY = "secret_key"
    PASSWORD = "password"
    WEBHOOK_SECRET = "webhook_secret"
    INTEGRATION = "integration"
    CUSTOM = "custom"


class CredentialStatus(str, Enum):
    """Status of a credential."""

    ACTIVE = "active"
    DISABLED = "disabled"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING_ROTATION = "pending_rotation"


@dataclass
class Credential:
    """A stored credential."""

    id: str = field(default_factory=lambda: f"cred_{uuid4().hex[:16]}")
    name: str = ""
    description: str = ""
    credential_type: CredentialType = CredentialType.SECRET_KEY
    status: CredentialStatus = CredentialStatus.ACTIVE

    # Ownership
    organization_id: str = ""
    created_by: Optional[str] = None

    # The encrypted value
    encrypted_value: Optional[bytes] = None
    encryption_key_id: Optional[str] = None

    # Metadata
    provider: Optional[str] = None  # e.g., "aws", "openai", "twilio"
    environment: str = "production"  # production, staging, development
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Expiration
    expires_at: Optional[datetime] = None
    rotation_interval_days: Optional[int] = None
    last_rotated_at: Optional[datetime] = None

    # Audit
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_expired(self) -> bool:
        """Check if credential is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def needs_rotation(self) -> bool:
        """Check if credential needs rotation."""
        if not self.rotation_interval_days:
            return False
        last_rotation = self.last_rotated_at or self.created_at
        days_since = (datetime.utcnow() - last_rotation).days
        return days_since >= self.rotation_interval_days

    def to_dict(self, include_encrypted: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "credential_type": self.credential_type.value,
            "status": self.status.value,
            "organization_id": self.organization_id,
            "created_by": self.created_by,
            "provider": self.provider,
            "environment": self.environment,
            "tags": self.tags,
            "metadata": self.metadata,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "rotation_interval_days": self.rotation_interval_days,
            "last_rotated_at": self.last_rotated_at.isoformat() if self.last_rotated_at else None,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_expired": self.is_expired,
            "needs_rotation": self.needs_rotation,
        }
        if include_encrypted:
            data["encrypted_value"] = base64.b64encode(self.encrypted_value).decode() if self.encrypted_value else None
            data["encryption_key_id"] = self.encryption_key_id
        return data


@dataclass
class CredentialVersion:
    """A version of a credential (for rotation history)."""

    id: str = field(default_factory=lambda: f"credv_{uuid4().hex[:16]}")
    credential_id: str = ""
    version: int = 1
    encrypted_value: bytes = b""
    encryption_key_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    is_current: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessPolicy:
    """Policy controlling access to credentials."""

    id: str = field(default_factory=lambda: f"policy_{uuid4().hex[:12]}")
    credential_id: str = ""

    # Who can access
    allowed_users: List[str] = field(default_factory=list)
    allowed_roles: List[str] = field(default_factory=list)
    allowed_services: List[str] = field(default_factory=list)

    # What they can do
    can_read: bool = True
    can_update: bool = False
    can_delete: bool = False
    can_rotate: bool = False

    # Restrictions
    allowed_ips: List[str] = field(default_factory=list)
    max_access_per_hour: Optional[int] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    created_at: datetime = field(default_factory=datetime.utcnow)


class CredentialStore(ABC):
    """Abstract base class for credential storage."""

    @abstractmethod
    async def store(self, credential: Credential) -> None:
        """Store a credential."""
        pass

    @abstractmethod
    async def get(self, credential_id: str) -> Optional[Credential]:
        """Get a credential by ID."""
        pass

    @abstractmethod
    async def list(
        self,
        organization_id: str,
        credential_type: Optional[CredentialType] = None,
        provider: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> List[Credential]:
        """List credentials."""
        pass

    @abstractmethod
    async def delete(self, credential_id: str) -> bool:
        """Delete a credential."""
        pass

    @abstractmethod
    async def update(self, credential: Credential) -> None:
        """Update a credential."""
        pass


class InMemoryCredentialStore(CredentialStore):
    """In-memory credential store for development/testing."""

    def __init__(self):
        self._credentials: Dict[str, Credential] = {}
        self._by_org: Dict[str, Set[str]] = defaultdict(set)
        self._versions: Dict[str, List[CredentialVersion]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def store(self, credential: Credential) -> None:
        """Store a credential."""
        async with self._lock:
            self._credentials[credential.id] = credential
            self._by_org[credential.organization_id].add(credential.id)

    async def get(self, credential_id: str) -> Optional[Credential]:
        """Get a credential by ID."""
        return self._credentials.get(credential_id)

    async def list(
        self,
        organization_id: str,
        credential_type: Optional[CredentialType] = None,
        provider: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> List[Credential]:
        """List credentials."""
        cred_ids = self._by_org.get(organization_id, set())
        credentials = []

        for cred_id in cred_ids:
            cred = self._credentials.get(cred_id)
            if not cred:
                continue
            if credential_type and cred.credential_type != credential_type:
                continue
            if provider and cred.provider != provider:
                continue
            if environment and cred.environment != environment:
                continue
            credentials.append(cred)

        return credentials

    async def delete(self, credential_id: str) -> bool:
        """Delete a credential."""
        async with self._lock:
            cred = self._credentials.get(credential_id)
            if not cred:
                return False

            del self._credentials[credential_id]
            self._by_org[cred.organization_id].discard(credential_id)
            return True

    async def update(self, credential: Credential) -> None:
        """Update a credential."""
        async with self._lock:
            self._credentials[credential.id] = credential


class EncryptedFileCredentialStore(CredentialStore):
    """File-based credential store with encryption."""

    def __init__(
        self,
        storage_path: str,
        encryption_key: bytes,
    ):
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._encryptor = DataEncryptor(encryption_key)
        self._lock = asyncio.Lock()
        self._cache: Dict[str, Credential] = {}

    async def store(self, credential: Credential) -> None:
        """Store an encrypted credential."""
        async with self._lock:
            # Serialize and encrypt credential data
            cred_data = credential.to_dict(include_encrypted=True)
            encrypted = self._encryptor.encrypt(json.dumps(cred_data).encode())

            # Write to file
            file_path = self._storage_path / f"{credential.id}.cred"
            with open(file_path, "wb") as f:
                f.write(encrypted.to_bytes())

            self._cache[credential.id] = credential

    async def get(self, credential_id: str) -> Optional[Credential]:
        """Get and decrypt a credential."""
        # Check cache
        if credential_id in self._cache:
            return self._cache[credential_id]

        file_path = self._storage_path / f"{credential_id}.cred"
        if not file_path.exists():
            return None

        with open(file_path, "rb") as f:
            encrypted = EncryptedData.from_bytes(f.read())

        decrypted = self._encryptor.decrypt(encrypted)
        data = json.loads(decrypted)

        credential = Credential(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            credential_type=CredentialType(data["credential_type"]),
            status=CredentialStatus(data["status"]),
            organization_id=data["organization_id"],
            created_by=data.get("created_by"),
            encrypted_value=base64.b64decode(data["encrypted_value"]) if data.get("encrypted_value") else None,
            encryption_key_id=data.get("encryption_key_id"),
            provider=data.get("provider"),
            environment=data.get("environment", "production"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            rotation_interval_days=data.get("rotation_interval_days"),
            last_rotated_at=datetime.fromisoformat(data["last_rotated_at"]) if data.get("last_rotated_at") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
        )

        self._cache[credential_id] = credential
        return credential

    async def list(
        self,
        organization_id: str,
        credential_type: Optional[CredentialType] = None,
        provider: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> List[Credential]:
        """List credentials for organization."""
        credentials = []

        for file_path in self._storage_path.glob("*.cred"):
            cred_id = file_path.stem
            cred = await self.get(cred_id)
            if not cred:
                continue
            if cred.organization_id != organization_id:
                continue
            if credential_type and cred.credential_type != credential_type:
                continue
            if provider and cred.provider != provider:
                continue
            if environment and cred.environment != environment:
                continue
            credentials.append(cred)

        return credentials

    async def delete(self, credential_id: str) -> bool:
        """Delete a credential."""
        async with self._lock:
            file_path = self._storage_path / f"{credential_id}.cred"
            if file_path.exists():
                file_path.unlink()
                self._cache.pop(credential_id, None)
                return True
            return False

    async def update(self, credential: Credential) -> None:
        """Update a credential."""
        await self.store(credential)


class CredentialManager:
    """
    Credential Management Service.

    Provides secure storage, retrieval, and rotation of credentials
    with encryption at rest and access control.
    """

    def __init__(
        self,
        store: CredentialStore,
        encryption_key: bytes,
    ):
        self._store = store
        self._encryptor = DataEncryptor(encryption_key)
        self._access_policies: Dict[str, List[AccessPolicy]] = defaultdict(list)
        self._access_log: List[Dict[str, Any]] = []
        self._rotation_callbacks: List[Callable[[Credential, str, str], None]] = []
        self._logger = structlog.get_logger("credential_manager")
        self._lock = asyncio.Lock()

    def add_rotation_callback(
        self,
        callback: Callable[[Credential, str, str], None],
    ) -> None:
        """Add callback for credential rotation events."""
        self._rotation_callbacks.append(callback)

    async def create(
        self,
        organization_id: str,
        name: str,
        value: str,
        credential_type: CredentialType,
        *,
        description: str = "",
        provider: Optional[str] = None,
        environment: str = "production",
        expires_at: Optional[datetime] = None,
        rotation_interval_days: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> Credential:
        """
        Create a new credential.

        Args:
            organization_id: Organization owning the credential
            name: Name/identifier for the credential
            value: The secret value to store
            credential_type: Type of credential
            description: Optional description
            provider: Optional provider name (e.g., "aws", "twilio")
            environment: Environment (production, staging, development)
            expires_at: Optional expiration time
            rotation_interval_days: Optional rotation interval
            tags: Optional tags
            metadata: Optional metadata
            created_by: User who created the credential

        Returns:
            Created credential
        """
        # Encrypt the value
        encrypted = self._encryptor.encrypt(value.encode())

        credential = Credential(
            name=name,
            description=description,
            credential_type=credential_type,
            organization_id=organization_id,
            created_by=created_by,
            encrypted_value=encrypted.to_bytes(),
            encryption_key_id=encrypted.key_id,
            provider=provider,
            environment=environment,
            tags=tags or [],
            metadata=metadata or {},
            expires_at=expires_at,
            rotation_interval_days=rotation_interval_days,
        )

        await self._store.store(credential)
        self._logger.info(f"Created credential: {credential.id} ({name})")

        return credential

    async def get(
        self,
        credential_id: str,
        *,
        accessor_id: Optional[str] = None,
        accessor_ip: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get a credential's decrypted value.

        Args:
            credential_id: ID of the credential
            accessor_id: ID of user/service accessing the credential
            accessor_ip: IP address of accessor

        Returns:
            Decrypted credential value, or None if not found
        """
        credential = await self._store.get(credential_id)
        if not credential:
            return None

        # Check access policy
        if not await self._check_access(credential, accessor_id, "read"):
            self._logger.warning(
                f"Access denied to credential {credential_id}",
                accessor=accessor_id,
            )
            return None

        # Check status
        if credential.status != CredentialStatus.ACTIVE:
            self._logger.warning(
                f"Credential {credential_id} is not active: {credential.status}",
            )
            return None

        # Check expiration
        if credential.is_expired:
            self._logger.warning(f"Credential {credential_id} is expired")
            credential.status = CredentialStatus.EXPIRED
            await self._store.update(credential)
            return None

        # Decrypt value
        if not credential.encrypted_value:
            return None

        encrypted = EncryptedData.from_bytes(credential.encrypted_value)
        decrypted = self._encryptor.decrypt(encrypted)

        # Update access metadata
        credential.last_accessed_at = datetime.utcnow()
        credential.access_count += 1
        await self._store.update(credential)

        # Log access
        self._log_access(credential_id, accessor_id, accessor_ip, "read")

        return decrypted.decode()

    async def get_metadata(self, credential_id: str) -> Optional[Credential]:
        """Get credential metadata without decrypting value."""
        return await self._store.get(credential_id)

    async def list(
        self,
        organization_id: str,
        *,
        credential_type: Optional[CredentialType] = None,
        provider: Optional[str] = None,
        environment: Optional[str] = None,
        include_expired: bool = False,
    ) -> List[Credential]:
        """List credentials for an organization."""
        credentials = await self._store.list(
            organization_id,
            credential_type=credential_type,
            provider=provider,
            environment=environment,
        )

        if not include_expired:
            credentials = [c for c in credentials if not c.is_expired]

        return credentials

    async def update(
        self,
        credential_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: Optional[CredentialStatus] = None,
        updater_id: Optional[str] = None,
    ) -> Optional[Credential]:
        """Update credential metadata (not the value)."""
        credential = await self._store.get(credential_id)
        if not credential:
            return None

        if name:
            credential.name = name
        if description is not None:
            credential.description = description
        if tags is not None:
            credential.tags = tags
        if metadata is not None:
            credential.metadata = metadata
        if status:
            credential.status = status

        credential.updated_at = datetime.utcnow()
        await self._store.update(credential)

        self._logger.info(f"Updated credential: {credential_id}", updater=updater_id)
        return credential

    async def rotate(
        self,
        credential_id: str,
        new_value: str,
        *,
        rotator_id: Optional[str] = None,
    ) -> Optional[Credential]:
        """
        Rotate a credential to a new value.

        Args:
            credential_id: ID of the credential
            new_value: New secret value
            rotator_id: ID of user/service performing rotation

        Returns:
            Updated credential
        """
        credential = await self._store.get(credential_id)
        if not credential:
            return None

        old_value = None
        if credential.encrypted_value:
            encrypted = EncryptedData.from_bytes(credential.encrypted_value)
            old_value = self._encryptor.decrypt(encrypted).decode()

        # Encrypt new value
        encrypted = self._encryptor.encrypt(new_value.encode())
        credential.encrypted_value = encrypted.to_bytes()
        credential.encryption_key_id = encrypted.key_id
        credential.last_rotated_at = datetime.utcnow()
        credential.updated_at = datetime.utcnow()
        credential.status = CredentialStatus.ACTIVE

        await self._store.update(credential)

        self._logger.info(
            f"Rotated credential: {credential_id}",
            rotator=rotator_id,
        )

        # Call rotation callbacks
        for callback in self._rotation_callbacks:
            try:
                callback(credential, old_value or "", new_value)
            except Exception as e:
                self._logger.error(f"Rotation callback error: {e}")

        return credential

    async def delete(
        self,
        credential_id: str,
        *,
        deleter_id: Optional[str] = None,
    ) -> bool:
        """Delete a credential."""
        result = await self._store.delete(credential_id)
        if result:
            self._logger.info(
                f"Deleted credential: {credential_id}",
                deleter=deleter_id,
            )
        return result

    async def revoke(
        self,
        credential_id: str,
        *,
        reason: str = "",
        revoker_id: Optional[str] = None,
    ) -> Optional[Credential]:
        """Revoke a credential (mark as unusable)."""
        credential = await self._store.get(credential_id)
        if not credential:
            return None

        credential.status = CredentialStatus.REVOKED
        credential.updated_at = datetime.utcnow()
        credential.metadata["revoked_reason"] = reason
        credential.metadata["revoked_by"] = revoker_id
        credential.metadata["revoked_at"] = datetime.utcnow().isoformat()

        await self._store.update(credential)

        self._logger.warning(
            f"Revoked credential: {credential_id}",
            reason=reason,
            revoker=revoker_id,
        )

        return credential

    async def set_access_policy(
        self,
        credential_id: str,
        policy: AccessPolicy,
    ) -> None:
        """Set access policy for a credential."""
        policy.credential_id = credential_id
        self._access_policies[credential_id].append(policy)

    async def get_credentials_needing_rotation(
        self,
        organization_id: str,
    ) -> List[Credential]:
        """Get credentials that need rotation."""
        credentials = await self._store.list(organization_id)
        return [c for c in credentials if c.needs_rotation and c.status == CredentialStatus.ACTIVE]

    async def get_expiring_credentials(
        self,
        organization_id: str,
        within_days: int = 7,
    ) -> List[Credential]:
        """Get credentials expiring soon."""
        credentials = await self._store.list(organization_id)
        cutoff = datetime.utcnow() + timedelta(days=within_days)
        return [
            c for c in credentials
            if c.expires_at and c.expires_at <= cutoff and c.status == CredentialStatus.ACTIVE
        ]

    async def _check_access(
        self,
        credential: Credential,
        accessor_id: Optional[str],
        action: str,
    ) -> bool:
        """Check if accessor has permission."""
        policies = self._access_policies.get(credential.id, [])

        if not policies:
            return True  # No policies = allow all

        for policy in policies:
            # Check time validity
            now = datetime.utcnow()
            if policy.valid_from and now < policy.valid_from:
                continue
            if policy.valid_until and now > policy.valid_until:
                continue

            # Check accessor
            if accessor_id:
                if accessor_id in policy.allowed_users:
                    return self._check_action_permission(policy, action)
                if any(s in policy.allowed_services for s in [accessor_id]):
                    return self._check_action_permission(policy, action)

        return False

    def _check_action_permission(self, policy: AccessPolicy, action: str) -> bool:
        """Check if action is allowed by policy."""
        if action == "read" and policy.can_read:
            return True
        if action == "update" and policy.can_update:
            return True
        if action == "delete" and policy.can_delete:
            return True
        if action == "rotate" and policy.can_rotate:
            return True
        return False

    def _log_access(
        self,
        credential_id: str,
        accessor_id: Optional[str],
        accessor_ip: Optional[str],
        action: str,
    ) -> None:
        """Log credential access."""
        self._access_log.append({
            "credential_id": credential_id,
            "accessor_id": accessor_id,
            "accessor_ip": accessor_ip,
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Keep log bounded
        if len(self._access_log) > 10000:
            self._access_log = self._access_log[-5000:]

    async def get_access_log(
        self,
        credential_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get access log entries."""
        logs = self._access_log
        if credential_id:
            logs = [l for l in logs if l["credential_id"] == credential_id]
        return logs[-limit:]

    async def get_statistics(
        self,
        organization_id: str,
    ) -> Dict[str, Any]:
        """Get credential statistics."""
        credentials = await self._store.list(organization_id)

        by_type: Dict[str, int] = defaultdict(int)
        by_status: Dict[str, int] = defaultdict(int)
        by_environment: Dict[str, int] = defaultdict(int)

        needs_rotation = 0
        expiring_soon = 0
        cutoff = datetime.utcnow() + timedelta(days=7)

        for cred in credentials:
            by_type[cred.credential_type.value] += 1
            by_status[cred.status.value] += 1
            by_environment[cred.environment] += 1

            if cred.needs_rotation:
                needs_rotation += 1
            if cred.expires_at and cred.expires_at <= cutoff:
                expiring_soon += 1

        return {
            "total": len(credentials),
            "by_type": dict(by_type),
            "by_status": dict(by_status),
            "by_environment": dict(by_environment),
            "needs_rotation": needs_rotation,
            "expiring_soon": expiring_soon,
        }


# Convenience function to create credential manager
def create_credential_manager(
    storage_path: Optional[str] = None,
    encryption_key: Optional[bytes] = None,
) -> CredentialManager:
    """Create a credential manager with default settings."""
    key = encryption_key or os.urandom(32)

    if storage_path:
        store = EncryptedFileCredentialStore(storage_path, key)
    else:
        store = InMemoryCredentialStore()

    return CredentialManager(store, key)


__all__ = [
    "CredentialType",
    "CredentialStatus",
    "Credential",
    "CredentialVersion",
    "AccessPolicy",
    "CredentialStore",
    "InMemoryCredentialStore",
    "EncryptedFileCredentialStore",
    "CredentialManager",
    "create_credential_manager",
]
