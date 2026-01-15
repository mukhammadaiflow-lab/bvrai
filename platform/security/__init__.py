"""
Security Engine
===============

Comprehensive security infrastructure for the Voice AI platform including
audit logging, encryption, key management, and access control.

Author: Platform Security Team
Version: 2.0.0
"""

from platform.security.audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditCategory,
    AuditFilter,
    AuditStore,
    InMemoryAuditStore,
    FileAuditStore,
)
from platform.security.encryption import (
    EncryptionService,
    EncryptionAlgorithm,
    KeyType,
    EncryptedData,
    KeyMetadata,
    DataEncryptor,
    FieldEncryptor,
)
from platform.security.keys import (
    KeyManager,
    KeyStore,
    InMemoryKeyStore,
    EncryptedKeyStore,
    KeyRotationPolicy,
    MasterKey,
    DataKey,
)
from platform.security.access import (
    AccessControl,
    Permission,
    Role,
    Policy,
    PolicyEffect,
    ResourceMatcher,
    AccessDecision,
    RBACManager,
    ABACManager,
)
from platform.security.monitoring import (
    SecurityMonitor,
    SecurityAlert,
    AlertSeverity,
    ThreatDetector,
    AnomalyDetector,
    SecurityMetrics,
)
from platform.security.credentials import (
    CredentialManager,
    CredentialStore,
    InMemoryCredentialStore,
    EncryptedFileCredentialStore,
    Credential,
    CredentialType,
    CredentialStatus,
    AccessPolicy,
    create_credential_manager,
)

__all__ = [
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "AuditCategory",
    "AuditFilter",
    "AuditStore",
    "InMemoryAuditStore",
    "FileAuditStore",
    # Encryption
    "EncryptionService",
    "EncryptionAlgorithm",
    "KeyType",
    "EncryptedData",
    "KeyMetadata",
    "DataEncryptor",
    "FieldEncryptor",
    # Keys
    "KeyManager",
    "KeyStore",
    "InMemoryKeyStore",
    "EncryptedKeyStore",
    "KeyRotationPolicy",
    "MasterKey",
    "DataKey",
    # Access Control
    "AccessControl",
    "Permission",
    "Role",
    "Policy",
    "PolicyEffect",
    "ResourceMatcher",
    "AccessDecision",
    "RBACManager",
    "ABACManager",
    # Monitoring
    "SecurityMonitor",
    "SecurityAlert",
    "AlertSeverity",
    "ThreatDetector",
    "AnomalyDetector",
    "SecurityMetrics",
    # Credentials
    "CredentialManager",
    "CredentialStore",
    "InMemoryCredentialStore",
    "EncryptedFileCredentialStore",
    "Credential",
    "CredentialType",
    "CredentialStatus",
    "AccessPolicy",
    "create_credential_manager",
]
