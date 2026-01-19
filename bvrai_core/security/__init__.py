"""
Security Engine
===============

Comprehensive security infrastructure for the Voice AI platform including
audit logging, encryption, key management, and access control.

Author: Platform Security Team
Version: 2.0.0
"""

from bvrai_core.security.audit import (
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
from bvrai_core.security.encryption import (
    EncryptionService,
    EncryptionAlgorithm,
    KeyType,
    EncryptedData,
    KeyMetadata,
    DataEncryptor,
    FieldEncryptor,
)
from bvrai_core.security.keys import (
    KeyManager,
    KeyStore,
    InMemoryKeyStore,
    EncryptedKeyStore,
    KeyRotationPolicy,
    MasterKey,
    DataKey,
)
from bvrai_core.security.access import (
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
from bvrai_core.security.monitoring import (
    SecurityMonitor,
    SecurityAlert,
    AlertSeverity,
    ThreatDetector,
    AnomalyDetector,
    SecurityMetrics,
)
from bvrai_core.security.credentials import (
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
