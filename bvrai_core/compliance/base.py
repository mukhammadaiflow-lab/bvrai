"""
Compliance & Security Base Types Module

This module defines core types for regulatory compliance, security controls,
audit logging, and data protection across the voice agent platform.
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)


# =============================================================================
# Compliance Framework Enums
# =============================================================================


class ComplianceFramework(str, Enum):
    """Regulatory compliance frameworks."""

    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    GDPR = "gdpr"  # General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    SOC2 = "soc2"  # Service Organization Control 2
    SOX = "sox"  # Sarbanes-Oxley Act
    GLBA = "glba"  # Gramm-Leach-Bliley Act
    FERPA = "ferpa"  # Family Educational Rights and Privacy Act
    TCPA = "tcpa"  # Telephone Consumer Protection Act
    COPPA = "coppa"  # Children's Online Privacy Protection Act
    ISO27001 = "iso27001"  # Information Security Management
    NIST = "nist"  # NIST Cybersecurity Framework
    CUSTOM = "custom"


class ComplianceStatus(str, Enum):
    """Compliance assessment status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    REMEDIATION_REQUIRED = "remediation_required"
    NOT_APPLICABLE = "not_applicable"


class RiskLevel(str, Enum):
    """Risk severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class ControlCategory(str, Enum):
    """Security control categories."""

    ACCESS_CONTROL = "access_control"
    DATA_PROTECTION = "data_protection"
    ENCRYPTION = "encryption"
    AUDIT_LOGGING = "audit_logging"
    INCIDENT_RESPONSE = "incident_response"
    BUSINESS_CONTINUITY = "business_continuity"
    PHYSICAL_SECURITY = "physical_security"
    NETWORK_SECURITY = "network_security"
    APPLICATION_SECURITY = "application_security"
    VENDOR_MANAGEMENT = "vendor_management"
    PRIVACY = "privacy"
    CONSENT_MANAGEMENT = "consent_management"


class DataClassification(str, Enum):
    """Data classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"  # Personally Identifiable Information
    PHI = "phi"  # Protected Health Information
    PCI = "pci"  # Payment Card Information


class ConsentType(str, Enum):
    """Types of consent."""

    CALL_RECORDING = "call_recording"
    DATA_PROCESSING = "data_processing"
    MARKETING = "marketing"
    THIRD_PARTY_SHARING = "third_party_sharing"
    AI_PROCESSING = "ai_processing"
    CROSS_BORDER_TRANSFER = "cross_border_transfer"


class ConsentStatus(str, Enum):
    """Consent status."""

    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    EXPIRED = "expired"


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLED = "mfa_enabled"

    # Authorization
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGE = "permission_change"
    ROLE_CHANGE = "role_change"

    # Data access
    DATA_READ = "data_read"
    DATA_CREATE = "data_create"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"

    # Call operations
    CALL_STARTED = "call_started"
    CALL_ENDED = "call_ended"
    RECORDING_STARTED = "recording_started"
    RECORDING_ACCESSED = "recording_accessed"
    TRANSCRIPTION_CREATED = "transcription_created"

    # Administrative
    CONFIG_CHANGE = "config_change"
    POLICY_CHANGE = "policy_change"
    USER_CREATED = "user_created"
    USER_DEACTIVATED = "user_deactivated"

    # Compliance
    CONSENT_GRANTED = "consent_granted"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    DATA_SUBJECT_REQUEST = "data_subject_request"
    COMPLIANCE_ALERT = "compliance_alert"


class DataSubjectRequestType(str, Enum):
    """Types of data subject requests (GDPR/CCPA)."""

    ACCESS = "access"  # Right to access
    RECTIFICATION = "rectification"  # Right to correct
    ERASURE = "erasure"  # Right to be forgotten
    PORTABILITY = "portability"  # Right to data portability
    RESTRICTION = "restriction"  # Right to restrict processing
    OBJECTION = "objection"  # Right to object
    OPT_OUT_SALE = "opt_out_sale"  # CCPA opt-out of sale


# =============================================================================
# Compliance Control Types
# =============================================================================


@dataclass
class ComplianceControl:
    """A compliance control requirement."""

    id: str
    framework: ComplianceFramework
    control_id: str  # Framework-specific control ID
    name: str
    description: str

    # Classification
    category: ControlCategory
    risk_level: RiskLevel = RiskLevel.MEDIUM

    # Requirements
    requirements: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)

    # Status
    status: ComplianceStatus = ComplianceStatus.UNDER_REVIEW
    implementation_notes: str = ""

    # Assessment
    last_assessed: Optional[datetime] = None
    next_assessment: Optional[datetime] = None
    assessed_by: Optional[str] = None

    # Remediation
    remediation_plan: Optional[str] = None
    remediation_deadline: Optional[datetime] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"ctrl_{uuid.uuid4().hex[:24]}"

    def is_compliant(self) -> bool:
        """Check if control is compliant."""
        return self.status in [ComplianceStatus.COMPLIANT, ComplianceStatus.NOT_APPLICABLE]

    def needs_remediation(self) -> bool:
        """Check if remediation is needed."""
        return self.status in [
            ComplianceStatus.NON_COMPLIANT,
            ComplianceStatus.PARTIALLY_COMPLIANT,
            ComplianceStatus.REMEDIATION_REQUIRED,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "framework": self.framework.value,
            "control_id": self.control_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "risk_level": self.risk_level.value,
            "requirements": self.requirements,
            "evidence_requirements": self.evidence_requirements,
            "status": self.status.value,
            "implementation_notes": self.implementation_notes,
            "last_assessed": self.last_assessed.isoformat() if self.last_assessed else None,
            "next_assessment": self.next_assessment.isoformat() if self.next_assessment else None,
            "assessed_by": self.assessed_by,
            "remediation_plan": self.remediation_plan,
            "remediation_deadline": self.remediation_deadline.isoformat() if self.remediation_deadline else None,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceControl":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            framework=ComplianceFramework(data["framework"]),
            control_id=data["control_id"],
            name=data["name"],
            description=data["description"],
            category=ControlCategory(data["category"]),
            risk_level=RiskLevel(data.get("risk_level", "medium")),
            requirements=data.get("requirements", []),
            evidence_requirements=data.get("evidence_requirements", []),
            status=ComplianceStatus(data.get("status", "under_review")),
            implementation_notes=data.get("implementation_notes", ""),
            last_assessed=datetime.fromisoformat(data["last_assessed"]) if data.get("last_assessed") else None,
            next_assessment=datetime.fromisoformat(data["next_assessment"]) if data.get("next_assessment") else None,
            assessed_by=data.get("assessed_by"),
            remediation_plan=data.get("remediation_plan"),
            remediation_deadline=datetime.fromisoformat(data["remediation_deadline"]) if data.get("remediation_deadline") else None,
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
        )


# =============================================================================
# Consent Management
# =============================================================================


@dataclass
class ConsentRecord:
    """Record of consent given by a data subject."""

    id: str
    organization_id: str

    # Data subject
    subject_id: str  # External identifier
    subject_type: str = "customer"  # customer, caller, user
    subject_email: Optional[str] = None
    subject_phone: Optional[str] = None

    # Consent details
    consent_type: ConsentType = ConsentType.DATA_PROCESSING
    status: ConsentStatus = ConsentStatus.PENDING
    purpose: str = ""
    legal_basis: str = ""  # e.g., "legitimate_interest", "explicit_consent"

    # Scope
    data_categories: List[str] = field(default_factory=list)
    processing_activities: List[str] = field(default_factory=list)
    third_parties: List[str] = field(default_factory=list)

    # Validity
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None

    # Evidence
    consent_method: str = ""  # e.g., "verbal", "written", "electronic"
    consent_evidence: Optional[str] = None  # Recording ID, document ID, etc.
    ip_address: Optional[str] = None

    # Metadata
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"cns_{uuid.uuid4().hex[:24]}"

    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.status != ConsentStatus.GRANTED:
            return False
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        if self.withdrawn_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "subject_id": self.subject_id,
            "subject_type": self.subject_type,
            "subject_email": self.subject_email,
            "subject_phone": self.subject_phone,
            "consent_type": self.consent_type.value,
            "status": self.status.value,
            "purpose": self.purpose,
            "legal_basis": self.legal_basis,
            "data_categories": self.data_categories,
            "processing_activities": self.processing_activities,
            "third_parties": self.third_parties,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "withdrawn_at": self.withdrawn_at.isoformat() if self.withdrawn_at else None,
            "consent_method": self.consent_method,
            "consent_evidence": self.consent_evidence,
            "ip_address": self.ip_address,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsentRecord":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            organization_id=data["organization_id"],
            subject_id=data["subject_id"],
            subject_type=data.get("subject_type", "customer"),
            subject_email=data.get("subject_email"),
            subject_phone=data.get("subject_phone"),
            consent_type=ConsentType(data.get("consent_type", "data_processing")),
            status=ConsentStatus(data.get("status", "pending")),
            purpose=data.get("purpose", ""),
            legal_basis=data.get("legal_basis", ""),
            data_categories=data.get("data_categories", []),
            processing_activities=data.get("processing_activities", []),
            third_parties=data.get("third_parties", []),
            granted_at=datetime.fromisoformat(data["granted_at"]) if data.get("granted_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            withdrawn_at=datetime.fromisoformat(data["withdrawn_at"]) if data.get("withdrawn_at") else None,
            consent_method=data.get("consent_method", ""),
            consent_evidence=data.get("consent_evidence"),
            ip_address=data.get("ip_address"),
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
        )


# =============================================================================
# Data Subject Requests
# =============================================================================


@dataclass
class DataSubjectRequest:
    """A data subject request (GDPR/CCPA)."""

    id: str
    organization_id: str

    # Request details
    request_type: DataSubjectRequestType
    subject_id: str
    subject_email: Optional[str] = None
    subject_phone: Optional[str] = None

    # Request info
    description: str = ""
    verification_status: str = "pending"  # pending, verified, failed

    # Processing
    status: str = "pending"  # pending, in_progress, completed, rejected
    assigned_to: Optional[str] = None
    priority: str = "normal"

    # Timeline
    received_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Response
    response_summary: Optional[str] = None
    data_exported: bool = False
    data_deleted: bool = False

    # Metadata
    source: str = ""  # email, portal, phone
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"dsr_{uuid.uuid4().hex[:24]}"
        if not self.deadline:
            # Default 30-day deadline per GDPR
            self.deadline = self.received_at + timedelta(days=30)

    def is_overdue(self) -> bool:
        """Check if request is overdue."""
        if self.status == "completed":
            return False
        return self.deadline and datetime.utcnow() > self.deadline

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "request_type": self.request_type.value,
            "subject_id": self.subject_id,
            "subject_email": self.subject_email,
            "subject_phone": self.subject_phone,
            "description": self.description,
            "verification_status": self.verification_status,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "priority": self.priority,
            "received_at": self.received_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "response_summary": self.response_summary,
            "data_exported": self.data_exported,
            "data_deleted": self.data_deleted,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSubjectRequest":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            organization_id=data["organization_id"],
            request_type=DataSubjectRequestType(data["request_type"]),
            subject_id=data["subject_id"],
            subject_email=data.get("subject_email"),
            subject_phone=data.get("subject_phone"),
            description=data.get("description", ""),
            verification_status=data.get("verification_status", "pending"),
            status=data.get("status", "pending"),
            assigned_to=data.get("assigned_to"),
            priority=data.get("priority", "normal"),
            received_at=datetime.fromisoformat(data["received_at"]) if "received_at" in data else datetime.utcnow(),
            acknowledged_at=datetime.fromisoformat(data["acknowledged_at"]) if data.get("acknowledged_at") else None,
            deadline=datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            response_summary=data.get("response_summary"),
            data_exported=data.get("data_exported", False),
            data_deleted=data.get("data_deleted", False),
            source=data.get("source", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
        )


# =============================================================================
# Audit Logging
# =============================================================================


@dataclass
class AuditEvent:
    """An audit log event."""

    id: str
    organization_id: str
    event_type: AuditEventType

    # Actor information
    actor_id: Optional[str] = None
    actor_email: Optional[str] = None
    actor_type: str = "user"  # user, system, agent, api

    # Target information
    resource_type: Optional[str] = None  # recording, call, agent, etc.
    resource_id: Optional[str] = None

    # Event details
    action: str = ""
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # Result
    success: bool = True
    error_message: Optional[str] = None

    # Context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    # Data classification
    data_classification: Optional[DataClassification] = None
    contains_pii: bool = False

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"evt_{uuid.uuid4().hex[:24]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "event_type": self.event_type.value,
            "actor_id": self.actor_id,
            "actor_email": self.actor_email,
            "actor_type": self.actor_type,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "description": self.description,
            "details": self.details,
            "success": self.success,
            "error_message": self.error_message,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "data_classification": self.data_classification.value if self.data_classification else None,
            "contains_pii": self.contains_pii,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            organization_id=data["organization_id"],
            event_type=AuditEventType(data["event_type"]),
            actor_id=data.get("actor_id"),
            actor_email=data.get("actor_email"),
            actor_type=data.get("actor_type", "user"),
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            action=data.get("action", ""),
            description=data.get("description", ""),
            details=data.get("details", {}),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            session_id=data.get("session_id"),
            request_id=data.get("request_id"),
            data_classification=DataClassification(data["data_classification"]) if data.get("data_classification") else None,
            contains_pii=data.get("contains_pii", False),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
        )


# =============================================================================
# Security Policy
# =============================================================================


@dataclass
class SecurityPolicy:
    """Organization security policy configuration."""

    id: str
    organization_id: str
    name: str

    # Password policy
    min_password_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    password_expiry_days: int = 90
    password_history_count: int = 5

    # Session policy
    session_timeout_minutes: int = 30
    max_concurrent_sessions: int = 3
    require_mfa: bool = True
    mfa_methods: List[str] = field(default_factory=lambda: ["totp", "sms"])

    # Access policy
    ip_whitelist: List[str] = field(default_factory=list)
    allowed_countries: List[str] = field(default_factory=list)
    require_vpn: bool = False

    # Data policy
    data_retention_days: int = 365
    encryption_required: bool = True
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True

    # Recording policy
    call_recording_enabled: bool = True
    recording_consent_required: bool = True
    recording_notification_required: bool = True

    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"sp_{uuid.uuid4().hex[:24]}"

    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against policy.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if len(password) < self.min_password_length:
            errors.append(f"Password must be at least {self.min_password_length} characters")

        if self.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")

        if self.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")

        if self.require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")

        if self.require_special_chars:
            special = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special for c in password):
                errors.append("Password must contain at least one special character")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "min_password_length": self.min_password_length,
            "require_uppercase": self.require_uppercase,
            "require_lowercase": self.require_lowercase,
            "require_numbers": self.require_numbers,
            "require_special_chars": self.require_special_chars,
            "password_expiry_days": self.password_expiry_days,
            "password_history_count": self.password_history_count,
            "session_timeout_minutes": self.session_timeout_minutes,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "require_mfa": self.require_mfa,
            "mfa_methods": self.mfa_methods,
            "ip_whitelist": self.ip_whitelist,
            "allowed_countries": self.allowed_countries,
            "require_vpn": self.require_vpn,
            "data_retention_days": self.data_retention_days,
            "encryption_required": self.encryption_required,
            "encryption_at_rest": self.encryption_at_rest,
            "encryption_in_transit": self.encryption_in_transit,
            "call_recording_enabled": self.call_recording_enabled,
            "recording_consent_required": self.recording_consent_required,
            "recording_notification_required": self.recording_notification_required,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityPolicy":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            organization_id=data["organization_id"],
            name=data["name"],
            min_password_length=data.get("min_password_length", 12),
            require_uppercase=data.get("require_uppercase", True),
            require_lowercase=data.get("require_lowercase", True),
            require_numbers=data.get("require_numbers", True),
            require_special_chars=data.get("require_special_chars", True),
            password_expiry_days=data.get("password_expiry_days", 90),
            password_history_count=data.get("password_history_count", 5),
            session_timeout_minutes=data.get("session_timeout_minutes", 30),
            max_concurrent_sessions=data.get("max_concurrent_sessions", 3),
            require_mfa=data.get("require_mfa", True),
            mfa_methods=data.get("mfa_methods", ["totp", "sms"]),
            ip_whitelist=data.get("ip_whitelist", []),
            allowed_countries=data.get("allowed_countries", []),
            require_vpn=data.get("require_vpn", False),
            data_retention_days=data.get("data_retention_days", 365),
            encryption_required=data.get("encryption_required", True),
            encryption_at_rest=data.get("encryption_at_rest", True),
            encryption_in_transit=data.get("encryption_in_transit", True),
            call_recording_enabled=data.get("call_recording_enabled", True),
            recording_consent_required=data.get("recording_consent_required", True),
            recording_notification_required=data.get("recording_notification_required", True),
            is_active=data.get("is_active", True),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
        )


# =============================================================================
# Exceptions
# =============================================================================


class ComplianceError(Exception):
    """Base exception for compliance errors."""
    pass


class ConsentError(ComplianceError):
    """Error with consent operations."""
    pass


class AuditError(ComplianceError):
    """Error with audit operations."""
    pass


class PolicyViolationError(ComplianceError):
    """Security policy violation."""
    pass


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "ComplianceFramework",
    "ComplianceStatus",
    "RiskLevel",
    "ControlCategory",
    "DataClassification",
    "ConsentType",
    "ConsentStatus",
    "AuditEventType",
    "DataSubjectRequestType",
    # Types
    "ComplianceControl",
    "ConsentRecord",
    "DataSubjectRequest",
    "AuditEvent",
    "SecurityPolicy",
    # Exceptions
    "ComplianceError",
    "ConsentError",
    "AuditError",
    "PolicyViolationError",
]
