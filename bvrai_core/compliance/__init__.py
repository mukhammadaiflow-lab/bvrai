"""
Compliance & Security Module

This module provides comprehensive compliance management for voice agent platforms,
including regulatory framework support, audit logging, consent management,
and security policy enforcement.

Supported Frameworks:
- HIPAA: Healthcare data protection
- PCI-DSS: Payment card data security
- GDPR: EU data protection regulation
- CCPA: California consumer privacy
- SOC2: Service organization controls
- TCPA: Telephone consumer protection

Features:
- Audit logging with tamper-evident storage
- Consent management and tracking
- Data subject request handling (GDPR/CCPA)
- Compliance control assessment
- Security policy management
- Call recording consent enforcement

Example usage:

    from bvrai_core.compliance import (
        ComplianceService,
        ComplianceFramework,
        ConsentType,
        DataSubjectRequestType,
        AuditEventType,
    )

    # Initialize compliance service
    service = ComplianceService()

    # Create security policy
    policy = await service.create_security_policy(
        organization_id="org_123",
        name="Enterprise Security",
        require_mfa=True,
        min_password_length=14,
    )

    # Initialize HIPAA compliance
    controls = await service.initialize_framework(
        organization_id="org_123",
        framework=ComplianceFramework.HIPAA,
    )

    # Record consent
    consent = await service.consent.record_consent(
        organization_id="org_123",
        subject_id="customer_456",
        consent_type=ConsentType.CALL_RECORDING,
        purpose="Quality assurance and training",
        granted=True,
    )

    # Check consent before recording
    has_consent, record = await service.check_call_recording_consent(
        organization_id="org_123",
        subject_id="customer_456",
    )

    # Log audit event
    await service.audit.log(
        organization_id="org_123",
        event_type=AuditEventType.CALL_STARTED,
        actor_id="agent_789",
        resource_type="call",
        resource_id="call_001",
        action="start_call",
    )

    # Handle data subject request
    dsr = await service.dsr.create_request(
        organization_id="org_123",
        request_type=DataSubjectRequestType.ACCESS,
        subject_id="customer_456",
        subject_email="customer@example.com",
    )

    # Get compliance dashboard
    dashboard = await service.get_compliance_dashboard("org_123")
"""

# Base types and enums
from .base import (
    # Framework enums
    ComplianceFramework,
    ComplianceStatus,
    RiskLevel,
    ControlCategory,
    DataClassification,
    # Consent enums
    ConsentType,
    ConsentStatus,
    # Audit enums
    AuditEventType,
    DataSubjectRequestType,
    # Types
    ComplianceControl,
    ConsentRecord,
    DataSubjectRequest,
    AuditEvent,
    SecurityPolicy,
    # Exceptions
    ComplianceError,
    ConsentError,
    AuditError,
    PolicyViolationError,
)

# Services
from .service import (
    AuditLogService,
    ConsentManager,
    DataSubjectRequestHandler,
    ComplianceAssessmentService,
    ComplianceService,
)


__all__ = [
    # Framework enums
    "ComplianceFramework",
    "ComplianceStatus",
    "RiskLevel",
    "ControlCategory",
    "DataClassification",
    # Consent enums
    "ConsentType",
    "ConsentStatus",
    # Audit enums
    "AuditEventType",
    "DataSubjectRequestType",
    # Types
    "ComplianceControl",
    "ConsentRecord",
    "DataSubjectRequest",
    "AuditEvent",
    "SecurityPolicy",
    # Services
    "AuditLogService",
    "ConsentManager",
    "DataSubjectRequestHandler",
    "ComplianceAssessmentService",
    "ComplianceService",
    # Exceptions
    "ComplianceError",
    "ConsentError",
    "AuditError",
    "PolicyViolationError",
]
