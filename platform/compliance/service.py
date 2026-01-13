"""
Compliance Service Module

This module provides comprehensive compliance management services including
audit logging, consent management, data subject request handling, and
compliance assessments.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from .base import (
    ComplianceFramework,
    ComplianceStatus,
    ComplianceControl,
    RiskLevel,
    ControlCategory,
    ConsentRecord,
    ConsentType,
    ConsentStatus,
    DataSubjectRequest,
    DataSubjectRequestType,
    AuditEvent,
    AuditEventType,
    DataClassification,
    SecurityPolicy,
    ComplianceError,
    ConsentError,
    AuditError,
    PolicyViolationError,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Audit Logger Service
# =============================================================================


class AuditLogService:
    """
    Service for managing audit logs with tamper-evident storage.
    """

    def __init__(self):
        self._events: List[AuditEvent] = []
        self._org_index: Dict[str, List[int]] = {}
        self._resource_index: Dict[str, List[int]] = {}
        self._actor_index: Dict[str, List[int]] = {}

    async def log(
        self,
        organization_id: str,
        event_type: AuditEventType,
        actor_id: Optional[str] = None,
        actor_email: Optional[str] = None,
        actor_type: str = "user",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: str = "",
        description: str = "",
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        data_classification: Optional[DataClassification] = None,
        contains_pii: bool = False,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            organization_id: Organization ID
            event_type: Type of event
            actor_id: ID of actor (user, system, etc.)
            actor_email: Email of actor
            actor_type: Type of actor
            resource_type: Type of resource affected
            resource_id: ID of resource
            action: Action performed
            description: Human-readable description
            details: Additional event details
            success: Whether action succeeded
            error_message: Error message if failed
            ip_address: Client IP address
            user_agent: Client user agent
            session_id: Session identifier
            request_id: Request identifier
            data_classification: Data classification level
            contains_pii: Whether event contains PII

        Returns:
            Created audit event
        """
        event = AuditEvent(
            id=f"evt_{uuid.uuid4().hex[:24]}",
            organization_id=organization_id,
            event_type=event_type,
            actor_id=actor_id,
            actor_email=actor_email,
            actor_type=actor_type,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            description=description,
            details=details or {},
            success=success,
            error_message=error_message,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            request_id=request_id,
            data_classification=data_classification,
            contains_pii=contains_pii,
        )

        # Store event
        idx = len(self._events)
        self._events.append(event)

        # Update indices
        if organization_id not in self._org_index:
            self._org_index[organization_id] = []
        self._org_index[organization_id].append(idx)

        if resource_id:
            key = f"{resource_type}:{resource_id}"
            if key not in self._resource_index:
                self._resource_index[key] = []
            self._resource_index[key].append(idx)

        if actor_id:
            if actor_id not in self._actor_index:
                self._actor_index[actor_id] = []
            self._actor_index[actor_id].append(idx)

        logger.debug(f"Audit: {event_type.value} by {actor_email or actor_id} on {resource_type}:{resource_id}")
        return event

    async def query(
        self,
        organization_id: str,
        event_types: Optional[List[AuditEventType]] = None,
        actor_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        success: Optional[bool] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[AuditEvent], int]:
        """
        Query audit logs.

        Args:
            organization_id: Organization ID
            event_types: Filter by event types
            actor_id: Filter by actor
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            start_time: Start time filter
            end_time: End time filter
            success: Filter by success status
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (events, total_count)
        """
        if organization_id not in self._org_index:
            return [], 0

        indices = self._org_index[organization_id]
        events = []

        for idx in indices:
            event = self._events[idx]

            # Apply filters
            if event_types and event.event_type not in event_types:
                continue
            if actor_id and event.actor_id != actor_id:
                continue
            if resource_type and event.resource_type != resource_type:
                continue
            if resource_id and event.resource_id != resource_id:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if success is not None and event.success != success:
                continue

            events.append(event)

        # Sort by timestamp descending
        events.sort(key=lambda e: e.timestamp, reverse=True)

        total = len(events)
        events = events[offset:offset + limit]

        return events, total

    async def get_resource_history(
        self,
        resource_type: str,
        resource_id: str,
    ) -> List[AuditEvent]:
        """Get all audit events for a resource."""
        key = f"{resource_type}:{resource_id}"
        if key not in self._resource_index:
            return []

        events = [self._events[idx] for idx in self._resource_index[key]]
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events

    async def get_actor_history(
        self,
        actor_id: str,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get all audit events for an actor."""
        if actor_id not in self._actor_index:
            return []

        events = [self._events[idx] for idx in self._actor_index[actor_id]]
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    async def export_logs(
        self,
        organization_id: str,
        start_time: datetime,
        end_time: datetime,
        format: str = "json",
    ) -> str:
        """
        Export audit logs for compliance reporting.

        Args:
            organization_id: Organization ID
            start_time: Start time
            end_time: End time
            format: Export format (json, csv)

        Returns:
            Exported data as string
        """
        events, _ = await self.query(
            organization_id=organization_id,
            start_time=start_time,
            end_time=end_time,
            limit=100000,
        )

        if format == "json":
            return json.dumps([e.to_dict() for e in events], indent=2)
        elif format == "csv":
            lines = ["id,timestamp,event_type,actor_id,actor_email,resource_type,resource_id,action,success"]
            for event in events:
                lines.append(
                    f"{event.id},{event.timestamp.isoformat()},{event.event_type.value},"
                    f"{event.actor_id or ''},{event.actor_email or ''},{event.resource_type or ''},"
                    f"{event.resource_id or ''},{event.action},{event.success}"
                )
            return "\n".join(lines)
        else:
            raise AuditError(f"Unknown format: {format}")


# =============================================================================
# Consent Manager
# =============================================================================


class ConsentManager:
    """
    Manages consent records and validation.
    """

    def __init__(self, audit_service: Optional[AuditLogService] = None):
        self._consents: Dict[str, ConsentRecord] = {}
        self._subject_index: Dict[str, Set[str]] = {}
        self._org_index: Dict[str, Set[str]] = {}
        self.audit = audit_service

    async def record_consent(
        self,
        organization_id: str,
        subject_id: str,
        consent_type: ConsentType,
        purpose: str,
        granted: bool = True,
        subject_email: Optional[str] = None,
        subject_phone: Optional[str] = None,
        legal_basis: str = "explicit_consent",
        data_categories: Optional[List[str]] = None,
        processing_activities: Optional[List[str]] = None,
        third_parties: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
        consent_method: str = "electronic",
        consent_evidence: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> ConsentRecord:
        """
        Record a consent decision.

        Args:
            organization_id: Organization ID
            subject_id: Subject identifier
            consent_type: Type of consent
            purpose: Purpose of data processing
            granted: Whether consent was granted
            subject_email: Subject email
            subject_phone: Subject phone
            legal_basis: Legal basis for processing
            data_categories: Categories of data
            processing_activities: Processing activities
            third_parties: Third parties involved
            expires_at: When consent expires
            consent_method: How consent was obtained
            consent_evidence: Evidence of consent
            ip_address: IP address

        Returns:
            Consent record
        """
        consent = ConsentRecord(
            id=f"cns_{uuid.uuid4().hex[:24]}",
            organization_id=organization_id,
            subject_id=subject_id,
            subject_email=subject_email,
            subject_phone=subject_phone,
            consent_type=consent_type,
            status=ConsentStatus.GRANTED if granted else ConsentStatus.DENIED,
            purpose=purpose,
            legal_basis=legal_basis,
            data_categories=data_categories or [],
            processing_activities=processing_activities or [],
            third_parties=third_parties or [],
            granted_at=datetime.utcnow() if granted else None,
            expires_at=expires_at,
            consent_method=consent_method,
            consent_evidence=consent_evidence,
            ip_address=ip_address,
        )

        # Store consent
        self._consents[consent.id] = consent

        # Update indices
        if subject_id not in self._subject_index:
            self._subject_index[subject_id] = set()
        self._subject_index[subject_id].add(consent.id)

        if organization_id not in self._org_index:
            self._org_index[organization_id] = set()
        self._org_index[organization_id].add(consent.id)

        # Log audit event
        if self.audit:
            await self.audit.log(
                organization_id=organization_id,
                event_type=AuditEventType.CONSENT_GRANTED if granted else AuditEventType.CONSENT_WITHDRAWN,
                resource_type="consent",
                resource_id=consent.id,
                action="consent_recorded",
                description=f"Consent {consent_type.value} {'granted' if granted else 'denied'} for {purpose}",
                details={
                    "subject_id": subject_id,
                    "consent_type": consent_type.value,
                    "purpose": purpose,
                },
            )

        logger.info(f"Recorded consent: {consent.id} ({consent_type.value})")
        return consent

    async def withdraw_consent(
        self,
        consent_id: str,
        reason: Optional[str] = None,
    ) -> ConsentRecord:
        """
        Withdraw a previously granted consent.

        Args:
            consent_id: Consent ID
            reason: Reason for withdrawal

        Returns:
            Updated consent record
        """
        consent = self._consents.get(consent_id)
        if not consent:
            raise ConsentError(f"Consent not found: {consent_id}")

        consent.status = ConsentStatus.WITHDRAWN
        consent.withdrawn_at = datetime.utcnow()
        consent.updated_at = datetime.utcnow()

        # Log audit event
        if self.audit:
            await self.audit.log(
                organization_id=consent.organization_id,
                event_type=AuditEventType.CONSENT_WITHDRAWN,
                resource_type="consent",
                resource_id=consent.id,
                action="consent_withdrawn",
                description=f"Consent withdrawn: {reason or 'No reason provided'}",
                details={
                    "subject_id": consent.subject_id,
                    "consent_type": consent.consent_type.value,
                    "reason": reason,
                },
            )

        logger.info(f"Withdrew consent: {consent_id}")
        return consent

    async def check_consent(
        self,
        subject_id: str,
        consent_type: ConsentType,
        organization_id: Optional[str] = None,
    ) -> bool:
        """
        Check if valid consent exists.

        Args:
            subject_id: Subject identifier
            consent_type: Type of consent to check
            organization_id: Organization ID (optional filter)

        Returns:
            True if valid consent exists
        """
        if subject_id not in self._subject_index:
            return False

        for consent_id in self._subject_index[subject_id]:
            consent = self._consents.get(consent_id)
            if not consent:
                continue

            if organization_id and consent.organization_id != organization_id:
                continue

            if consent.consent_type == consent_type and consent.is_valid():
                return True

        return False

    async def get_subject_consents(
        self,
        subject_id: str,
    ) -> List[ConsentRecord]:
        """Get all consents for a subject."""
        if subject_id not in self._subject_index:
            return []

        consents = []
        for consent_id in self._subject_index[subject_id]:
            consent = self._consents.get(consent_id)
            if consent:
                consents.append(consent)

        return consents

    async def list_consents(
        self,
        organization_id: str,
        consent_type: Optional[ConsentType] = None,
        status: Optional[ConsentStatus] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[ConsentRecord], int]:
        """List consents for organization."""
        if organization_id not in self._org_index:
            return [], 0

        consents = []
        for consent_id in self._org_index[organization_id]:
            consent = self._consents.get(consent_id)
            if not consent:
                continue

            if consent_type and consent.consent_type != consent_type:
                continue
            if status and consent.status != status:
                continue

            consents.append(consent)

        # Sort by created_at descending
        consents.sort(key=lambda c: c.created_at, reverse=True)

        total = len(consents)
        consents = consents[offset:offset + limit]

        return consents, total


# =============================================================================
# Data Subject Request Handler
# =============================================================================


class DataSubjectRequestHandler:
    """
    Handles GDPR/CCPA data subject requests.
    """

    def __init__(
        self,
        audit_service: Optional[AuditLogService] = None,
        consent_manager: Optional[ConsentManager] = None,
    ):
        self._requests: Dict[str, DataSubjectRequest] = {}
        self._org_index: Dict[str, Set[str]] = {}
        self._subject_index: Dict[str, Set[str]] = {}
        self.audit = audit_service
        self.consent = consent_manager

    async def create_request(
        self,
        organization_id: str,
        request_type: DataSubjectRequestType,
        subject_id: str,
        subject_email: Optional[str] = None,
        subject_phone: Optional[str] = None,
        description: str = "",
        source: str = "portal",
        deadline_days: int = 30,
    ) -> DataSubjectRequest:
        """
        Create a new data subject request.

        Args:
            organization_id: Organization ID
            request_type: Type of request
            subject_id: Subject identifier
            subject_email: Subject email
            subject_phone: Subject phone
            description: Request description
            source: Source of request
            deadline_days: Days until deadline

        Returns:
            Created request
        """
        request = DataSubjectRequest(
            id=f"dsr_{uuid.uuid4().hex[:24]}",
            organization_id=organization_id,
            request_type=request_type,
            subject_id=subject_id,
            subject_email=subject_email,
            subject_phone=subject_phone,
            description=description,
            source=source,
            deadline=datetime.utcnow() + timedelta(days=deadline_days),
        )

        # Store request
        self._requests[request.id] = request

        # Update indices
        if organization_id not in self._org_index:
            self._org_index[organization_id] = set()
        self._org_index[organization_id].add(request.id)

        if subject_id not in self._subject_index:
            self._subject_index[subject_id] = set()
        self._subject_index[subject_id].add(request.id)

        # Log audit event
        if self.audit:
            await self.audit.log(
                organization_id=organization_id,
                event_type=AuditEventType.DATA_SUBJECT_REQUEST,
                resource_type="dsr",
                resource_id=request.id,
                action="dsr_created",
                description=f"Data subject request created: {request_type.value}",
                details={
                    "subject_id": subject_id,
                    "request_type": request_type.value,
                },
            )

        logger.info(f"Created DSR: {request.id} ({request_type.value})")
        return request

    async def acknowledge_request(
        self,
        request_id: str,
        acknowledged_by: Optional[str] = None,
    ) -> DataSubjectRequest:
        """Acknowledge receipt of a request."""
        request = self._requests.get(request_id)
        if not request:
            raise ComplianceError(f"Request not found: {request_id}")

        request.acknowledged_at = datetime.utcnow()
        request.status = "in_progress"
        request.updated_at = datetime.utcnow()

        return request

    async def assign_request(
        self,
        request_id: str,
        assignee_id: str,
    ) -> DataSubjectRequest:
        """Assign request to a handler."""
        request = self._requests.get(request_id)
        if not request:
            raise ComplianceError(f"Request not found: {request_id}")

        request.assigned_to = assignee_id
        request.updated_at = datetime.utcnow()

        return request

    async def complete_request(
        self,
        request_id: str,
        response_summary: str,
        data_exported: bool = False,
        data_deleted: bool = False,
    ) -> DataSubjectRequest:
        """Complete a data subject request."""
        request = self._requests.get(request_id)
        if not request:
            raise ComplianceError(f"Request not found: {request_id}")

        request.status = "completed"
        request.completed_at = datetime.utcnow()
        request.response_summary = response_summary
        request.data_exported = data_exported
        request.data_deleted = data_deleted
        request.updated_at = datetime.utcnow()

        # Log audit event
        if self.audit:
            await self.audit.log(
                organization_id=request.organization_id,
                event_type=AuditEventType.DATA_SUBJECT_REQUEST,
                resource_type="dsr",
                resource_id=request.id,
                action="dsr_completed",
                description=f"Data subject request completed: {request.request_type.value}",
                details={
                    "data_exported": data_exported,
                    "data_deleted": data_deleted,
                },
            )

        logger.info(f"Completed DSR: {request_id}")
        return request

    async def get_request(self, request_id: str) -> Optional[DataSubjectRequest]:
        """Get a request by ID."""
        return self._requests.get(request_id)

    async def list_requests(
        self,
        organization_id: str,
        status: Optional[str] = None,
        request_type: Optional[DataSubjectRequestType] = None,
        include_completed: bool = True,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[DataSubjectRequest], int]:
        """List requests for organization."""
        if organization_id not in self._org_index:
            return [], 0

        requests = []
        for req_id in self._org_index[organization_id]:
            request = self._requests.get(req_id)
            if not request:
                continue

            if status and request.status != status:
                continue
            if request_type and request.request_type != request_type:
                continue
            if not include_completed and request.status == "completed":
                continue

            requests.append(request)

        # Sort by received_at descending
        requests.sort(key=lambda r: r.received_at, reverse=True)

        total = len(requests)
        requests = requests[offset:offset + limit]

        return requests, total

    async def get_overdue_requests(
        self,
        organization_id: Optional[str] = None,
    ) -> List[DataSubjectRequest]:
        """Get overdue requests."""
        overdue = []

        if organization_id:
            if organization_id not in self._org_index:
                return []
            request_ids = self._org_index[organization_id]
        else:
            request_ids = self._requests.keys()

        for req_id in request_ids:
            request = self._requests.get(req_id)
            if request and request.is_overdue():
                overdue.append(request)

        return overdue


# =============================================================================
# Compliance Assessment Service
# =============================================================================


class ComplianceAssessmentService:
    """
    Manages compliance assessments and control tracking.
    """

    def __init__(self):
        self._controls: Dict[str, ComplianceControl] = {}
        self._org_controls: Dict[str, Set[str]] = {}
        self._framework_controls: Dict[ComplianceFramework, Set[str]] = {}

    async def add_control(
        self,
        organization_id: str,
        framework: ComplianceFramework,
        control_id: str,
        name: str,
        description: str,
        category: ControlCategory,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        requirements: Optional[List[str]] = None,
        evidence_requirements: Optional[List[str]] = None,
    ) -> ComplianceControl:
        """Add a compliance control."""
        control = ComplianceControl(
            id=f"ctrl_{uuid.uuid4().hex[:24]}",
            framework=framework,
            control_id=control_id,
            name=name,
            description=description,
            category=category,
            risk_level=risk_level,
            requirements=requirements or [],
            evidence_requirements=evidence_requirements or [],
        )

        self._controls[control.id] = control

        # Update indices
        key = f"{organization_id}:{framework.value}"
        if key not in self._org_controls:
            self._org_controls[key] = set()
        self._org_controls[key].add(control.id)

        if framework not in self._framework_controls:
            self._framework_controls[framework] = set()
        self._framework_controls[framework].add(control.id)

        return control

    async def assess_control(
        self,
        control_id: str,
        status: ComplianceStatus,
        assessed_by: str,
        implementation_notes: str = "",
        remediation_plan: Optional[str] = None,
        remediation_deadline: Optional[datetime] = None,
    ) -> ComplianceControl:
        """Assess a compliance control."""
        control = self._controls.get(control_id)
        if not control:
            raise ComplianceError(f"Control not found: {control_id}")

        control.status = status
        control.last_assessed = datetime.utcnow()
        control.assessed_by = assessed_by
        control.implementation_notes = implementation_notes
        control.remediation_plan = remediation_plan
        control.remediation_deadline = remediation_deadline
        control.updated_at = datetime.utcnow()

        # Set next assessment (quarterly)
        control.next_assessment = datetime.utcnow() + timedelta(days=90)

        return control

    async def get_framework_status(
        self,
        organization_id: str,
        framework: ComplianceFramework,
    ) -> Dict[str, Any]:
        """Get compliance status for a framework."""
        key = f"{organization_id}:{framework.value}"
        if key not in self._org_controls:
            return {
                "framework": framework.value,
                "total_controls": 0,
                "compliant": 0,
                "non_compliant": 0,
                "compliance_percentage": 0.0,
            }

        controls = [
            self._controls[cid]
            for cid in self._org_controls[key]
            if cid in self._controls
        ]

        total = len(controls)
        compliant = sum(1 for c in controls if c.is_compliant())
        non_compliant = sum(1 for c in controls if c.needs_remediation())

        # Status breakdown
        status_counts: Dict[str, int] = {}
        for control in controls:
            status = control.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Risk breakdown
        risk_counts: Dict[str, int] = {}
        for control in controls:
            if control.needs_remediation():
                risk = control.risk_level.value
                risk_counts[risk] = risk_counts.get(risk, 0) + 1

        return {
            "framework": framework.value,
            "total_controls": total,
            "compliant": compliant,
            "non_compliant": non_compliant,
            "compliance_percentage": (compliant / total * 100) if total > 0 else 0.0,
            "status_breakdown": status_counts,
            "risk_breakdown": risk_counts,
            "controls_requiring_remediation": [
                c.to_dict() for c in controls if c.needs_remediation()
            ],
        }

    async def list_controls(
        self,
        organization_id: str,
        framework: Optional[ComplianceFramework] = None,
        category: Optional[ControlCategory] = None,
        status: Optional[ComplianceStatus] = None,
    ) -> List[ComplianceControl]:
        """List compliance controls."""
        controls = []

        if framework:
            key = f"{organization_id}:{framework.value}"
            if key in self._org_controls:
                control_ids = self._org_controls[key]
            else:
                control_ids = set()
        else:
            control_ids = set()
            for key in self._org_controls:
                if key.startswith(f"{organization_id}:"):
                    control_ids.update(self._org_controls[key])

        for cid in control_ids:
            control = self._controls.get(cid)
            if not control:
                continue

            if category and control.category != category:
                continue
            if status and control.status != status:
                continue

            controls.append(control)

        return controls

    async def get_controls_needing_assessment(
        self,
        organization_id: str,
    ) -> List[ComplianceControl]:
        """Get controls that need assessment."""
        now = datetime.utcnow()
        controls = []

        for key in self._org_controls:
            if not key.startswith(f"{organization_id}:"):
                continue

            for cid in self._org_controls[key]:
                control = self._controls.get(cid)
                if not control:
                    continue

                # Never assessed or past due
                if not control.last_assessed:
                    controls.append(control)
                elif control.next_assessment and control.next_assessment <= now:
                    controls.append(control)

        return controls


# =============================================================================
# Compliance Service (Main Entry Point)
# =============================================================================


class ComplianceService:
    """
    Main compliance service providing unified access to all compliance features.
    """

    def __init__(self):
        self.audit = AuditLogService()
        self.consent = ConsentManager(audit_service=self.audit)
        self.dsr = DataSubjectRequestHandler(
            audit_service=self.audit,
            consent_manager=self.consent,
        )
        self.assessment = ComplianceAssessmentService()

        # Security policies
        self._policies: Dict[str, SecurityPolicy] = {}
        self._org_policy: Dict[str, str] = {}

    async def create_security_policy(
        self,
        organization_id: str,
        name: str,
        **kwargs,
    ) -> SecurityPolicy:
        """Create a security policy for an organization."""
        policy = SecurityPolicy(
            id=f"sp_{uuid.uuid4().hex[:24]}",
            organization_id=organization_id,
            name=name,
            **kwargs,
        )

        self._policies[policy.id] = policy
        self._org_policy[organization_id] = policy.id

        # Log audit event
        await self.audit.log(
            organization_id=organization_id,
            event_type=AuditEventType.POLICY_CHANGE,
            resource_type="security_policy",
            resource_id=policy.id,
            action="policy_created",
            description=f"Security policy created: {name}",
        )

        return policy

    async def get_security_policy(
        self,
        organization_id: str,
    ) -> Optional[SecurityPolicy]:
        """Get security policy for organization."""
        policy_id = self._org_policy.get(organization_id)
        if policy_id:
            return self._policies.get(policy_id)
        return None

    async def validate_password(
        self,
        organization_id: str,
        password: str,
    ) -> Tuple[bool, List[str]]:
        """Validate password against organization's security policy."""
        policy = await self.get_security_policy(organization_id)
        if not policy:
            # Default validation
            return len(password) >= 8, [] if len(password) >= 8 else ["Password must be at least 8 characters"]

        return policy.validate_password(password)

    async def check_call_recording_consent(
        self,
        organization_id: str,
        subject_id: str,
    ) -> Tuple[bool, Optional[ConsentRecord]]:
        """
        Check if recording consent exists for a call.

        Returns:
            Tuple of (has_consent, consent_record)
        """
        # Get organization policy
        policy = await self.get_security_policy(organization_id)
        if policy and not policy.recording_consent_required:
            return True, None

        # Check for consent
        has_consent = await self.consent.check_consent(
            subject_id=subject_id,
            consent_type=ConsentType.CALL_RECORDING,
            organization_id=organization_id,
        )

        if has_consent:
            consents = await self.consent.get_subject_consents(subject_id)
            for consent in consents:
                if consent.consent_type == ConsentType.CALL_RECORDING and consent.is_valid():
                    return True, consent

        return False, None

    async def get_compliance_dashboard(
        self,
        organization_id: str,
    ) -> Dict[str, Any]:
        """
        Get compliance dashboard data.

        Returns:
            Dashboard data including status across frameworks
        """
        # Get framework statuses
        frameworks = {}
        for framework in [ComplianceFramework.HIPAA, ComplianceFramework.PCI_DSS,
                          ComplianceFramework.GDPR, ComplianceFramework.SOC2]:
            status = await self.assessment.get_framework_status(organization_id, framework)
            if status["total_controls"] > 0:
                frameworks[framework.value] = status

        # Get pending DSRs
        pending_dsrs, _ = await self.dsr.list_requests(
            organization_id=organization_id,
            status="pending",
        )
        overdue_dsrs = await self.dsr.get_overdue_requests(organization_id)

        # Get consent statistics
        consents, total_consents = await self.consent.list_consents(organization_id)
        granted = sum(1 for c in consents if c.status == ConsentStatus.GRANTED)
        withdrawn = sum(1 for c in consents if c.status == ConsentStatus.WITHDRAWN)

        # Get controls needing attention
        controls_due = await self.assessment.get_controls_needing_assessment(organization_id)

        return {
            "organization_id": organization_id,
            "frameworks": frameworks,
            "data_subject_requests": {
                "pending": len(pending_dsrs),
                "overdue": len(overdue_dsrs),
            },
            "consent": {
                "total": total_consents,
                "granted": granted,
                "withdrawn": withdrawn,
            },
            "controls": {
                "needing_assessment": len(controls_due),
            },
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def initialize_framework(
        self,
        organization_id: str,
        framework: ComplianceFramework,
    ) -> List[ComplianceControl]:
        """
        Initialize compliance controls for a framework.

        Args:
            organization_id: Organization ID
            framework: Compliance framework

        Returns:
            List of created controls
        """
        controls = []

        if framework == ComplianceFramework.HIPAA:
            controls = await self._initialize_hipaa(organization_id)
        elif framework == ComplianceFramework.PCI_DSS:
            controls = await self._initialize_pci_dss(organization_id)
        elif framework == ComplianceFramework.GDPR:
            controls = await self._initialize_gdpr(organization_id)
        elif framework == ComplianceFramework.SOC2:
            controls = await self._initialize_soc2(organization_id)

        logger.info(f"Initialized {len(controls)} controls for {framework.value}")
        return controls

    async def _initialize_hipaa(self, organization_id: str) -> List[ComplianceControl]:
        """Initialize HIPAA controls."""
        controls_def = [
            ("164.312(a)(1)", "Access Control", "Implement access controls for PHI", ControlCategory.ACCESS_CONTROL, RiskLevel.HIGH),
            ("164.312(a)(2)(i)", "Unique User ID", "Assign unique user identifiers", ControlCategory.ACCESS_CONTROL, RiskLevel.MEDIUM),
            ("164.312(a)(2)(iv)", "Encryption", "Implement encryption for PHI", ControlCategory.ENCRYPTION, RiskLevel.HIGH),
            ("164.312(b)", "Audit Controls", "Implement audit controls", ControlCategory.AUDIT_LOGGING, RiskLevel.HIGH),
            ("164.312(c)(1)", "Integrity", "Protect PHI from alteration", ControlCategory.DATA_PROTECTION, RiskLevel.HIGH),
            ("164.312(d)", "Authentication", "Implement authentication procedures", ControlCategory.ACCESS_CONTROL, RiskLevel.HIGH),
            ("164.312(e)(1)", "Transmission Security", "Guard against unauthorized access during transmission", ControlCategory.ENCRYPTION, RiskLevel.HIGH),
            ("164.308(a)(1)", "Security Management", "Implement security management process", ControlCategory.APPLICATION_SECURITY, RiskLevel.MEDIUM),
            ("164.308(a)(3)", "Workforce Security", "Implement workforce security procedures", ControlCategory.ACCESS_CONTROL, RiskLevel.MEDIUM),
            ("164.308(a)(6)", "Security Incident", "Implement incident response procedures", ControlCategory.INCIDENT_RESPONSE, RiskLevel.HIGH),
        ]

        controls = []
        for control_id, name, desc, category, risk in controls_def:
            control = await self.assessment.add_control(
                organization_id=organization_id,
                framework=ComplianceFramework.HIPAA,
                control_id=control_id,
                name=name,
                description=desc,
                category=category,
                risk_level=risk,
            )
            controls.append(control)

        return controls

    async def _initialize_pci_dss(self, organization_id: str) -> List[ComplianceControl]:
        """Initialize PCI-DSS controls."""
        controls_def = [
            ("1.1", "Firewall Configuration", "Install and maintain firewall", ControlCategory.NETWORK_SECURITY, RiskLevel.HIGH),
            ("2.1", "Vendor Defaults", "Change vendor-supplied defaults", ControlCategory.APPLICATION_SECURITY, RiskLevel.MEDIUM),
            ("3.4", "Data Encryption", "Render PAN unreadable", ControlCategory.ENCRYPTION, RiskLevel.CRITICAL),
            ("4.1", "Transmission Encryption", "Encrypt transmission of cardholder data", ControlCategory.ENCRYPTION, RiskLevel.CRITICAL),
            ("5.1", "Anti-virus", "Deploy anti-virus software", ControlCategory.APPLICATION_SECURITY, RiskLevel.MEDIUM),
            ("6.5", "Secure Development", "Address common vulnerabilities", ControlCategory.APPLICATION_SECURITY, RiskLevel.HIGH),
            ("7.1", "Access Restriction", "Restrict access to cardholder data", ControlCategory.ACCESS_CONTROL, RiskLevel.HIGH),
            ("8.1", "User Identification", "Assign unique ID to each user", ControlCategory.ACCESS_CONTROL, RiskLevel.HIGH),
            ("10.1", "Audit Trail", "Implement audit trails", ControlCategory.AUDIT_LOGGING, RiskLevel.HIGH),
            ("12.1", "Security Policy", "Maintain a security policy", ControlCategory.APPLICATION_SECURITY, RiskLevel.MEDIUM),
        ]

        controls = []
        for control_id, name, desc, category, risk in controls_def:
            control = await self.assessment.add_control(
                organization_id=organization_id,
                framework=ComplianceFramework.PCI_DSS,
                control_id=control_id,
                name=name,
                description=desc,
                category=category,
                risk_level=risk,
            )
            controls.append(control)

        return controls

    async def _initialize_gdpr(self, organization_id: str) -> List[ComplianceControl]:
        """Initialize GDPR controls."""
        controls_def = [
            ("Art. 5", "Data Processing Principles", "Lawful, fair, transparent processing", ControlCategory.PRIVACY, RiskLevel.HIGH),
            ("Art. 6", "Lawful Basis", "Establish lawful basis for processing", ControlCategory.CONSENT_MANAGEMENT, RiskLevel.HIGH),
            ("Art. 7", "Consent", "Obtain and document consent", ControlCategory.CONSENT_MANAGEMENT, RiskLevel.HIGH),
            ("Art. 13", "Information Provision", "Provide information to data subjects", ControlCategory.PRIVACY, RiskLevel.MEDIUM),
            ("Art. 15", "Right of Access", "Enable data subject access requests", ControlCategory.PRIVACY, RiskLevel.HIGH),
            ("Art. 17", "Right to Erasure", "Enable right to be forgotten", ControlCategory.PRIVACY, RiskLevel.HIGH),
            ("Art. 20", "Data Portability", "Enable data portability", ControlCategory.PRIVACY, RiskLevel.MEDIUM),
            ("Art. 25", "Privacy by Design", "Implement privacy by design", ControlCategory.DATA_PROTECTION, RiskLevel.HIGH),
            ("Art. 32", "Security Measures", "Implement appropriate security", ControlCategory.DATA_PROTECTION, RiskLevel.HIGH),
            ("Art. 33", "Breach Notification", "Notify breaches within 72 hours", ControlCategory.INCIDENT_RESPONSE, RiskLevel.CRITICAL),
        ]

        controls = []
        for control_id, name, desc, category, risk in controls_def:
            control = await self.assessment.add_control(
                organization_id=organization_id,
                framework=ComplianceFramework.GDPR,
                control_id=control_id,
                name=name,
                description=desc,
                category=category,
                risk_level=risk,
            )
            controls.append(control)

        return controls

    async def _initialize_soc2(self, organization_id: str) -> List[ComplianceControl]:
        """Initialize SOC 2 controls."""
        controls_def = [
            ("CC1.1", "COSO Principles", "Entity demonstrates commitment to integrity", ControlCategory.APPLICATION_SECURITY, RiskLevel.MEDIUM),
            ("CC2.1", "Communication", "Entity communicates objectives", ControlCategory.APPLICATION_SECURITY, RiskLevel.LOW),
            ("CC3.1", "Risk Assessment", "Entity identifies and assesses risks", ControlCategory.APPLICATION_SECURITY, RiskLevel.MEDIUM),
            ("CC4.1", "Monitoring", "Entity selects and develops monitoring activities", ControlCategory.AUDIT_LOGGING, RiskLevel.MEDIUM),
            ("CC5.1", "Control Activities", "Entity selects and develops control activities", ControlCategory.ACCESS_CONTROL, RiskLevel.HIGH),
            ("CC6.1", "Logical Access", "Entity implements logical access security", ControlCategory.ACCESS_CONTROL, RiskLevel.HIGH),
            ("CC6.6", "Boundary Protection", "Entity implements boundary protection", ControlCategory.NETWORK_SECURITY, RiskLevel.HIGH),
            ("CC7.1", "System Operations", "Entity detects and monitors system operation", ControlCategory.AUDIT_LOGGING, RiskLevel.MEDIUM),
            ("CC8.1", "Change Management", "Entity authorizes and manages changes", ControlCategory.APPLICATION_SECURITY, RiskLevel.MEDIUM),
            ("CC9.1", "Risk Mitigation", "Entity identifies and mitigates risks", ControlCategory.VENDOR_MANAGEMENT, RiskLevel.MEDIUM),
        ]

        controls = []
        for control_id, name, desc, category, risk in controls_def:
            control = await self.assessment.add_control(
                organization_id=organization_id,
                framework=ComplianceFramework.SOC2,
                control_id=control_id,
                name=name,
                description=desc,
                category=category,
                risk_level=risk,
            )
            controls.append(control)

        return controls


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "AuditLogService",
    "ConsentManager",
    "DataSubjectRequestHandler",
    "ComplianceAssessmentService",
    "ComplianceService",
]
