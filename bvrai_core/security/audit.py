"""
Audit Logging System
====================

Comprehensive audit logging for security, compliance, and forensic analysis.

Author: Platform Security Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    MFA_CHALLENGE = "mfa_challenge"
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    SESSION_REVOKED = "session_revoked"
    TOKEN_ISSUED = "token_issued"
    TOKEN_REVOKED = "token_revoked"
    TOKEN_REFRESHED = "token_refreshed"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REMOVED = "role_removed"

    # Data events
    DATA_READ = "data_read"
    DATA_CREATE = "data_create"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    BULK_OPERATION = "bulk_operation"

    # API events
    API_CALL = "api_call"
    API_ERROR = "api_error"
    RATE_LIMIT_HIT = "rate_limit_hit"
    WEBHOOK_SENT = "webhook_sent"
    WEBHOOK_RECEIVED = "webhook_received"

    # Voice/Call events
    CALL_STARTED = "call_started"
    CALL_ENDED = "call_ended"
    CALL_TRANSFERRED = "call_transferred"
    RECORDING_STARTED = "recording_started"
    RECORDING_STOPPED = "recording_stopped"
    RECORDING_ACCESSED = "recording_accessed"
    RECORDING_DELETED = "recording_deleted"
    TRANSCRIPT_GENERATED = "transcript_generated"

    # Admin events
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    USER_SUSPENDED = "user_suspended"
    USER_ACTIVATED = "user_activated"
    ORG_CREATED = "org_created"
    ORG_UPDATED = "org_updated"
    ORG_DELETED = "org_deleted"
    SETTINGS_CHANGED = "settings_changed"
    CONFIG_CHANGED = "config_changed"

    # Integration events
    INTEGRATION_CONNECTED = "integration_connected"
    INTEGRATION_DISCONNECTED = "integration_disconnected"
    INTEGRATION_SYNC = "integration_sync"
    INTEGRATION_ERROR = "integration_error"

    # Security events
    SECURITY_ALERT = "security_alert"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE_DETECTED = "brute_force_detected"
    IP_BLOCKED = "ip_blocked"
    IP_UNBLOCKED = "ip_unblocked"
    KEY_ROTATED = "key_rotated"
    ENCRYPTION_OPERATION = "encryption_operation"

    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    MAINTENANCE_MODE = "maintenance_mode"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"

    # Compliance events
    CONSENT_GIVEN = "consent_given"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    DATA_RETENTION_APPLIED = "data_retention_applied"
    GDPR_REQUEST = "gdpr_request"
    DATA_ANONYMIZED = "data_anonymized"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditCategory(str, Enum):
    """Categories for audit events."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    API = "api"
    VOICE = "voice"
    ADMIN = "admin"
    INTEGRATION = "integration"
    SECURITY = "security"
    SYSTEM = "system"
    COMPLIANCE = "compliance"


@dataclass
class AuditEvent:
    """Represents an audit event."""

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: AuditEventType = AuditEventType.API_CALL
    severity: AuditSeverity = AuditSeverity.INFO
    category: AuditCategory = AuditCategory.API

    # Actor information
    actor_id: Optional[str] = None
    actor_type: str = "user"  # user, service, system, api_key
    actor_email: Optional[str] = None
    actor_ip: Optional[str] = None
    actor_user_agent: Optional[str] = None

    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None

    # Organization context
    organization_id: Optional[str] = None
    tenant_id: Optional[str] = None

    # Action details
    action: str = ""
    description: str = ""
    outcome: str = "success"  # success, failure, partial
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Request context
    request_id: Optional[str] = None
    request_method: Optional[str] = None
    request_path: Optional[str] = None
    request_params: Dict[str, Any] = field(default_factory=dict)

    # Change tracking
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    changed_fields: List[str] = field(default_factory=list)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Integrity
    checksum: Optional[str] = None
    sequence_number: Optional[int] = None

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate checksum for integrity verification."""
        data = f"{self.id}:{self.timestamp.isoformat()}:{self.event_type.value}:{self.actor_id}:{self.action}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify event integrity."""
        expected = self._calculate_checksum()
        return self.checksum == expected

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "category": self.category.value,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "actor_email": self.actor_email,
            "actor_ip": self.actor_ip,
            "actor_user_agent": self.actor_user_agent,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "organization_id": self.organization_id,
            "tenant_id": self.tenant_id,
            "action": self.action,
            "description": self.description,
            "outcome": self.outcome,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "request_id": self.request_id,
            "request_method": self.request_method,
            "request_path": self.request_path,
            "request_params": self.request_params,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "changed_fields": self.changed_fields,
            "metadata": self.metadata,
            "tags": self.tags,
            "checksum": self.checksum,
            "sequence_number": self.sequence_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            event_type=AuditEventType(data.get("event_type", "api_call")),
            severity=AuditSeverity(data.get("severity", "info")),
            category=AuditCategory(data.get("category", "api")),
            actor_id=data.get("actor_id"),
            actor_type=data.get("actor_type", "user"),
            actor_email=data.get("actor_email"),
            actor_ip=data.get("actor_ip"),
            actor_user_agent=data.get("actor_user_agent"),
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            resource_name=data.get("resource_name"),
            organization_id=data.get("organization_id"),
            tenant_id=data.get("tenant_id"),
            action=data.get("action", ""),
            description=data.get("description", ""),
            outcome=data.get("outcome", "success"),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            request_id=data.get("request_id"),
            request_method=data.get("request_method"),
            request_path=data.get("request_path"),
            request_params=data.get("request_params", {}),
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            changed_fields=data.get("changed_fields", []),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            checksum=data.get("checksum"),
            sequence_number=data.get("sequence_number"),
        )


@dataclass
class AuditFilter:
    """Filter for querying audit events."""

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    severities: Optional[List[AuditSeverity]] = None
    categories: Optional[List[AuditCategory]] = None
    actor_ids: Optional[List[str]] = None
    actor_types: Optional[List[str]] = None
    organization_ids: Optional[List[str]] = None
    resource_types: Optional[List[str]] = None
    resource_ids: Optional[List[str]] = None
    outcomes: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    search_text: Optional[str] = None

    def matches(self, event: AuditEvent) -> bool:
        """Check if event matches filter."""
        if self.start_time and event.timestamp < self.start_time:
            return False
        if self.end_time and event.timestamp > self.end_time:
            return False
        if self.event_types and event.event_type not in self.event_types:
            return False
        if self.severities and event.severity not in self.severities:
            return False
        if self.categories and event.category not in self.categories:
            return False
        if self.actor_ids and event.actor_id not in self.actor_ids:
            return False
        if self.actor_types and event.actor_type not in self.actor_types:
            return False
        if self.organization_ids and event.organization_id not in self.organization_ids:
            return False
        if self.resource_types and event.resource_type not in self.resource_types:
            return False
        if self.resource_ids and event.resource_id not in self.resource_ids:
            return False
        if self.outcomes and event.outcome not in self.outcomes:
            return False
        if self.tags:
            if not any(tag in event.tags for tag in self.tags):
                return False
        if self.search_text:
            text = self.search_text.lower()
            searchable = f"{event.action} {event.description} {event.resource_name or ''}".lower()
            if text not in searchable:
                return False
        return True


class AuditStore(ABC):
    """Abstract base class for audit event storage."""

    @abstractmethod
    async def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        pass

    @abstractmethod
    async def query(
        self,
        filter: AuditFilter,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[AuditEvent], int]:
        """Query audit events."""
        pass

    @abstractmethod
    async def get_by_id(self, event_id: str) -> Optional[AuditEvent]:
        """Get event by ID."""
        pass

    @abstractmethod
    async def count(self, filter: AuditFilter) -> int:
        """Count events matching filter."""
        pass

    async def store_batch(self, events: List[AuditEvent]) -> int:
        """Store multiple events."""
        count = 0
        for event in events:
            await self.store(event)
            count += 1
        return count


class InMemoryAuditStore(AuditStore):
    """In-memory audit store for development/testing."""

    def __init__(self, max_events: int = 100000):
        self._events: Deque[AuditEvent] = deque(maxlen=max_events)
        self._events_by_id: Dict[str, AuditEvent] = {}
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        async with self._lock:
            self._sequence += 1
            event.sequence_number = self._sequence
            self._events.append(event)
            self._events_by_id[event.id] = event

    async def query(
        self,
        filter: AuditFilter,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[AuditEvent], int]:
        """Query audit events."""
        matching = [e for e in self._events if filter.matches(e)]
        matching.sort(key=lambda e: e.timestamp, reverse=True)
        total = len(matching)
        return matching[offset:offset + limit], total

    async def get_by_id(self, event_id: str) -> Optional[AuditEvent]:
        """Get event by ID."""
        return self._events_by_id.get(event_id)

    async def count(self, filter: AuditFilter) -> int:
        """Count events matching filter."""
        return sum(1 for e in self._events if filter.matches(e))


class FileAuditStore(AuditStore):
    """File-based audit store with rotation and compression."""

    def __init__(
        self,
        base_path: str,
        max_file_size_mb: int = 100,
        compress_after_days: int = 7,
        retention_days: int = 365,
    ):
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._max_file_size = max_file_size_mb * 1024 * 1024
        self._compress_after_days = compress_after_days
        self._retention_days = retention_days
        self._current_file: Optional[Path] = None
        self._current_file_size = 0
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._index: Dict[str, Tuple[str, int]] = {}  # event_id -> (file, line)

    def _get_current_file_path(self) -> Path:
        """Get path for current log file."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        hour_str = datetime.utcnow().strftime("%H")
        return self._base_path / f"audit_{date_str}_{hour_str}.jsonl"

    async def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        async with self._lock:
            self._sequence += 1
            event.sequence_number = self._sequence

            file_path = self._get_current_file_path()

            # Rotate if needed
            if file_path != self._current_file:
                self._current_file = file_path
                self._current_file_size = file_path.stat().st_size if file_path.exists() else 0

            # Write event
            event_json = json.dumps(event.to_dict()) + "\n"
            event_bytes = event_json.encode()

            with open(file_path, "a") as f:
                f.write(event_json)

            self._current_file_size += len(event_bytes)

            # Update index
            line_number = sum(1 for _ in open(file_path)) if file_path.exists() else 1
            self._index[event.id] = (str(file_path), line_number)

    async def query(
        self,
        filter: AuditFilter,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[AuditEvent], int]:
        """Query audit events."""
        matching: List[AuditEvent] = []

        # Get relevant files based on time range
        files = self._get_files_in_range(filter.start_time, filter.end_time)

        for file_path in files:
            events = await self._read_file(file_path)
            for event in events:
                if filter.matches(event):
                    matching.append(event)

        matching.sort(key=lambda e: e.timestamp, reverse=True)
        total = len(matching)
        return matching[offset:offset + limit], total

    async def get_by_id(self, event_id: str) -> Optional[AuditEvent]:
        """Get event by ID."""
        if event_id in self._index:
            file_path, line_number = self._index[event_id]
            events = await self._read_file(Path(file_path))
            for event in events:
                if event.id == event_id:
                    return event
        return None

    async def count(self, filter: AuditFilter) -> int:
        """Count events matching filter."""
        count = 0
        files = self._get_files_in_range(filter.start_time, filter.end_time)
        for file_path in files:
            events = await self._read_file(file_path)
            count += sum(1 for e in events if filter.matches(e))
        return count

    def _get_files_in_range(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> List[Path]:
        """Get audit files within time range."""
        files = []
        for file_path in self._base_path.glob("audit_*.jsonl*"):
            files.append(file_path)
        return sorted(files, reverse=True)

    async def _read_file(self, file_path: Path) -> List[AuditEvent]:
        """Read events from file."""
        events = []

        if not file_path.exists():
            return events

        if file_path.suffix == ".gz":
            with gzip.open(file_path, "rt") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        events.append(AuditEvent.from_dict(data))
        else:
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        events.append(AuditEvent.from_dict(data))

        return events

    async def compress_old_files(self) -> int:
        """Compress old audit files."""
        compressed = 0
        cutoff = datetime.utcnow() - timedelta(days=self._compress_after_days)

        for file_path in self._base_path.glob("audit_*.jsonl"):
            if file_path.suffix == ".gz":
                continue

            # Parse date from filename
            try:
                date_str = file_path.stem.split("_")[1]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff:
                    # Compress file
                    with open(file_path, "rb") as f_in:
                        with gzip.open(f"{file_path}.gz", "wb") as f_out:
                            f_out.write(f_in.read())
                    file_path.unlink()
                    compressed += 1
            except (IndexError, ValueError):
                continue

        return compressed

    async def cleanup_old_files(self) -> int:
        """Remove files older than retention period."""
        removed = 0
        cutoff = datetime.utcnow() - timedelta(days=self._retention_days)

        for file_path in self._base_path.glob("audit_*"):
            try:
                date_str = file_path.stem.split("_")[1]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff:
                    file_path.unlink()
                    removed += 1
            except (IndexError, ValueError):
                continue

        return removed


class AuditLogger:
    """
    High-level audit logging service.

    Provides easy-to-use methods for logging various types of events
    with automatic context capture and batching support.
    """

    def __init__(
        self,
        store: AuditStore,
        default_organization_id: Optional[str] = None,
        batch_size: int = 100,
        flush_interval_seconds: float = 5.0,
    ):
        self._store = store
        self._default_org_id = default_organization_id
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds
        self._batch: List[AuditEvent] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._hooks: List[Callable[[AuditEvent], None]] = []
        self._logger = structlog.get_logger("audit_logger")

    async def start(self) -> None:
        """Start the audit logger with background flushing."""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        self._logger.info("Audit logger started")

    async def stop(self) -> None:
        """Stop the audit logger and flush remaining events."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush()
        self._logger.info("Audit logger stopped")

    def add_hook(self, hook: Callable[[AuditEvent], None]) -> None:
        """Add a hook to be called for each event."""
        self._hooks.append(hook)

    async def log(
        self,
        event_type: AuditEventType,
        action: str,
        *,
        actor_id: Optional[str] = None,
        actor_type: str = "user",
        actor_email: Optional[str] = None,
        actor_ip: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_name: Optional[str] = None,
        organization_id: Optional[str] = None,
        description: str = "",
        outcome: str = "success",
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        old_value: Optional[Dict[str, Any]] = None,
        new_value: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        request_method: Optional[str] = None,
        request_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        severity: Optional[AuditSeverity] = None,
        category: Optional[AuditCategory] = None,
    ) -> AuditEvent:
        """Log an audit event."""
        # Determine category and severity from event type if not provided
        if not category:
            category = self._get_category_for_event(event_type)
        if not severity:
            severity = self._get_severity_for_outcome(outcome)

        # Calculate changed fields
        changed_fields = []
        if old_value and new_value:
            all_keys = set(old_value.keys()) | set(new_value.keys())
            for key in all_keys:
                if old_value.get(key) != new_value.get(key):
                    changed_fields.append(key)

        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            category=category,
            actor_id=actor_id,
            actor_type=actor_type,
            actor_email=actor_email,
            actor_ip=actor_ip,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            organization_id=organization_id or self._default_org_id,
            action=action,
            description=description,
            outcome=outcome,
            error_code=error_code,
            error_message=error_message,
            old_value=old_value,
            new_value=new_value,
            changed_fields=changed_fields,
            request_id=request_id,
            request_method=request_method,
            request_path=request_path,
            metadata=metadata or {},
            tags=tags or [],
        )

        # Call hooks
        for hook in self._hooks:
            try:
                hook(event)
            except Exception as e:
                self._logger.error(f"Audit hook error: {e}")

        # Add to batch
        async with self._lock:
            self._batch.append(event)
            if len(self._batch) >= self._batch_size:
                await self._flush()

        return event

    async def _flush(self) -> None:
        """Flush batched events to storage."""
        async with self._lock:
            if not self._batch:
                return
            events = self._batch
            self._batch = []

        try:
            await self._store.store_batch(events)
            self._logger.debug(f"Flushed {len(events)} audit events")
        except Exception as e:
            self._logger.error(f"Failed to flush audit events: {e}")
            # Re-add events on failure
            async with self._lock:
                self._batch = events + self._batch

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            await asyncio.sleep(self._flush_interval)
            await self._flush()

    def _get_category_for_event(self, event_type: AuditEventType) -> AuditCategory:
        """Determine category from event type."""
        auth_events = {
            AuditEventType.LOGIN_SUCCESS, AuditEventType.LOGIN_FAILURE,
            AuditEventType.LOGOUT, AuditEventType.PASSWORD_CHANGE,
            AuditEventType.PASSWORD_RESET, AuditEventType.MFA_ENABLED,
            AuditEventType.MFA_DISABLED, AuditEventType.MFA_CHALLENGE,
            AuditEventType.SESSION_CREATED, AuditEventType.SESSION_EXPIRED,
            AuditEventType.SESSION_REVOKED, AuditEventType.TOKEN_ISSUED,
            AuditEventType.TOKEN_REVOKED, AuditEventType.TOKEN_REFRESHED,
        }
        authz_events = {
            AuditEventType.ACCESS_GRANTED, AuditEventType.ACCESS_DENIED,
            AuditEventType.PERMISSION_GRANTED, AuditEventType.PERMISSION_REVOKED,
            AuditEventType.ROLE_ASSIGNED, AuditEventType.ROLE_REMOVED,
        }
        data_events = {
            AuditEventType.DATA_READ, AuditEventType.DATA_CREATE,
            AuditEventType.DATA_UPDATE, AuditEventType.DATA_DELETE,
            AuditEventType.DATA_EXPORT, AuditEventType.DATA_IMPORT,
            AuditEventType.BULK_OPERATION,
        }
        voice_events = {
            AuditEventType.CALL_STARTED, AuditEventType.CALL_ENDED,
            AuditEventType.CALL_TRANSFERRED, AuditEventType.RECORDING_STARTED,
            AuditEventType.RECORDING_STOPPED, AuditEventType.RECORDING_ACCESSED,
            AuditEventType.RECORDING_DELETED, AuditEventType.TRANSCRIPT_GENERATED,
        }
        security_events = {
            AuditEventType.SECURITY_ALERT, AuditEventType.SUSPICIOUS_ACTIVITY,
            AuditEventType.BRUTE_FORCE_DETECTED, AuditEventType.IP_BLOCKED,
            AuditEventType.IP_UNBLOCKED, AuditEventType.KEY_ROTATED,
            AuditEventType.ENCRYPTION_OPERATION,
        }
        compliance_events = {
            AuditEventType.CONSENT_GIVEN, AuditEventType.CONSENT_WITHDRAWN,
            AuditEventType.DATA_RETENTION_APPLIED, AuditEventType.GDPR_REQUEST,
            AuditEventType.DATA_ANONYMIZED,
        }

        if event_type in auth_events:
            return AuditCategory.AUTHENTICATION
        if event_type in authz_events:
            return AuditCategory.AUTHORIZATION
        if event_type in data_events:
            return AuditCategory.DATA_ACCESS
        if event_type in voice_events:
            return AuditCategory.VOICE
        if event_type in security_events:
            return AuditCategory.SECURITY
        if event_type in compliance_events:
            return AuditCategory.COMPLIANCE

        return AuditCategory.API

    def _get_severity_for_outcome(self, outcome: str) -> AuditSeverity:
        """Determine severity from outcome."""
        if outcome == "failure":
            return AuditSeverity.ERROR
        if outcome == "partial":
            return AuditSeverity.WARNING
        return AuditSeverity.INFO

    # Convenience methods for common event types

    async def log_login(
        self,
        actor_id: str,
        success: bool,
        *,
        actor_email: Optional[str] = None,
        actor_ip: Optional[str] = None,
        method: str = "password",
        error_message: Optional[str] = None,
        **kwargs,
    ) -> AuditEvent:
        """Log login attempt."""
        return await self.log(
            AuditEventType.LOGIN_SUCCESS if success else AuditEventType.LOGIN_FAILURE,
            f"login_via_{method}",
            actor_id=actor_id,
            actor_email=actor_email,
            actor_ip=actor_ip,
            outcome="success" if success else "failure",
            error_message=error_message,
            **kwargs,
        )

    async def log_data_access(
        self,
        actor_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        *,
        old_value: Optional[Dict[str, Any]] = None,
        new_value: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AuditEvent:
        """Log data access/modification."""
        event_type_map = {
            "read": AuditEventType.DATA_READ,
            "create": AuditEventType.DATA_CREATE,
            "update": AuditEventType.DATA_UPDATE,
            "delete": AuditEventType.DATA_DELETE,
        }
        event_type = event_type_map.get(action, AuditEventType.DATA_READ)

        return await self.log(
            event_type,
            f"{action}_{resource_type}",
            actor_id=actor_id,
            resource_type=resource_type,
            resource_id=resource_id,
            old_value=old_value,
            new_value=new_value,
            **kwargs,
        )

    async def log_api_call(
        self,
        request_method: str,
        request_path: str,
        *,
        actor_id: Optional[str] = None,
        actor_ip: Optional[str] = None,
        status_code: int = 200,
        duration_ms: Optional[float] = None,
        **kwargs,
    ) -> AuditEvent:
        """Log API call."""
        outcome = "success" if 200 <= status_code < 400 else "failure"

        return await self.log(
            AuditEventType.API_CALL,
            f"{request_method} {request_path}",
            actor_id=actor_id,
            actor_ip=actor_ip,
            request_method=request_method,
            request_path=request_path,
            outcome=outcome,
            metadata={"status_code": status_code, "duration_ms": duration_ms},
            **kwargs,
        )

    async def log_security_event(
        self,
        event_type: AuditEventType,
        description: str,
        *,
        actor_id: Optional[str] = None,
        actor_ip: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.WARNING,
        **kwargs,
    ) -> AuditEvent:
        """Log security-related event."""
        return await self.log(
            event_type,
            "security_event",
            actor_id=actor_id,
            actor_ip=actor_ip,
            description=description,
            severity=severity,
            **kwargs,
        )

    async def log_call_event(
        self,
        event_type: AuditEventType,
        call_id: str,
        *,
        actor_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        **kwargs,
    ) -> AuditEvent:
        """Log call-related event."""
        return await self.log(
            event_type,
            event_type.value,
            actor_id=actor_id,
            resource_type="call",
            resource_id=call_id,
            metadata={
                "agent_id": agent_id,
                "duration_seconds": duration_seconds,
            },
            **kwargs,
        )

    async def query(
        self,
        filter: AuditFilter,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[AuditEvent], int]:
        """Query audit events."""
        return await self._store.query(filter, limit, offset)

    async def get_statistics(
        self,
        start_time: datetime,
        end_time: datetime,
        organization_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get audit statistics for time range."""
        filter = AuditFilter(
            start_time=start_time,
            end_time=end_time,
            organization_ids=[organization_id] if organization_id else None,
        )

        events, total = await self.query(filter, limit=10000)

        # Calculate statistics
        by_type: Dict[str, int] = defaultdict(int)
        by_category: Dict[str, int] = defaultdict(int)
        by_severity: Dict[str, int] = defaultdict(int)
        by_outcome: Dict[str, int] = defaultdict(int)
        by_actor: Dict[str, int] = defaultdict(int)

        for event in events:
            by_type[event.event_type.value] += 1
            by_category[event.category.value] += 1
            by_severity[event.severity.value] += 1
            by_outcome[event.outcome] += 1
            if event.actor_id:
                by_actor[event.actor_id] += 1

        return {
            "total_events": total,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "by_type": dict(by_type),
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "by_outcome": dict(by_outcome),
            "top_actors": dict(sorted(by_actor.items(), key=lambda x: x[1], reverse=True)[:10]),
        }
