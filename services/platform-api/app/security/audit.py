"""
Security Audit Logging

Comprehensive audit logging with:
- Security event tracking
- Compliance logging
- Tamper-evident logs
- Search and analysis
"""

from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextvars import ContextVar
from functools import wraps
import asyncio
import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    # Authentication
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    PASSWORD_CHANGE = "auth.password.change"
    PASSWORD_RESET = "auth.password.reset"
    MFA_ENABLE = "auth.mfa.enable"
    MFA_DISABLE = "auth.mfa.disable"
    TOKEN_ISSUED = "auth.token.issued"
    TOKEN_REVOKED = "auth.token.revoked"

    # Authorization
    ACCESS_GRANTED = "authz.access.granted"
    ACCESS_DENIED = "authz.access.denied"
    PERMISSION_CHANGE = "authz.permission.change"
    ROLE_ASSIGNED = "authz.role.assigned"
    ROLE_REVOKED = "authz.role.revoked"

    # Data access
    DATA_READ = "data.read"
    DATA_CREATE = "data.create"
    DATA_UPDATE = "data.update"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"

    # Administration
    CONFIG_CHANGE = "admin.config.change"
    USER_CREATE = "admin.user.create"
    USER_UPDATE = "admin.user.update"
    USER_DELETE = "admin.user.delete"
    USER_SUSPEND = "admin.user.suspend"

    # Security
    SECURITY_ALERT = "security.alert"
    INTRUSION_ATTEMPT = "security.intrusion"
    RATE_LIMIT_EXCEEDED = "security.ratelimit"
    ENCRYPTION_KEY_ROTATE = "security.key.rotate"
    CERTIFICATE_EXPIRE = "security.cert.expire"

    # System
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    BACKUP_CREATE = "system.backup.create"
    BACKUP_RESTORE = "system.backup.restore"

    # API
    API_CALL = "api.call"
    API_ERROR = "api.error"
    WEBHOOK_SENT = "api.webhook.sent"
    WEBHOOK_RECEIVED = "api.webhook.received"

    # Compliance
    CONSENT_GIVEN = "compliance.consent.given"
    CONSENT_REVOKED = "compliance.consent.revoked"
    DATA_RETENTION_PURGE = "compliance.retention.purge"
    GDPR_REQUEST = "compliance.gdpr.request"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditContext:
    """Context for audit events."""
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "correlation_id": self.correlation_id,
        }


@dataclass
class AuditEvent:
    """Audit event record."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    severity: AuditSeverity
    actor: Optional[str]  # Who performed the action
    action: str  # Description of the action
    resource_type: Optional[str]  # Type of resource affected
    resource_id: Optional[str]  # ID of resource affected
    outcome: str  # "success", "failure", "error"
    context: AuditContext
    details: Dict[str, Any] = field(default_factory=dict)
    previous_hash: Optional[str] = None
    hash: Optional[str] = None

    def __post_init__(self):
        """Calculate hash after initialization."""
        if self.hash is None:
            self.hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate tamper-evident hash."""
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "action": self.action,
            "outcome": self.outcome,
            "previous_hash": self.previous_hash,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "actor": self.actor,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "outcome": self.outcome,
            "context": self.context.to_dict(),
            "details": self.details,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=AuditEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            severity=AuditSeverity(data["severity"]),
            actor=data.get("actor"),
            action=data["action"],
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            outcome=data["outcome"],
            context=AuditContext(**data.get("context", {})),
            details=data.get("details", {}),
            previous_hash=data.get("previous_hash"),
            hash=data.get("hash"),
        )


# Context variable for audit context
_audit_context: ContextVar[Optional[AuditContext]] = ContextVar(
    "_audit_context",
    default=None,
)


def get_audit_context() -> Optional[AuditContext]:
    """Get current audit context."""
    return _audit_context.get()


def set_audit_context(context: AuditContext) -> None:
    """Set audit context for current execution."""
    _audit_context.set(context)


class AuditStore:
    """Abstract audit event storage."""

    async def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        raise NotImplementedError

    async def search(
        self,
        event_types: Optional[List[AuditEventType]] = None,
        actor: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Search audit events."""
        raise NotImplementedError

    async def verify_chain(self, events: List[AuditEvent]) -> bool:
        """Verify hash chain integrity."""
        for i, event in enumerate(events):
            if i > 0:
                if event.previous_hash != events[i-1].hash:
                    return False
            if event.hash != event._calculate_hash():
                return False
        return True


class InMemoryAuditStore(AuditStore):
    """In-memory audit store for development."""

    def __init__(self, max_events: int = 10000):
        self._events: List[AuditEvent] = []
        self._max_events = max_events
        self._lock = asyncio.Lock()

    async def store(self, event: AuditEvent) -> None:
        """Store event."""
        async with self._lock:
            self._events.append(event)
            # Trim if over limit
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

    async def search(
        self,
        event_types: Optional[List[AuditEventType]] = None,
        actor: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Search events."""
        results = []

        for event in reversed(self._events):
            if event_types and event.event_type not in event_types:
                continue
            if actor and event.actor != actor:
                continue
            if resource_type and event.resource_type != resource_type:
                continue
            if resource_id and event.resource_id != resource_id:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            results.append(event)
            if len(results) >= limit:
                break

        return results


class FileAuditStore(AuditStore):
    """File-based audit store with rotation."""

    def __init__(
        self,
        directory: str,
        max_file_size_mb: int = 100,
        retention_days: int = 365,
    ):
        import os
        self.directory = directory
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.retention_days = retention_days
        self._current_file = None
        self._current_size = 0

        os.makedirs(directory, exist_ok=True)

    async def store(self, event: AuditEvent) -> None:
        """Store event to file."""
        import os

        # Get current log file
        if self._current_file is None or self._current_size >= self.max_file_size:
            self._rotate_file()

        # Write event
        event_json = json.dumps(event.to_dict()) + "\n"
        with open(self._current_file, 'a') as f:
            f.write(event_json)
        self._current_size += len(event_json)

    def _rotate_file(self) -> None:
        """Rotate to new log file."""
        import os
        filename = f"audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
        self._current_file = os.path.join(self.directory, filename)
        self._current_size = 0
        logger.info(f"Rotated audit log to {filename}")

    async def search(
        self,
        event_types: Optional[List[AuditEventType]] = None,
        actor: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Search events in files."""
        import os
        results = []

        # Get all log files
        files = sorted(
            [f for f in os.listdir(self.directory) if f.startswith("audit_")],
            reverse=True,
        )

        for filename in files:
            filepath = os.path.join(self.directory, filename)

            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        event = AuditEvent.from_dict(data)

                        if event_types and event.event_type not in event_types:
                            continue
                        if actor and event.actor != actor:
                            continue
                        if resource_type and event.resource_type != resource_type:
                            continue
                        if resource_id and event.resource_id != resource_id:
                            continue
                        if start_time and event.timestamp < start_time:
                            continue
                        if end_time and event.timestamp > end_time:
                            continue

                        results.append(event)
                        if len(results) >= limit:
                            return results
                    except Exception:
                        continue

        return results


class AuditLogger:
    """
    Enterprise audit logging service.

    Provides tamper-evident logging with chain verification.
    """

    def __init__(
        self,
        store: Optional[AuditStore] = None,
        default_severity: AuditSeverity = AuditSeverity.INFO,
    ):
        self.store = store or InMemoryAuditStore()
        self.default_severity = default_severity
        self._last_hash: Optional[str] = None
        self._lock = asyncio.Lock()

        # Event handlers
        self._handlers: List[Callable[[AuditEvent], None]] = []

    def add_handler(self, handler: Callable[[AuditEvent], None]) -> None:
        """Add event handler."""
        self._handlers.append(handler)

    async def log(
        self,
        event_type: AuditEventType,
        action: str,
        outcome: str = "success",
        actor: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[AuditContext] = None,
    ) -> AuditEvent:
        """Log an audit event."""
        async with self._lock:
            # Use context from context variable if not provided
            if context is None:
                context = get_audit_context() or AuditContext()

            # Create event
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.utcnow(),
                severity=severity or self.default_severity,
                actor=actor or context.user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                outcome=outcome,
                context=context,
                details=details or {},
                previous_hash=self._last_hash,
            )

            # Store event
            await self.store.store(event)
            self._last_hash = event.hash

            # Call handlers
            for handler in self._handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Audit handler error: {e}")

            logger.debug(f"Audit: {event_type.value} - {action}")
            return event

    async def log_login(
        self,
        user_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log login attempt."""
        return await self.log(
            event_type=AuditEventType.LOGIN_SUCCESS if success else AuditEventType.LOGIN_FAILURE,
            action=f"User login {'succeeded' if success else 'failed'}",
            outcome="success" if success else "failure",
            actor=user_id,
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            context=AuditContext(user_id=user_id, ip_address=ip_address),
            details=details,
        )

    async def log_data_access(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log data access."""
        event_type_map = {
            "read": AuditEventType.DATA_READ,
            "create": AuditEventType.DATA_CREATE,
            "update": AuditEventType.DATA_UPDATE,
            "delete": AuditEventType.DATA_DELETE,
        }

        event_type = event_type_map.get(action.lower(), AuditEventType.DATA_READ)

        return await self.log(
            event_type=event_type,
            action=f"{action} {resource_type}",
            resource_type=resource_type,
            resource_id=resource_id,
            actor=user_id,
            details=details,
        )

    async def log_security_event(
        self,
        event_type: AuditEventType,
        action: str,
        severity: AuditSeverity = AuditSeverity.WARNING,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log security event."""
        return await self.log(
            event_type=event_type,
            action=action,
            severity=severity,
            details=details,
        )

    async def search(
        self,
        **kwargs,
    ) -> List[AuditEvent]:
        """Search audit events."""
        return await self.store.search(**kwargs)

    async def verify_integrity(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Verify audit log integrity."""
        events = await self.store.search(
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        # Verify chain
        is_valid = await self.store.verify_chain(events)

        return {
            "is_valid": is_valid,
            "events_checked": len(events),
            "start_time": events[0].timestamp.isoformat() if events else None,
            "end_time": events[-1].timestamp.isoformat() if events else None,
        }


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def set_audit_logger(logger: AuditLogger) -> None:
    """Set global audit logger."""
    global _audit_logger
    _audit_logger = logger


# Decorator for automatic audit logging
def audit_log(
    event_type: AuditEventType,
    action_template: str = "{func_name}",
    resource_type: Optional[str] = None,
    resource_id_param: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
) -> Callable:
    """
    Decorator for automatic audit logging.

    Usage:
        @audit_log(AuditEventType.DATA_READ, resource_type="user")
        async def get_user(user_id: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            audit = get_audit_logger()

            # Build action string
            action = action_template.format(
                func_name=func.__name__,
                **kwargs,
            )

            # Extract resource ID
            res_id = None
            if resource_id_param and resource_id_param in kwargs:
                res_id = str(kwargs[resource_id_param])

            # Build details
            details = {}
            if log_args:
                details["args"] = str(kwargs)

            try:
                result = await func(*args, **kwargs)

                if log_result:
                    details["result_type"] = type(result).__name__

                await audit.log(
                    event_type=event_type,
                    action=action,
                    outcome="success",
                    resource_type=resource_type,
                    resource_id=res_id,
                    details=details,
                )

                return result

            except Exception as e:
                details["error"] = str(e)

                await audit.log(
                    event_type=event_type,
                    action=action,
                    outcome="error",
                    resource_type=resource_type,
                    resource_id=res_id,
                    severity=AuditSeverity.ERROR,
                    details=details,
                )

                raise

        return wrapper
    return decorator
