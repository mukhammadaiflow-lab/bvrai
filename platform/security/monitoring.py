"""
Security Monitoring System
==========================

Real-time security monitoring, threat detection, and anomaly detection
for the Voice AI platform.

Author: Platform Security Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
from uuid import uuid4

import structlog

from platform.security.audit import AuditEvent, AuditEventType, AuditSeverity

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Severity levels for security alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Status of a security alert."""

    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class ThreatType(str, Enum):
    """Types of security threats."""

    BRUTE_FORCE = "brute_force"
    CREDENTIAL_STUFFING = "credential_stuffing"
    ACCOUNT_TAKEOVER = "account_takeover"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    INJECTION_ATTEMPT = "injection_attempt"
    SUSPICIOUS_API_USAGE = "suspicious_api_usage"
    RATE_LIMIT_ABUSE = "rate_limit_abuse"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    IP_REPUTATION = "ip_reputation"
    SESSION_HIJACKING = "session_hijacking"


@dataclass
class SecurityAlert:
    """A security alert."""

    id: str = field(default_factory=lambda: f"alert_{uuid4().hex[:16]}")
    threat_type: ThreatType = ThreatType.ANOMALOUS_BEHAVIOR
    severity: AlertSeverity = AlertSeverity.MEDIUM
    status: AlertStatus = AlertStatus.NEW

    # Alert details
    title: str = ""
    description: str = ""
    source: str = ""  # What triggered the alert

    # Context
    actor_id: Optional[str] = None
    actor_ip: Optional[str] = None
    organization_id: Optional[str] = None
    resource: Optional[str] = None

    # Evidence
    related_events: List[str] = field(default_factory=list)  # Event IDs
    indicators: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    # Response
    recommended_actions: List[str] = field(default_factory=list)
    auto_mitigated: bool = False
    mitigation_action: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    # Tracking
    assigned_to: Optional[str] = None
    notes: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def acknowledge(self, user_id: str) -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
        self.notes.append({
            "action": "acknowledged",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def resolve(self, user_id: str, resolution_note: str = "") -> None:
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
        self.notes.append({
            "action": "resolved",
            "user_id": user_id,
            "note": resolution_note,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "threat_type": self.threat_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "actor_id": self.actor_id,
            "actor_ip": self.actor_ip,
            "organization_id": self.organization_id,
            "resource": self.resource,
            "related_events": self.related_events,
            "indicators": self.indicators,
            "recommended_actions": self.recommended_actions,
            "auto_mitigated": self.auto_mitigated,
            "mitigation_action": self.mitigation_action,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "assigned_to": self.assigned_to,
            "tags": self.tags,
        }


@dataclass
class SecurityMetrics:
    """Security metrics and statistics."""

    time_window_minutes: int = 60
    login_attempts: int = 0
    failed_logins: int = 0
    successful_logins: int = 0
    unique_ips: int = 0
    blocked_ips: int = 0
    api_calls: int = 0
    api_errors: int = 0
    access_denied_count: int = 0
    data_exports: int = 0
    alerts_generated: int = 0
    alerts_by_severity: Dict[str, int] = field(default_factory=dict)
    alerts_by_type: Dict[str, int] = field(default_factory=dict)
    top_offending_ips: List[Tuple[str, int]] = field(default_factory=list)
    top_targeted_resources: List[Tuple[str, int]] = field(default_factory=list)
    calculated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time_window_minutes": self.time_window_minutes,
            "login_attempts": self.login_attempts,
            "failed_logins": self.failed_logins,
            "successful_logins": self.successful_logins,
            "login_failure_rate": self.failed_logins / max(self.login_attempts, 1),
            "unique_ips": self.unique_ips,
            "blocked_ips": self.blocked_ips,
            "api_calls": self.api_calls,
            "api_errors": self.api_errors,
            "api_error_rate": self.api_errors / max(self.api_calls, 1),
            "access_denied_count": self.access_denied_count,
            "data_exports": self.data_exports,
            "alerts_generated": self.alerts_generated,
            "alerts_by_severity": self.alerts_by_severity,
            "alerts_by_type": self.alerts_by_type,
            "top_offending_ips": self.top_offending_ips,
            "top_targeted_resources": self.top_targeted_resources,
            "calculated_at": self.calculated_at.isoformat(),
        }


class ThreatDetector:
    """
    Detects security threats based on patterns in events.
    """

    def __init__(
        self,
        failed_login_threshold: int = 5,
        failed_login_window_seconds: int = 300,
        api_rate_threshold: int = 1000,
        api_rate_window_seconds: int = 60,
        data_export_threshold: int = 10,
        data_export_window_seconds: int = 3600,
    ):
        self._failed_login_threshold = failed_login_threshold
        self._failed_login_window = failed_login_window_seconds
        self._api_rate_threshold = api_rate_threshold
        self._api_rate_window = api_rate_window_seconds
        self._data_export_threshold = data_export_threshold
        self._data_export_window = data_export_window_seconds

        # Tracking
        self._failed_logins: Dict[str, Deque[datetime]] = defaultdict(lambda: deque(maxlen=100))
        self._api_calls: Dict[str, Deque[datetime]] = defaultdict(lambda: deque(maxlen=10000))
        self._data_exports: Dict[str, Deque[datetime]] = defaultdict(lambda: deque(maxlen=100))
        self._blocked_ips: Set[str] = set()
        self._suspicious_patterns: Dict[str, List[str]] = {}

        self._logger = structlog.get_logger("threat_detector")

    def analyze_event(self, event: AuditEvent) -> List[SecurityAlert]:
        """Analyze an event for potential threats."""
        alerts: List[SecurityAlert] = []

        # Check for brute force
        if event.event_type == AuditEventType.LOGIN_FAILURE:
            alert = self._check_brute_force(event)
            if alert:
                alerts.append(alert)

        # Check for rate limit abuse
        if event.event_type == AuditEventType.API_CALL:
            alert = self._check_rate_abuse(event)
            if alert:
                alerts.append(alert)

        # Check for data exfiltration
        if event.event_type == AuditEventType.DATA_EXPORT:
            alert = self._check_data_exfiltration(event)
            if alert:
                alerts.append(alert)

        # Check for privilege escalation
        if event.event_type in (AuditEventType.ROLE_ASSIGNED, AuditEventType.PERMISSION_GRANTED):
            alert = self._check_privilege_escalation(event)
            if alert:
                alerts.append(alert)

        # Check for suspicious API patterns
        if event.event_type == AuditEventType.API_ERROR:
            alert = self._check_injection_attempt(event)
            if alert:
                alerts.append(alert)

        return alerts

    def _check_brute_force(self, event: AuditEvent) -> Optional[SecurityAlert]:
        """Check for brute force attack."""
        key = event.actor_ip or event.actor_id or "unknown"
        now = datetime.utcnow()

        # Record failed login
        self._failed_logins[key].append(now)

        # Count recent failures
        cutoff = now - timedelta(seconds=self._failed_login_window)
        recent_failures = sum(1 for t in self._failed_logins[key] if t > cutoff)

        if recent_failures >= self._failed_login_threshold:
            self._blocked_ips.add(event.actor_ip) if event.actor_ip else None

            return SecurityAlert(
                threat_type=ThreatType.BRUTE_FORCE,
                severity=AlertSeverity.HIGH,
                title=f"Brute force attack detected from {key}",
                description=f"Detected {recent_failures} failed login attempts in {self._failed_login_window} seconds",
                source="threat_detector",
                actor_id=event.actor_id,
                actor_ip=event.actor_ip,
                organization_id=event.organization_id,
                related_events=[event.id],
                indicators={
                    "failed_attempts": recent_failures,
                    "window_seconds": self._failed_login_window,
                    "threshold": self._failed_login_threshold,
                },
                recommended_actions=[
                    f"Block IP: {event.actor_ip}" if event.actor_ip else "Review user account",
                    "Enable account lockout",
                    "Enforce MFA",
                ],
                auto_mitigated=bool(event.actor_ip),
                mitigation_action=f"IP {event.actor_ip} blocked" if event.actor_ip else None,
            )

        return None

    def _check_rate_abuse(self, event: AuditEvent) -> Optional[SecurityAlert]:
        """Check for API rate limit abuse."""
        key = event.actor_ip or event.actor_id or "unknown"
        now = datetime.utcnow()

        # Record API call
        self._api_calls[key].append(now)

        # Count recent calls
        cutoff = now - timedelta(seconds=self._api_rate_window)
        recent_calls = sum(1 for t in self._api_calls[key] if t > cutoff)

        if recent_calls >= self._api_rate_threshold:
            return SecurityAlert(
                threat_type=ThreatType.RATE_LIMIT_ABUSE,
                severity=AlertSeverity.MEDIUM,
                title=f"Rate limit abuse detected from {key}",
                description=f"Detected {recent_calls} API calls in {self._api_rate_window} seconds",
                source="threat_detector",
                actor_id=event.actor_id,
                actor_ip=event.actor_ip,
                organization_id=event.organization_id,
                related_events=[event.id],
                indicators={
                    "api_calls": recent_calls,
                    "window_seconds": self._api_rate_window,
                    "threshold": self._api_rate_threshold,
                },
                recommended_actions=[
                    "Review API usage patterns",
                    "Consider implementing stricter rate limits",
                    "Contact user if legitimate",
                ],
            )

        return None

    def _check_data_exfiltration(self, event: AuditEvent) -> Optional[SecurityAlert]:
        """Check for potential data exfiltration."""
        key = event.actor_id or event.actor_ip or "unknown"
        now = datetime.utcnow()

        # Record data export
        self._data_exports[key].append(now)

        # Count recent exports
        cutoff = now - timedelta(seconds=self._data_export_window)
        recent_exports = sum(1 for t in self._data_exports[key] if t > cutoff)

        if recent_exports >= self._data_export_threshold:
            return SecurityAlert(
                threat_type=ThreatType.DATA_EXFILTRATION,
                severity=AlertSeverity.CRITICAL,
                title=f"Potential data exfiltration by {key}",
                description=f"Detected {recent_exports} data exports in {self._data_export_window // 3600} hours",
                source="threat_detector",
                actor_id=event.actor_id,
                actor_ip=event.actor_ip,
                organization_id=event.organization_id,
                related_events=[event.id],
                indicators={
                    "data_exports": recent_exports,
                    "window_hours": self._data_export_window // 3600,
                    "threshold": self._data_export_threshold,
                },
                recommended_actions=[
                    "Immediately review user's activity",
                    "Consider suspending account",
                    "Check what data was exported",
                    "Notify security team",
                ],
            )

        return None

    def _check_privilege_escalation(self, event: AuditEvent) -> Optional[SecurityAlert]:
        """Check for suspicious privilege escalation."""
        # Alert on admin role assignments
        new_value = event.new_value or {}
        if "admin" in str(new_value).lower() or "super" in str(new_value).lower():
            return SecurityAlert(
                threat_type=ThreatType.PRIVILEGE_ESCALATION,
                severity=AlertSeverity.HIGH,
                title=f"Admin privilege granted to {event.resource_id}",
                description="High-privilege role was assigned",
                source="threat_detector",
                actor_id=event.actor_id,
                actor_ip=event.actor_ip,
                organization_id=event.organization_id,
                resource=event.resource_id,
                related_events=[event.id],
                indicators={
                    "role_assigned": new_value,
                    "assigned_by": event.actor_id,
                },
                recommended_actions=[
                    "Verify this was an authorized change",
                    "Review the requesting user's permissions",
                    "Check for other suspicious activity",
                ],
            )

        return None

    def _check_injection_attempt(self, event: AuditEvent) -> Optional[SecurityAlert]:
        """Check for injection attempts in API errors."""
        error_message = event.error_message or ""
        request_params = event.request_params or {}

        # Check for SQL injection patterns
        sql_patterns = ["select", "union", "insert", "delete", "drop", "--", ";--", "/*"]
        for pattern in sql_patterns:
            if pattern.lower() in str(request_params).lower():
                return SecurityAlert(
                    threat_type=ThreatType.INJECTION_ATTEMPT,
                    severity=AlertSeverity.HIGH,
                    title=f"Potential SQL injection attempt from {event.actor_ip}",
                    description=f"Suspicious pattern detected in request parameters",
                    source="threat_detector",
                    actor_id=event.actor_id,
                    actor_ip=event.actor_ip,
                    organization_id=event.organization_id,
                    resource=event.request_path,
                    related_events=[event.id],
                    indicators={
                        "pattern_detected": pattern,
                        "request_path": event.request_path,
                    },
                    recommended_actions=[
                        f"Block IP: {event.actor_ip}",
                        "Review application logs",
                        "Verify input validation",
                    ],
                    tags=["injection", "sql"],
                )

        return None

    def is_ip_blocked(self, ip: str) -> bool:
        """Check if an IP is blocked."""
        return ip in self._blocked_ips

    def block_ip(self, ip: str) -> None:
        """Block an IP address."""
        self._blocked_ips.add(ip)
        self._logger.warning(f"IP blocked: {ip}")

    def unblock_ip(self, ip: str) -> None:
        """Unblock an IP address."""
        self._blocked_ips.discard(ip)
        self._logger.info(f"IP unblocked: {ip}")


class AnomalyDetector:
    """
    Detects anomalies in user behavior and system patterns.
    """

    def __init__(
        self,
        baseline_window_hours: int = 24,
        deviation_threshold: float = 3.0,
    ):
        self._baseline_window = baseline_window_hours
        self._deviation_threshold = deviation_threshold

        # User behavior baselines
        self._user_activity: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=1000))
        self._user_baselines: Dict[str, Dict[str, float]] = {}

        # System baselines
        self._system_metrics: Deque[Dict[str, float]] = deque(maxlen=1440)  # 24 hours at 1-min intervals
        self._system_baseline: Dict[str, float] = {}

        self._logger = structlog.get_logger("anomaly_detector")

    def record_user_activity(
        self,
        user_id: str,
        activity_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record user activity for baseline."""
        self._user_activity[user_id].append({
            "type": activity_type,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {},
        })

    def analyze_user_behavior(
        self,
        user_id: str,
        event: AuditEvent,
    ) -> Optional[SecurityAlert]:
        """Analyze user behavior for anomalies."""
        # Record activity
        self.record_user_activity(
            user_id,
            event.event_type.value,
            {"ip": event.actor_ip, "resource": event.resource_type},
        )

        activities = list(self._user_activity[user_id])
        if len(activities) < 10:
            return None  # Not enough data

        # Calculate baseline
        baseline = self._calculate_user_baseline(user_id)

        # Check for anomalies
        now = datetime.utcnow()
        hour = now.hour
        day_of_week = now.weekday()

        # Unusual login time
        usual_hours = baseline.get("usual_hours", [])
        if event.event_type == AuditEventType.LOGIN_SUCCESS:
            if usual_hours and hour not in usual_hours:
                return SecurityAlert(
                    threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                    severity=AlertSeverity.LOW,
                    title=f"Unusual login time for {user_id}",
                    description=f"User logged in at {hour}:00, outside usual hours {usual_hours}",
                    source="anomaly_detector",
                    actor_id=user_id,
                    actor_ip=event.actor_ip,
                    organization_id=event.organization_id,
                    related_events=[event.id],
                    indicators={
                        "login_hour": hour,
                        "usual_hours": usual_hours,
                    },
                    recommended_actions=[
                        "Verify user identity",
                        "Check for compromised credentials",
                    ],
                    tags=["anomaly", "login_time"],
                )

        # Unusual location (IP-based)
        usual_ips = baseline.get("usual_ips", [])
        if event.actor_ip and usual_ips and event.actor_ip not in usual_ips:
            return SecurityAlert(
                threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                severity=AlertSeverity.MEDIUM,
                title=f"New IP address for {user_id}",
                description=f"Activity from new IP: {event.actor_ip}",
                source="anomaly_detector",
                actor_id=user_id,
                actor_ip=event.actor_ip,
                organization_id=event.organization_id,
                related_events=[event.id],
                indicators={
                    "new_ip": event.actor_ip,
                    "usual_ips": usual_ips[:5],  # Limit for readability
                },
                recommended_actions=[
                    "Confirm user is aware of new location",
                    "Review recent activity",
                ],
                tags=["anomaly", "new_ip"],
            )

        # Activity spike
        recent_count = len([a for a in activities if (now - a["timestamp"]).total_seconds() < 300])
        avg_rate = baseline.get("avg_activity_per_5min", 10)
        if recent_count > avg_rate * self._deviation_threshold:
            return SecurityAlert(
                threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                severity=AlertSeverity.MEDIUM,
                title=f"Activity spike for {user_id}",
                description=f"Unusual activity rate: {recent_count} actions in 5 minutes (avg: {avg_rate})",
                source="anomaly_detector",
                actor_id=user_id,
                actor_ip=event.actor_ip,
                organization_id=event.organization_id,
                related_events=[event.id],
                indicators={
                    "recent_activity_count": recent_count,
                    "average_rate": avg_rate,
                    "deviation_factor": recent_count / max(avg_rate, 1),
                },
                recommended_actions=[
                    "Review recent activity",
                    "Check for automation or scripting",
                ],
                tags=["anomaly", "activity_spike"],
            )

        return None

    def _calculate_user_baseline(self, user_id: str) -> Dict[str, Any]:
        """Calculate behavior baseline for user."""
        if user_id in self._user_baselines:
            return self._user_baselines[user_id]

        activities = list(self._user_activity[user_id])
        if not activities:
            return {}

        # Calculate usual hours
        hours = [a["timestamp"].hour for a in activities]
        usual_hours = list(set(hours))

        # Calculate usual IPs
        ips = [a["metadata"].get("ip") for a in activities if a["metadata"].get("ip")]
        usual_ips = list(set(ips))

        # Calculate average activity rate
        if len(activities) >= 2:
            time_span = (activities[-1]["timestamp"] - activities[0]["timestamp"]).total_seconds()
            intervals = time_span / 300  # 5-minute intervals
            avg_rate = len(activities) / max(intervals, 1)
        else:
            avg_rate = 1

        baseline = {
            "usual_hours": usual_hours,
            "usual_ips": usual_ips,
            "avg_activity_per_5min": avg_rate,
        }

        self._user_baselines[user_id] = baseline
        return baseline

    def record_system_metrics(self, metrics: Dict[str, float]) -> None:
        """Record system metrics for baseline."""
        metrics["timestamp"] = datetime.utcnow().timestamp()
        self._system_metrics.append(metrics)

    def analyze_system_metrics(self, metrics: Dict[str, float]) -> List[SecurityAlert]:
        """Analyze system metrics for anomalies."""
        alerts = []

        if len(self._system_metrics) < 60:
            return alerts  # Not enough data

        # Calculate baseline
        baseline = self._calculate_system_baseline()

        for metric_name, value in metrics.items():
            if metric_name == "timestamp":
                continue

            baseline_value = baseline.get(metric_name, {}).get("mean", value)
            baseline_std = baseline.get(metric_name, {}).get("std", 1)

            if baseline_std > 0:
                deviation = abs(value - baseline_value) / baseline_std
                if deviation > self._deviation_threshold:
                    alerts.append(SecurityAlert(
                        threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                        severity=AlertSeverity.MEDIUM if deviation < 5 else AlertSeverity.HIGH,
                        title=f"System metric anomaly: {metric_name}",
                        description=f"Metric {metric_name} is {deviation:.1f} standard deviations from baseline",
                        source="anomaly_detector",
                        indicators={
                            "metric": metric_name,
                            "current_value": value,
                            "baseline_mean": baseline_value,
                            "baseline_std": baseline_std,
                            "deviation": deviation,
                        },
                        recommended_actions=[
                            "Investigate system health",
                            "Check for resource exhaustion",
                            "Review recent deployments",
                        ],
                        tags=["anomaly", "system_metrics"],
                    ))

        return alerts

    def _calculate_system_baseline(self) -> Dict[str, Dict[str, float]]:
        """Calculate system metrics baseline."""
        metrics_list = list(self._system_metrics)
        if not metrics_list:
            return {}

        baseline = {}
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())

        for key in all_keys:
            if key == "timestamp":
                continue

            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                baseline[key] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                }

        return baseline


class SecurityMonitor:
    """
    Central security monitoring service.

    Coordinates threat detection, anomaly detection, and alert management.
    """

    def __init__(self):
        self._threat_detector = ThreatDetector()
        self._anomaly_detector = AnomalyDetector()
        self._alerts: Dict[str, SecurityAlert] = {}
        self._alert_handlers: List[Callable[[SecurityAlert], None]] = []
        self._logger = structlog.get_logger("security_monitor")
        self._running = False
        self._metrics_task: Optional[asyncio.Task] = None

        # Metrics tracking
        self._event_counts: Dict[str, int] = defaultdict(int)
        self._alert_counts: Dict[str, int] = defaultdict(int)

    @property
    def threat_detector(self) -> ThreatDetector:
        """Get threat detector."""
        return self._threat_detector

    @property
    def anomaly_detector(self) -> AnomalyDetector:
        """Get anomaly detector."""
        return self._anomaly_detector

    def add_alert_handler(self, handler: Callable[[SecurityAlert], None]) -> None:
        """Add handler to be called when alerts are generated."""
        self._alert_handlers.append(handler)

    async def start(self) -> None:
        """Start the security monitor."""
        self._running = True
        self._metrics_task = asyncio.create_task(self._metrics_loop())
        self._logger.info("Security monitor started")

    async def stop(self) -> None:
        """Stop the security monitor."""
        self._running = False
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        self._logger.info("Security monitor stopped")

    async def process_event(self, event: AuditEvent) -> List[SecurityAlert]:
        """
        Process an audit event for security analysis.

        Returns list of alerts generated.
        """
        alerts: List[SecurityAlert] = []

        # Track event
        self._event_counts[event.event_type.value] += 1

        # Run threat detection
        threat_alerts = self._threat_detector.analyze_event(event)
        alerts.extend(threat_alerts)

        # Run anomaly detection for authenticated events
        if event.actor_id:
            anomaly_alert = self._anomaly_detector.analyze_user_behavior(
                event.actor_id, event
            )
            if anomaly_alert:
                alerts.append(anomaly_alert)

        # Store and notify for new alerts
        for alert in alerts:
            self._alerts[alert.id] = alert
            self._alert_counts[alert.severity.value] += 1
            self._notify_alert(alert)

        return alerts

    def _notify_alert(self, alert: SecurityAlert) -> None:
        """Notify handlers of new alert."""
        self._logger.warning(
            "Security alert",
            alert_id=alert.id,
            type=alert.threat_type.value,
            severity=alert.severity.value,
            title=alert.title,
        )

        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self._logger.error(f"Alert handler error: {e}")

    async def get_alert(self, alert_id: str) -> Optional[SecurityAlert]:
        """Get an alert by ID."""
        return self._alerts.get(alert_id)

    async def list_alerts(
        self,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        threat_type: Optional[ThreatType] = None,
        organization_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[SecurityAlert]:
        """List alerts with filtering."""
        alerts = list(self._alerts.values())

        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if threat_type:
            alerts = [a for a in alerts if a.threat_type == threat_type]
        if organization_id:
            alerts = [a for a in alerts if a.organization_id == organization_id]

        alerts.sort(key=lambda a: a.created_at, reverse=True)
        return alerts[:limit]

    async def acknowledge_alert(self, alert_id: str, user_id: str) -> Optional[SecurityAlert]:
        """Acknowledge an alert."""
        alert = self._alerts.get(alert_id)
        if alert:
            alert.acknowledge(user_id)
        return alert

    async def resolve_alert(
        self,
        alert_id: str,
        user_id: str,
        resolution_note: str = "",
    ) -> Optional[SecurityAlert]:
        """Resolve an alert."""
        alert = self._alerts.get(alert_id)
        if alert:
            alert.resolve(user_id, resolution_note)
        return alert

    async def get_metrics(self, window_minutes: int = 60) -> SecurityMetrics:
        """Get security metrics."""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)

        # Filter recent alerts
        recent_alerts = [
            a for a in self._alerts.values()
            if a.created_at > cutoff
        ]

        # Count by severity
        by_severity: Dict[str, int] = defaultdict(int)
        for alert in recent_alerts:
            by_severity[alert.severity.value] += 1

        # Count by type
        by_type: Dict[str, int] = defaultdict(int)
        for alert in recent_alerts:
            by_type[alert.threat_type.value] += 1

        # Top offending IPs
        ip_counts: Dict[str, int] = defaultdict(int)
        for alert in recent_alerts:
            if alert.actor_ip:
                ip_counts[alert.actor_ip] += 1
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Top targeted resources
        resource_counts: Dict[str, int] = defaultdict(int)
        for alert in recent_alerts:
            if alert.resource:
                resource_counts[alert.resource] += 1
        top_resources = sorted(resource_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return SecurityMetrics(
            time_window_minutes=window_minutes,
            login_attempts=self._event_counts.get("login_success", 0) + self._event_counts.get("login_failure", 0),
            failed_logins=self._event_counts.get("login_failure", 0),
            successful_logins=self._event_counts.get("login_success", 0),
            unique_ips=len(set(a.actor_ip for a in recent_alerts if a.actor_ip)),
            blocked_ips=len(self._threat_detector._blocked_ips),
            api_calls=self._event_counts.get("api_call", 0),
            api_errors=self._event_counts.get("api_error", 0),
            access_denied_count=self._event_counts.get("access_denied", 0),
            data_exports=self._event_counts.get("data_export", 0),
            alerts_generated=len(recent_alerts),
            alerts_by_severity=dict(by_severity),
            alerts_by_type=dict(by_type),
            top_offending_ips=top_ips,
            top_targeted_resources=top_resources,
        )

    async def _metrics_loop(self) -> None:
        """Background loop to collect system metrics."""
        while self._running:
            await asyncio.sleep(60)  # Every minute

            # Collect system metrics
            import os
            try:
                load_avg = os.getloadavg()[0]
            except (AttributeError, OSError):
                load_avg = 0

            metrics = {
                "alerts_per_minute": len([
                    a for a in self._alerts.values()
                    if (datetime.utcnow() - a.created_at).total_seconds() < 60
                ]),
                "blocked_ips": len(self._threat_detector._blocked_ips),
                "system_load": load_avg,
            }

            self._anomaly_detector.record_system_metrics(metrics)
