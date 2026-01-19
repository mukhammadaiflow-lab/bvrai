"""Alert system for call monitoring."""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import re

from app.monitoring.calls.tracker import (
    LiveCallTracker,
    CallEvent,
    CallInfo,
    CallState,
    get_live_call_tracker,
)

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    """Types of alerts."""
    LONG_WAIT = "long_wait"
    LONG_CALL = "long_call"
    NEGATIVE_SENTIMENT = "negative_sentiment"
    HIGH_VOLUME = "high_volume"
    AGENT_UNAVAILABLE = "agent_unavailable"
    FAILED_CALL = "failed_call"
    TRANSFER_REQUIRED = "transfer_required"
    ESCALATION = "escalation"
    SLA_BREACH = "sla_breach"
    CUSTOM = "custom"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"


@dataclass
class Alert:
    """A monitoring alert."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    call_id: Optional[str] = None
    agent_id: Optional[str] = None
    account_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "call_id": self.call_id,
            "agent_id": self.agent_id,
            "account_id": self.account_id,
            "metadata": self.metadata,
        }


@dataclass
class AlertRule:
    """A rule for triggering alerts."""
    rule_id: str
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: str  # Expression or identifier
    threshold: Optional[float] = None
    duration_seconds: Optional[int] = None
    cooldown_seconds: int = 300  # Prevent repeated alerts
    is_enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    account_id: Optional[str] = None  # If None, applies to all accounts
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "condition": self.condition,
            "threshold": self.threshold,
            "duration_seconds": self.duration_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "is_enabled": self.is_enabled,
            "notification_channels": self.notification_channels,
            "account_id": self.account_id,
            "metadata": self.metadata,
        }


class AlertManager:
    """
    Manages alerts for call monitoring.

    Usage:
        alert_mgr = AlertManager(tracker)

        # Add alert rule
        rule = AlertRule(
            rule_id="rule_1",
            name="Long Wait Alert",
            alert_type=AlertType.LONG_WAIT,
            severity=AlertSeverity.WARNING,
            condition="wait_time",
            threshold=30,  # 30 seconds
        )
        alert_mgr.add_rule(rule)

        # Start monitoring
        await alert_mgr.start()

        # Subscribe to alerts
        alert_mgr.on_alert(handle_alert)

        # Manually create alert
        alert = await alert_mgr.create_alert(
            AlertType.ESCALATION,
            AlertSeverity.CRITICAL,
            "Escalation Required",
            "Customer requested supervisor",
            call_id="call_123",
        )
    """

    def __init__(
        self,
        tracker: Optional[LiveCallTracker] = None,
        check_interval: float = 5.0,
    ):
        self.tracker = tracker or get_live_call_tracker()
        self.check_interval = check_interval

        self._alerts: Dict[str, Alert] = {}
        self._rules: Dict[str, AlertRule] = {}
        self._callbacks: List[Callable[[Alert], Awaitable[None]]] = []
        self._last_triggered: Dict[str, datetime] = {}  # For cooldown
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Set up event listener
        self.tracker.on_call_event(self._handle_call_event)

        # Add default rules
        self._add_default_rules()

    def _add_default_rules(self) -> None:
        """Add default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="default_long_wait",
                name="Long Wait Time",
                alert_type=AlertType.LONG_WAIT,
                severity=AlertSeverity.WARNING,
                condition="wait_time",
                threshold=30,
            ),
            AlertRule(
                rule_id="default_long_call",
                name="Long Call Duration",
                alert_type=AlertType.LONG_CALL,
                severity=AlertSeverity.INFO,
                condition="call_duration",
                threshold=600,  # 10 minutes
            ),
            AlertRule(
                rule_id="default_negative_sentiment",
                name="Negative Customer Sentiment",
                alert_type=AlertType.NEGATIVE_SENTIMENT,
                severity=AlertSeverity.WARNING,
                condition="sentiment",
                threshold=-0.5,
            ),
            AlertRule(
                rule_id="default_failed_call",
                name="Failed Call",
                alert_type=AlertType.FAILED_CALL,
                severity=AlertSeverity.WARNING,
                condition="call_failed",
            ),
        ]

        for rule in default_rules:
            self._rules[rule.rule_id] = rule

    async def start(self) -> None:
        """Start the alert monitoring."""
        self._running = True
        self._check_task = asyncio.create_task(self._check_loop())
        logger.info("Alert manager started")

    async def stop(self) -> None:
        """Stop the alert monitoring."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Alert manager stopped")

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._rules[rule.rule_id] = rule
        logger.info(f"Alert rule added: {rule.rule_id}")

    def remove_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Remove an alert rule."""
        return self._rules.pop(rule_id, None)

    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get an alert rule."""
        return self._rules.get(rule_id)

    def get_all_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        return list(self._rules.values())

    def enable_rule(self, rule_id: str) -> bool:
        """Enable an alert rule."""
        rule = self._rules.get(rule_id)
        if rule:
            rule.is_enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable an alert rule."""
        rule = self._rules.get(rule_id)
        if rule:
            rule.is_enabled = False
            return True
        return False

    async def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        call_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        account_id: Optional[str] = None,
        **metadata,
    ) -> Alert:
        """Create a new alert manually."""
        import uuid

        alert = Alert(
            alert_id=f"alert_{uuid.uuid4().hex[:12]}",
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            call_id=call_id,
            agent_id=agent_id,
            account_id=account_id,
            metadata=metadata,
        )

        async with self._lock:
            self._alerts[alert.alert_id] = alert

        await self._notify(alert)

        logger.info(f"Alert created: {alert.alert_id} - {alert.title}")

        return alert

    async def acknowledge(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> Optional[Alert]:
        """Acknowledge an alert."""
        async with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                return None

            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by

        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")

        return alert

    async def resolve(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_note: Optional[str] = None,
    ) -> Optional[Alert]:
        """Resolve an alert."""
        async with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                return None

            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = resolved_by
            if resolution_note:
                alert.metadata["resolution_note"] = resolution_note

        logger.info(f"Alert resolved: {alert_id} by {resolved_by}")

        return alert

    async def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get a specific alert."""
        async with self._lock:
            return self._alerts.get(alert_id)

    async def get_active_alerts(
        self,
        account_id: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get active alerts."""
        async with self._lock:
            alerts = [
                a for a in self._alerts.values()
                if a.status == AlertStatus.ACTIVE
            ]

        if account_id:
            alerts = [a for a in alerts if a.account_id == account_id]

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    async def get_all_alerts(
        self,
        include_resolved: bool = False,
        limit: int = 100,
    ) -> List[Alert]:
        """Get all alerts."""
        async with self._lock:
            if include_resolved:
                alerts = list(self._alerts.values())
            else:
                alerts = [
                    a for a in self._alerts.values()
                    if a.status != AlertStatus.RESOLVED
                ]

        # Sort by created_at descending
        alerts.sort(key=lambda a: a.created_at, reverse=True)

        return alerts[:limit]

    async def get_alerts_for_call(self, call_id: str) -> List[Alert]:
        """Get alerts for a specific call."""
        async with self._lock:
            return [
                a for a in self._alerts.values()
                if a.call_id == call_id
            ]

    async def cleanup_old_alerts(
        self,
        max_age_hours: int = 24,
    ) -> int:
        """Remove old resolved alerts."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        removed = 0

        async with self._lock:
            to_remove = [
                alert_id for alert_id, alert in self._alerts.items()
                if alert.status == AlertStatus.RESOLVED
                and alert.resolved_at
                and alert.resolved_at < cutoff
            ]

            for alert_id in to_remove:
                del self._alerts[alert_id]
                removed += 1

        if removed > 0:
            logger.info(f"Cleaned up {removed} old alerts")

        return removed

    def on_alert(
        self,
        callback: Callable[[Alert], Awaitable[None]],
    ) -> None:
        """Register callback for new alerts."""
        self._callbacks.append(callback)

    async def _notify(self, alert: Alert) -> None:
        """Notify callbacks of new alert."""
        for callback in self._callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    async def _check_loop(self) -> None:
        """Periodic alert check loop."""
        while self._running:
            try:
                await self._check_rules()
            except Exception as e:
                logger.error(f"Alert check error: {e}")

            await asyncio.sleep(self.check_interval)

    async def _check_rules(self) -> None:
        """Check all alert rules against current state."""
        calls = await self.tracker.get_active_calls()

        for rule in self._rules.values():
            if not rule.is_enabled:
                continue

            # Check cooldown
            last_triggered = self._last_triggered.get(rule.rule_id)
            if last_triggered:
                if datetime.utcnow() - last_triggered < timedelta(seconds=rule.cooldown_seconds):
                    continue

            # Apply rule filter
            if rule.account_id:
                rule_calls = [c for c in calls if c.account_id == rule.account_id]
            else:
                rule_calls = calls

            # Check condition
            await self._evaluate_rule(rule, rule_calls)

    async def _evaluate_rule(
        self,
        rule: AlertRule,
        calls: List[CallInfo],
    ) -> None:
        """Evaluate a single rule."""
        if rule.condition == "wait_time":
            await self._check_wait_time(rule, calls)
        elif rule.condition == "call_duration":
            await self._check_call_duration(rule, calls)
        elif rule.condition == "sentiment":
            await self._check_sentiment(rule, calls)
        elif rule.condition == "high_volume":
            await self._check_volume(rule, calls)
        # Add more condition handlers as needed

    async def _check_wait_time(
        self,
        rule: AlertRule,
        calls: List[CallInfo],
    ) -> None:
        """Check for long wait times."""
        threshold = rule.threshold or 30

        for call in calls:
            if not call.answered_at and call.state == CallState.RINGING:
                wait_time = (datetime.utcnow() - call.started_at).total_seconds()

                if wait_time >= threshold:
                    await self._trigger_alert(
                        rule,
                        f"Call {call.call_id} waiting for {wait_time:.0f}s",
                        call_id=call.call_id,
                        agent_id=call.agent_id,
                        account_id=call.account_id,
                        wait_time=wait_time,
                    )

    async def _check_call_duration(
        self,
        rule: AlertRule,
        calls: List[CallInfo],
    ) -> None:
        """Check for long call durations."""
        threshold = rule.threshold or 600

        for call in calls:
            if call.duration_seconds >= threshold:
                await self._trigger_alert(
                    rule,
                    f"Call {call.call_id} duration is {call.duration_seconds:.0f}s",
                    call_id=call.call_id,
                    agent_id=call.agent_id,
                    account_id=call.account_id,
                    duration=call.duration_seconds,
                )

    async def _check_sentiment(
        self,
        rule: AlertRule,
        calls: List[CallInfo],
    ) -> None:
        """Check for negative sentiment."""
        threshold = rule.threshold or -0.5

        for call in calls:
            if call.sentiment_score <= threshold:
                await self._trigger_alert(
                    rule,
                    f"Negative sentiment detected on call {call.call_id}",
                    call_id=call.call_id,
                    agent_id=call.agent_id,
                    account_id=call.account_id,
                    sentiment_score=call.sentiment_score,
                )

    async def _check_volume(
        self,
        rule: AlertRule,
        calls: List[CallInfo],
    ) -> None:
        """Check for high call volume."""
        threshold = rule.threshold or 100

        if len(calls) >= threshold:
            await self._trigger_alert(
                rule,
                f"High call volume: {len(calls)} active calls",
                active_call_count=len(calls),
            )

    async def _trigger_alert(
        self,
        rule: AlertRule,
        message: str,
        call_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        account_id: Optional[str] = None,
        **metadata,
    ) -> None:
        """Trigger an alert from a rule."""
        # Check if similar alert already exists
        existing = await self._find_similar_alert(rule.alert_type, call_id)
        if existing:
            return

        await self.create_alert(
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=rule.name,
            message=message,
            call_id=call_id,
            agent_id=agent_id,
            account_id=account_id or rule.account_id,
            rule_id=rule.rule_id,
            **metadata,
        )

        self._last_triggered[rule.rule_id] = datetime.utcnow()

    async def _find_similar_alert(
        self,
        alert_type: AlertType,
        call_id: Optional[str],
    ) -> Optional[Alert]:
        """Find a similar active alert."""
        async with self._lock:
            for alert in self._alerts.values():
                if alert.status == AlertStatus.ACTIVE:
                    if alert.alert_type == alert_type:
                        if call_id and alert.call_id == call_id:
                            return alert
        return None

    async def _handle_call_event(self, event: CallEvent) -> None:
        """Handle call events for alert rules."""
        # Handle failed calls
        if event.event_type == "call.state_changed":
            new_state = event.data.get("new_state")

            if new_state == "failed":
                # Check if there's a failed call rule
                for rule in self._rules.values():
                    if rule.is_enabled and rule.condition == "call_failed":
                        await self._trigger_alert(
                            rule,
                            f"Call {event.call_id} failed",
                            call_id=event.call_id,
                            failure_data=event.data,
                        )

        # Handle sentiment updates
        elif event.event_type == "call.sentiment_updated":
            sentiment = event.data.get("sentiment_score", 0)

            for rule in self._rules.values():
                if rule.is_enabled and rule.condition == "sentiment":
                    threshold = rule.threshold or -0.5
                    if sentiment <= threshold:
                        await self._trigger_alert(
                            rule,
                            f"Negative sentiment on call {event.call_id}: {sentiment:.2f}",
                            call_id=event.call_id,
                            sentiment_score=sentiment,
                        )


class NotificationChannel:
    """Base class for alert notification channels."""

    async def send(self, alert: Alert) -> bool:
        """Send alert notification."""
        raise NotImplementedError


class WebhookNotificationChannel(NotificationChannel):
    """Send alerts via webhook."""

    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}

    async def send(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=alert.to_dict(),
                    headers=self.headers,
                ) as response:
                    return response.status in [200, 201, 202]
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False


class EmailNotificationChannel(NotificationChannel):
    """Send alerts via email."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_address: str,
        to_addresses: List[str],
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_address = from_address
        self.to_addresses = to_addresses

    async def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        # In a real implementation, this would send an email
        logger.info(f"Email alert would be sent to {self.to_addresses}: {alert.title}")
        return True


class SlackNotificationChannel(NotificationChannel):
    """Send alerts to Slack."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        import aiohttp

        severity_emoji = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.CRITICAL: ":rotating_light:",
            AlertSeverity.EMERGENCY: ":fire:",
        }

        emoji = severity_emoji.get(alert.severity, ":bell:")

        payload = {
            "text": f"{emoji} *{alert.title}*",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"{emoji} {alert.title}"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": alert.message}
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"*Severity:* {alert.severity.value}"},
                        {"type": "mrkdwn", "text": f"*Type:* {alert.alert_type.value}"},
                    ]
                }
            ]
        }

        if alert.call_id:
            payload["blocks"][2]["elements"].append(
                {"type": "mrkdwn", "text": f"*Call:* {alert.call_id}"}
            )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False
