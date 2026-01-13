"""Alert management system for call monitoring."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import hashlib

import redis.asyncio as redis

logger = logging.getLogger(__name__)


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
    SILENCED = "silenced"


class AlertCondition(str, Enum):
    """Types of alert conditions."""
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    THRESHOLD_BELOW = "threshold_below"
    RATE_CHANGE = "rate_change"
    ANOMALY = "anomaly"
    ABSENCE = "absence"
    PATTERN = "pattern"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    id: str
    name: str
    description: str
    metric_name: str
    condition: AlertCondition
    threshold: float
    comparison_period: int  # seconds
    severity: AlertSeverity
    organization_id: str
    labels: Dict[str, str] = field(default_factory=dict)
    cooldown_period: int = 300  # 5 minutes default
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    escalation_policy: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "condition": self.condition.value,
            "severity": self.severity.value,
        }


@dataclass
class Alert:
    """An active or historical alert."""
    id: str
    rule_id: str
    rule_name: str
    organization_id: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "severity": self.severity.value,
            "status": self.status.value,
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


class NotificationChannel:
    """Base class for notification channels."""

    async def send(self, alert: Alert, rule: AlertRule) -> bool:
        """Send notification for an alert."""
        raise NotImplementedError


class WebhookNotificationChannel(NotificationChannel):
    """Send alerts via webhook."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {}

    async def send(self, alert: Alert, rule: AlertRule) -> bool:
        """Send webhook notification."""
        import httpx

        payload = {
            "alert": alert.to_dict(),
            "rule": rule.to_dict(),
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url,
                    json=payload,
                    headers=self.headers,
                    timeout=10.0,
                )
                return response.status_code < 400
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """Send alerts to Slack."""

    def __init__(self, webhook_url: str, channel: Optional[str] = None):
        self.webhook_url = webhook_url
        self.channel = channel

    async def send(self, alert: Alert, rule: AlertRule) -> bool:
        """Send Slack notification."""
        import httpx

        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9800",
            AlertSeverity.CRITICAL: "#f44336",
            AlertSeverity.EMERGENCY: "#9c27b0",
        }.get(alert.severity, "#808080")

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"🚨 {alert.severity.value.upper()}: {alert.rule_name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Value", "value": f"{alert.metric_value:.2f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold:.2f}", "short": True},
                        {"title": "Status", "value": alert.status.value, "short": True},
                    ],
                    "footer": f"Alert ID: {alert.id}",
                    "ts": int(alert.triggered_at.timestamp()),
                }
            ]
        }

        if self.channel:
            payload["channel"] = self.channel

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
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

    async def send(self, alert: Alert, rule: AlertRule) -> bool:
        """Send email notification."""
        import aiosmtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        subject = f"[{alert.severity.value.upper()}] {alert.rule_name}"

        html = f"""
        <html>
        <body>
            <h2 style="color: {'red' if alert.severity == AlertSeverity.CRITICAL else 'orange'}">
                🚨 {alert.rule_name}
            </h2>
            <p>{alert.message}</p>
            <table border="1" cellpadding="5">
                <tr><td><b>Metric</b></td><td>{alert.metric_name}</td></tr>
                <tr><td><b>Value</b></td><td>{alert.metric_value:.2f}</td></tr>
                <tr><td><b>Threshold</b></td><td>{alert.threshold:.2f}</td></tr>
                <tr><td><b>Severity</b></td><td>{alert.severity.value}</td></tr>
                <tr><td><b>Status</b></td><td>{alert.status.value}</td></tr>
                <tr><td><b>Triggered At</b></td><td>{alert.triggered_at.isoformat()}</td></tr>
            </table>
            <p><small>Alert ID: {alert.id}</small></p>
        </body>
        </html>
        """

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = self.from_address
        message["To"] = ", ".join(self.to_addresses)
        message.attach(MIMEText(html, "html"))

        try:
            await aiosmtplib.send(
                message,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.username,
                password=self.password,
                use_tls=True,
            )
            return True
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False


class AlertManager:
    """Manages alert rules, evaluation, and notifications."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        evaluation_interval: int = 30,
    ):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.evaluation_interval = evaluation_interval

        # Rule storage
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_cooldowns: Dict[str, datetime] = {}
        self._silenced_rules: Set[str] = set()
        self._lock = asyncio.Lock()

        # Notification channels
        self._channels: Dict[str, NotificationChannel] = {}

        # Metric provider callback
        self._get_metric: Optional[Callable[[str, Dict[str, str]], asyncio.coroutine]] = None

        # Event callbacks
        self._on_alert_triggered: List[Callable[[Alert], asyncio.coroutine]] = []
        self._on_alert_resolved: List[Callable[[Alert], asyncio.coroutine]] = []

        # Background tasks
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start the alert manager."""
        self.redis = redis.from_url(self.redis_url, decode_responses=True)

        # Load rules from Redis
        await self._load_rules()

        # Start evaluation loop
        self._tasks.append(asyncio.create_task(self._evaluation_loop()))
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))

        logger.info("Alert manager started")

    async def stop(self) -> None:
        """Stop the alert manager."""
        for task in self._tasks:
            task.cancel()

        if self.redis:
            await self.redis.close()

        logger.info("Alert manager stopped")

    def set_metric_provider(
        self,
        provider: Callable[[str, Dict[str, str]], asyncio.coroutine],
    ) -> None:
        """Set the callback for getting metric values."""
        self._get_metric = provider

    def register_channel(self, name: str, channel: NotificationChannel) -> None:
        """Register a notification channel."""
        self._channels[name] = channel
        logger.info(f"Registered notification channel: {name}")

    def on_alert_triggered(self, callback: Callable[[Alert], asyncio.coroutine]) -> None:
        """Register callback for when an alert is triggered."""
        self._on_alert_triggered.append(callback)

    def on_alert_resolved(self, callback: Callable[[Alert], asyncio.coroutine]) -> None:
        """Register callback for when an alert is resolved."""
        self._on_alert_resolved.append(callback)

    # Rule management

    async def create_rule(self, rule: AlertRule) -> AlertRule:
        """Create a new alert rule."""
        async with self._lock:
            self._rules[rule.id] = rule

        # Persist to Redis
        if self.redis:
            await self.redis.hset(
                f"alert_rules:{rule.organization_id}",
                rule.id,
                json.dumps(rule.to_dict()),
            )

        logger.info(f"Created alert rule: {rule.name} ({rule.id})")
        return rule

    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> Optional[AlertRule]:
        """Update an existing alert rule."""
        async with self._lock:
            if rule_id not in self._rules:
                return None

            rule = self._rules[rule_id]
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)

            self._rules[rule_id] = rule

        # Persist to Redis
        if self.redis:
            await self.redis.hset(
                f"alert_rules:{rule.organization_id}",
                rule.id,
                json.dumps(rule.to_dict()),
            )

        logger.info(f"Updated alert rule: {rule.name}")
        return rule

    async def delete_rule(self, rule_id: str) -> bool:
        """Delete an alert rule."""
        async with self._lock:
            if rule_id not in self._rules:
                return False

            rule = self._rules.pop(rule_id)

        # Remove from Redis
        if self.redis:
            await self.redis.hdel(f"alert_rules:{rule.organization_id}", rule_id)

        logger.info(f"Deleted alert rule: {rule.name}")
        return True

    async def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get an alert rule by ID."""
        async with self._lock:
            return self._rules.get(rule_id)

    async def list_rules(
        self,
        organization_id: Optional[str] = None,
    ) -> List[AlertRule]:
        """List alert rules."""
        async with self._lock:
            rules = list(self._rules.values())

        if organization_id:
            rules = [r for r in rules if r.organization_id == organization_id]

        return rules

    # Alert management

    async def get_active_alerts(
        self,
        organization_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get active alerts."""
        async with self._lock:
            alerts = list(self._active_alerts.values())

        if organization_id:
            alerts = [a for a in alerts if a.organization_id == organization_id]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)

    async def acknowledge_alert(
        self,
        alert_id: str,
        user_id: str,
    ) -> Optional[Alert]:
        """Acknowledge an active alert."""
        async with self._lock:
            if alert_id not in self._active_alerts:
                return None

            alert = self._active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = user_id

        logger.info(f"Alert acknowledged: {alert_id} by {user_id}")
        return alert

    async def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        """Manually resolve an alert."""
        async with self._lock:
            if alert_id not in self._active_alerts:
                return None

            alert = self._active_alerts.pop(alert_id)
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()

        # Store in history
        await self._store_alert_history(alert)

        # Notify callbacks
        for callback in self._on_alert_resolved:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert resolved callback error: {e}")

        logger.info(f"Alert resolved: {alert_id}")
        return alert

    async def silence_rule(
        self,
        rule_id: str,
        duration_seconds: int,
    ) -> bool:
        """Silence an alert rule for a period."""
        async with self._lock:
            if rule_id not in self._rules:
                return False

            self._silenced_rules.add(rule_id)

        # Schedule unsilence
        asyncio.create_task(self._unsilence_after(rule_id, duration_seconds))

        logger.info(f"Silenced rule {rule_id} for {duration_seconds}s")
        return True

    async def _unsilence_after(self, rule_id: str, seconds: int) -> None:
        """Unsilence a rule after a delay."""
        await asyncio.sleep(seconds)

        async with self._lock:
            self._silenced_rules.discard(rule_id)

        logger.info(f"Unsilenced rule {rule_id}")

    async def get_alert_history(
        self,
        organization_id: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Alert]:
        """Get alert history."""
        if not self.redis:
            return []

        # Query from Redis sorted set
        key = f"alert_history:{organization_id}"
        end_time = end_time or datetime.utcnow()
        start_time = start_time or (end_time - timedelta(days=7))

        results = await self.redis.zrangebyscore(
            key,
            start_time.timestamp(),
            end_time.timestamp(),
            start=0,
            num=limit,
        )

        alerts = []
        for data in results:
            try:
                alert_dict = json.loads(data)
                alert_dict["severity"] = AlertSeverity(alert_dict["severity"])
                alert_dict["status"] = AlertStatus(alert_dict["status"])
                alert_dict["triggered_at"] = datetime.fromisoformat(alert_dict["triggered_at"])
                if alert_dict.get("resolved_at"):
                    alert_dict["resolved_at"] = datetime.fromisoformat(alert_dict["resolved_at"])
                if alert_dict.get("acknowledged_at"):
                    alert_dict["acknowledged_at"] = datetime.fromisoformat(alert_dict["acknowledged_at"])
                alerts.append(Alert(**alert_dict))
            except Exception as e:
                logger.warning(f"Failed to parse alert: {e}")

        return alerts

    # Internal methods

    async def _load_rules(self) -> None:
        """Load rules from Redis."""
        if not self.redis:
            return

        # Get all organization rule hashes
        async for key in self.redis.scan_iter(match="alert_rules:*"):
            rules_data = await self.redis.hgetall(key)
            for rule_id, data in rules_data.items():
                try:
                    rule_dict = json.loads(data)
                    rule_dict["condition"] = AlertCondition(rule_dict["condition"])
                    rule_dict["severity"] = AlertSeverity(rule_dict["severity"])
                    rule = AlertRule(**rule_dict)

                    async with self._lock:
                        self._rules[rule.id] = rule
                except Exception as e:
                    logger.warning(f"Failed to load rule {rule_id}: {e}")

        logger.info(f"Loaded {len(self._rules)} alert rules")

    async def _evaluation_loop(self) -> None:
        """Periodically evaluate alert rules."""
        while True:
            try:
                await asyncio.sleep(self.evaluation_interval)
                await self._evaluate_all_rules()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Evaluation error: {e}")

    async def _evaluate_all_rules(self) -> None:
        """Evaluate all active alert rules."""
        async with self._lock:
            rules = [r for r in self._rules.values() if r.enabled]
            silenced = set(self._silenced_rules)

        for rule in rules:
            if rule.id in silenced:
                continue

            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.id}: {e}")

    async def _evaluate_rule(self, rule: AlertRule) -> None:
        """Evaluate a single alert rule."""
        if not self._get_metric:
            return

        # Check cooldown
        cooldown_key = f"{rule.id}:{rule.metric_name}"
        if cooldown_key in self._alert_cooldowns:
            if datetime.utcnow() < self._alert_cooldowns[cooldown_key]:
                return

        # Get metric value
        try:
            value = await self._get_metric(rule.metric_name, rule.labels)
        except Exception as e:
            logger.warning(f"Failed to get metric {rule.metric_name}: {e}")
            return

        if value is None:
            return

        # Check condition
        triggered = False

        if rule.condition == AlertCondition.THRESHOLD_EXCEEDED:
            triggered = value > rule.threshold
        elif rule.condition == AlertCondition.THRESHOLD_BELOW:
            triggered = value < rule.threshold
        elif rule.condition == AlertCondition.RATE_CHANGE:
            # Compare with previous value (stored in Redis)
            triggered = await self._check_rate_change(rule, value)

        if triggered:
            await self._trigger_alert(rule, value)
        else:
            # Check if we can resolve an existing alert
            await self._check_resolve_alert(rule)

    async def _trigger_alert(self, rule: AlertRule, value: float) -> None:
        """Trigger an alert."""
        # Check if alert already exists
        alert_key = self._make_alert_key(rule)

        async with self._lock:
            if alert_key in self._active_alerts:
                return  # Alert already active

        # Create alert
        alert = Alert(
            id=hashlib.sha256(f"{alert_key}:{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16],
            rule_id=rule.id,
            rule_name=rule.name,
            organization_id=rule.organization_id,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=f"{rule.name}: {rule.metric_name} is {value:.2f} (threshold: {rule.threshold})",
            metric_name=rule.metric_name,
            metric_value=value,
            threshold=rule.threshold,
            triggered_at=datetime.utcnow(),
            labels=rule.labels,
        )

        async with self._lock:
            self._active_alerts[alert_key] = alert
            self._alert_cooldowns[f"{rule.id}:{rule.metric_name}"] = (
                datetime.utcnow() + timedelta(seconds=rule.cooldown_period)
            )

        # Send notifications
        await self._send_notifications(alert, rule)

        # Notify callbacks
        for callback in self._on_alert_triggered:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert triggered callback error: {e}")

        logger.info(f"Alert triggered: {rule.name} - value={value}, threshold={rule.threshold}")

    async def _check_resolve_alert(self, rule: AlertRule) -> None:
        """Check if an existing alert can be resolved."""
        alert_key = self._make_alert_key(rule)

        async with self._lock:
            if alert_key not in self._active_alerts:
                return

            alert = self._active_alerts.pop(alert_key)
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()

        # Store in history
        await self._store_alert_history(alert)

        # Notify callbacks
        for callback in self._on_alert_resolved:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert resolved callback error: {e}")

        logger.info(f"Alert auto-resolved: {rule.name}")

    async def _check_rate_change(self, rule: AlertRule, current_value: float) -> bool:
        """Check if metric rate of change exceeds threshold."""
        if not self.redis:
            return False

        key = f"metric_history:{rule.metric_name}:{json.dumps(rule.labels)}"

        # Get previous value
        prev = await self.redis.get(key)

        # Store current value
        await self.redis.set(key, str(current_value), ex=rule.comparison_period * 2)

        if prev is None:
            return False

        prev_value = float(prev)
        if prev_value == 0:
            return False

        rate_change = abs((current_value - prev_value) / prev_value) * 100
        return rate_change > rule.threshold

    def _make_alert_key(self, rule: AlertRule) -> str:
        """Create a unique key for an alert."""
        labels_str = json.dumps(rule.labels, sort_keys=True)
        return f"{rule.id}:{rule.metric_name}:{labels_str}"

    async def _send_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send notifications for an alert."""
        for channel_name in rule.notification_channels:
            channel = self._channels.get(channel_name)
            if channel:
                try:
                    success = await channel.send(alert, rule)
                    if success:
                        logger.info(f"Notification sent via {channel_name}")
                    else:
                        logger.warning(f"Notification failed via {channel_name}")
                except Exception as e:
                    logger.error(f"Notification error ({channel_name}): {e}")

    async def _store_alert_history(self, alert: Alert) -> None:
        """Store alert in history."""
        if not self.redis:
            return

        key = f"alert_history:{alert.organization_id}"
        await self.redis.zadd(
            key,
            {json.dumps(alert.to_dict()): alert.triggered_at.timestamp()},
        )

        # Trim to last 30 days
        cutoff = datetime.utcnow() - timedelta(days=30)
        await self.redis.zremrangebyscore(key, 0, cutoff.timestamp())

    async def _cleanup_loop(self) -> None:
        """Periodically clean up old data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Hourly
                await self._cleanup_expired_cooldowns()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _cleanup_expired_cooldowns(self) -> None:
        """Remove expired cooldowns."""
        now = datetime.utcnow()

        async with self._lock:
            expired = [k for k, v in self._alert_cooldowns.items() if v < now]
            for key in expired:
                del self._alert_cooldowns[key]

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired cooldowns")
