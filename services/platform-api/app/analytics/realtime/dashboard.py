"""Real-time analytics dashboard data."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json


logger = structlog.get_logger()


@dataclass
class LiveCall:
    """Information about a live call."""
    call_id: str
    agent_id: str
    agent_name: str
    direction: str
    started_at: datetime
    caller_number: Optional[str] = None
    current_state: str = "connected"
    current_speaker: Optional[str] = None
    turn_count: int = 0
    last_transcript: Optional[str] = None


@dataclass
class DashboardSnapshot:
    """Point-in-time dashboard data."""
    timestamp: datetime
    active_calls: int = 0
    calls_today: int = 0
    avg_duration_today: float = 0.0
    success_rate_today: float = 0.0
    calls_per_hour: List[int] = field(default_factory=list)
    active_agents: int = 0
    queue_size: int = 0
    avg_wait_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "active_calls": self.active_calls,
            "calls_today": self.calls_today,
            "avg_duration_today": round(self.avg_duration_today, 1),
            "success_rate_today": round(self.success_rate_today * 100, 1),
            "calls_per_hour": self.calls_per_hour,
            "active_agents": self.active_agents,
            "queue_size": self.queue_size,
            "avg_wait_time": round(self.avg_wait_time, 1),
        }


class RealTimeDashboard:
    """
    Real-time analytics dashboard.

    Provides:
    - Live call tracking
    - Real-time metrics
    - WebSocket streaming
    - Auto-refresh data
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client

        # Live call tracking
        self._live_calls: Dict[str, LiveCall] = {}

        # Recent metrics (sliding window)
        self._recent_calls: deque = deque(maxlen=1000)
        self._recent_durations: deque = deque(maxlen=100)

        # WebSocket subscribers
        self._subscribers: Set[Any] = set()

        # State
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

        # Cached snapshot
        self._last_snapshot: Optional[DashboardSnapshot] = None
        self._snapshot_interval = 1.0  # seconds

    async def start(self) -> None:
        """Start the dashboard service."""
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("realtime_dashboard_started")

    async def stop(self) -> None:
        """Stop the dashboard service."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("realtime_dashboard_stopped")

    async def call_started(
        self,
        call_id: str,
        agent_id: str,
        agent_name: str,
        direction: str,
        caller_number: Optional[str] = None,
    ) -> None:
        """Track a new call."""
        call = LiveCall(
            call_id=call_id,
            agent_id=agent_id,
            agent_name=agent_name,
            direction=direction,
            started_at=datetime.utcnow(),
            caller_number=caller_number,
        )

        self._live_calls[call_id] = call
        self._recent_calls.append({
            "call_id": call_id,
            "timestamp": datetime.utcnow(),
            "event": "started",
        })

        await self._broadcast_update({
            "type": "call_started",
            "call": self._call_to_dict(call),
        })

    async def call_updated(
        self,
        call_id: str,
        state: Optional[str] = None,
        speaker: Optional[str] = None,
        transcript: Optional[str] = None,
    ) -> None:
        """Update live call information."""
        call = self._live_calls.get(call_id)
        if not call:
            return

        if state:
            call.current_state = state
        if speaker:
            call.current_speaker = speaker
        if transcript:
            call.last_transcript = transcript
            call.turn_count += 1

        await self._broadcast_update({
            "type": "call_updated",
            "call_id": call_id,
            "state": call.current_state,
            "speaker": call.current_speaker,
            "turn_count": call.turn_count,
        })

    async def call_ended(
        self,
        call_id: str,
        duration_seconds: float,
        success: bool,
        end_reason: str,
    ) -> None:
        """Handle call end."""
        call = self._live_calls.pop(call_id, None)

        self._recent_calls.append({
            "call_id": call_id,
            "timestamp": datetime.utcnow(),
            "event": "ended",
            "duration": duration_seconds,
            "success": success,
            "reason": end_reason,
        })

        self._recent_durations.append(duration_seconds)

        await self._broadcast_update({
            "type": "call_ended",
            "call_id": call_id,
            "duration_seconds": duration_seconds,
            "success": success,
            "end_reason": end_reason,
        })

    async def get_snapshot(self) -> DashboardSnapshot:
        """Get current dashboard snapshot."""
        if self._last_snapshot:
            return self._last_snapshot

        return await self._compute_snapshot()

    async def get_live_calls(self) -> List[Dict[str, Any]]:
        """Get all live calls."""
        return [
            self._call_to_dict(call)
            for call in self._live_calls.values()
        ]

    async def get_live_call(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific live call."""
        call = self._live_calls.get(call_id)
        if call:
            return self._call_to_dict(call)
        return None

    def subscribe(self, websocket: Any) -> None:
        """Subscribe to real-time updates."""
        self._subscribers.add(websocket)
        logger.debug("dashboard_subscriber_added", count=len(self._subscribers))

    def unsubscribe(self, websocket: Any) -> None:
        """Unsubscribe from updates."""
        self._subscribers.discard(websocket)
        logger.debug("dashboard_subscriber_removed", count=len(self._subscribers))

    async def _update_loop(self) -> None:
        """Periodic snapshot update."""
        while self._running:
            try:
                await asyncio.sleep(self._snapshot_interval)
                self._last_snapshot = await self._compute_snapshot()

                # Broadcast snapshot to subscribers
                await self._broadcast_update({
                    "type": "snapshot",
                    "data": self._last_snapshot.to_dict(),
                })

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("dashboard_update_error", error=str(e))

    async def _compute_snapshot(self) -> DashboardSnapshot:
        """Compute current dashboard snapshot."""
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Count today's calls
        today_calls = [
            c for c in self._recent_calls
            if c["timestamp"] >= today_start
        ]

        ended_today = [c for c in today_calls if c["event"] == "ended"]
        successful_today = [c for c in ended_today if c.get("success", False)]

        # Calculate calls per hour
        calls_per_hour = [0] * 24
        for call in today_calls:
            if call["event"] == "started":
                hour = call["timestamp"].hour
                calls_per_hour[hour] += 1

        # Calculate averages
        durations = [c.get("duration", 0) for c in ended_today if c.get("duration")]
        avg_duration = sum(durations) / len(durations) if durations else 0

        success_rate = len(successful_today) / len(ended_today) if ended_today else 0

        # Count active agents
        active_agents = len(set(c.agent_id for c in self._live_calls.values()))

        return DashboardSnapshot(
            timestamp=now,
            active_calls=len(self._live_calls),
            calls_today=len([c for c in today_calls if c["event"] == "started"]),
            avg_duration_today=avg_duration,
            success_rate_today=success_rate,
            calls_per_hour=calls_per_hour[:now.hour + 1],
            active_agents=active_agents,
            queue_size=0,  # Would come from telephony
            avg_wait_time=0,  # Would come from telephony
        )

    async def _broadcast_update(self, data: Dict[str, Any]) -> None:
        """Broadcast update to all subscribers."""
        if not self._subscribers:
            return

        message = json.dumps(data)
        dead_sockets = []

        for ws in self._subscribers:
            try:
                await ws.send_text(message)
            except Exception:
                dead_sockets.append(ws)

        # Clean up dead connections
        for ws in dead_sockets:
            self._subscribers.discard(ws)

    def _call_to_dict(self, call: LiveCall) -> Dict[str, Any]:
        """Convert LiveCall to dictionary."""
        duration = (datetime.utcnow() - call.started_at).total_seconds()

        return {
            "call_id": call.call_id,
            "agent_id": call.agent_id,
            "agent_name": call.agent_name,
            "direction": call.direction,
            "started_at": call.started_at.isoformat(),
            "duration_seconds": round(duration, 1),
            "caller_number": call.caller_number,
            "current_state": call.current_state,
            "current_speaker": call.current_speaker,
            "turn_count": call.turn_count,
            "last_transcript": call.last_transcript,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        return {
            "live_calls": len(self._live_calls),
            "subscribers": len(self._subscribers),
            "recent_calls_tracked": len(self._recent_calls),
        }


class AlertManager:
    """
    Manages real-time alerts.

    Triggers alerts based on:
    - Thresholds (high queue, low success rate)
    - Anomalies
    - System events
    """

    def __init__(self):
        self._rules: List[Dict[str, Any]] = []
        self._active_alerts: Dict[str, Dict[str, Any]] = {}
        self._alert_history: deque = deque(maxlen=100)

    def add_rule(
        self,
        rule_id: str,
        metric: str,
        condition: str,  # "gt", "lt", "eq"
        threshold: float,
        severity: str = "warning",
        cooldown_seconds: int = 300,
    ) -> None:
        """Add an alert rule."""
        self._rules.append({
            "rule_id": rule_id,
            "metric": metric,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "cooldown_seconds": cooldown_seconds,
            "last_triggered": None,
        })

    async def check_metrics(
        self,
        metrics: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Check metrics against rules."""
        now = datetime.utcnow()
        triggered = []

        for rule in self._rules:
            metric_value = metrics.get(rule["metric"])
            if metric_value is None:
                continue

            # Check cooldown
            if rule["last_triggered"]:
                elapsed = (now - rule["last_triggered"]).total_seconds()
                if elapsed < rule["cooldown_seconds"]:
                    continue

            # Check condition
            should_alert = False
            if rule["condition"] == "gt" and metric_value > rule["threshold"]:
                should_alert = True
            elif rule["condition"] == "lt" and metric_value < rule["threshold"]:
                should_alert = True
            elif rule["condition"] == "eq" and metric_value == rule["threshold"]:
                should_alert = True

            if should_alert:
                alert = {
                    "alert_id": f"{rule['rule_id']}-{now.timestamp()}",
                    "rule_id": rule["rule_id"],
                    "metric": rule["metric"],
                    "value": metric_value,
                    "threshold": rule["threshold"],
                    "severity": rule["severity"],
                    "triggered_at": now.isoformat(),
                }

                rule["last_triggered"] = now
                self._active_alerts[alert["alert_id"]] = alert
                self._alert_history.append(alert)
                triggered.append(alert)

        return triggered

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id]["acknowledged"] = True
            self._active_alerts[alert_id]["acknowledged_at"] = datetime.utcnow().isoformat()
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        alert = self._active_alerts.pop(alert_id, None)
        if alert:
            alert["resolved_at"] = datetime.utcnow().isoformat()
            return True
        return False

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return list(self._active_alerts.values())

    def get_alert_history(
        self,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        return list(self._alert_history)[-limit:]


# Global instances
realtime_dashboard = RealTimeDashboard()
alert_manager = AlertManager()
