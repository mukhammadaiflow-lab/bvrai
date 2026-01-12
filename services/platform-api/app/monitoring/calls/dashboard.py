"""Real-time call monitoring dashboard data."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import logging

from app.monitoring.calls.tracker import (
    LiveCallTracker,
    CallInfo,
    CallState,
    CallDirection,
    get_live_call_tracker,
)

logger = logging.getLogger(__name__)


@dataclass
class CallMetrics:
    """Aggregated call metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Volume metrics
    total_calls: int = 0
    active_calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0
    inbound_calls: int = 0
    outbound_calls: int = 0

    # Duration metrics
    total_duration_seconds: float = 0.0
    total_talk_time_seconds: float = 0.0
    average_duration_seconds: float = 0.0
    average_talk_time_seconds: float = 0.0
    average_wait_time_seconds: float = 0.0

    # Performance metrics
    answer_rate: float = 0.0
    abandonment_rate: float = 0.0
    transfer_rate: float = 0.0
    supervised_rate: float = 0.0

    # Sentiment metrics
    average_sentiment: float = 0.0
    positive_sentiment_count: int = 0
    negative_sentiment_count: int = 0
    neutral_sentiment_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "volume": {
                "total": self.total_calls,
                "active": self.active_calls,
                "completed": self.completed_calls,
                "failed": self.failed_calls,
                "inbound": self.inbound_calls,
                "outbound": self.outbound_calls,
            },
            "duration": {
                "total_seconds": self.total_duration_seconds,
                "total_talk_seconds": self.total_talk_time_seconds,
                "average_seconds": self.average_duration_seconds,
                "average_talk_seconds": self.average_talk_time_seconds,
                "average_wait_seconds": self.average_wait_time_seconds,
            },
            "performance": {
                "answer_rate": self.answer_rate,
                "abandonment_rate": self.abandonment_rate,
                "transfer_rate": self.transfer_rate,
                "supervised_rate": self.supervised_rate,
            },
            "sentiment": {
                "average": self.average_sentiment,
                "positive_count": self.positive_sentiment_count,
                "negative_count": self.negative_sentiment_count,
                "neutral_count": self.neutral_sentiment_count,
            },
        }


@dataclass
class AgentMetrics:
    """Metrics for a specific agent."""
    agent_id: str
    agent_name: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Status
    is_available: bool = True
    current_call_id: Optional[str] = None
    status: str = "available"  # "available", "on_call", "busy", "offline"

    # Volume
    total_calls: int = 0
    active_calls: int = 0
    completed_calls: int = 0
    missed_calls: int = 0

    # Duration
    total_talk_time_seconds: float = 0.0
    average_talk_time_seconds: float = 0.0
    longest_call_seconds: float = 0.0

    # Performance
    average_sentiment: float = 0.0
    resolution_rate: float = 0.0
    transfer_rate: float = 0.0

    # Time utilization
    utilization_percent: float = 0.0
    idle_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "status": {
                "is_available": self.is_available,
                "current_call_id": self.current_call_id,
                "status": self.status,
            },
            "volume": {
                "total": self.total_calls,
                "active": self.active_calls,
                "completed": self.completed_calls,
                "missed": self.missed_calls,
            },
            "duration": {
                "total_talk_seconds": self.total_talk_time_seconds,
                "average_talk_seconds": self.average_talk_time_seconds,
                "longest_call_seconds": self.longest_call_seconds,
            },
            "performance": {
                "average_sentiment": self.average_sentiment,
                "resolution_rate": self.resolution_rate,
                "transfer_rate": self.transfer_rate,
            },
            "utilization": {
                "utilization_percent": self.utilization_percent,
                "idle_time_seconds": self.idle_time_seconds,
            },
        }


@dataclass
class QueueMetrics:
    """Metrics for call queues."""
    queue_id: str
    queue_name: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Volume
    calls_waiting: int = 0
    calls_in_progress: int = 0
    total_calls_today: int = 0

    # Wait times
    average_wait_seconds: float = 0.0
    max_wait_seconds: float = 0.0
    current_longest_wait_seconds: float = 0.0

    # Service level
    service_level_percent: float = 0.0
    service_level_target_seconds: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "queue_id": self.queue_id,
            "queue_name": self.queue_name,
            "timestamp": self.timestamp.isoformat(),
            "volume": {
                "waiting": self.calls_waiting,
                "in_progress": self.calls_in_progress,
                "total_today": self.total_calls_today,
            },
            "wait_times": {
                "average_seconds": self.average_wait_seconds,
                "max_seconds": self.max_wait_seconds,
                "current_longest_seconds": self.current_longest_wait_seconds,
            },
            "service_level": {
                "percent": self.service_level_percent,
                "target_seconds": self.service_level_target_seconds,
            },
        }


@dataclass
class DashboardData:
    """Complete dashboard data."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    call_metrics: CallMetrics = field(default_factory=CallMetrics)
    agent_metrics: List[AgentMetrics] = field(default_factory=list)
    queue_metrics: List[QueueMetrics] = field(default_factory=list)
    active_calls: List[Dict[str, Any]] = field(default_factory=list)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "call_metrics": self.call_metrics.to_dict(),
            "agent_metrics": [a.to_dict() for a in self.agent_metrics],
            "queue_metrics": [q.to_dict() for q in self.queue_metrics],
            "active_calls": self.active_calls,
            "recent_events": self.recent_events,
        }


class CallDashboard:
    """
    Real-time call monitoring dashboard.

    Usage:
        dashboard = CallDashboard(tracker)

        # Get current dashboard data
        data = await dashboard.get_dashboard_data()

        # Get specific metrics
        metrics = await dashboard.get_call_metrics()

        # Get agent performance
        agent_data = await dashboard.get_agent_metrics("agent_1")

        # Subscribe to updates
        dashboard.on_update(handle_update)
    """

    def __init__(
        self,
        tracker: Optional[LiveCallTracker] = None,
        update_interval: float = 5.0,
    ):
        self.tracker = tracker or get_live_call_tracker()
        self.update_interval = update_interval

        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        self._callbacks: List[callable] = []
        self._metrics_history: List[CallMetrics] = []
        self._max_history = 100
        self._recent_events: List[Dict[str, Any]] = []
        self._max_events = 50

        # Set up event listener
        self.tracker.on_call_event(self._handle_call_event)

    async def start(self) -> None:
        """Start the dashboard update loop."""
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Call dashboard started")

    async def stop(self) -> None:
        """Stop the dashboard update loop."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("Call dashboard stopped")

    async def get_dashboard_data(
        self,
        account_id: Optional[str] = None,
    ) -> DashboardData:
        """Get complete dashboard data."""
        call_metrics = await self.get_call_metrics(account_id)
        agent_metrics = await self.get_all_agent_metrics(account_id)
        active_calls = await self.get_active_calls_list(account_id)

        return DashboardData(
            timestamp=datetime.utcnow(),
            call_metrics=call_metrics,
            agent_metrics=agent_metrics,
            active_calls=active_calls,
            recent_events=self._recent_events.copy(),
        )

    async def get_call_metrics(
        self,
        account_id: Optional[str] = None,
        time_range_minutes: int = 60,
    ) -> CallMetrics:
        """Get aggregated call metrics."""
        calls = await self.tracker.get_all_calls(include_completed=True)

        if account_id:
            calls = [c for c in calls if c.account_id == account_id]

        # Filter by time range
        cutoff = datetime.utcnow() - timedelta(minutes=time_range_minutes)
        calls = [c for c in calls if c.started_at >= cutoff]

        metrics = CallMetrics(timestamp=datetime.utcnow())

        if not calls:
            return metrics

        # Volume metrics
        metrics.total_calls = len(calls)
        metrics.active_calls = len([c for c in calls if c.is_active])
        metrics.completed_calls = len([c for c in calls if c.state == CallState.COMPLETED])
        metrics.failed_calls = len([c for c in calls if c.state == CallState.FAILED])
        metrics.inbound_calls = len([c for c in calls if c.direction == CallDirection.INBOUND])
        metrics.outbound_calls = len([c for c in calls if c.direction == CallDirection.OUTBOUND])

        # Duration metrics
        completed = [c for c in calls if not c.is_active]
        if completed:
            metrics.total_duration_seconds = sum(c.duration_seconds for c in completed)
            metrics.total_talk_time_seconds = sum(c.talk_time_seconds for c in completed)
            metrics.average_duration_seconds = metrics.total_duration_seconds / len(completed)
            metrics.average_talk_time_seconds = metrics.total_talk_time_seconds / len(completed)

            # Calculate wait time (ring time)
            wait_times = []
            for c in completed:
                if c.answered_at:
                    wait = (c.answered_at - c.started_at).total_seconds()
                    wait_times.append(wait)
            if wait_times:
                metrics.average_wait_time_seconds = sum(wait_times) / len(wait_times)

        # Performance metrics
        answered = len([c for c in calls if c.answered_at is not None])
        metrics.answer_rate = answered / len(calls) if calls else 0

        abandoned = len([
            c for c in calls
            if c.state == CallState.FAILED and not c.answered_at
        ])
        metrics.abandonment_rate = abandoned / len(calls) if calls else 0

        transferred = len([c for c in calls if c.metadata.get("transferred", False)])
        metrics.transfer_rate = transferred / len(calls) if calls else 0

        supervised = len([c for c in calls if c.is_supervised or c.metadata.get("was_supervised", False)])
        metrics.supervised_rate = supervised / len(calls) if calls else 0

        # Sentiment metrics
        sentiments = [c.sentiment_score for c in calls if c.sentiment_score != 0]
        if sentiments:
            metrics.average_sentiment = sum(sentiments) / len(sentiments)
            metrics.positive_sentiment_count = len([s for s in sentiments if s > 0.3])
            metrics.negative_sentiment_count = len([s for s in sentiments if s < -0.3])
            metrics.neutral_sentiment_count = len(sentiments) - metrics.positive_sentiment_count - metrics.negative_sentiment_count

        return metrics

    async def get_agent_metrics(
        self,
        agent_id: str,
        time_range_minutes: int = 60,
    ) -> AgentMetrics:
        """Get metrics for a specific agent."""
        calls = await self.tracker.get_all_calls(include_completed=True)
        calls = [c for c in calls if c.agent_id == agent_id]

        # Filter by time range
        cutoff = datetime.utcnow() - timedelta(minutes=time_range_minutes)
        calls = [c for c in calls if c.started_at >= cutoff]

        metrics = AgentMetrics(
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
        )

        if not calls:
            return metrics

        active = [c for c in calls if c.is_active]
        completed = [c for c in calls if not c.is_active]

        # Status
        metrics.is_available = len(active) == 0
        metrics.active_calls = len(active)
        if active:
            metrics.current_call_id = active[0].call_id
            metrics.status = "on_call"
        else:
            metrics.status = "available"

        # Volume
        metrics.total_calls = len(calls)
        metrics.completed_calls = len(completed)

        # Duration
        if completed:
            talk_times = [c.talk_time_seconds for c in completed]
            metrics.total_talk_time_seconds = sum(talk_times)
            metrics.average_talk_time_seconds = metrics.total_talk_time_seconds / len(completed)
            metrics.longest_call_seconds = max(talk_times) if talk_times else 0

        # Performance
        if calls:
            sentiments = [c.sentiment_score for c in calls if c.sentiment_score != 0]
            if sentiments:
                metrics.average_sentiment = sum(sentiments) / len(sentiments)

            transferred = len([c for c in calls if c.metadata.get("transferred", False)])
            metrics.transfer_rate = transferred / len(calls)

        # Utilization (simplified calculation)
        if completed:
            total_possible = time_range_minutes * 60
            metrics.utilization_percent = min(100, (metrics.total_talk_time_seconds / total_possible) * 100)
            metrics.idle_time_seconds = max(0, total_possible - metrics.total_talk_time_seconds)

        return metrics

    async def get_all_agent_metrics(
        self,
        account_id: Optional[str] = None,
        time_range_minutes: int = 60,
    ) -> List[AgentMetrics]:
        """Get metrics for all agents."""
        calls = await self.tracker.get_all_calls(include_completed=True)

        if account_id:
            calls = [c for c in calls if c.account_id == account_id]

        # Get unique agent IDs
        agent_ids = set(c.agent_id for c in calls)

        metrics = []
        for agent_id in agent_ids:
            agent_metrics = await self.get_agent_metrics(agent_id, time_range_minutes)
            metrics.append(agent_metrics)

        return metrics

    async def get_active_calls_list(
        self,
        account_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get list of active calls for dashboard display."""
        calls = await self.tracker.get_active_calls(account_id=account_id)

        return [
            {
                "call_id": c.call_id,
                "agent_id": c.agent_id,
                "state": c.state.value,
                "direction": c.direction.value,
                "caller_number": c.caller_number,
                "duration_seconds": c.duration_seconds,
                "current_intent": c.current_intent,
                "sentiment_score": c.sentiment_score,
                "is_supervised": c.is_supervised,
                "started_at": c.started_at.isoformat(),
            }
            for c in calls
        ]

    async def get_metrics_history(
        self,
        limit: int = 60,
    ) -> List[Dict[str, Any]]:
        """Get historical metrics data."""
        return [m.to_dict() for m in self._metrics_history[-limit:]]

    def on_update(self, callback: callable) -> None:
        """Register callback for dashboard updates."""
        self._callbacks.append(callback)

    async def _update_loop(self) -> None:
        """Periodic dashboard update loop."""
        while self._running:
            try:
                # Get current metrics
                metrics = await self.get_call_metrics()

                # Add to history
                self._metrics_history.append(metrics)
                if len(self._metrics_history) > self._max_history:
                    self._metrics_history = self._metrics_history[-self._max_history:]

                # Get dashboard data
                data = await self.get_dashboard_data()

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
                    except Exception as e:
                        logger.error(f"Dashboard callback error: {e}")

            except Exception as e:
                logger.error(f"Dashboard update error: {e}")

            await asyncio.sleep(self.update_interval)

    async def _handle_call_event(self, event) -> None:
        """Handle call events for recent events list."""
        event_data = {
            "event_type": event.event_type,
            "call_id": event.call_id,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data,
        }

        self._recent_events.insert(0, event_data)
        if len(self._recent_events) > self._max_events:
            self._recent_events = self._recent_events[:self._max_events]


class RealTimeUpdater:
    """
    WebSocket-based real-time dashboard updater.

    Usage:
        updater = RealTimeUpdater(dashboard)

        # Start streaming updates
        async for update in updater.stream_updates():
            await websocket.send_json(update)
    """

    def __init__(
        self,
        dashboard: CallDashboard,
        update_interval: float = 2.0,
    ):
        self.dashboard = dashboard
        self.update_interval = update_interval

    async def stream_updates(
        self,
        account_id: Optional[str] = None,
    ):
        """Stream dashboard updates."""
        while True:
            try:
                data = await self.dashboard.get_dashboard_data(account_id)
                yield data.to_dict()
            except Exception as e:
                logger.error(f"Stream update error: {e}")
                yield {"error": str(e)}

            await asyncio.sleep(self.update_interval)

    async def get_single_update(
        self,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a single dashboard update."""
        try:
            data = await self.dashboard.get_dashboard_data(account_id)
            return data.to_dict()
        except Exception as e:
            logger.error(f"Single update error: {e}")
            return {"error": str(e)}
