"""Real-time call monitoring dashboard service."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum

import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class CallStatus(str, Enum):
    """Call status enum."""
    QUEUED = "queued"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BUSY = "busy"
    NO_ANSWER = "no_answer"


class EventType(str, Enum):
    """Dashboard event types."""
    CALL_STARTED = "call.started"
    CALL_RINGING = "call.ringing"
    CALL_ANSWERED = "call.answered"
    CALL_ENDED = "call.ended"
    CALL_FAILED = "call.failed"
    TRANSCRIPTION = "transcription"
    AGENT_SPEAKING = "agent.speaking"
    AGENT_THINKING = "agent.thinking"
    SENTIMENT_UPDATE = "sentiment.update"
    METRICS_UPDATE = "metrics.update"
    ALERT = "alert"


@dataclass
class ActiveCall:
    """Active call data structure."""
    id: str
    agent_id: str
    agent_name: str
    to_number: str
    from_number: str
    direction: str
    status: CallStatus
    started_at: datetime
    duration: int = 0
    current_sentiment: Optional[str] = None
    sentiment_score: float = 0.0
    transcript_preview: str = ""
    is_speaking: bool = False
    organization_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "started_at": self.started_at.isoformat(),
            "status": self.status.value,
        }


@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics."""
    active_calls: int = 0
    calls_today: int = 0
    calls_this_hour: int = 0
    avg_duration: float = 0.0
    success_rate: float = 0.0
    queued_calls: int = 0
    agents_in_use: int = 0
    total_agents: int = 0
    concurrent_limit: int = 0
    current_concurrency: int = 0
    error_rate: float = 0.0
    avg_wait_time: float = 0.0
    sentiment_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ConnectionManager:
    """WebSocket connection manager for dashboard clients."""

    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}  # org_id -> websockets
        self._user_subscriptions: Dict[WebSocket, Set[str]] = {}  # websocket -> call_ids
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, organization_id: str) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()

        async with self._lock:
            if organization_id not in self._connections:
                self._connections[organization_id] = set()
            self._connections[organization_id].add(websocket)
            self._user_subscriptions[websocket] = set()

        logger.info(f"Dashboard client connected for org {organization_id}")

    async def disconnect(self, websocket: WebSocket, organization_id: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if organization_id in self._connections:
                self._connections[organization_id].discard(websocket)
                if not self._connections[organization_id]:
                    del self._connections[organization_id]

            if websocket in self._user_subscriptions:
                del self._user_subscriptions[websocket]

        logger.info(f"Dashboard client disconnected from org {organization_id}")

    async def subscribe_to_call(self, websocket: WebSocket, call_id: str) -> None:
        """Subscribe a client to specific call updates."""
        async with self._lock:
            if websocket in self._user_subscriptions:
                self._user_subscriptions[websocket].add(call_id)

    async def unsubscribe_from_call(self, websocket: WebSocket, call_id: str) -> None:
        """Unsubscribe a client from specific call updates."""
        async with self._lock:
            if websocket in self._user_subscriptions:
                self._user_subscriptions[websocket].discard(call_id)

    async def broadcast_to_org(
        self,
        organization_id: str,
        event_type: EventType,
        data: Dict[str, Any],
    ) -> None:
        """Broadcast an event to all clients in an organization."""
        message = json.dumps({
            "type": event_type.value,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        })

        async with self._lock:
            connections = self._connections.get(organization_id, set()).copy()

        if not connections:
            return

        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            await self.disconnect(ws, organization_id)

    async def send_to_call_subscribers(
        self,
        organization_id: str,
        call_id: str,
        event_type: EventType,
        data: Dict[str, Any],
    ) -> None:
        """Send an event to clients subscribed to a specific call."""
        message = json.dumps({
            "type": event_type.value,
            "call_id": call_id,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        })

        async with self._lock:
            connections = self._connections.get(organization_id, set()).copy()

        for websocket in connections:
            subscriptions = self._user_subscriptions.get(websocket, set())
            if call_id in subscriptions or not subscriptions:  # Send if subscribed or no filter
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.warning(f"Failed to send to subscriber: {e}")

    def get_connection_count(self, organization_id: str) -> int:
        """Get number of connected clients for an organization."""
        return len(self._connections.get(organization_id, set()))


class CallMonitorDashboard:
    """Real-time call monitoring dashboard service."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        metrics_interval: int = 5,
    ):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.connection_manager = ConnectionManager()
        self.metrics_interval = metrics_interval

        # In-memory state
        self._active_calls: Dict[str, ActiveCall] = {}
        self._metrics_cache: Dict[str, DashboardMetrics] = {}
        self._lock = asyncio.Lock()

        # Event handlers
        self._event_handlers: Dict[EventType, List[Callable]] = {}

        # Background tasks
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start the dashboard service."""
        self.redis = redis.from_url(self.redis_url, decode_responses=True)

        # Start background tasks
        self._tasks.append(asyncio.create_task(self._metrics_broadcaster()))
        self._tasks.append(asyncio.create_task(self._redis_subscriber()))
        self._tasks.append(asyncio.create_task(self._cleanup_stale_calls()))

        logger.info("Call monitoring dashboard started")

    async def stop(self) -> None:
        """Stop the dashboard service."""
        for task in self._tasks:
            task.cancel()

        if self.redis:
            await self.redis.close()

        logger.info("Call monitoring dashboard stopped")

    async def handle_websocket(
        self,
        websocket: WebSocket,
        organization_id: str,
        user_id: str,
    ) -> None:
        """Handle a WebSocket connection from a dashboard client."""
        await self.connection_manager.connect(websocket, organization_id)

        try:
            # Send initial state
            await self._send_initial_state(websocket, organization_id)

            # Handle messages
            while True:
                data = await websocket.receive_json()
                await self._handle_client_message(websocket, organization_id, data)

        except WebSocketDisconnect:
            logger.info(f"Client disconnected: {user_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await self.connection_manager.disconnect(websocket, organization_id)

    async def _send_initial_state(
        self,
        websocket: WebSocket,
        organization_id: str,
    ) -> None:
        """Send initial state to a newly connected client."""
        # Get active calls for this org
        active_calls = await self.get_active_calls(organization_id)

        # Get current metrics
        metrics = await self.get_metrics(organization_id)

        initial_state = {
            "type": "initial_state",
            "data": {
                "active_calls": [call.to_dict() for call in active_calls],
                "metrics": metrics.to_dict(),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        await websocket.send_json(initial_state)

    async def _handle_client_message(
        self,
        websocket: WebSocket,
        organization_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Handle a message from a dashboard client."""
        action = data.get("action")

        if action == "subscribe_call":
            call_id = data.get("call_id")
            if call_id:
                await self.connection_manager.subscribe_to_call(websocket, call_id)
                await self._send_call_details(websocket, call_id)

        elif action == "unsubscribe_call":
            call_id = data.get("call_id")
            if call_id:
                await self.connection_manager.unsubscribe_from_call(websocket, call_id)

        elif action == "refresh_metrics":
            metrics = await self.get_metrics(organization_id)
            await websocket.send_json({
                "type": "metrics.update",
                "data": metrics.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
            })

        elif action == "get_call_transcript":
            call_id = data.get("call_id")
            if call_id:
                transcript = await self._get_call_transcript(call_id)
                await websocket.send_json({
                    "type": "call.transcript",
                    "call_id": call_id,
                    "data": transcript,
                    "timestamp": datetime.utcnow().isoformat(),
                })

    async def _send_call_details(self, websocket: WebSocket, call_id: str) -> None:
        """Send detailed call information to a client."""
        async with self._lock:
            call = self._active_calls.get(call_id)

        if call:
            await websocket.send_json({
                "type": "call.details",
                "data": call.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
            })

    async def _get_call_transcript(self, call_id: str) -> List[Dict[str, Any]]:
        """Get transcript for a call."""
        if not self.redis:
            return []

        transcript_key = f"call:{call_id}:transcript"
        messages = await self.redis.lrange(transcript_key, 0, -1)

        return [json.loads(msg) for msg in messages]

    # Call event handlers

    async def on_call_started(self, event: Dict[str, Any]) -> None:
        """Handle call started event."""
        call_id = event.get("call_id")
        organization_id = event.get("organization_id")

        if not call_id or not organization_id:
            return

        call = ActiveCall(
            id=call_id,
            agent_id=event.get("agent_id", ""),
            agent_name=event.get("agent_name", "Unknown"),
            to_number=event.get("to_number", ""),
            from_number=event.get("from_number", ""),
            direction=event.get("direction", "outbound"),
            status=CallStatus.QUEUED,
            started_at=datetime.utcnow(),
            organization_id=organization_id,
            metadata=event.get("metadata", {}),
        )

        async with self._lock:
            self._active_calls[call_id] = call

        # Store in Redis
        if self.redis:
            await self.redis.hset(
                f"org:{organization_id}:active_calls",
                call_id,
                json.dumps(call.to_dict()),
            )

        # Broadcast to clients
        await self.connection_manager.broadcast_to_org(
            organization_id,
            EventType.CALL_STARTED,
            call.to_dict(),
        )

        logger.info(f"Call started: {call_id}")

    async def on_call_ringing(self, event: Dict[str, Any]) -> None:
        """Handle call ringing event."""
        call_id = event.get("call_id")
        organization_id = event.get("organization_id")

        async with self._lock:
            if call_id in self._active_calls:
                self._active_calls[call_id].status = CallStatus.RINGING

        await self.connection_manager.broadcast_to_org(
            organization_id,
            EventType.CALL_RINGING,
            {"call_id": call_id, "status": "ringing"},
        )

    async def on_call_answered(self, event: Dict[str, Any]) -> None:
        """Handle call answered event."""
        call_id = event.get("call_id")
        organization_id = event.get("organization_id")

        async with self._lock:
            if call_id in self._active_calls:
                self._active_calls[call_id].status = CallStatus.IN_PROGRESS

        await self.connection_manager.broadcast_to_org(
            organization_id,
            EventType.CALL_ANSWERED,
            {"call_id": call_id, "status": "in_progress"},
        )

    async def on_call_ended(self, event: Dict[str, Any]) -> None:
        """Handle call ended event."""
        call_id = event.get("call_id")
        organization_id = event.get("organization_id")
        duration = event.get("duration", 0)
        end_reason = event.get("end_reason", "unknown")

        async with self._lock:
            call = self._active_calls.pop(call_id, None)

        # Remove from Redis
        if self.redis and organization_id:
            await self.redis.hdel(f"org:{organization_id}:active_calls", call_id)

        # Broadcast to clients
        if organization_id:
            await self.connection_manager.broadcast_to_org(
                organization_id,
                EventType.CALL_ENDED,
                {
                    "call_id": call_id,
                    "duration": duration,
                    "end_reason": end_reason,
                    "status": "completed",
                },
            )

        logger.info(f"Call ended: {call_id}, duration: {duration}s")

    async def on_transcription(self, event: Dict[str, Any]) -> None:
        """Handle transcription event."""
        call_id = event.get("call_id")
        organization_id = event.get("organization_id")
        text = event.get("text", "")
        role = event.get("role", "user")
        is_final = event.get("is_final", False)

        # Update transcript preview
        async with self._lock:
            if call_id in self._active_calls:
                if role == "user":
                    self._active_calls[call_id].transcript_preview = text[:100]

        # Store in Redis transcript history
        if self.redis and is_final:
            transcript_entry = {
                "role": role,
                "text": text,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await self.redis.rpush(
                f"call:{call_id}:transcript",
                json.dumps(transcript_entry),
            )

        # Send to subscribed clients
        if organization_id:
            await self.connection_manager.send_to_call_subscribers(
                organization_id,
                call_id,
                EventType.TRANSCRIPTION,
                {
                    "call_id": call_id,
                    "role": role,
                    "text": text,
                    "is_final": is_final,
                },
            )

    async def on_agent_speaking(self, event: Dict[str, Any]) -> None:
        """Handle agent speaking event."""
        call_id = event.get("call_id")
        organization_id = event.get("organization_id")
        is_speaking = event.get("is_speaking", False)

        async with self._lock:
            if call_id in self._active_calls:
                self._active_calls[call_id].is_speaking = is_speaking

        if organization_id:
            await self.connection_manager.send_to_call_subscribers(
                organization_id,
                call_id,
                EventType.AGENT_SPEAKING,
                {"call_id": call_id, "is_speaking": is_speaking},
            )

    async def on_sentiment_update(self, event: Dict[str, Any]) -> None:
        """Handle sentiment analysis update."""
        call_id = event.get("call_id")
        organization_id = event.get("organization_id")
        sentiment = event.get("sentiment", "neutral")
        score = event.get("score", 0.0)

        async with self._lock:
            if call_id in self._active_calls:
                self._active_calls[call_id].current_sentiment = sentiment
                self._active_calls[call_id].sentiment_score = score

        if organization_id:
            await self.connection_manager.send_to_call_subscribers(
                organization_id,
                call_id,
                EventType.SENTIMENT_UPDATE,
                {
                    "call_id": call_id,
                    "sentiment": sentiment,
                    "score": score,
                },
            )

    # Data retrieval methods

    async def get_active_calls(self, organization_id: str) -> List[ActiveCall]:
        """Get all active calls for an organization."""
        async with self._lock:
            return [
                call for call in self._active_calls.values()
                if call.organization_id == organization_id
            ]

    async def get_metrics(self, organization_id: str) -> DashboardMetrics:
        """Get current metrics for an organization."""
        # Check cache first
        if organization_id in self._metrics_cache:
            return self._metrics_cache[organization_id]

        # Calculate metrics
        metrics = await self._calculate_metrics(organization_id)
        self._metrics_cache[organization_id] = metrics

        return metrics

    async def _calculate_metrics(self, organization_id: str) -> DashboardMetrics:
        """Calculate dashboard metrics for an organization."""
        active_calls = await self.get_active_calls(organization_id)

        # Count by status
        in_progress = sum(1 for c in active_calls if c.status == CallStatus.IN_PROGRESS)
        queued = sum(1 for c in active_calls if c.status == CallStatus.QUEUED)

        # Sentiment breakdown
        sentiment_breakdown = {}
        for call in active_calls:
            if call.current_sentiment:
                sentiment_breakdown[call.current_sentiment] = (
                    sentiment_breakdown.get(call.current_sentiment, 0) + 1
                )

        # Get additional stats from Redis
        calls_today = 0
        calls_this_hour = 0
        success_rate = 0.0
        avg_duration = 0.0

        if self.redis:
            today_key = f"org:{organization_id}:calls:{datetime.utcnow().strftime('%Y-%m-%d')}"
            hour_key = f"org:{organization_id}:calls:{datetime.utcnow().strftime('%Y-%m-%d-%H')}"

            calls_today = await self.redis.get(today_key) or 0
            calls_this_hour = await self.redis.get(hour_key) or 0

            # Get success rate from last 100 calls
            success_count = await self.redis.get(f"org:{organization_id}:success_count") or 0
            total_count = await self.redis.get(f"org:{organization_id}:total_count") or 1
            success_rate = int(success_count) / max(int(total_count), 1)

            # Get average duration
            avg_duration_val = await self.redis.get(f"org:{organization_id}:avg_duration")
            avg_duration = float(avg_duration_val) if avg_duration_val else 0.0

        return DashboardMetrics(
            active_calls=len(active_calls),
            calls_today=int(calls_today),
            calls_this_hour=int(calls_this_hour),
            avg_duration=avg_duration,
            success_rate=success_rate,
            queued_calls=queued,
            current_concurrency=in_progress,
            sentiment_breakdown=sentiment_breakdown,
        )

    # Background tasks

    async def _metrics_broadcaster(self) -> None:
        """Periodically broadcast metrics to all connected clients."""
        while True:
            try:
                await asyncio.sleep(self.metrics_interval)

                # Get all organizations with connected clients
                orgs = list(self.connection_manager._connections.keys())

                for org_id in orgs:
                    metrics = await self.get_metrics(org_id)
                    await self.connection_manager.broadcast_to_org(
                        org_id,
                        EventType.METRICS_UPDATE,
                        metrics.to_dict(),
                    )

                # Clear metrics cache
                self._metrics_cache.clear()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics broadcaster error: {e}")

    async def _redis_subscriber(self) -> None:
        """Subscribe to Redis pub/sub for call events."""
        if not self.redis:
            return

        pubsub = self.redis.pubsub()
        await pubsub.subscribe("call_events")

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        event = json.loads(message["data"])
                        await self._handle_redis_event(event)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in Redis message")
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe("call_events")

    async def _handle_redis_event(self, event: Dict[str, Any]) -> None:
        """Handle an event from Redis pub/sub."""
        event_type = event.get("type")

        handlers = {
            "call.started": self.on_call_started,
            "call.ringing": self.on_call_ringing,
            "call.answered": self.on_call_answered,
            "call.ended": self.on_call_ended,
            "transcription": self.on_transcription,
            "agent.speaking": self.on_agent_speaking,
            "sentiment.update": self.on_sentiment_update,
        }

        handler = handlers.get(event_type)
        if handler:
            await handler(event)

    async def _cleanup_stale_calls(self) -> None:
        """Clean up calls that have been active too long (stale)."""
        max_duration = timedelta(hours=2)

        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.utcnow()
                stale_calls = []

                async with self._lock:
                    for call_id, call in self._active_calls.items():
                        if now - call.started_at > max_duration:
                            stale_calls.append(call_id)

                for call_id in stale_calls:
                    async with self._lock:
                        call = self._active_calls.pop(call_id, None)

                    if call:
                        logger.warning(f"Cleaned up stale call: {call_id}")
                        await self.connection_manager.broadcast_to_org(
                            call.organization_id,
                            EventType.CALL_ENDED,
                            {
                                "call_id": call_id,
                                "end_reason": "timeout",
                                "status": "failed",
                            },
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
