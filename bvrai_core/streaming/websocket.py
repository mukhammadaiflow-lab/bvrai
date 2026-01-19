"""
WebSocket Management
====================

WebSocket connection management for real-time bidirectional
communication with clients.

Author: Platform Engineering Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
)
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""

    # Control messages
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"

    # Audio messages
    AUDIO_DATA = "audio_data"
    AUDIO_START = "audio_start"
    AUDIO_STOP = "audio_stop"
    AUDIO_CONFIG = "audio_config"

    # Transcription messages
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"

    # Agent messages
    AGENT_RESPONSE = "agent_response"
    AGENT_THINKING = "agent_thinking"
    AGENT_ACTION = "agent_action"

    # Call messages
    CALL_START = "call_start"
    CALL_END = "call_end"
    CALL_TRANSFER = "call_transfer"
    CALL_HOLD = "call_hold"

    # Custom
    CUSTOM = "custom"
    DATA = "data"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""

    id: str = field(default_factory=lambda: f"msg_{uuid4().hex[:12]}")
    type: MessageType = MessageType.DATA
    payload: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sequence: int = 0

    # Binary data (for audio)
    binary_data: Optional[bytes] = None

    # Metadata
    session_id: Optional[str] = None
    call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "id": self.id,
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
            "session_id": self.session_id,
            "call_id": self.call_id,
            "metadata": self.metadata,
        })

    def to_bytes(self) -> bytes:
        """Convert to bytes (for binary messages)."""
        if self.binary_data:
            # Prepend header with type and sequence
            header = struct.pack(
                "!BHI",
                self.type.value.encode()[0] if isinstance(self.type.value, str) else 0,
                len(self.binary_data),
                self.sequence,
            )
            return header + self.binary_data
        return self.to_json().encode()

    @classmethod
    def from_json(cls, data: str) -> "WebSocketMessage":
        """Create from JSON string."""
        parsed = json.loads(data)
        return cls(
            id=parsed.get("id", f"msg_{uuid4().hex[:12]}"),
            type=MessageType(parsed.get("type", "data")),
            payload=parsed.get("payload"),
            timestamp=datetime.fromisoformat(parsed["timestamp"]) if parsed.get("timestamp") else datetime.utcnow(),
            sequence=parsed.get("sequence", 0),
            session_id=parsed.get("session_id"),
            call_id=parsed.get("call_id"),
            metadata=parsed.get("metadata", {}),
        )

    @classmethod
    def audio(
        cls,
        data: bytes,
        session_id: Optional[str] = None,
        sequence: int = 0,
    ) -> "WebSocketMessage":
        """Create an audio message."""
        return cls(
            type=MessageType.AUDIO_DATA,
            binary_data=data,
            session_id=session_id,
            sequence=sequence,
        )

    @classmethod
    def transcript(
        cls,
        text: str,
        is_final: bool = False,
        confidence: float = 0.0,
        **kwargs,
    ) -> "WebSocketMessage":
        """Create a transcript message."""
        return cls(
            type=MessageType.TRANSCRIPT_FINAL if is_final else MessageType.TRANSCRIPT_PARTIAL,
            payload={
                "text": text,
                "is_final": is_final,
                "confidence": confidence,
                **kwargs,
            },
        )


# Import struct for binary message handling
import struct


@dataclass
class WebSocketSession:
    """Represents a WebSocket connection session."""

    id: str = field(default_factory=lambda: f"ws_{uuid4().hex[:12]}")
    connection: Any = None  # Actual WebSocket connection object

    # Identity
    organization_id: Optional[str] = None
    user_id: Optional[str] = None
    call_id: Optional[str] = None

    # State
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    is_authenticated: bool = False

    # Configuration
    audio_format: str = "pcm"
    sample_rate: int = 16000
    channels: int = 1

    # Statistics
    messages_received: int = 0
    messages_sent: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    sequence: int = 0

    # Subscriptions
    subscribed_events: Set[str] = field(default_factory=set)

    # Metadata
    client_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_connected(self) -> bool:
        """Check if session is connected."""
        return self.connection is not None

    @property
    def duration_seconds(self) -> float:
        """Get session duration."""
        return (datetime.utcnow() - self.connected_at).total_seconds()

    def next_sequence(self) -> int:
        """Get next sequence number."""
        self.sequence += 1
        return self.sequence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "call_id": self.call_id,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "duration_seconds": self.duration_seconds,
            "is_authenticated": self.is_authenticated,
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "bytes_received": self.bytes_received,
            "bytes_sent": self.bytes_sent,
        }


class WebSocketManager:
    """
    Manages WebSocket connections and messaging.

    Handles connection lifecycle, message routing, broadcasting,
    and session management.
    """

    def __init__(
        self,
        max_connections: int = 10000,
        ping_interval_seconds: int = 30,
        idle_timeout_seconds: int = 300,
    ):
        self._max_connections = max_connections
        self._ping_interval = ping_interval_seconds
        self._idle_timeout = idle_timeout_seconds

        self._sessions: Dict[str, WebSocketSession] = {}
        self._sessions_by_call: Dict[str, Set[str]] = defaultdict(set)
        self._sessions_by_org: Dict[str, Set[str]] = defaultdict(set)
        self._sessions_by_user: Dict[str, Set[str]] = defaultdict(set)

        self._message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        self._lock = asyncio.Lock()
        self._running = False
        self._ping_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        self._logger = structlog.get_logger("websocket_manager")

        # Metrics
        self._total_connections = 0
        self._total_messages = 0
        self._total_bytes = 0

    async def start(self) -> None:
        """Start the WebSocket manager."""
        self._running = True
        self._ping_task = asyncio.create_task(self._ping_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._logger.info("WebSocket manager started")

    async def stop(self) -> None:
        """Stop the WebSocket manager."""
        self._running = False

        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all sessions
        for session_id in list(self._sessions.keys()):
            await self.disconnect(session_id)

        self._logger.info("WebSocket manager stopped")

    async def connect(
        self,
        connection: Any,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        call_id: Optional[str] = None,
        **kwargs,
    ) -> WebSocketSession:
        """
        Register a new WebSocket connection.

        Args:
            connection: The WebSocket connection object
            organization_id: Organization ID
            user_id: User ID
            call_id: Associated call ID
            **kwargs: Additional session attributes

        Returns:
            WebSocketSession
        """
        if len(self._sessions) >= self._max_connections:
            raise RuntimeError(f"Maximum connections ({self._max_connections}) reached")

        session = WebSocketSession(
            connection=connection,
            organization_id=organization_id,
            user_id=user_id,
            call_id=call_id,
            **kwargs,
        )

        async with self._lock:
            self._sessions[session.id] = session
            if organization_id:
                self._sessions_by_org[organization_id].add(session.id)
            if user_id:
                self._sessions_by_user[user_id].add(session.id)
            if call_id:
                self._sessions_by_call[call_id].add(session.id)
            self._total_connections += 1

        self._logger.info(
            f"WebSocket connected: {session.id}",
            org=organization_id,
            user=user_id,
            call=call_id,
        )

        await self._emit_event("connected", session)

        # Send connect acknowledgment
        await self.send(session.id, WebSocketMessage(
            type=MessageType.CONNECT,
            payload={"session_id": session.id},
        ))

        return session

    async def disconnect(self, session_id: str, reason: str = "") -> bool:
        """Disconnect a WebSocket session."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        async with self._lock:
            del self._sessions[session_id]
            if session.organization_id:
                self._sessions_by_org[session.organization_id].discard(session_id)
            if session.user_id:
                self._sessions_by_user[session.user_id].discard(session_id)
            if session.call_id:
                self._sessions_by_call[session.call_id].discard(session_id)

            self._total_bytes += session.bytes_received + session.bytes_sent
            self._total_messages += session.messages_received + session.messages_sent

        # Close the actual connection
        if session.connection:
            try:
                await self._close_connection(session.connection, reason)
            except Exception as e:
                self._logger.error(f"Error closing connection: {e}")

        self._logger.info(
            f"WebSocket disconnected: {session_id}",
            reason=reason,
            duration=session.duration_seconds,
        )

        await self._emit_event("disconnected", session, reason)
        return True

    async def _close_connection(self, connection: Any, reason: str) -> None:
        """Close the actual WebSocket connection."""
        # Implementation depends on the WebSocket library used
        if hasattr(connection, "close"):
            if asyncio.iscoroutinefunction(connection.close):
                await connection.close(reason=reason)
            else:
                connection.close(reason=reason)

    async def send(
        self,
        session_id: str,
        message: WebSocketMessage,
    ) -> bool:
        """Send a message to a session."""
        session = self._sessions.get(session_id)
        if not session or not session.connection:
            return False

        try:
            message.sequence = session.next_sequence()

            if message.binary_data:
                await self._send_binary(session.connection, message.binary_data)
                session.bytes_sent += len(message.binary_data)
            else:
                json_data = message.to_json()
                await self._send_text(session.connection, json_data)
                session.bytes_sent += len(json_data)

            session.messages_sent += 1
            session.last_activity = datetime.utcnow()
            return True

        except Exception as e:
            self._logger.error(f"Error sending message: {e}")
            return False

    async def _send_text(self, connection: Any, text: str) -> None:
        """Send text message through connection."""
        if hasattr(connection, "send_text"):
            await connection.send_text(text)
        elif hasattr(connection, "send"):
            await connection.send(text)

    async def _send_binary(self, connection: Any, data: bytes) -> None:
        """Send binary message through connection."""
        if hasattr(connection, "send_bytes"):
            await connection.send_bytes(data)
        elif hasattr(connection, "send"):
            await connection.send(data)

    async def broadcast(
        self,
        message: WebSocketMessage,
        organization_id: Optional[str] = None,
        call_id: Optional[str] = None,
        exclude: Optional[Set[str]] = None,
    ) -> int:
        """
        Broadcast a message to multiple sessions.

        Args:
            message: Message to broadcast
            organization_id: Filter by organization
            call_id: Filter by call
            exclude: Session IDs to exclude

        Returns:
            Number of messages sent
        """
        exclude = exclude or set()

        if call_id:
            session_ids = self._sessions_by_call.get(call_id, set())
        elif organization_id:
            session_ids = self._sessions_by_org.get(organization_id, set())
        else:
            session_ids = set(self._sessions.keys())

        sent = 0
        for session_id in session_ids:
            if session_id not in exclude:
                if await self.send(session_id, message):
                    sent += 1

        return sent

    async def receive(self, session_id: str, data: Union[str, bytes]) -> None:
        """
        Handle received data from a session.

        Args:
            session_id: Session that received the data
            data: Received data (text or binary)
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        session.last_activity = datetime.utcnow()
        session.messages_received += 1

        if isinstance(data, bytes):
            session.bytes_received += len(data)
            message = WebSocketMessage(
                type=MessageType.AUDIO_DATA,
                binary_data=data,
                session_id=session_id,
            )
        else:
            session.bytes_received += len(data)
            try:
                message = WebSocketMessage.from_json(data)
                message.session_id = session_id
            except json.JSONDecodeError:
                message = WebSocketMessage(
                    type=MessageType.DATA,
                    payload=data,
                    session_id=session_id,
                )

        # Handle ping/pong
        if message.type == MessageType.PING:
            await self.send(session_id, WebSocketMessage(type=MessageType.PONG))
            return

        # Call message handlers
        await self._dispatch_message(message, session)

    async def _dispatch_message(
        self,
        message: WebSocketMessage,
        session: WebSocketSession,
    ) -> None:
        """Dispatch message to handlers."""
        handlers = self._message_handlers.get(message.type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message, session)
                else:
                    handler(message, session)
            except Exception as e:
                self._logger.error(f"Message handler error: {e}")

    def on_message(
        self,
        message_type: MessageType,
        handler: Callable,
    ) -> None:
        """Register a message handler."""
        self._message_handlers[message_type].append(handler)

    def on_event(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        self._event_handlers[event].append(handler)

    async def _emit_event(self, event: str, *args, **kwargs) -> None:
        """Emit an event to handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                self._logger.error(f"Event handler error: {e}")

    async def get_session(self, session_id: str) -> Optional[WebSocketSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def get_sessions_for_call(self, call_id: str) -> List[WebSocketSession]:
        """Get all sessions for a call."""
        session_ids = self._sessions_by_call.get(call_id, set())
        return [self._sessions[sid] for sid in session_ids if sid in self._sessions]

    async def get_sessions_for_user(self, user_id: str) -> List[WebSocketSession]:
        """Get all sessions for a user."""
        session_ids = self._sessions_by_user.get(user_id, set())
        return [self._sessions[sid] for sid in session_ids if sid in self._sessions]

    async def _ping_loop(self) -> None:
        """Send periodic pings to all sessions."""
        while self._running:
            await asyncio.sleep(self._ping_interval)

            for session_id in list(self._sessions.keys()):
                await self.send(session_id, WebSocketMessage(type=MessageType.PING))

    async def _cleanup_loop(self) -> None:
        """Clean up idle sessions."""
        while self._running:
            await asyncio.sleep(60)

            now = datetime.utcnow()
            idle_sessions = []

            for session_id, session in self._sessions.items():
                idle_seconds = (now - session.last_activity).total_seconds()
                if idle_seconds > self._idle_timeout:
                    idle_sessions.append(session_id)

            for session_id in idle_sessions:
                self._logger.info(f"Disconnecting idle session: {session_id}")
                await self.disconnect(session_id, "idle timeout")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get WebSocket manager metrics."""
        active_sessions = len(self._sessions)

        current_bytes = sum(
            s.bytes_received + s.bytes_sent
            for s in self._sessions.values()
        )
        current_messages = sum(
            s.messages_received + s.messages_sent
            for s in self._sessions.values()
        )

        return {
            "total_connections": self._total_connections,
            "active_connections": active_sessions,
            "max_connections": self._max_connections,
            "total_messages": self._total_messages + current_messages,
            "total_bytes": self._total_bytes + current_bytes,
            "sessions_by_org": {
                org: len(sessions)
                for org, sessions in self._sessions_by_org.items()
            },
            "sessions_by_call": len(self._sessions_by_call),
        }
