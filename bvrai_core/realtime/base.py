"""
Real-time Base Types

This module defines the core data structures for real-time
communication including connections, messages, and events.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)


class ConnectionState(str, Enum):
    """Connection state."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class ConnectionType(str, Enum):
    """Type of connection."""
    WEBSOCKET = "websocket"
    WEBRTC = "webrtc"
    SSE = "sse"  # Server-Sent Events
    LONG_POLL = "long_poll"


class MessageType(str, Enum):
    """Types of messages."""
    # Connection
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"

    # Authentication
    AUTH = "auth"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILED = "auth_failed"

    # Call events
    CALL_STARTED = "call_started"
    CALL_ENDED = "call_ended"
    CALL_TRANSFERRED = "call_transferred"
    CALL_HELD = "call_held"
    CALL_RESUMED = "call_resumed"

    # Audio
    AUDIO_START = "audio_start"
    AUDIO_STOP = "audio_stop"
    AUDIO_DATA = "audio_data"
    AUDIO_CONFIG = "audio_config"

    # Transcription
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"

    # Agent
    AGENT_SPEAKING = "agent_speaking"
    AGENT_LISTENING = "agent_listening"
    AGENT_THINKING = "agent_thinking"
    AGENT_ACTION = "agent_action"

    # Intent/Conversation
    INTENT_DETECTED = "intent_detected"
    SLOT_FILLED = "slot_filled"
    FLOW_CHANGED = "flow_changed"
    CONVERSATION_UPDATE = "conversation_update"

    # Errors
    ERROR = "error"
    WARNING = "warning"

    # Custom
    CUSTOM = "custom"


class MessagePriority(int, Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class EventType(str, Enum):
    """Types of events."""
    # System events
    SYSTEM_START = "system:start"
    SYSTEM_STOP = "system:stop"
    SYSTEM_ERROR = "system:error"

    # Connection events
    CONNECTION_OPENED = "connection:opened"
    CONNECTION_CLOSED = "connection:closed"
    CONNECTION_ERROR = "connection:error"
    CONNECTION_AUTHENTICATED = "connection:authenticated"

    # Call events
    CALL_INITIATED = "call:initiated"
    CALL_RINGING = "call:ringing"
    CALL_ANSWERED = "call:answered"
    CALL_ENDED = "call:ended"
    CALL_FAILED = "call:failed"

    # Audio events
    AUDIO_STARTED = "audio:started"
    AUDIO_STOPPED = "audio:stopped"
    AUDIO_RECEIVED = "audio:received"
    AUDIO_SENT = "audio:sent"

    # Speech events
    SPEECH_STARTED = "speech:started"
    SPEECH_ENDED = "speech:ended"
    SPEECH_RECOGNIZED = "speech:recognized"
    SPEECH_SYNTHESIS_STARTED = "speech:synthesis_started"
    SPEECH_SYNTHESIS_COMPLETED = "speech:synthesis_completed"

    # Agent events
    AGENT_TURN_STARTED = "agent:turn_started"
    AGENT_TURN_ENDED = "agent:turn_ended"
    AGENT_RESPONSE = "agent:response"

    # Conversation events
    CONVERSATION_STARTED = "conversation:started"
    CONVERSATION_ENDED = "conversation:ended"
    TURN_COMPLETED = "turn:completed"

    # Custom
    CUSTOM = "custom"


class EventPriority(int, Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class SessionState(str, Enum):
    """Session state."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDING = "ending"
    ENDED = "ended"
    ERROR = "error"


@dataclass
class ConnectionConfig:
    """Configuration for a connection."""

    # Connection settings
    connection_type: ConnectionType = ConnectionType.WEBSOCKET
    timeout_seconds: int = 30
    ping_interval_seconds: int = 30
    max_message_size: int = 1024 * 1024  # 1MB

    # Buffer settings
    buffer_size: int = 1024 * 64  # 64KB
    max_queue_size: int = 1000

    # Reconnection
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_delay_seconds: float = 1.0
    reconnect_backoff_multiplier: float = 2.0

    # Authentication
    require_auth: bool = True
    auth_timeout_seconds: int = 10

    # Compression
    enable_compression: bool = True
    compression_level: int = 6


@dataclass
class Connection:
    """Represents a client connection."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ConnectionType = ConnectionType.WEBSOCKET
    state: ConnectionState = ConnectionState.CONNECTING

    # Client info
    client_id: Optional[str] = None
    user_id: Optional[str] = None
    organization_id: Optional[str] = None

    # Session association
    session_id: Optional[str] = None

    # Connection details
    remote_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: Optional[datetime] = None
    authenticated_at: Optional[datetime] = None
    last_activity_at: datetime = field(default_factory=datetime.utcnow)
    disconnected_at: Optional[datetime] = None

    # Statistics
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_authenticated(self) -> bool:
        """Check if connection is authenticated."""
        return self.state == ConnectionState.AUTHENTICATED

    def is_active(self) -> bool:
        """Check if connection is active."""
        return self.state in [
            ConnectionState.CONNECTED,
            ConnectionState.AUTHENTICATED,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "state": self.state.value,
            "client_id": self.client_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
        }


@dataclass
class Message:
    """Message sent through a connection."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.CUSTOM
    priority: MessagePriority = MessagePriority.NORMAL

    # Routing
    source_id: Optional[str] = None  # Connection/client ID
    target_id: Optional[str] = None  # Target connection ID
    session_id: Optional[str] = None

    # Content
    payload: Dict[str, Any] = field(default_factory=dict)
    binary_data: Optional[bytes] = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sequence: int = 0

    # Acknowledgment
    requires_ack: bool = False
    ack_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "session_id": self.session_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", "custom")),
            priority=MessagePriority(data.get("priority", 1)),
            source_id=data.get("source_id"),
            target_id=data.get("target_id"),
            session_id=data.get("session_id"),
            payload=data.get("payload", {}),
            sequence=data.get("sequence", 0),
        )


@dataclass
class Event:
    """Event in the event system."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.CUSTOM
    priority: EventPriority = EventPriority.NORMAL

    # Source
    source: str = ""
    source_id: Optional[str] = None

    # Context
    session_id: Optional[str] = None
    connection_id: Optional[str] = None
    call_id: Optional[str] = None

    # Data
    data: Dict[str, Any] = field(default_factory=dict)

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Processing
    processed: bool = False
    cancelled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "source": self.source,
            "session_id": self.session_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RealtimeSession:
    """Real-time voice agent session."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: SessionState = SessionState.INITIALIZING

    # Agent info
    agent_id: str = ""
    agent_name: str = ""

    # Call info
    call_id: Optional[str] = None
    caller_number: Optional[str] = None
    called_number: Optional[str] = None

    # Connection
    connection_id: Optional[str] = None

    # Organization
    organization_id: Optional[str] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # State
    is_agent_speaking: bool = False
    is_caller_speaking: bool = False
    current_intent: Optional[str] = None
    current_flow: Optional[str] = None

    # Turn tracking
    turn_count: int = 0
    last_turn_at: Optional[datetime] = None

    # Statistics
    total_audio_seconds: float = 0.0
    agent_audio_seconds: float = 0.0
    caller_audio_seconds: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Start the session."""
        self.state = SessionState.ACTIVE
        self.started_at = datetime.utcnow()

    def end(self) -> None:
        """End the session."""
        self.state = SessionState.ENDED
        self.ended_at = datetime.utcnow()

    def get_duration(self) -> float:
        """Get session duration in seconds."""
        if self.ended_at and self.started_at:
            return (self.ended_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.utcnow() - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "state": self.state.value,
            "agent_id": self.agent_id,
            "call_id": self.call_id,
            "created_at": self.created_at.isoformat(),
            "turn_count": self.turn_count,
            "duration": self.get_duration(),
        }


__all__ = [
    # Enums
    "ConnectionState",
    "ConnectionType",
    "MessageType",
    "MessagePriority",
    "EventType",
    "EventPriority",
    "SessionState",
    # Data classes
    "ConnectionConfig",
    "Connection",
    "Message",
    "Event",
    "RealtimeSession",
]
