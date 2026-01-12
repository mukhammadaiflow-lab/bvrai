"""Signaling message models."""

from datetime import datetime
from enum import Enum
from typing import Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SignalingMessageType(str, Enum):
    """Types of signaling messages."""

    # Connection management
    CONNECT = "connect"
    CONNECTED = "connected"
    DISCONNECT = "disconnect"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"

    # WebRTC signaling
    OFFER = "offer"
    ANSWER = "answer"
    ICE_CANDIDATE = "ice_candidate"

    # Session management
    SESSION_START = "session_start"
    SESSION_STARTED = "session_started"
    SESSION_END = "session_end"
    SESSION_ENDED = "session_ended"

    # Call events
    CALL_RINGING = "call_ringing"
    CALL_CONNECTED = "call_connected"
    CALL_ENDED = "call_ended"

    # Audio/Transcript events
    TRANSCRIPT = "transcript"
    AGENT_SPEAKING = "agent_speaking"
    AGENT_SILENT = "agent_silent"
    USER_SPEAKING = "user_speaking"
    USER_SILENT = "user_silent"

    # Configuration
    CONFIG_UPDATE = "config_update"


class SignalingMessage(BaseModel):
    """Base signaling message."""

    type: SignalingMessageType
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Optional[dict] = None


class ConnectPayload(BaseModel):
    """Payload for connect message."""

    agent_id: UUID
    api_key: Optional[str] = None
    metadata: Optional[dict] = None


class ConnectedPayload(BaseModel):
    """Payload for connected message."""

    session_id: str
    ice_servers: list[dict]


class OfferPayload(BaseModel):
    """Payload for SDP offer."""

    sdp: str
    type: str = "offer"


class AnswerPayload(BaseModel):
    """Payload for SDP answer."""

    sdp: str
    type: str = "answer"


class IceCandidatePayload(BaseModel):
    """Payload for ICE candidate."""

    candidate: str
    sdp_mid: Optional[str] = None
    sdp_m_line_index: Optional[int] = None
    username_fragment: Optional[str] = None


class TranscriptPayload(BaseModel):
    """Payload for transcript update."""

    speaker: str  # "user" or "agent"
    text: str
    is_final: bool = False
    confidence: Optional[float] = None


class ErrorPayload(BaseModel):
    """Payload for error message."""

    code: str
    message: str
    details: Optional[dict] = None


class SessionConfig(BaseModel):
    """Session configuration from agent."""

    agent_id: UUID
    agent_name: str
    voice_id: str
    language: str = "en-US"
    greeting_message: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: list[dict] = Field(default_factory=list)
