"""
Telephony Base Types

This module defines the core data structures for telephony operations
including calls, sessions, recordings, and configuration.
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


class CallDirection(str, Enum):
    """Direction of a call."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class CallState(str, Enum):
    """State of a call."""
    INITIATED = "initiated"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    TRANSFERRING = "transferring"
    COMPLETED = "completed"
    FAILED = "failed"
    BUSY = "busy"
    NO_ANSWER = "no_answer"
    CANCELED = "canceled"


class CallEventType(str, Enum):
    """Types of call events."""
    INITIATED = "initiated"
    RINGING = "ringing"
    ANSWERED = "answered"
    HOLD_STARTED = "hold_started"
    HOLD_ENDED = "hold_ended"
    TRANSFER_STARTED = "transfer_started"
    TRANSFER_COMPLETED = "transfer_completed"
    TRANSFER_FAILED = "transfer_failed"
    DTMF_RECEIVED = "dtmf_received"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    RECORDING_STARTED = "recording_started"
    RECORDING_STOPPED = "recording_stopped"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


class RecordingFormat(str, Enum):
    """Recording audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    MULAW = "mulaw"
    ALAW = "alaw"


class RecordingState(str, Enum):
    """State of a recording."""
    PENDING = "pending"
    RECORDING = "recording"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class DTMFMode(str, Enum):
    """DTMF signaling mode."""
    INBAND = "inband"
    RFC2833 = "rfc2833"
    INFO = "info"


@dataclass
class DTMFEvent:
    """DTMF digit event."""

    digit: str
    duration_ms: int = 100
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        # Validate DTMF digit
        valid_digits = "0123456789*#ABCD"
        if self.digit not in valid_digits:
            raise ValueError(f"Invalid DTMF digit: {self.digit}")


@dataclass
class CallMetadata:
    """Metadata associated with a call."""

    # Caller info
    caller_name: Optional[str] = None
    caller_location: Optional[str] = None

    # Custom data
    custom_data: Dict[str, Any] = field(default_factory=dict)

    # Tags
    tags: List[str] = field(default_factory=list)

    # Agent info
    agent_id: Optional[str] = None
    organization_id: Optional[str] = None

    # Campaign/tracking
    campaign_id: Optional[str] = None
    tracking_source: Optional[str] = None


@dataclass
class CallLeg:
    """A leg of a call (one participant connection)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str = ""

    # Endpoints
    from_number: str = ""
    to_number: str = ""

    # State
    state: CallState = CallState.INITIATED
    direction: CallDirection = CallDirection.INBOUND

    # Timing
    started_at: Optional[datetime] = None
    answered_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Media
    is_muted: bool = False
    is_on_hold: bool = False

    # Provider info
    provider_leg_id: Optional[str] = None
    provider: str = ""


@dataclass
class CallEvent:
    """Event that occurred during a call."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str = ""
    type: CallEventType = CallEventType.INITIATED

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Data
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "call_id": self.call_id,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "error": self.error,
        }


@dataclass
class Recording:
    """Call recording."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str = ""

    # State
    state: RecordingState = RecordingState.PENDING

    # Format
    format: RecordingFormat = RecordingFormat.WAV
    sample_rate: int = 8000
    channels: int = 1

    # Storage
    storage_url: Optional[str] = None
    storage_path: Optional[str] = None
    file_size_bytes: int = 0

    # Duration
    duration_seconds: float = 0.0

    # Timing
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Options
    stereo_legs: bool = False  # Separate channels for each party
    transcribe: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "call_id": self.call_id,
            "state": self.state.value,
            "format": self.format.value,
            "storage_url": self.storage_url,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class Call:
    """Represents a phone call."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Basic info
    direction: CallDirection = CallDirection.INBOUND
    state: CallState = CallState.INITIATED

    # Phone numbers
    from_number: str = ""
    to_number: str = ""
    forwarded_from: Optional[str] = None

    # Provider info
    provider: str = ""
    provider_call_id: Optional[str] = None
    provider_account_id: Optional[str] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    answered_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Duration
    ring_duration_seconds: float = 0.0
    talk_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0

    # Status
    answer_state: Optional[str] = None  # human_answered, machine_answered, etc.
    hangup_cause: Optional[str] = None
    hangup_source: Optional[str] = None  # caller, callee, system

    # Media
    is_muted: bool = False
    is_on_hold: bool = False
    is_recording: bool = False

    # Legs
    legs: List[CallLeg] = field(default_factory=list)

    # Events
    events: List[CallEvent] = field(default_factory=list)

    # Recording
    recording: Optional[Recording] = None

    # Metadata
    metadata: CallMetadata = field(default_factory=CallMetadata)

    def add_event(self, event_type: CallEventType, data: Optional[Dict] = None) -> CallEvent:
        """Add an event to the call."""
        event = CallEvent(
            call_id=self.id,
            type=event_type,
            data=data or {},
        )
        self.events.append(event)
        return event

    def get_duration(self) -> float:
        """Get call duration in seconds."""
        if self.ended_at and self.answered_at:
            return (self.ended_at - self.answered_at).total_seconds()
        elif self.answered_at:
            return (datetime.utcnow() - self.answered_at).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "direction": self.direction.value,
            "state": self.state.value,
            "from_number": self.from_number,
            "to_number": self.to_number,
            "provider": self.provider,
            "created_at": self.created_at.isoformat(),
            "talk_duration_seconds": self.talk_duration_seconds,
            "total_duration_seconds": self.total_duration_seconds,
        }


@dataclass
class SessionConfig:
    """Configuration for a call session."""

    # Audio settings
    sample_rate: int = 8000
    audio_encoding: str = "mulaw"
    channels: int = 1

    # Timeouts
    ring_timeout_seconds: int = 30
    silence_timeout_seconds: int = 30
    max_duration_seconds: int = 3600

    # Recording
    record_call: bool = True
    recording_format: RecordingFormat = RecordingFormat.WAV

    # Features
    enable_dtmf: bool = True
    dtmf_mode: DTMFMode = DTMFMode.RFC2833

    # Speech detection
    enable_speech_detection: bool = True
    speech_detection_sensitivity: float = 0.5

    # Answering machine detection
    enable_amd: bool = False
    amd_timeout_seconds: int = 5


@dataclass
class CallSession:
    """Active call session with state and media."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call: Call = field(default_factory=Call)
    config: SessionConfig = field(default_factory=SessionConfig)

    # Session state
    is_active: bool = False
    is_agent_speaking: bool = False
    is_caller_speaking: bool = False

    # Buffers
    audio_buffer_size: int = 0

    # DTMF collection
    dtmf_buffer: str = ""
    dtmf_events: List[DTMFEvent] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity_at: datetime = field(default_factory=datetime.utcnow)

    def clear_dtmf_buffer(self) -> str:
        """Clear and return DTMF buffer."""
        buffer = self.dtmf_buffer
        self.dtmf_buffer = ""
        return buffer

    def add_dtmf(self, digit: str) -> None:
        """Add DTMF digit to buffer."""
        event = DTMFEvent(digit=digit)
        self.dtmf_events.append(event)
        self.dtmf_buffer += digit
        self.last_activity_at = datetime.utcnow()


@dataclass
class ProviderConfig:
    """Configuration for a telephony provider."""

    name: str = ""
    enabled: bool = True

    # Credentials
    account_sid: Optional[str] = None
    auth_token: Optional[str] = None
    api_key: Optional[str] = None

    # Endpoints
    api_url: Optional[str] = None
    webhook_url: Optional[str] = None
    stream_url: Optional[str] = None

    # Phone numbers
    default_from_number: Optional[str] = None
    phone_numbers: List[str] = field(default_factory=list)

    # Limits
    max_concurrent_calls: int = 100
    calls_per_second: float = 10.0

    # Features
    supports_recording: bool = True
    supports_streaming: bool = True
    supports_dtmf: bool = True
    supports_amd: bool = True

    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelephonyConfig:
    """Main telephony configuration."""

    # Provider configs
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    default_provider: str = "twilio"

    # Global settings
    default_session_config: SessionConfig = field(default_factory=SessionConfig)

    # Webhook settings
    webhook_base_url: str = ""
    webhook_timeout_seconds: int = 30

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Logging
    log_calls: bool = True
    log_events: bool = True

    # Callbacks
    on_call_started: Optional[Callable[[Call], None]] = None
    on_call_ended: Optional[Callable[[Call], None]] = None
    on_call_error: Optional[Callable[[Call, str], None]] = None


__all__ = [
    # Enums
    "CallDirection",
    "CallState",
    "CallEventType",
    "RecordingFormat",
    "RecordingState",
    "DTMFMode",
    # Data classes
    "DTMFEvent",
    "CallMetadata",
    "CallLeg",
    "CallEvent",
    "Recording",
    "Call",
    "SessionConfig",
    "CallSession",
    "ProviderConfig",
    "TelephonyConfig",
]
