"""
Data Models for Streaming Orchestrator Service.

This module defines all data models, events, and schemas used throughout
the ultra-low latency streaming pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid
import time

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums and Constants
# =============================================================================


class SessionState(str, Enum):
    """State of a streaming session."""

    CREATED = "created"
    CONNECTING = "connecting"
    ACTIVE = "active"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    PAUSED = "paused"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class AudioFormat(str, Enum):
    """Supported audio formats."""

    PCM_16KHZ_16BIT = "pcm_16khz_16bit"
    PCM_8KHZ_16BIT = "pcm_8khz_16bit"
    PCM_24KHZ_16BIT = "pcm_24khz_16bit"
    PCM_44KHZ_16BIT = "pcm_44khz_16bit"
    MULAW_8KHZ = "mulaw_8khz"
    ALAW_8KHZ = "alaw_8khz"
    OPUS_48KHZ = "opus_48khz"
    MP3 = "mp3"
    AAC = "aac"


class TranscriptStatus(str, Enum):
    """Status of transcript processing."""

    PARTIAL = "partial"
    STABLE = "stable"
    FINAL = "final"


class EventType(str, Enum):
    """Types of events in the streaming pipeline."""

    # Session events
    SESSION_CREATED = "session_created"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    SESSION_ERROR = "session_error"

    # Audio events
    AUDIO_RECEIVED = "audio_received"
    AUDIO_SENT = "audio_sent"
    AUDIO_LEVEL = "audio_level"

    # Speech events
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    VAD_EVENT = "vad_event"

    # Transcription events
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_STABLE = "transcript_stable"
    TRANSCRIPT_FINAL = "transcript_final"

    # LLM events
    LLM_START = "llm_start"
    LLM_TOKEN = "llm_token"
    LLM_COMPLETE = "llm_complete"
    LLM_ERROR = "llm_error"

    # Speculative execution events
    SPECULATION_START = "speculation_start"
    SPECULATION_VALIDATED = "speculation_validated"
    SPECULATION_ABANDONED = "speculation_abandoned"

    # TTS events
    TTS_START = "tts_start"
    TTS_AUDIO = "tts_audio"
    TTS_COMPLETE = "tts_complete"
    TTS_ERROR = "tts_error"

    # Interruption events
    INTERRUPTION_DETECTED = "interruption_detected"
    INTERRUPTION_HANDLED = "interruption_handled"

    # Turn events
    TURN_START = "turn_start"
    TURN_END = "turn_end"

    # Latency events
    LATENCY_WARNING = "latency_warning"
    LATENCY_CRITICAL = "latency_critical"

    # System events
    PROVIDER_FALLBACK = "provider_fallback"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_CLOSE = "circuit_close"


# =============================================================================
# Audio Models
# =============================================================================


@dataclass
class AudioChunk:
    """A chunk of audio data with metadata."""

    data: bytes
    timestamp_ms: float
    duration_ms: float
    format: AudioFormat = AudioFormat.PCM_16KHZ_16BIT
    sample_rate: int = 16000
    channels: int = 1
    sequence_number: int = 0

    # Audio analysis
    rms_level: Optional[float] = None
    peak_level: Optional[float] = None
    is_speech: Optional[bool] = None

    @property
    def size_bytes(self) -> int:
        """Get size of audio data in bytes."""
        return len(self.data)

    @property
    def samples(self) -> int:
        """Get number of samples."""
        bytes_per_sample = 2  # 16-bit
        return len(self.data) // (bytes_per_sample * self.channels)


@dataclass
class AudioBuffer:
    """Buffer for accumulating audio chunks."""

    chunks: List[AudioChunk] = field(default_factory=list)
    max_duration_ms: float = 30000.0
    created_at: float = field(default_factory=time.time)

    def add(self, chunk: AudioChunk) -> None:
        """Add a chunk to the buffer."""
        self.chunks.append(chunk)

        # Trim if exceeding max duration
        total_duration = sum(c.duration_ms for c in self.chunks)
        while total_duration > self.max_duration_ms and self.chunks:
            removed = self.chunks.pop(0)
            total_duration -= removed.duration_ms

    @property
    def total_duration_ms(self) -> float:
        """Get total duration of buffered audio."""
        return sum(c.duration_ms for c in self.chunks)

    @property
    def total_bytes(self) -> int:
        """Get total size of buffered audio."""
        return sum(c.size_bytes for c in self.chunks)

    def get_audio_data(self) -> bytes:
        """Get concatenated audio data."""
        return b"".join(c.data for c in self.chunks)

    def clear(self) -> None:
        """Clear the buffer."""
        self.chunks.clear()


# =============================================================================
# Transcript Models
# =============================================================================


@dataclass
class TranscriptWord:
    """A single word in a transcript with timing."""

    word: str
    start_ms: float
    end_ms: float
    confidence: float = 1.0
    speaker: Optional[str] = None


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""

    text: str
    status: TranscriptStatus
    confidence: float
    words: List[TranscriptWord] = field(default_factory=list)
    start_ms: float = 0.0
    end_ms: float = 0.0
    language: Optional[str] = None
    speaker: Optional[str] = None

    # Stability metrics (for speculative execution)
    stability_score: float = 0.0
    change_count: int = 0
    last_change_at: float = field(default_factory=time.time)


@dataclass
class TranscriptResult:
    """Complete transcription result."""

    segments: List[TranscriptSegment] = field(default_factory=list)
    full_text: str = ""
    is_final: bool = False
    processing_time_ms: float = 0.0
    provider: str = ""

    @property
    def word_count(self) -> int:
        """Get total word count."""
        return sum(len(s.words) for s in self.segments)


# =============================================================================
# LLM Models
# =============================================================================


@dataclass
class LLMMessage:
    """A message in the LLM conversation."""

    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMResponse:
    """Response from LLM generation."""

    text: str
    tokens_generated: int = 0
    finish_reason: Optional[str] = None
    model: str = ""
    provider: str = ""

    # Timing
    first_token_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Speculative execution
    is_speculative: bool = False
    speculation_validated: bool = False


@dataclass
class LLMStreamToken:
    """A single token from LLM streaming."""

    token: str
    token_id: Optional[int] = None
    logprob: Optional[float] = None
    is_first: bool = False
    is_last: bool = False
    cumulative_text: str = ""
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# TTS Models
# =============================================================================


@dataclass
class TTSRequest:
    """Request for TTS synthesis."""

    text: str
    voice_id: str
    speed: float = 1.0
    pitch: float = 1.0
    output_format: AudioFormat = AudioFormat.PCM_16KHZ_16BIT
    sample_rate: int = 16000

    # Streaming options
    stream: bool = True
    chunk_size_ms: int = 100

    # Emotion/style
    emotion: Optional[str] = None
    style: Optional[str] = None


@dataclass
class TTSChunk:
    """A chunk of TTS audio output."""

    audio: AudioChunk
    text_position: int  # Character position in original text
    text_length: int    # Total text length
    is_first: bool = False
    is_last: bool = False

    @property
    def progress(self) -> float:
        """Get synthesis progress (0.0 to 1.0)."""
        if self.text_length == 0:
            return 1.0
        return min(1.0, self.text_position / self.text_length)


# =============================================================================
# Session Models
# =============================================================================


class SessionConfig(BaseModel):
    """Configuration for a streaming session."""

    # Identifiers
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    agent_id: str
    call_id: Optional[str] = None

    # Audio configuration
    input_format: AudioFormat = AudioFormat.PCM_16KHZ_16BIT
    input_sample_rate: int = 16000
    output_format: AudioFormat = AudioFormat.PCM_16KHZ_16BIT
    output_sample_rate: int = 16000

    # Provider overrides
    asr_provider: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    tts_provider: Optional[str] = None
    tts_voice: Optional[str] = None

    # Behavior options
    enable_speculative_execution: bool = True
    enable_interruption: bool = True
    enable_backchanneling: bool = False

    # Timing options
    silence_timeout_ms: int = 700
    max_turn_duration_ms: int = 60000

    # Context
    system_prompt: Optional[str] = None
    initial_context: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class SessionMetrics:
    """Real-time metrics for a session."""

    # Latency metrics (all in milliseconds)
    avg_asr_latency_ms: float = 0.0
    avg_llm_first_token_ms: float = 0.0
    avg_tts_first_audio_ms: float = 0.0
    avg_e2e_latency_ms: float = 0.0

    p50_e2e_latency_ms: float = 0.0
    p95_e2e_latency_ms: float = 0.0
    p99_e2e_latency_ms: float = 0.0

    # Count metrics
    total_turns: int = 0
    total_audio_received_ms: float = 0.0
    total_audio_sent_ms: float = 0.0
    total_words_transcribed: int = 0
    total_tokens_generated: int = 0

    # Speculative execution metrics
    speculations_attempted: int = 0
    speculations_validated: int = 0
    speculations_abandoned: int = 0

    # Interruption metrics
    interruptions_detected: int = 0
    interruptions_handled: int = 0

    # Error metrics
    asr_errors: int = 0
    llm_errors: int = 0
    tts_errors: int = 0
    fallbacks_triggered: int = 0

    @property
    def speculation_success_rate(self) -> float:
        """Get speculative execution success rate."""
        if self.speculations_attempted == 0:
            return 0.0
        return self.speculations_validated / self.speculations_attempted

    def update_latency(self, e2e_latency_ms: float, latencies: List[float]) -> None:
        """Update latency metrics with new measurement."""
        if not latencies:
            return

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        self.avg_e2e_latency_ms = sum(latencies_sorted) / n
        self.p50_e2e_latency_ms = latencies_sorted[int(n * 0.5)]
        self.p95_e2e_latency_ms = latencies_sorted[int(n * 0.95)] if n > 1 else latencies_sorted[-1]
        self.p99_e2e_latency_ms = latencies_sorted[int(n * 0.99)] if n > 1 else latencies_sorted[-1]


@dataclass
class Turn:
    """A single conversation turn."""

    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    # Input
    user_audio_duration_ms: float = 0.0
    user_transcript: str = ""
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)

    # Output
    assistant_response: str = ""
    assistant_audio_duration_ms: float = 0.0

    # Latency breakdown (all in milliseconds)
    speech_start_at: Optional[float] = None
    speech_end_at: Optional[float] = None
    asr_complete_at: Optional[float] = None
    llm_first_token_at: Optional[float] = None
    llm_complete_at: Optional[float] = None
    tts_first_audio_at: Optional[float] = None
    tts_complete_at: Optional[float] = None

    # Computed latencies
    @property
    def asr_latency_ms(self) -> Optional[float]:
        if self.speech_end_at and self.asr_complete_at:
            return (self.asr_complete_at - self.speech_end_at) * 1000
        return None

    @property
    def llm_first_token_latency_ms(self) -> Optional[float]:
        if self.asr_complete_at and self.llm_first_token_at:
            return (self.llm_first_token_at - self.asr_complete_at) * 1000
        return None

    @property
    def tts_first_audio_latency_ms(self) -> Optional[float]:
        if self.llm_first_token_at and self.tts_first_audio_at:
            return (self.tts_first_audio_at - self.llm_first_token_at) * 1000
        return None

    @property
    def e2e_latency_ms(self) -> Optional[float]:
        """End-to-end latency from speech end to first audio."""
        if self.speech_end_at and self.tts_first_audio_at:
            return (self.tts_first_audio_at - self.speech_end_at) * 1000
        return None

    # Speculative execution
    was_speculative: bool = False
    speculation_validated: bool = False

    # Interruption
    was_interrupted: bool = False
    interruption_at_ms: Optional[float] = None


# =============================================================================
# Event Models
# =============================================================================


@dataclass
class StreamEvent:
    """An event in the streaming pipeline."""

    event_type: EventType
    session_id: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    # Context
    turn_id: Optional[str] = None
    sequence_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "turn_id": self.turn_id,
            "sequence_number": self.sequence_number,
        }


# =============================================================================
# API Models
# =============================================================================


class CreateSessionRequest(BaseModel):
    """Request to create a new streaming session."""

    tenant_id: str = Field(..., description="Tenant identifier")
    agent_id: str = Field(..., description="Agent identifier")
    call_id: Optional[str] = Field(None, description="Associated call ID")

    # Audio configuration
    input_format: AudioFormat = Field(
        default=AudioFormat.PCM_16KHZ_16BIT,
        description="Input audio format",
    )
    output_format: AudioFormat = Field(
        default=AudioFormat.PCM_16KHZ_16BIT,
        description="Output audio format",
    )

    # Provider overrides
    asr_provider: Optional[str] = Field(None, description="Override ASR provider")
    llm_provider: Optional[str] = Field(None, description="Override LLM provider")
    llm_model: Optional[str] = Field(None, description="Override LLM model")
    tts_provider: Optional[str] = Field(None, description="Override TTS provider")
    tts_voice: Optional[str] = Field(None, description="Override TTS voice")

    # Behavior
    enable_speculative_execution: bool = Field(
        default=True,
        description="Enable speculative LLM execution",
    )
    enable_interruption: bool = Field(
        default=True,
        description="Enable interruption detection",
    )

    # Context
    system_prompt: Optional[str] = Field(None, description="System prompt override")
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Previous conversation history",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class SessionResponse(BaseModel):
    """Response containing session information."""

    session_id: str
    state: SessionState
    websocket_url: str
    created_at: datetime
    config: SessionConfig
    metrics: Optional[Dict[str, Any]] = None


class SessionListResponse(BaseModel):
    """Response containing list of sessions."""

    sessions: List[SessionResponse]
    total: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    uptime_seconds: float
    active_sessions: int
    providers: Dict[str, str]


# =============================================================================
# WebSocket Message Models
# =============================================================================


class WSMessageType(str, Enum):
    """WebSocket message types."""

    # Client -> Server
    AUDIO = "audio"
    CONFIG = "config"
    CONTROL = "control"
    PING = "ping"

    # Server -> Client
    TRANSCRIPT = "transcript"
    RESPONSE = "response"
    AUDIO_OUT = "audio_out"
    EVENT = "event"
    ERROR = "error"
    PONG = "pong"


class WSMessage(BaseModel):
    """WebSocket message structure."""

    type: WSMessageType
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    sequence: int = 0


class WSAudioMessage(BaseModel):
    """Audio message from WebSocket."""

    type: WSMessageType = WSMessageType.AUDIO
    audio_base64: str
    format: AudioFormat = AudioFormat.PCM_16KHZ_16BIT
    timestamp: float = Field(default_factory=time.time)
    sequence: int = 0


class WSControlMessage(BaseModel):
    """Control message for WebSocket."""

    type: WSMessageType = WSMessageType.CONTROL
    action: str  # "pause", "resume", "stop", "interrupt", "clear"
    params: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Export All Models
# =============================================================================


__all__ = [
    # Enums
    "SessionState",
    "AudioFormat",
    "TranscriptStatus",
    "EventType",
    "WSMessageType",
    # Audio
    "AudioChunk",
    "AudioBuffer",
    # Transcript
    "TranscriptWord",
    "TranscriptSegment",
    "TranscriptResult",
    # LLM
    "LLMMessage",
    "LLMResponse",
    "LLMStreamToken",
    # TTS
    "TTSRequest",
    "TTSChunk",
    # Session
    "SessionConfig",
    "SessionMetrics",
    "Turn",
    # Events
    "StreamEvent",
    # API
    "CreateSessionRequest",
    "SessionResponse",
    "SessionListResponse",
    "HealthResponse",
    # WebSocket
    "WSMessage",
    "WSAudioMessage",
    "WSControlMessage",
]
