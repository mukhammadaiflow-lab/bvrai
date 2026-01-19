"""
Call Orchestrator Base Types Module

This module defines core types and data structures for
the call orchestration system.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Union,
)


class CallState(str, Enum):
    """States in the call lifecycle."""

    # Pre-connection states
    INITIALIZING = "initializing"
    QUEUED = "queued"
    RINGING = "ringing"

    # Active states
    CONNECTED = "connected"
    AGENT_SPEAKING = "agent_speaking"
    CUSTOMER_SPEAKING = "customer_speaking"
    PROCESSING = "processing"
    SILENCE = "silence"

    # Transfer/hold states
    TRANSFERRING = "transferring"
    ON_HOLD = "on_hold"

    # Terminal states
    COMPLETED = "completed"
    FAILED = "failed"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    CANCELED = "canceled"
    VOICEMAIL = "voicemail"


class CallDirection(str, Enum):
    """Call direction."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"


class ParticipantRole(str, Enum):
    """Role of a participant in the call."""

    AGENT = "agent"
    CUSTOMER = "customer"
    SUPERVISOR = "supervisor"


class EventType(str, Enum):
    """Types of orchestration events."""

    # Call lifecycle
    CALL_STARTED = "call.started"
    CALL_INITIATED = "call.initiated"
    CALL_RINGING = "call.ringing"
    CALL_ANSWERED = "call.answered"
    CALL_CONNECTED = "call.connected"
    CALL_ENDED = "call.ended"
    CALL_FAILED = "call.failed"
    CALL_TRANSFERRED = "call.transferred"

    # Audio events
    AUDIO_RECEIVED = "audio.received"
    AUDIO_SENT = "audio.sent"
    SPEECH_STARTED = "speech.started"
    SPEECH_ENDED = "speech.ended"
    SILENCE_DETECTED = "silence.detected"

    # Transcription events
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"
    TRANSCRIPT_UPDATED = "transcript.updated"

    # Agent events
    AGENT_THINKING = "agent.thinking"
    AGENT_RESPONDING = "agent.responding"
    AGENT_RESPONSE_COMPLETE = "agent.response_complete"
    AGENT_INTERRUPTED = "agent.interrupted"

    # Turn events
    END_OF_TURN = "turn.ended"
    INTERRUPTION_DETECTED = "interruption.detected"

    # Function events
    FUNCTION_CALLED = "function.called"
    FUNCTION_COMPLETED = "function.completed"
    FUNCTION_FAILED = "function.failed"

    # State events
    STATE_CHANGED = "state.changed"

    # Special events
    VOICEMAIL_DETECTED = "voicemail.detected"
    DTMF_RECEIVED = "dtmf.received"
    TRANSFER_INITIATED = "transfer.initiated"
    TRANSFER_COMPLETED = "transfer.completed"

    # Error events
    ERROR_OCCURRED = "error.occurred"


@dataclass
class CallConfig:
    """Configuration for a call session."""

    # Agent configuration
    agent_id: str
    agent_name: str = ""
    system_prompt: str = ""
    first_message: Optional[str] = None

    # Voice configuration
    voice_provider: str = "elevenlabs"
    voice_id: str = ""
    voice_speed: float = 1.0

    # LLM configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 150

    # STT configuration
    stt_provider: str = "deepgram"
    language: str = "en-US"

    # Behavior configuration
    interruption_enabled: bool = True
    interruption_sensitivity: float = 0.5
    silence_timeout_seconds: int = 10
    max_call_duration_seconds: int = 1800

    # Knowledge base IDs
    knowledge_base_ids: List[str] = field(default_factory=list)

    # Functions available to agent
    functions: List[Dict[str, Any]] = field(default_factory=list)

    # Recording
    record_call: bool = True

    # Industry context
    industry: Optional[str] = None

    # Custom context variables
    context_variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallMetrics:
    """Metrics for a call session."""

    # Duration
    total_duration_seconds: float = 0.0
    agent_talk_time_seconds: float = 0.0
    customer_talk_time_seconds: float = 0.0
    silence_time_seconds: float = 0.0
    processing_time_seconds: float = 0.0

    # Latency
    time_to_connect_ms: float = 0.0
    time_to_first_response_ms: float = 0.0
    avg_stt_latency_ms: float = 0.0
    avg_llm_latency_ms: float = 0.0
    avg_tts_latency_ms: float = 0.0

    # Counts
    agent_turn_count: int = 0
    customer_turn_count: int = 0
    interruption_count: int = 0
    function_calls: int = 0
    transfer_count: int = 0

    # LLM metrics
    total_llm_tokens: int = 0
    total_llm_requests: int = 0

    # Audio metrics
    audio_bytes_received: int = 0
    audio_bytes_sent: int = 0

    # Quality
    transcription_confidence_avg: float = 0.0


@dataclass
class TranscriptEntry:
    """An entry in the call transcript."""

    role: ParticipantRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Timing
    start_time_ms: int = 0
    end_time_ms: int = 0
    duration_ms: int = 0

    # Metadata
    confidence: float = 1.0
    is_final: bool = True
    function_call: Optional[Dict[str, Any]] = None


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: ParticipantRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # For agent turns
    thinking_time_ms: int = 0
    generation_time_ms: int = 0
    speech_time_ms: int = 0

    # For customer turns
    transcription_confidence: float = 1.0

    # Function calls
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    function_results: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    was_interrupted: bool = False
    tokens_used: int = 0


@dataclass
class OrchestratorEvent:
    """Event in the orchestration system."""

    id: str
    session_id: str
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)

    # Context
    state_before: Optional[CallState] = None
    state_after: Optional[CallState] = None

    # For error events
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    # Aliases for backwards compatibility
    @property
    def type(self) -> EventType:
        return self.event_type

    @property
    def call_id(self) -> str:
        return self.session_id


@dataclass
class CallSession:
    """A call session being orchestrated."""

    # Identifiers
    id: str
    organization_id: str
    agent_id: str

    # Direction and participants
    direction: CallDirection
    from_number: str
    to_number: str

    # State
    state: CallState = CallState.INITIALIZING
    previous_state: Optional[CallState] = None

    # Configuration
    config: CallConfig = field(default_factory=lambda: CallConfig(agent_id=""))

    # Conversation
    transcript: List[TranscriptEntry] = field(default_factory=list)
    turns: List[ConversationTurn] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)  # LLM message format

    # Current state
    current_speaker: Optional[ParticipantRole] = None
    pending_audio: bytes = b""
    pending_transcript: str = ""

    # Metrics
    metrics: CallMetrics = field(default_factory=CallMetrics)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: Optional[datetime] = None
    answered_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Results
    end_reason: Optional[str] = None
    recording_url: Optional[str] = None
    voicemail_detected: bool = False

    # Event history
    events: List[OrchestratorEvent] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Event counter for generating IDs
    _event_counter: int = field(default=0, repr=False)

    def add_event(self, event_type: EventType, data: Optional[Dict] = None) -> OrchestratorEvent:
        """Add an event to the session."""
        import uuid
        self._event_counter += 1
        event = OrchestratorEvent(
            id=f"evt_{uuid.uuid4().hex[:12]}",
            session_id=self.id,
            event_type=event_type,
            data=data or {},
            state_before=self.previous_state,
            state_after=self.state,
        )
        self.events.append(event)
        return event

    def transition_state(
        self,
        new_state: CallState,
        reason: Optional[str] = None,
    ) -> None:
        """Transition to a new state."""
        self.previous_state = self.state
        self.state = new_state
        self.add_event(EventType.STATE_CHANGED, {
            "from_state": self.previous_state.value if self.previous_state else None,
            "to_state": new_state.value,
            "reason": reason,
        })

    def add_transcript_entry(
        self,
        role: ParticipantRole,
        content: str,
        **kwargs,
    ) -> TranscriptEntry:
        """Add a transcript entry."""
        entry = TranscriptEntry(role=role, content=content, **kwargs)
        self.transcript.append(entry)
        return entry

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a conversation turn."""
        self.turns.append(turn)

        # Update messages for LLM
        role_map = {
            ParticipantRole.AGENT: "assistant",
            ParticipantRole.CUSTOMER: "user",
        }
        self.messages.append({
            "role": role_map.get(turn.role, "user"),
            "content": turn.content,
        })

        # Update metrics
        if turn.role == ParticipantRole.AGENT:
            self.metrics.agent_turn_count += 1
            self.metrics.agent_talk_time_seconds += turn.speech_time_ms / 1000
            self.metrics.total_llm_tokens += turn.tokens_used
        else:
            self.metrics.customer_turn_count += 1

    def get_conversation_context(self) -> List[Dict[str, Any]]:
        """Get conversation history in LLM message format."""
        context = []

        # Add system prompt
        if self.config.system_prompt:
            context.append({
                "role": "system",
                "content": self.config.system_prompt,
            })

        # Add conversation history
        context.extend(self.messages)

        return context

    @property
    def duration_seconds(self) -> float:
        """Get call duration in seconds."""
        if not self.started_at:
            return 0.0
        end = self.ended_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    @property
    def is_active(self) -> bool:
        """Check if call is active."""
        return self.state not in {
            CallState.COMPLETED,
            CallState.FAILED,
            CallState.NO_ANSWER,
            CallState.BUSY,
            CallState.CANCELED,
            CallState.VOICEMAIL,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "agent_id": self.agent_id,
            "direction": self.direction.value,
            "from_number": self.from_number,
            "to_number": self.to_number,
            "state": self.state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "config": {
                "agent_id": self.config.agent_id,
                "agent_name": self.config.agent_name,
                "system_prompt": self.config.system_prompt,
                "voice_provider": self.config.voice_provider,
                "voice_id": self.config.voice_id,
                "llm_provider": self.config.llm_provider,
                "llm_model": self.config.llm_model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "industry": self.config.industry,
            },
            "transcript": [
                {
                    "role": entry.role.value,
                    "content": entry.content,
                    "timestamp": entry.timestamp.isoformat(),
                    "confidence": entry.confidence,
                }
                for entry in self.transcript
            ],
            "messages": self.messages,
            "metrics": {
                "total_duration_seconds": self.metrics.total_duration_seconds,
                "agent_talk_time_seconds": self.metrics.agent_talk_time_seconds,
                "customer_talk_time_seconds": self.metrics.customer_talk_time_seconds,
                "interruption_count": self.metrics.interruption_count,
                "function_calls": self.metrics.function_calls,
                "transfer_count": self.metrics.transfer_count,
            },
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat(),
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "end_reason": self.end_reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CallSession":
        """Deserialize session from dictionary."""
        from datetime import datetime

        config = CallConfig(
            agent_id=data.get("config", {}).get("agent_id", ""),
            agent_name=data.get("config", {}).get("agent_name", ""),
            system_prompt=data.get("config", {}).get("system_prompt", ""),
            voice_provider=data.get("config", {}).get("voice_provider", "elevenlabs"),
            voice_id=data.get("config", {}).get("voice_id", ""),
            llm_provider=data.get("config", {}).get("llm_provider", "openai"),
            llm_model=data.get("config", {}).get("llm_model", "gpt-4"),
            temperature=data.get("config", {}).get("temperature", 0.7),
            max_tokens=data.get("config", {}).get("max_tokens", 150),
            industry=data.get("config", {}).get("industry"),
        )

        session = cls(
            id=data["id"],
            organization_id=data["organization_id"],
            agent_id=data["agent_id"],
            direction=CallDirection(data["direction"]),
            from_number=data["from_number"],
            to_number=data["to_number"],
            state=CallState(data["state"]),
            config=config,
            messages=data.get("messages", []),
            metadata=data.get("metadata", {}),
        )

        # Restore state
        if data.get("previous_state"):
            session.previous_state = CallState(data["previous_state"])

        # Restore timestamps
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("connected_at"):
            session.connected_at = datetime.fromisoformat(data["connected_at"])
        if data.get("ended_at"):
            session.ended_at = datetime.fromisoformat(data["ended_at"])

        session.end_reason = data.get("end_reason")

        # Restore metrics
        metrics_data = data.get("metrics", {})
        session.metrics.total_duration_seconds = metrics_data.get("total_duration_seconds", 0)
        session.metrics.agent_talk_time_seconds = metrics_data.get("agent_talk_time_seconds", 0)
        session.metrics.customer_talk_time_seconds = metrics_data.get("customer_talk_time_seconds", 0)
        session.metrics.interruption_count = metrics_data.get("interruption_count", 0)
        session.metrics.function_calls = metrics_data.get("function_calls", 0)
        session.metrics.transfer_count = metrics_data.get("transfer_count", 0)

        # Restore transcript
        for entry_data in data.get("transcript", []):
            entry = TranscriptEntry(
                role=ParticipantRole(entry_data["role"]),
                content=entry_data["content"],
                timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                confidence=entry_data.get("confidence", 1.0),
            )
            session.transcript.append(entry)

        return session


@dataclass
class PipelineContext:
    """Context passed through the processing pipeline."""

    session: CallSession

    # Current audio chunk
    audio_chunk: Optional[bytes] = None
    audio_format: str = "pcm16"
    sample_rate: int = 16000

    # Current transcript
    partial_transcript: str = ""
    final_transcript: str = ""
    transcript_confidence: float = 0.0

    # Agent response
    agent_response: str = ""
    agent_audio: Optional[bytes] = None

    # Function calls
    pending_function_calls: List[Dict[str, Any]] = field(default_factory=list)
    function_results: List[Dict[str, Any]] = field(default_factory=list)

    # Flags
    speech_detected: bool = False
    silence_detected: bool = False
    interruption_detected: bool = False
    end_of_turn: bool = False

    # Timing
    pipeline_start_time: float = 0.0
    stt_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0


__all__ = [
    "CallState",
    "CallDirection",
    "ParticipantRole",
    "EventType",
    "CallConfig",
    "CallMetrics",
    "TranscriptEntry",
    "ConversationTurn",
    "OrchestratorEvent",
    "CallSession",
    "PipelineContext",
]
