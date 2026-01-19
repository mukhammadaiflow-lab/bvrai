"""
Conversation Engine - Base Classes and Types

This module defines the foundational types for the conversation engine.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)


class ConversationState(str, Enum):
    """High-level conversation states."""

    # Initial states
    INITIALIZED = "initialized"
    GREETING = "greeting"

    # Active conversation
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    WAITING_INPUT = "waiting_input"

    # Slot filling
    COLLECTING_SLOT = "collecting_slot"
    VALIDATING_SLOT = "validating_slot"

    # Flow states
    EXECUTING_ACTION = "executing_action"
    BRANCHING = "branching"
    TRANSFERRING = "transferring"

    # Hold states
    ON_HOLD = "on_hold"
    WAITING_CALLBACK = "waiting_callback"

    # Terminal states
    COMPLETED = "completed"
    TRANSFERRED = "transferred"
    ABANDONED = "abandoned"
    ERROR = "error"


class TurnState(str, Enum):
    """State within a conversation turn."""

    STARTED = "started"
    USER_INPUT_RECEIVED = "user_input_received"
    INTENT_DETECTED = "intent_detected"
    SLOTS_EXTRACTED = "slots_extracted"
    CONTEXT_UPDATED = "context_updated"
    RESPONSE_GENERATED = "response_generated"
    RESPONSE_DELIVERED = "response_delivered"
    COMPLETED = "completed"
    FAILED = "failed"


class ResponseType(str, Enum):
    """Types of responses from the conversation engine."""

    TEXT = "text"                    # Plain text response
    AUDIO = "audio"                  # Pre-recorded audio
    SSML = "ssml"                    # SSML for TTS
    ACTION = "action"                # Execute an action
    TRANSFER = "transfer"            # Transfer to agent/queue
    HOLD = "hold"                    # Put on hold with message
    END = "end"                      # End conversation
    SILENT = "silent"                # No response (listening)
    THINKING = "thinking"            # Filler while processing
    CLARIFY = "clarify"              # Ask for clarification


class ConversationEventType(str, Enum):
    """Types of conversation events."""

    # Session lifecycle
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    SESSION_ERROR = "session_error"

    # Turn events
    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"

    # Input/Output events
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    SILENCE_DETECTED = "silence_detected"

    # Intent events
    INTENT_MATCHED = "intent_matched"
    INTENT_UNKNOWN = "intent_unknown"
    INTENT_CHANGED = "intent_changed"

    # Slot events
    SLOT_REQUESTED = "slot_requested"
    SLOT_FILLED = "slot_filled"
    SLOT_VALIDATION_FAILED = "slot_validation_failed"

    # Flow events
    FLOW_STARTED = "flow_started"
    FLOW_NODE_ENTERED = "flow_node_entered"
    FLOW_TRANSITION = "flow_transition"
    FLOW_COMPLETED = "flow_completed"
    FLOW_ERROR = "flow_error"

    # State events
    STATE_CHANGED = "state_changed"

    # Action events
    ACTION_STARTED = "action_started"
    ACTION_COMPLETED = "action_completed"
    ACTION_FAILED = "action_failed"

    # Context events
    CONTEXT_UPDATED = "context_updated"

    # Transfer events
    TRANSFER_INITIATED = "transfer_initiated"
    TRANSFER_COMPLETED = "transfer_completed"


@dataclass
class ConversationEvent:
    """Event emitted during conversation."""

    type: ConversationEventType
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    # Context
    session_id: Optional[str] = None
    turn_id: Optional[str] = None
    flow_id: Optional[str] = None
    node_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "flow_id": self.flow_id,
            "node_id": self.node_id,
        }


@dataclass
class ConversationConfig:
    """Configuration for conversation handling."""

    # Timing
    max_silence_ms: int = 5000          # Max silence before prompting
    response_timeout_ms: int = 30000     # Max time to generate response
    turn_timeout_ms: int = 60000         # Max turn duration

    # Behavior
    allow_interruptions: bool = True
    confirm_on_silence: bool = True
    max_clarifications: int = 3
    max_retries: int = 2

    # Greetings
    enable_greeting: bool = True
    greeting_message: Optional[str] = None

    # Error handling
    error_message: str = "I'm sorry, I encountered an issue. Let me try that again."
    fallback_message: str = "I'm not sure I understood. Could you please rephrase?"

    # Context
    context_window_turns: int = 10
    persist_context: bool = True

    # Language
    default_language: str = "en-US"

    # Metadata
    agent_name: str = "Assistant"
    agent_persona: Optional[str] = None


@dataclass
class UserInput:
    """Represents user input to the conversation."""

    # Content
    text: str = ""
    audio_url: Optional[str] = None
    transcription_confidence: float = 1.0

    # Metadata
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    language: Optional[str] = None

    # Channel info
    channel: str = "voice"  # voice, chat, sms
    channel_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Represents agent response in conversation."""

    # Content
    text: str = ""
    response_type: ResponseType = ResponseType.TEXT

    # Alternative content
    ssml: Optional[str] = None
    audio_url: Optional[str] = None

    # Actions
    actions: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    source: str = ""  # Which component generated this

    # Flow control
    expects_response: bool = True
    timeout_ms: Optional[int] = None
    silence_prompt: Optional[str] = None

    # Transfer info
    transfer_target: Optional[str] = None
    transfer_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "response_type": self.response_type.value,
            "ssml": self.ssml,
            "actions": self.actions,
            "expects_response": self.expects_response,
        }


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""

    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Input
    user_input: Optional[UserInput] = None

    # Processing
    detected_intent: Optional[str] = None
    intent_confidence: float = 0.0
    extracted_slots: Dict[str, Any] = field(default_factory=dict)
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)

    # Output
    response: Optional[AgentResponse] = None

    # State tracking
    state: TurnState = TurnState.STARTED
    flow_node_id: Optional[str] = None

    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # Errors
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Calculate turn duration."""
        if self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return (time.time() - self.started_at) * 1000

    def complete(self) -> None:
        """Mark turn as completed."""
        self.completed_at = time.time()
        self.state = TurnState.COMPLETED


__all__ = [
    "ConversationState",
    "TurnState",
    "ResponseType",
    "ConversationEventType",
    "ConversationEvent",
    "ConversationConfig",
    "UserInput",
    "AgentResponse",
    "ConversationTurn",
]
