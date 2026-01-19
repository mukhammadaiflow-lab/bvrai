"""Conversation state machine for managing call flow."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Any
import structlog

logger = structlog.get_logger()


class ConversationState(str, Enum):
    """States of a voice conversation."""

    IDLE = "idle"  # Waiting for call to start
    GREETING = "greeting"  # Playing initial greeting
    LISTENING = "listening"  # Listening to user speech
    PROCESSING = "processing"  # Processing user input (ASR complete, waiting for LLM)
    SPEAKING = "speaking"  # Playing TTS response
    THINKING = "thinking"  # Brief pause between listening and speaking
    TRANSFERRING = "transferring"  # Transferring call
    ENDING = "ending"  # Call ending
    ENDED = "ended"  # Call completed


class ConversationEvent(str, Enum):
    """Events that trigger state transitions."""

    # Call lifecycle
    CALL_STARTED = "call_started"
    CALL_ENDED = "call_ended"

    # Greeting
    GREETING_STARTED = "greeting_started"
    GREETING_COMPLETED = "greeting_completed"

    # User speech
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    TRANSCRIPT_INTERIM = "transcript_interim"
    TRANSCRIPT_FINAL = "transcript_final"

    # Agent response
    LLM_RESPONSE_STARTED = "llm_response_started"
    LLM_RESPONSE_COMPLETED = "llm_response_completed"
    TTS_STARTED = "tts_started"
    TTS_COMPLETED = "tts_completed"

    # Interruption
    BARGE_IN = "barge_in"  # User interrupts agent

    # Actions
    TRANSFER_INITIATED = "transfer_initiated"
    TRANSFER_COMPLETED = "transfer_completed"
    HANGUP_REQUESTED = "hangup_requested"

    # Errors
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class StateTransition:
    """Represents a state transition."""
    from_state: ConversationState
    to_state: ConversationState
    event: ConversationEvent
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context maintained throughout the conversation."""

    session_id: str
    agent_id: str

    # Current state
    state: ConversationState = ConversationState.IDLE

    # Conversation history
    messages: list[dict] = field(default_factory=list)
    turn_count: int = 0

    # Current turn
    current_transcript: str = ""
    current_response: str = ""
    pending_audio: list[bytes] = field(default_factory=list)

    # Timing
    last_speech_time: float = 0.0
    last_response_time: float = 0.0
    turn_start_time: float = 0.0

    # Flags
    is_interrupted: bool = False
    awaiting_response: bool = False

    # State history
    state_history: list[StateTransition] = field(default_factory=list)

    # Extracted entities
    entities: dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationStateMachine:
    """
    State machine for managing voice conversation flow.

    Handles:
    - Turn-taking between user and agent
    - Barge-in (interruption) handling
    - Timeout management
    - State transition validation
    """

    # Valid state transitions
    TRANSITIONS: dict[ConversationState, dict[ConversationEvent, ConversationState]] = {
        ConversationState.IDLE: {
            ConversationEvent.CALL_STARTED: ConversationState.GREETING,
        },
        ConversationState.GREETING: {
            ConversationEvent.GREETING_COMPLETED: ConversationState.LISTENING,
            ConversationEvent.SPEECH_STARTED: ConversationState.LISTENING,  # User interrupts greeting
            ConversationEvent.BARGE_IN: ConversationState.LISTENING,
            ConversationEvent.CALL_ENDED: ConversationState.ENDED,
        },
        ConversationState.LISTENING: {
            ConversationEvent.TRANSCRIPT_FINAL: ConversationState.PROCESSING,
            ConversationEvent.SPEECH_ENDED: ConversationState.PROCESSING,
            ConversationEvent.TIMEOUT: ConversationState.SPEAKING,  # No speech, prompt again
            ConversationEvent.CALL_ENDED: ConversationState.ENDED,
            ConversationEvent.HANGUP_REQUESTED: ConversationState.ENDING,
        },
        ConversationState.PROCESSING: {
            ConversationEvent.LLM_RESPONSE_STARTED: ConversationState.SPEAKING,
            ConversationEvent.TTS_STARTED: ConversationState.SPEAKING,
            ConversationEvent.TRANSFER_INITIATED: ConversationState.TRANSFERRING,
            ConversationEvent.HANGUP_REQUESTED: ConversationState.ENDING,
            ConversationEvent.ERROR: ConversationState.SPEAKING,  # Fallback response
            ConversationEvent.CALL_ENDED: ConversationState.ENDED,
        },
        ConversationState.SPEAKING: {
            ConversationEvent.TTS_COMPLETED: ConversationState.LISTENING,
            ConversationEvent.BARGE_IN: ConversationState.LISTENING,  # User interrupts
            ConversationEvent.SPEECH_STARTED: ConversationState.LISTENING,  # User starts speaking
            ConversationEvent.HANGUP_REQUESTED: ConversationState.ENDING,
            ConversationEvent.CALL_ENDED: ConversationState.ENDED,
        },
        ConversationState.TRANSFERRING: {
            ConversationEvent.TRANSFER_COMPLETED: ConversationState.ENDED,
            ConversationEvent.ERROR: ConversationState.SPEAKING,
            ConversationEvent.CALL_ENDED: ConversationState.ENDED,
        },
        ConversationState.ENDING: {
            ConversationEvent.TTS_COMPLETED: ConversationState.ENDED,
            ConversationEvent.CALL_ENDED: ConversationState.ENDED,
        },
        ConversationState.ENDED: {},  # Terminal state
    }

    def __init__(self, context: ConversationContext) -> None:
        self.context = context
        self.logger = logger.bind(
            session_id=context.session_id,
            agent_id=context.agent_id,
        )

        # Callbacks
        self._on_state_change: list[Callable[[StateTransition], None]] = []
        self._on_event: list[Callable[[ConversationEvent, dict], None]] = []

        # State-specific handlers
        self._state_handlers: dict[ConversationState, Callable] = {}

    def on_state_change(self, callback: Callable[[StateTransition], None]) -> None:
        """Register callback for state changes."""
        self._on_state_change.append(callback)

    def on_event(self, callback: Callable[[ConversationEvent, dict], None]) -> None:
        """Register callback for events."""
        self._on_event.append(callback)

    def register_handler(
        self,
        state: ConversationState,
        handler: Callable,
    ) -> None:
        """Register a handler for a specific state."""
        self._state_handlers[state] = handler

    def can_transition(self, event: ConversationEvent) -> bool:
        """Check if transition is valid for current state."""
        current_state = self.context.state
        valid_transitions = self.TRANSITIONS.get(current_state, {})
        return event in valid_transitions

    def get_next_state(self, event: ConversationEvent) -> Optional[ConversationState]:
        """Get the next state for an event, or None if invalid."""
        current_state = self.context.state
        valid_transitions = self.TRANSITIONS.get(current_state, {})
        return valid_transitions.get(event)

    async def handle_event(
        self,
        event: ConversationEvent,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Handle an event and perform state transition if valid.

        Returns True if transition occurred, False otherwise.
        """
        metadata = metadata or {}

        # Notify event listeners
        for callback in self._on_event:
            try:
                callback(event, metadata)
            except Exception as e:
                self.logger.error("Event callback error", error=str(e))

        # Check if transition is valid
        next_state = self.get_next_state(event)
        if next_state is None:
            self.logger.debug(
                "Invalid event for current state",
                current_state=self.context.state.value,
                event=event.value,
            )
            return False

        # Perform transition
        previous_state = self.context.state
        self.context.state = next_state

        # Record transition
        transition = StateTransition(
            from_state=previous_state,
            to_state=next_state,
            event=event,
            metadata=metadata,
        )
        self.context.state_history.append(transition)

        self.logger.info(
            "State transition",
            from_state=previous_state.value,
            to_state=next_state.value,
            event=event.value,
        )

        # Notify state change listeners
        for callback in self._on_state_change:
            try:
                callback(transition)
            except Exception as e:
                self.logger.error("State change callback error", error=str(e))

        # Execute state handler if registered
        handler = self._state_handlers.get(next_state)
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(self.context, transition)
                else:
                    handler(self.context, transition)
            except Exception as e:
                self.logger.error("State handler error", error=str(e))

        return True

    def is_user_turn(self) -> bool:
        """Check if it's the user's turn to speak."""
        return self.context.state in [
            ConversationState.LISTENING,
            ConversationState.IDLE,
        ]

    def is_agent_turn(self) -> bool:
        """Check if it's the agent's turn to speak."""
        return self.context.state in [
            ConversationState.SPEAKING,
            ConversationState.GREETING,
        ]

    def is_active(self) -> bool:
        """Check if conversation is active (not ended)."""
        return self.context.state not in [
            ConversationState.ENDED,
            ConversationState.ENDING,
        ]

    def add_user_message(self, text: str) -> None:
        """Add a user message to conversation history."""
        self.context.messages.append({
            "role": "user",
            "content": text,
            "timestamp": time.time(),
        })
        self.context.current_transcript = text
        self.context.turn_count += 1

    def add_agent_message(self, text: str) -> None:
        """Add an agent message to conversation history."""
        self.context.messages.append({
            "role": "assistant",
            "content": text,
            "timestamp": time.time(),
        })
        self.context.current_response = text

    def get_conversation_history(self, max_messages: int = 20) -> list[dict]:
        """Get recent conversation history for LLM context."""
        return self.context.messages[-max_messages:]

    def set_entity(self, key: str, value: Any) -> None:
        """Set an extracted entity."""
        self.context.entities[key] = value

    def get_entity(self, key: str, default: Any = None) -> Any:
        """Get an extracted entity."""
        return self.context.entities.get(key, default)
