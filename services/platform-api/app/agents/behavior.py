"""
Agent Behavior System

Behavioral configurations and patterns:
- Interruption handling
- Silence management
- Error recovery
- Turn-taking
- Conversation flow
"""

from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import asyncio
import logging

logger = logging.getLogger(__name__)


class InterruptionMode(str, Enum):
    """How agent handles interruptions."""
    ALLOW = "allow"  # Allow interruptions anytime
    ALLOW_AFTER_SENTENCE = "allow_after_sentence"  # Allow after completing sentence
    ALLOW_AFTER_PHRASE = "allow_after_phrase"  # Allow after completing phrase
    IGNORE = "ignore"  # Ignore all interruptions
    QUEUE = "queue"  # Queue interruption and respond after


class SilenceAction(str, Enum):
    """What to do during silence."""
    WAIT = "wait"  # Wait for user
    PROMPT = "prompt"  # Prompt user to continue
    REPEAT = "repeat"  # Repeat last message
    TRANSFER = "transfer"  # Transfer to human
    END = "end"  # End conversation


class ErrorAction(str, Enum):
    """How to handle errors."""
    RETRY = "retry"  # Retry the operation
    FALLBACK = "fallback"  # Use fallback response
    ESCALATE = "escalate"  # Escalate to human
    LOG_CONTINUE = "log_continue"  # Log and continue
    ABORT = "abort"  # Abort conversation


class TurnTakingMode(str, Enum):
    """Turn-taking strategy."""
    STRICT = "strict"  # Strict turn alternation
    FLEXIBLE = "flexible"  # Flexible timing
    OVERLAP_ALLOWED = "overlap_allowed"  # Allow speech overlap
    BARGE_IN = "barge_in"  # Allow barge-in


class ConversationPhase(str, Enum):
    """Conversation phases."""
    GREETING = "greeting"
    IDENTIFICATION = "identification"
    MAIN = "main"
    RESOLUTION = "resolution"
    FAREWELL = "farewell"
    ESCALATION = "escalation"


@dataclass
class InterruptionBehavior:
    """Configuration for interruption handling."""
    mode: InterruptionMode = InterruptionMode.ALLOW_AFTER_SENTENCE

    # Sensitivity (0-1, higher = more sensitive to interruptions)
    sensitivity: float = 0.5

    # Minimum speech before allowing interruption (seconds)
    min_speech_duration: float = 1.0

    # Debounce period for interruptions (seconds)
    debounce_period: float = 0.3

    # Words to complete before allowing interruption
    min_words_before_interrupt: int = 3

    # Allow interruption during important phrases
    allow_during_important: bool = False

    # Important phrases that shouldn't be interrupted
    protected_phrases: List[str] = field(default_factory=list)

    # Response when interrupted
    interruption_acknowledgement: str = "Yes?"

    # Track interruption statistics
    track_stats: bool = True

    def should_allow_interruption(
        self,
        current_speech_duration: float,
        current_word_count: int,
        current_text: str,
    ) -> bool:
        """Determine if interruption should be allowed."""
        # Check mode
        if self.mode == InterruptionMode.IGNORE:
            return False
        if self.mode == InterruptionMode.ALLOW:
            return True

        # Check minimum duration
        if current_speech_duration < self.min_speech_duration:
            return False

        # Check minimum words
        if current_word_count < self.min_words_before_interrupt:
            return False

        # Check protected phrases
        if not self.allow_during_important:
            for phrase in self.protected_phrases:
                if phrase.lower() in current_text.lower():
                    return False

        return True


@dataclass
class SilenceBehavior:
    """Configuration for silence handling."""
    # Silence detection
    silence_threshold_ms: int = 2000  # Silence duration before action
    extended_silence_threshold_ms: int = 5000  # Extended silence
    max_silence_count: int = 3  # Max silences before escalation

    # Actions
    initial_action: SilenceAction = SilenceAction.PROMPT
    extended_action: SilenceAction = SilenceAction.REPEAT
    max_silence_action: SilenceAction = SilenceAction.TRANSFER

    # Prompts
    prompt_templates: List[str] = field(default_factory=lambda: [
        "Are you still there?",
        "I'm here when you're ready.",
        "Take your time, I'm listening.",
    ])

    # Repeat settings
    repeat_delay_ms: int = 1000
    max_repeats: int = 2

    # Context awareness
    adjust_for_complexity: bool = True  # Allow longer silence for complex topics
    complexity_multiplier: float = 1.5

    def get_action(self, silence_duration_ms: int, silence_count: int) -> SilenceAction:
        """Get action based on silence duration and count."""
        if silence_count >= self.max_silence_count:
            return self.max_silence_action
        elif silence_duration_ms >= self.extended_silence_threshold_ms:
            return self.extended_action
        elif silence_duration_ms >= self.silence_threshold_ms:
            return self.initial_action
        return SilenceAction.WAIT

    def get_prompt(self, count: int) -> str:
        """Get silence prompt based on count."""
        if not self.prompt_templates:
            return "Are you still there?"
        index = min(count, len(self.prompt_templates) - 1)
        return self.prompt_templates[index]


@dataclass
class ErrorBehavior:
    """Configuration for error handling."""
    # Retry settings
    max_retries: int = 3
    retry_delay_ms: int = 1000
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0

    # Error actions by type
    transcription_error_action: ErrorAction = ErrorAction.RETRY
    llm_error_action: ErrorAction = ErrorAction.FALLBACK
    tts_error_action: ErrorAction = ErrorAction.RETRY
    network_error_action: ErrorAction = ErrorAction.RETRY
    timeout_error_action: ErrorAction = ErrorAction.FALLBACK

    # Fallback responses
    fallback_responses: List[str] = field(default_factory=lambda: [
        "I apologize, I'm having some technical difficulties. Please bear with me.",
        "I'm sorry, could you please repeat that?",
        "Let me transfer you to someone who can better assist you.",
    ])

    # Escalation settings
    escalate_after_errors: int = 3
    escalation_message: str = "I apologize for the difficulties. Let me connect you with a human agent."

    # Logging
    log_errors: bool = True
    alert_on_error: bool = False

    def get_retry_delay(self, attempt: int) -> int:
        """Get retry delay for attempt."""
        if self.exponential_backoff:
            return int(self.retry_delay_ms * (self.backoff_multiplier ** attempt))
        return self.retry_delay_ms

    def get_fallback_response(self, error_count: int) -> str:
        """Get fallback response based on error count."""
        if not self.fallback_responses:
            return "I apologize for the inconvenience."
        index = min(error_count, len(self.fallback_responses) - 1)
        return self.fallback_responses[index]


@dataclass
class TurnTakingBehavior:
    """Configuration for turn-taking."""
    mode: TurnTakingMode = TurnTakingMode.FLEXIBLE

    # Timing
    min_turn_duration_ms: int = 500
    max_turn_duration_ms: int = 30000
    response_delay_ms: int = 200  # Delay before responding

    # End-of-turn detection
    end_of_turn_silence_ms: int = 700
    use_prosody_detection: bool = True
    use_semantic_detection: bool = True

    # Backchanneling
    enable_backchannel: bool = True
    backchannel_phrases: List[str] = field(default_factory=lambda: [
        "Uh-huh",
        "I see",
        "Okay",
        "Right",
        "Mm-hmm",
    ])
    backchannel_interval_ms: int = 3000

    # Floor holding
    floor_holding_phrases: List[str] = field(default_factory=lambda: [
        "So...",
        "Let me see...",
        "One moment...",
    ])

    def should_backchannel(self, turn_duration_ms: int) -> bool:
        """Check if backchannel is appropriate."""
        if not self.enable_backchannel:
            return False
        return turn_duration_ms >= self.backchannel_interval_ms

    def get_backchannel(self) -> str:
        """Get random backchannel phrase."""
        import random
        return random.choice(self.backchannel_phrases) if self.backchannel_phrases else "Mm-hmm"


@dataclass
class PhaseBehavior:
    """Behavior configuration for conversation phase."""
    phase: ConversationPhase

    # Timing
    max_duration_seconds: int = 300
    expected_turns: int = 5

    # Goals
    goals: List[str] = field(default_factory=list)
    required_slots: List[str] = field(default_factory=list)

    # Transitions
    next_phase: Optional[ConversationPhase] = None
    transition_conditions: List[Dict[str, Any]] = field(default_factory=list)

    # Behavior adjustments
    interruption_mode: Optional[InterruptionMode] = None
    silence_threshold_multiplier: float = 1.0

    def should_transition(self, context: Dict[str, Any]) -> bool:
        """Check if should transition to next phase."""
        for condition in self.transition_conditions:
            cond_type = condition.get("type")

            if cond_type == "slots_filled":
                required = condition.get("slots", [])
                filled = context.get("filled_slots", [])
                if not all(s in filled for s in required):
                    return False

            elif cond_type == "intent_detected":
                required_intent = condition.get("intent")
                current_intent = context.get("current_intent")
                if current_intent != required_intent:
                    return False

            elif cond_type == "turns_exceeded":
                max_turns = condition.get("turns", self.expected_turns)
                current_turns = context.get("turn_count", 0)
                if current_turns < max_turns:
                    return False

        return True


@dataclass
class ConversationBehavior:
    """Complete conversation behavior configuration."""
    behavior_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Core behaviors
    interruption: InterruptionBehavior = field(default_factory=InterruptionBehavior)
    silence: SilenceBehavior = field(default_factory=SilenceBehavior)
    error: ErrorBehavior = field(default_factory=ErrorBehavior)
    turn_taking: TurnTakingBehavior = field(default_factory=TurnTakingBehavior)

    # Phase behaviors
    phase_behaviors: Dict[ConversationPhase, PhaseBehavior] = field(default_factory=dict)

    # General settings
    max_conversation_duration_seconds: int = 1800  # 30 minutes
    max_turns: int = 100

    # Timeout settings
    inactivity_timeout_seconds: int = 120
    response_timeout_seconds: int = 30

    # Language
    primary_language: str = "en"
    language_detection: bool = True
    auto_translate: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def get_phase_behavior(self, phase: ConversationPhase) -> PhaseBehavior:
        """Get behavior for phase, creating default if needed."""
        if phase not in self.phase_behaviors:
            self.phase_behaviors[phase] = PhaseBehavior(phase=phase)
        return self.phase_behaviors[phase]


class BehaviorEngine:
    """
    Engine for managing agent behaviors.

    Features:
    - Behavior state management
    - Event-driven behavior triggers
    - Adaptive behavior adjustment
    """

    def __init__(self, behavior: ConversationBehavior):
        self.behavior = behavior
        self._state: Dict[str, Any] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._metrics: Dict[str, Any] = {
            "interruptions": 0,
            "silences": 0,
            "errors": 0,
            "backchannel_count": 0,
        }

    def initialize(self) -> None:
        """Initialize behavior state."""
        self._state = {
            "current_phase": ConversationPhase.GREETING,
            "turn_count": 0,
            "error_count": 0,
            "silence_count": 0,
            "last_activity": datetime.utcnow(),
            "conversation_start": datetime.utcnow(),
        }

    def on_event(self, event: str, handler: Callable) -> None:
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Emit event to handlers."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    async def handle_user_speech_start(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle start of user speech."""
        self._state["last_activity"] = datetime.utcnow()

        # Check if interrupting agent
        if context.get("agent_speaking"):
            current_duration = context.get("agent_speech_duration", 0)
            current_words = context.get("agent_word_count", 0)
            current_text = context.get("agent_current_text", "")

            should_allow = self.behavior.interruption.should_allow_interruption(
                current_duration,
                current_words,
                current_text,
            )

            self._metrics["interruptions"] += 1
            self._emit("interruption", {"allowed": should_allow, "context": context})

            return {
                "action": "allow_interruption" if should_allow else "continue_speech",
                "acknowledgement": self.behavior.interruption.interruption_acknowledgement if should_allow else None,
            }

        return {"action": "listen"}

    async def handle_silence(self, duration_ms: int) -> Dict[str, Any]:
        """Handle silence period."""
        silence_count = self._state.get("silence_count", 0)

        action = self.behavior.silence.get_action(duration_ms, silence_count)

        result = {"action": action.value}

        if action == SilenceAction.PROMPT:
            result["prompt"] = self.behavior.silence.get_prompt(silence_count)
            self._state["silence_count"] = silence_count + 1

        elif action == SilenceAction.REPEAT:
            result["repeat_last"] = True

        elif action == SilenceAction.TRANSFER:
            result["transfer"] = True

        elif action == SilenceAction.END:
            result["end_conversation"] = True

        self._metrics["silences"] += 1
        self._emit("silence", {"duration_ms": duration_ms, "action": action})

        return result

    async def handle_error(
        self,
        error_type: str,
        error: Exception,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle error occurrence."""
        error_count = self._state.get("error_count", 0) + 1
        self._state["error_count"] = error_count

        # Get action based on error type
        action_map = {
            "transcription": self.behavior.error.transcription_error_action,
            "llm": self.behavior.error.llm_error_action,
            "tts": self.behavior.error.tts_error_action,
            "network": self.behavior.error.network_error_action,
            "timeout": self.behavior.error.timeout_error_action,
        }
        action = action_map.get(error_type, ErrorAction.FALLBACK)

        result = {"action": action.value}

        # Check if should escalate
        if error_count >= self.behavior.error.escalate_after_errors:
            result["action"] = ErrorAction.ESCALATE.value
            result["message"] = self.behavior.error.escalation_message
        elif action == ErrorAction.RETRY:
            retry_count = context.get("retry_count", 0)
            if retry_count < self.behavior.error.max_retries:
                result["retry_delay_ms"] = self.behavior.error.get_retry_delay(retry_count)
            else:
                result["action"] = ErrorAction.FALLBACK.value
                result["message"] = self.behavior.error.get_fallback_response(error_count)
        elif action == ErrorAction.FALLBACK:
            result["message"] = self.behavior.error.get_fallback_response(error_count)

        self._metrics["errors"] += 1
        self._emit("error", {"type": error_type, "action": action, "count": error_count})

        return result

    async def handle_turn_end(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle end of turn."""
        self._state["turn_count"] = self._state.get("turn_count", 0) + 1
        self._state["last_activity"] = datetime.utcnow()
        self._state["silence_count"] = 0  # Reset silence count

        result = {
            "delay_ms": self.behavior.turn_taking.response_delay_ms,
        }

        # Check for backchannel opportunity
        turn_duration = context.get("turn_duration_ms", 0)
        if self.behavior.turn_taking.should_backchannel(turn_duration):
            result["backchannel"] = self.behavior.turn_taking.get_backchannel()
            self._metrics["backchannel_count"] += 1

        # Check phase transition
        current_phase = self._state.get("current_phase", ConversationPhase.GREETING)
        phase_behavior = self.behavior.get_phase_behavior(current_phase)

        if phase_behavior.should_transition(context):
            if phase_behavior.next_phase:
                self._state["current_phase"] = phase_behavior.next_phase
                result["phase_transition"] = phase_behavior.next_phase.value

        self._emit("turn_end", {"turn_count": self._state["turn_count"]})

        return result

    def check_timeouts(self) -> Dict[str, Any]:
        """Check for timeout conditions."""
        now = datetime.utcnow()

        # Check inactivity timeout
        last_activity = self._state.get("last_activity", now)
        inactivity_seconds = (now - last_activity).total_seconds()

        if inactivity_seconds >= self.behavior.inactivity_timeout_seconds:
            return {
                "timeout": True,
                "type": "inactivity",
                "action": "end_conversation",
            }

        # Check max conversation duration
        conversation_start = self._state.get("conversation_start", now)
        duration_seconds = (now - conversation_start).total_seconds()

        if duration_seconds >= self.behavior.max_conversation_duration_seconds:
            return {
                "timeout": True,
                "type": "max_duration",
                "action": "end_conversation",
            }

        # Check max turns
        turn_count = self._state.get("turn_count", 0)
        if turn_count >= self.behavior.max_turns:
            return {
                "timeout": True,
                "type": "max_turns",
                "action": "end_conversation",
            }

        return {"timeout": False}

    def get_metrics(self) -> Dict[str, Any]:
        """Get behavior metrics."""
        return {
            **self._metrics,
            "turn_count": self._state.get("turn_count", 0),
            "current_phase": self._state.get("current_phase", ConversationPhase.GREETING).value,
            "error_count": self._state.get("error_count", 0),
        }

    def reset(self) -> None:
        """Reset behavior state."""
        self.initialize()
        self._metrics = {
            "interruptions": 0,
            "silences": 0,
            "errors": 0,
            "backchannel_count": 0,
        }


class BehaviorManager:
    """
    Manages multiple behavior configurations.

    Features:
    - Behavior storage and retrieval
    - Default behaviors
    - Behavior inheritance
    """

    def __init__(self):
        self._behaviors: Dict[str, ConversationBehavior] = {}
        self._engines: Dict[str, BehaviorEngine] = {}
        self._defaults = self._create_defaults()

    def _create_defaults(self) -> Dict[str, ConversationBehavior]:
        """Create default behavior configurations."""
        return {
            "standard": ConversationBehavior(
                name="Standard",
                description="Standard conversation behavior",
            ),
            "patient": ConversationBehavior(
                name="Patient",
                description="Patient, slower-paced behavior",
                silence=SilenceBehavior(
                    silence_threshold_ms=4000,
                    extended_silence_threshold_ms=8000,
                ),
                turn_taking=TurnTakingBehavior(
                    response_delay_ms=500,
                    end_of_turn_silence_ms=1000,
                ),
            ),
            "efficient": ConversationBehavior(
                name="Efficient",
                description="Quick, efficient behavior",
                silence=SilenceBehavior(
                    silence_threshold_ms=1500,
                    extended_silence_threshold_ms=3000,
                ),
                turn_taking=TurnTakingBehavior(
                    response_delay_ms=100,
                    end_of_turn_silence_ms=500,
                ),
            ),
            "empathetic": ConversationBehavior(
                name="Empathetic",
                description="Empathetic, supportive behavior",
                interruption=InterruptionBehavior(
                    mode=InterruptionMode.ALLOW_AFTER_SENTENCE,
                    min_speech_duration=2.0,
                ),
                turn_taking=TurnTakingBehavior(
                    enable_backchannel=True,
                    backchannel_interval_ms=2000,
                ),
            ),
        }

    def register(self, behavior: ConversationBehavior) -> None:
        """Register behavior configuration."""
        self._behaviors[behavior.behavior_id] = behavior

    def get(self, behavior_id: str) -> Optional[ConversationBehavior]:
        """Get behavior by ID."""
        return self._behaviors.get(behavior_id) or self._defaults.get(behavior_id)

    def get_default(self, name: str = "standard") -> ConversationBehavior:
        """Get default behavior."""
        return self._defaults.get(name, self._defaults["standard"])

    def create_engine(self, behavior_id: str) -> BehaviorEngine:
        """Create behavior engine for behavior."""
        behavior = self.get(behavior_id)
        if not behavior:
            behavior = self.get_default()

        engine = BehaviorEngine(behavior)
        engine.initialize()

        self._engines[behavior_id] = engine
        return engine

    def get_engine(self, behavior_id: str) -> Optional[BehaviorEngine]:
        """Get existing engine."""
        return self._engines.get(behavior_id)

    def delete(self, behavior_id: str) -> bool:
        """Delete behavior."""
        self._engines.pop(behavior_id, None)
        return self._behaviors.pop(behavior_id, None) is not None

    def list_behaviors(self) -> List[ConversationBehavior]:
        """List all behaviors."""
        all_behaviors = list(self._behaviors.values())
        all_behaviors.extend(self._defaults.values())
        return all_behaviors

    def clone_behavior(
        self,
        source_id: str,
        new_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ConversationBehavior:
        """Clone behavior with overrides."""
        import copy

        source = self.get(source_id)
        if not source:
            source = self.get_default()

        new_behavior = copy.deepcopy(source)
        new_behavior.behavior_id = str(uuid.uuid4())
        new_behavior.name = new_name
        new_behavior.created_at = datetime.utcnow()
        new_behavior.updated_at = datetime.utcnow()

        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(new_behavior, key):
                    setattr(new_behavior, key, value)

        self.register(new_behavior)
        return new_behavior

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "total_behaviors": len(self._behaviors),
            "active_engines": len(self._engines),
            "default_behaviors": len(self._defaults),
        }


class AdaptiveBehavior:
    """
    Adaptive behavior that learns from interactions.

    Features:
    - Response time adaptation
    - Interruption pattern learning
    - User preference inference
    """

    def __init__(self, base_behavior: ConversationBehavior):
        self.base_behavior = base_behavior
        self._history: List[Dict[str, Any]] = []
        self._adaptations: Dict[str, Any] = {}
        self._learning_rate: float = 0.1

    def record_interaction(self, interaction: Dict[str, Any]) -> None:
        """Record interaction for learning."""
        self._history.append({
            **interaction,
            "timestamp": datetime.utcnow(),
        })

        # Keep limited history
        if len(self._history) > 100:
            self._history = self._history[-100:]

        # Update adaptations
        self._update_adaptations()

    def _update_adaptations(self) -> None:
        """Update behavior adaptations based on history."""
        if len(self._history) < 5:
            return

        recent = self._history[-10:]

        # Adapt silence threshold based on user response times
        response_times = [i.get("user_response_time_ms", 2000) for i in recent]
        avg_response_time = sum(response_times) / len(response_times)

        current_threshold = self._adaptations.get(
            "silence_threshold",
            self.base_behavior.silence.silence_threshold_ms,
        )

        # Adjust towards user's natural response time
        new_threshold = current_threshold + self._learning_rate * (avg_response_time - current_threshold)
        self._adaptations["silence_threshold"] = int(new_threshold)

        # Adapt interruption sensitivity based on interruption success
        interruptions = [i for i in recent if i.get("type") == "interruption"]
        if interruptions:
            successful = sum(1 for i in interruptions if i.get("was_helpful"))
            success_rate = successful / len(interruptions)

            current_sensitivity = self._adaptations.get(
                "interruption_sensitivity",
                self.base_behavior.interruption.sensitivity,
            )

            # Increase sensitivity if interruptions are helpful
            adjustment = 0.1 if success_rate > 0.7 else -0.1 if success_rate < 0.3 else 0
            new_sensitivity = max(0, min(1, current_sensitivity + adjustment))
            self._adaptations["interruption_sensitivity"] = new_sensitivity

    def get_adapted_behavior(self) -> ConversationBehavior:
        """Get behavior with current adaptations."""
        import copy

        adapted = copy.deepcopy(self.base_behavior)

        if "silence_threshold" in self._adaptations:
            adapted.silence.silence_threshold_ms = self._adaptations["silence_threshold"]

        if "interruption_sensitivity" in self._adaptations:
            adapted.interruption.sensitivity = self._adaptations["interruption_sensitivity"]

        return adapted

    def get_adaptations(self) -> Dict[str, Any]:
        """Get current adaptations."""
        return self._adaptations.copy()

    def reset_adaptations(self) -> None:
        """Reset all adaptations."""
        self._adaptations = {}
        self._history = []


# Factory functions
def create_standard_behavior() -> ConversationBehavior:
    """Create standard conversation behavior."""
    return ConversationBehavior(
        name="Standard",
        description="Standard balanced conversation behavior",
    )


def create_customer_service_behavior() -> ConversationBehavior:
    """Create customer service optimized behavior."""
    return ConversationBehavior(
        name="Customer Service",
        description="Optimized for customer service interactions",
        interruption=InterruptionBehavior(
            mode=InterruptionMode.ALLOW_AFTER_SENTENCE,
            sensitivity=0.6,
            protected_phrases=["account number", "confirmation code", "important"],
        ),
        silence=SilenceBehavior(
            silence_threshold_ms=3000,
            prompt_templates=[
                "I'm still here. Take your time.",
                "Is there anything else I can help you with?",
                "Let me know if you need more information.",
            ],
        ),
        error=ErrorBehavior(
            fallback_responses=[
                "I apologize for the inconvenience. Let me try that again.",
                "I'm sorry, I'm having some difficulty. Could you please repeat that?",
                "I apologize, let me connect you with a specialist who can better assist you.",
            ],
        ),
        turn_taking=TurnTakingBehavior(
            enable_backchannel=True,
            backchannel_phrases=["I understand", "I see", "Of course", "Certainly"],
        ),
    )


def create_sales_behavior() -> ConversationBehavior:
    """Create sales optimized behavior."""
    return ConversationBehavior(
        name="Sales",
        description="Optimized for sales conversations",
        interruption=InterruptionBehavior(
            mode=InterruptionMode.ALLOW,
            sensitivity=0.7,
        ),
        silence=SilenceBehavior(
            silence_threshold_ms=2000,
            prompt_templates=[
                "Do you have any questions?",
                "Would you like me to explain any features?",
                "Is there anything specific you'd like to know more about?",
            ],
        ),
        turn_taking=TurnTakingBehavior(
            response_delay_ms=150,
            enable_backchannel=True,
            backchannel_phrases=["Absolutely", "Great question", "I understand completely"],
        ),
    )
