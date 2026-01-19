"""
Dialogue Management

Manages conversation flow:
- Turn-taking
- State tracking
- Policy decisions
- Flow control
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import re

logger = logging.getLogger(__name__)


class DialogueState(str, Enum):
    """Dialogue states."""
    GREETING = "greeting"
    LISTENING = "listening"
    UNDERSTANDING = "understanding"
    RESPONDING = "responding"
    ASKING = "asking"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    TRANSFERRING = "transferring"
    CLOSING = "closing"
    ENDED = "ended"


class TurnType(str, Enum):
    """Turn types."""
    USER_SPEECH = "user_speech"
    USER_DTMF = "user_dtmf"
    USER_ACTION = "user_action"
    AGENT_SPEECH = "agent_speech"
    AGENT_ACTION = "agent_action"
    SYSTEM_EVENT = "system_event"
    SILENCE = "silence"
    INTERRUPTION = "interruption"


@dataclass
class DialogueTurn:
    """Represents a dialogue turn."""
    turn_id: str
    turn_type: TurnType
    speaker: str  # "user", "agent", "system"

    # Content
    content: str = ""
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0

    # Audio
    audio_url: Optional[str] = None
    transcription_confidence: float = 1.0

    # Processing
    was_interrupted: bool = False
    was_barged_in: bool = False

    # Response
    response_content: str = ""
    response_audio_url: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Check if turn is complete."""
        return self.end_time is not None


@dataclass
class DialogueContext:
    """Current dialogue context."""
    dialogue_id: str

    # State
    current_state: DialogueState = DialogueState.GREETING
    previous_state: Optional[DialogueState] = None

    # Current turn
    current_turn: Optional[DialogueTurn] = None
    turn_count: int = 0

    # Intent tracking
    current_intent: Optional[str] = None
    intent_confidence: float = 0.0
    pending_intents: List[str] = field(default_factory=list)

    # Slot filling
    required_slots: List[str] = field(default_factory=list)
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)

    # Confirmation
    awaiting_confirmation: bool = False
    confirmation_value: Optional[Any] = None
    confirmation_attempts: int = 0

    # Error handling
    error_count: int = 0
    no_input_count: int = 0
    no_match_count: int = 0

    # History
    state_history: List[DialogueState] = field(default_factory=list)

    def transition_to(self, new_state: DialogueState) -> None:
        """Transition to new state."""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_history.append(new_state)

    def all_slots_filled(self) -> bool:
        """Check if all required slots are filled."""
        return all(slot in self.filled_slots for slot in self.required_slots)

    def get_next_missing_slot(self) -> Optional[str]:
        """Get next slot to fill."""
        for slot in self.required_slots:
            if slot not in self.filled_slots:
                return slot
        return None


class DialoguePolicy(ABC):
    """Abstract dialogue policy."""

    @abstractmethod
    async def decide_action(
        self,
        context: DialogueContext,
        turn: DialogueTurn,
    ) -> Dict[str, Any]:
        """
        Decide next action based on context and turn.

        Returns action specification:
        {
            "action": "respond" | "ask" | "confirm" | "execute" | "transfer",
            "content": str,
            "slots_to_ask": List[str],
            "intent_to_execute": str,
            ...
        }
        """
        pass


class RuleBasedPolicy(DialoguePolicy):
    """Rule-based dialogue policy."""

    def __init__(self):
        self._rules: List[Dict[str, Any]] = []
        self._default_responses: Dict[str, str] = {
            "greeting": "Hello! How can I help you today?",
            "no_input": "I didn't hear anything. Could you please repeat that?",
            "no_match": "I'm sorry, I didn't understand. Could you please rephrase?",
            "confirm": "Just to confirm, you said {}. Is that correct?",
            "goodbye": "Thank you for calling. Goodbye!",
        }

    def add_rule(
        self,
        condition: Callable[[DialogueContext, DialogueTurn], bool],
        action: Dict[str, Any],
        priority: int = 0,
    ) -> None:
        """Add policy rule."""
        self._rules.append({
            "condition": condition,
            "action": action,
            "priority": priority,
        })
        self._rules.sort(key=lambda r: r["priority"], reverse=True)

    async def decide_action(
        self,
        context: DialogueContext,
        turn: DialogueTurn,
    ) -> Dict[str, Any]:
        """Decide action using rules."""
        # Check rules in priority order
        for rule in self._rules:
            try:
                if rule["condition"](context, turn):
                    return rule["action"]
            except Exception as e:
                logger.error(f"Rule evaluation error: {e}")

        # Default behavior based on state
        return await self._default_action(context, turn)

    async def _default_action(
        self,
        context: DialogueContext,
        turn: DialogueTurn,
    ) -> Dict[str, Any]:
        """Default action based on state."""
        state = context.current_state

        if state == DialogueState.GREETING:
            return {
                "action": "respond",
                "content": self._default_responses["greeting"],
                "next_state": DialogueState.LISTENING,
            }

        if turn.turn_type == TurnType.SILENCE:
            context.no_input_count += 1
            return {
                "action": "ask",
                "content": self._default_responses["no_input"],
            }

        if not turn.intent:
            context.no_match_count += 1
            return {
                "action": "ask",
                "content": self._default_responses["no_match"],
            }

        # Check if slots need filling
        if not context.all_slots_filled():
            next_slot = context.get_next_missing_slot()
            if next_slot:
                return {
                    "action": "ask",
                    "slot": next_slot,
                    "content": f"What is your {next_slot.replace('_', ' ')}?",
                }

        # Confirm before execution
        if not context.awaiting_confirmation:
            return {
                "action": "confirm",
                "content": self._default_responses["confirm"].format(
                    turn.content[:100]
                ),
            }

        # Execute intent
        return {
            "action": "execute",
            "intent": context.current_intent,
            "slots": context.filled_slots,
        }


class NeuralPolicy(DialoguePolicy):
    """Neural network-based dialogue policy."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None

    async def load_model(self) -> None:
        """Load policy model."""
        # In production: load trained model
        pass

    async def decide_action(
        self,
        context: DialogueContext,
        turn: DialogueTurn,
    ) -> Dict[str, Any]:
        """Decide action using neural model."""
        # Encode context and turn
        features = self._encode_features(context, turn)

        # Get model prediction
        # In production: use actual model
        # action_probs = self._model.predict(features)

        # For now, fall back to rule-based
        fallback = RuleBasedPolicy()
        return await fallback.decide_action(context, turn)

    def _encode_features(
        self,
        context: DialogueContext,
        turn: DialogueTurn,
    ) -> List[float]:
        """Encode context and turn as features."""
        features = []

        # State encoding
        states = list(DialogueState)
        state_vec = [1.0 if s == context.current_state else 0.0 for s in states]
        features.extend(state_vec)

        # Turn type encoding
        turn_types = list(TurnType)
        turn_vec = [1.0 if t == turn.turn_type else 0.0 for t in turn_types]
        features.extend(turn_vec)

        # Counts
        features.append(context.turn_count / 100.0)
        features.append(context.error_count / 10.0)
        features.append(context.no_input_count / 5.0)

        # Slot filling progress
        if context.required_slots:
            progress = len(context.filled_slots) / len(context.required_slots)
        else:
            progress = 1.0
        features.append(progress)

        # Confidence
        features.append(context.intent_confidence)
        features.append(turn.transcription_confidence)

        return features


class DialogueManager:
    """
    Manages dialogue flow.

    Handles:
    - Turn management
    - State transitions
    - Policy decisions
    - Slot filling
    """

    def __init__(
        self,
        policy: Optional[DialoguePolicy] = None,
    ):
        self.policy = policy or RuleBasedPolicy()

        # Active dialogues
        self._dialogues: Dict[str, DialogueContext] = {}
        self._lock = asyncio.Lock()

        # Slot definitions
        self._slot_definitions: Dict[str, Dict[str, Any]] = {}

        # Intent handlers
        self._intent_handlers: Dict[str, Callable[..., Awaitable[Any]]] = {}

    async def start_dialogue(self, dialogue_id: str) -> DialogueContext:
        """Start new dialogue."""
        async with self._lock:
            context = DialogueContext(dialogue_id=dialogue_id)
            self._dialogues[dialogue_id] = context
            return context

    async def get_context(self, dialogue_id: str) -> Optional[DialogueContext]:
        """Get dialogue context."""
        return self._dialogues.get(dialogue_id)

    async def end_dialogue(self, dialogue_id: str) -> bool:
        """End dialogue."""
        async with self._lock:
            context = self._dialogues.get(dialogue_id)
            if context:
                context.transition_to(DialogueState.ENDED)
                return True
            return False

    async def process_turn(
        self,
        dialogue_id: str,
        turn: DialogueTurn,
    ) -> Dict[str, Any]:
        """
        Process dialogue turn.

        Returns action to take.
        """
        context = await self.get_context(dialogue_id)
        if not context:
            context = await self.start_dialogue(dialogue_id)

        # Update context with turn info
        context.current_turn = turn
        context.turn_count += 1

        if turn.intent:
            context.current_intent = turn.intent
            context.intent_confidence = turn.transcription_confidence

        # Extract and fill slots
        await self._fill_slots(context, turn)

        # Get policy decision
        action = await self.policy.decide_action(context, turn)

        # Apply state transition
        if "next_state" in action:
            context.transition_to(action["next_state"])

        # Handle action
        result = await self._handle_action(context, action)

        # Mark turn complete
        turn.end_time = datetime.utcnow()
        if turn.start_time:
            turn.duration_ms = (turn.end_time - turn.start_time).total_seconds() * 1000

        return result

    async def _fill_slots(
        self,
        context: DialogueContext,
        turn: DialogueTurn,
    ) -> None:
        """Fill slots from turn entities."""
        for slot_name, slot_value in turn.entities.items():
            if slot_name in context.required_slots:
                # Validate slot value
                if self._validate_slot(slot_name, slot_value):
                    context.filled_slots[slot_name] = slot_value

        # Update missing slots
        context.missing_slots = [
            s for s in context.required_slots
            if s not in context.filled_slots
        ]

    def _validate_slot(self, slot_name: str, value: Any) -> bool:
        """Validate slot value."""
        definition = self._slot_definitions.get(slot_name)
        if not definition:
            return True

        # Type validation
        expected_type = definition.get("type")
        if expected_type:
            if expected_type == "string" and not isinstance(value, str):
                return False
            if expected_type == "number" and not isinstance(value, (int, float)):
                return False
            if expected_type == "date" and not self._is_valid_date(value):
                return False

        # Pattern validation
        pattern = definition.get("pattern")
        if pattern and isinstance(value, str):
            if not re.match(pattern, value):
                return False

        # Allowed values
        allowed = definition.get("allowed_values")
        if allowed and value not in allowed:
            return False

        return True

    def _is_valid_date(self, value: Any) -> bool:
        """Check if value is valid date."""
        if isinstance(value, datetime):
            return True
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value)
                return True
            except ValueError:
                pass
        return False

    async def _handle_action(
        self,
        context: DialogueContext,
        action: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle policy action."""
        action_type = action.get("action", "respond")

        if action_type == "respond":
            return {
                "type": "response",
                "content": action.get("content", ""),
            }

        if action_type == "ask":
            slot = action.get("slot")
            if slot:
                context.transition_to(DialogueState.ASKING)

            return {
                "type": "question",
                "content": action.get("content", ""),
                "slot": slot,
            }

        if action_type == "confirm":
            context.awaiting_confirmation = True
            context.confirmation_value = action.get("value")
            context.transition_to(DialogueState.CONFIRMING)

            return {
                "type": "confirmation",
                "content": action.get("content", ""),
                "value": context.confirmation_value,
            }

        if action_type == "execute":
            context.transition_to(DialogueState.EXECUTING)
            intent = action.get("intent")
            slots = action.get("slots", {})

            result = await self._execute_intent(intent, slots)

            return {
                "type": "execution",
                "intent": intent,
                "result": result,
            }

        if action_type == "transfer":
            context.transition_to(DialogueState.TRANSFERRING)

            return {
                "type": "transfer",
                "target": action.get("target"),
                "context": action.get("transfer_context", {}),
            }

        return {
            "type": "unknown",
            "action": action,
        }

    async def _execute_intent(
        self,
        intent: str,
        slots: Dict[str, Any],
    ) -> Any:
        """Execute intent handler."""
        handler = self._intent_handlers.get(intent)
        if handler:
            try:
                return await handler(**slots)
            except Exception as e:
                logger.error(f"Intent handler error: {e}")
                return {"error": str(e)}

        return {"error": f"No handler for intent: {intent}"}

    def register_intent_handler(
        self,
        intent: str,
        handler: Callable[..., Awaitable[Any]],
    ) -> None:
        """Register intent handler."""
        self._intent_handlers[intent] = handler

    def define_slot(
        self,
        name: str,
        slot_type: str = "string",
        required: bool = False,
        pattern: Optional[str] = None,
        allowed_values: Optional[List[Any]] = None,
        prompt: Optional[str] = None,
    ) -> None:
        """Define slot."""
        self._slot_definitions[name] = {
            "type": slot_type,
            "required": required,
            "pattern": pattern,
            "allowed_values": allowed_values,
            "prompt": prompt,
        }

    def set_required_slots(
        self,
        dialogue_id: str,
        slots: List[str],
    ) -> None:
        """Set required slots for dialogue."""
        context = self._dialogues.get(dialogue_id)
        if context:
            context.required_slots = slots
            context.missing_slots = [
                s for s in slots if s not in context.filled_slots
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        states: Dict[str, int] = {}
        for ctx in self._dialogues.values():
            state = ctx.current_state.value
            states[state] = states.get(state, 0) + 1

        return {
            "active_dialogues": len(self._dialogues),
            "states": states,
            "registered_handlers": len(self._intent_handlers),
        }
