"""Dialogue manager for conversation flow control."""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import asyncio

from app.nlu.processor import NLUResult

logger = logging.getLogger(__name__)


class DialogueState(str, Enum):
    """States of the dialogue."""
    GREETING = "greeting"
    ELICITING = "eliciting"
    CONFIRMING = "confirming"
    PROCESSING = "processing"
    INFORMING = "informing"
    CLARIFYING = "clarifying"
    TRANSFERRING = "transferring"
    CLOSING = "closing"
    ERROR = "error"


@dataclass
class DialogueAction:
    """Action to be taken by the dialogue system."""
    type: str  # speak, listen, transfer, hangup, etc.
    content: Optional[str] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "content": self.content,
            "slots": self.slots,
            "metadata": self.metadata,
        }


@dataclass
class DialogueContext:
    """Context for dialogue management."""
    session_id: str
    current_state: DialogueState = DialogueState.GREETING
    current_intent: Optional[str] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    required_slots: List[str] = field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    last_action: Optional[DialogueAction] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "current_state": self.current_state.value,
            "current_intent": self.current_intent,
            "slots": self.slots,
            "filled_slots": self.filled_slots,
            "required_slots": self.required_slots,
            "turn_count": self.turn_count,
            "last_action": self.last_action.to_dict() if self.last_action else None,
        }

    def set_slot(self, name: str, value: Any) -> None:
        """Set a slot value."""
        self.slots[name] = value
        self.filled_slots[name] = value

    def get_slot(self, name: str) -> Optional[Any]:
        """Get a slot value."""
        return self.filled_slots.get(name)

    def has_slot(self, name: str) -> bool:
        """Check if a slot is filled."""
        return name in self.filled_slots and self.filled_slots[name] is not None

    def get_missing_slots(self) -> List[str]:
        """Get list of required but unfilled slots."""
        return [s for s in self.required_slots if not self.has_slot(s)]

    def all_slots_filled(self) -> bool:
        """Check if all required slots are filled."""
        return len(self.get_missing_slots()) == 0

    def add_history(self, role: str, content: str, nlu_result: Optional[NLUResult] = None) -> None:
        """Add to conversation history."""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if nlu_result:
            entry["intent"] = nlu_result.intent.name
            entry["entities"] = [e.to_dict() for e in nlu_result.entities]
        self.conversation_history.append(entry)
        self.turn_count += 1


class DialoguePolicy:
    """Base class for dialogue policies."""

    async def select_action(
        self,
        context: DialogueContext,
        nlu_result: Optional[NLUResult] = None,
    ) -> DialogueAction:
        """Select the next action based on context."""
        raise NotImplementedError


class RuleBasedPolicy(DialoguePolicy):
    """
    Rule-based dialogue policy.

    Uses if-then rules to select actions.
    """

    def __init__(self):
        self._rules: List[Callable[[DialogueContext, Optional[NLUResult]], Optional[DialogueAction]]] = []
        self._intent_handlers: Dict[str, Callable[[DialogueContext, NLUResult], Awaitable[DialogueAction]]] = {}
        self._state_handlers: Dict[DialogueState, Callable[[DialogueContext], Awaitable[DialogueAction]]] = {}
        self._slot_prompts: Dict[str, str] = {}
        self._setup_default_handlers()

    def _setup_default_handlers(self) -> None:
        """Setup default handlers."""
        # Default state handlers
        self._state_handlers[DialogueState.GREETING] = self._handle_greeting
        self._state_handlers[DialogueState.ELICITING] = self._handle_eliciting
        self._state_handlers[DialogueState.CONFIRMING] = self._handle_confirming
        self._state_handlers[DialogueState.CLOSING] = self._handle_closing
        self._state_handlers[DialogueState.ERROR] = self._handle_error

    def add_rule(
        self,
        rule: Callable[[DialogueContext, Optional[NLUResult]], Optional[DialogueAction]],
    ) -> None:
        """Add a rule to the policy."""
        self._rules.append(rule)

    def add_intent_handler(
        self,
        intent: str,
        handler: Callable[[DialogueContext, NLUResult], Awaitable[DialogueAction]],
    ) -> None:
        """Add a handler for a specific intent."""
        self._intent_handlers[intent] = handler

    def add_state_handler(
        self,
        state: DialogueState,
        handler: Callable[[DialogueContext], Awaitable[DialogueAction]],
    ) -> None:
        """Add a handler for a specific state."""
        self._state_handlers[state] = handler

    def set_slot_prompt(self, slot_name: str, prompt: str) -> None:
        """Set the prompt for a slot."""
        self._slot_prompts[slot_name] = prompt

    async def select_action(
        self,
        context: DialogueContext,
        nlu_result: Optional[NLUResult] = None,
    ) -> DialogueAction:
        """Select action using rules."""
        # Check custom rules first
        for rule in self._rules:
            action = rule(context, nlu_result)
            if action:
                return action

        # Handle specific intents
        if nlu_result and nlu_result.intent.name in self._intent_handlers:
            handler = self._intent_handlers[nlu_result.intent.name]
            return await handler(context, nlu_result)

        # Handle special intents
        if nlu_result:
            if nlu_result.intent.name == "goodbye":
                context.current_state = DialogueState.CLOSING
            elif nlu_result.intent.name == "transfer":
                context.current_state = DialogueState.TRANSFERRING
            elif nlu_result.intent.name == "cancel":
                return DialogueAction(
                    type="speak",
                    content="No problem. Is there anything else I can help you with?",
                )

        # Handle current state
        if context.current_state in self._state_handlers:
            handler = self._state_handlers[context.current_state]
            return await handler(context)

        # Check for missing slots
        missing_slots = context.get_missing_slots()
        if missing_slots:
            slot_name = missing_slots[0]
            prompt = self._slot_prompts.get(
                slot_name,
                f"What is your {slot_name.replace('_', ' ')}?",
            )
            return DialogueAction(
                type="speak",
                content=prompt,
                metadata={"asking_slot": slot_name},
            )

        # Default action
        return DialogueAction(
            type="speak",
            content="How can I help you?",
        )

    async def _handle_greeting(self, context: DialogueContext) -> DialogueAction:
        """Handle greeting state."""
        context.current_state = DialogueState.ELICITING
        return DialogueAction(
            type="speak",
            content="Hello! How can I help you today?",
        )

    async def _handle_eliciting(self, context: DialogueContext) -> DialogueAction:
        """Handle eliciting state."""
        return DialogueAction(
            type="listen",
        )

    async def _handle_confirming(self, context: DialogueContext) -> DialogueAction:
        """Handle confirming state."""
        slots_summary = ", ".join(
            f"{k}: {v}" for k, v in context.filled_slots.items()
        )
        return DialogueAction(
            type="speak",
            content=f"Let me confirm: {slots_summary}. Is that correct?",
        )

    async def _handle_closing(self, context: DialogueContext) -> DialogueAction:
        """Handle closing state."""
        return DialogueAction(
            type="speak",
            content="Thank you for calling. Have a great day!",
            metadata={"end_call": True},
        )

    async def _handle_error(self, context: DialogueContext) -> DialogueAction:
        """Handle error state."""
        return DialogueAction(
            type="speak",
            content="I'm sorry, I encountered an error. Let me transfer you to an agent.",
            metadata={"transfer": True},
        )


class DialogueManager:
    """
    Main dialogue manager.

    Orchestrates the conversation flow using NLU and dialogue policy.

    Usage:
        manager = DialogueManager()
        manager.set_required_slots(["name", "phone", "appointment_date"])

        # Process user input
        action = await manager.process_turn("I'd like to schedule an appointment")
        print(action.content)  # "What is your name?"
    """

    def __init__(
        self,
        session_id: str,
        policy: Optional[DialoguePolicy] = None,
        nlu_processor: Optional[Any] = None,
    ):
        self.session_id = session_id
        self.policy = policy or RuleBasedPolicy()
        self._nlu_processor = nlu_processor
        self.context = DialogueContext(session_id=session_id)

    @property
    def nlu_processor(self):
        """Get NLU processor, lazy loaded."""
        if self._nlu_processor is None:
            from app.nlu.processor import get_nlu_processor
            self._nlu_processor = get_nlu_processor()
        return self._nlu_processor

    def set_required_slots(self, slots: List[str]) -> None:
        """Set required slots for the conversation."""
        self.context.required_slots = slots

    def set_slot_prompts(self, prompts: Dict[str, str]) -> None:
        """Set prompts for slots."""
        if isinstance(self.policy, RuleBasedPolicy):
            for slot, prompt in prompts.items():
                self.policy.set_slot_prompt(slot, prompt)

    async def process_turn(
        self,
        user_input: str,
        nlu_result: Optional[NLUResult] = None,
    ) -> DialogueAction:
        """
        Process a user turn and return the next action.

        Args:
            user_input: User's text input
            nlu_result: Pre-computed NLU result (optional)

        Returns:
            Next action to take
        """
        # Get NLU result if not provided
        if nlu_result is None:
            nlu_result = await self.nlu_processor.process(
                user_input,
                context=self.context.to_dict(),
            )

        # Add to history
        self.context.add_history("user", user_input, nlu_result)

        # Update context with intent
        self.context.current_intent = nlu_result.intent.name

        # Extract slot values from entities
        self._fill_slots_from_entities(nlu_result)

        # Handle affirmative/negative in confirmation state
        if self.context.current_state == DialogueState.CONFIRMING:
            if nlu_result.intent.name == "affirmative":
                self.context.current_state = DialogueState.PROCESSING
            elif nlu_result.intent.name == "negative":
                # Clear slots and restart
                self.context.filled_slots = {}
                self.context.current_state = DialogueState.ELICITING

        # Select action
        action = await self.policy.select_action(self.context, nlu_result)

        # Update context with action
        self.context.last_action = action

        # Add assistant response to history
        if action.content:
            self.context.add_history("assistant", action.content)

        # Check if all slots are filled
        if (
            self.context.all_slots_filled()
            and self.context.current_state == DialogueState.ELICITING
        ):
            self.context.current_state = DialogueState.CONFIRMING
            action = await self.policy.select_action(self.context, nlu_result)

        return action

    def _fill_slots_from_entities(self, nlu_result: NLUResult) -> None:
        """Fill slots from extracted entities."""
        # Map entity types to slot names
        entity_slot_mapping = {
            "date": "date",
            "time": "time",
            "phone_number": "phone",
            "email": "email",
            "number": "number",
            "person": "name",
        }

        for entity in nlu_result.entities:
            slot_name = entity_slot_mapping.get(entity.type.value)
            if slot_name and slot_name in self.context.required_slots:
                self.context.set_slot(slot_name, entity.value)

            # Also check custom entity metadata
            if entity.metadata.get("entity_type"):
                custom_type = entity.metadata["entity_type"]
                if custom_type in self.context.required_slots:
                    self.context.set_slot(custom_type, entity.value)

    async def start_conversation(self) -> DialogueAction:
        """Start a new conversation."""
        return await self.policy.select_action(self.context, None)

    def get_context(self) -> DialogueContext:
        """Get current dialogue context."""
        return self.context

    def reset(self) -> None:
        """Reset the dialogue manager."""
        self.context = DialogueContext(session_id=self.session_id)


# Global dialogue manager factory
_dialogue_managers: Dict[str, DialogueManager] = {}


def get_dialogue_manager(session_id: str) -> DialogueManager:
    """Get or create a dialogue manager for a session."""
    if session_id not in _dialogue_managers:
        _dialogue_managers[session_id] = DialogueManager(session_id)
    return _dialogue_managers[session_id]


def remove_dialogue_manager(session_id: str) -> None:
    """Remove a dialogue manager for a session."""
    if session_id in _dialogue_managers:
        del _dialogue_managers[session_id]
