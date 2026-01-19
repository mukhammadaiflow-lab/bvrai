"""Slot filling for dialogue management."""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

from app.nlu.processor import NLUResult
from app.nlu.entities import Entity, EntityType

logger = logging.getLogger(__name__)


class SlotType(str, Enum):
    """Types of slots."""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    TIME = "time"
    PHONE = "phone"
    EMAIL = "email"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    CUSTOM = "custom"


@dataclass
class Slot:
    """Definition of a slot to be filled."""
    name: str
    type: SlotType
    prompt: str
    required: bool = True
    choices: List[str] = field(default_factory=list)
    default: Optional[Any] = None
    validator: Optional[Callable[[Any], bool]] = None
    transformer: Optional[Callable[[Any], Any]] = None
    reprompt: Optional[str] = None
    max_attempts: int = 3
    entity_types: List[EntityType] = field(default_factory=list)

    def validate(self, value: Any) -> bool:
        """Validate a slot value."""
        if value is None:
            return not self.required

        # Type-specific validation
        if self.type == SlotType.NUMBER:
            try:
                float(value)
            except (TypeError, ValueError):
                return False

        elif self.type == SlotType.BOOLEAN:
            return isinstance(value, bool) or value in ["yes", "no", "true", "false"]

        elif self.type == SlotType.CHOICE:
            return value.lower() in [c.lower() for c in self.choices]

        elif self.type == SlotType.EMAIL:
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, str(value)))

        elif self.type == SlotType.PHONE:
            pattern = r'^\+?[\d\s\-()]{10,}$'
            return bool(re.match(pattern, str(value)))

        # Custom validator
        if self.validator:
            return self.validator(value)

        return True

    def transform(self, value: Any) -> Any:
        """Transform a slot value."""
        if value is None:
            return self.default

        # Type-specific transformation
        if self.type == SlotType.NUMBER:
            try:
                return float(value) if '.' in str(value) else int(value)
            except (TypeError, ValueError):
                return value

        elif self.type == SlotType.BOOLEAN:
            if isinstance(value, bool):
                return value
            return value.lower() in ["yes", "true", "yeah", "yep", "correct"]

        elif self.type == SlotType.TEXT:
            return str(value).strip()

        # Custom transformer
        if self.transformer:
            return self.transformer(value)

        return value


@dataclass
class SlotFillResult:
    """Result of attempting to fill a slot."""
    slot_name: str
    filled: bool
    value: Optional[Any] = None
    confidence: float = 0.0
    source: str = "unknown"  # entity, text, default


class SlotFiller:
    """
    Fills slots from NLU results.

    Maps entities and text to slot values.
    """

    # Default entity type to slot type mapping
    ENTITY_SLOT_MAPPING = {
        EntityType.DATE: SlotType.DATE,
        EntityType.TIME: SlotType.TIME,
        EntityType.PHONE_NUMBER: SlotType.PHONE,
        EntityType.EMAIL: SlotType.EMAIL,
        EntityType.NUMBER: SlotType.NUMBER,
        EntityType.CURRENCY: SlotType.NUMBER,
        EntityType.PERCENTAGE: SlotType.NUMBER,
    }

    def __init__(self, slots: List[Slot]):
        self.slots = {slot.name: slot for slot in slots}
        self._custom_fillers: Dict[str, Callable] = {}

    def add_custom_filler(
        self,
        slot_name: str,
        filler: Callable[[str, NLUResult], Optional[Any]],
    ) -> None:
        """Add a custom filler for a slot."""
        self._custom_fillers[slot_name] = filler

    def fill_from_nlu(
        self,
        nlu_result: NLUResult,
        target_slot: Optional[str] = None,
    ) -> List[SlotFillResult]:
        """
        Fill slots from NLU result.

        Args:
            nlu_result: NLU analysis result
            target_slot: Specific slot to fill (optional)

        Returns:
            List of slot fill results
        """
        results = []

        slots_to_fill = (
            [self.slots[target_slot]] if target_slot
            else self.slots.values()
        )

        for slot in slots_to_fill:
            result = self._fill_slot(slot, nlu_result)
            if result.filled:
                results.append(result)

        return results

    def _fill_slot(self, slot: Slot, nlu_result: NLUResult) -> SlotFillResult:
        """Fill a single slot."""
        # Try custom filler first
        if slot.name in self._custom_fillers:
            value = self._custom_fillers[slot.name](nlu_result.text, nlu_result)
            if value is not None and slot.validate(value):
                return SlotFillResult(
                    slot_name=slot.name,
                    filled=True,
                    value=slot.transform(value),
                    confidence=0.9,
                    source="custom",
                )

        # Try to fill from entities
        for entity in nlu_result.entities:
            if self._entity_matches_slot(entity, slot):
                value = entity.value
                if slot.validate(value):
                    return SlotFillResult(
                        slot_name=slot.name,
                        filled=True,
                        value=slot.transform(value),
                        confidence=entity.confidence,
                        source="entity",
                    )

        # Try to fill from text for simple types
        if slot.type == SlotType.BOOLEAN:
            intent = nlu_result.intent.name
            if intent in ["affirmative", "yes"]:
                return SlotFillResult(
                    slot_name=slot.name,
                    filled=True,
                    value=True,
                    confidence=nlu_result.intent.confidence,
                    source="intent",
                )
            elif intent in ["negative", "no"]:
                return SlotFillResult(
                    slot_name=slot.name,
                    filled=True,
                    value=False,
                    confidence=nlu_result.intent.confidence,
                    source="intent",
                )

        elif slot.type == SlotType.CHOICE:
            text_lower = nlu_result.text.lower()
            for choice in slot.choices:
                if choice.lower() in text_lower:
                    return SlotFillResult(
                        slot_name=slot.name,
                        filled=True,
                        value=choice,
                        confidence=0.8,
                        source="text",
                    )

        elif slot.type == SlotType.TEXT:
            # For text slots, use the entire text if no entity found
            if len(nlu_result.entities) == 0:
                value = nlu_result.text.strip()
                if value and slot.validate(value):
                    return SlotFillResult(
                        slot_name=slot.name,
                        filled=True,
                        value=slot.transform(value),
                        confidence=0.6,
                        source="text",
                    )

        return SlotFillResult(
            slot_name=slot.name,
            filled=False,
        )

    def _entity_matches_slot(self, entity: Entity, slot: Slot) -> bool:
        """Check if an entity matches a slot."""
        # Check explicit entity types
        if slot.entity_types and entity.type in slot.entity_types:
            return True

        # Check default mapping
        expected_slot_type = self.ENTITY_SLOT_MAPPING.get(entity.type)
        if expected_slot_type == slot.type:
            return True

        return False


class FormManager:
    """
    Manages slot-filling forms.

    Guides the conversation through filling required slots.
    """

    def __init__(self, slots: List[Slot]):
        self.slots = slots
        self.slot_filler = SlotFiller(slots)
        self._filled_values: Dict[str, Any] = {}
        self._attempts: Dict[str, int] = {}
        self._current_slot_index = 0

    @property
    def current_slot(self) -> Optional[Slot]:
        """Get the current slot being filled."""
        if self._current_slot_index >= len(self.slots):
            return None
        return self.slots[self._current_slot_index]

    @property
    def is_complete(self) -> bool:
        """Check if all required slots are filled."""
        for slot in self.slots:
            if slot.required and slot.name not in self._filled_values:
                return False
        return True

    def get_filled_values(self) -> Dict[str, Any]:
        """Get all filled slot values."""
        return self._filled_values.copy()

    def get_missing_slots(self) -> List[Slot]:
        """Get slots that still need to be filled."""
        return [
            slot for slot in self.slots
            if slot.required and slot.name not in self._filled_values
        ]

    def process_input(self, nlu_result: NLUResult) -> Optional[str]:
        """
        Process user input and try to fill slots.

        Returns the next prompt or None if complete.
        """
        current = self.current_slot
        if not current:
            return None

        # Try to fill current slot
        results = self.slot_filler.fill_from_nlu(nlu_result, current.name)

        if results and results[0].filled:
            self._filled_values[current.name] = results[0].value
            self._current_slot_index += 1
            self._attempts[current.name] = 0
        else:
            # Increment attempts
            self._attempts[current.name] = self._attempts.get(current.name, 0) + 1

            # Check max attempts
            if self._attempts[current.name] >= current.max_attempts:
                if current.default is not None:
                    self._filled_values[current.name] = current.default
                    self._current_slot_index += 1
                else:
                    return f"I'm sorry, I couldn't understand. {current.reprompt or current.prompt}"

            return current.reprompt or current.prompt

        # Also try to fill other slots from entities
        for slot in self.slots[self._current_slot_index:]:
            if slot.name not in self._filled_values:
                results = self.slot_filler.fill_from_nlu(nlu_result, slot.name)
                if results and results[0].filled:
                    self._filled_values[slot.name] = results[0].value

        # Get next prompt
        return self.get_next_prompt()

    def get_next_prompt(self) -> Optional[str]:
        """Get the prompt for the next unfilled slot."""
        for i, slot in enumerate(self.slots[self._current_slot_index:], self._current_slot_index):
            if slot.name not in self._filled_values:
                self._current_slot_index = i
                return slot.prompt
        return None

    def set_value(self, slot_name: str, value: Any) -> bool:
        """Manually set a slot value."""
        if slot_name in {s.name for s in self.slots}:
            slot = next(s for s in self.slots if s.name == slot_name)
            if slot.validate(value):
                self._filled_values[slot_name] = slot.transform(value)
                return True
        return False

    def reset(self) -> None:
        """Reset the form."""
        self._filled_values = {}
        self._attempts = {}
        self._current_slot_index = 0


class SlotFillingDialogue:
    """
    Dialogue handler for slot-filling conversations.

    Manages the entire slot-filling flow with validation and reprompts.
    """

    def __init__(
        self,
        slots: List[Slot],
        intro_message: str = "Let me collect some information from you.",
        confirmation_message: str = "Let me confirm the details:",
        completion_message: str = "Thank you! I have all the information I need.",
        on_complete: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ):
        self.slots = slots
        self.form = FormManager(slots)
        self.intro_message = intro_message
        self.confirmation_message = confirmation_message
        self.completion_message = completion_message
        self.on_complete = on_complete
        self._started = False
        self._confirmed = False

    async def start(self) -> str:
        """Start the slot-filling dialogue."""
        self._started = True
        first_prompt = self.form.get_next_prompt()
        if first_prompt:
            return f"{self.intro_message} {first_prompt}"
        return self.intro_message

    async def process(self, nlu_result: NLUResult) -> str:
        """Process user input and return response."""
        if not self._started:
            return await self.start()

        # Check for confirmation response
        if self.form.is_complete and not self._confirmed:
            if nlu_result.intent.name == "affirmative":
                self._confirmed = True
                if self.on_complete:
                    await self.on_complete(self.form.get_filled_values())
                return self.completion_message
            elif nlu_result.intent.name == "negative":
                # Restart form
                self.form.reset()
                return f"No problem, let's start over. {self.form.get_next_prompt()}"

        # Process slot filling
        next_prompt = self.form.process_input(nlu_result)

        if next_prompt:
            return next_prompt

        # All slots filled, ask for confirmation
        values = self.form.get_filled_values()
        summary = ", ".join(f"{k}: {v}" for k, v in values.items())
        return f"{self.confirmation_message} {summary}. Is that correct?"

    @property
    def is_complete(self) -> bool:
        """Check if dialogue is complete."""
        return self._confirmed

    def get_values(self) -> Dict[str, Any]:
        """Get filled values."""
        return self.form.get_filled_values()
