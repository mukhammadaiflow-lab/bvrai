"""
Slot Filling and Entity Extraction

This module handles slot definitions, validation, and filling
for collecting structured information during conversations.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Pattern,
    Tuple,
    Union,
)

from .context import ConversationContext


logger = logging.getLogger(__name__)


class SlotType(str, Enum):
    """Common slot types with built-in extraction."""

    TEXT = "text"
    NUMBER = "number"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    PHONE = "phone"
    EMAIL = "email"
    URL = "url"
    CURRENCY = "currency"
    DURATION = "duration"
    CHOICE = "choice"        # From predefined options
    ENTITY = "entity"        # Named entity
    CUSTOM = "custom"        # Custom extraction


@dataclass
class SlotValue:
    """Extracted slot value."""

    raw_value: str
    normalized_value: Any
    confidence: float = 1.0

    # Extraction details
    source: str = ""  # What extracted this
    start_pos: int = 0
    end_pos: int = 0

    # Validation
    is_valid: bool = True
    validation_error: Optional[str] = None


@dataclass
class SlotValidationResult:
    """Result of slot validation."""

    is_valid: bool
    normalized_value: Any = None
    error_message: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)


@dataclass
class Slot:
    """Definition of a slot to be filled."""

    # Identification
    name: str
    description: str = ""
    slot_type: SlotType = SlotType.TEXT

    # Prompts
    prompt: str = ""  # Question to ask
    reprompt: str = ""  # If validation fails
    examples: List[str] = field(default_factory=list)

    # Validation
    required: bool = True
    validation_pattern: Optional[str] = None
    validator: Optional[Callable[[Any], SlotValidationResult]] = None

    # For CHOICE type
    choices: List[str] = field(default_factory=list)
    allow_synonyms: Dict[str, str] = field(default_factory=dict)  # synonym -> canonical

    # Extraction
    extraction_patterns: List[str] = field(default_factory=list)
    entity_type: Optional[str] = None  # For ENTITY type

    # Behavior
    confirm: bool = False  # Ask for confirmation
    max_attempts: int = 3
    allow_skip: bool = False
    skip_phrase: str = "skip"

    # Default
    default_value: Any = None

    def __post_init__(self):
        # Compile patterns
        self._compiled_patterns: List[Pattern] = []
        for pattern in self.extraction_patterns:
            try:
                self._compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid pattern for slot {self.name}: {e}")

        if self.validation_pattern:
            try:
                self._validation_pattern = re.compile(
                    self.validation_pattern,
                    re.IGNORECASE
                )
            except re.error:
                self._validation_pattern = None
        else:
            self._validation_pattern = None


class SlotExtractor(ABC):
    """Abstract base class for slot extractors."""

    @abstractmethod
    def extract(
        self,
        text: str,
        slot: Slot,
        context: ConversationContext,
    ) -> Optional[SlotValue]:
        """Extract slot value from text."""
        pass


class RegexSlotExtractor(SlotExtractor):
    """Pattern-based slot extractor."""

    # Built-in patterns for common types
    TYPE_PATTERNS = {
        SlotType.EMAIL: r'[\w\.-]+@[\w\.-]+\.\w+',
        SlotType.PHONE: r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{4,6}',
        SlotType.URL: r'https?://[^\s]+',
        SlotType.NUMBER: r'-?\d+(?:\.\d+)?',
        SlotType.INTEGER: r'-?\d+',
        SlotType.DATE: r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        SlotType.TIME: r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?',
        SlotType.CURRENCY: r'\$?\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:dollars?|usd)',
    }

    def extract(
        self,
        text: str,
        slot: Slot,
        context: ConversationContext,
    ) -> Optional[SlotValue]:
        """Extract using patterns."""
        # Try slot-specific patterns first
        for pattern in slot._compiled_patterns:
            match = pattern.search(text)
            if match:
                return SlotValue(
                    raw_value=match.group(0),
                    normalized_value=self._normalize(match.group(0), slot),
                    source="slot_pattern",
                    start_pos=match.start(),
                    end_pos=match.end(),
                )

        # Try built-in type patterns
        if slot.slot_type in self.TYPE_PATTERNS:
            pattern = re.compile(self.TYPE_PATTERNS[slot.slot_type], re.IGNORECASE)
            match = pattern.search(text)
            if match:
                return SlotValue(
                    raw_value=match.group(0),
                    normalized_value=self._normalize(match.group(0), slot),
                    source="type_pattern",
                    start_pos=match.start(),
                    end_pos=match.end(),
                )

        # For CHOICE type, match against options
        if slot.slot_type == SlotType.CHOICE and slot.choices:
            text_lower = text.lower()

            # Check synonyms first
            for synonym, canonical in slot.allow_synonyms.items():
                if synonym.lower() in text_lower:
                    return SlotValue(
                        raw_value=synonym,
                        normalized_value=canonical,
                        source="synonym",
                    )

            # Check choices
            for choice in slot.choices:
                if choice.lower() in text_lower:
                    return SlotValue(
                        raw_value=choice,
                        normalized_value=choice,
                        source="choice",
                    )

        # For TEXT type, return the whole text if no pattern
        if slot.slot_type == SlotType.TEXT and not slot.extraction_patterns:
            return SlotValue(
                raw_value=text,
                normalized_value=text.strip(),
                source="full_text",
            )

        return None

    def _normalize(self, value: str, slot: Slot) -> Any:
        """Normalize extracted value based on type."""
        try:
            if slot.slot_type == SlotType.INTEGER:
                return int(value.replace(',', ''))
            elif slot.slot_type in (SlotType.NUMBER, SlotType.FLOAT):
                return float(value.replace(',', ''))
            elif slot.slot_type == SlotType.BOOLEAN:
                return value.lower() in ('yes', 'true', '1', 'yeah', 'yep')
            elif slot.slot_type == SlotType.EMAIL:
                return value.lower()
            elif slot.slot_type == SlotType.PHONE:
                # Remove non-digits except +
                return re.sub(r'[^\d+]', '', value)
            elif slot.slot_type == SlotType.CURRENCY:
                # Extract numeric value
                num_match = re.search(r'[\d,]+(?:\.\d+)?', value)
                if num_match:
                    return float(num_match.group().replace(',', ''))
        except (ValueError, AttributeError):
            pass

        return value


class LLMSlotExtractor(SlotExtractor):
    """LLM-based slot extractor for complex extractions."""

    def __init__(
        self,
        llm_client: Any,
        model: str = "gpt-4o-mini",
    ):
        self._llm = llm_client
        self._model = model

    def extract(
        self,
        text: str,
        slot: Slot,
        context: ConversationContext,
    ) -> Optional[SlotValue]:
        """Extract using LLM."""
        import asyncio

        async def _extract():
            prompt = f"""Extract the following information from the user's message:

Slot: {slot.name}
Description: {slot.description}
Type: {slot.slot_type.value}
"""
            if slot.choices:
                prompt += f"Valid options: {', '.join(slot.choices)}\n"

            if slot.examples:
                prompt += f"Examples: {', '.join(slot.examples)}\n"

            prompt += f"""
User message: "{text}"

If you can find the requested information, respond with JSON:
{{"value": "<extracted value>", "confidence": <0.0-1.0>}}

If you cannot find the information, respond with:
{{"value": null, "confidence": 0}}"""

            try:
                from ..llm.base import LLMMessage, LLMConfig

                response = await self._llm.complete(
                    messages=[LLMMessage.user(prompt)],
                    config=LLMConfig(
                        model=self._model,
                        temperature=0.1,
                        response_format={"type": "json_object"},
                    ),
                )

                import json
                result = json.loads(response.content)

                if result.get("value") is not None:
                    return SlotValue(
                        raw_value=text,
                        normalized_value=result["value"],
                        confidence=float(result.get("confidence", 0.8)),
                        source="llm",
                    )

            except Exception as e:
                logger.warning(f"LLM slot extraction failed: {e}")

            return None

        return asyncio.run(_extract())


class SlotValidator:
    """Validates slot values."""

    def validate(
        self,
        value: Any,
        slot: Slot,
        context: ConversationContext,
    ) -> SlotValidationResult:
        """Validate a slot value."""
        # Custom validator takes precedence
        if slot.validator:
            try:
                return slot.validator(value)
            except Exception as e:
                return SlotValidationResult(
                    is_valid=False,
                    error_message=f"Validation error: {e}",
                )

        # Pattern validation
        if slot._validation_pattern and isinstance(value, str):
            if not slot._validation_pattern.match(value):
                return SlotValidationResult(
                    is_valid=False,
                    error_message=slot.reprompt or f"Invalid format for {slot.name}",
                )

        # Choice validation
        if slot.slot_type == SlotType.CHOICE and slot.choices:
            value_lower = str(value).lower()

            # Check synonyms
            if value_lower in [s.lower() for s in slot.allow_synonyms]:
                canonical = slot.allow_synonyms.get(value, value)
                return SlotValidationResult(
                    is_valid=True,
                    normalized_value=canonical,
                )

            # Check choices
            for choice in slot.choices:
                if choice.lower() == value_lower:
                    return SlotValidationResult(
                        is_valid=True,
                        normalized_value=choice,
                    )

            return SlotValidationResult(
                is_valid=False,
                error_message=f"Please choose from: {', '.join(slot.choices)}",
                suggestions=slot.choices,
            )

        # Type-specific validation
        if slot.slot_type == SlotType.EMAIL:
            if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', str(value)):
                return SlotValidationResult(
                    is_valid=False,
                    error_message="Please provide a valid email address",
                )

        elif slot.slot_type == SlotType.PHONE:
            # Basic phone validation
            digits = re.sub(r'\D', '', str(value))
            if len(digits) < 10:
                return SlotValidationResult(
                    is_valid=False,
                    error_message="Please provide a valid phone number",
                )

        elif slot.slot_type == SlotType.INTEGER:
            try:
                int(value)
            except (ValueError, TypeError):
                return SlotValidationResult(
                    is_valid=False,
                    error_message="Please provide a whole number",
                )

        elif slot.slot_type == SlotType.NUMBER:
            try:
                float(value)
            except (ValueError, TypeError):
                return SlotValidationResult(
                    is_valid=False,
                    error_message="Please provide a number",
                )

        return SlotValidationResult(
            is_valid=True,
            normalized_value=value,
        )


class SlotFiller:
    """
    Manages slot filling process during conversation.
    """

    def __init__(
        self,
        extractor: Optional[SlotExtractor] = None,
        validator: Optional[SlotValidator] = None,
    ):
        self._extractor = extractor or RegexSlotExtractor()
        self._validator = validator or SlotValidator()

        # Track fill attempts per session
        self._attempts: Dict[str, Dict[str, int]] = {}

    def fill(
        self,
        text: str,
        slots: List[Slot],
        context: ConversationContext,
    ) -> Dict[str, SlotValue]:
        """
        Extract and validate slots from text.

        Returns dictionary of slot name -> SlotValue for successfully filled slots.
        """
        filled = {}

        for slot in slots:
            # Skip if already filled in context
            if context.get_slot(slot.name) is not None and not slot.required:
                continue

            # Try extraction
            value = self._extractor.extract(text, slot, context)

            if value:
                # Validate
                validation = self._validator.validate(
                    value.normalized_value,
                    slot,
                    context,
                )

                if validation.is_valid:
                    value.normalized_value = validation.normalized_value or value.normalized_value
                    filled[slot.name] = value
                else:
                    value.is_valid = False
                    value.validation_error = validation.error_message
                    filled[slot.name] = value

        return filled

    def get_next_required_slot(
        self,
        slots: List[Slot],
        context: ConversationContext,
    ) -> Optional[Slot]:
        """Get the next unfilled required slot."""
        session_id = context.session_id

        for slot in slots:
            if not slot.required:
                continue

            # Check if already filled
            if context.get_slot(slot.name) is not None:
                continue

            # Check if max attempts reached
            if session_id in self._attempts:
                attempts = self._attempts[session_id].get(slot.name, 0)
                if attempts >= slot.max_attempts:
                    continue

            return slot

        return None

    def record_attempt(
        self,
        slot_name: str,
        session_id: str,
    ) -> int:
        """Record a fill attempt and return total attempts."""
        if session_id not in self._attempts:
            self._attempts[session_id] = {}

        self._attempts[session_id][slot_name] = (
            self._attempts[session_id].get(slot_name, 0) + 1
        )

        return self._attempts[session_id][slot_name]

    def reset_attempts(self, session_id: str) -> None:
        """Reset attempt counts for a session."""
        if session_id in self._attempts:
            del self._attempts[session_id]

    def get_prompt_for_slot(
        self,
        slot: Slot,
        context: ConversationContext,
        is_reprompt: bool = False,
    ) -> str:
        """Get the prompt to ask for a slot."""
        if is_reprompt and slot.reprompt:
            prompt = slot.reprompt
        else:
            prompt = slot.prompt or f"What is your {slot.name}?"

        # Add examples if available
        if slot.examples and not is_reprompt:
            prompt += f" (for example: {slot.examples[0]})"

        # Add choices if available
        if slot.slot_type == SlotType.CHOICE and slot.choices:
            prompt += f" Options: {', '.join(slot.choices)}"

        return prompt


# Pre-defined common slots
COMMON_SLOTS = {
    "name": Slot(
        name="name",
        description="Person's name",
        slot_type=SlotType.TEXT,
        prompt="May I have your name please?",
        reprompt="I didn't catch your name. Could you spell it for me?",
    ),
    "email": Slot(
        name="email",
        description="Email address",
        slot_type=SlotType.EMAIL,
        prompt="What's your email address?",
        reprompt="That doesn't look like a valid email. Please provide your email address.",
    ),
    "phone": Slot(
        name="phone",
        description="Phone number",
        slot_type=SlotType.PHONE,
        prompt="What's the best phone number to reach you?",
        reprompt="Please provide a valid phone number with area code.",
    ),
    "confirmation": Slot(
        name="confirmation",
        description="Yes/No confirmation",
        slot_type=SlotType.CHOICE,
        choices=["yes", "no"],
        allow_synonyms={
            "yeah": "yes",
            "yep": "yes",
            "sure": "yes",
            "correct": "yes",
            "nope": "no",
            "nah": "no",
        },
        prompt="Is that correct?",
    ),
}


__all__ = [
    "SlotType",
    "SlotValue",
    "SlotValidationResult",
    "Slot",
    "SlotExtractor",
    "RegexSlotExtractor",
    "LLMSlotExtractor",
    "SlotValidator",
    "SlotFiller",
    "COMMON_SLOTS",
]
