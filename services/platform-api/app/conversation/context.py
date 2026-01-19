"""
Context Management

Manages conversation context:
- Slot tracking
- Context frames
- Memory management
- Variable resolution
"""

from typing import Optional, Dict, Any, List, Callable, Generic, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import copy
import re

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SlotType(str, Enum):
    """Slot data types."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    PHONE = "phone"
    EMAIL = "email"
    CURRENCY = "currency"
    LIST = "list"
    ENTITY = "entity"
    CUSTOM = "custom"


class SlotStatus(str, Enum):
    """Slot filling status."""
    EMPTY = "empty"
    PENDING = "pending"
    FILLED = "filled"
    CONFIRMED = "confirmed"
    INVALID = "invalid"


@dataclass
class ContextSlot:
    """Context slot for storing information."""
    name: str
    slot_type: SlotType = SlotType.STRING
    value: Optional[Any] = None
    status: SlotStatus = SlotStatus.EMPTY

    # Constraints
    required: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None

    # Prompts
    prompt: str = ""
    reprompt: str = ""
    confirmation_prompt: str = ""

    # Metadata
    confidence: float = 0.0
    source: str = ""  # "user", "system", "default"
    filled_at: Optional[datetime] = None
    attempts: int = 0
    max_attempts: int = 3

    def set_value(
        self,
        value: Any,
        source: str = "user",
        confidence: float = 1.0,
    ) -> bool:
        """Set slot value with validation."""
        if self.validate(value):
            self.value = value
            self.source = source
            self.confidence = confidence
            self.status = SlotStatus.FILLED
            self.filled_at = datetime.utcnow()
            return True

        self.status = SlotStatus.INVALID
        return False

    def validate(self, value: Any) -> bool:
        """Validate slot value."""
        if value is None:
            return not self.required

        # Type validation
        if self.slot_type == SlotType.STRING:
            if not isinstance(value, str):
                return False
            if self.pattern and not re.match(self.pattern, value):
                return False

        elif self.slot_type == SlotType.NUMBER:
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False

        elif self.slot_type == SlotType.BOOLEAN:
            if not isinstance(value, bool):
                if isinstance(value, str):
                    value = value.lower() in ("yes", "true", "1", "yeah", "yep")
                else:
                    return False

        elif self.slot_type == SlotType.PHONE:
            if not isinstance(value, str):
                return False
            # Basic phone validation
            digits = re.sub(r"[^\d]", "", value)
            if len(digits) < 10 or len(digits) > 15:
                return False

        elif self.slot_type == SlotType.EMAIL:
            if not isinstance(value, str):
                return False
            if not re.match(r"[^@]+@[^@]+\.[^@]+", value):
                return False

        # Allowed values check
        if self.allowed_values and value not in self.allowed_values:
            return False

        return True

    def clear(self) -> None:
        """Clear slot value."""
        self.value = None
        self.status = SlotStatus.EMPTY
        self.confidence = 0.0
        self.filled_at = None

    def confirm(self) -> None:
        """Confirm slot value."""
        if self.status == SlotStatus.FILLED:
            self.status = SlotStatus.CONFIRMED

    @property
    def is_filled(self) -> bool:
        """Check if slot is filled."""
        return self.status in (SlotStatus.FILLED, SlotStatus.CONFIRMED)

    @property
    def needs_confirmation(self) -> bool:
        """Check if slot needs confirmation."""
        return self.status == SlotStatus.FILLED and self.confidence < 0.9


@dataclass
class ContextFrame:
    """Context frame for scoped information."""
    frame_id: str
    name: str
    parent_id: Optional[str] = None

    # Slots
    slots: Dict[str, ContextSlot] = field(default_factory=dict)

    # Variables
    variables: Dict[str, Any] = field(default_factory=dict)

    # State
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def add_slot(self, slot: ContextSlot) -> None:
        """Add slot to frame."""
        self.slots[slot.name] = slot

    def get_slot(self, name: str) -> Optional[ContextSlot]:
        """Get slot by name."""
        return self.slots.get(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set variable."""
        self.variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get variable."""
        return self.variables.get(name, default)

    def get_filled_slots(self) -> Dict[str, Any]:
        """Get all filled slot values."""
        return {
            name: slot.value
            for name, slot in self.slots.items()
            if slot.is_filled
        }

    def get_missing_required_slots(self) -> List[str]:
        """Get list of missing required slots."""
        return [
            name for name, slot in self.slots.items()
            if slot.required and not slot.is_filled
        ]

    @property
    def is_expired(self) -> bool:
        """Check if frame is expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False


class ContextStack:
    """
    Stack-based context management.

    Supports:
    - Push/pop context frames
    - Variable scoping
    - Inheritance
    """

    def __init__(self):
        self._frames: List[ContextFrame] = []
        self._frame_map: Dict[str, ContextFrame] = {}

    def push(self, frame: ContextFrame) -> None:
        """Push frame onto stack."""
        if self._frames:
            frame.parent_id = self._frames[-1].frame_id

        self._frames.append(frame)
        self._frame_map[frame.frame_id] = frame

    def pop(self) -> Optional[ContextFrame]:
        """Pop frame from stack."""
        if self._frames:
            frame = self._frames.pop()
            self._frame_map.pop(frame.frame_id, None)
            return frame
        return None

    def peek(self) -> Optional[ContextFrame]:
        """Get top frame without removing."""
        return self._frames[-1] if self._frames else None

    def get_frame(self, frame_id: str) -> Optional[ContextFrame]:
        """Get frame by ID."""
        return self._frame_map.get(frame_id)

    def resolve_slot(self, name: str) -> Optional[ContextSlot]:
        """Resolve slot from stack (bottom-up)."""
        for frame in reversed(self._frames):
            slot = frame.get_slot(name)
            if slot:
                return slot
        return None

    def resolve_variable(self, name: str, default: Any = None) -> Any:
        """Resolve variable from stack (bottom-up)."""
        for frame in reversed(self._frames):
            if name in frame.variables:
                return frame.variables[name]
        return default

    def get_all_slots(self) -> Dict[str, ContextSlot]:
        """Get all slots from all frames."""
        slots = {}
        for frame in self._frames:
            slots.update(frame.slots)
        return slots

    def get_all_variables(self) -> Dict[str, Any]:
        """Get all variables from all frames."""
        variables = {}
        for frame in self._frames:
            variables.update(frame.variables)
        return variables

    @property
    def depth(self) -> int:
        """Get stack depth."""
        return len(self._frames)


@dataclass
class ConversationContext:
    """Complete conversation context."""
    context_id: str
    conversation_id: str

    # Context stack
    stack: ContextStack = field(default_factory=ContextStack)

    # Global context
    global_variables: Dict[str, Any] = field(default_factory=dict)

    # User information
    user_id: Optional[str] = None
    user_profile: Dict[str, Any] = field(default_factory=dict)

    # Session information
    session_id: Optional[str] = None
    session_data: Dict[str, Any] = field(default_factory=dict)

    # History
    intent_history: List[str] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def set_variable(
        self,
        name: str,
        value: Any,
        scope: str = "frame",
    ) -> None:
        """Set variable in specified scope."""
        if scope == "global":
            self.global_variables[name] = value
        elif scope == "session":
            self.session_data[name] = value
        else:
            frame = self.stack.peek()
            if frame:
                frame.set_variable(name, value)

        self.last_updated = datetime.utcnow()

    def get_variable(
        self,
        name: str,
        default: Any = None,
    ) -> Any:
        """Get variable (searches all scopes)."""
        # Check stack first
        value = self.stack.resolve_variable(name)
        if value is not None:
            return value

        # Check session
        if name in self.session_data:
            return self.session_data[name]

        # Check global
        if name in self.global_variables:
            return self.global_variables[name]

        # Check user profile
        if name in self.user_profile:
            return self.user_profile[name]

        return default

    def resolve(self, expression: str) -> Any:
        """
        Resolve expression with variable substitution.

        Supports: ${variable}, ${slot.value}, ${user.name}
        """
        def replace_var(match):
            var_path = match.group(1)
            parts = var_path.split(".")

            if parts[0] == "user":
                value = self.user_profile.get(parts[1] if len(parts) > 1 else "id")
            elif parts[0] == "session":
                value = self.session_data.get(parts[1] if len(parts) > 1 else "id")
            elif parts[0] == "slot":
                slot = self.stack.resolve_slot(parts[1] if len(parts) > 1 else "")
                value = slot.value if slot else None
            else:
                value = self.get_variable(var_path)

            return str(value) if value is not None else ""

        return re.sub(r"\$\{([^}]+)\}", replace_var, expression)

    def add_intent(self, intent: str) -> None:
        """Add intent to history."""
        self.intent_history.append(intent)
        self.last_updated = datetime.utcnow()

    def add_action(self, action: Dict[str, Any]) -> None:
        """Add action to history."""
        action["timestamp"] = datetime.utcnow().isoformat()
        self.action_history.append(action)
        self.last_updated = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "context_id": self.context_id,
            "conversation_id": self.conversation_id,
            "global_variables": self.global_variables,
            "session_data": self.session_data,
            "user_profile": self.user_profile,
            "slots": {
                name: {
                    "value": slot.value,
                    "status": slot.status.value,
                    "confidence": slot.confidence,
                }
                for name, slot in self.stack.get_all_slots().items()
            },
            "intent_history": self.intent_history[-10:],
            "action_count": len(self.action_history),
        }


class ContextManager:
    """
    Manages conversation contexts.

    Features:
    - Context creation/retrieval
    - Slot management
    - Variable resolution
    - Context persistence
    """

    def __init__(self):
        self._contexts: Dict[str, ConversationContext] = {}
        self._slot_templates: Dict[str, ContextSlot] = {}
        self._frame_templates: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()

    async def create_context(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
        initial_frame: Optional[str] = None,
    ) -> ConversationContext:
        """Create new conversation context."""
        import uuid

        async with self._lock:
            context = ConversationContext(
                context_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                user_id=user_id,
            )

            # Create initial frame
            frame = ContextFrame(
                frame_id=str(uuid.uuid4()),
                name=initial_frame or "main",
            )

            # Add template slots if defined
            if initial_frame and initial_frame in self._frame_templates:
                for slot_name in self._frame_templates[initial_frame]:
                    if slot_name in self._slot_templates:
                        slot = copy.deepcopy(self._slot_templates[slot_name])
                        frame.add_slot(slot)

            context.stack.push(frame)
            self._contexts[conversation_id] = context

            return context

    async def get_context(
        self,
        conversation_id: str,
    ) -> Optional[ConversationContext]:
        """Get conversation context."""
        return self._contexts.get(conversation_id)

    async def delete_context(self, conversation_id: str) -> bool:
        """Delete conversation context."""
        async with self._lock:
            return self._contexts.pop(conversation_id, None) is not None

    def define_slot_template(self, slot: ContextSlot) -> None:
        """Define reusable slot template."""
        self._slot_templates[slot.name] = slot

    def define_frame_template(
        self,
        frame_name: str,
        slot_names: List[str],
    ) -> None:
        """Define frame template with slots."""
        self._frame_templates[frame_name] = slot_names

    async def push_frame(
        self,
        conversation_id: str,
        frame_name: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Optional[ContextFrame]:
        """Push new frame onto context stack."""
        import uuid

        context = await self.get_context(conversation_id)
        if not context:
            return None

        frame = ContextFrame(
            frame_id=str(uuid.uuid4()),
            name=frame_name,
            variables=variables or {},
        )

        # Add template slots
        if frame_name in self._frame_templates:
            for slot_name in self._frame_templates[frame_name]:
                if slot_name in self._slot_templates:
                    slot = copy.deepcopy(self._slot_templates[slot_name])
                    frame.add_slot(slot)

        context.stack.push(frame)
        return frame

    async def pop_frame(
        self,
        conversation_id: str,
    ) -> Optional[ContextFrame]:
        """Pop frame from context stack."""
        context = await self.get_context(conversation_id)
        if not context:
            return None

        return context.stack.pop()

    async def fill_slot(
        self,
        conversation_id: str,
        slot_name: str,
        value: Any,
        source: str = "user",
        confidence: float = 1.0,
    ) -> bool:
        """Fill slot in current frame."""
        context = await self.get_context(conversation_id)
        if not context:
            return False

        slot = context.stack.resolve_slot(slot_name)
        if not slot:
            return False

        return slot.set_value(value, source, confidence)

    async def get_slot_value(
        self,
        conversation_id: str,
        slot_name: str,
    ) -> Optional[Any]:
        """Get slot value."""
        context = await self.get_context(conversation_id)
        if not context:
            return None

        slot = context.stack.resolve_slot(slot_name)
        return slot.value if slot else None

    async def get_missing_slots(
        self,
        conversation_id: str,
    ) -> List[ContextSlot]:
        """Get list of missing required slots."""
        context = await self.get_context(conversation_id)
        if not context:
            return []

        missing = []
        for slot in context.stack.get_all_slots().values():
            if slot.required and not slot.is_filled:
                missing.append(slot)

        return missing

    async def clear_slots(
        self,
        conversation_id: str,
        slot_names: Optional[List[str]] = None,
    ) -> int:
        """Clear slots in current frame."""
        context = await self.get_context(conversation_id)
        if not context:
            return 0

        frame = context.stack.peek()
        if not frame:
            return 0

        cleared = 0
        for name, slot in frame.slots.items():
            if slot_names is None or name in slot_names:
                slot.clear()
                cleared += 1

        return cleared

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "active_contexts": len(self._contexts),
            "slot_templates": len(self._slot_templates),
            "frame_templates": len(self._frame_templates),
        }
