"""
Builder Engine Conversation Engine

A sophisticated conversation management system that provides:
- State machine-based dialog management
- Intent recognition and handling
- Slot filling for entity extraction
- Multi-turn conversation context
- Dynamic flow control
- Conditional branching and routing
- Action and webhook execution

Designed for building intelligent, context-aware AI voice agents.
"""

from .base import (
    ConversationState,
    ConversationEvent,
    ConversationEventType,
    TurnState,
    ResponseType,
    ConversationConfig,
)
from .context import (
    ConversationContext,
    ContextVariable,
    ContextScope,
    ContextManager,
)
from .intents import (
    Intent,
    IntentMatch,
    IntentHandler,
    IntentRouter,
    StaticIntentMatcher,
    LLMIntentMatcher,
)
from .slots import (
    Slot,
    SlotType,
    SlotValue,
    SlotFiller,
    SlotValidationResult,
)
from .flows import (
    DialogNode,
    DialogFlow,
    FlowTransition,
    FlowCondition,
    FlowAction,
    FlowBuilder,
)
from .state import (
    StateMachine,
    StateTransition,
    StateHandler,
)
from .engine import (
    ConversationEngine,
    EngineConfig,
    ConversationSession,
    TurnResult,
)

__all__ = [
    # Base
    "ConversationState",
    "ConversationEvent",
    "ConversationEventType",
    "TurnState",
    "ResponseType",
    "ConversationConfig",
    # Context
    "ConversationContext",
    "ContextVariable",
    "ContextScope",
    "ContextManager",
    # Intents
    "Intent",
    "IntentMatch",
    "IntentHandler",
    "IntentRouter",
    "StaticIntentMatcher",
    "LLMIntentMatcher",
    # Slots
    "Slot",
    "SlotType",
    "SlotValue",
    "SlotFiller",
    "SlotValidationResult",
    # Flows
    "DialogNode",
    "DialogFlow",
    "FlowTransition",
    "FlowCondition",
    "FlowAction",
    "FlowBuilder",
    # State
    "StateMachine",
    "StateTransition",
    "StateHandler",
    # Engine
    "ConversationEngine",
    "EngineConfig",
    "ConversationSession",
    "TurnResult",
]
