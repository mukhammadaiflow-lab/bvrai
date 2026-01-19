"""
Conversation Engine

AI-powered conversation management with:
- Dialogue management
- Turn-taking
- Context management
- Intent handling
- Response generation
"""

from app.conversation.engine import (
    ConversationEngine,
    ConversationConfig,
    Conversation,
    ConversationState,
    ConversationManager,
)

from app.conversation.dialogue import (
    DialogueManager,
    DialogueState,
    DialogueContext,
    DialogueTurn,
    TurnType,
    DialoguePolicy,
)

from app.conversation.context import (
    ContextManager,
    ConversationContext,
    ContextSlot,
    SlotType,
    ContextFrame,
    ContextStack,
)

from app.conversation.response import (
    ResponseGenerator,
    ResponseTemplate,
    ResponseType,
    GeneratedResponse,
    TemplateEngine,
)

from app.conversation.handler import (
    ConversationHandler,
    IntentHandler,
    ActionHandler,
    FallbackHandler,
    HandlerRegistry,
)

__all__ = [
    # Engine
    "ConversationEngine",
    "ConversationConfig",
    "Conversation",
    "ConversationState",
    "ConversationManager",
    # Dialogue
    "DialogueManager",
    "DialogueState",
    "DialogueContext",
    "DialogueTurn",
    "TurnType",
    "DialoguePolicy",
    # Context
    "ContextManager",
    "ConversationContext",
    "ContextSlot",
    "SlotType",
    "ContextFrame",
    "ContextStack",
    # Response
    "ResponseGenerator",
    "ResponseTemplate",
    "ResponseType",
    "GeneratedResponse",
    "TemplateEngine",
    # Handler
    "ConversationHandler",
    "IntentHandler",
    "ActionHandler",
    "FallbackHandler",
    "HandlerRegistry",
]
