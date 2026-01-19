"""
Conversation Handlers

Intent and action handling:
- Intent handlers
- Action handlers
- Fallback handling
- Handler registry
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import re

logger = logging.getLogger(__name__)


class HandlerPriority(int, Enum):
    """Handler priority levels."""
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    FALLBACK = 0


class HandlerResult(str, Enum):
    """Handler execution result."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CONTINUE = "continue"
    DELEGATE = "delegate"


@dataclass
class HandlerContext:
    """Context passed to handlers."""
    conversation_id: str
    message_id: str
    user_id: Optional[str] = None
    tenant_id: str = ""

    # Input
    text: str = ""
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    # Conversation context
    slots: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    session_data: Dict[str, Any] = field(default_factory=dict)

    # State
    turn_count: int = 0
    last_intent: Optional[str] = None
    error_count: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandlerResponse:
    """Response from handler."""
    result: HandlerResult
    response_text: Optional[str] = None
    response_ssml: Optional[str] = None

    # Actions to take
    actions: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Slot updates
    slot_updates: Dict[str, Any] = field(default_factory=dict)
    variable_updates: Dict[str, Any] = field(default_factory=dict)

    # Flow control
    should_continue: bool = True
    should_end_conversation: bool = False
    delegate_to: Optional[str] = None

    # Metadata
    handler_name: str = ""
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationHandler(ABC):
    """Abstract conversation handler."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Handler name."""
        pass

    @property
    def priority(self) -> int:
        """Handler priority."""
        return HandlerPriority.NORMAL

    @abstractmethod
    async def can_handle(self, context: HandlerContext) -> bool:
        """Check if handler can handle this context."""
        pass

    @abstractmethod
    async def handle(self, context: HandlerContext) -> HandlerResponse:
        """Handle the context."""
        pass


class IntentHandler(ConversationHandler):
    """
    Handler for specific intents.

    Matches and handles conversation intents.
    """

    def __init__(
        self,
        intent_name: str,
        handler_func: Callable[[HandlerContext], Awaitable[HandlerResponse]],
        aliases: Optional[List[str]] = None,
        priority: int = HandlerPriority.NORMAL,
    ):
        self._intent_name = intent_name
        self._handler_func = handler_func
        self._aliases = set(aliases or [])
        self._aliases.add(intent_name)
        self._priority = priority

    @property
    def name(self) -> str:
        return f"intent:{self._intent_name}"

    @property
    def priority(self) -> int:
        return self._priority

    async def can_handle(self, context: HandlerContext) -> bool:
        """Check if intent matches."""
        if context.intent:
            return context.intent.lower() in {a.lower() for a in self._aliases}
        return False

    async def handle(self, context: HandlerContext) -> HandlerResponse:
        """Execute handler function."""
        try:
            return await self._handler_func(context)
        except Exception as e:
            logger.error(f"Intent handler error: {e}")
            return HandlerResponse(
                result=HandlerResult.FAILED,
                handler_name=self.name,
                metadata={"error": str(e)},
            )


class PatternHandler(ConversationHandler):
    """
    Handler for text patterns.

    Matches text using regex patterns.
    """

    def __init__(
        self,
        pattern: str,
        handler_func: Callable[[HandlerContext, re.Match], Awaitable[HandlerResponse]],
        handler_name: str = "",
        priority: int = HandlerPriority.NORMAL,
        flags: int = re.IGNORECASE,
    ):
        self._pattern = re.compile(pattern, flags)
        self._handler_func = handler_func
        self._name = handler_name or f"pattern:{pattern[:20]}"
        self._priority = priority
        self._last_match: Optional[re.Match] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    async def can_handle(self, context: HandlerContext) -> bool:
        """Check if text matches pattern."""
        if context.text:
            self._last_match = self._pattern.search(context.text)
            return self._last_match is not None
        return False

    async def handle(self, context: HandlerContext) -> HandlerResponse:
        """Execute handler with match."""
        try:
            if self._last_match:
                return await self._handler_func(context, self._last_match)
            return HandlerResponse(
                result=HandlerResult.SKIPPED,
                handler_name=self.name,
            )
        except Exception as e:
            logger.error(f"Pattern handler error: {e}")
            return HandlerResponse(
                result=HandlerResult.FAILED,
                handler_name=self.name,
                metadata={"error": str(e)},
            )


class ActionHandler(ConversationHandler):
    """
    Handler for actions/commands.

    Handles specific action types.
    """

    def __init__(
        self,
        action_type: str,
        handler_func: Callable[[HandlerContext, Dict[str, Any]], Awaitable[HandlerResponse]],
        priority: int = HandlerPriority.NORMAL,
    ):
        self._action_type = action_type
        self._handler_func = handler_func
        self._priority = priority

    @property
    def name(self) -> str:
        return f"action:{self._action_type}"

    @property
    def priority(self) -> int:
        return self._priority

    async def can_handle(self, context: HandlerContext) -> bool:
        """Check if action type matches."""
        action = context.metadata.get("action", {})
        return action.get("type") == self._action_type

    async def handle(self, context: HandlerContext) -> HandlerResponse:
        """Execute action handler."""
        try:
            action = context.metadata.get("action", {})
            return await self._handler_func(context, action)
        except Exception as e:
            logger.error(f"Action handler error: {e}")
            return HandlerResponse(
                result=HandlerResult.FAILED,
                handler_name=self.name,
                metadata={"error": str(e)},
            )


class ConditionHandler(ConversationHandler):
    """
    Handler with custom condition.

    Matches based on custom condition function.
    """

    def __init__(
        self,
        condition_func: Callable[[HandlerContext], bool],
        handler_func: Callable[[HandlerContext], Awaitable[HandlerResponse]],
        handler_name: str,
        priority: int = HandlerPriority.NORMAL,
    ):
        self._condition_func = condition_func
        self._handler_func = handler_func
        self._name = handler_name
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    async def can_handle(self, context: HandlerContext) -> bool:
        """Check custom condition."""
        try:
            return self._condition_func(context)
        except Exception:
            return False

    async def handle(self, context: HandlerContext) -> HandlerResponse:
        """Execute handler."""
        try:
            return await self._handler_func(context)
        except Exception as e:
            logger.error(f"Condition handler error: {e}")
            return HandlerResponse(
                result=HandlerResult.FAILED,
                handler_name=self.name,
                metadata={"error": str(e)},
            )


class FallbackHandler(ConversationHandler):
    """
    Fallback handler for unmatched inputs.

    Always handles when no other handler matches.
    """

    def __init__(
        self,
        response_templates: Optional[List[str]] = None,
        max_fallbacks: int = 3,
    ):
        self._templates = response_templates or [
            "I'm sorry, I didn't understand that. Could you please rephrase?",
            "I'm not sure what you mean. Could you try saying that differently?",
            "I didn't quite catch that. Let me transfer you to someone who can help.",
        ]
        self._max_fallbacks = max_fallbacks
        self._fallback_counts: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "fallback"

    @property
    def priority(self) -> int:
        return HandlerPriority.FALLBACK

    async def can_handle(self, context: HandlerContext) -> bool:
        """Always returns True (fallback)."""
        return True

    async def handle(self, context: HandlerContext) -> HandlerResponse:
        """Handle with fallback response."""
        conv_id = context.conversation_id
        count = self._fallback_counts.get(conv_id, 0)

        # Increment fallback count
        self._fallback_counts[conv_id] = count + 1

        # Select response based on count
        index = min(count, len(self._templates) - 1)
        response_text = self._templates[index]

        # Check if max fallbacks reached
        should_end = count >= self._max_fallbacks - 1

        return HandlerResponse(
            result=HandlerResult.SUCCESS,
            response_text=response_text,
            handler_name=self.name,
            should_end_conversation=should_end,
            metadata={"fallback_count": count + 1},
        )

    def reset_count(self, conversation_id: str) -> None:
        """Reset fallback count for conversation."""
        self._fallback_counts.pop(conversation_id, None)


class CompositeHandler(ConversationHandler):
    """
    Composite handler that chains multiple handlers.
    """

    def __init__(
        self,
        handlers: List[ConversationHandler],
        handler_name: str = "composite",
        stop_on_success: bool = True,
    ):
        self._handlers = handlers
        self._name = handler_name
        self._stop_on_success = stop_on_success

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return max((h.priority for h in self._handlers), default=0)

    async def can_handle(self, context: HandlerContext) -> bool:
        """Check if any handler can handle."""
        for handler in self._handlers:
            if await handler.can_handle(context):
                return True
        return False

    async def handle(self, context: HandlerContext) -> HandlerResponse:
        """Execute handlers in sequence."""
        responses = []

        for handler in self._handlers:
            if await handler.can_handle(context):
                response = await handler.handle(context)
                responses.append(response)

                if self._stop_on_success and response.result == HandlerResult.SUCCESS:
                    return response

        if responses:
            return responses[-1]

        return HandlerResponse(
            result=HandlerResult.SKIPPED,
            handler_name=self.name,
        )


class HandlerRegistry:
    """
    Registry for conversation handlers.

    Features:
    - Handler registration
    - Priority-based execution
    - Handler lookup
    """

    def __init__(self):
        self._handlers: List[ConversationHandler] = []
        self._handlers_by_name: Dict[str, ConversationHandler] = {}
        self._intent_handlers: Dict[str, IntentHandler] = {}
        self._fallback: Optional[FallbackHandler] = None

    def register(self, handler: ConversationHandler) -> None:
        """Register handler."""
        self._handlers.append(handler)
        self._handlers_by_name[handler.name] = handler

        if isinstance(handler, IntentHandler):
            self._intent_handlers[handler._intent_name] = handler

        if isinstance(handler, FallbackHandler):
            self._fallback = handler

        # Sort by priority
        self._handlers.sort(key=lambda h: h.priority, reverse=True)

    def unregister(self, handler_name: str) -> bool:
        """Unregister handler by name."""
        handler = self._handlers_by_name.pop(handler_name, None)
        if handler:
            self._handlers = [h for h in self._handlers if h.name != handler_name]
            return True
        return False

    def get_handler(self, name: str) -> Optional[ConversationHandler]:
        """Get handler by name."""
        return self._handlers_by_name.get(name)

    def register_intent(
        self,
        intent_name: str,
        handler_func: Callable[[HandlerContext], Awaitable[HandlerResponse]],
        aliases: Optional[List[str]] = None,
        priority: int = HandlerPriority.NORMAL,
    ) -> IntentHandler:
        """Register intent handler."""
        handler = IntentHandler(
            intent_name=intent_name,
            handler_func=handler_func,
            aliases=aliases,
            priority=priority,
        )
        self.register(handler)
        return handler

    def register_pattern(
        self,
        pattern: str,
        handler_func: Callable[[HandlerContext, re.Match], Awaitable[HandlerResponse]],
        name: str = "",
        priority: int = HandlerPriority.NORMAL,
    ) -> PatternHandler:
        """Register pattern handler."""
        handler = PatternHandler(
            pattern=pattern,
            handler_func=handler_func,
            handler_name=name,
            priority=priority,
        )
        self.register(handler)
        return handler

    def register_action(
        self,
        action_type: str,
        handler_func: Callable[[HandlerContext, Dict[str, Any]], Awaitable[HandlerResponse]],
        priority: int = HandlerPriority.NORMAL,
    ) -> ActionHandler:
        """Register action handler."""
        handler = ActionHandler(
            action_type=action_type,
            handler_func=handler_func,
            priority=priority,
        )
        self.register(handler)
        return handler

    def set_fallback(
        self,
        templates: Optional[List[str]] = None,
        max_fallbacks: int = 3,
    ) -> FallbackHandler:
        """Set fallback handler."""
        handler = FallbackHandler(
            response_templates=templates,
            max_fallbacks=max_fallbacks,
        )
        self.register(handler)
        return handler

    async def handle(self, context: HandlerContext) -> HandlerResponse:
        """Find and execute matching handler."""
        import time

        start_time = time.time()

        # Try handlers in priority order
        for handler in self._handlers:
            if await handler.can_handle(context):
                response = await handler.handle(context)
                response.processing_time_ms = (time.time() - start_time) * 1000

                if response.result in (HandlerResult.SUCCESS, HandlerResult.FAILED):
                    return response

                if response.result == HandlerResult.DELEGATE and response.delegate_to:
                    delegate = self._handlers_by_name.get(response.delegate_to)
                    if delegate:
                        return await delegate.handle(context)

        # Use fallback
        if self._fallback:
            response = await self._fallback.handle(context)
            response.processing_time_ms = (time.time() - start_time) * 1000
            return response

        # No handler found
        return HandlerResponse(
            result=HandlerResult.SKIPPED,
            handler_name="none",
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    def decorator(
        self,
        intent: Optional[str] = None,
        pattern: Optional[str] = None,
        priority: int = HandlerPriority.NORMAL,
    ):
        """Decorator for registering handlers."""
        def wrapper(func: Callable):
            if intent:
                self.register_intent(
                    intent_name=intent,
                    handler_func=func,
                    priority=priority,
                )
            elif pattern:
                self.register_pattern(
                    pattern=pattern,
                    handler_func=func,
                    priority=priority,
                )
            return func

        return wrapper

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_handlers": len(self._handlers),
            "intent_handlers": len(self._intent_handlers),
            "has_fallback": self._fallback is not None,
            "handlers": [
                {"name": h.name, "priority": h.priority}
                for h in self._handlers
            ],
        }


# Convenience function for creating handlers
def intent_handler(
    intent: str,
    aliases: Optional[List[str]] = None,
    priority: int = HandlerPriority.NORMAL,
):
    """Decorator for creating intent handlers."""
    def decorator(func: Callable[[HandlerContext], Awaitable[HandlerResponse]]):
        return IntentHandler(
            intent_name=intent,
            handler_func=func,
            aliases=aliases,
            priority=priority,
        )
    return decorator


def pattern_handler(
    pattern: str,
    name: str = "",
    priority: int = HandlerPriority.NORMAL,
):
    """Decorator for creating pattern handlers."""
    def decorator(func: Callable[[HandlerContext, re.Match], Awaitable[HandlerResponse]]):
        return PatternHandler(
            pattern=pattern,
            handler_func=func,
            handler_name=name,
            priority=priority,
        )
    return decorator
