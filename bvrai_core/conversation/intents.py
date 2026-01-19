"""
Intent Recognition and Handling

This module provides intent matching and routing for conversation flows.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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


@dataclass
class Intent:
    """Definition of a conversation intent."""

    # Identification
    name: str
    description: str = ""

    # Matching patterns
    patterns: List[str] = field(default_factory=list)  # Regex patterns
    examples: List[str] = field(default_factory=list)  # Example utterances
    keywords: List[str] = field(default_factory=list)  # Keywords to match

    # Priority (higher = checked first)
    priority: int = 0

    # Parameters/slots associated with this intent
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)

    # Behavior
    cancels_current_flow: bool = False
    requires_confirmation: bool = False
    can_interrupt: bool = True  # Can interrupt other flows

    # Handler
    handler: Optional[str] = None  # Handler function name or flow ID

    # Metadata
    category: str = "general"
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Compile regex patterns
        self._compiled_patterns: List[Pattern] = []
        for pattern in self.patterns:
            try:
                self._compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Invalid pattern for intent {self.name}: {e}")


@dataclass
class IntentMatch:
    """Result of intent matching."""

    intent: Intent
    confidence: float
    matched_text: str = ""

    # Extracted from pattern groups
    extracted_values: Dict[str, Any] = field(default_factory=dict)

    # Matching details
    match_type: str = ""  # pattern, keyword, llm, etc.
    pattern_matched: Optional[str] = None

    @property
    def name(self) -> str:
        return self.intent.name


class IntentMatcher(ABC):
    """Abstract base class for intent matchers."""

    @abstractmethod
    async def match(
        self,
        text: str,
        context: ConversationContext,
        intents: List[Intent],
    ) -> Optional[IntentMatch]:
        """Match user text against intents."""
        pass


class StaticIntentMatcher(IntentMatcher):
    """
    Pattern-based intent matcher using regex and keywords.
    """

    def __init__(
        self,
        min_keyword_matches: int = 1,
        keyword_confidence: float = 0.6,
        pattern_confidence: float = 0.9,
    ):
        self._min_keyword_matches = min_keyword_matches
        self._keyword_confidence = keyword_confidence
        self._pattern_confidence = pattern_confidence

    async def match(
        self,
        text: str,
        context: ConversationContext,
        intents: List[Intent],
    ) -> Optional[IntentMatch]:
        """Match using patterns and keywords."""
        text_lower = text.lower().strip()

        # Sort by priority
        sorted_intents = sorted(intents, key=lambda x: x.priority, reverse=True)

        best_match: Optional[IntentMatch] = None
        best_confidence = 0.0

        for intent in sorted_intents:
            match = self._match_intent(text, text_lower, intent)
            if match and match.confidence > best_confidence:
                best_match = match
                best_confidence = match.confidence

        return best_match

    def _match_intent(
        self,
        text: str,
        text_lower: str,
        intent: Intent,
    ) -> Optional[IntentMatch]:
        """Match against a single intent."""
        # Try pattern matching first
        for i, pattern in enumerate(intent._compiled_patterns):
            match = pattern.search(text)
            if match:
                extracted = match.groupdict() if match.lastgroup else {}
                return IntentMatch(
                    intent=intent,
                    confidence=self._pattern_confidence,
                    matched_text=match.group(0),
                    extracted_values=extracted,
                    match_type="pattern",
                    pattern_matched=intent.patterns[i],
                )

        # Try keyword matching
        if intent.keywords:
            matched_keywords = sum(
                1 for kw in intent.keywords
                if kw.lower() in text_lower
            )

            if matched_keywords >= self._min_keyword_matches:
                ratio = matched_keywords / len(intent.keywords)
                confidence = self._keyword_confidence * ratio
                return IntentMatch(
                    intent=intent,
                    confidence=confidence,
                    matched_text=text,
                    match_type="keyword",
                )

        return None


class LLMIntentMatcher(IntentMatcher):
    """
    LLM-based intent matcher for complex/nuanced matching.
    """

    def __init__(
        self,
        llm_client: Any,  # LLM provider from llm module
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        fallback_matcher: Optional[IntentMatcher] = None,
    ):
        self._llm = llm_client
        self._model = model
        self._temperature = temperature
        self._fallback = fallback_matcher

    async def match(
        self,
        text: str,
        context: ConversationContext,
        intents: List[Intent],
    ) -> Optional[IntentMatch]:
        """Match using LLM."""
        if not intents:
            return None

        # Build intent descriptions for prompt
        intent_descriptions = []
        for intent in intents:
            desc = f"- {intent.name}: {intent.description}"
            if intent.examples:
                desc += f" (examples: {', '.join(intent.examples[:3])})"
            intent_descriptions.append(desc)

        # Build context
        context_info = ""
        if context.current_intent:
            context_info = f"\nCurrent conversation intent: {context.current_intent}"
        if context.slots:
            context_info += f"\nCollected information: {context.slots}"

        prompt = f"""Classify the user's message into one of these intents:

{chr(10).join(intent_descriptions)}
- unknown: The message doesn't match any intent

User message: "{text}"
{context_info}

Respond with JSON:
{{"intent": "<intent_name>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

        try:
            from ..llm.base import LLMMessage, LLMConfig

            response = await self._llm.complete(
                messages=[LLMMessage.user(prompt)],
                config=LLMConfig(
                    model=self._model,
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                ),
            )

            import json
            result = json.loads(response.content)

            intent_name = result.get("intent", "unknown")
            confidence = float(result.get("confidence", 0.5))

            if intent_name == "unknown":
                return None

            # Find matching intent
            for intent in intents:
                if intent.name == intent_name:
                    return IntentMatch(
                        intent=intent,
                        confidence=confidence,
                        matched_text=text,
                        match_type="llm",
                    )

        except Exception as e:
            logger.warning(f"LLM intent matching failed: {e}")

            # Fall back to static matching
            if self._fallback:
                return await self._fallback.match(text, context, intents)

        return None


class IntentHandler:
    """Handler for a matched intent."""

    def __init__(
        self,
        intent: Intent,
        handler_fn: Optional[Callable] = None,
        flow_id: Optional[str] = None,
    ):
        self.intent = intent
        self.handler_fn = handler_fn
        self.flow_id = flow_id

    async def handle(
        self,
        match: IntentMatch,
        context: ConversationContext,
    ) -> Dict[str, Any]:
        """Handle the matched intent."""
        result = {
            "handled": False,
            "response": None,
            "next_flow": self.flow_id,
            "slots_extracted": match.extracted_values,
        }

        if self.handler_fn:
            try:
                handler_result = self.handler_fn(match, context)
                if hasattr(handler_result, '__await__'):
                    handler_result = await handler_result
                result.update(handler_result)
                result["handled"] = True
            except Exception as e:
                logger.error(f"Intent handler error: {e}")
                result["error"] = str(e)

        return result


class IntentRouter:
    """
    Routes user input to appropriate intent handlers.
    """

    def __init__(
        self,
        matcher: Optional[IntentMatcher] = None,
        default_handler: Optional[Callable] = None,
    ):
        self._matcher = matcher or StaticIntentMatcher()
        self._intents: Dict[str, Intent] = {}
        self._handlers: Dict[str, IntentHandler] = {}
        self._default_handler = default_handler

        # Global intents that can interrupt any flow
        self._global_intents: List[str] = []

    def register(
        self,
        intent: Intent,
        handler_fn: Optional[Callable] = None,
        flow_id: Optional[str] = None,
        is_global: bool = False,
    ) -> None:
        """Register an intent with its handler."""
        self._intents[intent.name] = intent
        self._handlers[intent.name] = IntentHandler(
            intent=intent,
            handler_fn=handler_fn,
            flow_id=flow_id,
        )

        if is_global:
            self._global_intents.append(intent.name)

    def unregister(self, intent_name: str) -> None:
        """Unregister an intent."""
        if intent_name in self._intents:
            del self._intents[intent_name]
        if intent_name in self._handlers:
            del self._handlers[intent_name]
        if intent_name in self._global_intents:
            self._global_intents.remove(intent_name)

    async def route(
        self,
        text: str,
        context: ConversationContext,
        allowed_intents: Optional[List[str]] = None,
    ) -> Tuple[Optional[IntentMatch], Optional[Dict[str, Any]]]:
        """
        Route user input to the appropriate handler.

        Args:
            text: User input text
            context: Conversation context
            allowed_intents: Optional list of allowed intent names

        Returns:
            Tuple of (IntentMatch, handler result)
        """
        # Get intents to match against
        intents_to_check = list(self._intents.values())

        if allowed_intents:
            # Include global intents + allowed intents
            allowed_set = set(allowed_intents) | set(self._global_intents)
            intents_to_check = [
                i for i in intents_to_check
                if i.name in allowed_set
            ]

        # Match intent
        match = await self._matcher.match(text, context, intents_to_check)

        if match:
            # Update context
            context.set_intent(match.name)

            # Get handler
            handler = self._handlers.get(match.name)
            if handler:
                result = await handler.handle(match, context)
                return match, result

        # No match - use default handler if available
        if self._default_handler:
            try:
                result = self._default_handler(text, context)
                if hasattr(result, '__await__'):
                    result = await result
                return None, {"handled": True, **result}
            except Exception as e:
                logger.error(f"Default handler error: {e}")

        return match, None

    def get_intent(self, name: str) -> Optional[Intent]:
        """Get an intent by name."""
        return self._intents.get(name)

    def list_intents(
        self,
        category: Optional[str] = None,
    ) -> List[Intent]:
        """List registered intents."""
        intents = list(self._intents.values())

        if category:
            intents = [i for i in intents if i.category == category]

        return intents


# Pre-defined common intents
COMMON_INTENTS = {
    "greeting": Intent(
        name="greeting",
        description="User greeting or hello",
        patterns=[
            r"^(hi|hello|hey|good\s*(morning|afternoon|evening))[\s!.,]*$",
        ],
        keywords=["hi", "hello", "hey", "greetings"],
        priority=10,
        category="general",
    ),
    "goodbye": Intent(
        name="goodbye",
        description="User wants to end conversation",
        patterns=[
            r"^(bye|goodbye|see\s*ya|talk\s*later|have\s*a\s*good\s*(day|one))[\s!.,]*$",
        ],
        keywords=["bye", "goodbye", "later"],
        priority=10,
        category="general",
        cancels_current_flow=True,
    ),
    "help": Intent(
        name="help",
        description="User needs help or assistance",
        patterns=[
            r"(help|assist|support|what\s*can\s*you\s*do)",
        ],
        keywords=["help", "assist", "support", "options"],
        priority=5,
        category="general",
    ),
    "cancel": Intent(
        name="cancel",
        description="User wants to cancel current operation",
        patterns=[
            r"^(cancel|stop|nevermind|never\s*mind|forget\s*it)[\s!.,]*$",
        ],
        keywords=["cancel", "stop", "abort", "quit"],
        priority=20,
        category="control",
        cancels_current_flow=True,
    ),
    "repeat": Intent(
        name="repeat",
        description="User wants information repeated",
        patterns=[
            r"(repeat|say\s*that\s*again|what\s*did\s*you\s*say|pardon)",
        ],
        keywords=["repeat", "again", "pardon", "what"],
        priority=5,
        category="control",
    ),
    "yes": Intent(
        name="affirmative",
        description="User confirms or agrees",
        patterns=[
            r"^(yes|yeah|yep|sure|correct|right|ok|okay|that's right|exactly)[\s!.,]*$",
        ],
        keywords=["yes", "yeah", "correct", "right", "ok"],
        priority=5,
        category="confirmation",
    ),
    "no": Intent(
        name="negative",
        description="User declines or disagrees",
        patterns=[
            r"^(no|nope|not|don't|wrong|incorrect)[\s!.,]*$",
        ],
        keywords=["no", "nope", "wrong", "incorrect"],
        priority=5,
        category="confirmation",
    ),
    "transfer": Intent(
        name="transfer_request",
        description="User wants to speak to a human agent",
        patterns=[
            r"(speak|talk)\s*(to|with)\s*(a|an)?\s*(human|agent|person|representative)",
            r"transfer\s*(me)?",
        ],
        keywords=["human", "agent", "representative", "transfer", "person"],
        priority=15,
        category="control",
        cancels_current_flow=True,
    ),
}


def get_common_intents() -> Dict[str, Intent]:
    """Get dictionary of common intents."""
    return COMMON_INTENTS.copy()


__all__ = [
    "Intent",
    "IntentMatch",
    "IntentMatcher",
    "StaticIntentMatcher",
    "LLMIntentMatcher",
    "IntentHandler",
    "IntentRouter",
    "COMMON_INTENTS",
    "get_common_intents",
]
