"""
Industry Behaviors Module

This module defines industry-specific conversation behaviors,
response patterns, and interaction guidelines.
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .base import (
    IndustryType,
    ConversationPhase,
    SentimentCategory,
    BehaviorGuideline,
    ConversationPattern,
    EmergencyProtocol,
)


logger = logging.getLogger(__name__)


@dataclass
class ResponseStyle:
    """Configuration for response styling."""

    tone: str = "professional"  # professional, friendly, formal, casual
    formality_level: int = 3  # 1-5 (1=very casual, 5=very formal)
    empathy_level: int = 3  # 1-5
    urgency_responsiveness: int = 3  # 1-5

    # Language preferences
    use_contractions: bool = True
    use_technical_terms: bool = False
    explain_jargon: bool = True

    # Interaction style
    proactive_suggestions: bool = True
    confirm_understanding: bool = True
    summarize_key_points: bool = True


@dataclass
class IntentHandler:
    """Handler for specific customer intents."""

    intent: str
    description: str
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)

    # Response configuration
    initial_responses: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)

    # Actions
    collect_info: List[str] = field(default_factory=list)  # Info to collect
    offer_services: List[str] = field(default_factory=list)  # Services to offer

    # Routing
    requires_transfer: bool = False
    transfer_to: str = ""
    priority: int = 0


@dataclass
class SentimentResponse:
    """Response configuration based on sentiment."""

    sentiment: SentimentCategory
    acknowledgment_phrases: List[str] = field(default_factory=list)
    tone_adjustments: Dict[str, Any] = field(default_factory=dict)
    priority_boost: int = 0


@dataclass
class TimeBasedBehavior:
    """Behavior that varies based on time."""

    name: str
    start_time: time = time(9, 0)
    end_time: time = time(17, 0)
    days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    # Behavior during this period
    greeting: str = ""
    offer_callback: bool = False
    transfer_enabled: bool = True
    urgency_threshold: int = 3


# Sentiment responses by category
SENTIMENT_RESPONSES: Dict[SentimentCategory, SentimentResponse] = {
    SentimentCategory.FRUSTRATED: SentimentResponse(
        sentiment=SentimentCategory.FRUSTRATED,
        acknowledgment_phrases=[
            "I understand this has been frustrating for you.",
            "I can hear that you're frustrated, and I want to help.",
            "I'm sorry you've had this experience. Let me see what I can do.",
        ],
        tone_adjustments={
            "patience": "increase",
            "empathy": "increase",
            "pace": "slower",
        },
        priority_boost=1,
    ),
    SentimentCategory.ANGRY: SentimentResponse(
        sentiment=SentimentCategory.ANGRY,
        acknowledgment_phrases=[
            "I completely understand why you're upset.",
            "You have every right to be frustrated. I want to make this right.",
            "I sincerely apologize for this situation. Let me help resolve this.",
        ],
        tone_adjustments={
            "patience": "maximum",
            "empathy": "maximum",
            "defensiveness": "none",
        },
        priority_boost=2,
    ),
    SentimentCategory.ANXIOUS: SentimentResponse(
        sentiment=SentimentCategory.ANXIOUS,
        acknowledgment_phrases=[
            "I understand this situation is concerning.",
            "I can help you through this. Let's take it step by step.",
            "Let me reassure you - we'll work through this together.",
        ],
        tone_adjustments={
            "reassurance": "increase",
            "pace": "calm",
            "detail": "increase",
        },
        priority_boost=1,
    ),
    SentimentCategory.URGENT: SentimentResponse(
        sentiment=SentimentCategory.URGENT,
        acknowledgment_phrases=[
            "I understand this is urgent. Let me help you right away.",
            "I hear that this needs immediate attention.",
            "Let's address this as quickly as possible.",
        ],
        tone_adjustments={
            "pace": "efficient",
            "focus": "solution",
        },
        priority_boost=2,
    ),
    SentimentCategory.CONFUSED: SentimentResponse(
        sentiment=SentimentCategory.CONFUSED,
        acknowledgment_phrases=[
            "Let me help clarify things for you.",
            "I understand this can be confusing. Let me explain.",
            "No worries - I'm happy to walk you through this.",
        ],
        tone_adjustments={
            "clarity": "increase",
            "pace": "slower",
            "detail": "increase",
        },
        priority_boost=0,
    ),
    SentimentCategory.APPRECIATIVE: SentimentResponse(
        sentiment=SentimentCategory.APPRECIATIVE,
        acknowledgment_phrases=[
            "Thank you - I'm happy I could help!",
            "It's my pleasure to assist you.",
            "I appreciate your kind words. Is there anything else I can help with?",
        ],
        tone_adjustments={
            "warmth": "increase",
        },
        priority_boost=0,
    ),
}


class ConversationBehavior:
    """
    Manages conversation behaviors based on industry and context.

    Provides:
    - Intent detection and handling
    - Sentiment-appropriate responses
    - Phase-based conversation flow
    - Time-based behavior adjustments
    """

    def __init__(
        self,
        industry_type: IndustryType,
        response_style: Optional[ResponseStyle] = None,
    ):
        """
        Initialize conversation behavior.

        Args:
            industry_type: Industry type
            response_style: Response style configuration
        """
        self.industry_type = industry_type
        self.response_style = response_style or ResponseStyle()

        # Load industry-specific handlers
        self._intent_handlers: Dict[str, IntentHandler] = {}
        self._emergency_protocols: List[EmergencyProtocol] = []
        self._patterns: List[ConversationPattern] = []
        self._guidelines: List[BehaviorGuideline] = []

        self._load_industry_behaviors()

        # Conversation state
        self._current_phase = ConversationPhase.GREETING
        self._detected_sentiment = SentimentCategory.NEUTRAL
        self._collected_info: Dict[str, Any] = {}

    def _load_industry_behaviors(self) -> None:
        """Load behaviors for the industry."""
        from .profiles import get_industry_profile

        profile = get_industry_profile(self.industry_type)
        if profile:
            self._patterns = profile.conversation_patterns
            self._guidelines = profile.behavior_guidelines
            self._emergency_protocols = profile.emergency_protocols

            # Set response style from profile
            self.response_style.tone = profile.tone
            self.response_style.formality_level = profile.formality_level
            self.response_style.empathy_level = profile.empathy_level
            self.response_style.urgency_responsiveness = profile.urgency_sensitivity

    def detect_intent(self, text: str) -> Optional[str]:
        """
        Detect customer intent from text.

        Args:
            text: Customer message

        Returns:
            Detected intent or None
        """
        text_lower = text.lower()

        # Check patterns for matching intent
        for pattern in self._patterns:
            # Check trigger keywords
            for keyword in pattern.trigger_keywords:
                if keyword.lower() in text_lower:
                    return pattern.trigger_intents[0] if pattern.trigger_intents else pattern.id

            # Check trigger intents (as keywords)
            for intent in pattern.trigger_intents:
                intent_keywords = intent.replace("_", " ").split()
                if all(kw in text_lower for kw in intent_keywords):
                    return intent

        return None

    def detect_sentiment(self, text: str) -> SentimentCategory:
        """
        Detect customer sentiment from text.

        Args:
            text: Customer message

        Returns:
            Detected sentiment category
        """
        text_lower = text.lower()

        # Urgent indicators
        urgent_keywords = [
            "urgent", "emergency", "immediately", "right now", "asap",
            "can't wait", "need help now", "serious", "critical",
        ]
        if any(kw in text_lower for kw in urgent_keywords):
            return SentimentCategory.URGENT

        # Angry indicators
        angry_keywords = [
            "unacceptable", "ridiculous", "outrageous", "terrible",
            "worst", "hate", "furious", "demand", "sue", "lawyer",
        ]
        angry_patterns = [
            r"what the (?:hell|heck)",
            r"this is (?:bs|bull)",
            r"i(?:'m| am) (?:so )?(?:angry|mad|pissed)",
        ]
        if any(kw in text_lower for kw in angry_keywords):
            return SentimentCategory.ANGRY
        for pattern in angry_patterns:
            if re.search(pattern, text_lower):
                return SentimentCategory.ANGRY

        # Frustrated indicators
        frustrated_keywords = [
            "frustrated", "annoyed", "disappointed", "keep calling",
            "already told", "still waiting", "how many times",
            "no one", "nobody helps", "been waiting",
        ]
        if any(kw in text_lower for kw in frustrated_keywords):
            return SentimentCategory.FRUSTRATED

        # Anxious indicators
        anxious_keywords = [
            "worried", "concerned", "nervous", "scared", "afraid",
            "what if", "is it serious", "should i be",
        ]
        if any(kw in text_lower for kw in anxious_keywords):
            return SentimentCategory.ANXIOUS

        # Confused indicators
        confused_keywords = [
            "confused", "don't understand", "what does", "explain",
            "not sure", "clarify", "what do you mean", "lost",
        ]
        confused_patterns = [
            r"i(?:'m| am) not (?:sure|clear)",
            r"what (?:is|does|do) (?:that|this|it) mean",
        ]
        if any(kw in text_lower for kw in confused_keywords):
            return SentimentCategory.CONFUSED
        for pattern in confused_patterns:
            if re.search(pattern, text_lower):
                return SentimentCategory.CONFUSED

        # Appreciative indicators
        appreciative_keywords = [
            "thank you", "thanks", "appreciate", "grateful",
            "wonderful", "excellent", "great help", "so helpful",
        ]
        if any(kw in text_lower for kw in appreciative_keywords):
            return SentimentCategory.APPRECIATIVE

        # Satisfied indicators
        satisfied_keywords = [
            "sounds good", "perfect", "that works", "great",
            "yes please", "let's do that",
        ]
        if any(kw in text_lower for kw in satisfied_keywords):
            return SentimentCategory.SATISFIED

        return SentimentCategory.NEUTRAL

    def check_emergency(self, text: str) -> Optional[EmergencyProtocol]:
        """
        Check if text indicates an emergency situation.

        Args:
            text: Customer message

        Returns:
            Matching emergency protocol or None
        """
        text_lower = text.lower()

        for protocol in self._emergency_protocols:
            # Check trigger keywords
            for keyword in protocol.trigger_keywords:
                if keyword.lower() in text_lower:
                    return protocol

            # Check trigger phrases
            for phrase in protocol.trigger_phrases:
                if phrase.lower() in text_lower:
                    return protocol

        return None

    def get_sentiment_response(
        self,
        sentiment: SentimentCategory,
    ) -> SentimentResponse:
        """
        Get appropriate response configuration for sentiment.

        Args:
            sentiment: Detected sentiment

        Returns:
            Sentiment response configuration
        """
        return SENTIMENT_RESPONSES.get(
            sentiment,
            SentimentResponse(sentiment=SentimentCategory.NEUTRAL),
        )

    def get_pattern_for_intent(self, intent: str) -> Optional[ConversationPattern]:
        """
        Get conversation pattern for an intent.

        Args:
            intent: Detected intent

        Returns:
            Matching conversation pattern
        """
        for pattern in self._patterns:
            if intent in pattern.trigger_intents or intent == pattern.id:
                return pattern
        return None

    def get_guidelines_for_phase(
        self,
        phase: ConversationPhase,
    ) -> List[BehaviorGuideline]:
        """
        Get applicable guidelines for a conversation phase.

        Args:
            phase: Current conversation phase

        Returns:
            Applicable guidelines
        """
        applicable = []

        for guideline in self._guidelines:
            # Check if guideline applies to this phase
            if not guideline.applies_when:
                applicable.append(guideline)
            elif phase.value in guideline.applies_when:
                applicable.append(guideline)

        return sorted(applicable, key=lambda g: g.priority, reverse=True)

    def suggest_response(
        self,
        customer_message: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Suggest a response based on customer message.

        Args:
            customer_message: Latest customer message
            conversation_history: Previous messages

        Returns:
            Response suggestion with metadata
        """
        # Detect intent and sentiment
        intent = self.detect_intent(customer_message)
        sentiment = self.detect_sentiment(customer_message)
        self._detected_sentiment = sentiment

        # Check for emergency
        emergency = self.check_emergency(customer_message)
        if emergency:
            return {
                "type": "emergency",
                "response": emergency.acknowledgment_script,
                "immediate_action": emergency.immediate_response,
                "transfer_required": emergency.transfer_required,
                "transfer_to": emergency.transfer_to,
                "severity": emergency.severity,
            }

        # Get sentiment response
        sentiment_response = self.get_sentiment_response(sentiment)

        # Get pattern for intent
        pattern = self.get_pattern_for_intent(intent) if intent else None

        # Build suggestion
        suggestion = {
            "type": "normal",
            "detected_intent": intent,
            "detected_sentiment": sentiment.value,
            "sentiment_acknowledgment": (
                sentiment_response.acknowledgment_phrases[0]
                if sentiment_response.acknowledgment_phrases
                else None
            ),
            "tone_adjustments": sentiment_response.tone_adjustments,
            "priority_boost": sentiment_response.priority_boost,
        }

        if pattern:
            suggestion.update({
                "suggested_responses": pattern.response_templates,
                "follow_up_questions": pattern.follow_up_questions,
                "expected_outcomes": pattern.expected_outcomes,
                "phase": pattern.phase.value,
            })
            self._current_phase = pattern.phase

        # Add applicable guidelines
        guidelines = self.get_guidelines_for_phase(self._current_phase)
        if guidelines:
            suggestion["guidelines"] = [
                {
                    "guideline": g.guideline,
                    "do_examples": g.do_examples,
                    "dont_examples": g.dont_examples,
                }
                for g in guidelines[:3]  # Top 3 guidelines
            ]

        return suggestion

    def update_phase(self, new_phase: ConversationPhase) -> None:
        """Update current conversation phase."""
        self._current_phase = new_phase

    def collect_information(self, key: str, value: Any) -> None:
        """Store collected information."""
        self._collected_info[key] = value

    def get_collected_info(self) -> Dict[str, Any]:
        """Get all collected information."""
        return dict(self._collected_info)


class BehaviorEngine:
    """
    Engine for managing conversation behaviors across sessions.

    Provides:
    - Session-specific behavior management
    - Cross-session learning (placeholder)
    - Behavior analytics
    """

    def __init__(self):
        """Initialize behavior engine."""
        self._behaviors: Dict[str, ConversationBehavior] = {}
        self._session_stats: Dict[str, Dict] = {}

    def get_behavior_for_session(
        self,
        session_id: str,
        industry_type: IndustryType,
    ) -> ConversationBehavior:
        """
        Get or create behavior handler for a session.

        Args:
            session_id: Session ID
            industry_type: Industry type

        Returns:
            Conversation behavior handler
        """
        if session_id not in self._behaviors:
            self._behaviors[session_id] = ConversationBehavior(industry_type)
            self._session_stats[session_id] = {
                "created_at": datetime.utcnow(),
                "messages_processed": 0,
                "intents_detected": {},
                "sentiments_detected": {},
            }

        return self._behaviors[session_id]

    def process_message(
        self,
        session_id: str,
        industry_type: IndustryType,
        message: str,
    ) -> Dict[str, Any]:
        """
        Process a message through the behavior engine.

        Args:
            session_id: Session ID
            industry_type: Industry type
            message: Customer message

        Returns:
            Processing result with suggestions
        """
        behavior = self.get_behavior_for_session(session_id, industry_type)
        suggestion = behavior.suggest_response(message)

        # Update stats
        stats = self._session_stats[session_id]
        stats["messages_processed"] += 1

        if suggestion.get("detected_intent"):
            intent = suggestion["detected_intent"]
            stats["intents_detected"][intent] = stats["intents_detected"].get(intent, 0) + 1

        sentiment = suggestion.get("detected_sentiment", "neutral")
        stats["sentiments_detected"][sentiment] = stats["sentiments_detected"].get(sentiment, 0) + 1

        return suggestion

    def end_session(self, session_id: str) -> Optional[Dict]:
        """
        End a session and return summary.

        Args:
            session_id: Session ID

        Returns:
            Session summary or None
        """
        behavior = self._behaviors.pop(session_id, None)
        stats = self._session_stats.pop(session_id, None)

        if stats:
            stats["ended_at"] = datetime.utcnow()
            stats["collected_info"] = behavior.get_collected_info() if behavior else {}

        return stats

    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get stats for a session."""
        return self._session_stats.get(session_id)


def create_behavior_handler(
    industry_type: IndustryType,
    response_style: Optional[ResponseStyle] = None,
) -> ConversationBehavior:
    """
    Create a conversation behavior handler.

    Args:
        industry_type: Industry type
        response_style: Custom response style

    Returns:
        Configured behavior handler
    """
    return ConversationBehavior(industry_type, response_style)


def create_behavior_engine() -> BehaviorEngine:
    """Create a behavior engine instance."""
    return BehaviorEngine()


__all__ = [
    "ResponseStyle",
    "IntentHandler",
    "SentimentResponse",
    "TimeBasedBehavior",
    "SENTIMENT_RESPONSES",
    "ConversationBehavior",
    "BehaviorEngine",
    "create_behavior_handler",
    "create_behavior_engine",
]
