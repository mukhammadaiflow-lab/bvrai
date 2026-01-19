"""Intent classification for NLU."""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """Detected intent."""
    name: str
    confidence: float
    slots: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "confidence": self.confidence,
            "slots": self.slots,
            "raw_text": self.raw_text,
        }


@dataclass
class IntentConfig:
    """Configuration for intent classifier."""
    patterns: Dict[str, List[str]] = field(default_factory=dict)
    keywords: Dict[str, List[str]] = field(default_factory=dict)
    threshold: float = 0.5
    fallback_intent: str = "unknown"


class IntentClassifier:
    """Base class for intent classifiers."""

    async def classify(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Intent:
        """Classify intent from text."""
        raise NotImplementedError


class RuleBasedIntentClassifier(IntentClassifier):
    """
    Rule-based intent classifier using patterns and keywords.

    Usage:
        classifier = RuleBasedIntentClassifier()
        classifier.add_pattern("greeting", r"(hello|hi|hey)")
        classifier.add_pattern("goodbye", r"(bye|goodbye|see you)")

        intent = await classifier.classify("Hello, how are you?")
    """

    def __init__(self, config: Optional[IntentConfig] = None):
        self.config = config or IntentConfig()
        self._patterns: Dict[str, List[re.Pattern]] = {}
        self._keywords: Dict[str, List[str]] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load patterns and keywords from config."""
        for intent, patterns in self.config.patterns.items():
            for pattern in patterns:
                self.add_pattern(intent, pattern)

        for intent, keywords in self.config.keywords.items():
            for keyword in keywords:
                self.add_keyword(intent, keyword)

    def add_pattern(self, intent: str, pattern: str) -> None:
        """Add a regex pattern for an intent."""
        if intent not in self._patterns:
            self._patterns[intent] = []
        self._patterns[intent].append(re.compile(pattern, re.IGNORECASE))

    def add_keyword(self, intent: str, keyword: str) -> None:
        """Add a keyword for an intent."""
        if intent not in self._keywords:
            self._keywords[intent] = []
        self._keywords[intent].append(keyword.lower())

    async def classify(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Intent:
        """Classify intent using rules."""
        text_lower = text.lower()
        scores: Dict[str, float] = {}

        # Check patterns (higher weight)
        for intent, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    scores[intent] = scores.get(intent, 0) + 0.8
                    break

        # Check keywords (lower weight)
        for intent, keywords in self._keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[intent] = scores.get(intent, 0) + 0.4
                    break

        if not scores:
            return Intent(
                name=self.config.fallback_intent,
                confidence=0.0,
                raw_text=text,
            )

        # Get best intent
        best_intent = max(scores.items(), key=lambda x: x[1])
        confidence = min(best_intent[1], 1.0)

        if confidence < self.config.threshold:
            return Intent(
                name=self.config.fallback_intent,
                confidence=confidence,
                raw_text=text,
            )

        return Intent(
            name=best_intent[0],
            confidence=confidence,
            raw_text=text,
        )


class MLIntentClassifier(IntentClassifier):
    """
    Machine learning-based intent classifier.

    Uses external API or local model for classification.
    """

    def __init__(
        self,
        model_endpoint: Optional[str] = None,
        intents: Optional[List[str]] = None,
        threshold: float = 0.5,
    ):
        self.model_endpoint = model_endpoint
        self.intents = intents or []
        self.threshold = threshold
        self._fallback = RuleBasedIntentClassifier()

    async def classify(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Intent:
        """Classify intent using ML model."""
        if not self.model_endpoint:
            # Fall back to rule-based
            return await self._fallback.classify(text, context)

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.model_endpoint,
                    json={
                        "text": text,
                        "context": context,
                        "intents": self.intents,
                    },
                    timeout=5.0,
                )
                response.raise_for_status()

                data = response.json()
                return Intent(
                    name=data.get("intent", "unknown"),
                    confidence=data.get("confidence", 0.0),
                    slots=data.get("slots", {}),
                    raw_text=text,
                )

        except Exception as e:
            logger.warning(f"ML intent classification failed: {e}")
            return await self._fallback.classify(text, context)


class HybridIntentClassifier(IntentClassifier):
    """
    Hybrid classifier combining rule-based and ML approaches.

    Uses rules for high-confidence matches, falls back to ML.
    """

    def __init__(
        self,
        rule_classifier: Optional[RuleBasedIntentClassifier] = None,
        ml_classifier: Optional[MLIntentClassifier] = None,
        rule_threshold: float = 0.8,
    ):
        self.rule_classifier = rule_classifier or RuleBasedIntentClassifier()
        self.ml_classifier = ml_classifier or MLIntentClassifier()
        self.rule_threshold = rule_threshold

    async def classify(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Intent:
        """Classify using hybrid approach."""
        # Try rules first
        rule_intent = await self.rule_classifier.classify(text, context)

        if rule_intent.confidence >= self.rule_threshold:
            return rule_intent

        # Try ML
        ml_intent = await self.ml_classifier.classify(text, context)

        # Return higher confidence
        if ml_intent.confidence > rule_intent.confidence:
            return ml_intent
        return rule_intent


# Predefined intent patterns
COMMON_INTENTS = IntentConfig(
    patterns={
        "greeting": [
            r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b",
        ],
        "goodbye": [
            r"\b(bye|goodbye|see you|talk later|have a good|take care)\b",
        ],
        "affirmative": [
            r"^(yes|yeah|yep|sure|ok|okay|correct|right|absolutely|definitely)$",
            r"\b(yes please|that's right|sounds good|i agree)\b",
        ],
        "negative": [
            r"^(no|nope|nah|not really|negative)$",
            r"\b(no thanks|i don't think so|i disagree)\b",
        ],
        "help": [
            r"\b(help|assist|support|what can you|how do i)\b",
        ],
        "repeat": [
            r"\b(repeat|say again|what did you say|pardon|excuse me)\b",
        ],
        "cancel": [
            r"\b(cancel|stop|never mind|forget it)\b",
        ],
        "transfer": [
            r"\b(transfer|speak to|talk to|human|agent|representative|person)\b",
        ],
        "hold": [
            r"\b(hold on|wait|one moment|just a second|give me a moment)\b",
        ],
        "thanks": [
            r"\b(thanks|thank you|appreciate|grateful)\b",
        ],
    },
    keywords={
        "appointment": ["appointment", "schedule", "booking", "reservation"],
        "billing": ["bill", "payment", "invoice", "charge", "cost", "price"],
        "order": ["order", "purchase", "buy", "delivery", "shipping"],
        "complaint": ["complaint", "issue", "problem", "wrong", "broken", "not working"],
        "information": ["information", "details", "tell me", "what is", "how does"],
    },
)


# Global classifier
_intent_classifier: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create the global intent classifier."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = RuleBasedIntentClassifier(COMMON_INTENTS)
    return _intent_classifier


def setup_intent_classifier(classifier: IntentClassifier) -> None:
    """Set up the global intent classifier."""
    global _intent_classifier
    _intent_classifier = classifier
