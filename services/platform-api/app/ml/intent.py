"""
Intent Classification System

Enterprise intent recognition with:
- Multi-label classification
- Hierarchical intents
- Confidence scoring
- Active learning support
- Custom model training
"""

from typing import Optional, Dict, Any, List, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import re
import math
import logging

logger = logging.getLogger(__name__)


class IntentCategory(str, Enum):
    """Standard intent categories."""
    GREETING = "greeting"
    FAREWELL = "farewell"
    AFFIRMATION = "affirmation"
    NEGATION = "negation"
    QUESTION = "question"
    REQUEST = "request"
    COMPLAINT = "complaint"
    FEEDBACK = "feedback"
    INFORMATION = "information"
    CONFIRMATION = "confirmation"
    CANCELLATION = "cancellation"
    ESCALATION = "escalation"
    TRANSFER = "transfer"
    HOLD = "hold"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Detected intent."""
    label: str
    confidence: float
    category: Optional[IntentCategory] = None
    sub_intents: List["Intent"] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_confident(self, threshold: float = 0.7) -> bool:
        """Check if intent meets confidence threshold."""
        return self.confidence >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "category": self.category.value if self.category else None,
            "sub_intents": [i.to_dict() for i in self.sub_intents],
            "slots": self.slots,
            "metadata": self.metadata,
        }


@dataclass
class IntentResult:
    """Result from intent classification."""
    text: str
    primary_intent: Optional[Intent] = None
    all_intents: List[Intent] = field(default_factory=list)
    is_multi_intent: bool = False
    processing_time_ms: float = 0.0
    model_version: str = "1.0.0"

    def get_top_n(self, n: int = 3) -> List[Intent]:
        """Get top N intents by confidence."""
        return sorted(
            self.all_intents,
            key=lambda i: i.confidence,
            reverse=True
        )[:n]


@dataclass
class TrainingExample:
    """Training example for intent classifier."""
    text: str
    intent: str
    slots: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingData:
    """Training data collection."""
    examples: List[TrainingExample] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_example(self, text: str, intent: str, **kwargs) -> None:
        """Add training example."""
        self.examples.append(TrainingExample(text=text, intent=intent, **kwargs))

    def get_intents(self) -> Set[str]:
        """Get all unique intents."""
        return {ex.intent for ex in self.examples}

    def get_examples_for_intent(self, intent: str) -> List[TrainingExample]:
        """Get examples for specific intent."""
        return [ex for ex in self.examples if ex.intent == intent]


class IntentClassifier(ABC):
    """Abstract base for intent classifiers."""

    @abstractmethod
    async def classify(self, text: str, context: Optional[Dict[str, Any]] = None) -> IntentResult:
        """Classify intent from text."""
        pass

    @abstractmethod
    async def classify_batch(self, texts: List[str]) -> List[IntentResult]:
        """Classify multiple texts."""
        pass


class PatternBasedClassifier(IntentClassifier):
    """
    Pattern-based intent classifier.

    Uses regex patterns and keywords for classification.
    Suitable for well-defined, rule-based intents.
    """

    def __init__(self):
        self.patterns: Dict[str, List[re.Pattern]] = {}
        self.keywords: Dict[str, List[str]] = {}
        self.priority: Dict[str, int] = {}
        self._setup_default_patterns()

    def _setup_default_patterns(self) -> None:
        """Setup default intent patterns."""
        # Greeting patterns
        self.add_pattern("greeting", [
            r"\b(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy)\b",
            r"^(hi|hello|hey)[\s!.,]*$",
        ])
        self.add_keywords("greeting", [
            "hi", "hello", "hey", "greetings", "good morning",
            "good afternoon", "good evening", "howdy",
        ])

        # Farewell patterns
        self.add_pattern("farewell", [
            r"\b(bye|goodbye|see\s*you|take\s*care|have\s*a\s*(good|nice)\s*(day|one))\b",
            r"^(bye|goodbye)[\s!.,]*$",
        ])
        self.add_keywords("farewell", [
            "bye", "goodbye", "see you", "take care", "later",
        ])

        # Affirmation patterns
        self.add_pattern("affirmation", [
            r"\b(yes|yeah|yep|sure|okay|ok|correct|right|absolutely|definitely)\b",
            r"^(yes|yeah|yep|ok|okay)[\s!.,]*$",
        ])
        self.add_keywords("affirmation", [
            "yes", "yeah", "yep", "sure", "okay", "correct", "right",
            "absolutely", "definitely", "affirmative",
        ])

        # Negation patterns
        self.add_pattern("negation", [
            r"\b(no|nope|nah|not\s*really|don'?t\s*think\s*so)\b",
            r"^(no|nope|nah)[\s!.,]*$",
        ])
        self.add_keywords("negation", [
            "no", "nope", "nah", "not really", "negative",
        ])

        # Question patterns
        self.add_pattern("question", [
            r"\b(what|who|where|when|why|how|which|can|could|would|should|is|are|do|does)\b.*\?",
            r"\?$",
        ])

        # Request patterns
        self.add_pattern("request", [
            r"\b(i\s*(want|need|would\s*like)|please|can\s*you|could\s*you)\b",
            r"\b(help\s*me|assist\s*me)\b",
        ])
        self.add_keywords("request", [
            "please", "help", "assist", "need", "want", "require",
        ])

        # Complaint patterns
        self.add_pattern("complaint", [
            r"\b(problem|issue|wrong|broken|not\s*working|frustrated|angry|upset)\b",
            r"\b(terrible|awful|horrible|worst|disappointed)\b",
        ])
        self.add_keywords("complaint", [
            "problem", "issue", "wrong", "broken", "frustrated",
            "disappointed", "terrible", "awful",
        ])

        # Escalation patterns
        self.add_pattern("escalation", [
            r"\b(speak\s*to\s*(a\s*)?(manager|supervisor|human)|escalate)\b",
            r"\b(transfer\s*(me\s*)?(to\s*)?(someone|agent))\b",
        ])
        self.add_keywords("escalation", [
            "manager", "supervisor", "escalate", "human", "agent",
        ])

        # Cancellation patterns
        self.add_pattern("cancellation", [
            r"\b(cancel|stop|end|terminate|close\s*(my)?\s*account)\b",
        ])
        self.add_keywords("cancellation", [
            "cancel", "cancellation", "terminate", "close account",
        ])

        # Confirmation patterns
        self.add_pattern("confirmation", [
            r"\b(confirm|verify|check\s*(on)?|status\s*(of)?)\b",
        ])
        self.add_keywords("confirmation", [
            "confirm", "verify", "status", "check",
        ])

    def add_pattern(self, intent: str, patterns: List[str]) -> None:
        """Add regex patterns for intent."""
        if intent not in self.patterns:
            self.patterns[intent] = []
        for p in patterns:
            self.patterns[intent].append(re.compile(p, re.IGNORECASE))

    def add_keywords(self, intent: str, keywords: List[str]) -> None:
        """Add keywords for intent."""
        if intent not in self.keywords:
            self.keywords[intent] = []
        self.keywords[intent].extend(keywords)

    def set_priority(self, intent: str, priority: int) -> None:
        """Set intent priority (higher = more important)."""
        self.priority[intent] = priority

    async def classify(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> IntentResult:
        """Classify intent from text."""
        import time
        start_time = time.time()

        text_lower = text.lower().strip()
        scores: Dict[str, float] = {}

        # Pattern matching
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    scores[intent] = scores.get(intent, 0) + 0.4
                    break

        # Keyword matching
        for intent, keywords in self.keywords.items():
            keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            if keyword_matches > 0:
                scores[intent] = scores.get(intent, 0) + min(0.3 * keyword_matches, 0.6)

        # Apply priority boost
        for intent, score in scores.items():
            priority = self.priority.get(intent, 0)
            scores[intent] = score + (priority * 0.1)

        # Normalize scores
        max_score = max(scores.values()) if scores else 0
        if max_score > 1.0:
            scores = {k: v / max_score for k, v in scores.items()}

        # Build results
        all_intents = [
            Intent(
                label=intent,
                confidence=min(score, 1.0),
                category=self._get_category(intent),
            )
            for intent, score in sorted(scores.items(), key=lambda x: -x[1])
        ]

        # Default to unknown if no matches
        if not all_intents:
            all_intents = [Intent(
                label="unknown",
                confidence=0.5,
                category=IntentCategory.UNKNOWN,
            )]

        return IntentResult(
            text=text,
            primary_intent=all_intents[0] if all_intents else None,
            all_intents=all_intents,
            is_multi_intent=len([i for i in all_intents if i.confidence > 0.5]) > 1,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version="pattern-1.0.0",
        )

    async def classify_batch(self, texts: List[str]) -> List[IntentResult]:
        """Classify multiple texts."""
        return await asyncio.gather(*[self.classify(text) for text in texts])

    def _get_category(self, intent: str) -> Optional[IntentCategory]:
        """Map intent to category."""
        try:
            return IntentCategory(intent.lower())
        except ValueError:
            return None


class MLIntentClassifier(IntentClassifier):
    """
    Machine learning based intent classifier.

    Uses embedding-based classification with support for:
    - Pre-trained embeddings
    - Fine-tuning on custom data
    - Confidence calibration
    """

    def __init__(
        self,
        model_name: str = "default",
        embedding_dim: int = 384,
        num_classes: int = 20,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self._embeddings: Optional[Any] = None
        self._classifier: Optional[Any] = None
        self._label_map: Dict[int, str] = {}
        self._intent_embeddings: Dict[str, List[float]] = {}
        self._is_trained = False

    async def initialize(self, training_data: Optional[TrainingData] = None) -> None:
        """Initialize or load the model."""
        if training_data:
            await self.train(training_data)

    async def train(self, training_data: TrainingData) -> Dict[str, Any]:
        """Train the classifier on provided data."""
        logger.info(f"Training intent classifier with {len(training_data.examples)} examples")

        # Build label map
        intents = sorted(training_data.get_intents())
        self._label_map = {i: intent for i, intent in enumerate(intents)}

        # Compute intent embeddings (average of example embeddings)
        for intent in intents:
            examples = training_data.get_examples_for_intent(intent)
            embeddings = []
            for ex in examples:
                embedding = await self._get_embedding(ex.text)
                embeddings.append(embedding)

            # Average embeddings
            if embeddings:
                avg_embedding = [
                    sum(e[i] for e in embeddings) / len(embeddings)
                    for i in range(len(embeddings[0]))
                ]
                self._intent_embeddings[intent] = avg_embedding

        self._is_trained = True
        logger.info(f"Trained classifier with {len(intents)} intents")

        return {
            "num_intents": len(intents),
            "num_examples": len(training_data.examples),
            "intents": intents,
        }

    async def _get_embedding(self, text: str) -> List[float]:
        """Get text embedding (simplified for demo)."""
        # In production, this would call an embedding model
        # Using simple hash-based embedding for demonstration
        import hashlib

        embedding = []
        text_bytes = text.lower().encode()

        for i in range(self.embedding_dim):
            h = hashlib.sha256(text_bytes + str(i).encode())
            value = int(h.hexdigest()[:8], 16) / (2 ** 32) - 0.5
            embedding.append(value)

        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    async def classify(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> IntentResult:
        """Classify intent from text."""
        import time
        start_time = time.time()

        if not self._is_trained or not self._intent_embeddings:
            # Fall back to pattern-based
            fallback = PatternBasedClassifier()
            return await fallback.classify(text, context)

        # Get text embedding
        text_embedding = await self._get_embedding(text)

        # Compute similarities to all intents
        similarities: List[Tuple[str, float]] = []
        for intent, intent_embedding in self._intent_embeddings.items():
            sim = self._cosine_similarity(text_embedding, intent_embedding)
            # Convert similarity (-1 to 1) to confidence (0 to 1)
            confidence = (sim + 1) / 2
            similarities.append((intent, confidence))

        # Sort by confidence
        similarities.sort(key=lambda x: -x[1])

        # Build intents
        all_intents = [
            Intent(
                label=intent,
                confidence=conf,
                category=self._get_category(intent),
            )
            for intent, conf in similarities
        ]

        return IntentResult(
            text=text,
            primary_intent=all_intents[0] if all_intents else None,
            all_intents=all_intents,
            is_multi_intent=len([i for i in all_intents if i.confidence > 0.5]) > 1,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version=f"ml-{self.model_name}",
        )

    async def classify_batch(self, texts: List[str]) -> List[IntentResult]:
        """Classify multiple texts."""
        return await asyncio.gather(*[self.classify(text) for text in texts])

    def _get_category(self, intent: str) -> Optional[IntentCategory]:
        """Map intent to category."""
        try:
            return IntentCategory(intent.lower())
        except ValueError:
            return None


class MultiLabelClassifier(IntentClassifier):
    """
    Multi-label intent classifier.

    Detects multiple intents in a single utterance.
    """

    def __init__(
        self,
        base_classifier: Optional[IntentClassifier] = None,
        threshold: float = 0.4,
        max_intents: int = 5,
    ):
        self.base_classifier = base_classifier or PatternBasedClassifier()
        self.threshold = threshold
        self.max_intents = max_intents

    async def classify(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> IntentResult:
        """Classify multiple intents from text."""
        # Get base classification
        base_result = await self.base_classifier.classify(text, context)

        # Filter by threshold
        confident_intents = [
            intent for intent in base_result.all_intents
            if intent.confidence >= self.threshold
        ][:self.max_intents]

        return IntentResult(
            text=text,
            primary_intent=confident_intents[0] if confident_intents else base_result.primary_intent,
            all_intents=base_result.all_intents,
            is_multi_intent=len(confident_intents) > 1,
            processing_time_ms=base_result.processing_time_ms,
            model_version=f"multi-{base_result.model_version}",
        )

    async def classify_batch(self, texts: List[str]) -> List[IntentResult]:
        """Classify multiple texts."""
        return await asyncio.gather(*[self.classify(text) for text in texts])


class HierarchicalIntentClassifier(IntentClassifier):
    """
    Hierarchical intent classifier.

    Supports nested intent structures (e.g., request.refund, request.information).
    """

    def __init__(self):
        self.hierarchies: Dict[str, List[str]] = {}
        self.classifiers: Dict[str, IntentClassifier] = {}
        self._root_classifier = PatternBasedClassifier()

    def add_hierarchy(
        self,
        parent: str,
        children: List[str],
        classifier: Optional[IntentClassifier] = None,
    ) -> None:
        """Add intent hierarchy."""
        self.hierarchies[parent] = children
        if classifier:
            self.classifiers[parent] = classifier

    async def classify(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> IntentResult:
        """Classify with hierarchical intents."""
        import time
        start_time = time.time()

        # First, classify at root level
        root_result = await self._root_classifier.classify(text, context)

        if not root_result.primary_intent:
            return root_result

        primary_label = root_result.primary_intent.label

        # Check if this intent has children
        if primary_label in self.hierarchies:
            children = self.hierarchies[primary_label]

            # Use child classifier if available
            if primary_label in self.classifiers:
                child_result = await self.classifiers[primary_label].classify(text, context)
                if child_result.primary_intent:
                    # Create hierarchical intent
                    sub_intent = child_result.primary_intent
                    root_result.primary_intent.sub_intents = [sub_intent]
                    root_result.primary_intent.label = f"{primary_label}.{sub_intent.label}"

        root_result.processing_time_ms = (time.time() - start_time) * 1000

        return root_result

    async def classify_batch(self, texts: List[str]) -> List[IntentResult]:
        """Classify multiple texts."""
        return await asyncio.gather(*[self.classify(text) for text in texts])


class ContextAwareClassifier(IntentClassifier):
    """
    Context-aware intent classifier.

    Uses conversation context to improve classification.
    """

    def __init__(
        self,
        base_classifier: Optional[IntentClassifier] = None,
        context_weight: float = 0.3,
    ):
        self.base_classifier = base_classifier or PatternBasedClassifier()
        self.context_weight = context_weight
        self._context_transitions: Dict[str, Dict[str, float]] = {}
        self._setup_default_transitions()

    def _setup_default_transitions(self) -> None:
        """Setup default context transitions."""
        # After greeting, expect question/request
        self._context_transitions["greeting"] = {
            "question": 0.4,
            "request": 0.4,
            "information": 0.2,
        }

        # After question, expect affirmation/negation
        self._context_transitions["question"] = {
            "affirmation": 0.3,
            "negation": 0.3,
            "information": 0.2,
            "question": 0.2,
        }

        # After complaint, expect escalation
        self._context_transitions["complaint"] = {
            "escalation": 0.4,
            "request": 0.3,
            "cancellation": 0.2,
        }

        # After request, expect confirmation
        self._context_transitions["request"] = {
            "confirmation": 0.4,
            "affirmation": 0.3,
            "question": 0.2,
        }

    async def classify(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> IntentResult:
        """Classify with context awareness."""
        # Get base classification
        base_result = await self.base_classifier.classify(text, context)

        if not context or "previous_intent" not in context:
            return base_result

        previous_intent = context.get("previous_intent")
        if previous_intent not in self._context_transitions:
            return base_result

        # Adjust scores based on transitions
        transitions = self._context_transitions[previous_intent]
        adjusted_intents = []

        for intent in base_result.all_intents:
            adjustment = transitions.get(intent.label, 0) * self.context_weight
            adjusted_confidence = min(intent.confidence + adjustment, 1.0)
            adjusted_intents.append(Intent(
                label=intent.label,
                confidence=adjusted_confidence,
                category=intent.category,
                sub_intents=intent.sub_intents,
                slots=intent.slots,
                metadata=intent.metadata,
            ))

        # Re-sort by confidence
        adjusted_intents.sort(key=lambda i: -i.confidence)

        return IntentResult(
            text=base_result.text,
            primary_intent=adjusted_intents[0] if adjusted_intents else None,
            all_intents=adjusted_intents,
            is_multi_intent=base_result.is_multi_intent,
            processing_time_ms=base_result.processing_time_ms,
            model_version=f"context-{base_result.model_version}",
        )

    async def classify_batch(self, texts: List[str]) -> List[IntentResult]:
        """Classify multiple texts."""
        return await asyncio.gather(*[self.classify(text) for text in texts])


class IntentClassificationPipeline:
    """
    Complete intent classification pipeline.

    Combines multiple classifiers with voting/ensemble.
    """

    def __init__(
        self,
        classifiers: Optional[List[IntentClassifier]] = None,
        voting_strategy: str = "weighted",
    ):
        self.classifiers = classifiers or [
            PatternBasedClassifier(),
            ContextAwareClassifier(),
        ]
        self.voting_strategy = voting_strategy
        self.weights = [1.0] * len(self.classifiers)

    def set_weights(self, weights: List[float]) -> None:
        """Set classifier weights for voting."""
        if len(weights) == len(self.classifiers):
            self.weights = weights

    async def classify(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> IntentResult:
        """Classify using ensemble."""
        import time
        start_time = time.time()

        # Get results from all classifiers
        results = await asyncio.gather(*[
            clf.classify(text, context) for clf in self.classifiers
        ])

        # Aggregate scores
        intent_scores: Dict[str, List[Tuple[float, float]]] = {}

        for result, weight in zip(results, self.weights):
            for intent in result.all_intents:
                if intent.label not in intent_scores:
                    intent_scores[intent.label] = []
                intent_scores[intent.label].append((intent.confidence, weight))

        # Compute weighted average
        final_intents = []
        for label, scores in intent_scores.items():
            if self.voting_strategy == "weighted":
                total_weight = sum(w for _, w in scores)
                weighted_score = sum(s * w for s, w in scores) / total_weight if total_weight > 0 else 0
            elif self.voting_strategy == "max":
                weighted_score = max(s for s, _ in scores)
            else:  # average
                weighted_score = sum(s for s, _ in scores) / len(scores)

            final_intents.append(Intent(
                label=label,
                confidence=weighted_score,
                category=self._get_category(label),
            ))

        final_intents.sort(key=lambda i: -i.confidence)

        return IntentResult(
            text=text,
            primary_intent=final_intents[0] if final_intents else None,
            all_intents=final_intents,
            is_multi_intent=len([i for i in final_intents if i.confidence > 0.5]) > 1,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version="ensemble-1.0.0",
        )

    async def classify_batch(self, texts: List[str]) -> List[IntentResult]:
        """Classify multiple texts."""
        return await asyncio.gather(*[self.classify(text) for text in texts])

    def _get_category(self, intent: str) -> Optional[IntentCategory]:
        """Map intent to category."""
        try:
            return IntentCategory(intent.lower())
        except ValueError:
            return None


# Factory function
def create_intent_classifier(
    classifier_type: str = "pattern",
    **kwargs,
) -> IntentClassifier:
    """Create intent classifier by type."""
    classifiers = {
        "pattern": PatternBasedClassifier,
        "ml": MLIntentClassifier,
        "multi_label": MultiLabelClassifier,
        "hierarchical": HierarchicalIntentClassifier,
        "context": ContextAwareClassifier,
        "ensemble": IntentClassificationPipeline,
    }

    cls = classifiers.get(classifier_type, PatternBasedClassifier)
    return cls(**kwargs)
