"""NLU processor combining all NLU components."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import asyncio
import logging

from app.nlu.intent import (
    Intent,
    IntentClassifier,
    get_intent_classifier,
)
from app.nlu.entities import (
    Entity,
    EntityExtractor,
    get_entity_extractor,
)
from app.nlu.sentiment import (
    SentimentScore,
    SentimentAnalyzer,
    get_sentiment_analyzer,
)

logger = logging.getLogger(__name__)


@dataclass
class NLUResult:
    """Complete NLU analysis result."""
    text: str
    intent: Intent
    entities: List[Entity]
    sentiment: SentimentScore
    language: str = "en"
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "intent": self.intent.to_dict(),
            "entities": [e.to_dict() for e in self.entities],
            "sentiment": self.sentiment.to_dict(),
            "language": self.language,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }

    def get_entity(self, entity_type: str) -> Optional[Entity]:
        """Get first entity of a specific type."""
        for entity in self.entities:
            if entity.type.value == entity_type:
                return entity
        return None

    def get_entities(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities if e.type.value == entity_type]

    def get_entity_values(self, entity_type: str) -> List[Any]:
        """Get values of all entities of a specific type."""
        return [e.value for e in self.get_entities(entity_type)]


class NLUProcessor:
    """
    Main NLU processor.

    Combines intent classification, entity extraction, and sentiment analysis.

    Usage:
        processor = NLUProcessor()
        result = await processor.process("I want to schedule an appointment for tomorrow at 3pm")

        print(result.intent.name)  # "appointment"
        print(result.entities)  # [Entity(type=DATE, value="2024-01-02"), Entity(type=TIME, value="15:00")]
        print(result.sentiment.sentiment)  # "neutral"
    """

    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
    ):
        self.intent_classifier = intent_classifier or get_intent_classifier()
        self.entity_extractor = entity_extractor or get_entity_extractor()
        self.sentiment_analyzer = sentiment_analyzer or get_sentiment_analyzer()

    async def process(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> NLUResult:
        """
        Process text through all NLU components.

        Args:
            text: Input text to analyze
            context: Optional context for analysis

        Returns:
            Complete NLU result
        """
        import time
        start_time = time.perf_counter()

        # Run all analyses in parallel
        intent_task = asyncio.create_task(
            self.intent_classifier.classify(text, context)
        )
        entities_task = asyncio.create_task(
            self.entity_extractor.extract(text, context)
        )
        sentiment_task = asyncio.create_task(
            self.sentiment_analyzer.analyze(text, context)
        )

        # Wait for all tasks
        intent, entities, sentiment = await asyncio.gather(
            intent_task,
            entities_task,
            sentiment_task,
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(intent, Exception):
            logger.error(f"Intent classification failed: {intent}")
            intent = Intent(name="unknown", confidence=0.0, raw_text=text)

        if isinstance(entities, Exception):
            logger.error(f"Entity extraction failed: {entities}")
            entities = []

        if isinstance(sentiment, Exception):
            logger.error(f"Sentiment analysis failed: {sentiment}")
            from app.nlu.sentiment import Sentiment
            sentiment = SentimentScore(
                sentiment=Sentiment.NEUTRAL,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                confidence=0.0,
            )

        # Calculate overall confidence
        confidence = (
            intent.confidence * 0.5 +
            sentiment.confidence * 0.3 +
            (1.0 if entities else 0.0) * 0.2
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return NLUResult(
            text=text,
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
        )

    async def classify_intent(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Intent:
        """Classify intent only."""
        return await self.intent_classifier.classify(text, context)

    async def extract_entities(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Entity]:
        """Extract entities only."""
        return await self.entity_extractor.extract(text, context)

    async def analyze_sentiment(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SentimentScore:
        """Analyze sentiment only."""
        return await self.sentiment_analyzer.analyze(text, context)


class NLUPipeline:
    """
    Configurable NLU pipeline.

    Allows adding custom processors and transformers.

    Usage:
        pipeline = NLUPipeline()
        pipeline.add_preprocessor(lowercase_text)
        pipeline.add_postprocessor(filter_low_confidence)

        result = await pipeline.process("HELLO WORLD")
    """

    def __init__(self, processor: Optional[NLUProcessor] = None):
        self.processor = processor or NLUProcessor()
        self._preprocessors: List[callable] = []
        self._postprocessors: List[callable] = []
        self._validators: List[callable] = []

    def add_preprocessor(self, func: callable) -> None:
        """Add a text preprocessor."""
        self._preprocessors.append(func)

    def add_postprocessor(self, func: callable) -> None:
        """Add a result postprocessor."""
        self._postprocessors.append(func)

    def add_validator(self, func: callable) -> None:
        """Add a result validator."""
        self._validators.append(func)

    async def process(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> NLUResult:
        """Process text through the pipeline."""
        # Run preprocessors
        processed_text = text
        for preprocessor in self._preprocessors:
            if asyncio.iscoroutinefunction(preprocessor):
                processed_text = await preprocessor(processed_text)
            else:
                processed_text = preprocessor(processed_text)

        # Run main processor
        result = await self.processor.process(processed_text, context)

        # Run postprocessors
        for postprocessor in self._postprocessors:
            if asyncio.iscoroutinefunction(postprocessor):
                result = await postprocessor(result)
            else:
                result = postprocessor(result)

        # Run validators
        for validator in self._validators:
            if asyncio.iscoroutinefunction(validator):
                await validator(result)
            else:
                validator(result)

        return result


class ConversationNLU:
    """
    NLU processor with conversation context.

    Maintains context across multiple turns.
    """

    def __init__(self, processor: Optional[NLUProcessor] = None):
        self.processor = processor or NLUProcessor()
        self._context: Dict[str, Any] = {}
        self._history: List[NLUResult] = []
        self._max_history = 10

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value."""
        self._context[key] = value

    def get_context(self, key: str) -> Optional[Any]:
        """Get a context value."""
        return self._context.get(key)

    def clear_context(self) -> None:
        """Clear all context."""
        self._context = {}
        self._history = []

    async def process(self, text: str) -> NLUResult:
        """Process text with conversation context."""
        # Build context from history
        context = {
            **self._context,
            "previous_intents": [r.intent.name for r in self._history[-3:]],
            "previous_entities": self._get_recent_entities(),
            "turn_count": len(self._history),
        }

        result = await self.processor.process(text, context)

        # Update history
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Update context with new entities
        for entity in result.entities:
            self._context[f"last_{entity.type.value}"] = entity.value

        return result

    def _get_recent_entities(self) -> Dict[str, Any]:
        """Get entities from recent history."""
        entities = {}
        for result in self._history[-3:]:
            for entity in result.entities:
                entities[entity.type.value] = entity.value
        return entities

    def get_history(self) -> List[NLUResult]:
        """Get conversation history."""
        return self._history.copy()


# Global NLU processor
_nlu_processor: Optional[NLUProcessor] = None


def get_nlu_processor() -> NLUProcessor:
    """Get or create the global NLU processor."""
    global _nlu_processor
    if _nlu_processor is None:
        _nlu_processor = NLUProcessor()
    return _nlu_processor


def setup_nlu_processor(processor: NLUProcessor) -> None:
    """Set up the global NLU processor."""
    global _nlu_processor
    _nlu_processor = processor


# Convenience function
async def process_text(
    text: str,
    context: Optional[Dict[str, Any]] = None,
) -> NLUResult:
    """Process text using the global NLU processor."""
    return await get_nlu_processor().process(text, context)
