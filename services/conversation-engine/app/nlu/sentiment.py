"""Sentiment analysis for NLU."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class Sentiment(str, Enum):
    """Sentiment labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class SentimentScore:
    """Sentiment analysis result."""
    sentiment: Sentiment
    positive_score: float
    negative_score: float
    neutral_score: float
    confidence: float
    aspects: Dict[str, Sentiment] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sentiment": self.sentiment.value,
            "positive_score": self.positive_score,
            "negative_score": self.negative_score,
            "neutral_score": self.neutral_score,
            "confidence": self.confidence,
            "aspects": {k: v.value for k, v in self.aspects.items()},
        }


class SentimentAnalyzer:
    """Base class for sentiment analyzers."""

    async def analyze(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SentimentScore:
        """Analyze sentiment of text."""
        raise NotImplementedError


class LexiconSentimentAnalyzer(SentimentAnalyzer):
    """
    Lexicon-based sentiment analyzer.

    Uses word lists and rules to determine sentiment.
    """

    # Positive words
    POSITIVE_WORDS = {
        # Strong positive
        "excellent": 2.0, "amazing": 2.0, "wonderful": 2.0, "fantastic": 2.0,
        "outstanding": 2.0, "perfect": 2.0, "brilliant": 2.0, "exceptional": 2.0,
        "superb": 2.0, "incredible": 2.0, "magnificent": 2.0, "marvelous": 2.0,
        # Moderate positive
        "good": 1.0, "great": 1.5, "nice": 1.0, "happy": 1.0, "love": 1.5,
        "like": 0.8, "appreciate": 1.0, "pleased": 1.0, "satisfied": 1.0,
        "delighted": 1.5, "enjoy": 1.0, "helpful": 1.0, "friendly": 1.0,
        "excellent": 1.5, "awesome": 1.5, "best": 1.5, "beautiful": 1.0,
        # Mild positive
        "okay": 0.3, "ok": 0.3, "fine": 0.3, "decent": 0.5, "alright": 0.3,
        "thanks": 0.5, "thank": 0.5, "grateful": 1.0, "glad": 0.8,
    }

    # Negative words
    NEGATIVE_WORDS = {
        # Strong negative
        "terrible": -2.0, "horrible": -2.0, "awful": -2.0, "worst": -2.0,
        "disgusting": -2.0, "dreadful": -2.0, "atrocious": -2.0, "abysmal": -2.0,
        # Moderate negative
        "bad": -1.0, "poor": -1.0, "hate": -1.5, "dislike": -1.0, "angry": -1.0,
        "frustrated": -1.0, "annoyed": -1.0, "disappointed": -1.0,
        "unhappy": -1.0, "upset": -1.0, "wrong": -0.8, "broken": -1.0,
        "slow": -0.8, "rude": -1.5, "incompetent": -1.5, "useless": -1.5,
        "waste": -1.0, "problem": -0.8, "issue": -0.5, "error": -0.8,
        # Mild negative
        "not good": -0.8, "could be better": -0.5, "mediocre": -0.5,
    }

    # Intensifiers
    INTENSIFIERS = {
        "very": 1.5, "really": 1.5, "extremely": 2.0, "absolutely": 2.0,
        "completely": 1.8, "totally": 1.8, "quite": 1.3, "rather": 1.2,
        "so": 1.5, "pretty": 1.2, "fairly": 1.1, "highly": 1.5,
    }

    # Negators
    NEGATORS = {
        "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
        "doesn't", "don't", "didn't", "won't", "wouldn't", "couldn't",
        "shouldn't", "can't", "cannot", "isn't", "aren't", "wasn't", "weren't",
    }

    def __init__(self):
        self._positive_pattern = self._build_pattern(self.POSITIVE_WORDS.keys())
        self._negative_pattern = self._build_pattern(self.NEGATIVE_WORDS.keys())

    def _build_pattern(self, words: List[str]) -> re.Pattern:
        """Build regex pattern from word list."""
        escaped = [re.escape(w) for w in words]
        pattern = r"\b(" + "|".join(escaped) + r")\b"
        return re.compile(pattern, re.IGNORECASE)

    async def analyze(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SentimentScore:
        """Analyze sentiment using lexicon approach."""
        text_lower = text.lower()
        words = text_lower.split()

        positive_score = 0.0
        negative_score = 0.0
        word_count = len(words)

        if word_count == 0:
            return SentimentScore(
                sentiment=Sentiment.NEUTRAL,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                confidence=0.5,
            )

        # Analyze each word with context
        for i, word in enumerate(words):
            # Check for intensifier before word
            intensifier = 1.0
            if i > 0 and words[i - 1] in self.INTENSIFIERS:
                intensifier = self.INTENSIFIERS[words[i - 1]]

            # Check for negation before word
            negated = False
            for j in range(max(0, i - 3), i):
                if words[j] in self.NEGATORS:
                    negated = True
                    break

            # Get word score
            if word in self.POSITIVE_WORDS:
                score = self.POSITIVE_WORDS[word] * intensifier
                if negated:
                    negative_score += score * 0.7
                else:
                    positive_score += score

            elif word in self.NEGATIVE_WORDS:
                score = abs(self.NEGATIVE_WORDS[word]) * intensifier
                if negated:
                    positive_score += score * 0.7
                else:
                    negative_score += score

        # Normalize scores
        total_score = positive_score + negative_score
        if total_score > 0:
            positive_normalized = positive_score / (total_score + word_count * 0.1)
            negative_normalized = negative_score / (total_score + word_count * 0.1)
        else:
            positive_normalized = 0.0
            negative_normalized = 0.0

        # Determine sentiment
        diff = positive_score - negative_score
        neutral_score = max(0.0, 1.0 - positive_normalized - negative_normalized)

        if abs(diff) < 0.5:
            sentiment = Sentiment.NEUTRAL
        elif diff > 0:
            sentiment = Sentiment.POSITIVE
        else:
            sentiment = Sentiment.NEGATIVE

        # Check for mixed sentiment
        if positive_score > 1.0 and negative_score > 1.0:
            sentiment = Sentiment.MIXED

        # Calculate confidence
        confidence = min(1.0, total_score / (word_count * 0.5))
        if sentiment == Sentiment.NEUTRAL:
            confidence = min(confidence, 0.7)

        return SentimentScore(
            sentiment=sentiment,
            positive_score=min(1.0, positive_normalized),
            negative_score=min(1.0, negative_normalized),
            neutral_score=neutral_score,
            confidence=confidence,
        )


class AspectBasedSentimentAnalyzer(SentimentAnalyzer):
    """
    Aspect-based sentiment analyzer.

    Analyzes sentiment for specific aspects of the text.
    """

    ASPECTS = {
        "service": ["service", "support", "help", "assistance", "staff", "team"],
        "product": ["product", "item", "order", "purchase", "quality"],
        "price": ["price", "cost", "value", "expensive", "cheap", "affordable"],
        "delivery": ["delivery", "shipping", "arrival", "package", "courier"],
        "experience": ["experience", "overall", "general", "total"],
    }

    def __init__(self):
        self._base_analyzer = LexiconSentimentAnalyzer()

    async def analyze(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SentimentScore:
        """Analyze sentiment with aspects."""
        # Get overall sentiment
        overall = await self._base_analyzer.analyze(text, context)

        # Analyze aspects
        aspects = {}
        text_lower = text.lower()
        sentences = self._split_sentences(text)

        for aspect, keywords in self.ASPECTS.items():
            aspect_text = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(kw in sentence_lower for kw in keywords):
                    aspect_text.append(sentence)

            if aspect_text:
                aspect_result = await self._base_analyzer.analyze(
                    " ".join(aspect_text), context
                )
                aspects[aspect] = aspect_result.sentiment

        return SentimentScore(
            sentiment=overall.sentiment,
            positive_score=overall.positive_score,
            negative_score=overall.negative_score,
            neutral_score=overall.neutral_score,
            confidence=overall.confidence,
            aspects=aspects,
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


class EmotionAnalyzer(SentimentAnalyzer):
    """
    Emotion detection analyzer.

    Detects specific emotions like anger, joy, sadness, etc.
    """

    EMOTIONS = {
        "anger": [
            "angry", "furious", "outraged", "mad", "irritated", "annoyed",
            "frustrated", "infuriated", "enraged", "livid",
        ],
        "joy": [
            "happy", "joyful", "delighted", "ecstatic", "thrilled", "excited",
            "pleased", "cheerful", "elated", "overjoyed",
        ],
        "sadness": [
            "sad", "unhappy", "depressed", "miserable", "heartbroken",
            "disappointed", "devastated", "gloomy", "melancholy",
        ],
        "fear": [
            "afraid", "scared", "terrified", "anxious", "worried", "nervous",
            "panicked", "frightened", "alarmed",
        ],
        "surprise": [
            "surprised", "amazed", "astonished", "shocked", "stunned",
            "startled", "bewildered",
        ],
        "disgust": [
            "disgusted", "repulsed", "revolted", "appalled", "nauseated",
            "sickened",
        ],
    }

    def __init__(self):
        self._base_analyzer = LexiconSentimentAnalyzer()
        self._emotion_patterns = {
            emotion: re.compile(
                r"\b(" + "|".join(re.escape(w) for w in words) + r")\b",
                re.IGNORECASE,
            )
            for emotion, words in self.EMOTIONS.items()
        }

    async def analyze(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SentimentScore:
        """Analyze sentiment with emotion detection."""
        # Get base sentiment
        base_result = await self._base_analyzer.analyze(text, context)

        # Detect emotions
        emotions = {}
        for emotion, pattern in self._emotion_patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Simple scoring based on match count
                score = min(1.0, len(matches) * 0.3)
                emotions[emotion] = (
                    Sentiment.NEGATIVE if emotion in ["anger", "sadness", "fear", "disgust"]
                    else Sentiment.POSITIVE if emotion in ["joy"]
                    else Sentiment.NEUTRAL
                )

        return SentimentScore(
            sentiment=base_result.sentiment,
            positive_score=base_result.positive_score,
            negative_score=base_result.negative_score,
            neutral_score=base_result.neutral_score,
            confidence=base_result.confidence,
            aspects=emotions,
        )


# Global sentiment analyzer
_sentiment_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create the global sentiment analyzer."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = AspectBasedSentimentAnalyzer()
    return _sentiment_analyzer


def setup_sentiment_analyzer(analyzer: SentimentAnalyzer) -> None:
    """Set up the global sentiment analyzer."""
    global _sentiment_analyzer
    _sentiment_analyzer = analyzer
