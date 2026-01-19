"""
Sentiment Analysis System

Enterprise sentiment and emotion detection with:
- Multi-class sentiment classification
- Fine-grained emotion detection
- Aspect-based sentiment analysis
- Sentiment trend tracking
- Real-time processing
"""

from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import re
import math
import logging

logger = logging.getLogger(__name__)


class SentimentLabel(str, Enum):
    """Sentiment labels."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class Emotion(str, Enum):
    """Emotion categories (Ekman's basic emotions + extended)."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    FRUSTRATION = "frustration"
    CONFUSION = "confusion"
    SATISFACTION = "satisfaction"
    DISAPPOINTMENT = "disappointment"
    NEUTRAL = "neutral"


@dataclass
class Sentiment:
    """Detected sentiment."""
    label: SentimentLabel
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    magnitude: float = 0.0  # Intensity

    @property
    def is_positive(self) -> bool:
        """Check if sentiment is positive."""
        return self.label in (SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE)

    @property
    def is_negative(self) -> bool:
        """Check if sentiment is negative."""
        return self.label in (SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label": self.label.value,
            "score": self.score,
            "confidence": self.confidence,
            "magnitude": self.magnitude,
        }


@dataclass
class EmotionScore:
    """Emotion detection result."""
    emotion: Emotion
    score: float  # 0.0 to 1.0
    confidence: float
    indicators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "emotion": self.emotion.value,
            "score": self.score,
            "confidence": self.confidence,
            "indicators": self.indicators,
        }


@dataclass
class AspectSentiment:
    """Aspect-based sentiment."""
    aspect: str
    sentiment: Sentiment
    mentions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "aspect": self.aspect,
            "sentiment": self.sentiment.to_dict(),
            "mentions": self.mentions,
        }


@dataclass
class SentimentResult:
    """Complete sentiment analysis result."""
    text: str
    sentiment: Sentiment
    emotions: List[EmotionScore] = field(default_factory=list)
    aspects: List[AspectSentiment] = field(default_factory=list)
    processing_time_ms: float = 0.0
    model_version: str = "1.0.0"

    def get_primary_emotion(self) -> Optional[EmotionScore]:
        """Get highest-scoring emotion."""
        if not self.emotions:
            return None
        return max(self.emotions, key=lambda e: e.score)

    def get_emotion(self, emotion: Emotion) -> Optional[EmotionScore]:
        """Get specific emotion score."""
        for e in self.emotions:
            if e.emotion == emotion:
                return e
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "sentiment": self.sentiment.to_dict(),
            "emotions": [e.to_dict() for e in self.emotions],
            "aspects": [a.to_dict() for a in self.aspects],
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
        }


class SentimentAnalyzer(ABC):
    """Abstract base for sentiment analyzers."""

    @abstractmethod
    async def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment in text."""
        pass

    @abstractmethod
    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment in multiple texts."""
        pass


class LexiconSentimentAnalyzer(SentimentAnalyzer):
    """
    Lexicon-based sentiment analyzer.

    Uses word lists and rules for sentiment classification.
    """

    def __init__(self):
        self._positive_words: Set[str] = set()
        self._negative_words: Set[str] = set()
        self._intensifiers: Set[str] = set()
        self._negators: Set[str] = set()
        self._emotion_lexicons: Dict[Emotion, Set[str]] = {}
        self._setup_lexicons()

    def _setup_lexicons(self) -> None:
        """Setup sentiment lexicons."""
        # Positive words
        self._positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "awesome", "perfect", "love", "like", "happy", "pleased", "satisfied",
            "helpful", "best", "better", "nice", "beautiful", "brilliant",
            "superb", "outstanding", "delightful", "impressive", "positive",
            "thank", "thanks", "appreciate", "glad", "excited", "enjoy",
            "pleasant", "remarkable", "terrific", "marvelous", "fabulous",
        }

        # Negative words
        self._negative_words = {
            "bad", "terrible", "awful", "horrible", "poor", "worst", "hate",
            "dislike", "unhappy", "sad", "angry", "frustrated", "disappointed",
            "annoyed", "upset", "problem", "issue", "wrong", "broken", "fail",
            "failed", "failure", "useless", "waste", "stupid", "ridiculous",
            "pathetic", "disgusting", "unacceptable", "never", "sucks", "suck",
            "complaint", "complain", "refund", "cancel", "cancelled",
        }

        # Intensifiers
        self._intensifiers = {
            "very", "extremely", "absolutely", "completely", "totally",
            "really", "quite", "highly", "incredibly", "so", "super",
            "especially", "particularly", "exceptionally", "truly",
        }

        # Negators
        self._negators = {
            "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
            "hardly", "barely", "scarcely", "doesn't", "don't", "didn't",
            "won't", "wouldn't", "couldn't", "shouldn't", "isn't", "aren't",
            "wasn't", "weren't", "haven't", "hasn't", "hadn't", "cannot", "can't",
        }

        # Emotion lexicons
        self._emotion_lexicons = {
            Emotion.JOY: {
                "happy", "joy", "excited", "delighted", "thrilled", "pleased",
                "glad", "cheerful", "elated", "ecstatic", "wonderful", "amazing",
            },
            Emotion.SADNESS: {
                "sad", "unhappy", "depressed", "disappointed", "heartbroken",
                "miserable", "gloomy", "sorrowful", "melancholy", "down",
            },
            Emotion.ANGER: {
                "angry", "furious", "outraged", "irritated", "annoyed", "mad",
                "enraged", "livid", "infuriated", "hostile", "resentful",
            },
            Emotion.FEAR: {
                "afraid", "scared", "terrified", "frightened", "anxious",
                "worried", "nervous", "panicked", "alarmed", "concerned",
            },
            Emotion.SURPRISE: {
                "surprised", "shocked", "amazed", "astonished", "stunned",
                "startled", "unexpected", "wow", "unbelievable",
            },
            Emotion.DISGUST: {
                "disgusted", "revolted", "repulsed", "sickened", "appalled",
                "gross", "nasty", "awful", "horrible", "terrible",
            },
            Emotion.FRUSTRATION: {
                "frustrated", "annoyed", "irritated", "stuck", "blocked",
                "confused", "exasperated", "aggravated", "impatient",
            },
            Emotion.SATISFACTION: {
                "satisfied", "content", "pleased", "fulfilled", "happy",
                "gratified", "comfortable", "resolved",
            },
            Emotion.DISAPPOINTMENT: {
                "disappointed", "let down", "dissatisfied", "unhappy",
                "dismayed", "disillusioned", "discouraged",
            },
            Emotion.CONFUSION: {
                "confused", "puzzled", "perplexed", "bewildered", "lost",
                "unclear", "uncertain", "unsure", "don't understand",
            },
            Emotion.TRUST: {
                "trust", "confident", "reliable", "dependable", "faith",
                "believe", "secure", "safe",
            },
            Emotion.ANTICIPATION: {
                "excited", "eager", "looking forward", "anticipating",
                "expecting", "hopeful", "optimistic",
            },
        }

    async def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment in text."""
        import time
        start_time = time.time()

        words = text.lower().split()
        cleaned_words = [re.sub(r'[^\w\s]', '', w) for w in words]

        # Calculate sentiment score
        positive_count = 0
        negative_count = 0
        intensifier_active = False
        negation_active = False

        for i, word in enumerate(cleaned_words):
            # Check for intensifiers
            if word in self._intensifiers:
                intensifier_active = True
                continue

            # Check for negators
            if word in self._negators:
                negation_active = True
                continue

            # Calculate word sentiment
            word_score = 0
            if word in self._positive_words:
                word_score = 1
            elif word in self._negative_words:
                word_score = -1

            # Apply modifiers
            if word_score != 0:
                if intensifier_active:
                    word_score *= 1.5
                    intensifier_active = False

                if negation_active:
                    word_score *= -1
                    negation_active = False

                if word_score > 0:
                    positive_count += word_score
                else:
                    negative_count += abs(word_score)

        # Calculate final score (-1 to 1)
        total = positive_count + negative_count
        if total == 0:
            score = 0.0
        else:
            score = (positive_count - negative_count) / (total + 1)

        # Map to label
        if score >= 0.5:
            label = SentimentLabel.VERY_POSITIVE
        elif score >= 0.15:
            label = SentimentLabel.POSITIVE
        elif score <= -0.5:
            label = SentimentLabel.VERY_NEGATIVE
        elif score <= -0.15:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        # Calculate confidence based on evidence
        confidence = min(0.5 + (total * 0.1), 0.95)

        # Calculate magnitude (intensity)
        magnitude = abs(score) * (1 + math.log(total + 1) * 0.2) if total > 0 else 0

        sentiment = Sentiment(
            label=label,
            score=score,
            confidence=confidence,
            magnitude=magnitude,
        )

        # Detect emotions
        emotions = await self._detect_emotions(text)

        return SentimentResult(
            text=text,
            sentiment=sentiment,
            emotions=emotions,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version="lexicon-1.0.0",
        )

    async def _detect_emotions(self, text: str) -> List[EmotionScore]:
        """Detect emotions in text."""
        text_lower = text.lower()
        emotions = []

        for emotion, words in self._emotion_lexicons.items():
            matches = [w for w in words if w in text_lower]
            if matches:
                score = min(len(matches) * 0.3, 1.0)
                emotions.append(EmotionScore(
                    emotion=emotion,
                    score=score,
                    confidence=min(0.5 + len(matches) * 0.1, 0.9),
                    indicators=matches,
                ))

        # Sort by score
        emotions.sort(key=lambda e: -e.score)

        # Add neutral if no emotions detected
        if not emotions:
            emotions.append(EmotionScore(
                emotion=Emotion.NEUTRAL,
                score=0.5,
                confidence=0.5,
            ))

        return emotions

    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment in multiple texts."""
        return await asyncio.gather(*[self.analyze(text) for text in texts])


class VADERSentimentAnalyzer(SentimentAnalyzer):
    """
    VADER-inspired sentiment analyzer.

    Implements rule-based sentiment with:
    - Punctuation handling
    - Capitalization awareness
    - Degree modifiers
    - Contrastive conjunctions
    """

    def __init__(self):
        self._lexicon: Dict[str, float] = {}
        self._booster_dict: Dict[str, float] = {}
        self._negation_list: List[str] = []
        self._setup_lexicon()

    def _setup_lexicon(self) -> None:
        """Setup VADER-style lexicon."""
        # Simplified sentiment lexicon with valence scores
        self._lexicon = {
            # Positive
            "good": 1.9, "great": 3.1, "excellent": 3.3, "amazing": 3.0,
            "wonderful": 3.0, "fantastic": 3.0, "awesome": 3.1, "perfect": 3.0,
            "love": 3.2, "like": 1.5, "happy": 2.7, "pleased": 2.3,
            "satisfied": 2.2, "helpful": 2.1, "best": 3.2, "better": 1.9,
            "nice": 1.8, "beautiful": 2.8, "brilliant": 3.0, "superb": 3.1,
            "thank": 2.0, "thanks": 2.0, "appreciate": 2.5, "glad": 2.4,
            "excited": 2.8, "enjoy": 2.5, "pleasant": 2.2,

            # Negative
            "bad": -2.5, "terrible": -3.4, "awful": -3.1, "horrible": -3.4,
            "poor": -2.1, "worst": -3.5, "hate": -3.4, "dislike": -2.3,
            "unhappy": -2.3, "sad": -2.1, "angry": -2.8, "frustrated": -2.6,
            "disappointed": -2.4, "annoyed": -2.2, "upset": -2.4,
            "problem": -2.1, "issue": -1.8, "wrong": -2.1, "broken": -2.5,
            "fail": -2.8, "failed": -2.8, "failure": -3.0, "useless": -3.1,
            "waste": -2.3, "stupid": -2.5, "ridiculous": -2.6,
            "sucks": -3.0, "complaint": -2.0, "refund": -1.5,
        }

        # Booster words
        self._booster_dict = {
            "very": 0.293, "extremely": 0.293, "absolutely": 0.293,
            "completely": 0.293, "totally": 0.293, "really": 0.293,
            "quite": 0.293, "highly": 0.293, "incredibly": 0.293,
            "so": 0.293, "super": 0.293, "somewhat": -0.293,
            "kind of": -0.293, "kinda": -0.293, "sort of": -0.293,
            "a bit": -0.293, "a little": -0.293, "slightly": -0.293,
        }

        # Negation words
        self._negation_list = [
            "not", "isn't", "doesn't", "wasn't", "shouldn't", "wouldn't",
            "couldn't", "won't", "don't", "didn't", "haven't", "hasn't",
            "hadn't", "never", "neither", "nor", "cannot", "can't",
        ]

    def _normalize_score(self, score: float, alpha: float = 15) -> float:
        """Normalize score to -1 to 1 range."""
        return score / math.sqrt(score * score + alpha)

    async def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment using VADER approach."""
        import time
        start_time = time.time()

        # Tokenize
        words = re.findall(r'\b[\w\']+\b', text.lower())

        sentiments = []
        for i, word in enumerate(words):
            if word in self._lexicon:
                valence = self._lexicon[word]

                # Check for negation in previous 3 words
                for j in range(max(0, i - 3), i):
                    if words[j] in self._negation_list:
                        valence *= -0.74
                        break

                # Check for boosters
                for j in range(max(0, i - 1), i):
                    if words[j] in self._booster_dict:
                        valence += self._booster_dict[words[j]] * (1 if valence > 0 else -1)

                # Handle ALL CAPS
                if word.isupper():
                    valence += 0.733 * (1 if valence > 0 else -1)

                sentiments.append(valence)

        # Handle punctuation
        exclamation_count = text.count('!')
        question_count = text.count('?')

        # Calculate compound score
        if sentiments:
            sum_s = sum(sentiments)
            # Add punctuation influence
            sum_s += min(exclamation_count, 4) * 0.292

            compound = self._normalize_score(sum_s)
        else:
            compound = 0.0

        # Determine label
        if compound >= 0.5:
            label = SentimentLabel.VERY_POSITIVE
        elif compound >= 0.05:
            label = SentimentLabel.POSITIVE
        elif compound <= -0.5:
            label = SentimentLabel.VERY_NEGATIVE
        elif compound <= -0.05:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        # Calculate positive/negative/neutral proportions
        pos_sum = sum(s for s in sentiments if s > 0)
        neg_sum = sum(abs(s) for s in sentiments if s < 0)
        total = pos_sum + neg_sum + 0.0001

        confidence = 0.5 + min(total * 0.05, 0.45)

        sentiment = Sentiment(
            label=label,
            score=compound,
            confidence=confidence,
            magnitude=abs(compound) * (1 + len(sentiments) * 0.1),
        )

        # Detect emotions
        emotions = await self._detect_emotions(text, compound)

        return SentimentResult(
            text=text,
            sentiment=sentiment,
            emotions=emotions,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version="vader-1.0.0",
        )

    async def _detect_emotions(self, text: str, compound: float) -> List[EmotionScore]:
        """Detect emotions based on sentiment."""
        emotions = []

        # Map compound score to basic emotions
        if compound > 0.5:
            emotions.append(EmotionScore(
                emotion=Emotion.JOY,
                score=compound,
                confidence=0.8,
            ))
        elif compound < -0.5:
            if "angry" in text.lower() or "furious" in text.lower():
                emotions.append(EmotionScore(
                    emotion=Emotion.ANGER,
                    score=abs(compound),
                    confidence=0.8,
                ))
            else:
                emotions.append(EmotionScore(
                    emotion=Emotion.SADNESS,
                    score=abs(compound),
                    confidence=0.7,
                ))

        if not emotions:
            emotions.append(EmotionScore(
                emotion=Emotion.NEUTRAL,
                score=0.5,
                confidence=0.6,
            ))

        return emotions

    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment in multiple texts."""
        return await asyncio.gather(*[self.analyze(text) for text in texts])


class EmotionDetector:
    """
    Fine-grained emotion detection.

    Detects multiple emotions with intensity scores.
    """

    def __init__(self):
        self._emotion_patterns: Dict[Emotion, List[re.Pattern]] = {}
        self._emotion_keywords: Dict[Emotion, Set[str]] = {}
        self._setup_patterns()

    def _setup_patterns(self) -> None:
        """Setup emotion detection patterns."""
        self._emotion_patterns = {
            Emotion.JOY: [
                re.compile(r'\b(so\s+happy|very\s+happy|really\s+happy)\b', re.I),
                re.compile(r'\b(love\s+it|loved\s+it)\b', re.I),
                re.compile(r'\b(thank\s+you\s+so\s+much)\b', re.I),
            ],
            Emotion.ANGER: [
                re.compile(r'\b(so\s+angry|very\s+angry|really\s+angry)\b', re.I),
                re.compile(r'\b(pissed\s+off|fed\s+up)\b', re.I),
                re.compile(r'\b(this\s+is\s+(ridiculous|unacceptable))\b', re.I),
            ],
            Emotion.FRUSTRATION: [
                re.compile(r'\b(so\s+frustrated|very\s+frustrated)\b', re.I),
                re.compile(r'\b(doesn\'t\s+work|not\s+working)\b', re.I),
                re.compile(r'\b(keep\s+getting|keeps\s+happening)\b', re.I),
            ],
            Emotion.FEAR: [
                re.compile(r'\b(worried\s+about|concerned\s+about)\b', re.I),
                re.compile(r'\b(afraid\s+that|scared\s+that)\b', re.I),
            ],
            Emotion.CONFUSION: [
                re.compile(r'\b(don\'t\s+understand|confused\s+about)\b', re.I),
                re.compile(r'\b(what\s+does\s+this\s+mean|how\s+do\s+i)\b', re.I),
            ],
        }

        self._emotion_keywords = {
            Emotion.JOY: {
                "happy", "joy", "excited", "delighted", "thrilled", "pleased",
                "glad", "cheerful", "wonderful", "amazing", "fantastic",
            },
            Emotion.SADNESS: {
                "sad", "unhappy", "depressed", "disappointed", "upset",
                "heartbroken", "miserable", "gloomy",
            },
            Emotion.ANGER: {
                "angry", "furious", "outraged", "irritated", "mad",
                "enraged", "livid", "hostile",
            },
            Emotion.FEAR: {
                "afraid", "scared", "terrified", "frightened", "anxious",
                "worried", "nervous", "panicked",
            },
            Emotion.SURPRISE: {
                "surprised", "shocked", "amazed", "astonished", "stunned",
                "startled", "unexpected", "wow",
            },
            Emotion.DISGUST: {
                "disgusted", "revolted", "repulsed", "gross", "nasty",
                "awful", "horrible",
            },
            Emotion.FRUSTRATION: {
                "frustrated", "annoyed", "irritated", "stuck",
                "exasperated", "aggravated",
            },
            Emotion.SATISFACTION: {
                "satisfied", "content", "pleased", "fulfilled",
                "gratified", "resolved",
            },
            Emotion.DISAPPOINTMENT: {
                "disappointed", "let down", "dissatisfied",
                "dismayed", "discouraged",
            },
            Emotion.CONFUSION: {
                "confused", "puzzled", "perplexed", "bewildered",
                "lost", "unclear", "uncertain",
            },
        }

    async def detect(self, text: str) -> List[EmotionScore]:
        """Detect emotions in text."""
        emotions: Dict[Emotion, EmotionScore] = {}
        text_lower = text.lower()

        # Pattern matching
        for emotion, patterns in self._emotion_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    if emotion not in emotions:
                        emotions[emotion] = EmotionScore(
                            emotion=emotion,
                            score=0.0,
                            confidence=0.0,
                            indicators=[],
                        )
                    emotions[emotion].score += 0.4
                    emotions[emotion].confidence += 0.3
                    emotions[emotion].indicators.append(pattern.pattern)

        # Keyword matching
        for emotion, keywords in self._emotion_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                if emotion not in emotions:
                    emotions[emotion] = EmotionScore(
                        emotion=emotion,
                        score=0.0,
                        confidence=0.0,
                        indicators=[],
                    )
                emotions[emotion].score += min(len(matches) * 0.2, 0.6)
                emotions[emotion].confidence += min(len(matches) * 0.1, 0.4)
                emotions[emotion].indicators.extend(matches)

        # Normalize scores
        for emotion in emotions.values():
            emotion.score = min(emotion.score, 1.0)
            emotion.confidence = min(emotion.confidence, 0.95)

        # Sort by score
        result = sorted(emotions.values(), key=lambda e: -e.score)

        # Add neutral if no emotions
        if not result:
            result.append(EmotionScore(
                emotion=Emotion.NEUTRAL,
                score=0.5,
                confidence=0.5,
            ))

        return result


class AspectSentimentAnalyzer:
    """
    Aspect-based sentiment analysis.

    Detects sentiment for specific aspects/features.
    """

    def __init__(
        self,
        base_analyzer: Optional[SentimentAnalyzer] = None,
        aspects: Optional[List[str]] = None,
    ):
        self.base_analyzer = base_analyzer or LexiconSentimentAnalyzer()
        self.aspects = aspects or [
            "service", "quality", "price", "support", "product",
            "delivery", "experience", "staff", "speed", "value",
        ]
        self._aspect_patterns: Dict[str, List[re.Pattern]] = {}
        self._setup_patterns()

    def _setup_patterns(self) -> None:
        """Setup aspect extraction patterns."""
        for aspect in self.aspects:
            self._aspect_patterns[aspect] = [
                re.compile(rf'\b{aspect}\b.*?(?=[.!?]|$)', re.I),
                re.compile(rf'\b(the|your|their)\s+{aspect}\b.*?(?=[.!?]|$)', re.I),
            ]

    async def analyze(self, text: str) -> List[AspectSentiment]:
        """Analyze sentiment by aspect."""
        results = []

        for aspect, patterns in self._aspect_patterns.items():
            mentions = []
            for pattern in patterns:
                for match in pattern.finditer(text):
                    mentions.append(match.group())

            if mentions:
                # Analyze sentiment for mentions
                combined_text = " ".join(mentions)
                sentiment_result = await self.base_analyzer.analyze(combined_text)

                results.append(AspectSentiment(
                    aspect=aspect,
                    sentiment=sentiment_result.sentiment,
                    mentions=mentions,
                ))

        return results


class SentimentTrendTracker:
    """
    Track sentiment trends over conversation.

    Monitors sentiment changes and detects shifts.
    """

    def __init__(
        self,
        analyzer: Optional[SentimentAnalyzer] = None,
        window_size: int = 5,
    ):
        self.analyzer = analyzer or LexiconSentimentAnalyzer()
        self.window_size = window_size
        self._history: List[Tuple[str, Sentiment]] = []

    async def add(self, text: str) -> Dict[str, Any]:
        """Add text and get trend analysis."""
        result = await self.analyzer.analyze(text)
        self._history.append((text, result.sentiment))

        # Keep window size
        if len(self._history) > self.window_size * 2:
            self._history = self._history[-self.window_size * 2:]

        return {
            "current": result.sentiment.to_dict(),
            "trend": self._calculate_trend(),
            "shift_detected": self._detect_shift(),
        }

    def _calculate_trend(self) -> str:
        """Calculate sentiment trend."""
        if len(self._history) < 2:
            return "stable"

        recent = [s.score for _, s in self._history[-self.window_size:]]
        earlier = [s.score for _, s in self._history[-self.window_size * 2:-self.window_size]]

        if not earlier:
            return "stable"

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        diff = recent_avg - earlier_avg

        if diff > 0.2:
            return "improving"
        elif diff < -0.2:
            return "declining"
        return "stable"

    def _detect_shift(self) -> bool:
        """Detect significant sentiment shift."""
        if len(self._history) < 2:
            return False

        current = self._history[-1][1]
        previous = self._history[-2][1]

        # Shift from positive to negative or vice versa
        if current.is_positive and previous.is_negative:
            return True
        if current.is_negative and previous.is_positive:
            return True

        # Large score change
        if abs(current.score - previous.score) > 0.5:
            return True

        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get conversation sentiment summary."""
        if not self._history:
            return {"average": 0.0, "trend": "no_data", "volatility": 0.0}

        scores = [s.score for _, s in self._history]
        avg = sum(scores) / len(scores)

        # Calculate volatility
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        volatility = math.sqrt(variance)

        return {
            "average_score": avg,
            "average_label": self._score_to_label(avg),
            "trend": self._calculate_trend(),
            "volatility": volatility,
            "sample_count": len(self._history),
        }

    def _score_to_label(self, score: float) -> str:
        """Convert score to label string."""
        if score >= 0.5:
            return "very_positive"
        elif score >= 0.15:
            return "positive"
        elif score <= -0.5:
            return "very_negative"
        elif score <= -0.15:
            return "negative"
        return "neutral"


class SentimentAnalysisPipeline:
    """
    Complete sentiment analysis pipeline.

    Combines multiple analyzers for comprehensive analysis.
    """

    def __init__(
        self,
        analyzers: Optional[List[SentimentAnalyzer]] = None,
        include_emotions: bool = True,
        include_aspects: bool = False,
        aspect_list: Optional[List[str]] = None,
    ):
        self.analyzers = analyzers or [
            LexiconSentimentAnalyzer(),
            VADERSentimentAnalyzer(),
        ]
        self.include_emotions = include_emotions
        self.include_aspects = include_aspects
        self._emotion_detector = EmotionDetector() if include_emotions else None
        self._aspect_analyzer = AspectSentimentAnalyzer(
            aspects=aspect_list
        ) if include_aspects else None

    async def analyze(self, text: str) -> SentimentResult:
        """Run complete sentiment analysis."""
        import time
        start_time = time.time()

        # Get results from all analyzers
        results = await asyncio.gather(*[
            analyzer.analyze(text) for analyzer in self.analyzers
        ])

        # Aggregate sentiment scores
        total_score = sum(r.sentiment.score for r in results)
        total_confidence = sum(r.sentiment.confidence for r in results)
        n = len(results)

        avg_score = total_score / n
        avg_confidence = total_confidence / n

        # Determine label
        if avg_score >= 0.5:
            label = SentimentLabel.VERY_POSITIVE
        elif avg_score >= 0.15:
            label = SentimentLabel.POSITIVE
        elif avg_score <= -0.5:
            label = SentimentLabel.VERY_NEGATIVE
        elif avg_score <= -0.15:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        sentiment = Sentiment(
            label=label,
            score=avg_score,
            confidence=avg_confidence,
            magnitude=sum(r.sentiment.magnitude for r in results) / n,
        )

        # Detect emotions
        emotions = []
        if self._emotion_detector:
            emotions = await self._emotion_detector.detect(text)

        # Aspect analysis
        aspects = []
        if self._aspect_analyzer:
            aspects = await self._aspect_analyzer.analyze(text)

        return SentimentResult(
            text=text,
            sentiment=sentiment,
            emotions=emotions,
            aspects=aspects,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version="pipeline-1.0.0",
        )

    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts."""
        return await asyncio.gather(*[self.analyze(text) for text in texts])


# Factory function
def create_sentiment_analyzer(
    analyzer_type: str = "lexicon",
    **kwargs,
) -> SentimentAnalyzer:
    """Create sentiment analyzer by type."""
    analyzers = {
        "lexicon": LexiconSentimentAnalyzer,
        "vader": VADERSentimentAnalyzer,
        "pipeline": SentimentAnalysisPipeline,
    }

    cls = analyzers.get(analyzer_type, LexiconSentimentAnalyzer)
    return cls(**kwargs)
