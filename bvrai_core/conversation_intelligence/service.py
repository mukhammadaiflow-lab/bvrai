"""
Conversation Intelligence Service Module

This module provides the service layer for real-time conversation analysis,
including sentiment analysis, intent detection, emotion recognition,
entity extraction, and more.

Key Services:
- SentimentAnalyzer: Real-time sentiment analysis
- EmotionDetector: Emotion recognition from text/speech
- IntentClassifier: Intent detection and classification
- EntityExtractor: Named entity recognition
- TopicTracker: Topic detection and context tracking
- PerformanceEvaluator: Agent performance scoring
- SatisfactionPredictor: Customer satisfaction prediction
- EscalationDetector: Escalation need detection
- ConversationSummarizer: Automatic summarization
- ConversationIntelligenceService: Main orchestrating service
"""

import asyncio
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from .base import (
    # Sentiment
    SentimentLabel,
    SentimentTrend,
    SentimentScore,
    ConversationSentiment,
    # Emotion
    EmotionType,
    EmotionScore,
    EmotionalProfile,
    # Intent
    IntentCategory,
    IntentPriority,
    Intent,
    IntentFlow,
    # Entity
    EntityType,
    Entity,
    EntityCollection,
    # Topic
    TopicCategory,
    Topic,
    ConversationContext,
    # Performance
    PerformanceMetricType,
    PerformanceMetric,
    AgentPerformanceScore,
    # Satisfaction
    SatisfactionLevel,
    SatisfactionPrediction,
    # Escalation
    EscalationReason,
    EscalationPriority,
    EscalationSignal,
    EscalationAssessment,
    # Summary
    ConversationSummary,
    # Main analysis
    ConversationAnalysis,
    # Exceptions
    ConversationIntelligenceError,
    AnalysisError,
    ModelNotLoadedError,
    InsufficientDataError,
)


# =============================================================================
# Utterance Type
# =============================================================================


@dataclass
class Utterance:
    """A single utterance in a conversation."""

    id: str
    text: str
    speaker: str  # "agent" or "customer"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Audio features (optional)
    audio_features: Optional[Dict[str, float]] = None

    # Analysis results (populated by analyzers)
    sentiment: Optional[SentimentScore] = None
    emotion: Optional[EmotionScore] = None
    intent: Optional[Intent] = None
    entities: List[Entity] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = f"utt_{uuid.uuid4().hex[:12]}"


# =============================================================================
# Sentiment Analyzer
# =============================================================================


class SentimentAnalyzer:
    """
    Real-time sentiment analysis service.

    Uses rule-based analysis with lexicon scoring, enhanced by
    contextual understanding of conversation flow.
    """

    def __init__(self):
        # Sentiment lexicons
        self._positive_words: Set[str] = {
            "great", "excellent", "amazing", "wonderful", "fantastic", "perfect",
            "good", "nice", "helpful", "appreciate", "thanks", "thank", "love",
            "happy", "pleased", "satisfied", "awesome", "brilliant", "superb",
            "outstanding", "remarkable", "terrific", "delighted", "grateful",
            "impressed", "excited", "joy", "cheerful", "optimistic", "positive",
        }
        self._negative_words: Set[str] = {
            "bad", "terrible", "awful", "horrible", "worst", "poor", "wrong",
            "disappointed", "frustrating", "annoying", "angry", "upset", "hate",
            "problem", "issue", "broken", "failed", "useless", "waste", "stupid",
            "ridiculous", "unacceptable", "pathetic", "disgusting", "furious",
            "outraged", "miserable", "dreadful", "atrocious", "incompetent",
        }
        self._intensifiers: Set[str] = {
            "very", "really", "extremely", "absolutely", "completely", "totally",
            "utterly", "highly", "incredibly", "exceptionally", "particularly",
        }
        self._negators: Set[str] = {
            "not", "no", "never", "none", "nothing", "nobody", "neither",
            "hardly", "barely", "scarcely", "without", "don't", "doesn't",
            "didn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't",
        }

    def analyze(self, text: str, speaker: str = "") -> SentimentScore:
        """Analyze sentiment of a single text."""
        words = text.lower().split()

        positive_count = 0
        negative_count = 0
        intensifier_boost = 0
        negation_active = False

        for i, word in enumerate(words):
            # Check for negation
            if word in self._negators:
                negation_active = True
                continue

            # Check for intensifiers
            if word in self._intensifiers:
                intensifier_boost += 0.2
                continue

            # Check sentiment words
            is_positive = word in self._positive_words
            is_negative = word in self._negative_words

            if negation_active:
                # Flip sentiment
                is_positive, is_negative = is_negative, is_positive
                negation_active = False

            if is_positive:
                positive_count += 1 + intensifier_boost
            elif is_negative:
                negative_count += 1 + intensifier_boost

            intensifier_boost = 0

        # Calculate scores
        total = max(positive_count + negative_count, 1)
        positive_score = positive_count / total
        negative_score = negative_count / total
        neutral_score = 1 - positive_score - negative_score

        # Calculate overall score (-1 to 1)
        score = (positive_score - negative_score)

        # Determine label
        if score > 0.3:
            label = SentimentLabel.VERY_POSITIVE if score > 0.6 else SentimentLabel.POSITIVE
        elif score < -0.3:
            label = SentimentLabel.VERY_NEGATIVE if score < -0.6 else SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        # Calculate confidence
        confidence = abs(score) * 0.5 + 0.5

        return SentimentScore(
            label=label,
            score=score,
            confidence=confidence,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score,
            speaker=speaker,
            text=text,
            timestamp=datetime.utcnow(),
        )

    def analyze_conversation(
        self, utterances: List[Utterance]
    ) -> ConversationSentiment:
        """Analyze sentiment for an entire conversation."""
        if not utterances:
            return ConversationSentiment(
                overall_score=0.0,
                overall_label=SentimentLabel.NEUTRAL,
            )

        sentiment_timeline = []
        customer_scores = []
        agent_scores = []

        for utterance in utterances:
            sentiment = self.analyze(utterance.text, utterance.speaker)
            sentiment_timeline.append(sentiment)

            if utterance.speaker == "customer":
                customer_scores.append(sentiment.score)
            else:
                agent_scores.append(sentiment.score)

        # Calculate overall scores
        all_scores = [s.score for s in sentiment_timeline]
        overall_score = sum(all_scores) / len(all_scores)

        customer_sentiment = sum(customer_scores) / len(customer_scores) if customer_scores else 0
        agent_sentiment = sum(agent_scores) / len(agent_scores) if agent_scores else 0

        # Determine trend
        if len(all_scores) >= 2:
            first_half = sum(all_scores[:len(all_scores)//2]) / (len(all_scores)//2)
            second_half = sum(all_scores[len(all_scores)//2:]) / (len(all_scores) - len(all_scores)//2)
            trend_delta = second_half - first_half

            if trend_delta > 0.1:
                trend = SentimentTrend.IMPROVING
            elif trend_delta < -0.1:
                trend = SentimentTrend.DECLINING
            else:
                trend = SentimentTrend.STABLE
        else:
            trend = SentimentTrend.STABLE
            trend_delta = 0.0

        # Determine overall label
        if overall_score > 0.3:
            overall_label = SentimentLabel.VERY_POSITIVE if overall_score > 0.6 else SentimentLabel.POSITIVE
        elif overall_score < -0.3:
            overall_label = SentimentLabel.VERY_NEGATIVE if overall_score < -0.6 else SentimentLabel.NEGATIVE
        else:
            overall_label = SentimentLabel.NEUTRAL

        # Find key moments
        most_positive = max(sentiment_timeline, key=lambda s: s.score)
        most_negative = min(sentiment_timeline, key=lambda s: s.score)

        return ConversationSentiment(
            overall_score=overall_score,
            overall_label=overall_label,
            customer_sentiment=customer_sentiment,
            agent_sentiment=agent_sentiment,
            trend=trend,
            trend_delta=trend_delta,
            sentiment_timeline=sentiment_timeline,
            most_positive_moment=most_positive,
            most_negative_moment=most_negative,
        )


# =============================================================================
# Emotion Detector
# =============================================================================


class EmotionDetector:
    """
    Emotion detection service.

    Detects emotions from text using keyword matching and
    contextual patterns.
    """

    def __init__(self):
        # Emotion keywords
        self._emotion_keywords: Dict[EmotionType, Set[str]] = {
            EmotionType.JOY: {
                "happy", "glad", "delighted", "pleased", "thrilled", "excited",
                "wonderful", "fantastic", "amazing", "love", "great", "awesome",
            },
            EmotionType.SADNESS: {
                "sad", "unhappy", "disappointed", "sorry", "unfortunate",
                "regret", "miss", "upset", "down", "depressed", "heartbroken",
            },
            EmotionType.ANGER: {
                "angry", "furious", "outraged", "mad", "irritated", "annoyed",
                "frustrated", "infuriated", "livid", "enraged", "hate",
            },
            EmotionType.FEAR: {
                "afraid", "scared", "worried", "anxious", "nervous", "terrified",
                "concerned", "fearful", "panicked", "alarmed",
            },
            EmotionType.SURPRISE: {
                "surprised", "shocked", "amazed", "astonished", "stunned",
                "unexpected", "unbelievable", "wow", "incredible",
            },
            EmotionType.FRUSTRATION: {
                "frustrated", "annoying", "irritating", "tiresome", "exasperating",
                "aggravating", "maddening", "infuriating",
            },
            EmotionType.CONFUSION: {
                "confused", "puzzled", "unclear", "uncertain", "unsure",
                "lost", "bewildered", "perplexed", "baffled",
            },
            EmotionType.SATISFACTION: {
                "satisfied", "content", "pleased", "fulfilled", "gratified",
                "happy", "good", "great", "perfect",
            },
            EmotionType.DISAPPOINTMENT: {
                "disappointed", "letdown", "underwhelmed", "unsatisfied",
                "displeased", "discouraged", "dismayed",
            },
            EmotionType.EXCITEMENT: {
                "excited", "thrilled", "eager", "enthusiastic", "pumped",
                "hyped", "stoked", "can't wait",
            },
            EmotionType.ANXIETY: {
                "anxious", "worried", "nervous", "uneasy", "apprehensive",
                "tense", "stressed", "on edge",
            },
            EmotionType.RELIEF: {
                "relieved", "grateful", "thankful", "glad", "phew",
                "finally", "at last", "thank goodness",
            },
            EmotionType.IMPATIENCE: {
                "hurry", "quickly", "asap", "urgent", "immediately",
                "waiting", "still", "yet", "already",
            },
        }

    def detect(self, text: str, speaker: str = "") -> EmotionScore:
        """Detect primary emotion in text."""
        words = set(text.lower().split())

        # Score each emotion
        emotion_scores: Dict[EmotionType, float] = {}
        for emotion, keywords in self._emotion_keywords.items():
            matches = words.intersection(keywords)
            if matches:
                emotion_scores[emotion] = len(matches) / len(keywords)

        if not emotion_scores:
            return EmotionScore(
                emotion=EmotionType.NEUTRAL,
                score=0.5,
                confidence=0.3,
                speaker=speaker,
            )

        # Get primary emotion
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        primary_score = emotion_scores[primary_emotion]

        # Get secondary emotions
        secondary = [
            (e, s) for e, s in emotion_scores.items()
            if e != primary_emotion and s > 0.1
        ]
        secondary.sort(key=lambda x: x[1], reverse=True)

        return EmotionScore(
            emotion=primary_emotion,
            score=min(primary_score * 2, 1.0),  # Scale up
            confidence=min(0.5 + primary_score, 0.95),
            speaker=speaker,
            secondary_emotions=secondary[:3],
        )

    def build_emotional_profile(
        self, utterances: List[Utterance], speaker: str
    ) -> EmotionalProfile:
        """Build emotional profile for a speaker."""
        speaker_utterances = [u for u in utterances if u.speaker == speaker]

        if not speaker_utterances:
            return EmotionalProfile(speaker=speaker)

        # Detect emotions for each utterance
        emotion_timeline = []
        emotion_counts: Dict[EmotionType, float] = {}

        for utterance in speaker_utterances:
            emotion = self.detect(utterance.text, speaker)
            emotion_timeline.append(emotion)

            if emotion.emotion not in emotion_counts:
                emotion_counts[emotion.emotion] = 0
            emotion_counts[emotion.emotion] += emotion.score

        # Calculate distribution
        total = sum(emotion_counts.values())
        distribution = {
            e: s / total if total > 0 else 0
            for e, s in emotion_counts.items()
        }

        # Get dominant emotion
        dominant = max(emotion_counts, key=emotion_counts.get)
        dominant_score = distribution.get(dominant, 0)

        # Calculate volatility (variance in emotions)
        if len(emotion_timeline) > 1:
            unique_emotions = len(set(e.emotion for e in emotion_timeline))
            volatility = unique_emotions / len(EmotionType)
        else:
            volatility = 0

        return EmotionalProfile(
            speaker=speaker,
            dominant_emotion=dominant,
            dominant_score=dominant_score,
            emotion_distribution=distribution,
            volatility=volatility,
            emotion_timeline=emotion_timeline,
        )


# =============================================================================
# Intent Classifier
# =============================================================================


class IntentClassifier:
    """
    Intent classification service.

    Classifies utterances into intent categories with slot extraction.
    """

    def __init__(self):
        # Intent patterns (regex patterns for each category)
        self._intent_patterns: Dict[IntentCategory, List[str]] = {
            IntentCategory.QUESTION: [
                r"\?$", r"^(what|who|where|when|why|how|which|can|could|would|is|are|do|does)\b",
            ],
            IntentCategory.REQUEST: [
                r"^(please|can you|could you|would you|i need|i want|i'd like)\b",
                r"(help me|assist me|show me|tell me)\b",
            ],
            IntentCategory.COMPLAINT: [
                r"(complaint|complain|unhappy|dissatisfied|terrible|worst|awful)\b",
                r"(not working|doesn't work|broken|failed)\b",
            ],
            IntentCategory.BOOKING: [
                r"(book|reserve|schedule|appointment|reservation)\b",
            ],
            IntentCategory.CANCELLATION: [
                r"(cancel|cancellation|terminate|end|stop)\b",
            ],
            IntentCategory.REFUND: [
                r"(refund|money back|reimburse|return)\b",
            ],
            IntentCategory.GREETING: [
                r"^(hi|hello|hey|good morning|good afternoon|good evening)\b",
            ],
            IntentCategory.GOODBYE: [
                r"(bye|goodbye|see you|take care|have a good)\b",
            ],
            IntentCategory.THANKS: [
                r"(thank|thanks|appreciate|grateful)\b",
            ],
            IntentCategory.ESCALATION: [
                r"(supervisor|manager|escalate|speak to someone|human)\b",
            ],
            IntentCategory.TECHNICAL_ISSUE: [
                r"(error|bug|crash|not loading|slow|timeout)\b",
            ],
            IntentCategory.BILLING: [
                r"(bill|invoice|charge|payment|price|cost)\b",
            ],
            IntentCategory.ACCOUNT: [
                r"(account|password|login|sign in|profile)\b",
            ],
        }

        # Priority mapping
        self._priority_map: Dict[IntentCategory, IntentPriority] = {
            IntentCategory.ESCALATION: IntentPriority.CRITICAL,
            IntentCategory.COMPLAINT: IntentPriority.HIGH,
            IntentCategory.TECHNICAL_ISSUE: IntentPriority.HIGH,
            IntentCategory.CANCELLATION: IntentPriority.HIGH,
            IntentCategory.REFUND: IntentPriority.MEDIUM,
            IntentCategory.BOOKING: IntentPriority.MEDIUM,
            IntentCategory.REQUEST: IntentPriority.MEDIUM,
            IntentCategory.QUESTION: IntentPriority.MEDIUM,
            IntentCategory.GREETING: IntentPriority.LOW,
            IntentCategory.GOODBYE: IntentPriority.LOW,
            IntentCategory.THANKS: IntentPriority.LOW,
        }

    def classify(self, text: str, speaker: str = "") -> Intent:
        """Classify the intent of an utterance."""
        text_lower = text.lower()
        matched_intents: List[Tuple[IntentCategory, float]] = []

        for category, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    # Higher confidence for more specific patterns
                    confidence = 0.6 + (0.1 * len(pattern) / 20)
                    matched_intents.append((category, min(confidence, 0.95)))
                    break

        if not matched_intents:
            return Intent(
                id="",
                category=IntentCategory.OTHER,
                confidence=0.3,
                utterance=text,
                speaker=speaker,
            )

        # Get highest confidence match
        matched_intents.sort(key=lambda x: x[1], reverse=True)
        category, confidence = matched_intents[0]

        # Determine priority
        priority = self._priority_map.get(category, IntentPriority.MEDIUM)
        requires_action = priority in [IntentPriority.CRITICAL, IntentPriority.HIGH]

        return Intent(
            id="",
            category=category,
            confidence=confidence,
            priority=priority,
            requires_action=requires_action,
            utterance=text,
            speaker=speaker,
        )

    def classify_conversation(
        self, utterances: List[Utterance]
    ) -> IntentFlow:
        """Classify intents for an entire conversation."""
        flow = IntentFlow()

        for utterance in utterances:
            intent = self.classify(utterance.text, utterance.speaker)
            flow.add_intent(intent)

        return flow


# =============================================================================
# Entity Extractor
# =============================================================================


class EntityExtractor:
    """
    Named entity extraction service.

    Extracts entities like dates, times, phone numbers, emails, etc.
    """

    def __init__(self):
        # Entity patterns
        self._patterns: Dict[EntityType, str] = {
            EntityType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            EntityType.PHONE: r'\b(\+?1?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            EntityType.URL: r'https?://[^\s]+',
            EntityType.DATE: r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s*\d{2,4}\b',
            EntityType.TIME: r'\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)?\b|\b(noon|midnight)\b',
            EntityType.MONEY: r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+\s*(?:dollars?|cents?|usd)\b',
            EntityType.PERCENTAGE: r'\b\d+(?:\.\d+)?%\b',
            EntityType.ORDER_NUMBER: r'\b(?:order|order\s*#|order\s*number|ord)[:\s]*([A-Z0-9-]+)\b',
            EntityType.ACCOUNT_NUMBER: r'\b(?:account|acct|account\s*#)[:\s]*([A-Z0-9-]+)\b',
        }

    def extract(self, text: str, speaker: str = "") -> List[Entity]:
        """Extract entities from text."""
        entities = []

        for entity_type, pattern in self._patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = Entity(
                    id="",
                    type=entity_type,
                    value=match.group(),
                    confidence=0.9,
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    speaker=speaker,
                )
                entities.append(entity)

        return entities

    def extract_from_conversation(
        self, utterances: List[Utterance]
    ) -> EntityCollection:
        """Extract entities from entire conversation."""
        collection = EntityCollection()

        for utterance in utterances:
            entities = self.extract(utterance.text, utterance.speaker)
            for entity in entities:
                entity.utterance_id = utterance.id
                collection.add_entity(entity)

        # Identify key entities (most frequent or most relevant)
        type_counts = {t: len(entities) for t, entities in collection.by_type.items()}
        if type_counts:
            # Get top entity types
            top_types = sorted(type_counts, key=type_counts.get, reverse=True)[:5]
            for entity_type in top_types:
                if collection.by_type[entity_type]:
                    collection.key_entities.append(collection.by_type[entity_type][0])

        return collection


# =============================================================================
# Topic Tracker
# =============================================================================


class TopicTracker:
    """
    Topic tracking and context management service.

    Tracks topics discussed in the conversation and maintains context.
    """

    def __init__(self):
        # Topic keywords
        self._topic_keywords: Dict[TopicCategory, Set[str]] = {
            TopicCategory.PRODUCT_INQUIRY: {
                "product", "item", "model", "version", "feature", "specs",
            },
            TopicCategory.SERVICE_INQUIRY: {
                "service", "plan", "subscription", "package", "offer",
            },
            TopicCategory.PRICING: {
                "price", "cost", "fee", "charge", "rate", "discount",
            },
            TopicCategory.TECHNICAL_SUPPORT: {
                "error", "bug", "issue", "problem", "broken", "not working",
            },
            TopicCategory.BILLING_SUPPORT: {
                "bill", "invoice", "payment", "charge", "refund",
            },
            TopicCategory.ACCOUNT_MANAGEMENT: {
                "account", "password", "login", "profile", "settings",
            },
            TopicCategory.ORDER_STATUS: {
                "order", "shipping", "delivery", "tracking", "status",
            },
            TopicCategory.APPOINTMENT: {
                "appointment", "schedule", "book", "reserve", "availability",
            },
            TopicCategory.COMPLAINT: {
                "complaint", "unhappy", "dissatisfied", "terrible", "worst",
            },
            TopicCategory.CANCELLATION: {
                "cancel", "terminate", "close", "end", "stop",
            },
        }

    def detect_topic(self, text: str) -> Optional[Topic]:
        """Detect topic from text."""
        text_lower = text.lower()
        words = set(text_lower.split())

        topic_scores: Dict[TopicCategory, float] = {}
        for category, keywords in self._topic_keywords.items():
            matches = words.intersection(keywords)
            if matches:
                topic_scores[category] = len(matches) / len(keywords)

        if not topic_scores:
            return None

        # Get top topic
        top_category = max(topic_scores, key=topic_scores.get)
        confidence = topic_scores[top_category]

        return Topic(
            id="",
            name=top_category.value.replace("_", " ").title(),
            category=top_category,
            confidence=min(confidence * 2, 0.95),
            keywords=list(words.intersection(self._topic_keywords.get(top_category, set()))),
        )

    def track_conversation(
        self, utterances: List[Utterance]
    ) -> ConversationContext:
        """Track topics and context throughout conversation."""
        context = ConversationContext()

        for utterance in utterances:
            topic = self.detect_topic(utterance.text)
            if topic:
                topic.utterance_count = 1

                # Check if topic changed
                if context.current_topic:
                    if topic.category != context.current_topic.category:
                        context.switch_topic(topic)
                    else:
                        # Same topic, increment count
                        context.current_topic.utterance_count += 1
                else:
                    context.current_topic = topic
                    context.primary_topic = topic

        return context


# =============================================================================
# Performance Evaluator
# =============================================================================


class PerformanceEvaluator:
    """
    Agent performance evaluation service.

    Evaluates agent performance based on various metrics.
    """

    def __init__(self):
        # Empathy phrases
        self._empathy_phrases: Set[str] = {
            "i understand", "i'm sorry", "i apologize", "i can see",
            "i appreciate", "thank you for", "let me help",
            "i'll do my best", "i hear you", "that must be",
        }

        # Professional phrases
        self._professional_phrases: Set[str] = {
            "please", "thank you", "of course", "certainly",
            "i'd be happy to", "absolutely", "my pleasure",
        }

    def evaluate(
        self,
        utterances: List[Utterance],
        sentiment: Optional[ConversationSentiment] = None,
    ) -> AgentPerformanceScore:
        """Evaluate agent performance."""
        agent_utterances = [u for u in utterances if u.speaker == "agent"]

        if not agent_utterances:
            return AgentPerformanceScore(overall_score=0)

        metrics = {}

        # Evaluate empathy
        empathy_score = self._evaluate_empathy(agent_utterances)
        metrics[PerformanceMetricType.EMPATHY] = empathy_score

        # Evaluate professionalism
        professionalism_score = self._evaluate_professionalism(agent_utterances)
        metrics[PerformanceMetricType.PROFESSIONALISM] = professionalism_score

        # Evaluate response clarity
        clarity_score = self._evaluate_clarity(agent_utterances)
        metrics[PerformanceMetricType.RESPONSE_CLARITY] = clarity_score

        # Calculate category scores
        empathy_final = metrics[PerformanceMetricType.EMPATHY].score * 100
        professionalism_final = metrics[PerformanceMetricType.PROFESSIONALISM].score * 100
        clarity_final = metrics[PerformanceMetricType.RESPONSE_CLARITY].score * 100

        # Calculate overall score
        overall_score = (empathy_final + professionalism_final + clarity_final) / 3

        # Determine strengths and improvements
        strengths = []
        improvements = []

        if empathy_final >= 70:
            strengths.append("Shows empathy and understanding")
        elif empathy_final < 50:
            improvements.append("Increase empathetic language")

        if professionalism_final >= 70:
            strengths.append("Maintains professional tone")
        elif professionalism_final < 50:
            improvements.append("Use more professional language")

        if clarity_final >= 70:
            strengths.append("Provides clear responses")
        elif clarity_final < 50:
            improvements.append("Improve response clarity")

        return AgentPerformanceScore(
            overall_score=overall_score,
            metrics=metrics,
            empathy_score=empathy_final,
            communication_score=professionalism_final,
            response_quality_score=clarity_final,
            strengths=strengths,
            improvements_needed=improvements,
        )

    def _evaluate_empathy(
        self, utterances: List[Utterance]
    ) -> PerformanceMetric:
        """Evaluate empathy in agent responses."""
        total_matches = 0
        evidence = []

        for utterance in utterances:
            text_lower = utterance.text.lower()
            for phrase in self._empathy_phrases:
                if phrase in text_lower:
                    total_matches += 1
                    evidence.append(f"Used empathetic phrase: '{phrase}'")
                    break

        score = min(total_matches / max(len(utterances), 1), 1.0)

        return PerformanceMetric(
            metric_type=PerformanceMetricType.EMPATHY,
            score=score,
            confidence=0.8,
            evidence=evidence[:5],
        )

    def _evaluate_professionalism(
        self, utterances: List[Utterance]
    ) -> PerformanceMetric:
        """Evaluate professionalism in agent responses."""
        total_matches = 0
        evidence = []

        for utterance in utterances:
            text_lower = utterance.text.lower()
            for phrase in self._professional_phrases:
                if phrase in text_lower:
                    total_matches += 1
                    evidence.append(f"Used professional phrase: '{phrase}'")
                    break

        score = min(total_matches / max(len(utterances), 1), 1.0)

        return PerformanceMetric(
            metric_type=PerformanceMetricType.PROFESSIONALISM,
            score=score,
            confidence=0.8,
            evidence=evidence[:5],
        )

    def _evaluate_clarity(
        self, utterances: List[Utterance]
    ) -> PerformanceMetric:
        """Evaluate response clarity."""
        # Simple heuristic: moderate length responses are clearer
        total_score = 0
        for utterance in utterances:
            word_count = len(utterance.text.split())
            # Optimal range: 10-50 words
            if 10 <= word_count <= 50:
                total_score += 1.0
            elif word_count < 10:
                total_score += 0.5
            else:
                total_score += 0.7

        score = total_score / max(len(utterances), 1)

        return PerformanceMetric(
            metric_type=PerformanceMetricType.RESPONSE_CLARITY,
            score=score,
            confidence=0.7,
        )


# =============================================================================
# Satisfaction Predictor
# =============================================================================


class SatisfactionPredictor:
    """
    Customer satisfaction prediction service.

    Predicts customer satisfaction based on conversation analysis.
    """

    def predict(
        self,
        sentiment: ConversationSentiment,
        emotions: Optional[EmotionalProfile] = None,
        intents: Optional[IntentFlow] = None,
        escalation: Optional[EscalationAssessment] = None,
    ) -> SatisfactionPrediction:
        """Predict customer satisfaction."""
        # Base score from sentiment
        base_score = (sentiment.customer_sentiment + 1) / 2 * 100  # Convert -1..1 to 0..100

        # Adjust based on trend
        if sentiment.trend == SentimentTrend.IMPROVING:
            base_score += 10
        elif sentiment.trend == SentimentTrend.DECLINING:
            base_score -= 15

        # Adjust based on emotions
        positive_factors = []
        negative_factors = []

        if emotions:
            if emotions.dominant_emotion in [EmotionType.JOY, EmotionType.SATISFACTION, EmotionType.RELIEF]:
                base_score += 10
                positive_factors.append(f"Expressed {emotions.dominant_emotion.value}")
            elif emotions.dominant_emotion in [EmotionType.ANGER, EmotionType.FRUSTRATION]:
                base_score -= 20
                negative_factors.append(f"Expressed {emotions.dominant_emotion.value}")

        # Adjust for escalation
        if escalation and escalation.should_escalate:
            base_score -= 25
            negative_factors.append("Escalation required")

        # Clamp score
        final_score = max(0, min(100, base_score))

        # Determine level
        if final_score >= 80:
            level = SatisfactionLevel.VERY_SATISFIED
        elif final_score >= 60:
            level = SatisfactionLevel.SATISFIED
        elif final_score >= 40:
            level = SatisfactionLevel.NEUTRAL
        elif final_score >= 20:
            level = SatisfactionLevel.DISSATISFIED
        else:
            level = SatisfactionLevel.VERY_DISSATISFIED

        # Calculate risks
        churn_risk = max(0, (100 - final_score) / 100)
        escalation_risk = 0.3 if escalation and escalation.should_escalate else 0.1

        # Recommendations
        recommendations = []
        if final_score < 60:
            recommendations.append("Follow up with customer within 24 hours")
        if negative_factors:
            recommendations.append("Address customer concerns proactively")
        if churn_risk > 0.5:
            recommendations.append("Offer compensation or special attention")

        return SatisfactionPrediction(
            predicted_level=level,
            predicted_score=final_score,
            confidence=0.75,
            positive_factors=positive_factors,
            negative_factors=negative_factors,
            churn_risk=churn_risk,
            escalation_risk=escalation_risk,
            retention_recommendations=recommendations,
        )


# =============================================================================
# Escalation Detector
# =============================================================================


class EscalationDetector:
    """
    Escalation detection service.

    Detects when a conversation should be escalated.
    """

    def __init__(self):
        # Escalation trigger phrases
        self._escalation_phrases: Dict[EscalationReason, Set[str]] = {
            EscalationReason.CUSTOMER_REQUEST: {
                "supervisor", "manager", "speak to someone", "human",
                "real person", "escalate", "higher up",
            },
            EscalationReason.FRUSTRATED_CUSTOMER: {
                "frustrated", "fed up", "had enough", "ridiculous",
                "unacceptable", "waste of time",
            },
            EscalationReason.ANGRY_CUSTOMER: {
                "angry", "furious", "outraged", "livid", "sue",
                "lawyer", "legal action", "report",
            },
            EscalationReason.REPEATED_ISSUE: {
                "again", "still", "same problem", "already told",
                "how many times", "keep happening",
            },
        }

    def detect(
        self,
        utterances: List[Utterance],
        sentiment: Optional[ConversationSentiment] = None,
        emotions: Optional[EmotionalProfile] = None,
    ) -> EscalationAssessment:
        """Detect if escalation is needed."""
        signals = []

        # Check for explicit escalation requests
        for utterance in utterances:
            if utterance.speaker == "customer":
                text_lower = utterance.text.lower()
                for reason, phrases in self._escalation_phrases.items():
                    for phrase in phrases:
                        if phrase in text_lower:
                            priority = EscalationPriority.HIGH
                            if reason == EscalationReason.ANGRY_CUSTOMER:
                                priority = EscalationPriority.CRITICAL

                            signals.append(EscalationSignal(
                                reason=reason,
                                priority=priority,
                                confidence=0.9,
                                trigger_text=utterance.text,
                                trigger_timestamp=utterance.timestamp,
                            ))
                            break

        # Check sentiment-based escalation
        if sentiment and sentiment.customer_sentiment < -0.5:
            signals.append(EscalationSignal(
                reason=EscalationReason.NEGATIVE_SENTIMENT,
                priority=EscalationPriority.MEDIUM,
                confidence=0.7,
                context="Customer sentiment is highly negative",
            ))

        # Check emotion-based escalation
        if emotions:
            if emotions.dominant_emotion in [EmotionType.ANGER, EmotionType.FRUSTRATION]:
                signals.append(EscalationSignal(
                    reason=EscalationReason.FRUSTRATED_CUSTOMER,
                    priority=EscalationPriority.HIGH,
                    confidence=0.8,
                    context=f"Customer showing {emotions.dominant_emotion.value}",
                ))

        # Determine overall escalation need
        should_escalate = len(signals) > 0 and any(
            s.priority in [EscalationPriority.CRITICAL, EscalationPriority.HIGH]
            for s in signals
        )

        probability = min(len(signals) * 0.3, 1.0) if signals else 0.0

        # Get primary reason and priority
        primary_reason = None
        priority = EscalationPriority.LOW
        if signals:
            signals.sort(key=lambda s: (
                0 if s.priority == EscalationPriority.CRITICAL else
                1 if s.priority == EscalationPriority.HIGH else
                2 if s.priority == EscalationPriority.MEDIUM else 3
            ))
            primary_reason = signals[0].reason
            priority = signals[0].priority

        # Recommendations
        recommendations = []
        if should_escalate:
            recommendations.append("Transfer to human agent")
            if primary_reason == EscalationReason.ANGRY_CUSTOMER:
                recommendations.append("Apologize sincerely")
                recommendations.append("Offer compensation if appropriate")

        return EscalationAssessment(
            should_escalate=should_escalate,
            escalation_probability=probability,
            priority=priority,
            signals=signals,
            primary_reason=primary_reason,
            recommended_actions=recommendations,
            recommended_target="supervisor" if should_escalate else "",
        )


# =============================================================================
# Conversation Summarizer
# =============================================================================


class ConversationSummarizer:
    """
    Conversation summarization service.

    Generates summaries of conversations.
    """

    def summarize(
        self,
        utterances: List[Utterance],
        context: Optional[ConversationContext] = None,
        sentiment: Optional[ConversationSentiment] = None,
        intents: Optional[IntentFlow] = None,
        entities: Optional[EntityCollection] = None,
    ) -> ConversationSummary:
        """Generate conversation summary."""
        if not utterances:
            return ConversationSummary()

        # Determine main topic
        main_topics = []
        if context and context.primary_topic:
            main_topics.append(context.primary_topic.name)

        # Determine outcome based on sentiment trend
        if sentiment:
            if sentiment.trend == SentimentTrend.IMPROVING:
                outcome = "resolved"
            elif sentiment.customer_sentiment > 0:
                outcome = "positive"
            elif sentiment.customer_sentiment < -0.3:
                outcome = "unresolved"
            else:
                outcome = "neutral"
        else:
            outcome = "unknown"

        # Extract key points
        key_points = []
        if intents and intents.primary_intent:
            key_points.append(
                f"Primary intent: {intents.primary_intent.category.value}"
            )

        if entities:
            for entity in entities.key_entities[:3]:
                key_points.append(f"Mentioned: {entity.type.value} - {entity.value}")

        # Generate brief summary
        customer_msgs = len([u for u in utterances if u.speaker == "customer"])
        agent_msgs = len([u for u in utterances if u.speaker == "agent"])
        brief = f"Conversation with {customer_msgs} customer and {agent_msgs} agent messages."

        if main_topics:
            brief += f" Main topic: {main_topics[0]}."
        if outcome != "unknown":
            brief += f" Outcome: {outcome}."

        # Check for follow-up needs
        follow_up = outcome in ["unresolved", "neutral"]
        follow_up_reason = ""
        if follow_up:
            follow_up_reason = "Issue may not be fully resolved"

        return ConversationSummary(
            brief_summary=brief,
            detailed_summary=brief,  # Same for now, can be enhanced
            key_points=key_points,
            outcome=outcome,
            main_topics=main_topics,
            follow_up_required=follow_up,
            follow_up_reason=follow_up_reason,
            action_items=[{"action": "Review conversation", "priority": "low"}] if follow_up else [],
        )


# =============================================================================
# Main Conversation Intelligence Service
# =============================================================================


class ConversationIntelligenceService:
    """
    Main service for conversation intelligence.

    This is the primary entry point for all conversation analysis operations.
    """

    def __init__(self):
        # Initialize analyzers
        self.sentiment_analyzer = SentimentAnalyzer()
        self.emotion_detector = EmotionDetector()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.topic_tracker = TopicTracker()
        self.performance_evaluator = PerformanceEvaluator()
        self.satisfaction_predictor = SatisfactionPredictor()
        self.escalation_detector = EscalationDetector()
        self.summarizer = ConversationSummarizer()

        # Analysis cache
        self._analysis_cache: Dict[str, ConversationAnalysis] = {}

    # -------------------------------------------------------------------------
    # Real-time Analysis
    # -------------------------------------------------------------------------

    def analyze_utterance(self, utterance: Utterance) -> Dict[str, Any]:
        """
        Analyze a single utterance in real-time.

        Returns analysis results for immediate use.
        """
        # Sentiment
        sentiment = self.sentiment_analyzer.analyze(
            utterance.text, utterance.speaker
        )
        utterance.sentiment = sentiment

        # Emotion
        emotion = self.emotion_detector.detect(
            utterance.text, utterance.speaker
        )
        utterance.emotion = emotion

        # Intent
        intent = self.intent_classifier.classify(
            utterance.text, utterance.speaker
        )
        utterance.intent = intent

        # Entities
        entities = self.entity_extractor.extract(
            utterance.text, utterance.speaker
        )
        utterance.entities = entities

        return {
            "utterance_id": utterance.id,
            "sentiment": sentiment.to_dict(),
            "emotion": emotion.to_dict(),
            "intent": intent.to_dict(),
            "entities": [e.to_dict() for e in entities],
        }

    def check_escalation(
        self,
        utterances: List[Utterance],
    ) -> EscalationAssessment:
        """
        Check if escalation is needed (real-time check).
        """
        # Quick sentiment check
        if utterances:
            recent = utterances[-3:]  # Check last 3 utterances
            sentiment = self.sentiment_analyzer.analyze_conversation(recent)
            customer_emotions = self.emotion_detector.build_emotional_profile(
                recent, "customer"
            )
            return self.escalation_detector.detect(
                recent, sentiment, customer_emotions
            )

        return EscalationAssessment()

    # -------------------------------------------------------------------------
    # Full Conversation Analysis
    # -------------------------------------------------------------------------

    async def analyze_conversation(
        self,
        conversation_id: str,
        organization_id: str,
        utterances: List[Utterance],
    ) -> ConversationAnalysis:
        """
        Perform full analysis of a conversation.
        """
        analysis = ConversationAnalysis(
            id="",
            conversation_id=conversation_id,
            organization_id=organization_id,
        )

        # Analyze each utterance
        for utterance in utterances:
            self.analyze_utterance(utterance)

        # Conversation-level analysis
        analysis.sentiment = self.sentiment_analyzer.analyze_conversation(utterances)

        analysis.customer_emotions = self.emotion_detector.build_emotional_profile(
            utterances, "customer"
        )
        analysis.agent_emotions = self.emotion_detector.build_emotional_profile(
            utterances, "agent"
        )

        analysis.intent_flow = self.intent_classifier.classify_conversation(utterances)

        analysis.entities = self.entity_extractor.extract_from_conversation(utterances)

        analysis.context = self.topic_tracker.track_conversation(utterances)

        analysis.agent_performance = self.performance_evaluator.evaluate(
            utterances, analysis.sentiment
        )

        analysis.escalation_assessment = self.escalation_detector.detect(
            utterances, analysis.sentiment, analysis.customer_emotions
        )

        analysis.satisfaction_prediction = self.satisfaction_predictor.predict(
            analysis.sentiment,
            analysis.customer_emotions,
            analysis.intent_flow,
            analysis.escalation_assessment,
        )

        analysis.summary = self.summarizer.summarize(
            utterances,
            analysis.context,
            analysis.sentiment,
            analysis.intent_flow,
            analysis.entities,
        )

        # Calculate metadata
        analysis.turn_count = len(utterances)
        analysis.word_count = sum(len(u.text.split()) for u in utterances)
        if utterances:
            first = min(u.timestamp for u in utterances)
            last = max(u.timestamp for u in utterances)
            analysis.duration_seconds = (last - first).total_seconds()

        analysis.analysis_completed_at = datetime.utcnow()

        # Cache the analysis
        self._analysis_cache[conversation_id] = analysis

        return analysis

    def get_cached_analysis(
        self, conversation_id: str
    ) -> Optional[ConversationAnalysis]:
        """Get cached analysis for a conversation."""
        return self._analysis_cache.get(conversation_id)

    # -------------------------------------------------------------------------
    # Individual Analysis Methods
    # -------------------------------------------------------------------------

    def get_sentiment(
        self, utterances: List[Utterance]
    ) -> ConversationSentiment:
        """Get conversation sentiment analysis."""
        return self.sentiment_analyzer.analyze_conversation(utterances)

    def get_emotions(
        self, utterances: List[Utterance], speaker: str
    ) -> EmotionalProfile:
        """Get emotional profile for a speaker."""
        return self.emotion_detector.build_emotional_profile(utterances, speaker)

    def get_intents(self, utterances: List[Utterance]) -> IntentFlow:
        """Get intent flow for conversation."""
        return self.intent_classifier.classify_conversation(utterances)

    def get_entities(self, utterances: List[Utterance]) -> EntityCollection:
        """Get extracted entities."""
        return self.entity_extractor.extract_from_conversation(utterances)

    def get_agent_performance(
        self, utterances: List[Utterance]
    ) -> AgentPerformanceScore:
        """Get agent performance score."""
        sentiment = self.sentiment_analyzer.analyze_conversation(utterances)
        return self.performance_evaluator.evaluate(utterances, sentiment)

    def get_satisfaction_prediction(
        self, utterances: List[Utterance]
    ) -> SatisfactionPrediction:
        """Get customer satisfaction prediction."""
        sentiment = self.sentiment_analyzer.analyze_conversation(utterances)
        emotions = self.emotion_detector.build_emotional_profile(utterances, "customer")
        intents = self.intent_classifier.classify_conversation(utterances)
        escalation = self.escalation_detector.detect(utterances, sentiment, emotions)

        return self.satisfaction_predictor.predict(
            sentiment, emotions, intents, escalation
        )

    def get_summary(self, utterances: List[Utterance]) -> ConversationSummary:
        """Get conversation summary."""
        context = self.topic_tracker.track_conversation(utterances)
        sentiment = self.sentiment_analyzer.analyze_conversation(utterances)
        intents = self.intent_classifier.classify_conversation(utterances)
        entities = self.entity_extractor.extract_from_conversation(utterances)

        return self.summarizer.summarize(
            utterances, context, sentiment, intents, entities
        )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Types
    "Utterance",
    # Analyzers
    "SentimentAnalyzer",
    "EmotionDetector",
    "IntentClassifier",
    "EntityExtractor",
    "TopicTracker",
    "PerformanceEvaluator",
    "SatisfactionPredictor",
    "EscalationDetector",
    "ConversationSummarizer",
    # Main Service
    "ConversationIntelligenceService",
]
