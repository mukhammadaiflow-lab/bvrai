"""
Conversation Intelligence Base Types

This module defines the core types and data structures for the
Conversation Intelligence Engine, which provides real-time analysis
of voice conversations including sentiment, intent, emotion, and more.

Key Features:
- Real-time sentiment analysis
- Intent detection and classification
- Emotion recognition
- Topic tracking and context management
- Entity extraction (NER)
- Conversation summarization
- Agent performance scoring
- Customer satisfaction prediction
- Escalation detection
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# =============================================================================
# Sentiment Analysis Types
# =============================================================================


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""

    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class SentimentTrend(str, Enum):
    """Sentiment trend direction."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass
class SentimentScore:
    """Sentiment analysis result."""

    # Core scores
    label: SentimentLabel
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0

    # Detailed breakdown
    positive_score: float = 0.0
    negative_score: float = 0.0
    neutral_score: float = 0.0

    # Context
    speaker: str = ""  # "agent" or "customer"
    text: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label": self.label.value,
            "score": self.score,
            "confidence": self.confidence,
            "positive_score": self.positive_score,
            "negative_score": self.negative_score,
            "neutral_score": self.neutral_score,
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConversationSentiment:
    """Aggregate sentiment for a conversation."""

    # Overall scores
    overall_score: float  # -1.0 to 1.0
    overall_label: SentimentLabel

    # Per-speaker breakdown
    customer_sentiment: float = 0.0
    agent_sentiment: float = 0.0

    # Trend analysis
    trend: SentimentTrend = SentimentTrend.STABLE
    trend_delta: float = 0.0  # Change from start to end

    # Timeline
    sentiment_timeline: List[SentimentScore] = field(default_factory=list)

    # Key moments
    most_positive_moment: Optional[SentimentScore] = None
    most_negative_moment: Optional[SentimentScore] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "overall_label": self.overall_label.value,
            "customer_sentiment": self.customer_sentiment,
            "agent_sentiment": self.agent_sentiment,
            "trend": self.trend.value,
            "trend_delta": self.trend_delta,
            "timeline_count": len(self.sentiment_timeline),
        }


# =============================================================================
# Emotion Recognition Types
# =============================================================================


class EmotionType(str, Enum):
    """Emotion types detected in speech."""

    # Primary emotions
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"

    # Secondary emotions
    FRUSTRATION = "frustration"
    CONFUSION = "confusion"
    SATISFACTION = "satisfaction"
    DISAPPOINTMENT = "disappointment"
    EXCITEMENT = "excitement"
    ANXIETY = "anxiety"
    RELIEF = "relief"
    IMPATIENCE = "impatience"

    # Neutral
    NEUTRAL = "neutral"


@dataclass
class EmotionScore:
    """Emotion detection result."""

    emotion: EmotionType
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0

    # Context
    speaker: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Additional emotions detected (secondary)
    secondary_emotions: List[Tuple[EmotionType, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "emotion": self.emotion.value,
            "score": self.score,
            "confidence": self.confidence,
            "speaker": self.speaker,
            "timestamp": self.timestamp.isoformat(),
            "secondary_emotions": [
                {"emotion": e.value, "score": s}
                for e, s in self.secondary_emotions
            ],
        }


@dataclass
class EmotionalProfile:
    """Emotional profile for a conversation participant."""

    speaker: str  # "agent" or "customer"

    # Dominant emotion
    dominant_emotion: EmotionType = EmotionType.NEUTRAL
    dominant_score: float = 0.0

    # Emotion distribution
    emotion_distribution: Dict[EmotionType, float] = field(default_factory=dict)

    # Emotional volatility (variance in emotions)
    volatility: float = 0.0  # 0.0 = stable, 1.0 = very volatile

    # Timeline
    emotion_timeline: List[EmotionScore] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "speaker": self.speaker,
            "dominant_emotion": self.dominant_emotion.value,
            "dominant_score": self.dominant_score,
            "emotion_distribution": {
                e.value: s for e, s in self.emotion_distribution.items()
            },
            "volatility": self.volatility,
            "timeline_count": len(self.emotion_timeline),
        }


# =============================================================================
# Intent Classification Types
# =============================================================================


class IntentCategory(str, Enum):
    """High-level intent categories."""

    # Information seeking
    QUESTION = "question"
    INQUIRY = "inquiry"
    CLARIFICATION = "clarification"

    # Actions
    REQUEST = "request"
    COMMAND = "command"
    BOOKING = "booking"
    CANCELLATION = "cancellation"
    MODIFICATION = "modification"

    # Support
    COMPLAINT = "complaint"
    FEEDBACK = "feedback"
    ESCALATION = "escalation"
    TECHNICAL_ISSUE = "technical_issue"

    # Transactional
    PURCHASE = "purchase"
    REFUND = "refund"
    BILLING = "billing"
    ACCOUNT = "account"

    # Conversational
    GREETING = "greeting"
    GOODBYE = "goodbye"
    THANKS = "thanks"
    ACKNOWLEDGMENT = "acknowledgment"

    # Other
    OTHER = "other"
    UNKNOWN = "unknown"


class IntentPriority(str, Enum):
    """Intent priority levels."""

    CRITICAL = "critical"  # Requires immediate attention
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Intent:
    """Detected intent from an utterance."""

    id: str
    category: IntentCategory
    confidence: float  # 0.0 to 1.0

    # Specific intent details
    name: str = ""  # Specific intent name within category
    description: str = ""

    # Priority
    priority: IntentPriority = IntentPriority.MEDIUM
    requires_action: bool = False

    # Context
    utterance: str = ""
    speaker: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Slots/parameters extracted
    slots: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"intent_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "category": self.category.value,
            "confidence": self.confidence,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.value,
            "requires_action": self.requires_action,
            "utterance": self.utterance,
            "speaker": self.speaker,
            "timestamp": self.timestamp.isoformat(),
            "slots": self.slots,
        }


@dataclass
class IntentFlow:
    """Intent flow tracking for a conversation."""

    intents: List[Intent] = field(default_factory=list)

    # Primary intent (most significant)
    primary_intent: Optional[Intent] = None

    # Intent transitions
    transitions: List[Tuple[IntentCategory, IntentCategory]] = field(
        default_factory=list
    )

    # Statistics
    intent_distribution: Dict[IntentCategory, int] = field(default_factory=dict)

    def add_intent(self, intent: Intent) -> None:
        """Add an intent and update tracking."""
        # Track transition
        if self.intents:
            self.transitions.append((self.intents[-1].category, intent.category))

        self.intents.append(intent)

        # Update distribution
        if intent.category not in self.intent_distribution:
            self.intent_distribution[intent.category] = 0
        self.intent_distribution[intent.category] += 1

        # Update primary intent (highest priority or most confident)
        if (
            not self.primary_intent
            or intent.priority.value < self.primary_intent.priority.value
            or (
                intent.priority == self.primary_intent.priority
                and intent.confidence > self.primary_intent.confidence
            )
        ):
            self.primary_intent = intent

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent_count": len(self.intents),
            "primary_intent": self.primary_intent.to_dict()
            if self.primary_intent else None,
            "intent_distribution": {
                k.value: v for k, v in self.intent_distribution.items()
            },
            "transition_count": len(self.transitions),
        }


# =============================================================================
# Entity Extraction Types (NER)
# =============================================================================


class EntityType(str, Enum):
    """Named entity types."""

    # Personal
    PERSON = "person"
    ORGANIZATION = "organization"

    # Location
    LOCATION = "location"
    ADDRESS = "address"
    CITY = "city"
    STATE = "state"
    COUNTRY = "country"

    # Time
    DATE = "date"
    TIME = "time"
    DURATION = "duration"
    DATE_RANGE = "date_range"

    # Contact
    PHONE = "phone"
    EMAIL = "email"
    URL = "url"

    # Business
    PRODUCT = "product"
    SERVICE = "service"
    ORDER_NUMBER = "order_number"
    ACCOUNT_NUMBER = "account_number"
    REFERENCE_NUMBER = "reference_number"

    # Financial
    MONEY = "money"
    CURRENCY = "currency"
    CREDIT_CARD = "credit_card"

    # Measurement
    QUANTITY = "quantity"
    PERCENTAGE = "percentage"

    # Other
    CUSTOM = "custom"


@dataclass
class Entity:
    """Extracted named entity."""

    id: str
    type: EntityType
    value: str
    confidence: float  # 0.0 to 1.0

    # Original text and position
    text: str = ""
    start_pos: int = 0
    end_pos: int = 0

    # Normalized value (if applicable)
    normalized_value: Optional[str] = None

    # Context
    speaker: str = ""
    utterance_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"entity_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "value": self.value,
            "confidence": self.confidence,
            "text": self.text,
            "normalized_value": self.normalized_value,
            "speaker": self.speaker,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EntityCollection:
    """Collection of extracted entities."""

    entities: List[Entity] = field(default_factory=list)

    # Grouped by type
    by_type: Dict[EntityType, List[Entity]] = field(default_factory=dict)

    # Key entities (most relevant/frequent)
    key_entities: List[Entity] = field(default_factory=list)

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the collection."""
        self.entities.append(entity)

        if entity.type not in self.by_type:
            self.by_type[entity.type] = []
        self.by_type[entity.type].append(entity)

    def get_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get entities of a specific type."""
        return self.by_type.get(entity_type, [])

    def get_unique_values(self, entity_type: EntityType) -> Set[str]:
        """Get unique values for an entity type."""
        return {e.value for e in self.get_by_type(entity_type)}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_entities": len(self.entities),
            "by_type": {
                t.value: len(entities)
                for t, entities in self.by_type.items()
            },
            "key_entities": [e.to_dict() for e in self.key_entities[:10]],
        }


# =============================================================================
# Topic and Context Tracking
# =============================================================================


class TopicCategory(str, Enum):
    """Topic categories for conversation tracking."""

    # Product/Service
    PRODUCT_INQUIRY = "product_inquiry"
    SERVICE_INQUIRY = "service_inquiry"
    PRICING = "pricing"
    AVAILABILITY = "availability"

    # Support
    TECHNICAL_SUPPORT = "technical_support"
    BILLING_SUPPORT = "billing_support"
    GENERAL_SUPPORT = "general_support"

    # Account
    ACCOUNT_MANAGEMENT = "account_management"
    SUBSCRIPTION = "subscription"
    CANCELLATION = "cancellation"

    # Orders
    ORDER_STATUS = "order_status"
    ORDER_ISSUE = "order_issue"
    RETURNS = "returns"
    REFUNDS = "refunds"

    # Scheduling
    APPOINTMENT = "appointment"
    SCHEDULING = "scheduling"

    # Feedback
    COMPLAINT = "complaint"
    PRAISE = "praise"
    SUGGESTION = "suggestion"

    # Other
    GENERAL = "general"
    OFF_TOPIC = "off_topic"


@dataclass
class Topic:
    """A topic detected in the conversation."""

    id: str
    name: str
    category: TopicCategory
    confidence: float  # 0.0 to 1.0

    # Duration tracking
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Relevance
    relevance_score: float = 0.0  # How relevant to main conversation
    is_primary: bool = False

    # Keywords associated with this topic
    keywords: List[str] = field(default_factory=list)

    # Utterances related to this topic
    utterance_count: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = f"topic_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "confidence": self.confidence,
            "duration_seconds": self.duration_seconds,
            "relevance_score": self.relevance_score,
            "is_primary": self.is_primary,
            "keywords": self.keywords,
            "utterance_count": self.utterance_count,
        }


@dataclass
class ConversationContext:
    """Conversation context tracking."""

    # Current topic
    current_topic: Optional[Topic] = None

    # Topic history
    topic_history: List[Topic] = field(default_factory=list)

    # Primary topic (most discussed)
    primary_topic: Optional[Topic] = None

    # Entities mentioned
    entities: EntityCollection = field(default_factory=EntityCollection)

    # Key facts/information gathered
    key_facts: List[Dict[str, Any]] = field(default_factory=list)

    # User preferences mentioned
    user_preferences: Dict[str, Any] = field(default_factory=dict)

    # Action items identified
    action_items: List[str] = field(default_factory=list)

    # Questions asked (pending answers)
    pending_questions: List[str] = field(default_factory=list)

    # Questions answered
    answered_questions: List[Dict[str, str]] = field(default_factory=list)

    def switch_topic(self, new_topic: Topic) -> None:
        """Switch to a new topic."""
        if self.current_topic:
            self.current_topic.end_time = datetime.utcnow()
            if self.current_topic.start_time:
                self.current_topic.duration_seconds = (
                    self.current_topic.end_time - self.current_topic.start_time
                ).total_seconds()
            self.topic_history.append(self.current_topic)

        self.current_topic = new_topic

        # Update primary topic
        if not self.primary_topic or (
            new_topic.relevance_score > self.primary_topic.relevance_score
        ):
            self.primary_topic = new_topic

    def add_key_fact(self, key: str, value: Any, source: str = "") -> None:
        """Add a key fact."""
        self.key_facts.append({
            "key": key,
            "value": value,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_topic": self.current_topic.to_dict()
            if self.current_topic else None,
            "topic_count": len(self.topic_history) + (1 if self.current_topic else 0),
            "primary_topic": self.primary_topic.to_dict()
            if self.primary_topic else None,
            "entities": self.entities.to_dict(),
            "key_facts_count": len(self.key_facts),
            "action_items": self.action_items,
            "pending_questions": self.pending_questions,
        }


# =============================================================================
# Agent Performance Types
# =============================================================================


class PerformanceMetricType(str, Enum):
    """Types of agent performance metrics."""

    # Response quality
    RESPONSE_RELEVANCE = "response_relevance"
    RESPONSE_CLARITY = "response_clarity"
    RESPONSE_ACCURACY = "response_accuracy"
    RESPONSE_COMPLETENESS = "response_completeness"

    # Timing
    RESPONSE_TIME = "response_time"
    RESOLUTION_TIME = "resolution_time"

    # Engagement
    EMPATHY = "empathy"
    PROFESSIONALISM = "professionalism"
    HELPFULNESS = "helpfulness"

    # Communication
    ACTIVE_LISTENING = "active_listening"
    CLEAR_COMMUNICATION = "clear_communication"

    # Problem solving
    PROBLEM_IDENTIFICATION = "problem_identification"
    SOLUTION_QUALITY = "solution_quality"


@dataclass
class PerformanceMetric:
    """A single performance metric."""

    metric_type: PerformanceMetricType
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0

    # Evidence
    evidence: List[str] = field(default_factory=list)

    # Feedback
    feedback: str = ""
    improvement_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "score": self.score,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "feedback": self.feedback,
            "improvement_suggestions": self.improvement_suggestions,
        }


@dataclass
class AgentPerformanceScore:
    """Complete agent performance assessment."""

    # Overall score
    overall_score: float  # 0.0 to 100.0
    grade: str = ""  # A, B, C, D, F

    # Individual metrics
    metrics: Dict[PerformanceMetricType, PerformanceMetric] = field(
        default_factory=dict
    )

    # Category scores
    response_quality_score: float = 0.0
    communication_score: float = 0.0
    problem_solving_score: float = 0.0
    empathy_score: float = 0.0

    # Timing metrics
    average_response_time_ms: float = 0.0
    total_speaking_time_seconds: float = 0.0
    talk_to_listen_ratio: float = 0.0

    # Strengths and areas for improvement
    strengths: List[str] = field(default_factory=list)
    improvements_needed: List[str] = field(default_factory=list)

    # Specific feedback
    feedback_summary: str = ""

    def calculate_grade(self) -> str:
        """Calculate letter grade from overall score."""
        if self.overall_score >= 90:
            return "A"
        elif self.overall_score >= 80:
            return "B"
        elif self.overall_score >= 70:
            return "C"
        elif self.overall_score >= 60:
            return "D"
        else:
            return "F"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "grade": self.grade or self.calculate_grade(),
            "metrics": {
                k.value: v.to_dict() for k, v in self.metrics.items()
            },
            "response_quality_score": self.response_quality_score,
            "communication_score": self.communication_score,
            "problem_solving_score": self.problem_solving_score,
            "empathy_score": self.empathy_score,
            "average_response_time_ms": self.average_response_time_ms,
            "talk_to_listen_ratio": self.talk_to_listen_ratio,
            "strengths": self.strengths,
            "improvements_needed": self.improvements_needed,
            "feedback_summary": self.feedback_summary,
        }


# =============================================================================
# Customer Satisfaction Types
# =============================================================================


class SatisfactionLevel(str, Enum):
    """Customer satisfaction levels."""

    VERY_DISSATISFIED = "very_dissatisfied"
    DISSATISFIED = "dissatisfied"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    VERY_SATISFIED = "very_satisfied"


@dataclass
class SatisfactionPrediction:
    """Predicted customer satisfaction."""

    # Prediction
    predicted_level: SatisfactionLevel
    predicted_score: float  # 0.0 to 100.0 (like NPS)
    confidence: float  # 0.0 to 1.0

    # Contributing factors
    positive_factors: List[str] = field(default_factory=list)
    negative_factors: List[str] = field(default_factory=list)

    # Risk assessment
    churn_risk: float = 0.0  # 0.0 to 1.0
    escalation_risk: float = 0.0  # 0.0 to 1.0

    # Recommendations
    retention_recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicted_level": self.predicted_level.value,
            "predicted_score": self.predicted_score,
            "confidence": self.confidence,
            "positive_factors": self.positive_factors,
            "negative_factors": self.negative_factors,
            "churn_risk": self.churn_risk,
            "escalation_risk": self.escalation_risk,
            "retention_recommendations": self.retention_recommendations,
        }


# =============================================================================
# Escalation Detection Types
# =============================================================================


class EscalationReason(str, Enum):
    """Reasons for escalation."""

    # Customer-driven
    CUSTOMER_REQUEST = "customer_request"
    REPEATED_ISSUE = "repeated_issue"
    HIGH_VALUE_CUSTOMER = "high_value_customer"

    # Issue-driven
    COMPLEX_ISSUE = "complex_issue"
    UNRESOLVED_ISSUE = "unresolved_issue"
    POLICY_EXCEPTION = "policy_exception"

    # Sentiment-driven
    NEGATIVE_SENTIMENT = "negative_sentiment"
    FRUSTRATED_CUSTOMER = "frustrated_customer"
    ANGRY_CUSTOMER = "angry_customer"

    # Agent-driven
    AGENT_LIMITATION = "agent_limitation"
    REQUIRES_SUPERVISOR = "requires_supervisor"

    # System-driven
    TECHNICAL_LIMITATION = "technical_limitation"
    SECURITY_CONCERN = "security_concern"


class EscalationPriority(str, Enum):
    """Escalation priority levels."""

    CRITICAL = "critical"  # Immediate attention required
    HIGH = "high"  # Urgent, handle soon
    MEDIUM = "medium"  # Standard escalation
    LOW = "low"  # Can wait


@dataclass
class EscalationSignal:
    """A signal indicating potential need for escalation."""

    reason: EscalationReason
    priority: EscalationPriority
    confidence: float  # 0.0 to 1.0

    # Evidence
    trigger_text: str = ""
    trigger_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Context
    context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reason": self.reason.value,
            "priority": self.priority.value,
            "confidence": self.confidence,
            "trigger_text": self.trigger_text,
            "trigger_timestamp": self.trigger_timestamp.isoformat(),
            "context": self.context,
        }


@dataclass
class EscalationAssessment:
    """Complete escalation assessment."""

    # Should escalate?
    should_escalate: bool = False
    escalation_probability: float = 0.0  # 0.0 to 1.0

    # Priority
    priority: EscalationPriority = EscalationPriority.MEDIUM

    # Signals detected
    signals: List[EscalationSignal] = field(default_factory=list)

    # Primary reason
    primary_reason: Optional[EscalationReason] = None

    # Recommended actions
    recommended_actions: List[str] = field(default_factory=list)

    # Target (who to escalate to)
    recommended_target: str = ""  # "supervisor", "specialist", "manager"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "should_escalate": self.should_escalate,
            "escalation_probability": self.escalation_probability,
            "priority": self.priority.value,
            "signals": [s.to_dict() for s in self.signals],
            "primary_reason": self.primary_reason.value
            if self.primary_reason else None,
            "recommended_actions": self.recommended_actions,
            "recommended_target": self.recommended_target,
        }


# =============================================================================
# Conversation Summary Types
# =============================================================================


@dataclass
class ConversationSummary:
    """Summary of a conversation."""

    # Brief summary
    brief_summary: str = ""  # 1-2 sentences
    detailed_summary: str = ""  # Paragraph

    # Key points
    key_points: List[str] = field(default_factory=list)

    # Outcome
    outcome: str = ""  # "resolved", "escalated", "follow_up_required", etc.
    resolution: str = ""

    # Action items
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_required: bool = False
    follow_up_reason: str = ""

    # Topics discussed
    main_topics: List[str] = field(default_factory=list)

    # Customer info gathered
    customer_info_gathered: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "brief_summary": self.brief_summary,
            "detailed_summary": self.detailed_summary,
            "key_points": self.key_points,
            "outcome": self.outcome,
            "resolution": self.resolution,
            "action_items": self.action_items,
            "follow_up_required": self.follow_up_required,
            "follow_up_reason": self.follow_up_reason,
            "main_topics": self.main_topics,
        }


# =============================================================================
# Complete Conversation Analysis
# =============================================================================


@dataclass
class ConversationAnalysis:
    """Complete analysis of a conversation."""

    id: str
    conversation_id: str
    organization_id: str

    # Timestamps
    analysis_started_at: datetime = field(default_factory=datetime.utcnow)
    analysis_completed_at: Optional[datetime] = None

    # Sentiment analysis
    sentiment: Optional[ConversationSentiment] = None

    # Emotion analysis
    customer_emotions: Optional[EmotionalProfile] = None
    agent_emotions: Optional[EmotionalProfile] = None

    # Intent analysis
    intent_flow: IntentFlow = field(default_factory=IntentFlow)

    # Entity extraction
    entities: EntityCollection = field(default_factory=EntityCollection)

    # Context
    context: ConversationContext = field(default_factory=ConversationContext)

    # Agent performance
    agent_performance: Optional[AgentPerformanceScore] = None

    # Customer satisfaction
    satisfaction_prediction: Optional[SatisfactionPrediction] = None

    # Escalation
    escalation_assessment: Optional[EscalationAssessment] = None

    # Summary
    summary: Optional[ConversationSummary] = None

    # Metadata
    duration_seconds: float = 0.0
    turn_count: int = 0
    word_count: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = f"analysis_{uuid.uuid4().hex[:14]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "organization_id": self.organization_id,
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "customer_emotions": self.customer_emotions.to_dict()
            if self.customer_emotions else None,
            "agent_emotions": self.agent_emotions.to_dict()
            if self.agent_emotions else None,
            "intent_flow": self.intent_flow.to_dict(),
            "entities": self.entities.to_dict(),
            "context": self.context.to_dict(),
            "agent_performance": self.agent_performance.to_dict()
            if self.agent_performance else None,
            "satisfaction_prediction": self.satisfaction_prediction.to_dict()
            if self.satisfaction_prediction else None,
            "escalation_assessment": self.escalation_assessment.to_dict()
            if self.escalation_assessment else None,
            "summary": self.summary.to_dict() if self.summary else None,
            "duration_seconds": self.duration_seconds,
            "turn_count": self.turn_count,
            "word_count": self.word_count,
        }


# =============================================================================
# Exceptions
# =============================================================================


class ConversationIntelligenceError(Exception):
    """Base exception for conversation intelligence errors."""
    pass


class AnalysisError(ConversationIntelligenceError):
    """Error during analysis."""
    pass


class ModelNotLoadedError(ConversationIntelligenceError):
    """ML model not loaded."""
    pass


class InsufficientDataError(ConversationIntelligenceError):
    """Not enough data for analysis."""
    pass


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Sentiment types
    "SentimentLabel",
    "SentimentTrend",
    "SentimentScore",
    "ConversationSentiment",
    # Emotion types
    "EmotionType",
    "EmotionScore",
    "EmotionalProfile",
    # Intent types
    "IntentCategory",
    "IntentPriority",
    "Intent",
    "IntentFlow",
    # Entity types
    "EntityType",
    "Entity",
    "EntityCollection",
    # Topic types
    "TopicCategory",
    "Topic",
    "ConversationContext",
    # Performance types
    "PerformanceMetricType",
    "PerformanceMetric",
    "AgentPerformanceScore",
    # Satisfaction types
    "SatisfactionLevel",
    "SatisfactionPrediction",
    # Escalation types
    "EscalationReason",
    "EscalationPriority",
    "EscalationSignal",
    "EscalationAssessment",
    # Summary types
    "ConversationSummary",
    # Main analysis
    "ConversationAnalysis",
    # Exceptions
    "ConversationIntelligenceError",
    "AnalysisError",
    "ModelNotLoadedError",
    "InsufficientDataError",
]
