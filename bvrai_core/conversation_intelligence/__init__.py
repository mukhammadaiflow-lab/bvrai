"""
Conversation Intelligence Module

This module provides comprehensive real-time conversation analysis
for AI voice agents, including sentiment analysis, emotion recognition,
intent classification, entity extraction, and more.

Key Features:
- Real-time sentiment analysis with trend tracking
- Emotion detection and emotional profiling
- Intent classification with slot extraction
- Named entity recognition (NER)
- Topic tracking and context management
- Agent performance scoring
- Customer satisfaction prediction
- Escalation detection and recommendations
- Automatic conversation summarization

Example usage:

    from bvrai_core.conversation_intelligence import (
        ConversationIntelligenceService,
        Utterance,
        SentimentLabel,
        EmotionType,
        IntentCategory,
    )
    from datetime import datetime

    # Initialize service
    service = ConversationIntelligenceService()

    # Create utterances
    utterances = [
        Utterance(
            id="utt_1",
            text="Hi, I need help with my order",
            speaker="customer",
            timestamp=datetime.now(),
        ),
        Utterance(
            id="utt_2",
            text="Of course! I'd be happy to help. Could you provide your order number?",
            speaker="agent",
            timestamp=datetime.now(),
        ),
        Utterance(
            id="utt_3",
            text="It's ORD-12345. The package hasn't arrived and I'm frustrated.",
            speaker="customer",
            timestamp=datetime.now(),
        ),
    ]

    # Real-time analysis (per utterance)
    for utterance in utterances:
        result = service.analyze_utterance(utterance)
        print(f"Sentiment: {result['sentiment']['label']}")
        print(f"Emotion: {result['emotion']['emotion']}")
        print(f"Intent: {result['intent']['category']}")

    # Check for escalation needs
    escalation = service.check_escalation(utterances)
    if escalation.should_escalate:
        print(f"Escalation needed: {escalation.primary_reason}")

    # Full conversation analysis
    analysis = await service.analyze_conversation(
        conversation_id="conv_123",
        organization_id="org_456",
        utterances=utterances,
    )

    # Access analysis results
    print(f"Overall sentiment: {analysis.sentiment.overall_label}")
    print(f"Customer emotion: {analysis.customer_emotions.dominant_emotion}")
    print(f"Primary intent: {analysis.intent_flow.primary_intent.category}")
    print(f"Agent score: {analysis.agent_performance.overall_score}")
    print(f"Satisfaction: {analysis.satisfaction_prediction.predicted_level}")
    print(f"Summary: {analysis.summary.brief_summary}")

    # Individual analysis methods
    sentiment = service.get_sentiment(utterances)
    emotions = service.get_emotions(utterances, "customer")
    intents = service.get_intents(utterances)
    entities = service.get_entities(utterances)
    performance = service.get_agent_performance(utterances)
    satisfaction = service.get_satisfaction_prediction(utterances)
    summary = service.get_summary(utterances)

Analysis Components:

    Sentiment Analysis:
    - Real-time sentiment scoring (-1 to +1)
    - Sentiment labels (very_negative to very_positive)
    - Trend detection (improving, stable, declining)
    - Per-speaker breakdown
    - Key moment identification

    Emotion Detection:
    - 14 emotion types (joy, sadness, anger, fear, etc.)
    - Emotional profiling per speaker
    - Emotional volatility tracking
    - Secondary emotion detection

    Intent Classification:
    - 20+ intent categories
    - Priority levels (critical, high, medium, low)
    - Intent flow tracking
    - Slot extraction

    Entity Extraction:
    - Phone numbers, emails, URLs
    - Dates, times, durations
    - Money, percentages
    - Order numbers, account numbers
    - Custom entity types

    Performance Evaluation:
    - Overall score (0-100)
    - Letter grade (A-F)
    - Category scores (empathy, communication, etc.)
    - Strengths and improvements

    Satisfaction Prediction:
    - Predicted satisfaction level
    - Churn risk assessment
    - Escalation risk
    - Retention recommendations
"""

# Base types
from .base import (
    # Sentiment types
    SentimentLabel,
    SentimentTrend,
    SentimentScore,
    ConversationSentiment,
    # Emotion types
    EmotionType,
    EmotionScore,
    EmotionalProfile,
    # Intent types
    IntentCategory,
    IntentPriority,
    Intent,
    IntentFlow,
    # Entity types
    EntityType,
    Entity,
    EntityCollection,
    # Topic types
    TopicCategory,
    Topic,
    ConversationContext,
    # Performance types
    PerformanceMetricType,
    PerformanceMetric,
    AgentPerformanceScore,
    # Satisfaction types
    SatisfactionLevel,
    SatisfactionPrediction,
    # Escalation types
    EscalationReason,
    EscalationPriority,
    EscalationSignal,
    EscalationAssessment,
    # Summary types
    ConversationSummary,
    # Main analysis
    ConversationAnalysis,
    # Exceptions
    ConversationIntelligenceError,
    AnalysisError,
    ModelNotLoadedError,
    InsufficientDataError,
)

# Services
from .service import (
    # Types
    Utterance,
    # Analyzers
    SentimentAnalyzer,
    EmotionDetector,
    IntentClassifier,
    EntityExtractor,
    TopicTracker,
    PerformanceEvaluator,
    SatisfactionPredictor,
    EscalationDetector,
    ConversationSummarizer,
    # Main Service
    ConversationIntelligenceService,
)


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
    # Exceptions
    "ConversationIntelligenceError",
    "AnalysisError",
    "ModelNotLoadedError",
    "InsufficientDataError",
]
