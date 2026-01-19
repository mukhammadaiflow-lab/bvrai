"""
ML Pipeline for Conversation Intelligence

Advanced machine learning capabilities:
- Intent classification
- Sentiment analysis
- Entity extraction
- Conversation summarization
- Topic modeling
- Embedding generation
"""

from app.ml.pipeline import (
    MLPipeline,
    PipelineConfig,
    PipelineStage,
    BatchProcessor,
)

from app.ml.intent import (
    IntentClassifier,
    Intent,
    IntentResult,
    TrainingData,
    MultiLabelClassifier,
)

from app.ml.sentiment import (
    SentimentAnalyzer,
    Sentiment,
    SentimentResult,
    EmotionDetector,
    Emotion,
)

from app.ml.entities import (
    EntityExtractor,
    Entity,
    EntityType,
    SpanEntity,
    EntityResult,
    NERModel,
)

from app.ml.embeddings import (
    EmbeddingGenerator,
    EmbeddingModel,
    EmbeddingConfig,
    TextEmbedding,
    SimilaritySearch,
)

from app.ml.summarization import (
    ConversationSummarizer,
    Summary,
    KeyPoint,
    ActionItem,
    ExtractiveSummarizer,
    AbstractiveSummarizer,
)

__all__ = [
    # Pipeline
    "MLPipeline",
    "PipelineConfig",
    "PipelineStage",
    "BatchProcessor",
    # Intent
    "IntentClassifier",
    "Intent",
    "IntentResult",
    "TrainingData",
    "MultiLabelClassifier",
    # Sentiment
    "SentimentAnalyzer",
    "Sentiment",
    "SentimentResult",
    "EmotionDetector",
    "Emotion",
    # Entities
    "EntityExtractor",
    "Entity",
    "EntityType",
    "SpanEntity",
    "EntityResult",
    "NERModel",
    # Embeddings
    "EmbeddingGenerator",
    "EmbeddingModel",
    "EmbeddingConfig",
    "TextEmbedding",
    "SimilaritySearch",
    # Summarization
    "ConversationSummarizer",
    "Summary",
    "KeyPoint",
    "ActionItem",
    "ExtractiveSummarizer",
    "AbstractiveSummarizer",
]
