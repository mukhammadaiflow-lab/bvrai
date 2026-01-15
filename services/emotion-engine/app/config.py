"""
Configuration for Emotion Engine Service.
"""

from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmotionCategory(str, Enum):
    """Primary emotion categories."""

    # Primary emotions (Ekman)
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"

    # Secondary emotions
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"
    EXCITED = "excited"
    BORED = "bored"

    # Nuanced emotions
    ANXIOUS = "anxious"
    CONTENT = "content"
    EMPATHETIC = "empathetic"
    URGENT = "urgent"

    # Neutral
    NEUTRAL = "neutral"


class ArousalLevel(str, Enum):
    """Arousal/energy level."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ValenceLevel(str, Enum):
    """Valence (positive/negative) level."""

    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    SLIGHTLY_NEGATIVE = "slightly_negative"
    NEUTRAL = "neutral"
    SLIGHTLY_POSITIVE = "slightly_positive"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class AnalysisMode(str, Enum):
    """Analysis mode options."""

    REALTIME = "realtime"          # Sub-100ms latency, streaming
    BALANCED = "balanced"          # Balance speed and accuracy
    HIGH_ACCURACY = "high_accuracy"  # Maximum accuracy, higher latency


class ProsodicsConfig(BaseSettings):
    """Configuration for prosodic analysis."""

    model_config = SettingsConfigDict(env_prefix="PROSODICS_")

    # Window sizes (in milliseconds)
    frame_size_ms: int = Field(default=25, description="Frame size for feature extraction")
    hop_size_ms: int = Field(default=10, description="Hop size between frames")
    analysis_window_ms: int = Field(default=500, description="Window for aggregate analysis")

    # Pitch analysis
    pitch_min_hz: float = Field(default=50.0, description="Minimum pitch frequency")
    pitch_max_hz: float = Field(default=500.0, description="Maximum pitch frequency")
    pitch_threshold: float = Field(default=0.3, description="Pitch detection threshold")

    # Energy analysis
    energy_smoothing: float = Field(default=0.9, description="Energy smoothing factor")

    # Speaking rate
    syllable_threshold: float = Field(default=0.1, description="Threshold for syllable detection")


class EmotionModelConfig(BaseSettings):
    """Configuration for emotion classification model."""

    model_config = SettingsConfigDict(env_prefix="EMOTION_MODEL_")

    # Model selection
    model_type: str = Field(
        default="heuristic",
        description="Model type: heuristic, ml, neural",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Path to trained model (for ml/neural)",
    )

    # Classification
    num_classes: int = Field(default=15, description="Number of emotion classes")
    confidence_threshold: float = Field(
        default=0.3,
        description="Minimum confidence for emotion detection",
    )
    smoothing_window: int = Field(
        default=5,
        description="Number of frames to smooth predictions",
    )

    # Feature extraction
    use_mfcc: bool = Field(default=True, description="Use MFCC features")
    use_pitch: bool = Field(default=True, description="Use pitch features")
    use_energy: bool = Field(default=True, description="Use energy features")
    use_spectral: bool = Field(default=True, description="Use spectral features")

    n_mfcc: int = Field(default=13, description="Number of MFCC coefficients")


class ContextConfig(BaseSettings):
    """Configuration for emotional context tracking."""

    model_config = SettingsConfigDict(env_prefix="CONTEXT_")

    # Temporal aggregation
    short_term_window_s: float = Field(
        default=5.0,
        description="Short-term context window (seconds)",
    )
    long_term_window_s: float = Field(
        default=60.0,
        description="Long-term context window (seconds)",
    )

    # Smoothing
    smoothing_window_ms: float = Field(
        default=5000.0,
        description="Sliding window for emotion smoothing (ms)",
    )
    ema_alpha: float = Field(
        default=0.3,
        description="Exponential moving average alpha (0-1)",
    )

    # Baseline establishment
    baseline_turns: int = Field(
        default=5,
        description="Number of turns to establish emotional baseline",
    )

    # State tracking
    state_change_threshold: float = Field(
        default=0.3,
        description="Threshold for significant state change",
    )
    shift_threshold: float = Field(
        default=0.4,
        description="Threshold for emotional shift detection",
    )
    trajectory_smoothing: float = Field(
        default=0.7,
        description="Smoothing for emotional trajectory",
    )

    # Memory
    max_history_events: int = Field(
        default=100,
        description="Maximum emotional events to keep in history",
    )


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Service configuration
    service_name: str = Field(default="emotion-engine", description="Service name")
    host: str = Field(default="0.0.0.0", description="Host to bind")
    port: int = Field(default=8090, ge=1024, le=65535, description="Port")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="info", description="Log level")

    # Analysis mode
    analysis_mode: AnalysisMode = Field(
        default=AnalysisMode.REALTIME,
        description="Analysis mode",
    )

    # Audio settings
    sample_rate: int = Field(default=16000, description="Expected sample rate")
    channels: int = Field(default=1, description="Expected channels")

    # Processing
    max_concurrent_sessions: int = Field(
        default=1000,
        description="Maximum concurrent sessions",
    )
    processing_timeout_ms: int = Field(
        default=100,
        description="Maximum processing time per chunk",
    )
    min_analysis_duration_ms: float = Field(
        default=100.0,
        description="Minimum audio duration for analysis (ms)",
    )

    # Redis (for caching/pub-sub)
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")

    # Sub-configurations
    prosodics: ProsodicsConfig = Field(default_factory=ProsodicsConfig)
    emotion_model: EmotionModelConfig = Field(default_factory=EmotionModelConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()


settings = get_settings()
