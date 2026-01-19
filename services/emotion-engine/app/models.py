"""
Data Models for Emotion Engine Service.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import uuid

from pydantic import BaseModel, Field

from .config import EmotionCategory, ArousalLevel, ValenceLevel


# =============================================================================
# Prosodic Features
# =============================================================================


@dataclass
class ProsodicsFeatures:
    """Extracted prosodic features from audio."""

    timestamp_ms: float

    # Pitch (fundamental frequency)
    pitch_mean_hz: float = 0.0
    pitch_std_hz: float = 0.0
    pitch_min_hz: float = 0.0
    pitch_max_hz: float = 0.0
    pitch_range_hz: float = 0.0
    pitch_slope: float = 0.0  # Rising/falling intonation

    # Energy/Intensity
    energy_mean: float = 0.0
    energy_std: float = 0.0
    energy_max: float = 0.0
    loudness_db: float = 0.0

    # Rhythm/Timing
    speaking_rate_sps: float = 0.0  # Syllables per second
    pause_ratio: float = 0.0
    voiced_ratio: float = 0.0

    # Spectral
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_rolloff: float = 0.0
    zero_crossing_rate: float = 0.0

    # Voice quality
    jitter: float = 0.0  # Pitch variation
    shimmer: float = 0.0  # Amplitude variation
    hnr_db: float = 0.0  # Harmonics-to-noise ratio

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "pitch_mean_hz": round(self.pitch_mean_hz, 2),
            "pitch_std_hz": round(self.pitch_std_hz, 2),
            "pitch_range_hz": round(self.pitch_range_hz, 2),
            "pitch_slope": round(self.pitch_slope, 3),
            "energy_mean": round(self.energy_mean, 4),
            "loudness_db": round(self.loudness_db, 2),
            "speaking_rate_sps": round(self.speaking_rate_sps, 2),
            "pause_ratio": round(self.pause_ratio, 3),
            "spectral_centroid": round(self.spectral_centroid, 2),
            "jitter": round(self.jitter, 4),
            "shimmer": round(self.shimmer, 4),
            "hnr_db": round(self.hnr_db, 2),
        }


# =============================================================================
# Emotion Detection
# =============================================================================


@dataclass
class EmotionScore:
    """Score for a single emotion category."""

    category: EmotionCategory
    confidence: float  # 0.0 to 1.0
    arousal: ArousalLevel = ArousalLevel.MEDIUM
    valence: ValenceLevel = ValenceLevel.NEUTRAL
    intensity: float = 0.5  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "confidence": round(self.confidence, 3),
            "arousal": self.arousal.value,
            "valence": self.valence.value,
            "intensity": round(self.intensity, 3),
        }


@dataclass
class EmotionPrediction:
    """Emotion prediction result."""

    timestamp_ms: float

    # Primary emotion
    primary_emotion: EmotionCategory = EmotionCategory.NEUTRAL
    primary_confidence: float = 0.0

    # All emotion scores (sorted by confidence)
    all_scores: List[EmotionScore] = field(default_factory=list)

    # Legacy alias for backwards compatibility
    @property
    def scores(self) -> List[EmotionScore]:
        return self.all_scores

    # Dimensional emotions (-1 to 1 range)
    arousal: float = 0.0  # -1.0 (calm) to 1.0 (excited)
    valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    dominance: float = 0.0  # -1.0 (submissive) to 1.0 (dominant)

    # Features used for classification
    features_used: List[str] = field(default_factory=list)

    # Voice characteristics
    is_speech: bool = True
    voice_stress_level: float = 0.0  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_emotion": self.primary_emotion.value,
            "primary_confidence": round(self.primary_confidence, 3),
            "all_scores": [s.to_dict() for s in self.all_scores[:5]],  # Top 5
            "arousal": round(self.arousal, 3),
            "valence": round(self.valence, 3),
            "dominance": round(self.dominance, 3),
            "features_used": self.features_used,
            "voice_stress_level": round(self.voice_stress_level, 3),
        }


# =============================================================================
# Emotional State
# =============================================================================


@dataclass
class EmotionalShift:
    """Represents a significant emotional shift."""

    timestamp_ms: float
    from_emotion: EmotionCategory
    to_emotion: EmotionCategory
    arousal_change: float
    valence_change: float
    magnitude: float
    direction: str  # positive, negative, escalation, de-escalation, lateral

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "from_emotion": self.from_emotion.value,
            "to_emotion": self.to_emotion.value,
            "arousal_change": round(self.arousal_change, 3),
            "valence_change": round(self.valence_change, 3),
            "magnitude": round(self.magnitude, 3),
            "direction": self.direction,
        }


@dataclass
class EmotionalState:
    """Current emotional state of a speaker with tracking context."""

    session_id: str = ""
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)

    # Current emotion (smoothed)
    current_emotion: EmotionCategory = EmotionCategory.NEUTRAL
    current_confidence: float = 0.0

    # Smoothed dimensional values (-1 to 1)
    smoothed_arousal: float = 0.0
    smoothed_valence: float = 0.0

    # Trends (slope per second)
    arousal_trend: float = 0.0
    valence_trend: float = 0.0

    # Stability (0 to 1, higher = more stable)
    stability: float = 1.0

    # Baseline reference
    baseline_arousal: Optional[float] = None
    baseline_valence: Optional[float] = None
    deviation_from_baseline: float = 0.0

    # Recent shift
    recent_shift: Optional[EmotionalShift] = None

    # Conversation trajectory
    conversation_trajectory: str = "establishing"

    # Full emotion distribution (smoothed)
    emotion_distribution: Dict[EmotionCategory, float] = field(default_factory=dict)

    # Legacy compatibility
    state_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def emotion_confidence(self) -> float:
        """Alias for backwards compatibility."""
        return self.current_confidence

    @property
    def arousal_value(self) -> float:
        """Alias for backwards compatibility."""
        return (self.smoothed_arousal + 1) / 2  # Convert to 0-1 range

    @property
    def valence_value(self) -> float:
        """Alias for backwards compatibility."""
        return (self.smoothed_valence + 1) / 2  # Convert to 0-1 range

    @property
    def state_stability(self) -> float:
        """Alias for backwards compatibility."""
        return self.stability

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state_id": self.state_id,
            "timestamp_ms": self.timestamp_ms,
            "current_emotion": self.current_emotion.value,
            "current_confidence": round(self.current_confidence, 3),
            "dimensional": {
                "arousal": round(self.smoothed_arousal, 3),
                "valence": round(self.smoothed_valence, 3),
            },
            "trends": {
                "arousal": round(self.arousal_trend, 4),
                "valence": round(self.valence_trend, 4),
            },
            "stability": round(self.stability, 3),
            "baseline": {
                "arousal": round(self.baseline_arousal, 3) if self.baseline_arousal else None,
                "valence": round(self.baseline_valence, 3) if self.baseline_valence else None,
                "deviation": round(self.deviation_from_baseline, 3),
            },
            "recent_shift": self.recent_shift.to_dict() if self.recent_shift else None,
            "trajectory": self.conversation_trajectory,
            "emotion_distribution": {
                k.value: round(v, 3) for k, v in self.emotion_distribution.items()
            },
        }


# =============================================================================
# Response Recommendations
# =============================================================================


class ResponseTone(str, Enum):
    """Recommended response tone."""

    EMPATHETIC = "empathetic"
    SUPPORTIVE = "supportive"
    CALMING = "calming"
    ENERGETIC = "energetic"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    REASSURING = "reassuring"
    PATIENT = "patient"


@dataclass
class ToneGuidance:
    """Detailed tone guidance for response generation."""

    # Warmth level
    warmth: str = "neutral"  # cold, formal, neutral, pleasant_professional, warm_friendly
    warmth_value: float = 0.5  # -1 to 1

    # Energy level
    energy: str = "moderate"  # calm, moderate, high_energy
    energy_value: float = 0.5  # 0 to 1

    # Formality
    formality: str = "semi_formal"  # casual, semi_formal, formal
    formality_value: float = 0.5  # 0 to 1

    # Key emotional dimensions
    empathy_level: float = 0.5  # 0 to 1
    directness_level: float = 0.5  # 0 to 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warmth": {"level": self.warmth, "value": round(self.warmth_value, 3)},
            "energy": {"level": self.energy, "value": round(self.energy_value, 3)},
            "formality": {"level": self.formality, "value": round(self.formality_value, 3)},
            "empathy_level": round(self.empathy_level, 3),
            "directness_level": round(self.directness_level, 3),
        }


@dataclass
class StyleAdjustment:
    """TTS style adjustments for emotional congruence."""

    # Voice modifiers (-1 to 1)
    speaking_rate_modifier: float = 0.0  # Slower to faster
    pitch_modifier: float = 0.0  # Lower to higher
    volume_modifier: float = 0.0  # Quieter to louder

    # Pacing
    pause_frequency: str = "normal"  # reduced, normal, increased
    emphasis_level: str = "normal"  # reduced, normal, high

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaking_rate_modifier": round(self.speaking_rate_modifier, 3),
            "pitch_modifier": round(self.pitch_modifier, 3),
            "volume_modifier": round(self.volume_modifier, 3),
            "pause_frequency": self.pause_frequency,
            "emphasis_level": self.emphasis_level,
        }


@dataclass
class ResponseRecommendation:
    """Comprehensive response recommendation based on emotional state."""

    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)

    # Strategy
    primary_strategy: str = "neutral"
    secondary_strategy: Optional[str] = None

    # Detailed guidance
    tone_guidance: Optional[ToneGuidance] = None
    style_adjustments: Optional[StyleAdjustment] = None

    # Actions and phrases
    priority_actions: List[str] = field(default_factory=list)
    suggested_phrases: List[str] = field(default_factory=list)
    phrases_to_avoid: List[str] = field(default_factory=list)

    # Urgency
    urgency_level: str = "normal"  # low, normal, medium, high

    # Meta
    confidence: float = 0.5
    reasoning: str = ""

    # Legacy fields for backwards compatibility
    recommended_tone: ResponseTone = ResponseTone.PROFESSIONAL
    tone_confidence: float = 0.5

    @property
    def empathy_phrases(self) -> List[str]:
        """Alias for backwards compatibility."""
        return self.suggested_phrases

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "strategy": {
                "primary": self.primary_strategy,
                "secondary": self.secondary_strategy,
            },
            "tone_guidance": self.tone_guidance.to_dict() if self.tone_guidance else None,
            "style_adjustments": self.style_adjustments.to_dict() if self.style_adjustments else None,
            "actions": {
                "priority": self.priority_actions[:5],
                "suggested_phrases": self.suggested_phrases[:5],
                "avoid": self.phrases_to_avoid[:5],
            },
            "urgency": self.urgency_level,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
        }


# =============================================================================
# Engine Result
# =============================================================================


@dataclass
class EmotionEngineResult:
    """Complete result from the emotion engine pipeline."""

    session_id: str
    timestamp_ms: float

    # Pipeline outputs
    features: ProsodicsFeatures
    prediction: EmotionPrediction
    state: EmotionalState
    recommendation: Optional[ResponseRecommendation] = None

    # Performance
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp_ms": self.timestamp_ms,
            "features": self.features.to_dict(),
            "prediction": self.prediction.to_dict(),
            "state": self.state.to_dict(),
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# =============================================================================
# Session Models
# =============================================================================


@dataclass
class EmotionEvent:
    """A significant emotional event in the session."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)
    event_type: str = "emotion_change"  # emotion_change, stress_peak, engagement_drop
    description: str = ""
    emotion: Optional[EmotionCategory] = None
    severity: float = 0.5  # 0.0 to 1.0


@dataclass
class SessionEmotionContext:
    """Emotional context for a session."""

    session_id: str
    started_at: float = field(default_factory=time.time)

    # Current state
    current_state: EmotionalState = field(default_factory=EmotionalState)

    # History
    emotion_history: List[EmotionPrediction] = field(default_factory=list)
    state_history: List[EmotionalState] = field(default_factory=list)
    events: List[EmotionEvent] = field(default_factory=list)

    # Aggregates
    dominant_emotion: EmotionCategory = EmotionCategory.NEUTRAL
    average_valence: float = 0.5
    average_arousal: float = 0.5
    peak_stress: float = 0.0
    emotion_variability: float = 0.0

    # Counts
    total_speech_duration_s: float = 0.0
    emotion_change_count: int = 0


# =============================================================================
# API Models
# =============================================================================


class AnalyzeRequest(BaseModel):
    """Request to analyze audio for emotion."""

    audio_base64: str = Field(..., description="Base64-encoded audio data")
    sample_rate: int = Field(default=16000, description="Sample rate")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    include_prosodics: bool = Field(default=True, description="Include prosodic features")
    include_recommendations: bool = Field(default=True, description="Include response recommendations")


class AnalyzeResponse(BaseModel):
    """Response from emotion analysis."""

    prediction: Dict[str, Any]
    prosodics: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, Any]] = None
    recommendation: Optional[Dict[str, Any]] = None
    processing_time_ms: float


class SessionStateResponse(BaseModel):
    """Response with full session emotional state."""

    session_id: str
    current_state: Dict[str, Any]
    recent_emotions: List[Dict[str, Any]]
    summary: Dict[str, Any]


class CreateSessionRequest(BaseModel):
    """Request to create an emotion tracking session."""

    session_id: Optional[str] = Field(None, description="Optional custom session ID")
    agent_id: Optional[str] = Field(None, description="Associated agent ID")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Health Models
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    uptime_seconds: float
    active_sessions: int
    analysis_mode: str


# =============================================================================
# Export
# =============================================================================


__all__ = [
    # Prosodics
    "ProsodicsFeatures",
    # Emotion
    "EmotionScore",
    "EmotionPrediction",
    # State
    "EmotionalShift",
    "EmotionalState",
    "EmotionEvent",
    "SessionEmotionContext",
    # Response
    "ResponseTone",
    "ToneGuidance",
    "StyleAdjustment",
    "ResponseRecommendation",
    # Engine
    "EmotionEngineResult",
    # API
    "AnalyzeRequest",
    "AnalyzeResponse",
    "SessionStateResponse",
    "CreateSessionRequest",
    "HealthResponse",
]
