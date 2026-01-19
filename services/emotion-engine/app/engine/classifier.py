"""
Emotion Classifier.

Maps prosodic features to emotion categories using a multi-layer
classification approach combining rule-based and ML-inspired techniques.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..config import (
    EmotionCategory,
    EmotionModelConfig,
    ArousalLevel,
    ValenceLevel,
    get_settings,
)
from ..models import EmotionScore, EmotionPrediction, ProsodicsFeatures

logger = logging.getLogger(__name__)


@dataclass
class EmotionProfile:
    """Feature profile for an emotion category."""

    # Pitch characteristics (normalized z-scores)
    pitch_mean_range: Tuple[float, float]  # Low to high
    pitch_variability: Tuple[float, float]  # Std range
    pitch_slope_range: Tuple[float, float]  # Rising/falling

    # Energy characteristics
    energy_level: Tuple[float, float]
    energy_variability: Tuple[float, float]

    # Rhythm characteristics
    speaking_rate: Tuple[float, float]
    pause_ratio: Tuple[float, float]

    # Spectral characteristics
    spectral_centroid: Tuple[float, float]

    # Voice quality
    jitter_range: Tuple[float, float]
    shimmer_range: Tuple[float, float]

    # Dimensional position
    arousal: ArousalLevel
    valence: ValenceLevel

    # Weight for this emotion
    prior_weight: float = 1.0


# Emotion profiles based on prosodic research
EMOTION_PROFILES: Dict[EmotionCategory, EmotionProfile] = {
    EmotionCategory.HAPPY: EmotionProfile(
        pitch_mean_range=(0.3, 1.5),
        pitch_variability=(0.4, 1.5),
        pitch_slope_range=(0.0, 1.0),
        energy_level=(0.4, 1.2),
        energy_variability=(0.3, 1.0),
        speaking_rate=(0.3, 1.0),
        pause_ratio=(-1.0, 0.2),
        spectral_centroid=(0.2, 1.0),
        jitter_range=(-0.5, 0.5),
        shimmer_range=(-0.5, 0.5),
        arousal=ArousalLevel.HIGH,
        valence=ValenceLevel.POSITIVE,
        prior_weight=1.0,
    ),
    EmotionCategory.SAD: EmotionProfile(
        pitch_mean_range=(-1.5, -0.2),
        pitch_variability=(-1.0, 0.0),
        pitch_slope_range=(-1.0, 0.0),
        energy_level=(-1.5, -0.3),
        energy_variability=(-1.0, 0.2),
        speaking_rate=(-1.0, -0.2),
        pause_ratio=(0.2, 1.5),
        spectral_centroid=(-1.0, 0.0),
        jitter_range=(0.0, 1.0),
        shimmer_range=(0.0, 1.0),
        arousal=ArousalLevel.LOW,
        valence=ValenceLevel.NEGATIVE,
        prior_weight=1.0,
    ),
    EmotionCategory.ANGRY: EmotionProfile(
        pitch_mean_range=(0.2, 1.5),
        pitch_variability=(0.5, 2.0),
        pitch_slope_range=(-0.5, 0.5),
        energy_level=(0.5, 2.0),
        energy_variability=(0.5, 1.5),
        speaking_rate=(0.2, 1.0),
        pause_ratio=(-1.0, 0.0),
        spectral_centroid=(0.5, 2.0),
        jitter_range=(0.2, 1.5),
        shimmer_range=(0.2, 1.0),
        arousal=ArousalLevel.HIGH,
        valence=ValenceLevel.NEGATIVE,
        prior_weight=1.0,
    ),
    EmotionCategory.FEARFUL: EmotionProfile(
        pitch_mean_range=(0.3, 1.5),
        pitch_variability=(0.5, 2.0),
        pitch_slope_range=(0.0, 1.0),
        energy_level=(-0.5, 0.5),
        energy_variability=(0.5, 1.5),
        speaking_rate=(0.3, 1.5),
        pause_ratio=(-0.5, 0.5),
        spectral_centroid=(0.0, 1.0),
        jitter_range=(0.3, 1.5),
        shimmer_range=(0.3, 1.5),
        arousal=ArousalLevel.HIGH,
        valence=ValenceLevel.NEGATIVE,
        prior_weight=0.8,
    ),
    EmotionCategory.SURPRISED: EmotionProfile(
        pitch_mean_range=(0.5, 2.0),
        pitch_variability=(0.5, 2.0),
        pitch_slope_range=(0.3, 1.5),
        energy_level=(0.3, 1.5),
        energy_variability=(0.5, 1.5),
        speaking_rate=(0.0, 0.8),
        pause_ratio=(-0.5, 0.5),
        spectral_centroid=(0.3, 1.5),
        jitter_range=(-0.5, 0.5),
        shimmer_range=(-0.5, 0.5),
        arousal=ArousalLevel.HIGH,
        valence=ValenceLevel.NEUTRAL,
        prior_weight=0.7,
    ),
    EmotionCategory.DISGUSTED: EmotionProfile(
        pitch_mean_range=(-0.5, 0.3),
        pitch_variability=(-0.5, 0.5),
        pitch_slope_range=(-0.5, 0.0),
        energy_level=(-0.5, 0.5),
        energy_variability=(-0.5, 0.5),
        speaking_rate=(-0.5, 0.3),
        pause_ratio=(0.0, 0.8),
        spectral_centroid=(-0.5, 0.3),
        jitter_range=(0.0, 0.8),
        shimmer_range=(0.0, 0.8),
        arousal=ArousalLevel.MEDIUM,
        valence=ValenceLevel.NEGATIVE,
        prior_weight=0.6,
    ),
    EmotionCategory.NEUTRAL: EmotionProfile(
        pitch_mean_range=(-0.3, 0.3),
        pitch_variability=(-0.5, 0.5),
        pitch_slope_range=(-0.3, 0.3),
        energy_level=(-0.3, 0.3),
        energy_variability=(-0.5, 0.5),
        speaking_rate=(-0.3, 0.3),
        pause_ratio=(-0.3, 0.5),
        spectral_centroid=(-0.3, 0.3),
        jitter_range=(-0.3, 0.3),
        shimmer_range=(-0.3, 0.3),
        arousal=ArousalLevel.MEDIUM,
        valence=ValenceLevel.NEUTRAL,
        prior_weight=1.5,  # Higher prior for neutral
    ),
    EmotionCategory.EXCITED: EmotionProfile(
        pitch_mean_range=(0.5, 2.0),
        pitch_variability=(0.5, 2.0),
        pitch_slope_range=(0.0, 1.0),
        energy_level=(0.5, 2.0),
        energy_variability=(0.5, 1.5),
        speaking_rate=(0.5, 1.5),
        pause_ratio=(-1.0, 0.0),
        spectral_centroid=(0.3, 1.5),
        jitter_range=(-0.3, 0.5),
        shimmer_range=(-0.3, 0.5),
        arousal=ArousalLevel.VERY_HIGH,
        valence=ValenceLevel.POSITIVE,
        prior_weight=0.8,
    ),
    EmotionCategory.FRUSTRATED: EmotionProfile(
        pitch_mean_range=(0.0, 0.8),
        pitch_variability=(0.3, 1.2),
        pitch_slope_range=(-0.3, 0.3),
        energy_level=(0.3, 1.2),
        energy_variability=(0.3, 1.0),
        speaking_rate=(0.0, 0.5),
        pause_ratio=(0.0, 0.5),
        spectral_centroid=(0.0, 0.8),
        jitter_range=(0.2, 1.0),
        shimmer_range=(0.2, 1.0),
        arousal=ArousalLevel.HIGH,
        valence=ValenceLevel.NEGATIVE,
        prior_weight=1.0,
    ),
    EmotionCategory.CONFUSED: EmotionProfile(
        pitch_mean_range=(-0.2, 0.5),
        pitch_variability=(0.3, 1.0),
        pitch_slope_range=(0.2, 1.0),  # Rising intonation
        energy_level=(-0.3, 0.3),
        energy_variability=(0.0, 0.8),
        speaking_rate=(-0.5, 0.0),
        pause_ratio=(0.3, 1.0),
        spectral_centroid=(-0.2, 0.5),
        jitter_range=(0.0, 0.5),
        shimmer_range=(0.0, 0.5),
        arousal=ArousalLevel.MEDIUM,
        valence=ValenceLevel.SLIGHTLY_NEGATIVE,
        prior_weight=0.9,
    ),
    EmotionCategory.BORED: EmotionProfile(
        pitch_mean_range=(-0.8, 0.0),
        pitch_variability=(-1.0, -0.3),
        pitch_slope_range=(-0.5, 0.0),
        energy_level=(-1.0, -0.3),
        energy_variability=(-1.0, -0.3),
        speaking_rate=(-0.8, 0.0),
        pause_ratio=(0.3, 1.0),
        spectral_centroid=(-0.8, 0.0),
        jitter_range=(-0.3, 0.3),
        shimmer_range=(-0.3, 0.3),
        arousal=ArousalLevel.LOW,
        valence=ValenceLevel.SLIGHTLY_NEGATIVE,
        prior_weight=0.7,
    ),
    EmotionCategory.ANXIOUS: EmotionProfile(
        pitch_mean_range=(0.2, 1.0),
        pitch_variability=(0.5, 1.5),
        pitch_slope_range=(0.0, 0.8),
        energy_level=(-0.3, 0.5),
        energy_variability=(0.3, 1.2),
        speaking_rate=(0.3, 1.0),
        pause_ratio=(-0.3, 0.3),
        spectral_centroid=(0.0, 0.8),
        jitter_range=(0.3, 1.2),
        shimmer_range=(0.3, 1.0),
        arousal=ArousalLevel.HIGH,
        valence=ValenceLevel.NEGATIVE,
        prior_weight=0.8,
    ),
    EmotionCategory.CONTENT: EmotionProfile(
        pitch_mean_range=(-0.2, 0.3),
        pitch_variability=(-0.3, 0.3),
        pitch_slope_range=(-0.2, 0.2),
        energy_level=(-0.2, 0.3),
        energy_variability=(-0.5, 0.3),
        speaking_rate=(-0.3, 0.2),
        pause_ratio=(-0.2, 0.5),
        spectral_centroid=(-0.2, 0.3),
        jitter_range=(-0.5, 0.2),
        shimmer_range=(-0.5, 0.2),
        arousal=ArousalLevel.LOW,
        valence=ValenceLevel.POSITIVE,
        prior_weight=0.9,
    ),
    EmotionCategory.EMPATHETIC: EmotionProfile(
        pitch_mean_range=(-0.3, 0.3),
        pitch_variability=(0.0, 0.8),
        pitch_slope_range=(-0.2, 0.3),
        energy_level=(-0.3, 0.3),
        energy_variability=(0.0, 0.5),
        speaking_rate=(-0.5, 0.0),
        pause_ratio=(0.0, 0.5),
        spectral_centroid=(-0.3, 0.3),
        jitter_range=(-0.3, 0.3),
        shimmer_range=(-0.3, 0.3),
        arousal=ArousalLevel.MEDIUM,
        valence=ValenceLevel.SLIGHTLY_POSITIVE,
        prior_weight=0.6,
    ),
    EmotionCategory.URGENT: EmotionProfile(
        pitch_mean_range=(0.3, 1.2),
        pitch_variability=(0.3, 1.0),
        pitch_slope_range=(-0.3, 0.5),
        energy_level=(0.5, 1.5),
        energy_variability=(0.3, 1.0),
        speaking_rate=(0.5, 1.5),
        pause_ratio=(-1.0, -0.3),
        spectral_centroid=(0.3, 1.2),
        jitter_range=(0.0, 0.5),
        shimmer_range=(0.0, 0.5),
        arousal=ArousalLevel.VERY_HIGH,
        valence=ValenceLevel.NEUTRAL,
        prior_weight=0.7,
    ),
}


class FeatureNormalizer:
    """
    Normalizes prosodic features to z-scores based on
    running statistics or reference values.
    """

    def __init__(self):
        """Initialize with default reference values."""
        # Reference values based on typical speech patterns
        self.reference_means = {
            "pitch_mean_hz": 150.0,
            "pitch_std_hz": 30.0,
            "pitch_slope": 0.0,
            "energy_mean": 0.1,
            "energy_std": 0.03,
            "speaking_rate_sps": 4.0,
            "pause_ratio": 0.3,
            "spectral_centroid": 1500.0,
            "jitter": 0.02,
            "shimmer": 0.05,
        }

        self.reference_stds = {
            "pitch_mean_hz": 50.0,
            "pitch_std_hz": 20.0,
            "pitch_slope": 5.0,
            "energy_mean": 0.05,
            "energy_std": 0.02,
            "speaking_rate_sps": 1.5,
            "pause_ratio": 0.15,
            "spectral_centroid": 500.0,
            "jitter": 0.01,
            "shimmer": 0.03,
        }

        # Running statistics for adaptation
        self._counts: Dict[str, int] = {}
        self._running_means: Dict[str, float] = {}
        self._running_m2: Dict[str, float] = {}

    def normalize(self, features: ProsodicsFeatures) -> Dict[str, float]:
        """Normalize features to z-scores."""
        normalized = {}

        feature_values = {
            "pitch_mean_hz": features.pitch_mean_hz,
            "pitch_std_hz": features.pitch_std_hz,
            "pitch_slope": features.pitch_slope,
            "energy_mean": features.energy_mean,
            "energy_std": features.energy_std,
            "speaking_rate_sps": features.speaking_rate_sps,
            "pause_ratio": features.pause_ratio,
            "spectral_centroid": features.spectral_centroid,
            "jitter": features.jitter,
            "shimmer": features.shimmer,
        }

        for key, value in feature_values.items():
            mean = self.reference_means.get(key, 0.0)
            std = self.reference_stds.get(key, 1.0)

            if std > 0:
                normalized[key] = (value - mean) / std
            else:
                normalized[key] = 0.0

        return normalized

    def update_statistics(self, features: ProsodicsFeatures) -> None:
        """Update running statistics with new features (Welford's algorithm)."""
        feature_values = {
            "pitch_mean_hz": features.pitch_mean_hz,
            "pitch_std_hz": features.pitch_std_hz,
            "energy_mean": features.energy_mean,
            "speaking_rate_sps": features.speaking_rate_sps,
        }

        for key, value in feature_values.items():
            if value <= 0:  # Skip invalid values
                continue

            if key not in self._counts:
                self._counts[key] = 0
                self._running_means[key] = 0.0
                self._running_m2[key] = 0.0

            self._counts[key] += 1
            delta = value - self._running_means[key]
            self._running_means[key] += delta / self._counts[key]
            delta2 = value - self._running_means[key]
            self._running_m2[key] += delta * delta2


class EmotionClassifier:
    """
    Classifies emotions from prosodic features.

    Uses a profile-matching approach where prosodic features
    are compared against characteristic profiles for each emotion.
    Outputs probability distribution over emotion categories.
    """

    def __init__(self, config: Optional[EmotionModelConfig] = None):
        """Initialize classifier."""
        self.config = config or get_settings().emotion_model
        self.normalizer = FeatureNormalizer()
        self.profiles = EMOTION_PROFILES

        # Feature weights for classification
        self.feature_weights = {
            "pitch_mean": 1.2,
            "pitch_variability": 1.0,
            "pitch_slope": 0.8,
            "energy": 1.1,
            "energy_variability": 0.7,
            "speaking_rate": 0.9,
            "pause_ratio": 0.8,
            "spectral": 0.6,
            "jitter": 0.5,
            "shimmer": 0.5,
        }

    def classify(
        self,
        features: ProsodicsFeatures,
        adapt_normalization: bool = True,
    ) -> EmotionPrediction:
        """
        Classify emotion from prosodic features.

        Args:
            features: Prosodic features extracted from audio
            adapt_normalization: Whether to update running statistics

        Returns:
            EmotionPrediction with scores for all emotions
        """
        # Normalize features
        normalized = self.normalizer.normalize(features)

        if adapt_normalization:
            self.normalizer.update_statistics(features)

        # Calculate match scores for each emotion
        scores: List[EmotionScore] = []

        for emotion, profile in self.profiles.items():
            score = self._calculate_match_score(normalized, profile)
            scores.append(
                EmotionScore(
                    category=emotion,
                    confidence=score,
                    arousal=profile.arousal,
                    valence=profile.valence,
                )
            )

        # Normalize to probabilities using softmax
        scores = self._softmax_normalize(scores)

        # Sort by confidence
        scores.sort(key=lambda s: s.confidence, reverse=True)

        # Get primary emotion
        primary = scores[0]

        # Calculate arousal and valence as weighted average
        arousal, valence = self._calculate_dimensional_emotions(scores)

        return EmotionPrediction(
            timestamp_ms=features.timestamp_ms,
            primary_emotion=primary.category,
            primary_confidence=primary.confidence,
            arousal=arousal,
            valence=valence,
            all_scores=scores,
            features_used=list(normalized.keys()),
        )

    def _calculate_match_score(
        self,
        normalized: Dict[str, float],
        profile: EmotionProfile,
    ) -> float:
        """Calculate how well features match an emotion profile."""
        scores = []
        weights = []

        # Pitch mean
        score = self._range_score(
            normalized.get("pitch_mean_hz", 0),
            profile.pitch_mean_range,
        )
        scores.append(score)
        weights.append(self.feature_weights["pitch_mean"])

        # Pitch variability
        score = self._range_score(
            normalized.get("pitch_std_hz", 0),
            profile.pitch_variability,
        )
        scores.append(score)
        weights.append(self.feature_weights["pitch_variability"])

        # Pitch slope
        score = self._range_score(
            normalized.get("pitch_slope", 0),
            profile.pitch_slope_range,
        )
        scores.append(score)
        weights.append(self.feature_weights["pitch_slope"])

        # Energy
        score = self._range_score(
            normalized.get("energy_mean", 0),
            profile.energy_level,
        )
        scores.append(score)
        weights.append(self.feature_weights["energy"])

        # Energy variability
        score = self._range_score(
            normalized.get("energy_std", 0),
            profile.energy_variability,
        )
        scores.append(score)
        weights.append(self.feature_weights["energy_variability"])

        # Speaking rate
        score = self._range_score(
            normalized.get("speaking_rate_sps", 0),
            profile.speaking_rate,
        )
        scores.append(score)
        weights.append(self.feature_weights["speaking_rate"])

        # Pause ratio
        score = self._range_score(
            normalized.get("pause_ratio", 0),
            profile.pause_ratio,
        )
        scores.append(score)
        weights.append(self.feature_weights["pause_ratio"])

        # Spectral centroid
        score = self._range_score(
            normalized.get("spectral_centroid", 0),
            profile.spectral_centroid,
        )
        scores.append(score)
        weights.append(self.feature_weights["spectral"])

        # Voice quality
        jitter_score = self._range_score(
            normalized.get("jitter", 0),
            profile.jitter_range,
        )
        shimmer_score = self._range_score(
            normalized.get("shimmer", 0),
            profile.shimmer_range,
        )
        scores.append((jitter_score + shimmer_score) / 2)
        weights.append(
            (self.feature_weights["jitter"] + self.feature_weights["shimmer"]) / 2
        )

        # Weighted average with prior
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Apply prior weight
        return weighted_score * profile.prior_weight

    def _range_score(
        self,
        value: float,
        range_tuple: Tuple[float, float],
    ) -> float:
        """
        Calculate score based on how well value fits in range.

        Returns 1.0 if value is in range, decreasing as value
        moves away from range boundaries.
        """
        low, high = range_tuple

        if low <= value <= high:
            return 1.0

        # Calculate distance from range
        if value < low:
            distance = low - value
        else:
            distance = value - high

        # Exponential decay
        return np.exp(-distance * 0.5)

    def _softmax_normalize(
        self,
        scores: List[EmotionScore],
        temperature: float = 1.0,
    ) -> List[EmotionScore]:
        """Normalize scores to probabilities using softmax."""
        confidences = np.array([s.confidence for s in scores])

        # Apply temperature
        confidences = confidences / temperature

        # Softmax
        exp_scores = np.exp(confidences - np.max(confidences))
        probabilities = exp_scores / np.sum(exp_scores)

        # Update scores
        for score, prob in zip(scores, probabilities):
            score.confidence = float(prob)

        return scores

    def _calculate_dimensional_emotions(
        self,
        scores: List[EmotionScore],
    ) -> Tuple[float, float]:
        """
        Calculate arousal and valence as weighted averages.

        Returns:
            Tuple of (arousal, valence) in range [-1, 1]
        """
        arousal_map = {
            ArousalLevel.VERY_LOW: -1.0,
            ArousalLevel.LOW: -0.5,
            ArousalLevel.MEDIUM: 0.0,
            ArousalLevel.HIGH: 0.5,
            ArousalLevel.VERY_HIGH: 1.0,
        }

        valence_map = {
            ValenceLevel.VERY_NEGATIVE: -1.0,
            ValenceLevel.NEGATIVE: -0.6,
            ValenceLevel.SLIGHTLY_NEGATIVE: -0.3,
            ValenceLevel.NEUTRAL: 0.0,
            ValenceLevel.SLIGHTLY_POSITIVE: 0.3,
            ValenceLevel.POSITIVE: 0.6,
            ValenceLevel.VERY_POSITIVE: 1.0,
        }

        total_weight = sum(s.confidence for s in scores)

        if total_weight == 0:
            return 0.0, 0.0

        arousal = sum(
            arousal_map.get(s.arousal, 0.0) * s.confidence for s in scores
        ) / total_weight

        valence = sum(
            valence_map.get(s.valence, 0.0) * s.confidence for s in scores
        ) / total_weight

        return float(arousal), float(valence)

    def get_top_emotions(
        self,
        prediction: EmotionPrediction,
        n: int = 3,
        min_confidence: float = 0.1,
    ) -> List[EmotionScore]:
        """Get top N emotions above confidence threshold."""
        return [
            s
            for s in prediction.all_scores[:n]
            if s.confidence >= min_confidence
        ]
