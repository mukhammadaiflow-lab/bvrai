"""
Emotional State Tracker.

Tracks emotional state over time, applying temporal smoothing
and detecting significant emotional shifts.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

from ..config import (
    EmotionCategory,
    ContextConfig,
    ArousalLevel,
    ValenceLevel,
    get_settings,
)
from ..models import (
    EmotionPrediction,
    EmotionScore,
    EmotionalState,
    EmotionalShift,
)

logger = logging.getLogger(__name__)


@dataclass
class EmotionWindow:
    """Sliding window of emotion predictions."""

    predictions: Deque[EmotionPrediction] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    window_size_ms: float = 5000.0

    def add(self, prediction: EmotionPrediction) -> None:
        """Add prediction to window."""
        self.predictions.append(prediction)
        self._prune_old()

    def _prune_old(self) -> None:
        """Remove predictions outside the window."""
        if not self.predictions:
            return

        current_time = self.predictions[-1].timestamp_ms
        cutoff = current_time - self.window_size_ms

        while self.predictions and self.predictions[0].timestamp_ms < cutoff:
            self.predictions.popleft()

    def get_weighted_emotions(self) -> Dict[EmotionCategory, float]:
        """Get time-weighted emotion scores."""
        if not self.predictions:
            return {}

        scores: Dict[EmotionCategory, float] = {}
        total_weight = 0.0

        current_time = self.predictions[-1].timestamp_ms

        for pred in self.predictions:
            # Exponential decay weight based on recency
            age_ms = current_time - pred.timestamp_ms
            weight = np.exp(-age_ms / (self.window_size_ms / 2))

            for emotion_score in pred.all_scores:
                category = emotion_score.category
                if category not in scores:
                    scores[category] = 0.0
                scores[category] += emotion_score.confidence * weight

            total_weight += weight

        # Normalize
        if total_weight > 0:
            scores = {k: v / total_weight for k, v in scores.items()}

        return scores

    def get_arousal_valence_trend(self) -> Tuple[float, float, float, float]:
        """
        Get arousal/valence mean and trend.

        Returns:
            (arousal_mean, valence_mean, arousal_slope, valence_slope)
        """
        if len(self.predictions) < 2:
            if self.predictions:
                pred = self.predictions[0]
                return pred.arousal, pred.valence, 0.0, 0.0
            return 0.0, 0.0, 0.0, 0.0

        times = []
        arousals = []
        valences = []

        for pred in self.predictions:
            times.append(pred.timestamp_ms)
            arousals.append(pred.arousal)
            valences.append(pred.valence)

        times = np.array(times)
        arousals = np.array(arousals)
        valences = np.array(valences)

        # Normalize time
        times = (times - times[0]) / 1000.0  # Convert to seconds

        # Calculate slopes
        arousal_slope = np.polyfit(times, arousals, 1)[0] if len(times) > 1 else 0.0
        valence_slope = np.polyfit(times, valences, 1)[0] if len(times) > 1 else 0.0

        return (
            float(np.mean(arousals)),
            float(np.mean(valences)),
            float(arousal_slope),
            float(valence_slope),
        )


@dataclass
class ConversationContext:
    """Tracks conversation-level emotional context."""

    session_id: str
    start_time: datetime = field(default_factory=datetime.utcnow)

    # Emotional baselines established from early conversation
    baseline_arousal: Optional[float] = None
    baseline_valence: Optional[float] = None
    baseline_established: bool = False

    # Cumulative emotional metrics
    total_turns: int = 0
    positive_turns: int = 0
    negative_turns: int = 0
    neutral_turns: int = 0

    # Emotion frequency
    emotion_counts: Dict[EmotionCategory, int] = field(default_factory=dict)

    # Shift history
    shifts: List[EmotionalShift] = field(default_factory=list)

    # Peak emotions
    peak_arousal: float = 0.0
    peak_valence: float = 0.0
    lowest_valence: float = 0.0

    def update_baseline(self, arousal: float, valence: float) -> None:
        """Update emotional baseline from early conversation."""
        if self.baseline_established:
            # Slow adaptation
            alpha = 0.1
            self.baseline_arousal = (
                alpha * arousal + (1 - alpha) * self.baseline_arousal
            )
            self.baseline_valence = (
                alpha * valence + (1 - alpha) * self.baseline_valence
            )
        else:
            self.baseline_arousal = arousal
            self.baseline_valence = valence
            self.baseline_established = True

    def record_emotion(self, primary: EmotionCategory, valence: float) -> None:
        """Record emotion occurrence."""
        self.total_turns += 1

        if valence > 0.2:
            self.positive_turns += 1
        elif valence < -0.2:
            self.negative_turns += 1
        else:
            self.neutral_turns += 1

        if primary not in self.emotion_counts:
            self.emotion_counts[primary] = 0
        self.emotion_counts[primary] += 1

    def get_emotional_trajectory(self) -> str:
        """Determine overall emotional trajectory."""
        if self.total_turns < 3:
            return "establishing"

        positive_ratio = self.positive_turns / self.total_turns
        negative_ratio = self.negative_turns / self.total_turns

        if positive_ratio > 0.6:
            return "positive"
        elif negative_ratio > 0.6:
            return "negative"
        elif positive_ratio > negative_ratio:
            return "slightly_positive"
        elif negative_ratio > positive_ratio:
            return "slightly_negative"
        else:
            return "neutral"


class EmotionalStateTracker:
    """
    Tracks emotional state over a conversation.

    Features:
    - Temporal smoothing of emotion predictions
    - Baseline establishment
    - Shift detection
    - Trend analysis
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        """Initialize tracker."""
        self.config = config or get_settings().context
        self.sessions: Dict[str, ConversationContext] = {}
        self.windows: Dict[str, EmotionWindow] = {}

        # EMA state for smoothing
        self._ema_arousal: Dict[str, float] = {}
        self._ema_valence: Dict[str, float] = {}
        self._ema_emotions: Dict[str, Dict[EmotionCategory, float]] = {}

        # Previous states for shift detection
        self._previous_states: Dict[str, EmotionalState] = {}

    def update(
        self,
        session_id: str,
        prediction: EmotionPrediction,
    ) -> EmotionalState:
        """
        Update emotional state with new prediction.

        Args:
            session_id: Conversation session ID
            prediction: Latest emotion prediction

        Returns:
            Updated EmotionalState
        """
        # Initialize session if needed
        if session_id not in self.sessions:
            self._init_session(session_id)

        context = self.sessions[session_id]
        window = self.windows[session_id]

        # Add to window
        window.add(prediction)

        # Get smoothed emotions
        smoothed_emotions = self._smooth_emotions(session_id, prediction)

        # Get primary emotion from smoothed
        primary_emotion = max(smoothed_emotions.items(), key=lambda x: x[1])

        # Calculate smoothed arousal/valence
        arousal, valence = self._smooth_dimensional(session_id, prediction)

        # Update baseline
        if context.total_turns < self.config.baseline_turns:
            context.update_baseline(arousal, valence)

        # Get trend
        _, _, arousal_slope, valence_slope = window.get_arousal_valence_trend()

        # Determine stability
        stability = self._calculate_stability(window)

        # Detect shift
        shift = self._detect_shift(session_id, arousal, valence, primary_emotion[0])

        # Record in context
        context.record_emotion(primary_emotion[0], valence)

        # Update peaks
        context.peak_arousal = max(context.peak_arousal, arousal)
        context.peak_valence = max(context.peak_valence, valence)
        context.lowest_valence = min(context.lowest_valence, valence)

        # Create state
        state = EmotionalState(
            session_id=session_id,
            timestamp_ms=prediction.timestamp_ms,
            current_emotion=primary_emotion[0],
            current_confidence=primary_emotion[1],
            smoothed_arousal=arousal,
            smoothed_valence=valence,
            arousal_trend=arousal_slope,
            valence_trend=valence_slope,
            stability=stability,
            baseline_arousal=context.baseline_arousal,
            baseline_valence=context.baseline_valence,
            deviation_from_baseline=self._calculate_baseline_deviation(
                arousal, valence, context
            ),
            recent_shift=shift,
            conversation_trajectory=context.get_emotional_trajectory(),
            emotion_distribution=smoothed_emotions,
        )

        # Store for next update
        self._previous_states[session_id] = state

        return state

    def _init_session(self, session_id: str) -> None:
        """Initialize a new session."""
        self.sessions[session_id] = ConversationContext(session_id=session_id)
        self.windows[session_id] = EmotionWindow(
            window_size_ms=self.config.smoothing_window_ms
        )
        self._ema_arousal[session_id] = 0.0
        self._ema_valence[session_id] = 0.0
        self._ema_emotions[session_id] = {}

    def _smooth_emotions(
        self,
        session_id: str,
        prediction: EmotionPrediction,
    ) -> Dict[EmotionCategory, float]:
        """Apply exponential moving average to emotion scores."""
        alpha = self.config.ema_alpha
        current = {s.category: s.confidence for s in prediction.all_scores}

        if session_id not in self._ema_emotions:
            self._ema_emotions[session_id] = current
            return current

        ema = self._ema_emotions[session_id]

        # Update EMA for all categories
        all_categories = set(current.keys()) | set(ema.keys())
        smoothed = {}

        for category in all_categories:
            curr_val = current.get(category, 0.0)
            prev_val = ema.get(category, 0.0)
            smoothed[category] = alpha * curr_val + (1 - alpha) * prev_val

        self._ema_emotions[session_id] = smoothed
        return smoothed

    def _smooth_dimensional(
        self,
        session_id: str,
        prediction: EmotionPrediction,
    ) -> Tuple[float, float]:
        """Smooth arousal and valence values."""
        alpha = self.config.ema_alpha

        # Update arousal
        prev_arousal = self._ema_arousal.get(session_id, prediction.arousal)
        smoothed_arousal = alpha * prediction.arousal + (1 - alpha) * prev_arousal
        self._ema_arousal[session_id] = smoothed_arousal

        # Update valence
        prev_valence = self._ema_valence.get(session_id, prediction.valence)
        smoothed_valence = alpha * prediction.valence + (1 - alpha) * prev_valence
        self._ema_valence[session_id] = smoothed_valence

        return smoothed_arousal, smoothed_valence

    def _calculate_stability(self, window: EmotionWindow) -> float:
        """Calculate emotional stability from window variance."""
        if len(window.predictions) < 3:
            return 1.0

        # Calculate variance of arousal and valence
        arousals = [p.arousal for p in window.predictions]
        valences = [p.valence for p in window.predictions]

        arousal_var = np.var(arousals)
        valence_var = np.var(valences)

        # Combined variance
        combined_var = (arousal_var + valence_var) / 2

        # Convert to stability score (higher variance = lower stability)
        stability = np.exp(-combined_var * 5)

        return float(np.clip(stability, 0.0, 1.0))

    def _calculate_baseline_deviation(
        self,
        arousal: float,
        valence: float,
        context: ConversationContext,
    ) -> float:
        """Calculate deviation from emotional baseline."""
        if not context.baseline_established:
            return 0.0

        arousal_dev = abs(arousal - context.baseline_arousal)
        valence_dev = abs(valence - context.baseline_valence)

        return float((arousal_dev + valence_dev) / 2)

    def _detect_shift(
        self,
        session_id: str,
        arousal: float,
        valence: float,
        current_emotion: EmotionCategory,
    ) -> Optional[EmotionalShift]:
        """Detect significant emotional shift."""
        if session_id not in self._previous_states:
            return None

        prev_state = self._previous_states[session_id]

        # Check for significant change
        arousal_change = arousal - prev_state.smoothed_arousal
        valence_change = valence - prev_state.smoothed_valence

        magnitude = np.sqrt(arousal_change ** 2 + valence_change ** 2)

        if magnitude < self.config.shift_threshold:
            return None

        # Determine shift direction
        if valence_change > 0.2:
            direction = "positive"
        elif valence_change < -0.2:
            direction = "negative"
        elif arousal_change > 0.2:
            direction = "escalation"
        elif arousal_change < -0.2:
            direction = "de-escalation"
        else:
            direction = "lateral"

        shift = EmotionalShift(
            timestamp_ms=prev_state.timestamp_ms,
            from_emotion=prev_state.current_emotion,
            to_emotion=current_emotion,
            arousal_change=arousal_change,
            valence_change=valence_change,
            magnitude=magnitude,
            direction=direction,
        )

        # Record in context
        context = self.sessions[session_id]
        context.shifts.append(shift)

        return shift

    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary of emotional state for a session."""
        if session_id not in self.sessions:
            return {"error": "Session not found"}

        context = self.sessions[session_id]
        window = self.windows[session_id]

        # Get current weighted emotions
        current_emotions = window.get_weighted_emotions()
        arousal_mean, valence_mean, arousal_trend, valence_trend = (
            window.get_arousal_valence_trend()
        )

        # Get dominant emotions
        sorted_emotions = sorted(
            context.emotion_counts.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "session_id": session_id,
            "duration_seconds": (
                datetime.utcnow() - context.start_time
            ).total_seconds(),
            "total_turns": context.total_turns,
            "trajectory": context.get_emotional_trajectory(),
            "baseline": {
                "arousal": context.baseline_arousal,
                "valence": context.baseline_valence,
            },
            "current": {
                "arousal": arousal_mean,
                "valence": valence_mean,
                "emotions": current_emotions,
            },
            "trends": {
                "arousal": arousal_trend,
                "valence": valence_trend,
            },
            "peaks": {
                "highest_arousal": context.peak_arousal,
                "highest_valence": context.peak_valence,
                "lowest_valence": context.lowest_valence,
            },
            "dominant_emotions": [
                {"emotion": e.value, "count": c}
                for e, c in sorted_emotions[:5]
            ],
            "sentiment_distribution": {
                "positive": context.positive_turns / max(1, context.total_turns),
                "negative": context.negative_turns / max(1, context.total_turns),
                "neutral": context.neutral_turns / max(1, context.total_turns),
            },
            "shift_count": len(context.shifts),
            "recent_shifts": [
                {
                    "from": s.from_emotion.value,
                    "to": s.to_emotion.value,
                    "direction": s.direction,
                    "magnitude": s.magnitude,
                }
                for s in context.shifts[-5:]
            ],
        }

    def clear_session(self, session_id: str) -> None:
        """Clear session data."""
        self.sessions.pop(session_id, None)
        self.windows.pop(session_id, None)
        self._ema_arousal.pop(session_id, None)
        self._ema_valence.pop(session_id, None)
        self._ema_emotions.pop(session_id, None)
        self._previous_states.pop(session_id, None)
