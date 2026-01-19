"""
Emotion Engine.

Main orchestrator that combines prosodic analysis, emotion classification,
state tracking, and response advising into a unified pipeline.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional, Tuple
import numpy as np

from ..config import get_settings, EmotionCategory
from ..models import (
    ProsodicsFeatures,
    EmotionPrediction,
    EmotionalState,
    ResponseRecommendation,
    EmotionEngineResult,
)
from .prosodics import ProsodicsAnalyzer
from .classifier import EmotionClassifier
from .state_tracker import EmotionalStateTracker
from .advisor import ResponseAdvisor

logger = logging.getLogger(__name__)


@dataclass
class EngineMetrics:
    """Metrics for engine performance."""

    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    avg_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float("inf")
    max_processing_time_ms: float = 0.0

    # Emotion distribution
    emotion_counts: Dict[str, int] = None

    def __post_init__(self):
        if self.emotion_counts is None:
            self.emotion_counts = {}

    def record_analysis(
        self,
        processing_time_ms: float,
        emotion: Optional[EmotionCategory] = None,
        success: bool = True,
    ) -> None:
        """Record analysis metrics."""
        self.total_analyses += 1

        if success:
            self.successful_analyses += 1
        else:
            self.failed_analyses += 1

        # Update processing times
        self.min_processing_time_ms = min(
            self.min_processing_time_ms, processing_time_ms
        )
        self.max_processing_time_ms = max(
            self.max_processing_time_ms, processing_time_ms
        )

        # Running average
        alpha = 0.1
        self.avg_processing_time_ms = (
            alpha * processing_time_ms + (1 - alpha) * self.avg_processing_time_ms
        )

        # Emotion counts
        if emotion:
            key = emotion.value
            if key not in self.emotion_counts:
                self.emotion_counts[key] = 0
            self.emotion_counts[key] += 1


class EmotionEngine:
    """
    Real-time emotion analysis engine.

    Provides end-to-end emotion detection from audio, including:
    - Prosodic feature extraction
    - Emotion classification
    - Temporal state tracking
    - Response recommendations

    Usage:
        engine = EmotionEngine()
        await engine.initialize()

        result = await engine.analyze(audio_data, sample_rate, session_id)
        print(result.state.current_emotion)
        print(result.recommendation.primary_strategy)
    """

    def __init__(self):
        """Initialize engine components."""
        self.settings = get_settings()

        # Core components
        self.prosodics_analyzer = ProsodicsAnalyzer()
        self.classifier = EmotionClassifier()
        self.state_tracker = EmotionalStateTracker()
        self.advisor = ResponseAdvisor()

        # State
        self._initialized = False
        self._metrics = EngineMetrics()
        self._active_sessions: Dict[str, datetime] = {}

        # Processing lock per session
        self._session_locks: Dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        """Initialize engine resources."""
        if self._initialized:
            return

        logger.info("Initializing Emotion Engine")

        # Load any required models or resources here
        # (Placeholder for ML model loading)

        self._initialized = True
        logger.info("Emotion Engine initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown engine and release resources."""
        logger.info("Shutting down Emotion Engine")

        # Clear all sessions
        for session_id in list(self._active_sessions.keys()):
            await self.end_session(session_id)

        self._initialized = False
        logger.info("Emotion Engine shutdown complete")

    async def start_session(self, session_id: str) -> None:
        """Start a new analysis session."""
        if session_id in self._active_sessions:
            logger.warning(f"Session {session_id} already exists")
            return

        self._active_sessions[session_id] = datetime.utcnow()
        self._session_locks[session_id] = asyncio.Lock()

        logger.info(f"Started emotion analysis session: {session_id}")

    async def end_session(self, session_id: str) -> Optional[Dict]:
        """
        End an analysis session and return summary.

        Returns:
            Session summary with emotional statistics
        """
        if session_id not in self._active_sessions:
            return None

        # Get summary before clearing
        summary = self.state_tracker.get_session_summary(session_id)

        # Clear session data
        self.state_tracker.clear_session(session_id)
        del self._active_sessions[session_id]
        del self._session_locks[session_id]

        logger.info(f"Ended emotion analysis session: {session_id}")

        return summary

    async def analyze(
        self,
        audio_data: bytes,
        sample_rate: int,
        session_id: str,
        timestamp_ms: Optional[float] = None,
        include_recommendation: bool = True,
    ) -> EmotionEngineResult:
        """
        Analyze audio for emotion.

        Args:
            audio_data: Raw audio bytes (int16 or float32)
            sample_rate: Audio sample rate in Hz
            session_id: Session identifier for tracking
            timestamp_ms: Optional timestamp for the audio segment
            include_recommendation: Whether to generate response recommendation

        Returns:
            EmotionEngineResult with features, prediction, state, and recommendation
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.utcnow()

        # Ensure session exists
        if session_id not in self._active_sessions:
            await self.start_session(session_id)

        # Get session lock
        lock = self._session_locks[session_id]

        try:
            async with lock:
                result = await self._process_audio(
                    audio_data,
                    sample_rate,
                    session_id,
                    timestamp_ms,
                    include_recommendation,
                )

            # Record metrics
            processing_time = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000
            self._metrics.record_analysis(
                processing_time,
                result.state.current_emotion,
                success=True,
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")

            processing_time = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000
            self._metrics.record_analysis(processing_time, success=False)

            raise

    async def _process_audio(
        self,
        audio_data: bytes,
        sample_rate: int,
        session_id: str,
        timestamp_ms: Optional[float],
        include_recommendation: bool,
    ) -> EmotionEngineResult:
        """Process audio through the emotion pipeline."""
        # Convert bytes to numpy array
        audio = self._bytes_to_numpy(audio_data)

        # Use current time if no timestamp provided
        if timestamp_ms is None:
            timestamp_ms = datetime.utcnow().timestamp() * 1000

        # Step 1: Extract prosodic features
        features = await asyncio.to_thread(
            self.prosodics_analyzer.analyze,
            audio,
            sample_rate,
            timestamp_ms,
        )

        # Step 2: Classify emotion
        prediction = await asyncio.to_thread(
            self.classifier.classify,
            features,
        )

        # Step 3: Update state tracker
        state = self.state_tracker.update(session_id, prediction)

        # Step 4: Get response recommendation (if requested)
        recommendation = None
        if include_recommendation:
            recommendation = self.advisor.get_recommendation(state)

        return EmotionEngineResult(
            session_id=session_id,
            timestamp_ms=timestamp_ms,
            features=features,
            prediction=prediction,
            state=state,
            recommendation=recommendation,
            processing_time_ms=(
                datetime.utcnow().timestamp() * 1000 - timestamp_ms
            ),
        )

    async def analyze_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int,
        session_id: str,
        chunk_duration_ms: float = 100.0,
    ) -> AsyncIterator[EmotionEngineResult]:
        """
        Analyze streaming audio.

        Args:
            audio_stream: Async iterator of audio chunks
            sample_rate: Audio sample rate
            session_id: Session identifier
            chunk_duration_ms: Expected duration of each chunk

        Yields:
            EmotionEngineResult for each processed chunk
        """
        if not self._initialized:
            await self.initialize()

        if session_id not in self._active_sessions:
            await self.start_session(session_id)

        current_time_ms = 0.0
        buffer = bytearray()
        min_buffer_size = int(
            sample_rate * self.settings.min_analysis_duration_ms / 1000 * 2
        )  # int16 = 2 bytes

        async for chunk in audio_stream:
            buffer.extend(chunk)
            current_time_ms += chunk_duration_ms

            # Process when buffer is large enough
            if len(buffer) >= min_buffer_size:
                result = await self.analyze(
                    bytes(buffer),
                    sample_rate,
                    session_id,
                    current_time_ms,
                )

                yield result

                # Keep overlap for continuity
                overlap_size = min_buffer_size // 4
                buffer = buffer[-overlap_size:]

    def get_quick_emotion(
        self,
        audio_data: bytes,
        sample_rate: int,
    ) -> Tuple[EmotionCategory, float]:
        """
        Quick synchronous emotion detection without tracking.

        Args:
            audio_data: Raw audio bytes
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (emotion_category, confidence)
        """
        audio = self._bytes_to_numpy(audio_data)
        features = self.prosodics_analyzer.analyze(audio, sample_rate)
        prediction = self.classifier.classify(features, adapt_normalization=False)

        return prediction.primary_emotion, prediction.primary_confidence

    def get_dimensional_emotion(
        self,
        audio_data: bytes,
        sample_rate: int,
    ) -> Tuple[float, float]:
        """
        Get arousal-valence coordinates without tracking.

        Args:
            audio_data: Raw audio bytes
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (arousal, valence) in range [-1, 1]
        """
        audio = self._bytes_to_numpy(audio_data)
        features = self.prosodics_analyzer.analyze(audio, sample_rate)
        prediction = self.classifier.classify(features, adapt_normalization=False)

        return prediction.arousal, prediction.valence

    async def get_session_state(
        self,
        session_id: str,
    ) -> Optional[Dict]:
        """Get current state for a session."""
        return self.state_tracker.get_session_summary(session_id)

    def get_metrics(self) -> Dict:
        """Get engine performance metrics."""
        return {
            "total_analyses": self._metrics.total_analyses,
            "successful_analyses": self._metrics.successful_analyses,
            "failed_analyses": self._metrics.failed_analyses,
            "success_rate": (
                self._metrics.successful_analyses / max(1, self._metrics.total_analyses)
            ),
            "processing_times": {
                "avg_ms": self._metrics.avg_processing_time_ms,
                "min_ms": (
                    self._metrics.min_processing_time_ms
                    if self._metrics.min_processing_time_ms != float("inf")
                    else 0.0
                ),
                "max_ms": self._metrics.max_processing_time_ms,
            },
            "emotion_distribution": self._metrics.emotion_counts,
            "active_sessions": len(self._active_sessions),
        }

    def _bytes_to_numpy(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        # Assume int16 format
        audio = np.frombuffer(audio_data, dtype=np.int16)

        # Convert to float32 normalized
        audio = audio.astype(np.float32) / 32768.0

        return audio

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._initialized

    @property
    def active_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._active_sessions)


# Singleton instance
_engine_instance: Optional[EmotionEngine] = None


async def get_engine() -> EmotionEngine:
    """Get or create the singleton engine instance."""
    global _engine_instance

    if _engine_instance is None:
        _engine_instance = EmotionEngine()
        await _engine_instance.initialize()

    return _engine_instance
