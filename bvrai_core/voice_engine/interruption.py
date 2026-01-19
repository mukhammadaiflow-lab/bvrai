"""
Interruption handling for conversational AI.

Provides sophisticated handling of user interruptions during AI speech:
- Immediate stop on user speech detection
- Graceful completion of current sentence
- Smart continuation after brief interruptions
- Barge-in support with different strategies

Critical for natural, human-like conversations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading

from .audio import AudioChunk, AudioBuffer
from .vad import VADEvent, VADEventType, VoiceActivityDetector

logger = logging.getLogger(__name__)


class InterruptionStrategy(str, Enum):
    """
    Strategies for handling user interruptions.

    IMMEDIATE: Stop AI speech immediately when user speaks
    SENTENCE: Complete current sentence before stopping
    SMART: Analyze interruption context to decide
    IGNORE: Don't stop AI speech (user must wait)
    HOLD: Pause briefly, resume if user stops quickly
    """
    IMMEDIATE = "immediate"
    SENTENCE = "sentence"
    SMART = "smart"
    IGNORE = "ignore"
    HOLD = "hold"


class InterruptionType(str, Enum):
    """Types of user interruptions."""
    BARGE_IN = "barge_in"           # User starts speaking while AI speaks
    QUICK_RESPONSE = "quick_response" # User responds before AI finishes
    CLARIFICATION = "clarification"  # User asks for clarification mid-speech
    AGREEMENT = "agreement"          # User says "yes", "uh-huh", etc.
    DISAGREEMENT = "disagreement"    # User says "no", "wait", etc.
    UNKNOWN = "unknown"


@dataclass
class InterruptionEvent:
    """
    Represents an interruption event.

    Attributes:
        event_type: Type of interruption
        timestamp_ms: When the interruption occurred
        duration_ms: How long the user spoke
        user_audio: Audio data of user speech (for analysis)
        ai_position_ms: Where in AI speech the interruption occurred
        ai_remaining_ms: How much AI speech was remaining
        action_taken: What action was taken (stop, pause, continue)
        transcription: Transcribed user speech (if available)
    """
    event_type: InterruptionType = InterruptionType.UNKNOWN
    timestamp_ms: float = 0.0
    duration_ms: float = 0.0
    user_audio: Optional[bytes] = None
    ai_position_ms: float = 0.0
    ai_remaining_ms: float = 0.0
    action_taken: str = ""
    transcription: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp_ms": self.timestamp_ms,
            "duration_ms": self.duration_ms,
            "ai_position_ms": self.ai_position_ms,
            "ai_remaining_ms": self.ai_remaining_ms,
            "action_taken": self.action_taken,
            "transcription": self.transcription,
            "metadata": self.metadata,
        }


@dataclass
class InterruptionConfig:
    """
    Configuration for interruption handling.

    Attributes:
        strategy: Primary interruption handling strategy
        min_interruption_duration_ms: Minimum speech duration to count as interruption
        hold_duration_ms: How long to hold before resuming (HOLD strategy)
        sentence_end_markers: Characters that mark sentence endings
        backoff_words: Number of words to back off when resuming

        agreement_words: Words that indicate agreement (not real interruption)
        disagreement_words: Words that indicate disagreement (stop immediately)

        sensitivity: How sensitive to interruptions (0.0 - 1.0)
        energy_threshold_db: Minimum energy to consider as interruption
    """
    strategy: InterruptionStrategy = InterruptionStrategy.SMART
    min_interruption_duration_ms: float = 150
    hold_duration_ms: float = 500
    sentence_end_markers: List[str] = field(default_factory=lambda: [".", "!", "?"])
    backoff_words: int = 3

    agreement_words: List[str] = field(default_factory=lambda: [
        "yes", "yeah", "yep", "uh-huh", "mm-hmm", "ok", "okay",
        "sure", "right", "got it", "i see", "understood"
    ])

    disagreement_words: List[str] = field(default_factory=lambda: [
        "no", "wait", "stop", "hold on", "actually", "but",
        "hang on", "one moment", "excuse me"
    ])

    sensitivity: float = 0.7
    energy_threshold_db: float = -40.0

    # Callback delays
    pre_interrupt_delay_ms: float = 50
    post_interrupt_delay_ms: float = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "min_interruption_duration_ms": self.min_interruption_duration_ms,
            "hold_duration_ms": self.hold_duration_ms,
            "sensitivity": self.sensitivity,
        }


class InterruptionHandler:
    """
    Handles user interruptions during AI speech.

    Monitors user audio input while AI is speaking and determines
    appropriate action based on configured strategy.
    """

    def __init__(
        self,
        config: InterruptionConfig,
        vad: Optional[VoiceActivityDetector] = None,
    ):
        self.config = config
        self.vad = vad

        # State
        self._is_ai_speaking = False
        self._ai_speech_start_ms: float = 0.0
        self._ai_speech_duration_ms: float = 0.0
        self._ai_current_position_ms: float = 0.0

        self._is_user_speaking = False
        self._user_speech_start_ms: float = 0.0
        self._user_speech_buffer = AudioBuffer(
            format=vad.config.sample_rate if vad else 16000,
            max_duration_ms=5000,
        ) if True else None

        self._is_holding = False
        self._hold_start_ms: float = 0.0

        # Interruption tracking
        self._current_interruption: Optional[InterruptionEvent] = None
        self._interruption_history: List[InterruptionEvent] = []
        self._max_history = 100

        # Callbacks
        self._on_interrupt: Optional[Callable[[InterruptionEvent], None]] = None
        self._on_resume: Optional[Callable[[], None]] = None
        self._on_stop: Optional[Callable[[], None]] = None

        # Text being spoken (for sentence detection)
        self._current_text: str = ""
        self._current_text_position: int = 0

        self._lock = threading.Lock()

    def set_callbacks(
        self,
        on_interrupt: Optional[Callable[[InterruptionEvent], None]] = None,
        on_resume: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
    ) -> None:
        """Set callbacks for interruption events."""
        self._on_interrupt = on_interrupt
        self._on_resume = on_resume
        self._on_stop = on_stop

    def start_ai_speech(
        self,
        text: str,
        duration_ms: float,
        timestamp_ms: Optional[float] = None,
    ) -> None:
        """
        Mark the start of AI speech.

        Args:
            text: Text being spoken
            duration_ms: Expected duration of speech
            timestamp_ms: Start timestamp (defaults to current time)
        """
        with self._lock:
            self._is_ai_speaking = True
            self._ai_speech_start_ms = timestamp_ms or time.time() * 1000
            self._ai_speech_duration_ms = duration_ms
            self._ai_current_position_ms = 0.0
            self._current_text = text
            self._current_text_position = 0

            logger.debug(f"AI speech started: {duration_ms:.0f}ms")

    def update_ai_position(self, position_ms: float) -> None:
        """Update current position in AI speech."""
        with self._lock:
            self._ai_current_position_ms = position_ms

            # Update text position estimate
            if self._current_text and self._ai_speech_duration_ms > 0:
                progress = position_ms / self._ai_speech_duration_ms
                self._current_text_position = int(len(self._current_text) * progress)

    def stop_ai_speech(self) -> None:
        """Mark the end of AI speech."""
        with self._lock:
            self._is_ai_speaking = False
            self._ai_speech_start_ms = 0.0
            self._ai_speech_duration_ms = 0.0
            self._ai_current_position_ms = 0.0
            self._current_text = ""
            self._current_text_position = 0

            logger.debug("AI speech stopped")

    async def process_user_audio(self, audio_chunk: AudioChunk) -> Optional[InterruptionEvent]:
        """
        Process user audio and detect interruptions.

        Args:
            audio_chunk: Audio from user

        Returns:
            InterruptionEvent if interruption detected, None otherwise
        """
        # Use VAD to detect speech
        if self.vad:
            vad_event = await self.vad.process(audio_chunk)

            if vad_event:
                if vad_event.event_type == VADEventType.SPEECH_START:
                    return await self._handle_user_speech_start(audio_chunk)
                elif vad_event.event_type == VADEventType.SPEECH_END:
                    return await self._handle_user_speech_end(audio_chunk)

            # Buffer audio during user speech
            if self.vad.is_speaking and self._user_speech_buffer:
                self._user_speech_buffer.append(audio_chunk)

        return None

    async def _handle_user_speech_start(
        self,
        audio_chunk: AudioChunk,
    ) -> Optional[InterruptionEvent]:
        """Handle user starting to speak."""
        current_time = audio_chunk.timestamp_ms or time.time() * 1000

        with self._lock:
            self._is_user_speaking = True
            self._user_speech_start_ms = current_time

            # Clear buffer and start collecting
            if self._user_speech_buffer:
                self._user_speech_buffer.clear()
                self._user_speech_buffer.append(audio_chunk)

            # Only process interruption if AI is speaking
            if not self._is_ai_speaking:
                return None

            # Create interruption event
            self._current_interruption = InterruptionEvent(
                event_type=InterruptionType.BARGE_IN,
                timestamp_ms=current_time,
                ai_position_ms=self._ai_current_position_ms,
                ai_remaining_ms=self._ai_speech_duration_ms - self._ai_current_position_ms,
            )

            # Apply strategy
            action = await self._apply_strategy_start()

            if action == "stop":
                self._current_interruption.action_taken = "stop"
                if self._on_interrupt:
                    await self._call_callback(self._on_interrupt, self._current_interruption)
                return self._current_interruption

            elif action == "hold":
                self._is_holding = True
                self._hold_start_ms = current_time
                self._current_interruption.action_taken = "hold"
                return None

            else:  # "continue"
                return None

    async def _handle_user_speech_end(
        self,
        audio_chunk: AudioChunk,
    ) -> Optional[InterruptionEvent]:
        """Handle user stopping speech."""
        current_time = audio_chunk.timestamp_ms or time.time() * 1000

        with self._lock:
            speech_duration = current_time - self._user_speech_start_ms
            self._is_user_speaking = False

            # Get buffered audio
            user_audio = None
            if self._user_speech_buffer:
                all_audio = self._user_speech_buffer.consume_all()
                if all_audio:
                    user_audio = all_audio.data

            # Update interruption event
            if self._current_interruption:
                self._current_interruption.duration_ms = speech_duration
                self._current_interruption.user_audio = user_audio

            # If we were holding, decide whether to resume or stop
            if self._is_holding:
                return await self._handle_hold_end(speech_duration)

            # Check if this was a significant interruption
            if speech_duration >= self.config.min_interruption_duration_ms:
                if self._current_interruption:
                    self._record_interruption(self._current_interruption)
                    return self._current_interruption

        return None

    async def _handle_hold_end(
        self,
        user_speech_duration: float,
    ) -> Optional[InterruptionEvent]:
        """Handle end of hold period."""
        self._is_holding = False

        # If user speech was brief, resume AI speech
        if user_speech_duration < self.config.hold_duration_ms:
            if self._on_resume:
                await self._call_callback(self._on_resume)

            if self._current_interruption:
                self._current_interruption.action_taken = "resumed"
                self._current_interruption.event_type = InterruptionType.AGREEMENT

            return None

        # User speech was long enough, treat as real interruption
        if self._current_interruption:
            self._current_interruption.action_taken = "stop"

            if self._on_stop:
                await self._call_callback(self._on_stop)

            self._record_interruption(self._current_interruption)
            return self._current_interruption

        return None

    async def _apply_strategy_start(self) -> str:
        """
        Apply interruption strategy at start of user speech.

        Returns:
            Action to take: "stop", "hold", or "continue"
        """
        strategy = self.config.strategy

        if strategy == InterruptionStrategy.IMMEDIATE:
            return "stop"

        elif strategy == InterruptionStrategy.IGNORE:
            return "continue"

        elif strategy == InterruptionStrategy.HOLD:
            return "hold"

        elif strategy == InterruptionStrategy.SENTENCE:
            # Complete current sentence
            if self._is_near_sentence_end():
                return "continue"
            return "hold"

        elif strategy == InterruptionStrategy.SMART:
            return await self._apply_smart_strategy()

        return "stop"

    async def _apply_smart_strategy(self) -> str:
        """
        Apply smart interruption strategy.

        Analyzes context to determine best action.
        """
        # If very early in AI speech, likely intentional interruption
        if self._ai_current_position_ms < 500:
            return "stop"

        # If near end of AI speech, let it finish
        if self._ai_remaining_ms() < 1000:
            return "continue"

        # If near sentence end, complete sentence
        if self._is_near_sentence_end():
            return "hold"

        # Default to hold for smart analysis
        return "hold"

    def _is_near_sentence_end(self, lookahead_chars: int = 20) -> bool:
        """Check if current position is near a sentence end."""
        if not self._current_text:
            return False

        pos = self._current_text_position
        lookahead = self._current_text[pos:pos + lookahead_chars]

        for marker in self.config.sentence_end_markers:
            if marker in lookahead:
                return True

        return False

    def _ai_remaining_ms(self) -> float:
        """Get remaining AI speech duration."""
        return max(0, self._ai_speech_duration_ms - self._ai_current_position_ms)

    def _record_interruption(self, event: InterruptionEvent) -> None:
        """Record an interruption event in history."""
        self._interruption_history.append(event)
        if len(self._interruption_history) > self._max_history:
            self._interruption_history.pop(0)

    async def _call_callback(
        self,
        callback: Callable,
        *args,
        **kwargs,
    ) -> None:
        """Call a callback safely."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in interruption callback: {e}")

    async def analyze_interruption(
        self,
        event: InterruptionEvent,
        transcription: str = "",
    ) -> InterruptionType:
        """
        Analyze an interruption to determine its type.

        Args:
            event: The interruption event
            transcription: Transcribed user speech

        Returns:
            Classified interruption type
        """
        if not transcription:
            return InterruptionType.UNKNOWN

        transcription_lower = transcription.lower().strip()
        event.transcription = transcription

        # Check for agreement indicators
        for word in self.config.agreement_words:
            if word in transcription_lower:
                return InterruptionType.AGREEMENT

        # Check for disagreement indicators
        for word in self.config.disagreement_words:
            if word in transcription_lower:
                return InterruptionType.DISAGREEMENT

        # Check for clarification requests
        clarification_indicators = [
            "what", "huh", "sorry", "repeat", "again",
            "didn't catch", "what do you mean", "could you"
        ]
        for indicator in clarification_indicators:
            if indicator in transcription_lower:
                return InterruptionType.CLARIFICATION

        # If it's a quick response (short duration, early in AI speech)
        if event.duration_ms < 1000 and event.ai_position_ms < 2000:
            return InterruptionType.QUICK_RESPONSE

        return InterruptionType.BARGE_IN

    def get_resume_position(self) -> Tuple[float, int]:
        """
        Get position to resume AI speech after interruption.

        Returns:
            Tuple of (position_ms, character_position)
        """
        with self._lock:
            if not self._current_text:
                return 0.0, 0

            # Find last sentence boundary before current position
            text_before = self._current_text[:self._current_text_position]
            last_boundary = 0

            for marker in self.config.sentence_end_markers:
                idx = text_before.rfind(marker)
                if idx > last_boundary:
                    last_boundary = idx

            # Back off by configured number of words
            words_before = text_before[:last_boundary + 1].split()
            if len(words_before) > self.config.backoff_words:
                # Find position of the word to resume from
                resume_from = len(words_before) - self.config.backoff_words
                resume_text = " ".join(words_before[:resume_from])
                resume_char_pos = len(resume_text)
            else:
                resume_char_pos = 0

            # Calculate time position
            if self._ai_speech_duration_ms > 0 and len(self._current_text) > 0:
                progress = resume_char_pos / len(self._current_text)
                resume_time_pos = progress * self._ai_speech_duration_ms
            else:
                resume_time_pos = 0.0

            return resume_time_pos, resume_char_pos

    def get_interruption_stats(self) -> Dict[str, Any]:
        """Get statistics about interruptions."""
        if not self._interruption_history:
            return {
                "total_interruptions": 0,
                "interruption_rate": 0.0,
            }

        by_type = {}
        by_action = {}
        total_duration = 0.0

        for event in self._interruption_history:
            event_type = event.event_type.value
            by_type[event_type] = by_type.get(event_type, 0) + 1

            action = event.action_taken
            by_action[action] = by_action.get(action, 0) + 1

            total_duration += event.duration_ms

        return {
            "total_interruptions": len(self._interruption_history),
            "by_type": by_type,
            "by_action": by_action,
            "avg_duration_ms": total_duration / len(self._interruption_history),
        }

    def reset(self) -> None:
        """Reset handler state."""
        with self._lock:
            self._is_ai_speaking = False
            self._is_user_speaking = False
            self._is_holding = False
            self._current_interruption = None
            self._current_text = ""
            self._current_text_position = 0
            if self._user_speech_buffer:
                self._user_speech_buffer.clear()


class InterruptionAnalyzer:
    """
    Analyzes interruption patterns for improving conversation flow.

    Tracks patterns and provides insights for optimizing interruption handling.
    """

    def __init__(self):
        self._events: List[InterruptionEvent] = []
        self._session_start: float = time.time()

    def record_event(self, event: InterruptionEvent) -> None:
        """Record an interruption event."""
        self._events.append(event)

    def get_pattern_analysis(self) -> Dict[str, Any]:
        """Analyze interruption patterns."""
        if not self._events:
            return {"patterns": [], "recommendations": []}

        patterns = []
        recommendations = []

        # Analyze timing patterns
        early_interruptions = [e for e in self._events if e.ai_position_ms < 1000]
        late_interruptions = [e for e in self._events if e.ai_remaining_ms < 1000]

        if len(early_interruptions) > len(self._events) * 0.3:
            patterns.append("frequent_early_interruptions")
            recommendations.append(
                "Users frequently interrupt early - consider shorter initial responses"
            )

        if len(late_interruptions) > len(self._events) * 0.3:
            patterns.append("frequent_near_end_interruptions")
            recommendations.append(
                "Users often interrupt near the end - consider sentence-based stopping"
            )

        # Analyze interruption types
        agreements = [e for e in self._events if e.event_type == InterruptionType.AGREEMENT]
        if len(agreements) > len(self._events) * 0.4:
            patterns.append("many_acknowledgments")
            recommendations.append(
                "Many interruptions are acknowledgments - consider HOLD strategy"
            )

        clarifications = [e for e in self._events if e.event_type == InterruptionType.CLARIFICATION]
        if len(clarifications) > len(self._events) * 0.2:
            patterns.append("frequent_clarification_requests")
            recommendations.append(
                "Users frequently ask for clarification - consider clearer responses"
            )

        return {
            "total_events": len(self._events),
            "session_duration_sec": time.time() - self._session_start,
            "patterns": patterns,
            "recommendations": recommendations,
            "type_distribution": self._get_type_distribution(),
        }

    def _get_type_distribution(self) -> Dict[str, float]:
        """Get distribution of interruption types."""
        if not self._events:
            return {}

        counts = {}
        for event in self._events:
            type_name = event.event_type.value
            counts[type_name] = counts.get(type_name, 0) + 1

        total = len(self._events)
        return {k: v / total for k, v in counts.items()}

    def suggest_strategy(self) -> InterruptionStrategy:
        """Suggest optimal interruption strategy based on patterns."""
        if not self._events:
            return InterruptionStrategy.SMART

        analysis = self.get_pattern_analysis()
        patterns = analysis.get("patterns", [])

        if "many_acknowledgments" in patterns:
            return InterruptionStrategy.HOLD

        if "frequent_early_interruptions" in patterns:
            return InterruptionStrategy.IMMEDIATE

        if "frequent_near_end_interruptions" in patterns:
            return InterruptionStrategy.SENTENCE

        return InterruptionStrategy.SMART

    def reset(self) -> None:
        """Reset analyzer state."""
        self._events.clear()
        self._session_start = time.time()
