"""Turn detection for determining when user has finished speaking."""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import structlog

logger = structlog.get_logger()


class TurnState(str, Enum):
    """State of the current turn."""
    IDLE = "idle"  # No speech detected
    SPEECH_STARTED = "speech_started"  # User started speaking
    SPEAKING = "speaking"  # User is speaking
    PAUSE = "pause"  # Brief pause in speech
    TURN_COMPLETE = "turn_complete"  # User finished speaking


@dataclass
class TurnConfig:
    """Configuration for turn detection."""

    # Silence duration to consider turn complete
    silence_threshold_ms: int = 700  # 700ms of silence

    # Minimum speech duration to consider valid
    min_speech_ms: int = 200  # At least 200ms of speech

    # Maximum pause within a turn
    max_pause_ms: int = 1500  # 1.5s pause still same turn

    # Timeout for no speech
    no_speech_timeout_ms: int = 10000  # 10s timeout

    # Use VAD events from ASR
    use_vad_events: bool = True

    # Use semantic endpointing (sentence completion)
    use_semantic_endpointing: bool = True

    # Sentences that indicate turn completion
    turn_ending_phrases: list[str] = None

    def __post_init__(self):
        if self.turn_ending_phrases is None:
            self.turn_ending_phrases = [
                "thank you",
                "thanks",
                "goodbye",
                "bye",
                "that's all",
                "that's it",
                "nevermind",
                "never mind",
            ]


@dataclass
class TurnInfo:
    """Information about the current turn."""
    state: TurnState = TurnState.IDLE
    started_at: float = 0.0
    last_speech_at: float = 0.0
    transcript: str = ""
    word_count: int = 0
    is_final: bool = False

    @property
    def duration_ms(self) -> int:
        """Duration of speech in milliseconds."""
        if self.started_at == 0:
            return 0
        return int((self.last_speech_at - self.started_at) * 1000)

    @property
    def silence_ms(self) -> int:
        """Duration of silence since last speech."""
        if self.last_speech_at == 0:
            return 0
        return int((time.time() - self.last_speech_at) * 1000)


class TurnDetector:
    """
    Detects when user has completed their turn in a conversation.

    Uses multiple signals:
    - Silence duration (primary)
    - VAD (Voice Activity Detection) events
    - Semantic endpointing (sentence completion)
    - Transcript finality
    """

    def __init__(
        self,
        config: Optional[TurnConfig] = None,
        on_turn_complete: Optional[Callable[[TurnInfo], None]] = None,
    ) -> None:
        self.config = config or TurnConfig()
        self._on_turn_complete = on_turn_complete

        self.turn = TurnInfo()
        self._check_task: Optional[asyncio.Task] = None
        self._is_running = False

        self.logger = logger.bind(component="turn_detector")

    async def start(self) -> None:
        """Start the turn detector background task."""
        self._is_running = True
        self._check_task = asyncio.create_task(self._check_loop())
        self.logger.debug("Turn detector started")

    async def stop(self) -> None:
        """Stop the turn detector."""
        self._is_running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        self.logger.debug("Turn detector stopped")

    def reset(self) -> None:
        """Reset for a new turn."""
        self.turn = TurnInfo()
        self.logger.debug("Turn reset")

    def on_speech_start(self) -> None:
        """Called when VAD detects speech start."""
        now = time.time()

        if self.turn.state == TurnState.IDLE:
            self.turn.state = TurnState.SPEECH_STARTED
            self.turn.started_at = now

        self.turn.last_speech_at = now
        self.turn.state = TurnState.SPEAKING

        self.logger.debug("Speech started")

    def on_speech_end(self) -> None:
        """Called when VAD detects speech end."""
        if self.turn.state == TurnState.SPEAKING:
            self.turn.state = TurnState.PAUSE
            self.logger.debug("Speech paused")

    def on_transcript(
        self,
        text: str,
        is_final: bool = False,
        speech_final: bool = False,
    ) -> Optional[TurnInfo]:
        """
        Process a transcript update.

        Returns TurnInfo if turn is complete, None otherwise.
        """
        now = time.time()

        # Update turn info
        self.turn.transcript = text
        self.turn.word_count = len(text.split())
        self.turn.is_final = is_final
        self.turn.last_speech_at = now

        if self.turn.state == TurnState.IDLE:
            self.turn.state = TurnState.SPEAKING
            self.turn.started_at = now

        # Check if turn is complete
        if speech_final:
            # ASR indicated end of speech
            return self._complete_turn("speech_final")

        if is_final and self._is_semantic_endpoint(text):
            # Semantic endpointing detected
            return self._complete_turn("semantic_endpoint")

        return None

    def _is_semantic_endpoint(self, text: str) -> bool:
        """Check if text ends at a semantic boundary."""
        if not self.config.use_semantic_endpointing:
            return False

        text_lower = text.lower().strip()

        # Check for explicit turn-ending phrases
        for phrase in self.config.turn_ending_phrases:
            if text_lower.endswith(phrase):
                return True

        # Check for sentence-ending punctuation
        if text.rstrip().endswith((".", "?", "!")):
            return True

        return False

    async def _check_loop(self) -> None:
        """Background loop to check for turn completion via silence."""
        while self._is_running:
            await asyncio.sleep(0.1)  # Check every 100ms

            if self.turn.state == TurnState.IDLE:
                continue

            if self.turn.state in [TurnState.SPEAKING, TurnState.PAUSE]:
                silence_ms = self.turn.silence_ms

                # Check if silence threshold exceeded
                if silence_ms >= self.config.silence_threshold_ms:
                    # Verify minimum speech duration
                    if self.turn.duration_ms >= self.config.min_speech_ms:
                        self._complete_turn("silence_threshold")

            # Check for no-speech timeout
            if self.turn.state == TurnState.IDLE:
                if self.turn.started_at == 0:
                    continue
                elapsed = (time.time() - self.turn.started_at) * 1000
                if elapsed >= self.config.no_speech_timeout_ms:
                    self._complete_turn("no_speech_timeout")

    def _complete_turn(self, reason: str) -> TurnInfo:
        """Mark turn as complete and notify."""
        self.turn.state = TurnState.TURN_COMPLETE

        self.logger.info(
            "Turn complete",
            reason=reason,
            duration_ms=self.turn.duration_ms,
            word_count=self.turn.word_count,
            transcript=self.turn.transcript[:100],
        )

        if self._on_turn_complete:
            self._on_turn_complete(self.turn)

        # Return a copy and reset
        completed_turn = TurnInfo(
            state=self.turn.state,
            started_at=self.turn.started_at,
            last_speech_at=self.turn.last_speech_at,
            transcript=self.turn.transcript,
            word_count=self.turn.word_count,
            is_final=True,
        )

        self.reset()
        return completed_turn

    def is_turn_complete(self) -> bool:
        """Check if current turn is complete."""
        return self.turn.state == TurnState.TURN_COMPLETE

    def get_turn_info(self) -> TurnInfo:
        """Get current turn information."""
        return self.turn
