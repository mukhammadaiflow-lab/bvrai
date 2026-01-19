"""Interrupt (barge-in) handler for detecting when user interrupts agent."""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import structlog

logger = structlog.get_logger()


class InterruptType(str, Enum):
    """Type of interruption."""
    NONE = "none"
    SOFT = "soft"  # Brief overlap, continue speaking
    HARD = "hard"  # User wants to take over, stop speaking


@dataclass
class InterruptConfig:
    """Configuration for interrupt detection."""

    # Enable interrupt detection
    enabled: bool = True

    # Minimum speech duration to trigger interrupt
    min_speech_ms: int = 200  # 200ms of user speech

    # Speech overlap tolerance before interrupt
    overlap_tolerance_ms: int = 300  # 300ms overlap allowed

    # Energy threshold for interrupt (relative to baseline)
    energy_threshold: float = 1.5

    # Debounce time to prevent false positives
    debounce_ms: int = 100

    # Hard interrupt phrases
    hard_interrupt_phrases: list[str] = None

    def __post_init__(self):
        if self.hard_interrupt_phrases is None:
            self.hard_interrupt_phrases = [
                "stop",
                "wait",
                "hold on",
                "excuse me",
                "actually",
                "sorry",
                "one moment",
            ]


@dataclass
class InterruptEvent:
    """An interrupt event."""
    type: InterruptType
    timestamp: float
    speech_duration_ms: int
    transcript: str = ""
    agent_was_speaking: bool = False


class InterruptHandler:
    """
    Handles barge-in detection during agent speech.

    Determines when user speech during agent speaking should:
    - Be ignored (background noise, overlap)
    - Trigger a soft interrupt (pause briefly)
    - Trigger a hard interrupt (stop speaking, listen to user)
    """

    def __init__(
        self,
        config: Optional[InterruptConfig] = None,
        on_interrupt: Optional[Callable[[InterruptEvent], None]] = None,
    ) -> None:
        self.config = config or InterruptConfig()
        self._on_interrupt = on_interrupt

        self._is_agent_speaking = False
        self._agent_speech_start: float = 0.0
        self._user_speech_start: Optional[float] = None
        self._last_interrupt_time: float = 0.0
        self._pending_transcript: str = ""

        self.logger = logger.bind(component="interrupt_handler")

    def agent_started_speaking(self) -> None:
        """Called when agent starts speaking."""
        self._is_agent_speaking = True
        self._agent_speech_start = time.time()
        self.logger.debug("Agent started speaking")

    def agent_stopped_speaking(self) -> None:
        """Called when agent stops speaking."""
        self._is_agent_speaking = False
        self._user_speech_start = None
        self.logger.debug("Agent stopped speaking")

    def on_user_speech_start(self) -> Optional[InterruptEvent]:
        """
        Called when user starts speaking.

        Returns InterruptEvent if interrupt should be triggered.
        """
        if not self.config.enabled:
            return None

        if not self._is_agent_speaking:
            return None  # No interrupt if agent isn't speaking

        now = time.time()
        self._user_speech_start = now

        self.logger.debug("User speech detected during agent speaking")

        return None  # Wait for more speech before deciding

    def on_user_speech_end(self) -> Optional[InterruptEvent]:
        """Called when user speech ends."""
        self._user_speech_start = None
        self._pending_transcript = ""
        return None

    def on_user_transcript(
        self,
        text: str,
        is_final: bool = False,
    ) -> Optional[InterruptEvent]:
        """
        Process user transcript during agent speaking.

        Returns InterruptEvent if interrupt should be triggered.
        """
        if not self.config.enabled:
            return None

        if not self._is_agent_speaking:
            return None

        self._pending_transcript = text
        now = time.time()

        # Check debounce
        if now - self._last_interrupt_time < self.config.debounce_ms / 1000:
            return None

        # Calculate speech duration
        if self._user_speech_start:
            speech_duration_ms = int((now - self._user_speech_start) * 1000)
        else:
            speech_duration_ms = 0

        # Check if speech is long enough
        if speech_duration_ms < self.config.min_speech_ms:
            return None

        # Determine interrupt type
        interrupt_type = self._determine_interrupt_type(text, speech_duration_ms)

        if interrupt_type != InterruptType.NONE:
            event = InterruptEvent(
                type=interrupt_type,
                timestamp=now,
                speech_duration_ms=speech_duration_ms,
                transcript=text,
                agent_was_speaking=True,
            )

            self._last_interrupt_time = now

            self.logger.info(
                "Interrupt detected",
                type=interrupt_type.value,
                speech_duration_ms=speech_duration_ms,
                transcript=text[:50],
            )

            if self._on_interrupt:
                self._on_interrupt(event)

            return event

        return None

    def _determine_interrupt_type(
        self,
        text: str,
        speech_duration_ms: int,
    ) -> InterruptType:
        """Determine the type of interrupt based on speech content and duration."""
        text_lower = text.lower().strip()

        # Check for hard interrupt phrases
        for phrase in self.config.hard_interrupt_phrases:
            if phrase in text_lower:
                return InterruptType.HARD

        # Long speech = hard interrupt
        if speech_duration_ms > 1000:  # More than 1 second
            return InterruptType.HARD

        # Multiple words = likely intentional
        word_count = len(text.split())
        if word_count >= 3:
            return InterruptType.HARD

        # Short single word might be overlap
        if word_count == 1 and speech_duration_ms < 500:
            return InterruptType.SOFT

        # Default to soft for short utterances
        if speech_duration_ms < self.config.overlap_tolerance_ms:
            return InterruptType.SOFT

        return InterruptType.HARD

    def should_stop_speaking(self) -> bool:
        """Check if agent should stop speaking due to interrupt."""
        if not self._is_agent_speaking:
            return False

        if not self._user_speech_start:
            return False

        now = time.time()
        speech_duration = (now - self._user_speech_start) * 1000

        return speech_duration >= self.config.min_speech_ms

    def clear(self) -> None:
        """Clear interrupt state."""
        self._user_speech_start = None
        self._pending_transcript = ""


class InterruptManager:
    """
    Manages interrupt handling and coordinates with TTS.

    Handles:
    - Clearing TTS audio buffer on hard interrupt
    - Pausing TTS on soft interrupt
    - Resuming TTS after soft interrupt
    """

    def __init__(
        self,
        handler: InterruptHandler,
        clear_audio_callback: Optional[Callable[[], asyncio.Future]] = None,
        pause_audio_callback: Optional[Callable[[], None]] = None,
        resume_audio_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.handler = handler
        self._clear_audio = clear_audio_callback
        self._pause_audio = pause_audio_callback
        self._resume_audio = resume_audio_callback

        self._is_paused = False
        self.logger = logger.bind(component="interrupt_manager")

    async def handle_interrupt(self, event: InterruptEvent) -> None:
        """Handle an interrupt event."""
        if event.type == InterruptType.HARD:
            await self._handle_hard_interrupt(event)
        elif event.type == InterruptType.SOFT:
            await self._handle_soft_interrupt(event)

    async def _handle_hard_interrupt(self, event: InterruptEvent) -> None:
        """Handle hard interrupt - stop speaking immediately."""
        self.logger.info("Handling hard interrupt")

        # Clear audio buffer
        if self._clear_audio:
            await self._clear_audio()

        # Notify that agent stopped speaking
        self.handler.agent_stopped_speaking()

    async def _handle_soft_interrupt(self, event: InterruptEvent) -> None:
        """Handle soft interrupt - pause briefly."""
        self.logger.debug("Handling soft interrupt")

        if self._pause_audio:
            self._pause_audio()
            self._is_paused = True

        # Wait briefly to see if user continues
        await asyncio.sleep(0.5)

        # If no more speech, resume
        if not self.handler.should_stop_speaking() and self._is_paused:
            if self._resume_audio:
                self._resume_audio()
            self._is_paused = False
        else:
            # Convert to hard interrupt
            if self._clear_audio:
                await self._clear_audio()
            self.handler.agent_stopped_speaking()
