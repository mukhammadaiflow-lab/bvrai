"""DTMF (Dual-Tone Multi-Frequency) detection and generation."""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging
import struct
import math

logger = logging.getLogger(__name__)


class DTMFTone(str, Enum):
    """DTMF tone characters."""
    ZERO = "0"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    STAR = "*"
    HASH = "#"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


# DTMF frequency pairs (low freq, high freq)
DTMF_FREQUENCIES: Dict[str, Tuple[int, int]] = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "A": (697, 1633),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477), "B": (770, 1633),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "C": (852, 1633),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477), "D": (941, 1633),
}

# Reverse lookup for frequency to tone
FREQUENCY_TO_TONE: Dict[Tuple[int, int], str] = {v: k for k, v in DTMF_FREQUENCIES.items()}

# Low frequencies
LOW_FREQUENCIES = [697, 770, 852, 941]
# High frequencies
HIGH_FREQUENCIES = [1209, 1336, 1477, 1633]


@dataclass
class DTMFEvent:
    """A detected DTMF event."""
    tone: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tone": self.tone,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "confidence": self.confidence,
        }


@dataclass
class DTMFSequence:
    """A sequence of DTMF tones."""
    sequence_id: str
    tones: str = ""
    events: List[DTMFEvent] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    is_complete: bool = False
    terminator: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sequence_id": self.sequence_id,
            "tones": self.tones,
            "events": [e.to_dict() for e in self.events],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "is_complete": self.is_complete,
            "terminator": self.terminator,
        }


class DTMFDetector:
    """
    Detects DTMF tones from audio using Goertzel algorithm.

    Usage:
        detector = DTMFDetector()

        # Process audio
        async for tone in detector.detect_tones(audio_stream):
            print(f"Detected: {tone}")

        # Or process individual chunks
        tone = detector.process_chunk(audio_data)
        if tone:
            print(f"Detected: {tone}")
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        detection_threshold: float = 0.3,
        min_duration_ms: float = 40,
        max_gap_ms: float = 100,
    ):
        self.sample_rate = sample_rate
        self.detection_threshold = detection_threshold
        self.min_duration_ms = min_duration_ms
        self.max_gap_ms = max_gap_ms

        # Goertzel parameters
        self._block_size = int(sample_rate * 0.020)  # 20ms blocks
        self._current_tone: Optional[str] = None
        self._tone_start_time: Optional[datetime] = None
        self._last_tone_time: Optional[datetime] = None
        self._callbacks: List[Callable[[DTMFEvent], Awaitable[None]]] = []

    def process_chunk(
        self,
        audio_data: bytes,
    ) -> Optional[str]:
        """
        Process an audio chunk and detect DTMF tone.

        Returns detected tone or None.
        """
        # Convert bytes to samples
        if len(audio_data) < self._block_size * 2:
            return None

        samples = struct.unpack(f'{len(audio_data) // 2}h', audio_data)

        # Process in blocks
        tone = None
        for i in range(0, len(samples) - self._block_size, self._block_size):
            block = samples[i:i + self._block_size]
            detected = self._detect_block(block)
            if detected:
                tone = detected

        return tone

    def _detect_block(
        self,
        samples: tuple,
    ) -> Optional[str]:
        """Detect DTMF in a single block using Goertzel algorithm."""
        if not samples:
            return None

        # Normalize samples
        max_sample = max(abs(s) for s in samples) or 1
        normalized = [s / max_sample for s in samples]

        # Calculate magnitudes for all DTMF frequencies
        low_mags = {}
        high_mags = {}

        for freq in LOW_FREQUENCIES:
            low_mags[freq] = self._goertzel(normalized, freq, self.sample_rate)

        for freq in HIGH_FREQUENCIES:
            high_mags[freq] = self._goertzel(normalized, freq, self.sample_rate)

        # Find strongest low and high frequency
        low_freq = max(low_mags, key=low_mags.get)
        high_freq = max(high_mags, key=high_mags.get)

        low_mag = low_mags[low_freq]
        high_mag = high_mags[high_freq]

        # Check if magnitudes exceed threshold
        if low_mag < self.detection_threshold or high_mag < self.detection_threshold:
            return None

        # Check if no other frequencies are present (twist check)
        avg_low = sum(low_mags.values()) / len(low_mags)
        avg_high = sum(high_mags.values()) / len(high_mags)

        if low_mag < avg_low * 2 or high_mag < avg_high * 2:
            return None

        # Look up tone
        tone_key = (low_freq, high_freq)
        return FREQUENCY_TO_TONE.get(tone_key)

    def _goertzel(
        self,
        samples: list,
        target_freq: float,
        sample_rate: int,
    ) -> float:
        """Calculate Goertzel magnitude for target frequency."""
        n = len(samples)
        if n == 0:
            return 0.0

        k = int(0.5 + n * target_freq / sample_rate)
        w = 2 * math.pi * k / n
        coeff = 2 * math.cos(w)

        s0 = 0.0
        s1 = 0.0
        s2 = 0.0

        for sample in samples:
            s0 = sample + coeff * s1 - s2
            s2 = s1
            s1 = s0

        # Calculate magnitude
        magnitude = math.sqrt(s1 * s1 + s2 * s2 - coeff * s1 * s2)
        return magnitude / n

    async def detect_tones(
        self,
        audio_stream,
    ):
        """
        Async generator that yields detected DTMF tones.

        Args:
            audio_stream: Async iterator yielding audio chunks
        """
        async for chunk in audio_stream:
            tone = self.process_chunk(chunk)
            if tone:
                event = DTMFEvent(
                    tone=tone,
                    confidence=0.9,  # In real impl, calculate from magnitudes
                )
                yield event

    def on_tone(
        self,
        callback: Callable[[DTMFEvent], Awaitable[None]],
    ) -> None:
        """Register callback for detected tones."""
        self._callbacks.append(callback)

    async def _notify(self, event: DTMFEvent) -> None:
        """Notify callbacks of detected tone."""
        for callback in self._callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"DTMF callback error: {e}")


class DTMFGenerator:
    """
    Generates DTMF tones as audio.

    Usage:
        generator = DTMFGenerator()

        # Generate single tone
        audio = generator.generate_tone("5", duration_ms=100)

        # Generate sequence
        audio = generator.generate_sequence("1234#")
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        amplitude: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.amplitude = amplitude

    def generate_tone(
        self,
        tone: str,
        duration_ms: float = 100,
    ) -> bytes:
        """Generate audio for a single DTMF tone."""
        if tone.upper() not in DTMF_FREQUENCIES:
            raise ValueError(f"Invalid DTMF tone: {tone}")

        low_freq, high_freq = DTMF_FREQUENCIES[tone.upper()]
        return self._generate_dual_tone(low_freq, high_freq, duration_ms)

    def generate_sequence(
        self,
        sequence: str,
        tone_duration_ms: float = 100,
        gap_duration_ms: float = 50,
    ) -> bytes:
        """Generate audio for a DTMF sequence."""
        audio_parts = []

        for i, tone in enumerate(sequence):
            if tone.upper() in DTMF_FREQUENCIES:
                audio_parts.append(self.generate_tone(tone, tone_duration_ms))

                # Add gap between tones (except after last)
                if i < len(sequence) - 1:
                    audio_parts.append(self._generate_silence(gap_duration_ms))

        return b''.join(audio_parts)

    def _generate_dual_tone(
        self,
        freq1: float,
        freq2: float,
        duration_ms: float,
    ) -> bytes:
        """Generate dual-tone audio."""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        samples = []

        for i in range(num_samples):
            t = i / self.sample_rate
            # Combine two sine waves
            sample = (
                math.sin(2 * math.pi * freq1 * t) +
                math.sin(2 * math.pi * freq2 * t)
            ) * self.amplitude / 2

            # Convert to 16-bit signed integer
            sample_int = int(sample * 32767)
            samples.append(sample_int)

        return struct.pack(f'{len(samples)}h', *samples)

    def _generate_silence(
        self,
        duration_ms: float,
    ) -> bytes:
        """Generate silence."""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        return struct.pack(f'{num_samples}h', *([0] * num_samples))


class DTMFHandler:
    """
    Handles DTMF input for call flows.

    Usage:
        handler = DTMFHandler()

        # Set up handler for sequence
        handler.expect_sequence(
            name="pin_entry",
            length=4,
            terminator="#",
            callback=handle_pin,
        )

        # Process audio
        await handler.process_audio(audio_chunk)

        # Or manually add tones
        handler.add_tone("5")
    """

    def __init__(
        self,
        detector: Optional[DTMFDetector] = None,
        inter_digit_timeout_ms: float = 3000,
    ):
        self.detector = detector or DTMFDetector()
        self.inter_digit_timeout_ms = inter_digit_timeout_ms

        self._current_sequence: Optional[DTMFSequence] = None
        self._expected: Dict[str, Dict[str, Any]] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._last_tone_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

    def expect_sequence(
        self,
        name: str,
        length: Optional[int] = None,
        terminator: Optional[str] = "#",
        callback: Optional[Callable[[str], Awaitable[None]]] = None,
        timeout_ms: Optional[float] = None,
    ) -> None:
        """
        Set up expectation for a DTMF sequence.

        Args:
            name: Name for this expectation
            length: Expected length (if known)
            terminator: Character that ends the sequence
            callback: Function to call when sequence is complete
            timeout_ms: Timeout for sequence entry
        """
        self._expected[name] = {
            "length": length,
            "terminator": terminator,
            "timeout_ms": timeout_ms or self.inter_digit_timeout_ms,
        }
        if callback:
            self._callbacks[name] = callback

        # Start new sequence if not active
        if not self._current_sequence:
            import uuid
            self._current_sequence = DTMFSequence(
                sequence_id=f"dtmf_{uuid.uuid4().hex[:12]}",
                terminator=terminator,
            )

    def clear_expectations(self) -> None:
        """Clear all expectations."""
        self._expected = {}
        self._callbacks = {}
        self._current_sequence = None

    async def process_audio(
        self,
        audio_data: bytes,
    ) -> Optional[str]:
        """Process audio and return any detected tone."""
        tone = self.detector.process_chunk(audio_data)
        if tone:
            await self.add_tone(tone)
        return tone

    async def add_tone(
        self,
        tone: str,
    ) -> None:
        """Manually add a detected tone."""
        async with self._lock:
            if not self._current_sequence:
                import uuid
                self._current_sequence = DTMFSequence(
                    sequence_id=f"dtmf_{uuid.uuid4().hex[:12]}",
                )

            # Create event
            event = DTMFEvent(
                tone=tone,
                confidence=1.0,
            )
            self._current_sequence.events.append(event)
            self._current_sequence.tones += tone
            self._last_tone_time = datetime.utcnow()

            logger.debug(f"DTMF tone added: {tone}, sequence: {self._current_sequence.tones}")

            # Check if sequence is complete
            await self._check_sequence_complete()

    async def _check_sequence_complete(self) -> None:
        """Check if current sequence is complete."""
        if not self._current_sequence:
            return

        sequence = self._current_sequence

        for name, expectation in self._expected.items():
            length = expectation.get("length")
            terminator = expectation.get("terminator")

            is_complete = False

            # Check terminator
            if terminator and sequence.tones.endswith(terminator):
                is_complete = True
                # Remove terminator from result
                result = sequence.tones[:-1]

            # Check length
            elif length and len(sequence.tones) >= length:
                is_complete = True
                result = sequence.tones[:length]

            if is_complete:
                sequence.is_complete = True
                sequence.completed_at = datetime.utcnow()

                # Call callback
                callback = self._callbacks.get(name)
                if callback:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                    except Exception as e:
                        logger.error(f"DTMF callback error: {e}")

                # Reset for next sequence
                self._current_sequence = None
                break

    async def check_timeout(self) -> bool:
        """Check if sequence has timed out."""
        if not self._last_tone_time or not self._current_sequence:
            return False

        elapsed = (datetime.utcnow() - self._last_tone_time).total_seconds() * 1000

        for expectation in self._expected.values():
            timeout = expectation.get("timeout_ms", self.inter_digit_timeout_ms)
            if elapsed > timeout:
                # Timeout - complete with what we have
                if self._current_sequence.tones:
                    await self._complete_on_timeout()
                    return True

        return False

    async def _complete_on_timeout(self) -> None:
        """Complete sequence on timeout."""
        if not self._current_sequence:
            return

        sequence = self._current_sequence
        sequence.is_complete = True
        sequence.completed_at = datetime.utcnow()

        # Try to find matching callback
        for name in self._expected:
            callback = self._callbacks.get(name)
            if callback:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(sequence.tones)
                    else:
                        callback(sequence.tones)
                except Exception as e:
                    logger.error(f"DTMF timeout callback error: {e}")
                break

        self._current_sequence = None

    def get_current_sequence(self) -> str:
        """Get the current partial sequence."""
        if self._current_sequence:
            return self._current_sequence.tones
        return ""

    def reset(self) -> None:
        """Reset handler state."""
        self._current_sequence = None
        self._last_tone_time = None


class DTMFMenuHandler:
    """
    Handles DTMF menu navigation.

    Usage:
        menu = DTMFMenuHandler()

        # Define menu options
        menu.add_option("1", "Sales", handle_sales)
        menu.add_option("2", "Support", handle_support)
        menu.add_option("0", "Operator", handle_operator)

        # Process input
        result = await menu.process_input("1")
    """

    def __init__(self):
        self._options: Dict[str, Dict[str, Any]] = {}
        self._default_handler: Optional[Callable] = None
        self._invalid_handler: Optional[Callable] = None

    def add_option(
        self,
        key: str,
        label: str,
        handler: Callable,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a menu option."""
        self._options[key] = {
            "label": label,
            "handler": handler,
            "metadata": metadata or {},
        }

    def set_default_handler(
        self,
        handler: Callable,
    ) -> None:
        """Set handler for default/timeout."""
        self._default_handler = handler

    def set_invalid_handler(
        self,
        handler: Callable,
    ) -> None:
        """Set handler for invalid input."""
        self._invalid_handler = handler

    def get_options(self) -> Dict[str, str]:
        """Get menu options as key->label mapping."""
        return {k: v["label"] for k, v in self._options.items()}

    def get_prompt_text(self) -> str:
        """Generate prompt text for menu."""
        lines = []
        for key, option in self._options.items():
            lines.append(f"Press {key} for {option['label']}")
        return ". ".join(lines)

    async def process_input(
        self,
        input_key: str,
    ) -> bool:
        """
        Process menu input.

        Returns True if valid option was selected.
        """
        option = self._options.get(input_key)

        if option:
            handler = option["handler"]
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
                return True
            except Exception as e:
                logger.error(f"Menu handler error: {e}")
                return False
        else:
            # Invalid input
            if self._invalid_handler:
                try:
                    if asyncio.iscoroutinefunction(self._invalid_handler):
                        await self._invalid_handler()
                    else:
                        self._invalid_handler()
                except Exception as e:
                    logger.error(f"Invalid handler error: {e}")
            return False

    async def handle_timeout(self) -> None:
        """Handle menu timeout."""
        if self._default_handler:
            try:
                if asyncio.iscoroutinefunction(self._default_handler):
                    await self._default_handler()
                else:
                    self._default_handler()
            except Exception as e:
                logger.error(f"Default handler error: {e}")
