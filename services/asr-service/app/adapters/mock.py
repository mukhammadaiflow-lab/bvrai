"""Mock ASR adapter for testing without external services."""

import asyncio
import random
import time
from typing import AsyncIterator, Optional

import structlog

from app.adapters.base import (
    ASRAdapter,
    TranscriptEvent,
    TranscriptResult,
    VADEvent,
    Word,
)

logger = structlog.get_logger()


class MockASRAdapter(ASRAdapter):
    """
    Mock ASR adapter for testing.

    Generates fake transcripts based on audio input patterns.
    Useful for development and testing without API costs.
    """

    # Sample responses for testing
    SAMPLE_PHRASES = [
        "hello how can i help you today",
        "i would like to schedule an appointment",
        "what are your business hours",
        "can you tell me more about your services",
        "i have a question about my account",
        "thank you for your help",
        "goodbye",
        "yes that sounds good",
        "no i don't think so",
        "can you repeat that please",
    ]

    def __init__(self, latency_ms: int = 100) -> None:
        super().__init__()
        self.latency_ms = latency_ms
        self._session_id: Optional[str] = None
        self._event_queue: asyncio.Queue[TranscriptEvent] = asyncio.Queue()
        self._audio_buffer: bytearray = bytearray()
        self._is_connected = False
        self._speech_active = False
        self._last_audio_time = 0.0
        self._silence_threshold = 0.5  # seconds

        self.logger = logger.bind(adapter="mock")

    @property
    def name(self) -> str:
        return "mock"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_interim_results(self) -> bool:
        return True

    async def connect(self, session_id: str, options: Optional[dict] = None) -> None:
        """Connect to mock ASR."""
        self._session_id = session_id
        self._is_connected = True
        self._audio_buffer = bytearray()

        self.logger.info("Mock ASR connected", session_id=session_id)

        # Start background processing
        asyncio.create_task(self._process_audio())

    async def disconnect(self) -> None:
        """Disconnect from mock ASR."""
        self._is_connected = False
        self.logger.info("Mock ASR disconnected", session_id=self._session_id)

    async def send_audio(self, audio: bytes) -> None:
        """Send audio to mock ASR."""
        if not self._is_connected:
            raise ConnectionError("Not connected")

        self._audio_buffer.extend(audio)
        self._last_audio_time = time.time()

        # Detect if this looks like speech (non-silence)
        if self._has_speech(audio) and not self._speech_active:
            self._speech_active = True
            event = TranscriptEvent(
                type="vad",
                vad_event=VADEvent.SPEECH_START,
                timestamp=time.time(),
            )
            await self._event_queue.put(event)

    def _has_speech(self, audio: bytes) -> bool:
        """Simple energy-based speech detection."""
        if len(audio) < 10:
            return False

        # Calculate average energy
        energy = sum(abs(b - 128) for b in audio) / len(audio)
        return energy > 10  # Threshold for "speech"

    async def _process_audio(self) -> None:
        """Background task to process audio and generate transcripts."""
        while self._is_connected:
            await asyncio.sleep(0.1)  # Check every 100ms

            # Check for silence (end of speech)
            if self._speech_active and time.time() - self._last_audio_time > self._silence_threshold:
                self._speech_active = False

                # Generate transcript
                if len(self._audio_buffer) > 1000:  # Minimum audio for transcript
                    await self._generate_transcript()

                # Send speech end event
                event = TranscriptEvent(
                    type="vad",
                    vad_event=VADEvent.SPEECH_END,
                    timestamp=time.time(),
                )
                await self._event_queue.put(event)

                self._audio_buffer = bytearray()

    async def _generate_transcript(self) -> None:
        """Generate a mock transcript based on audio buffer."""
        # Simulate processing latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Pick a random phrase
        text = random.choice(self.SAMPLE_PHRASES)

        # Generate words with timing
        words = []
        current_time = 0.0
        for word in text.split():
            word_duration = len(word) * 0.05  # ~50ms per character
            words.append(
                Word(
                    word=word,
                    start=current_time,
                    end=current_time + word_duration,
                    confidence=random.uniform(0.9, 1.0),
                )
            )
            current_time += word_duration + 0.1  # Gap between words

        # First send interim result
        interim = TranscriptResult(
            text=text[:len(text) // 2] + "...",
            is_final=False,
            confidence=0.8,
            words=words[:len(words) // 2],
            start_time=0.0,
            end_time=current_time / 2,
        )

        event = TranscriptEvent(
            type="transcript",
            transcript=interim,
            timestamp=time.time(),
        )
        await self._event_queue.put(event)
        await self._emit_event(event)

        # Small delay
        await asyncio.sleep(0.05)

        # Then send final result
        final = TranscriptResult(
            text=text,
            is_final=True,
            confidence=random.uniform(0.95, 0.99),
            words=words,
            start_time=0.0,
            end_time=current_time,
            speech_final=True,
        )

        event = TranscriptEvent(
            type="transcript",
            transcript=final,
            timestamp=time.time(),
        )
        await self._event_queue.put(event)
        await self._emit_event(event)

        self.logger.info(
            "Generated mock transcript",
            text=text,
            duration=current_time,
        )

    async def stream_transcripts(self) -> AsyncIterator[TranscriptEvent]:
        """Stream transcript events."""
        while self._is_connected:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=5.0,
                )
                yield event
            except asyncio.TimeoutError:
                continue

    async def finalize(self) -> Optional[TranscriptResult]:
        """Finalize and get last transcript."""
        if len(self._audio_buffer) > 1000:
            await self._generate_transcript()

            # Get the final transcript from queue
            while not self._event_queue.empty():
                event = await self._event_queue.get()
                if event.transcript and event.transcript.is_final:
                    return event.transcript

        return None
