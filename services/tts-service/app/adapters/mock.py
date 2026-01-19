"""Mock TTS adapter for testing without external services."""

import asyncio
import math
import struct
from typing import AsyncIterator, Optional

import structlog

from app.adapters.base import TTSAdapter, AudioChunk, VoiceInfo

logger = structlog.get_logger()


class MockTTSAdapter(TTSAdapter):
    """
    Mock TTS adapter for testing.

    Generates silence or simple tones instead of real speech.
    Useful for development and testing without API costs.
    """

    SAMPLE_RATE = 8000  # 8kHz for Twilio compatibility

    def __init__(self, latency_ms: int = 50) -> None:
        self.latency_ms = latency_ms
        self.logger = logger.bind(adapter="mock_tts")

    @property
    def name(self) -> str:
        return "mock"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """Generate mock audio for text."""
        # Calculate duration based on text length
        # Average speaking rate: ~150 words per minute = 2.5 words/sec
        # Average word length: ~5 characters
        words = len(text.split())
        duration_sec = words / 2.5

        # Generate μ-law encoded silence/tone
        samples = int(self.SAMPLE_RATE * duration_sec)
        audio = self._generate_mulaw_audio(samples, frequency=440 if kwargs.get("tone") else 0)

        self.logger.info(
            "Generated mock audio",
            text_length=len(text),
            duration_sec=duration_sec,
            audio_size=len(audio),
        )

        await asyncio.sleep(self.latency_ms / 1000)

        return audio

    async def synthesize_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[AudioChunk]:
        """Stream mock audio chunks."""
        words = len(text.split())
        duration_sec = words / 2.5
        total_samples = int(self.SAMPLE_RATE * duration_sec)

        # Stream in 20ms chunks (160 samples at 8kHz)
        chunk_samples = 160
        sequence = 0

        for i in range(0, total_samples, chunk_samples):
            remaining = min(chunk_samples, total_samples - i)
            audio = self._generate_mulaw_audio(remaining, frequency=0)

            sequence += 1
            yield AudioChunk(
                data=audio,
                sample_rate=self.SAMPLE_RATE,
                encoding="mulaw",
                sequence=sequence,
            )

            # Simulate real-time streaming delay
            await asyncio.sleep(0.02)  # 20ms per chunk

        # Final chunk
        yield AudioChunk(
            data=b"",
            sample_rate=self.SAMPLE_RATE,
            encoding="mulaw",
            sequence=sequence + 1,
            is_final=True,
        )

        self.logger.info(
            "Streamed mock audio",
            text_length=len(text),
            chunks=sequence,
        )

    def _generate_mulaw_audio(self, num_samples: int, frequency: float = 0) -> bytes:
        """Generate μ-law encoded audio samples."""
        output = bytearray()

        for i in range(num_samples):
            if frequency > 0:
                # Generate sine wave
                t = i / self.SAMPLE_RATE
                sample = int(16384 * math.sin(2 * math.pi * frequency * t))
            else:
                # Silence
                sample = 0

            # Encode to μ-law
            mulaw_byte = self._linear_to_mulaw(sample)
            output.append(mulaw_byte)

        return bytes(output)

    def _linear_to_mulaw(self, sample: int) -> int:
        """Convert 16-bit linear PCM to 8-bit μ-law."""
        MULAW_MAX = 0x1FFF
        MULAW_BIAS = 33

        sign = 0x80 if sample < 0 else 0
        if sample < 0:
            sample = -sample

        if sample > MULAW_MAX:
            sample = MULAW_MAX

        sample += MULAW_BIAS

        exponent = 7
        for exp in range(7, -1, -1):
            if sample & (1 << (exp + 7)):
                exponent = exp
                break

        mantissa = (sample >> (exponent + 3)) & 0x0F
        mulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF

        return mulaw_byte

    async def get_voices(self) -> list[VoiceInfo]:
        """Return mock voices."""
        return [
            VoiceInfo(
                voice_id="mock-female",
                name="Mock Female",
                description="Mock female voice for testing",
            ),
            VoiceInfo(
                voice_id="mock-male",
                name="Mock Male",
                description="Mock male voice for testing",
            ),
        ]
