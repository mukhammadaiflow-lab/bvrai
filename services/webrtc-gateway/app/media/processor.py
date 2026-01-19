"""Audio processing for WebRTC."""

import asyncio
from typing import AsyncIterator, Optional

import structlog

from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class AudioProcessor:
    """Processes audio for WebRTC sessions.

    Handles:
    - Audio format conversion (Opus to PCM and vice versa)
    - Sample rate conversion (48kHz WebRTC to 16kHz for ASR)
    - Audio buffering and chunking
    - Voice activity detection (basic)
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.logger = logger.bind(session_id=session_id, component="audio_processor")

        # Audio buffers
        self._input_buffer: list[bytes] = []
        self._output_buffer: list[bytes] = []

        # Processing state
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None

        # Configuration
        self.input_sample_rate = 48000  # WebRTC Opus
        self.output_sample_rate = settings.audio_sample_rate
        self.channels = settings.audio_channels

        # VAD state
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0

    async def start(self) -> None:
        """Start audio processing."""
        self._running = True
        self.logger.info("Audio processor started")

    async def stop(self) -> None:
        """Stop audio processing."""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        self._input_buffer.clear()
        self._output_buffer.clear()
        self.logger.info("Audio processor stopped")

    async def process_input(self, audio_data: bytes) -> Optional[bytes]:
        """Process incoming audio from WebRTC.

        Converts Opus/48kHz to PCM/16kHz for ASR.

        In production, this would use libraries like:
        - opuslib for Opus decoding
        - scipy or resampy for sample rate conversion
        """
        if not self._running:
            return None

        # Add to input buffer
        self._input_buffer.append(audio_data)

        # Basic VAD using energy threshold
        self._update_vad(audio_data)

        # In production: decode Opus, resample to 16kHz
        # For now, pass through (simulated)
        return audio_data

    async def process_output(self, audio_data: bytes) -> Optional[bytes]:
        """Process outgoing audio for WebRTC.

        Converts PCM/16kHz from TTS to Opus/48kHz for WebRTC.
        """
        if not self._running:
            return None

        # In production: upsample to 48kHz, encode to Opus
        # For now, pass through (simulated)
        return audio_data

    def _update_vad(self, audio_data: bytes) -> None:
        """Update voice activity detection state."""
        # Simple energy-based VAD
        if len(audio_data) < 2:
            return

        # Calculate RMS energy
        samples = [
            int.from_bytes(audio_data[i : i + 2], "little", signed=True)
            for i in range(0, len(audio_data), 2)
        ]

        if not samples:
            return

        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5

        # Threshold for speech detection
        speech_threshold = 500

        if rms > speech_threshold:
            self._speech_frames += 1
            self._silence_frames = 0

            if self._speech_frames > 3 and not self._is_speaking:
                self._is_speaking = True
                self.logger.debug("Speech detected")
        else:
            self._silence_frames += 1
            self._speech_frames = 0

            if self._silence_frames > 15 and self._is_speaking:
                self._is_speaking = False
                self.logger.debug("Silence detected")

    @property
    def is_speaking(self) -> bool:
        """Check if user is currently speaking."""
        return self._is_speaking

    async def stream_input(self) -> AsyncIterator[bytes]:
        """Stream processed input audio."""
        while self._running:
            if self._input_buffer:
                chunk = self._input_buffer.pop(0)
                yield chunk
            else:
                await asyncio.sleep(0.02)  # 20ms chunks


class AudioResampler:
    """Audio resampling utilities."""

    @staticmethod
    def resample(
        audio: bytes,
        from_rate: int,
        to_rate: int,
        channels: int = 1,
    ) -> bytes:
        """Resample audio to different sample rate.

        In production, use scipy.signal.resample or resampy.
        """
        if from_rate == to_rate:
            return audio

        # Simple linear interpolation (for demo)
        # Production should use proper resampling algorithm
        ratio = to_rate / from_rate

        samples = [
            int.from_bytes(audio[i : i + 2], "little", signed=True)
            for i in range(0, len(audio), 2)
        ]

        if not samples:
            return audio

        new_length = int(len(samples) * ratio)
        resampled = []

        for i in range(new_length):
            src_pos = i / ratio
            idx = int(src_pos)
            frac = src_pos - idx

            if idx + 1 < len(samples):
                sample = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
            else:
                sample = samples[idx]

            resampled.append(sample.to_bytes(2, "little", signed=True))

        return b"".join(resampled)


class OpusCodec:
    """Opus codec wrapper.

    In production, use opuslib or similar library.
    """

    def __init__(self, sample_rate: int = 48000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

    def encode(self, pcm_data: bytes) -> bytes:
        """Encode PCM to Opus.

        In production: use opuslib encoder.
        """
        # Placeholder - return as-is for demo
        return pcm_data

    def decode(self, opus_data: bytes) -> bytes:
        """Decode Opus to PCM.

        In production: use opuslib decoder.
        """
        # Placeholder - return as-is for demo
        return opus_data
