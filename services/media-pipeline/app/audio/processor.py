"""Audio processor - handles audio processing pipeline."""

import struct
import math
from typing import Optional, List
import numpy as np

from app.audio.vad import VoiceActivityDetector
from app.audio.agc import AutomaticGainControl
from app.audio.noise import NoiseSuppressor


class AudioProcessor:
    """
    Main audio processor for real-time audio enhancement.

    Features:
    - Automatic Gain Control (AGC)
    - Noise Suppression
    - Voice Activity Detection (VAD)
    - Level normalization
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        channels: int = 1,
        enable_agc: bool = True,
        enable_noise_suppression: bool = True,
        enable_vad: bool = True,
    ):
        self.sample_rate = sample_rate
        self.channels = channels

        # Initialize components
        self.agc = AutomaticGainControl(
            sample_rate=sample_rate,
            target_level_dbfs=-20.0,
        ) if enable_agc else None

        self.noise_suppressor = NoiseSuppressor(
            sample_rate=sample_rate,
        ) if enable_noise_suppression else None

        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate,
        ) if enable_vad else None

        # Processing state
        self._frame_count = 0
        self._total_energy = 0.0

    def process(self, audio_data: bytes) -> bytes:
        """
        Process audio through the enhancement pipeline.

        Args:
            audio_data: Raw PCM16 audio bytes

        Returns:
            Processed PCM16 audio bytes
        """
        # Convert to numpy array
        samples = self._bytes_to_samples(audio_data)

        # Noise suppression first
        if self.noise_suppressor:
            samples = self.noise_suppressor.process(samples)

        # Then AGC
        if self.agc:
            samples = self.agc.process(samples)

        self._frame_count += 1

        # Convert back to bytes
        return self._samples_to_bytes(samples)

    def detect_voice(self, audio_data: bytes) -> bool:
        """
        Detect voice activity in audio.

        Args:
            audio_data: PCM16 audio bytes

        Returns:
            True if voice detected
        """
        if not self.vad:
            return True  # Default to true if VAD disabled

        samples = self._bytes_to_samples(audio_data)
        return self.vad.is_speech(samples)

    def get_rms_level(self, audio_data: bytes) -> float:
        """
        Calculate RMS level of audio.

        Args:
            audio_data: PCM16 audio bytes

        Returns:
            RMS level in dBFS
        """
        samples = self._bytes_to_samples(audio_data)
        rms = np.sqrt(np.mean(samples ** 2))

        if rms > 0:
            return 20 * math.log10(rms / 32768.0)
        return -100.0

    def get_peak_level(self, audio_data: bytes) -> float:
        """
        Calculate peak level of audio.

        Args:
            audio_data: PCM16 audio bytes

        Returns:
            Peak level in dBFS
        """
        samples = self._bytes_to_samples(audio_data)
        peak = np.max(np.abs(samples))

        if peak > 0:
            return 20 * math.log10(peak / 32768.0)
        return -100.0

    def normalize(self, audio_data: bytes, target_dbfs: float = -3.0) -> bytes:
        """
        Normalize audio to target level.

        Args:
            audio_data: PCM16 audio bytes
            target_dbfs: Target level in dBFS

        Returns:
            Normalized PCM16 audio bytes
        """
        samples = self._bytes_to_samples(audio_data)
        peak = np.max(np.abs(samples))

        if peak == 0:
            return audio_data

        current_dbfs = 20 * math.log10(peak / 32768.0)
        gain_db = target_dbfs - current_dbfs
        gain_linear = 10 ** (gain_db / 20)

        # Apply gain with clipping protection
        samples = np.clip(samples * gain_linear, -32768, 32767).astype(np.int16)

        return self._samples_to_bytes(samples)

    def mix_audio(self, *audio_streams: bytes) -> bytes:
        """
        Mix multiple audio streams together.

        Args:
            audio_streams: Variable number of PCM16 audio byte streams

        Returns:
            Mixed PCM16 audio bytes
        """
        if not audio_streams:
            return b""

        if len(audio_streams) == 1:
            return audio_streams[0]

        # Convert all to numpy
        arrays = []
        max_len = 0

        for stream in audio_streams:
            samples = self._bytes_to_samples(stream)
            arrays.append(samples)
            max_len = max(max_len, len(samples))

        # Pad shorter arrays
        padded = []
        for arr in arrays:
            if len(arr) < max_len:
                arr = np.pad(arr, (0, max_len - len(arr)))
            padded.append(arr.astype(np.float32))

        # Mix with equal weighting
        mixed = sum(padded) / len(padded)

        # Clip and convert
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

        return self._samples_to_bytes(mixed)

    def resample(
        self,
        audio_data: bytes,
        from_rate: int,
        to_rate: int,
    ) -> bytes:
        """
        Resample audio to different sample rate.

        Args:
            audio_data: PCM16 audio bytes
            from_rate: Source sample rate
            to_rate: Target sample rate

        Returns:
            Resampled PCM16 audio bytes
        """
        if from_rate == to_rate:
            return audio_data

        samples = self._bytes_to_samples(audio_data)

        # Simple linear interpolation resampling
        ratio = to_rate / from_rate
        new_length = int(len(samples) * ratio)

        # Create interpolation indices
        indices = np.linspace(0, len(samples) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(samples)), samples)

        return self._samples_to_bytes(resampled.astype(np.int16))

    def _bytes_to_samples(self, audio_data: bytes) -> np.ndarray:
        """Convert PCM16 bytes to numpy array."""
        return np.frombuffer(audio_data, dtype=np.int16)

    def _samples_to_bytes(self, samples: np.ndarray) -> bytes:
        """Convert numpy array to PCM16 bytes."""
        return samples.astype(np.int16).tobytes()

    def reset(self) -> None:
        """Reset processor state."""
        self._frame_count = 0
        self._total_energy = 0.0

        if self.agc:
            self.agc.reset()
        if self.noise_suppressor:
            self.noise_suppressor.reset()
        if self.vad:
            self.vad.reset()
