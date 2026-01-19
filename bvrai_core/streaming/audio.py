"""
Audio Processing Module
=======================

Audio processing utilities including buffering, format conversion,
voice activity detection, and audio manipulation.

Author: Platform Engineering Team
Version: 2.0.0
"""

from __future__ import annotations

import math
import struct
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import (
    Any,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


class AudioFormat(str, Enum):
    """Audio formats."""

    PCM_S16LE = "pcm_s16le"      # 16-bit signed, little-endian
    PCM_S16BE = "pcm_s16be"      # 16-bit signed, big-endian
    PCM_F32LE = "pcm_f32le"      # 32-bit float, little-endian
    MULAW = "mulaw"              # G.711 mu-law
    ALAW = "alaw"                # G.711 A-law


class AudioCodec(str, Enum):
    """Audio codecs."""

    PCM = "pcm"
    OPUS = "opus"
    G711_ULAW = "g711_ulaw"
    G711_ALAW = "g711_alaw"
    AAC = "aac"
    MP3 = "mp3"


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""

    id: str = field(default_factory=lambda: f"chunk_{uuid4().hex[:12]}")
    data: bytes = b""
    timestamp_ms: int = 0
    sequence: int = 0
    duration_ms: int = 0

    # Audio properties
    sample_rate: int = 16000
    channels: int = 1
    format: AudioFormat = AudioFormat.PCM_S16LE

    # Metadata
    is_speech: bool = True
    audio_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        """Get number of samples in chunk."""
        bytes_per_sample = 2 if "16" in self.format.value else 4
        return len(self.data) // (bytes_per_sample * self.channels)

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return self.sample_count / self.sample_rate if self.sample_rate > 0 else 0

    def to_samples(self) -> List[float]:
        """Convert to normalized float samples (-1.0 to 1.0)."""
        if self.format in (AudioFormat.PCM_S16LE, AudioFormat.PCM_S16BE):
            fmt = "<h" if self.format == AudioFormat.PCM_S16LE else ">h"
            samples = []
            for i in range(0, len(self.data), 2):
                if i + 2 <= len(self.data):
                    sample = struct.unpack(fmt, self.data[i:i+2])[0]
                    samples.append(sample / 32768.0)
            return samples
        elif self.format == AudioFormat.PCM_F32LE:
            samples = []
            for i in range(0, len(self.data), 4):
                if i + 4 <= len(self.data):
                    sample = struct.unpack("<f", self.data[i:i+4])[0]
                    samples.append(sample)
            return samples
        return []

    @classmethod
    def from_samples(
        cls,
        samples: List[float],
        sample_rate: int = 16000,
        channels: int = 1,
        format: AudioFormat = AudioFormat.PCM_S16LE,
    ) -> "AudioChunk":
        """Create chunk from normalized float samples."""
        if format in (AudioFormat.PCM_S16LE, AudioFormat.PCM_S16BE):
            fmt = "<h" if format == AudioFormat.PCM_S16LE else ">h"
            data = bytearray()
            for sample in samples:
                clamped = max(-1.0, min(1.0, sample))
                int_sample = int(clamped * 32767)
                data.extend(struct.pack(fmt, int_sample))
            return cls(
                data=bytes(data),
                sample_rate=sample_rate,
                channels=channels,
                format=format,
            )
        return cls()


class AudioBuffer:
    """
    Audio buffer for managing streaming audio data.

    Provides buffering, chunking, and overflow handling.
    """

    def __init__(
        self,
        max_duration_ms: int = 5000,
        sample_rate: int = 16000,
        channels: int = 1,
        bits_per_sample: int = 16,
    ):
        self._max_duration = max_duration_ms
        self._sample_rate = sample_rate
        self._channels = channels
        self._bits_per_sample = bits_per_sample

        bytes_per_sample = bits_per_sample // 8
        self._bytes_per_ms = sample_rate * channels * bytes_per_sample // 1000
        self._max_size = self._bytes_per_ms * max_duration_ms

        self._buffer = bytearray()
        self._timestamp_ms = 0
        self._sequence = 0

        # Statistics
        self._total_bytes = 0
        self._overflow_count = 0

    def write(self, data: bytes) -> int:
        """Write data to buffer."""
        # Handle overflow
        if len(self._buffer) + len(data) > self._max_size:
            overflow = len(self._buffer) + len(data) - self._max_size
            self._buffer = self._buffer[overflow:]
            self._overflow_count += 1

        self._buffer.extend(data)
        self._total_bytes += len(data)
        return len(data)

    def read(self, size: int) -> bytes:
        """Read data from buffer."""
        data = bytes(self._buffer[:size])
        self._buffer = self._buffer[size:]
        return data

    def read_chunk(self, duration_ms: int = 20) -> Optional[AudioChunk]:
        """Read a chunk of specified duration."""
        chunk_size = self._bytes_per_ms * duration_ms

        if len(self._buffer) < chunk_size:
            return None

        data = self.read(chunk_size)
        chunk = AudioChunk(
            data=data,
            timestamp_ms=self._timestamp_ms,
            sequence=self._sequence,
            duration_ms=duration_ms,
            sample_rate=self._sample_rate,
            channels=self._channels,
        )

        self._timestamp_ms += duration_ms
        self._sequence += 1

        return chunk

    def iter_chunks(self, duration_ms: int = 20) -> Iterator[AudioChunk]:
        """Iterate over available chunks."""
        while True:
            chunk = self.read_chunk(duration_ms)
            if chunk is None:
                break
            yield chunk

    def peek(self, size: int) -> bytes:
        """Peek at data without consuming."""
        return bytes(self._buffer[:size])

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    @property
    def size(self) -> int:
        """Get current buffer size in bytes."""
        return len(self._buffer)

    @property
    def duration_ms(self) -> int:
        """Get current buffer duration in milliseconds."""
        return len(self._buffer) // self._bytes_per_ms if self._bytes_per_ms > 0 else 0

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0

    @property
    def overflow_count(self) -> int:
        """Get number of buffer overflows."""
        return self._overflow_count


class VoiceActivityDetector:
    """
    Voice Activity Detection (VAD).

    Detects presence of speech in audio using energy-based and
    zero-crossing rate methods.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.01,
        zcr_threshold: float = 0.1,
        speech_pad_ms: int = 300,
        min_speech_duration_ms: int = 250,
    ):
        self._sample_rate = sample_rate
        self._frame_duration = frame_duration_ms
        self._energy_threshold = energy_threshold
        self._zcr_threshold = zcr_threshold
        self._speech_pad = speech_pad_ms
        self._min_speech_duration = min_speech_duration_ms

        # Frame size in samples
        self._frame_size = sample_rate * frame_duration_ms // 1000

        # State
        self._is_speech = False
        self._speech_start_ms: Optional[int] = None
        self._silence_start_ms: Optional[int] = None
        self._current_time_ms = 0

        # Energy smoothing
        self._energy_history: Deque[float] = deque(maxlen=10)
        self._smoothed_energy = 0.0

    def process(self, chunk: AudioChunk) -> Tuple[bool, float]:
        """
        Process an audio chunk for voice activity.

        Returns:
            Tuple of (is_speech, confidence)
        """
        samples = chunk.to_samples()
        if not samples:
            return False, 0.0

        # Calculate frame energy
        energy = self._calculate_energy(samples)
        self._energy_history.append(energy)
        self._smoothed_energy = sum(self._energy_history) / len(self._energy_history)

        # Calculate zero-crossing rate
        zcr = self._calculate_zcr(samples)

        # Determine if speech
        is_active = (
            self._smoothed_energy > self._energy_threshold
            or (energy > self._energy_threshold * 0.5 and zcr < self._zcr_threshold)
        )

        # Apply speech padding and minimum duration
        if is_active:
            if not self._is_speech:
                self._speech_start_ms = self._current_time_ms
            self._silence_start_ms = None
            self._is_speech = True
        else:
            if self._is_speech:
                if self._silence_start_ms is None:
                    self._silence_start_ms = self._current_time_ms
                elif (self._current_time_ms - self._silence_start_ms) > self._speech_pad:
                    # Check minimum speech duration
                    if self._speech_start_ms is not None:
                        speech_duration = self._silence_start_ms - self._speech_start_ms
                        if speech_duration < self._min_speech_duration:
                            self._is_speech = False
                        else:
                            self._is_speech = False
                    else:
                        self._is_speech = False

        self._current_time_ms += chunk.duration_ms

        # Calculate confidence
        confidence = min(1.0, self._smoothed_energy / (self._energy_threshold * 2))
        if not self._is_speech:
            confidence = 0.0

        return self._is_speech, confidence

    def _calculate_energy(self, samples: List[float]) -> float:
        """Calculate RMS energy of samples."""
        if not samples:
            return 0.0
        squared = [s * s for s in samples]
        mean_squared = sum(squared) / len(squared)
        return math.sqrt(mean_squared)

    def _calculate_zcr(self, samples: List[float]) -> float:
        """Calculate zero-crossing rate."""
        if len(samples) < 2:
            return 0.0
        crossings = sum(
            1 for i in range(1, len(samples))
            if (samples[i] >= 0) != (samples[i-1] >= 0)
        )
        return crossings / len(samples)

    def reset(self) -> None:
        """Reset detector state."""
        self._is_speech = False
        self._speech_start_ms = None
        self._silence_start_ms = None
        self._energy_history.clear()
        self._smoothed_energy = 0.0


class AudioResampler:
    """
    Audio resampling utilities.

    Handles sample rate conversion and format conversion.
    """

    @staticmethod
    def resample(
        samples: List[float],
        input_rate: int,
        output_rate: int,
    ) -> List[float]:
        """
        Resample audio using linear interpolation.

        For production use, consider using a proper resampling library
        like scipy.signal.resample or librosa.resample.
        """
        if input_rate == output_rate:
            return samples

        ratio = output_rate / input_rate
        output_length = int(len(samples) * ratio)

        if output_length == 0:
            return []

        resampled = []
        for i in range(output_length):
            src_pos = i / ratio
            src_idx = int(src_pos)
            frac = src_pos - src_idx

            if src_idx + 1 < len(samples):
                sample = samples[src_idx] * (1 - frac) + samples[src_idx + 1] * frac
            else:
                sample = samples[src_idx] if src_idx < len(samples) else 0
            resampled.append(sample)

        return resampled

    @staticmethod
    def convert_format(
        data: bytes,
        input_format: AudioFormat,
        output_format: AudioFormat,
    ) -> bytes:
        """Convert between audio formats."""
        if input_format == output_format:
            return data

        # First convert to float samples
        if input_format in (AudioFormat.PCM_S16LE, AudioFormat.PCM_S16BE):
            fmt = "<h" if input_format == AudioFormat.PCM_S16LE else ">h"
            samples = []
            for i in range(0, len(data), 2):
                if i + 2 <= len(data):
                    sample = struct.unpack(fmt, data[i:i+2])[0]
                    samples.append(sample / 32768.0)
        elif input_format == AudioFormat.PCM_F32LE:
            samples = []
            for i in range(0, len(data), 4):
                if i + 4 <= len(data):
                    samples.append(struct.unpack("<f", data[i:i+4])[0])
        elif input_format == AudioFormat.MULAW:
            samples = [AudioResampler._mulaw_decode(b) for b in data]
        elif input_format == AudioFormat.ALAW:
            samples = [AudioResampler._alaw_decode(b) for b in data]
        else:
            return data

        # Then convert to output format
        if output_format in (AudioFormat.PCM_S16LE, AudioFormat.PCM_S16BE):
            fmt = "<h" if output_format == AudioFormat.PCM_S16LE else ">h"
            output = bytearray()
            for sample in samples:
                clamped = max(-1.0, min(1.0, sample))
                int_sample = int(clamped * 32767)
                output.extend(struct.pack(fmt, int_sample))
            return bytes(output)
        elif output_format == AudioFormat.PCM_F32LE:
            output = bytearray()
            for sample in samples:
                output.extend(struct.pack("<f", sample))
            return bytes(output)
        elif output_format == AudioFormat.MULAW:
            return bytes(AudioResampler._mulaw_encode(s) for s in samples)
        elif output_format == AudioFormat.ALAW:
            return bytes(AudioResampler._alaw_encode(s) for s in samples)

        return data

    @staticmethod
    def _mulaw_encode(sample: float) -> int:
        """Encode sample to mu-law."""
        MU = 255
        sign = 1 if sample >= 0 else -1
        sample = abs(sample)
        compressed = sign * math.log(1 + MU * sample) / math.log(1 + MU)
        return int((compressed + 1) / 2 * 255)

    @staticmethod
    def _mulaw_decode(encoded: int) -> float:
        """Decode mu-law to sample."""
        MU = 255
        normalized = (encoded / 255) * 2 - 1
        sign = 1 if normalized >= 0 else -1
        return sign * (math.pow(1 + MU, abs(normalized)) - 1) / MU

    @staticmethod
    def _alaw_encode(sample: float) -> int:
        """Encode sample to A-law."""
        A = 87.6
        sign = 1 if sample >= 0 else -1
        sample = abs(sample)

        if sample < 1/A:
            compressed = A * sample / (1 + math.log(A))
        else:
            compressed = (1 + math.log(A * sample)) / (1 + math.log(A))

        compressed = sign * compressed
        return int((compressed + 1) / 2 * 255)

    @staticmethod
    def _alaw_decode(encoded: int) -> float:
        """Decode A-law to sample."""
        A = 87.6
        normalized = (encoded / 255) * 2 - 1
        sign = 1 if normalized >= 0 else -1
        normalized = abs(normalized)

        log_a = math.log(A)
        if normalized < 1 / (1 + log_a):
            sample = normalized * (1 + log_a) / A
        else:
            sample = math.exp(normalized * (1 + log_a) - 1) / A

        return sign * sample


class AudioProcessor:
    """
    High-level audio processing service.

    Combines buffering, VAD, and format conversion for stream processing.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        format: AudioFormat = AudioFormat.PCM_S16LE,
        enable_vad: bool = True,
        buffer_duration_ms: int = 5000,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._format = format
        self._enable_vad = enable_vad

        self._buffer = AudioBuffer(
            max_duration_ms=buffer_duration_ms,
            sample_rate=sample_rate,
            channels=channels,
        )

        self._vad = VoiceActivityDetector(sample_rate=sample_rate) if enable_vad else None
        self._resampler = AudioResampler()

        # Processing state
        self._processed_chunks = 0
        self._speech_chunks = 0
        self._silence_chunks = 0
        self._total_duration_ms = 0

        self._logger = structlog.get_logger("audio_processor")

    def write(self, data: bytes) -> None:
        """Write raw audio data to processor."""
        self._buffer.write(data)

    def process(self, chunk_duration_ms: int = 20) -> Iterator[AudioChunk]:
        """Process buffered audio and yield chunks."""
        for chunk in self._buffer.iter_chunks(chunk_duration_ms):
            # Apply VAD
            if self._vad:
                is_speech, confidence = self._vad.process(chunk)
                chunk.is_speech = is_speech
                chunk.audio_level = confidence

                if is_speech:
                    self._speech_chunks += 1
                else:
                    self._silence_chunks += 1
            else:
                chunk.is_speech = True
                chunk.audio_level = self._calculate_level(chunk)

            self._processed_chunks += 1
            self._total_duration_ms += chunk.duration_ms

            yield chunk

    def _calculate_level(self, chunk: AudioChunk) -> float:
        """Calculate audio level for chunk."""
        samples = chunk.to_samples()
        if not samples:
            return 0.0
        # RMS level
        squared = [s * s for s in samples]
        return math.sqrt(sum(squared) / len(squared))

    def convert_sample_rate(
        self,
        data: bytes,
        input_rate: int,
        output_rate: int,
    ) -> bytes:
        """Convert sample rate of audio data."""
        chunk = AudioChunk(
            data=data,
            sample_rate=input_rate,
            format=self._format,
        )
        samples = chunk.to_samples()
        resampled = AudioResampler.resample(samples, input_rate, output_rate)
        output_chunk = AudioChunk.from_samples(
            resampled,
            sample_rate=output_rate,
            format=self._format,
        )
        return output_chunk.data

    def convert_format(
        self,
        data: bytes,
        input_format: AudioFormat,
        output_format: AudioFormat,
    ) -> bytes:
        """Convert audio format."""
        return AudioResampler.convert_format(data, input_format, output_format)

    def to_wav(self, data: bytes) -> bytes:
        """Convert raw audio to WAV format."""
        output = BytesIO()

        with wave.open(output, "wb") as wav:
            wav.setnchannels(self._channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self._sample_rate)
            wav.writeframes(data)

        return output.getvalue()

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "processed_chunks": self._processed_chunks,
            "speech_chunks": self._speech_chunks,
            "silence_chunks": self._silence_chunks,
            "total_duration_ms": self._total_duration_ms,
            "speech_ratio": (
                self._speech_chunks / self._processed_chunks
                if self._processed_chunks > 0 else 0
            ),
            "buffer_size": self._buffer.size,
            "buffer_overflows": self._buffer.overflow_count,
        }

    def reset(self) -> None:
        """Reset processor state."""
        self._buffer.clear()
        if self._vad:
            self._vad.reset()
        self._processed_chunks = 0
        self._speech_chunks = 0
        self._silence_chunks = 0
        self._total_duration_ms = 0
