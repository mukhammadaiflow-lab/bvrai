"""
Audio processing utilities for the voice pipeline.

Provides:
- Audio format definitions and conversions
- Audio buffering with thread-safe operations
- Resampling between different sample rates
- Encoding/decoding for various audio codecs
- Audio analysis utilities (RMS, peak detection)
"""

import asyncio
import audioop
import base64
import io
import logging
import struct
import time
import wave
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Deque, Dict, List, Optional, Tuple, Union
import threading

logger = logging.getLogger(__name__)


class AudioCodec(str, Enum):
    """Supported audio codecs."""
    PCM = "pcm"           # Raw PCM (Linear16)
    MULAW = "mulaw"       # G.711 mu-law
    ALAW = "alaw"         # G.711 A-law
    OPUS = "opus"         # Opus codec
    MP3 = "mp3"           # MP3
    AAC = "aac"           # AAC
    FLAC = "flac"         # FLAC (lossless)
    WAV = "wav"           # WAV container with PCM


class AudioChannels(int, Enum):
    """Audio channel configurations."""
    MONO = 1
    STEREO = 2


@dataclass
class AudioFormat:
    """
    Defines the format of an audio stream.

    Attributes:
        sample_rate: Samples per second (Hz)
        channels: Number of audio channels
        sample_width: Bytes per sample (1, 2, or 4)
        codec: Audio codec/encoding
    """
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit = 2 bytes
    codec: AudioCodec = AudioCodec.PCM

    @property
    def bytes_per_second(self) -> int:
        """Calculate bytes per second for this format."""
        return self.sample_rate * self.channels * self.sample_width

    @property
    def bytes_per_ms(self) -> float:
        """Calculate bytes per millisecond."""
        return self.bytes_per_second / 1000.0

    @property
    def frame_size(self) -> int:
        """Size of a single frame (all channels, one sample)."""
        return self.channels * self.sample_width

    def duration_ms(self, num_bytes: int) -> float:
        """Calculate duration in milliseconds for given byte count."""
        if self.bytes_per_second == 0:
            return 0.0
        return (num_bytes / self.bytes_per_second) * 1000.0

    def bytes_for_duration_ms(self, duration_ms: float) -> int:
        """Calculate bytes needed for given duration in milliseconds."""
        return int((duration_ms / 1000.0) * self.bytes_per_second)

    def is_compatible(self, other: "AudioFormat") -> bool:
        """Check if two formats are compatible (same core properties)."""
        return (
            self.sample_rate == other.sample_rate and
            self.channels == other.channels and
            self.sample_width == other.sample_width
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "codec": self.codec.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioFormat":
        """Create from dictionary."""
        return cls(
            sample_rate=data.get("sample_rate", 16000),
            channels=data.get("channels", 1),
            sample_width=data.get("sample_width", 2),
            codec=AudioCodec(data.get("codec", "pcm")),
        )

    # Common format presets
    @classmethod
    def telephony(cls) -> "AudioFormat":
        """Standard telephony format (8kHz, mono, 16-bit PCM)."""
        return cls(sample_rate=8000, channels=1, sample_width=2, codec=AudioCodec.PCM)

    @classmethod
    def telephony_mulaw(cls) -> "AudioFormat":
        """Telephony mu-law format (8kHz, mono, 8-bit mu-law)."""
        return cls(sample_rate=8000, channels=1, sample_width=1, codec=AudioCodec.MULAW)

    @classmethod
    def wideband(cls) -> "AudioFormat":
        """Wideband format (16kHz, mono, 16-bit PCM)."""
        return cls(sample_rate=16000, channels=1, sample_width=2, codec=AudioCodec.PCM)

    @classmethod
    def fullband(cls) -> "AudioFormat":
        """Fullband format (48kHz, mono, 16-bit PCM)."""
        return cls(sample_rate=48000, channels=1, sample_width=2, codec=AudioCodec.PCM)

    @classmethod
    def cd_quality(cls) -> "AudioFormat":
        """CD quality (44.1kHz, stereo, 16-bit PCM)."""
        return cls(sample_rate=44100, channels=2, sample_width=2, codec=AudioCodec.PCM)


@dataclass
class AudioChunk:
    """
    A chunk of audio data with metadata.

    Attributes:
        data: Raw audio bytes
        format: Audio format specification
        timestamp_ms: Timestamp in milliseconds (from start of stream)
        is_speech: Whether VAD detected speech in this chunk
        sequence: Sequence number for ordering
        metadata: Additional metadata
    """
    data: bytes
    format: AudioFormat
    timestamp_ms: float = 0.0
    is_speech: bool = False
    sequence: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration of this chunk in milliseconds."""
        return self.format.duration_ms(len(self.data))

    @property
    def num_samples(self) -> int:
        """Number of samples in this chunk."""
        return len(self.data) // self.format.frame_size

    @property
    def rms(self) -> float:
        """Calculate RMS (Root Mean Square) amplitude."""
        if not self.data or self.format.codec != AudioCodec.PCM:
            return 0.0
        try:
            return audioop.rms(self.data, self.format.sample_width)
        except Exception:
            return 0.0

    @property
    def peak(self) -> int:
        """Get peak amplitude value."""
        if not self.data or self.format.codec != AudioCodec.PCM:
            return 0
        try:
            return audioop.max(self.data, self.format.sample_width)
        except Exception:
            return 0

    @property
    def db(self) -> float:
        """Calculate decibel level (dBFS)."""
        rms = self.rms
        if rms <= 0:
            return -100.0

        # Max value for sample width
        max_val = (2 ** (self.format.sample_width * 8 - 1)) - 1
        import math
        return 20 * math.log10(rms / max_val + 1e-10) if rms > 0 else -100.0

    def to_base64(self) -> str:
        """Encode audio data as base64."""
        return base64.b64encode(self.data).decode("utf-8")

    @classmethod
    def from_base64(
        cls,
        data: str,
        format: AudioFormat,
        timestamp_ms: float = 0.0,
    ) -> "AudioChunk":
        """Create chunk from base64 encoded data."""
        return cls(
            data=base64.b64decode(data),
            format=format,
            timestamp_ms=timestamp_ms,
        )

    def split(self, chunk_size_ms: float) -> List["AudioChunk"]:
        """Split this chunk into smaller chunks of specified duration."""
        chunk_bytes = self.format.bytes_for_duration_ms(chunk_size_ms)
        chunks = []

        offset = 0
        seq = 0
        while offset < len(self.data):
            end = min(offset + chunk_bytes, len(self.data))
            chunk_data = self.data[offset:end]

            chunks.append(AudioChunk(
                data=chunk_data,
                format=self.format,
                timestamp_ms=self.timestamp_ms + self.format.duration_ms(offset),
                is_speech=self.is_speech,
                sequence=seq,
                metadata=self.metadata.copy(),
            ))

            offset = end
            seq += 1

        return chunks


class AudioBuffer:
    """
    Thread-safe audio buffer for accumulating audio chunks.

    Features:
    - Circular buffer with configurable max duration
    - Thread-safe append and consume operations
    - Automatic overflow handling
    - Statistics tracking
    """

    def __init__(
        self,
        format: AudioFormat,
        max_duration_ms: float = 30000,  # 30 seconds max
        chunk_duration_ms: float = 20,    # Default chunk size
    ):
        self.format = format
        self.max_duration_ms = max_duration_ms
        self.chunk_duration_ms = chunk_duration_ms

        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._timestamp_offset = 0.0

        # Statistics
        self._total_bytes_written = 0
        self._total_bytes_read = 0
        self._overflow_bytes = 0
        self._chunk_count = 0

    @property
    def duration_ms(self) -> float:
        """Current buffer duration in milliseconds."""
        with self._lock:
            return self.format.duration_ms(len(self._buffer))

    @property
    def size_bytes(self) -> int:
        """Current buffer size in bytes."""
        with self._lock:
            return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0

    def append(self, chunk: AudioChunk) -> None:
        """
        Append audio chunk to buffer.

        Handles format conversion if necessary and manages overflow.
        """
        if not chunk.data:
            return

        with self._lock:
            self._buffer.extend(chunk.data)
            self._total_bytes_written += len(chunk.data)
            self._chunk_count += 1

            # Handle overflow
            max_bytes = self.format.bytes_for_duration_ms(self.max_duration_ms)
            if len(self._buffer) > max_bytes:
                overflow = len(self._buffer) - max_bytes
                self._buffer = self._buffer[overflow:]
                self._overflow_bytes += overflow
                self._timestamp_offset += self.format.duration_ms(overflow)

    def append_bytes(self, data: bytes) -> None:
        """Append raw bytes to buffer."""
        if not data:
            return

        with self._lock:
            self._buffer.extend(data)
            self._total_bytes_written += len(data)

            # Handle overflow
            max_bytes = self.format.bytes_for_duration_ms(self.max_duration_ms)
            if len(self._buffer) > max_bytes:
                overflow = len(self._buffer) - max_bytes
                self._buffer = self._buffer[overflow:]
                self._overflow_bytes += overflow
                self._timestamp_offset += self.format.duration_ms(overflow)

    def consume(self, duration_ms: Optional[float] = None) -> Optional[AudioChunk]:
        """
        Consume and return audio from the buffer.

        Args:
            duration_ms: Duration to consume. If None, consumes chunk_duration_ms.

        Returns:
            AudioChunk or None if buffer is empty
        """
        duration_ms = duration_ms or self.chunk_duration_ms

        with self._lock:
            if not self._buffer:
                return None

            bytes_to_read = self.format.bytes_for_duration_ms(duration_ms)
            bytes_to_read = min(bytes_to_read, len(self._buffer))

            # Align to frame boundary
            bytes_to_read = (bytes_to_read // self.format.frame_size) * self.format.frame_size

            if bytes_to_read == 0:
                return None

            data = bytes(self._buffer[:bytes_to_read])
            self._buffer = self._buffer[bytes_to_read:]
            self._total_bytes_read += bytes_to_read

            timestamp = self._timestamp_offset
            self._timestamp_offset += self.format.duration_ms(bytes_to_read)

            return AudioChunk(
                data=data,
                format=self.format,
                timestamp_ms=timestamp,
            )

    def consume_all(self) -> Optional[AudioChunk]:
        """Consume all audio from the buffer."""
        with self._lock:
            if not self._buffer:
                return None

            data = bytes(self._buffer)
            timestamp = self._timestamp_offset

            self._buffer.clear()
            self._total_bytes_read += len(data)
            self._timestamp_offset += self.format.duration_ms(len(data))

            return AudioChunk(
                data=data,
                format=self.format,
                timestamp_ms=timestamp,
            )

    def peek(self, duration_ms: Optional[float] = None) -> Optional[AudioChunk]:
        """
        Peek at audio without consuming it.

        Args:
            duration_ms: Duration to peek. If None, peeks chunk_duration_ms.

        Returns:
            AudioChunk or None if buffer is empty
        """
        duration_ms = duration_ms or self.chunk_duration_ms

        with self._lock:
            if not self._buffer:
                return None

            bytes_to_read = self.format.bytes_for_duration_ms(duration_ms)
            bytes_to_read = min(bytes_to_read, len(self._buffer))
            bytes_to_read = (bytes_to_read // self.format.frame_size) * self.format.frame_size

            if bytes_to_read == 0:
                return None

            return AudioChunk(
                data=bytes(self._buffer[:bytes_to_read]),
                format=self.format,
                timestamp_ms=self._timestamp_offset,
            )

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._timestamp_offset = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                "size_bytes": len(self._buffer),
                "duration_ms": self.format.duration_ms(len(self._buffer)),
                "total_bytes_written": self._total_bytes_written,
                "total_bytes_read": self._total_bytes_read,
                "overflow_bytes": self._overflow_bytes,
                "chunk_count": self._chunk_count,
            }


class AudioResampler:
    """
    Audio resampling utility for converting between sample rates.

    Uses linear interpolation for simplicity. For production use,
    consider using a library like librosa or scipy for higher quality.
    """

    def __init__(
        self,
        input_rate: int,
        output_rate: int,
        channels: int = 1,
        sample_width: int = 2,
    ):
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.channels = channels
        self.sample_width = sample_width

        self.ratio = output_rate / input_rate
        self._state = None

    def resample(self, audio_data: bytes) -> bytes:
        """
        Resample audio data to target sample rate.

        Args:
            audio_data: Input audio bytes

        Returns:
            Resampled audio bytes
        """
        if self.input_rate == self.output_rate:
            return audio_data

        try:
            # Use audioop for basic resampling
            resampled, self._state = audioop.ratecv(
                audio_data,
                self.sample_width,
                self.channels,
                self.input_rate,
                self.output_rate,
                self._state,
            )
            return resampled
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            return audio_data

    def reset(self) -> None:
        """Reset resampler state."""
        self._state = None


class AudioEncoder(ABC):
    """Abstract base class for audio encoders."""

    @abstractmethod
    def encode(self, pcm_data: bytes, format: AudioFormat) -> bytes:
        """Encode PCM audio to target format."""
        pass

    @abstractmethod
    def get_output_format(self, input_format: AudioFormat) -> AudioFormat:
        """Get the output format after encoding."""
        pass


class AudioDecoder(ABC):
    """Abstract base class for audio decoders."""

    @abstractmethod
    def decode(self, encoded_data: bytes, format: AudioFormat) -> bytes:
        """Decode audio to PCM format."""
        pass

    @abstractmethod
    def get_output_format(self, input_format: AudioFormat) -> AudioFormat:
        """Get the output format after decoding."""
        pass


class MuLawEncoder(AudioEncoder):
    """G.711 mu-law encoder."""

    def encode(self, pcm_data: bytes, format: AudioFormat) -> bytes:
        """Encode 16-bit PCM to 8-bit mu-law."""
        if format.sample_width != 2:
            raise ValueError("mu-law encoding requires 16-bit PCM input")

        return audioop.lin2ulaw(pcm_data, format.sample_width)

    def get_output_format(self, input_format: AudioFormat) -> AudioFormat:
        return AudioFormat(
            sample_rate=input_format.sample_rate,
            channels=input_format.channels,
            sample_width=1,
            codec=AudioCodec.MULAW,
        )


class MuLawDecoder(AudioDecoder):
    """G.711 mu-law decoder."""

    def decode(self, encoded_data: bytes, format: AudioFormat) -> bytes:
        """Decode 8-bit mu-law to 16-bit PCM."""
        return audioop.ulaw2lin(encoded_data, 2)

    def get_output_format(self, input_format: AudioFormat) -> AudioFormat:
        return AudioFormat(
            sample_rate=input_format.sample_rate,
            channels=input_format.channels,
            sample_width=2,
            codec=AudioCodec.PCM,
        )


class ALawEncoder(AudioEncoder):
    """G.711 A-law encoder."""

    def encode(self, pcm_data: bytes, format: AudioFormat) -> bytes:
        """Encode 16-bit PCM to 8-bit A-law."""
        if format.sample_width != 2:
            raise ValueError("A-law encoding requires 16-bit PCM input")

        return audioop.lin2alaw(pcm_data, format.sample_width)

    def get_output_format(self, input_format: AudioFormat) -> AudioFormat:
        return AudioFormat(
            sample_rate=input_format.sample_rate,
            channels=input_format.channels,
            sample_width=1,
            codec=AudioCodec.ALAW,
        )


class ALawDecoder(AudioDecoder):
    """G.711 A-law decoder."""

    def decode(self, encoded_data: bytes, format: AudioFormat) -> bytes:
        """Decode 8-bit A-law to 16-bit PCM."""
        return audioop.alaw2lin(encoded_data, 2)

    def get_output_format(self, input_format: AudioFormat) -> AudioFormat:
        return AudioFormat(
            sample_rate=input_format.sample_rate,
            channels=input_format.channels,
            sample_width=2,
            codec=AudioCodec.PCM,
        )


class WAVEncoder(AudioEncoder):
    """WAV file encoder."""

    def encode(self, pcm_data: bytes, format: AudioFormat) -> bytes:
        """Encode PCM to WAV format."""
        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(format.channels)
            wav_file.setsampwidth(format.sample_width)
            wav_file.setframerate(format.sample_rate)
            wav_file.writeframes(pcm_data)

        return buffer.getvalue()

    def get_output_format(self, input_format: AudioFormat) -> AudioFormat:
        return AudioFormat(
            sample_rate=input_format.sample_rate,
            channels=input_format.channels,
            sample_width=input_format.sample_width,
            codec=AudioCodec.WAV,
        )


class WAVDecoder(AudioDecoder):
    """WAV file decoder."""

    def decode(self, encoded_data: bytes, format: AudioFormat) -> bytes:
        """Decode WAV to PCM format."""
        buffer = io.BytesIO(encoded_data)

        with wave.open(buffer, "rb") as wav_file:
            return wav_file.readframes(wav_file.getnframes())

    def get_output_format(self, input_format: AudioFormat) -> AudioFormat:
        return AudioFormat(
            sample_rate=input_format.sample_rate,
            channels=input_format.channels,
            sample_width=input_format.sample_width,
            codec=AudioCodec.PCM,
        )


class AudioConverter:
    """
    High-level audio format converter.

    Handles conversion between different audio formats including:
    - Sample rate conversion
    - Channel conversion (mono/stereo)
    - Codec conversion
    """

    def __init__(self):
        self._encoders: Dict[AudioCodec, AudioEncoder] = {
            AudioCodec.MULAW: MuLawEncoder(),
            AudioCodec.ALAW: ALawEncoder(),
            AudioCodec.WAV: WAVEncoder(),
        }

        self._decoders: Dict[AudioCodec, AudioDecoder] = {
            AudioCodec.MULAW: MuLawDecoder(),
            AudioCodec.ALAW: ALawDecoder(),
            AudioCodec.WAV: WAVDecoder(),
        }

        self._resamplers: Dict[Tuple[int, int], AudioResampler] = {}

    def convert(
        self,
        audio_data: bytes,
        input_format: AudioFormat,
        output_format: AudioFormat,
    ) -> bytes:
        """
        Convert audio between formats.

        Args:
            audio_data: Input audio bytes
            input_format: Current audio format
            output_format: Desired output format

        Returns:
            Converted audio bytes
        """
        data = audio_data
        current_format = input_format

        # Step 1: Decode to PCM if necessary
        if current_format.codec != AudioCodec.PCM:
            decoder = self._decoders.get(current_format.codec)
            if decoder:
                data = decoder.decode(data, current_format)
                current_format = decoder.get_output_format(current_format)

        # Step 2: Convert channels if necessary
        if current_format.channels != output_format.channels:
            data = self._convert_channels(
                data,
                current_format.channels,
                output_format.channels,
                current_format.sample_width,
            )
            current_format = AudioFormat(
                sample_rate=current_format.sample_rate,
                channels=output_format.channels,
                sample_width=current_format.sample_width,
                codec=AudioCodec.PCM,
            )

        # Step 3: Resample if necessary
        if current_format.sample_rate != output_format.sample_rate:
            resampler_key = (current_format.sample_rate, output_format.sample_rate)
            if resampler_key not in self._resamplers:
                self._resamplers[resampler_key] = AudioResampler(
                    input_rate=current_format.sample_rate,
                    output_rate=output_format.sample_rate,
                    channels=current_format.channels,
                    sample_width=current_format.sample_width,
                )

            data = self._resamplers[resampler_key].resample(data)
            current_format = AudioFormat(
                sample_rate=output_format.sample_rate,
                channels=current_format.channels,
                sample_width=current_format.sample_width,
                codec=AudioCodec.PCM,
            )

        # Step 4: Encode to target codec if necessary
        if output_format.codec != AudioCodec.PCM:
            encoder = self._encoders.get(output_format.codec)
            if encoder:
                data = encoder.encode(data, current_format)

        return data

    def _convert_channels(
        self,
        data: bytes,
        input_channels: int,
        output_channels: int,
        sample_width: int,
    ) -> bytes:
        """Convert between mono and stereo."""
        if input_channels == output_channels:
            return data

        if input_channels == 1 and output_channels == 2:
            # Mono to stereo: duplicate channel
            return audioop.tostereo(data, sample_width, 1, 1)

        elif input_channels == 2 and output_channels == 1:
            # Stereo to mono: average channels
            return audioop.tomono(data, sample_width, 0.5, 0.5)

        else:
            raise ValueError(f"Unsupported channel conversion: {input_channels} -> {output_channels}")


def calculate_rms(audio_data: bytes, sample_width: int = 2) -> float:
    """Calculate RMS amplitude of audio data."""
    if not audio_data:
        return 0.0
    try:
        return audioop.rms(audio_data, sample_width)
    except Exception:
        return 0.0


def calculate_db(audio_data: bytes, sample_width: int = 2) -> float:
    """Calculate decibel level (dBFS) of audio data."""
    rms = calculate_rms(audio_data, sample_width)
    if rms <= 0:
        return -100.0

    max_val = (2 ** (sample_width * 8 - 1)) - 1
    try:
        import math
        return 20 * math.log10(rms / max_val + 1e-10)
    except Exception:
        return -100.0


def detect_silence(
    audio_data: bytes,
    sample_width: int = 2,
    threshold_db: float = -40.0,
) -> bool:
    """Check if audio chunk is silence based on dB threshold."""
    db = calculate_db(audio_data, sample_width)
    return db < threshold_db


def normalize_audio(
    audio_data: bytes,
    sample_width: int = 2,
    target_db: float = -3.0,
) -> bytes:
    """Normalize audio to target dB level."""
    if not audio_data:
        return audio_data

    current_db = calculate_db(audio_data, sample_width)
    if current_db <= -100.0:
        return audio_data

    # Calculate required gain
    import math
    gain_db = target_db - current_db
    gain_factor = 10 ** (gain_db / 20)

    # Apply gain (clamp to prevent overflow)
    try:
        return audioop.mul(audio_data, sample_width, min(gain_factor, 10.0))
    except Exception:
        return audio_data


async def stream_audio_chunks(
    audio_data: bytes,
    format: AudioFormat,
    chunk_duration_ms: float = 20,
    real_time: bool = False,
) -> AsyncIterator[AudioChunk]:
    """
    Stream audio data as chunks.

    Args:
        audio_data: Complete audio data
        format: Audio format
        chunk_duration_ms: Duration of each chunk
        real_time: If True, adds delays to simulate real-time streaming

    Yields:
        AudioChunk objects
    """
    chunk_size = format.bytes_for_duration_ms(chunk_duration_ms)
    chunk_size = (chunk_size // format.frame_size) * format.frame_size

    offset = 0
    sequence = 0
    start_time = time.time() if real_time else None

    while offset < len(audio_data):
        end = min(offset + chunk_size, len(audio_data))
        chunk_data = audio_data[offset:end]

        timestamp = format.duration_ms(offset)

        if real_time:
            # Wait until the chunk should be delivered
            expected_time = start_time + (timestamp / 1000.0)
            current_time = time.time()
            if current_time < expected_time:
                await asyncio.sleep(expected_time - current_time)

        yield AudioChunk(
            data=chunk_data,
            format=format,
            timestamp_ms=timestamp,
            sequence=sequence,
        )

        offset = end
        sequence += 1
