"""
Audio Streaming Module

This module provides audio streaming capabilities including
buffering, format conversion, and stream management.
"""

import asyncio
import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from .base import Event, EventType


logger = logging.getLogger(__name__)


class AudioFormat(str, Enum):
    """Audio format types."""
    PCM_16 = "pcm_16"  # 16-bit PCM
    PCM_32 = "pcm_32"  # 32-bit PCM
    PCM_FLOAT = "pcm_float"  # 32-bit float
    MULAW = "mulaw"
    ALAW = "alaw"
    OPUS = "opus"
    MP3 = "mp3"


class StreamDirection(str, Enum):
    """Stream direction."""
    INBOUND = "inbound"  # From caller
    OUTBOUND = "outbound"  # To caller
    BIDIRECTIONAL = "bidirectional"


@dataclass
class AudioStreamConfig:
    """Configuration for audio streams."""

    # Audio format
    format: AudioFormat = AudioFormat.PCM_16
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16

    # Buffering
    buffer_size_ms: int = 100
    max_buffer_size_ms: int = 5000
    chunk_size_ms: int = 20

    # Processing
    enable_vad: bool = True
    enable_noise_reduction: bool = False
    enable_agc: bool = False  # Automatic gain control

    # Streaming
    direction: StreamDirection = StreamDirection.BIDIRECTIONAL

    @property
    def bytes_per_sample(self) -> int:
        """Get bytes per sample."""
        return self.bits_per_sample // 8

    @property
    def bytes_per_second(self) -> int:
        """Get bytes per second."""
        return self.sample_rate * self.channels * self.bytes_per_sample

    @property
    def chunk_size_bytes(self) -> int:
        """Get chunk size in bytes."""
        return int(self.bytes_per_second * self.chunk_size_ms / 1000)

    @property
    def buffer_size_bytes(self) -> int:
        """Get buffer size in bytes."""
        return int(self.bytes_per_second * self.buffer_size_ms / 1000)


@dataclass
class AudioBuffer:
    """Audio buffer for managing audio data."""

    config: AudioStreamConfig = field(default_factory=AudioStreamConfig)

    # Buffer
    _data: bytearray = field(default_factory=bytearray)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # State
    total_bytes_written: int = 0
    total_bytes_read: int = 0

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_write_at: Optional[datetime] = None
    last_read_at: Optional[datetime] = None

    async def write(self, data: bytes) -> int:
        """
        Write data to buffer.

        Args:
            data: Audio data

        Returns:
            Bytes written
        """
        async with self._lock:
            # Check max buffer size
            max_bytes = int(
                self.config.bytes_per_second *
                self.config.max_buffer_size_ms / 1000
            )

            available_space = max_bytes - len(self._data)
            if available_space <= 0:
                # Buffer full, drop oldest data
                drop_amount = len(data) - available_space
                self._data = self._data[drop_amount:]
                available_space = len(data)

            # Write data
            bytes_to_write = min(len(data), available_space)
            self._data.extend(data[:bytes_to_write])

            self.total_bytes_written += bytes_to_write
            self.last_write_at = datetime.utcnow()

            return bytes_to_write

    async def read(self, size: int = -1) -> bytes:
        """
        Read data from buffer.

        Args:
            size: Bytes to read (-1 for all)

        Returns:
            Audio data
        """
        async with self._lock:
            if size < 0:
                data = bytes(self._data)
                self._data.clear()
            else:
                data = bytes(self._data[:size])
                self._data = self._data[size:]

            self.total_bytes_read += len(data)
            self.last_read_at = datetime.utcnow()

            return data

    async def read_chunk(self) -> bytes:
        """Read a single chunk."""
        return await self.read(self.config.chunk_size_bytes)

    async def peek(self, size: int = -1) -> bytes:
        """Peek at buffer without consuming."""
        async with self._lock:
            if size < 0:
                return bytes(self._data)
            return bytes(self._data[:size])

    def available(self) -> int:
        """Get available bytes in buffer."""
        return len(self._data)

    def duration_ms(self) -> float:
        """Get buffered duration in milliseconds."""
        return (len(self._data) / self.config.bytes_per_second) * 1000

    async def clear(self) -> None:
        """Clear the buffer."""
        async with self._lock:
            self._data.clear()


@dataclass
class AudioStream:
    """Audio stream for a session."""

    id: str = ""
    session_id: str = ""
    direction: StreamDirection = StreamDirection.BIDIRECTIONAL

    # Configuration
    config: AudioStreamConfig = field(default_factory=AudioStreamConfig)

    # Buffers
    inbound_buffer: AudioBuffer = field(default_factory=AudioBuffer)
    outbound_buffer: AudioBuffer = field(default_factory=AudioBuffer)

    # State
    is_active: bool = False
    is_paused: bool = False

    # Statistics
    total_inbound_bytes: int = 0
    total_outbound_bytes: int = 0
    total_inbound_duration_ms: float = 0.0
    total_outbound_duration_ms: float = 0.0

    # Speech detection
    is_speech_detected: bool = False
    speech_start_at: Optional[datetime] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    # Callbacks
    on_audio_received: Optional[Callable[[bytes], None]] = None
    on_speech_start: Optional[Callable[[], None]] = None
    on_speech_end: Optional[Callable[[], None]] = None

    def __post_init__(self):
        self.inbound_buffer.config = self.config
        self.outbound_buffer.config = self.config

    async def start(self) -> None:
        """Start the stream."""
        self.is_active = True
        self.started_at = datetime.utcnow()

    async def stop(self) -> None:
        """Stop the stream."""
        self.is_active = False
        self.stopped_at = datetime.utcnow()

    async def pause(self) -> None:
        """Pause the stream."""
        self.is_paused = True

    async def resume(self) -> None:
        """Resume the stream."""
        self.is_paused = False

    async def write_inbound(self, data: bytes) -> None:
        """
        Write inbound audio (from caller).

        Args:
            data: Audio data
        """
        if not self.is_active or self.is_paused:
            return

        await self.inbound_buffer.write(data)
        self.total_inbound_bytes += len(data)
        self.total_inbound_duration_ms += (
            len(data) / self.config.bytes_per_second * 1000
        )

        if self.on_audio_received:
            await asyncio.to_thread(self.on_audio_received, data)

    async def write_outbound(self, data: bytes) -> None:
        """
        Write outbound audio (to caller).

        Args:
            data: Audio data
        """
        if not self.is_active or self.is_paused:
            return

        await self.outbound_buffer.write(data)
        self.total_outbound_bytes += len(data)
        self.total_outbound_duration_ms += (
            len(data) / self.config.bytes_per_second * 1000
        )

    async def read_inbound(self, size: int = -1) -> bytes:
        """Read inbound audio."""
        return await self.inbound_buffer.read(size)

    async def read_outbound(self, size: int = -1) -> bytes:
        """Read outbound audio."""
        return await self.outbound_buffer.read(size)

    async def read_outbound_chunk(self) -> bytes:
        """Read a chunk of outbound audio."""
        return await self.outbound_buffer.read_chunk()

    async def iter_outbound_chunks(self) -> AsyncIterator[bytes]:
        """Iterate over outbound audio chunks."""
        while self.is_active:
            if self.outbound_buffer.available() >= self.config.chunk_size_bytes:
                yield await self.outbound_buffer.read_chunk()
            else:
                await asyncio.sleep(self.config.chunk_size_ms / 1000 / 2)

    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "is_active": self.is_active,
            "total_inbound_bytes": self.total_inbound_bytes,
            "total_outbound_bytes": self.total_outbound_bytes,
            "total_inbound_duration_ms": self.total_inbound_duration_ms,
            "total_outbound_duration_ms": self.total_outbound_duration_ms,
            "inbound_buffer_ms": self.inbound_buffer.duration_ms(),
            "outbound_buffer_ms": self.outbound_buffer.duration_ms(),
        }


class AudioStreamManager:
    """
    Manages multiple audio streams.

    Handles stream lifecycle, format conversion, and audio processing.
    """

    def __init__(self, default_config: Optional[AudioStreamConfig] = None):
        """
        Initialize audio stream manager.

        Args:
            default_config: Default stream configuration
        """
        self.default_config = default_config or AudioStreamConfig()
        self._streams: Dict[str, AudioStream] = {}
        self._lock = asyncio.Lock()

    async def create_stream(
        self,
        session_id: str,
        stream_id: Optional[str] = None,
        config: Optional[AudioStreamConfig] = None,
    ) -> AudioStream:
        """
        Create a new audio stream.

        Args:
            session_id: Session ID
            stream_id: Optional stream ID
            config: Stream configuration

        Returns:
            Audio stream
        """
        import uuid

        stream = AudioStream(
            id=stream_id or str(uuid.uuid4()),
            session_id=session_id,
            config=config or self.default_config,
        )

        async with self._lock:
            self._streams[stream.id] = stream

        logger.info(f"Audio stream created: {stream.id}")

        return stream

    async def get_stream(self, stream_id: str) -> Optional[AudioStream]:
        """Get a stream by ID."""
        return self._streams.get(stream_id)

    async def get_streams_by_session(self, session_id: str) -> List[AudioStream]:
        """Get all streams for a session."""
        return [
            s for s in self._streams.values()
            if s.session_id == session_id
        ]

    async def start_stream(self, stream_id: str) -> bool:
        """Start a stream."""
        stream = self._streams.get(stream_id)
        if stream:
            await stream.start()
            return True
        return False

    async def stop_stream(self, stream_id: str) -> bool:
        """Stop a stream."""
        stream = self._streams.get(stream_id)
        if stream:
            await stream.stop()
            return True
        return False

    async def remove_stream(self, stream_id: str) -> None:
        """Remove a stream."""
        async with self._lock:
            stream = self._streams.pop(stream_id, None)
            if stream and stream.is_active:
                await stream.stop()

        logger.info(f"Audio stream removed: {stream_id}")

    async def write_audio(
        self,
        stream_id: str,
        data: bytes,
        direction: StreamDirection = StreamDirection.INBOUND,
    ) -> None:
        """
        Write audio to a stream.

        Args:
            stream_id: Stream ID
            data: Audio data
            direction: Stream direction
        """
        stream = self._streams.get(stream_id)
        if not stream:
            return

        if direction == StreamDirection.INBOUND:
            await stream.write_inbound(data)
        else:
            await stream.write_outbound(data)

    async def read_audio(
        self,
        stream_id: str,
        direction: StreamDirection = StreamDirection.OUTBOUND,
        size: int = -1,
    ) -> bytes:
        """
        Read audio from a stream.

        Args:
            stream_id: Stream ID
            direction: Stream direction
            size: Bytes to read

        Returns:
            Audio data
        """
        stream = self._streams.get(stream_id)
        if not stream:
            return b""

        if direction == StreamDirection.INBOUND:
            return await stream.read_inbound(size)
        else:
            return await stream.read_outbound(size)

    def get_active_streams(self) -> List[AudioStream]:
        """Get all active streams."""
        return [s for s in self._streams.values() if s.is_active]

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        active = sum(1 for s in self._streams.values() if s.is_active)
        total_inbound = sum(s.total_inbound_bytes for s in self._streams.values())
        total_outbound = sum(s.total_outbound_bytes for s in self._streams.values())

        return {
            "total_streams": len(self._streams),
            "active_streams": active,
            "total_inbound_bytes": total_inbound,
            "total_outbound_bytes": total_outbound,
        }


# Audio conversion utilities

def convert_mulaw_to_pcm16(mulaw_data: bytes) -> bytes:
    """Convert mu-law to 16-bit PCM."""
    MULAW_DECODE = [
        -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
        -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
        -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
        -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
        -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
        -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
        -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
        -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
        -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
        -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
        -876, -844, -812, -780, -748, -716, -684, -652,
        -620, -588, -556, -524, -492, -460, -428, -396,
        -372, -356, -340, -324, -308, -292, -276, -260,
        -244, -228, -212, -196, -180, -164, -148, -132,
        -120, -112, -104, -96, -88, -80, -72, -64,
        -56, -48, -40, -32, -24, -16, -8, 0,
        32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
        23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
        15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
        11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
        7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
        5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
        3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
        2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
        1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
        1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
        876, 844, 812, 780, 748, 716, 684, 652,
        620, 588, 556, 524, 492, 460, 428, 396,
        372, 356, 340, 324, 308, 292, 276, 260,
        244, 228, 212, 196, 180, 164, 148, 132,
        120, 112, 104, 96, 88, 80, 72, 64,
        56, 48, 40, 32, 24, 16, 8, 0,
    ]

    pcm_data = bytearray()
    for byte in mulaw_data:
        sample = MULAW_DECODE[byte ^ 0xFF]
        pcm_data.extend(struct.pack('<h', sample))

    return bytes(pcm_data)


def convert_pcm16_to_mulaw(pcm_data: bytes) -> bytes:
    """Convert 16-bit PCM to mu-law."""
    def encode_sample(sample: int) -> int:
        BIAS = 0x84
        CLIP = 32635

        sign = (sample >> 8) & 0x80
        if sign:
            sample = -sample
        sample = min(sample, CLIP)
        sample += BIAS

        exponent = 7
        mask = 0x4000
        while exponent > 0 and not (sample & mask):
            exponent -= 1
            mask >>= 1

        mantissa = (sample >> (exponent + 3)) & 0x0F
        return ~(sign | (exponent << 4) | mantissa) & 0xFF

    mulaw_data = bytearray()
    for i in range(0, len(pcm_data), 2):
        sample = struct.unpack('<h', pcm_data[i:i+2])[0]
        mulaw_data.append(encode_sample(sample))

    return bytes(mulaw_data)


def resample_audio(
    data: bytes,
    from_rate: int,
    to_rate: int,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """
    Resample audio data.

    Simple linear interpolation resampling.
    """
    if from_rate == to_rate:
        return data

    samples_in = len(data) // sample_width
    samples_out = int(samples_in * to_rate / from_rate)

    fmt = '<h' if sample_width == 2 else '<b'

    # Unpack samples
    in_samples = struct.unpack(f'{samples_in}{fmt[1]}', data)

    # Resample
    out_samples = []
    for i in range(samples_out):
        src_idx = i * from_rate / to_rate
        idx0 = int(src_idx)
        idx1 = min(idx0 + 1, samples_in - 1)
        frac = src_idx - idx0

        sample = int(in_samples[idx0] * (1 - frac) + in_samples[idx1] * frac)
        out_samples.append(sample)

    # Pack samples
    return struct.pack(f'{len(out_samples)}{fmt[1]}', *out_samples)


__all__ = [
    "AudioFormat",
    "StreamDirection",
    "AudioStreamConfig",
    "AudioBuffer",
    "AudioStream",
    "AudioStreamManager",
    "convert_mulaw_to_pcm16",
    "convert_pcm16_to_mulaw",
    "resample_audio",
]
