"""Audio streaming handler for WebSocket."""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import base64
import logging
import struct

logger = logging.getLogger(__name__)


class AudioCodec(str, Enum):
    """Supported audio codecs."""
    PCM = "pcm"
    MULAW = "mulaw"
    ALAW = "alaw"
    OPUS = "opus"
    MP3 = "mp3"
    WAV = "wav"


class AudioFormat(str, Enum):
    """Audio format types."""
    RAW = "raw"
    BASE64 = "base64"


@dataclass
class AudioConfig:
    """Audio configuration."""
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16
    codec: AudioCodec = AudioCodec.PCM
    format: AudioFormat = AudioFormat.RAW

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bits_per_sample": self.bits_per_sample,
            "codec": self.codec.value,
            "format": self.format.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioConfig":
        """Create from dictionary."""
        return cls(
            sample_rate=data.get("sample_rate", 16000),
            channels=data.get("channels", 1),
            bits_per_sample=data.get("bits_per_sample", 16),
            codec=AudioCodec(data.get("codec", "pcm")),
            format=AudioFormat(data.get("format", "raw")),
        )


@dataclass
class AudioChunk:
    """Audio chunk data."""
    data: bytes
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sequence: int = 0
    duration_ms: float = 0.0
    is_speech: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Get size of audio data in bytes."""
        return len(self.data)

    def to_base64(self) -> str:
        """Convert audio data to base64."""
        return base64.b64encode(self.data).decode("utf-8")

    @classmethod
    def from_base64(cls, b64_data: str, **kwargs) -> "AudioChunk":
        """Create from base64 data."""
        data = base64.b64decode(b64_data)
        return cls(data=data, **kwargs)


class AudioBuffer:
    """
    Buffer for accumulating audio chunks.

    Handles buffering, resampling, and chunk management.
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        max_size: int = 1024 * 1024 * 10,  # 10MB
        chunk_size_ms: int = 20,
    ):
        self.config = config or AudioConfig()
        self.max_size = max_size
        self.chunk_size_ms = chunk_size_ms

        self._buffer = bytearray()
        self._sequence = 0
        self._total_duration_ms = 0.0
        self._lock = asyncio.Lock()

    @property
    def bytes_per_ms(self) -> float:
        """Calculate bytes per millisecond."""
        return (
            self.config.sample_rate *
            self.config.channels *
            (self.config.bits_per_sample // 8)
        ) / 1000

    @property
    def chunk_size_bytes(self) -> int:
        """Calculate chunk size in bytes."""
        return int(self.bytes_per_ms * self.chunk_size_ms)

    async def write(self, data: bytes) -> None:
        """Write audio data to buffer."""
        async with self._lock:
            if len(self._buffer) + len(data) > self.max_size:
                # Remove old data
                overflow = len(self._buffer) + len(data) - self.max_size
                self._buffer = self._buffer[overflow:]

            self._buffer.extend(data)
            self._total_duration_ms += len(data) / self.bytes_per_ms

    async def read(self, size: Optional[int] = None) -> bytes:
        """Read audio data from buffer."""
        async with self._lock:
            if size is None:
                data = bytes(self._buffer)
                self._buffer.clear()
            else:
                data = bytes(self._buffer[:size])
                self._buffer = self._buffer[size:]
            return data

    async def read_chunk(self) -> Optional[AudioChunk]:
        """Read a chunk of audio data."""
        async with self._lock:
            if len(self._buffer) < self.chunk_size_bytes:
                return None

            data = bytes(self._buffer[:self.chunk_size_bytes])
            self._buffer = self._buffer[self.chunk_size_bytes:]
            self._sequence += 1

            return AudioChunk(
                data=data,
                sequence=self._sequence,
                duration_ms=self.chunk_size_ms,
            )

    async def read_all_chunks(self) -> List[AudioChunk]:
        """Read all available chunks."""
        chunks = []
        while True:
            chunk = await self.read_chunk()
            if chunk is None:
                break
            chunks.append(chunk)
        return chunks

    async def clear(self) -> None:
        """Clear the buffer."""
        async with self._lock:
            self._buffer.clear()
            self._sequence = 0
            self._total_duration_ms = 0.0

    @property
    def size(self) -> int:
        """Get current buffer size in bytes."""
        return len(self._buffer)

    @property
    def duration_ms(self) -> float:
        """Get duration of buffered audio in milliseconds."""
        return len(self._buffer) / self.bytes_per_ms

    @property
    def total_duration_ms(self) -> float:
        """Get total duration of audio written."""
        return self._total_duration_ms


class AudioStreamHandler:
    """
    Handler for audio streaming.

    Manages bidirectional audio streaming with buffering and events.

    Usage:
        handler = AudioStreamHandler()

        @handler.on_audio_received
        async def handle_audio(chunk: AudioChunk):
            await process_audio(chunk)

        await handler.receive_audio(websocket_data)
        await handler.send_audio(audio_bytes)
    """

    def __init__(
        self,
        input_config: Optional[AudioConfig] = None,
        output_config: Optional[AudioConfig] = None,
        buffer_size_ms: int = 200,
    ):
        self.input_config = input_config or AudioConfig()
        self.output_config = output_config or AudioConfig()

        self._input_buffer = AudioBuffer(
            config=self.input_config,
            chunk_size_ms=buffer_size_ms,
        )
        self._output_buffer = AudioBuffer(
            config=self.output_config,
            chunk_size_ms=buffer_size_ms,
        )

        self._on_audio_received: List[Callable[[AudioChunk], Awaitable[None]]] = []
        self._on_audio_sent: List[Callable[[AudioChunk], Awaitable[None]]] = []
        self._on_speech_start: List[Callable[[], Awaitable[None]]] = []
        self._on_speech_end: List[Callable[[], Awaitable[None]]] = []

        self._is_streaming = False
        self._is_speaking = False
        self._total_received = 0
        self._total_sent = 0

    def on_audio_received(self, handler: Callable[[AudioChunk], Awaitable[None]]) -> None:
        """Register handler for received audio."""
        self._on_audio_received.append(handler)

    def on_audio_sent(self, handler: Callable[[AudioChunk], Awaitable[None]]) -> None:
        """Register handler for sent audio."""
        self._on_audio_sent.append(handler)

    def on_speech_start(self, handler: Callable[[], Awaitable[None]]) -> None:
        """Register handler for speech start."""
        self._on_speech_start.append(handler)

    def on_speech_end(self, handler: Callable[[], Awaitable[None]]) -> None:
        """Register handler for speech end."""
        self._on_speech_end.append(handler)

    async def start_stream(self) -> None:
        """Start audio streaming."""
        self._is_streaming = True
        logger.info("Audio stream started")

    async def stop_stream(self) -> None:
        """Stop audio streaming."""
        self._is_streaming = False
        await self._input_buffer.clear()
        await self._output_buffer.clear()
        logger.info("Audio stream stopped")

    async def receive_audio(
        self,
        data: bytes,
        is_base64: bool = False,
    ) -> List[AudioChunk]:
        """
        Receive audio data.

        Args:
            data: Audio data (raw bytes or base64)
            is_base64: Whether data is base64 encoded

        Returns:
            List of audio chunks
        """
        if not self._is_streaming:
            return []

        if is_base64:
            data = base64.b64decode(data)

        self._total_received += len(data)
        await self._input_buffer.write(data)

        # Get chunks
        chunks = await self._input_buffer.read_all_chunks()

        # Notify handlers
        for chunk in chunks:
            for handler in self._on_audio_received:
                try:
                    await handler(chunk)
                except Exception as e:
                    logger.error(f"Audio handler error: {e}")

        return chunks

    async def send_audio(
        self,
        data: bytes,
        as_base64: bool = False,
    ) -> bytes:
        """
        Send audio data.

        Args:
            data: Audio data
            as_base64: Whether to return as base64

        Returns:
            Processed audio data
        """
        if not self._is_streaming:
            return b""

        self._total_sent += len(data)

        # Create chunk
        chunk = AudioChunk(data=data, duration_ms=len(data) / self._output_buffer.bytes_per_ms)

        # Notify handlers
        for handler in self._on_audio_sent:
            try:
                await handler(chunk)
            except Exception as e:
                logger.error(f"Audio send handler error: {e}")

        if as_base64:
            return base64.b64encode(data)
        return data

    async def notify_speech_start(self) -> None:
        """Notify speech started."""
        if self._is_speaking:
            return

        self._is_speaking = True
        for handler in self._on_speech_start:
            try:
                await handler()
            except Exception as e:
                logger.error(f"Speech start handler error: {e}")

    async def notify_speech_end(self) -> None:
        """Notify speech ended."""
        if not self._is_speaking:
            return

        self._is_speaking = False
        for handler in self._on_speech_end:
            try:
                await handler()
            except Exception as e:
                logger.error(f"Speech end handler error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            "is_streaming": self._is_streaming,
            "is_speaking": self._is_speaking,
            "total_received_bytes": self._total_received,
            "total_sent_bytes": self._total_sent,
            "input_buffer_size": self._input_buffer.size,
            "output_buffer_size": self._output_buffer.size,
            "input_config": self.input_config.to_dict(),
            "output_config": self.output_config.to_dict(),
        }


class AudioConverter:
    """
    Audio format converter.

    Handles conversion between different audio formats.
    """

    @staticmethod
    def pcm_to_mulaw(pcm_data: bytes) -> bytes:
        """Convert PCM to mu-law."""
        MULAW_BIAS = 0x84
        MULAW_CLIP = 32635

        output = bytearray()
        for i in range(0, len(pcm_data), 2):
            sample = struct.unpack("<h", pcm_data[i:i+2])[0]

            sign = (sample >> 8) & 0x80
            if sign:
                sample = -sample
            if sample > MULAW_CLIP:
                sample = MULAW_CLIP

            sample = sample + MULAW_BIAS
            exponent = AudioConverter._get_mulaw_exponent(sample)
            mantissa = (sample >> (exponent + 3)) & 0x0F
            mulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
            output.append(mulaw_byte)

        return bytes(output)

    @staticmethod
    def mulaw_to_pcm(mulaw_data: bytes) -> bytes:
        """Convert mu-law to PCM."""
        output = bytearray()
        for byte in mulaw_data:
            byte = ~byte & 0xFF
            sign = (byte & 0x80)
            exponent = (byte >> 4) & 0x07
            mantissa = byte & 0x0F
            sample = AudioConverter._decode_mulaw_sample(sign, exponent, mantissa)
            output.extend(struct.pack("<h", sample))
        return bytes(output)

    @staticmethod
    def _get_mulaw_exponent(sample: int) -> int:
        """Get mu-law exponent for sample."""
        exp_table = [0, 132, 396, 924, 1980, 4092, 8316, 16764]
        for i in range(7, -1, -1):
            if sample >= exp_table[i]:
                return i
        return 0

    @staticmethod
    def _decode_mulaw_sample(sign: int, exponent: int, mantissa: int) -> int:
        """Decode mu-law sample."""
        sample = ((mantissa << 3) + 0x84) << exponent
        sample -= 0x84
        if sign:
            return -sample
        return sample

    @staticmethod
    def resample(
        data: bytes,
        from_rate: int,
        to_rate: int,
        channels: int = 1,
        bits: int = 16,
    ) -> bytes:
        """Simple linear resampling."""
        if from_rate == to_rate:
            return data

        bytes_per_sample = (bits // 8) * channels
        samples_in = len(data) // bytes_per_sample
        samples_out = int(samples_in * to_rate / from_rate)

        output = bytearray()
        ratio = from_rate / to_rate

        for i in range(samples_out):
            src_idx = i * ratio
            idx_low = int(src_idx)
            idx_high = min(idx_low + 1, samples_in - 1)
            frac = src_idx - idx_low

            for ch in range(channels):
                offset_low = (idx_low * channels + ch) * (bits // 8)
                offset_high = (idx_high * channels + ch) * (bits // 8)

                if bits == 16:
                    sample_low = struct.unpack("<h", data[offset_low:offset_low+2])[0]
                    sample_high = struct.unpack("<h", data[offset_high:offset_high+2])[0]
                    sample = int(sample_low * (1 - frac) + sample_high * frac)
                    output.extend(struct.pack("<h", sample))
                else:
                    sample_low = data[offset_low]
                    sample_high = data[offset_high]
                    sample = int(sample_low * (1 - frac) + sample_high * frac)
                    output.append(sample)

        return bytes(output)


class VoiceActivityDetector:
    """
    Simple Voice Activity Detection (VAD).

    Detects speech in audio based on energy levels.
    """

    def __init__(
        self,
        threshold: float = 0.02,
        min_speech_ms: int = 200,
        min_silence_ms: int = 300,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.sample_rate = sample_rate

        self._is_speech = False
        self._speech_start_time: Optional[datetime] = None
        self._silence_start_time: Optional[datetime] = None
        self._energy_history: List[float] = []

    def process(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Process audio and detect voice activity.

        Returns dict with:
            - is_speech: Current speech state
            - energy: Current energy level
            - speech_started: True if speech just started
            - speech_ended: True if speech just ended
        """
        energy = self._calculate_energy(audio_data)
        self._energy_history.append(energy)
        if len(self._energy_history) > 50:
            self._energy_history = self._energy_history[-50:]

        # Adaptive threshold
        avg_energy = sum(self._energy_history) / len(self._energy_history)
        adaptive_threshold = max(self.threshold, avg_energy * 1.5)

        now = datetime.utcnow()
        speech_started = False
        speech_ended = False

        if energy > adaptive_threshold:
            # Potential speech
            if not self._is_speech:
                if self._speech_start_time is None:
                    self._speech_start_time = now
                elif (now - self._speech_start_time).total_seconds() * 1000 >= self.min_speech_ms:
                    self._is_speech = True
                    speech_started = True
            self._silence_start_time = None
        else:
            # Potential silence
            if self._is_speech:
                if self._silence_start_time is None:
                    self._silence_start_time = now
                elif (now - self._silence_start_time).total_seconds() * 1000 >= self.min_silence_ms:
                    self._is_speech = False
                    speech_ended = True
            self._speech_start_time = None

        return {
            "is_speech": self._is_speech,
            "energy": energy,
            "speech_started": speech_started,
            "speech_ended": speech_ended,
        }

    def _calculate_energy(self, audio_data: bytes) -> float:
        """Calculate RMS energy of audio."""
        if len(audio_data) < 2:
            return 0.0

        samples = []
        for i in range(0, len(audio_data) - 1, 2):
            sample = struct.unpack("<h", audio_data[i:i+2])[0]
            samples.append(sample / 32768.0)  # Normalize to -1.0 to 1.0

        if not samples:
            return 0.0

        # RMS energy
        sum_sq = sum(s * s for s in samples)
        return (sum_sq / len(samples)) ** 0.5

    def reset(self) -> None:
        """Reset VAD state."""
        self._is_speech = False
        self._speech_start_time = None
        self._silence_start_time = None
        self._energy_history.clear()
