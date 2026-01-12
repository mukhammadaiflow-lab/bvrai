"""
Audio Streaming

Real-time audio streaming:
- Bi-directional streaming
- Stream multiplexing
- Jitter buffer
- Packet handling
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Generic, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import struct
import time
import heapq

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StreamDirection(str, Enum):
    """Stream direction."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BIDIRECTIONAL = "bidirectional"


class StreamProtocol(str, Enum):
    """Streaming protocols."""
    RTP = "rtp"
    RTCP = "rtcp"
    WEBSOCKET = "websocket"
    HTTP_CHUNKED = "http_chunked"
    GRPC = "grpc"


@dataclass
class StreamConfig:
    """Stream configuration."""
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16
    frame_duration_ms: int = 20

    # Buffer settings
    jitter_buffer_ms: int = 50
    max_buffer_ms: int = 500

    # Timing
    packet_loss_concealment: bool = True
    silence_detection: bool = True

    # Quality
    target_latency_ms: int = 150
    max_latency_ms: int = 500

    @property
    def frame_size(self) -> int:
        """Get frame size in bytes."""
        samples = int(self.sample_rate * self.frame_duration_ms / 1000)
        return samples * self.channels * (self.bits_per_sample // 8)


@dataclass
class AudioChunk:
    """Audio chunk for streaming."""
    data: bytes
    sequence: int
    timestamp_ms: float
    duration_ms: float
    is_speech: bool = True

    # RTP-like fields
    payload_type: int = 0
    ssrc: int = 0
    marker: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "AudioChunk") -> bool:
        """Compare by sequence for heap operations."""
        return self.sequence < other.sequence


@dataclass
class RTPPacket:
    """RTP packet representation."""
    version: int = 2
    padding: bool = False
    extension: bool = False
    csrc_count: int = 0
    marker: bool = False
    payload_type: int = 0
    sequence: int = 0
    timestamp: int = 0
    ssrc: int = 0
    csrc: List[int] = field(default_factory=list)
    payload: bytes = b""

    def to_bytes(self) -> bytes:
        """Serialize RTP packet."""
        # First byte: V=2, P, X, CC
        first_byte = (self.version << 6) | (int(self.padding) << 5) | \
                     (int(self.extension) << 4) | self.csrc_count

        # Second byte: M, PT
        second_byte = (int(self.marker) << 7) | self.payload_type

        # Pack header
        header = struct.pack(
            "!BBHII",
            first_byte,
            second_byte,
            self.sequence,
            self.timestamp,
            self.ssrc,
        )

        # Add CSRC list
        for csrc in self.csrc:
            header += struct.pack("!I", csrc)

        return header + self.payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "RTPPacket":
        """Parse RTP packet from bytes."""
        if len(data) < 12:
            raise ValueError("RTP packet too short")

        first_byte = data[0]
        second_byte = data[1]

        version = (first_byte >> 6) & 0x03
        padding = bool((first_byte >> 5) & 0x01)
        extension = bool((first_byte >> 4) & 0x01)
        csrc_count = first_byte & 0x0F

        marker = bool((second_byte >> 7) & 0x01)
        payload_type = second_byte & 0x7F

        sequence, timestamp, ssrc = struct.unpack("!HII", data[2:12])

        # Parse CSRC
        offset = 12
        csrc = []
        for _ in range(csrc_count):
            csrc.append(struct.unpack("!I", data[offset:offset+4])[0])
            offset += 4

        payload = data[offset:]

        return cls(
            version=version,
            padding=padding,
            extension=extension,
            csrc_count=csrc_count,
            marker=marker,
            payload_type=payload_type,
            sequence=sequence,
            timestamp=timestamp,
            ssrc=ssrc,
            csrc=csrc,
            payload=payload,
        )


class JitterBuffer:
    """
    Jitter buffer for audio streaming.

    Buffers packets to smooth out network jitter.
    """

    def __init__(
        self,
        target_delay_ms: int = 50,
        max_delay_ms: int = 200,
        frame_duration_ms: int = 20,
    ):
        self.target_delay_ms = target_delay_ms
        self.max_delay_ms = max_delay_ms
        self.frame_duration_ms = frame_duration_ms

        # Packet buffer (min-heap by sequence)
        self._buffer: List[AudioChunk] = []
        self._next_sequence = 0
        self._initialized = False

        # Statistics
        self._packets_received = 0
        self._packets_dropped = 0
        self._packets_late = 0
        self._packets_concealed = 0

        # Timing
        self._last_packet_time = 0.0
        self._jitter_estimate_ms = 0.0

    def push(self, chunk: AudioChunk) -> None:
        """Push packet into buffer."""
        self._packets_received += 1

        # Initialize on first packet
        if not self._initialized:
            self._next_sequence = chunk.sequence
            self._initialized = True
            self._last_packet_time = time.time() * 1000

        # Update jitter estimate
        current_time = time.time() * 1000
        if self._last_packet_time > 0:
            arrival_delta = current_time - self._last_packet_time
            expected_delta = chunk.duration_ms
            jitter = abs(arrival_delta - expected_delta)
            self._jitter_estimate_ms = self._jitter_estimate_ms * 0.9 + jitter * 0.1
        self._last_packet_time = current_time

        # Check if packet is too old
        if chunk.sequence < self._next_sequence:
            self._packets_late += 1
            return

        # Check buffer size
        buffer_duration = len(self._buffer) * self.frame_duration_ms
        if buffer_duration >= self.max_delay_ms:
            self._packets_dropped += 1
            return

        # Add to buffer
        heapq.heappush(self._buffer, chunk)

    def pop(self) -> Optional[AudioChunk]:
        """Pop next packet from buffer."""
        # Check if buffer has enough delay
        buffer_duration = len(self._buffer) * self.frame_duration_ms
        if buffer_duration < self.target_delay_ms:
            return None

        # Get next expected packet
        while self._buffer:
            chunk = self._buffer[0]

            if chunk.sequence == self._next_sequence:
                heapq.heappop(self._buffer)
                self._next_sequence += 1
                return chunk

            elif chunk.sequence < self._next_sequence:
                # Old packet, discard
                heapq.heappop(self._buffer)
                continue

            else:
                # Packet loss, need concealment
                self._packets_concealed += 1
                self._next_sequence += 1
                return self._conceal_packet()

        return None

    def _conceal_packet(self) -> AudioChunk:
        """Generate concealment packet for lost audio."""
        # Generate silence or interpolate
        frame_size = int(16000 * self.frame_duration_ms / 1000) * 2
        silence = b"\x00" * frame_size

        return AudioChunk(
            data=silence,
            sequence=self._next_sequence - 1,
            timestamp_ms=0,
            duration_ms=float(self.frame_duration_ms),
            is_speech=False,
        )

    def get_delay_ms(self) -> float:
        """Get current buffer delay in milliseconds."""
        return len(self._buffer) * self.frame_duration_ms

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "buffer_size": len(self._buffer),
            "buffer_delay_ms": self.get_delay_ms(),
            "packets_received": self._packets_received,
            "packets_dropped": self._packets_dropped,
            "packets_late": self._packets_late,
            "packets_concealed": self._packets_concealed,
            "jitter_estimate_ms": self._jitter_estimate_ms,
        }

    def reset(self) -> None:
        """Reset buffer state."""
        self._buffer.clear()
        self._next_sequence = 0
        self._initialized = False


class StreamProcessor(ABC):
    """Abstract stream processor."""

    @abstractmethod
    async def process(self, chunk: AudioChunk) -> Optional[AudioChunk]:
        """Process audio chunk."""
        pass


class PassthroughProcessor(StreamProcessor):
    """Passthrough processor."""

    async def process(self, chunk: AudioChunk) -> Optional[AudioChunk]:
        """Pass through unchanged."""
        return chunk


class VolumeProcessor(StreamProcessor):
    """Volume adjustment processor."""

    def __init__(self, volume: float = 1.0):
        self.volume = volume

    async def process(self, chunk: AudioChunk) -> Optional[AudioChunk]:
        """Adjust volume."""
        if self.volume == 1.0:
            return chunk

        samples = list(struct.unpack(f"<{len(chunk.data)//2}h", chunk.data))
        adjusted = [int(s * self.volume) for s in samples]
        adjusted = [max(-32768, min(32767, s)) for s in adjusted]
        data = struct.pack(f"<{len(adjusted)}h", *adjusted)

        return AudioChunk(
            data=data,
            sequence=chunk.sequence,
            timestamp_ms=chunk.timestamp_ms,
            duration_ms=chunk.duration_ms,
            is_speech=chunk.is_speech,
            metadata=chunk.metadata,
        )


class StreamingPipeline:
    """
    Audio streaming pipeline.

    Processes audio through a chain of processors.
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()

        # Processors
        self._processors: List[StreamProcessor] = []

        # Jitter buffer
        self._jitter_buffer = JitterBuffer(
            target_delay_ms=self.config.jitter_buffer_ms,
            max_delay_ms=self.config.max_buffer_ms,
            frame_duration_ms=self.config.frame_duration_ms,
        )

        # State
        self._sequence = 0
        self._running = False

        # Statistics
        self._chunks_in = 0
        self._chunks_out = 0

    def add_processor(self, processor: StreamProcessor) -> "StreamingPipeline":
        """Add processor to pipeline."""
        self._processors.append(processor)
        return self

    async def push(self, audio: bytes, is_speech: bool = True) -> None:
        """Push audio into pipeline."""
        chunk = AudioChunk(
            data=audio,
            sequence=self._sequence,
            timestamp_ms=time.time() * 1000,
            duration_ms=float(self.config.frame_duration_ms),
            is_speech=is_speech,
        )
        self._sequence += 1
        self._chunks_in += 1

        self._jitter_buffer.push(chunk)

    async def pull(self) -> Optional[AudioChunk]:
        """Pull processed audio from pipeline."""
        chunk = self._jitter_buffer.pop()

        if chunk is None:
            return None

        # Process through pipeline
        result = chunk
        for processor in self._processors:
            if result is None:
                break
            result = await processor.process(result)

        if result:
            self._chunks_out += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "chunks_in": self._chunks_in,
            "chunks_out": self._chunks_out,
            "jitter_buffer": self._jitter_buffer.get_stats(),
        }


class BiDirectionalStream:
    """
    Bi-directional audio stream.

    Handles both inbound and outbound audio.
    """

    def __init__(
        self,
        stream_id: str,
        config: Optional[StreamConfig] = None,
    ):
        self.stream_id = stream_id
        self.config = config or StreamConfig()

        # Pipelines for each direction
        self._inbound_pipeline = StreamingPipeline(config)
        self._outbound_pipeline = StreamingPipeline(config)

        # Callbacks
        self._on_inbound: List[Callable[[AudioChunk], Awaitable[None]]] = []
        self._on_outbound: List[Callable[[AudioChunk], Awaitable[None]]] = []

        # State
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Statistics
        self._created_at = datetime.utcnow()

    async def start(self) -> None:
        """Start stream processing."""
        if self._running:
            return

        self._running = True

        # Start processing loops
        self._tasks.append(asyncio.create_task(self._inbound_loop()))
        self._tasks.append(asyncio.create_task(self._outbound_loop()))

    async def stop(self) -> None:
        """Stop stream processing."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

    async def send(self, audio: bytes, is_speech: bool = True) -> None:
        """Send audio to outbound stream."""
        await self._outbound_pipeline.push(audio, is_speech)

    async def receive(self, audio: bytes, is_speech: bool = True) -> None:
        """Receive audio into inbound stream."""
        await self._inbound_pipeline.push(audio, is_speech)

    def on_inbound(self, callback: Callable[[AudioChunk], Awaitable[None]]) -> None:
        """Register inbound audio callback."""
        self._on_inbound.append(callback)

    def on_outbound(self, callback: Callable[[AudioChunk], Awaitable[None]]) -> None:
        """Register outbound audio callback."""
        self._on_outbound.append(callback)

    async def _inbound_loop(self) -> None:
        """Process inbound audio."""
        while self._running:
            try:
                chunk = await self._inbound_pipeline.pull()
                if chunk:
                    for callback in self._on_inbound:
                        await callback(chunk)
                else:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Inbound processing error: {e}")

    async def _outbound_loop(self) -> None:
        """Process outbound audio."""
        while self._running:
            try:
                chunk = await self._outbound_pipeline.pull()
                if chunk:
                    for callback in self._on_outbound:
                        await callback(chunk)
                else:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Outbound processing error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        return {
            "stream_id": self.stream_id,
            "running": self._running,
            "uptime_seconds": (datetime.utcnow() - self._created_at).total_seconds(),
            "inbound": self._inbound_pipeline.get_stats(),
            "outbound": self._outbound_pipeline.get_stats(),
        }


class StreamMultiplexer:
    """
    Stream multiplexer.

    Manages multiple audio streams and routes audio between them.
    """

    def __init__(self):
        # Streams by ID
        self._streams: Dict[str, BiDirectionalStream] = {}

        # Routing table: source -> [destinations]
        self._routes: Dict[str, List[str]] = {}

        self._lock = asyncio.Lock()

    async def create_stream(
        self,
        stream_id: str,
        config: Optional[StreamConfig] = None,
    ) -> BiDirectionalStream:
        """Create new stream."""
        async with self._lock:
            stream = BiDirectionalStream(stream_id, config)
            self._streams[stream_id] = stream

            # Set up routing callback
            async def route_audio(chunk: AudioChunk):
                await self._route_audio(stream_id, chunk)

            stream.on_outbound(route_audio)

            return stream

    async def get_stream(self, stream_id: str) -> Optional[BiDirectionalStream]:
        """Get stream by ID."""
        return self._streams.get(stream_id)

    async def remove_stream(self, stream_id: str) -> bool:
        """Remove stream."""
        async with self._lock:
            stream = self._streams.pop(stream_id, None)
            if stream:
                await stream.stop()
                self._routes.pop(stream_id, None)
                return True
            return False

    def add_route(self, from_stream: str, to_stream: str) -> None:
        """Add routing from one stream to another."""
        if from_stream not in self._routes:
            self._routes[from_stream] = []
        if to_stream not in self._routes[from_stream]:
            self._routes[from_stream].append(to_stream)

    def remove_route(self, from_stream: str, to_stream: str) -> None:
        """Remove routing."""
        if from_stream in self._routes:
            if to_stream in self._routes[from_stream]:
                self._routes[from_stream].remove(to_stream)

    async def _route_audio(self, from_stream: str, chunk: AudioChunk) -> None:
        """Route audio to destination streams."""
        destinations = self._routes.get(from_stream, [])

        for dest_id in destinations:
            dest_stream = self._streams.get(dest_id)
            if dest_stream:
                await dest_stream.receive(chunk.data, chunk.is_speech)

    async def broadcast(self, audio: bytes, exclude: Optional[List[str]] = None) -> None:
        """Broadcast audio to all streams."""
        exclude = exclude or []

        for stream_id, stream in self._streams.items():
            if stream_id not in exclude:
                await stream.receive(audio)

    async def start_all(self) -> None:
        """Start all streams."""
        for stream in self._streams.values():
            await stream.start()

    async def stop_all(self) -> None:
        """Stop all streams."""
        for stream in self._streams.values():
            await stream.stop()

    def get_stats(self) -> Dict[str, Any]:
        """Get multiplexer statistics."""
        return {
            "stream_count": len(self._streams),
            "route_count": sum(len(r) for r in self._routes.values()),
            "streams": {
                sid: stream.get_stats()
                for sid, stream in self._streams.items()
            },
        }


class WebSocketAudioStream:
    """
    WebSocket-based audio streaming.

    Handles audio streaming over WebSocket connections.
    """

    def __init__(
        self,
        stream_id: str,
        config: Optional[StreamConfig] = None,
    ):
        self.stream_id = stream_id
        self.config = config or StreamConfig()

        # WebSocket connection (set externally)
        self._websocket: Optional[Any] = None

        # Pipeline
        self._pipeline = StreamingPipeline(config)

        # State
        self._running = False
        self._connected = False

        # Callbacks
        self._on_audio: List[Callable[[bytes], Awaitable[None]]] = []
        self._on_message: List[Callable[[Dict], Awaitable[None]]] = []

    def set_websocket(self, websocket: Any) -> None:
        """Set WebSocket connection."""
        self._websocket = websocket
        self._connected = True

    async def start(self) -> None:
        """Start streaming."""
        self._running = True

    async def stop(self) -> None:
        """Stop streaming."""
        self._running = False
        self._connected = False

    async def send_audio(self, audio: bytes) -> bool:
        """Send audio over WebSocket."""
        if not self._websocket or not self._connected:
            return False

        try:
            # Send as binary frame
            await self._websocket.send_bytes(audio)
            return True
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            self._connected = False
            return False

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send JSON message over WebSocket."""
        if not self._websocket or not self._connected:
            return False

        try:
            import json
            await self._websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            self._connected = False
            return False

    async def receive(self) -> Optional[bytes]:
        """Receive audio from WebSocket."""
        if not self._websocket or not self._connected:
            return None

        try:
            message = await self._websocket.receive()

            if message.get("type") == "websocket.receive":
                if "bytes" in message:
                    audio = message["bytes"]
                    for callback in self._on_audio:
                        await callback(audio)
                    return audio

                elif "text" in message:
                    import json
                    data = json.loads(message["text"])
                    for callback in self._on_message:
                        await callback(data)

            elif message.get("type") == "websocket.disconnect":
                self._connected = False

        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
            self._connected = False

        return None

    def on_audio(self, callback: Callable[[bytes], Awaitable[None]]) -> None:
        """Register audio callback."""
        self._on_audio.append(callback)

    def on_message(self, callback: Callable[[Dict], Awaitable[None]]) -> None:
        """Register message callback."""
        self._on_message.append(callback)

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected
