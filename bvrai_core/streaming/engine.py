"""
Streaming Engine Core
=====================

Core streaming engine for managing real-time audio and data streams
with session management, buffering, and metrics collection.

Author: Platform Engineering Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


class StreamState(str, Enum):
    """State of a stream session."""

    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    PAUSED = "paused"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


class StreamType(str, Enum):
    """Types of streams."""

    AUDIO_IN = "audio_in"
    AUDIO_OUT = "audio_out"
    VIDEO_IN = "video_in"
    VIDEO_OUT = "video_out"
    DATA = "data"
    CONTROL = "control"


@dataclass
class StreamConfig:
    """Configuration for a stream session."""

    # Basic settings
    session_id: str = field(default_factory=lambda: f"stream_{uuid4().hex[:12]}")
    stream_type: StreamType = StreamType.AUDIO_IN
    organization_id: str = ""
    call_id: Optional[str] = None

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16
    codec: str = "pcm"  # pcm, opus, g711

    # Buffer settings
    buffer_duration_ms: int = 100
    max_buffer_size: int = 32000  # bytes
    chunk_size_ms: int = 20

    # Quality settings
    enable_noise_reduction: bool = True
    enable_echo_cancellation: bool = True
    enable_auto_gain: bool = True

    # Timeout settings
    idle_timeout_seconds: int = 300
    connect_timeout_seconds: int = 30

    # Recording
    enable_recording: bool = False
    recording_format: str = "wav"

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamMetrics:
    """Metrics for a stream session."""

    session_id: str = ""
    stream_type: StreamType = StreamType.AUDIO_IN

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0

    # Volume
    bytes_received: int = 0
    bytes_sent: int = 0
    packets_received: int = 0
    packets_sent: int = 0

    # Quality
    packets_lost: int = 0
    packets_out_of_order: int = 0
    jitter_ms: float = 0.0
    latency_ms: float = 0.0

    # Buffer
    buffer_overflows: int = 0
    buffer_underflows: int = 0

    # Audio specific
    audio_level_avg: float = 0.0
    voice_activity_percent: float = 0.0
    silence_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "stream_type": self.stream_type.value,
            "start_time": self.start_time.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "duration_seconds": self.duration_seconds,
            "bytes_received": self.bytes_received,
            "bytes_sent": self.bytes_sent,
            "packets_received": self.packets_received,
            "packets_sent": self.packets_sent,
            "packets_lost": self.packets_lost,
            "jitter_ms": self.jitter_ms,
            "latency_ms": self.latency_ms,
            "buffer_overflows": self.buffer_overflows,
            "buffer_underflows": self.buffer_underflows,
            "audio_level_avg": self.audio_level_avg,
            "voice_activity_percent": self.voice_activity_percent,
        }


class StreamSession:
    """
    Represents a streaming session.

    Handles buffering, state management, and data flow for a single stream.
    """

    def __init__(self, config: StreamConfig):
        self.config = config
        self.id = config.session_id
        self.state = StreamState.INITIALIZING
        self.metrics = StreamMetrics(
            session_id=self.id,
            stream_type=config.stream_type,
        )

        # Buffers
        self._input_buffer: Deque[bytes] = deque(maxlen=1000)
        self._output_buffer: Deque[bytes] = deque(maxlen=1000)
        self._chunk_buffer = bytearray()

        # Event handling
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._data_handlers: List[Callable[[bytes], None]] = []

        # State
        self._lock = asyncio.Lock()
        self._connected_at: Optional[datetime] = None
        self._last_data_time: Optional[datetime] = None
        self._sequence_number = 0
        self._last_sequence = -1

        # Recording
        self._recording_buffer: List[bytes] = []

        self._logger = structlog.get_logger(f"stream_session.{self.id[:8]}")

    async def connect(self) -> bool:
        """Establish the stream connection."""
        self.state = StreamState.CONNECTING
        self._connected_at = datetime.utcnow()

        try:
            # Simulate connection setup
            await asyncio.sleep(0.01)
            self.state = StreamState.CONNECTED
            self._emit_event("connected")
            self._logger.info("Stream connected")
            return True
        except Exception as e:
            self.state = StreamState.ERROR
            self._emit_event("error", str(e))
            return False

    async def start(self) -> None:
        """Start streaming."""
        if self.state != StreamState.CONNECTED:
            raise RuntimeError(f"Cannot start stream in state: {self.state}")

        self.state = StreamState.STREAMING
        self.metrics.start_time = datetime.utcnow()
        self._emit_event("started")
        self._logger.info("Streaming started")

    async def pause(self) -> None:
        """Pause the stream."""
        if self.state == StreamState.STREAMING:
            self.state = StreamState.PAUSED
            self._emit_event("paused")

    async def resume(self) -> None:
        """Resume the stream."""
        if self.state == StreamState.PAUSED:
            self.state = StreamState.STREAMING
            self._emit_event("resumed")

    async def stop(self) -> None:
        """Stop and close the stream."""
        self.state = StreamState.CLOSING

        # Flush buffers
        await self._flush_buffers()

        self.state = StreamState.CLOSED
        self.metrics.duration_seconds = (
            datetime.utcnow() - self.metrics.start_time
        ).total_seconds()

        self._emit_event("closed")
        self._logger.info(
            "Stream closed",
            duration=self.metrics.duration_seconds,
            bytes_received=self.metrics.bytes_received,
        )

    async def write(self, data: bytes) -> None:
        """Write data to the stream (outbound)."""
        if self.state not in (StreamState.STREAMING, StreamState.CONNECTED):
            return

        async with self._lock:
            self._output_buffer.append(data)
            self.metrics.bytes_sent += len(data)
            self.metrics.packets_sent += 1
            self.metrics.last_activity = datetime.utcnow()

    async def read(self) -> Optional[bytes]:
        """Read data from the stream (inbound)."""
        if not self._input_buffer:
            return None

        async with self._lock:
            if self._input_buffer:
                return self._input_buffer.popleft()
        return None

    async def receive(self, data: bytes, sequence: int = -1) -> None:
        """Receive data into the stream (from external source)."""
        if self.state not in (StreamState.STREAMING, StreamState.CONNECTED):
            return

        async with self._lock:
            # Check sequence
            if sequence >= 0:
                if sequence <= self._last_sequence:
                    self.metrics.packets_out_of_order += 1
                    return
                if sequence > self._last_sequence + 1:
                    self.metrics.packets_lost += (sequence - self._last_sequence - 1)
                self._last_sequence = sequence

            self._input_buffer.append(data)
            self.metrics.bytes_received += len(data)
            self.metrics.packets_received += 1
            self.metrics.last_activity = datetime.utcnow()
            self._last_data_time = datetime.utcnow()

            # Recording
            if self.config.enable_recording:
                self._recording_buffer.append(data)

            # Notify handlers
            for handler in self._data_handlers:
                try:
                    handler(data)
                except Exception as e:
                    self._logger.error(f"Data handler error: {e}")

    async def read_chunk(self) -> Optional[bytes]:
        """Read a chunk of configured size."""
        chunk_bytes = int(
            self.config.sample_rate
            * self.config.channels
            * (self.config.bits_per_sample // 8)
            * (self.config.chunk_size_ms / 1000)
        )

        while len(self._chunk_buffer) < chunk_bytes:
            data = await self.read()
            if data is None:
                if len(self._chunk_buffer) > 0:
                    # Return partial chunk
                    chunk = bytes(self._chunk_buffer)
                    self._chunk_buffer.clear()
                    return chunk
                return None
            self._chunk_buffer.extend(data)

        chunk = bytes(self._chunk_buffer[:chunk_bytes])
        self._chunk_buffer = self._chunk_buffer[chunk_bytes:]
        return chunk

    async def iter_chunks(self, timeout: float = 1.0) -> AsyncIterator[bytes]:
        """Iterate over incoming chunks."""
        while self.state == StreamState.STREAMING:
            chunk = await self.read_chunk()
            if chunk:
                yield chunk
            else:
                await asyncio.sleep(0.01)

    async def _flush_buffers(self) -> None:
        """Flush remaining buffer data."""
        self._input_buffer.clear()
        self._output_buffer.clear()
        self._chunk_buffer.clear()

    def on_data(self, handler: Callable[[bytes], None]) -> None:
        """Register a data handler."""
        self._data_handlers.append(handler)

    def on_event(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        self._event_handlers[event].append(handler)

    def _emit_event(self, event: str, *args, **kwargs) -> None:
        """Emit an event to handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                handler(*args, **kwargs)
            except Exception as e:
                self._logger.error(f"Event handler error for {event}: {e}")

    def get_recording(self) -> bytes:
        """Get recorded audio data."""
        return b"".join(self._recording_buffer)

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state in (StreamState.STREAMING, StreamState.CONNECTED, StreamState.PAUSED)

    @property
    def is_idle(self) -> bool:
        """Check if session is idle."""
        if not self._last_data_time:
            return False
        idle_seconds = (datetime.utcnow() - self._last_data_time).total_seconds()
        return idle_seconds > self.config.idle_timeout_seconds


class StreamingEngine:
    """
    Central streaming engine.

    Manages multiple streaming sessions, handles resource allocation,
    and provides unified streaming capabilities.
    """

    def __init__(
        self,
        max_concurrent_sessions: int = 1000,
        cleanup_interval_seconds: int = 60,
    ):
        self._max_sessions = max_concurrent_sessions
        self._cleanup_interval = cleanup_interval_seconds
        self._sessions: Dict[str, StreamSession] = {}
        self._sessions_by_call: Dict[str, Set[str]] = defaultdict(set)
        self._sessions_by_org: Dict[str, Set[str]] = defaultdict(set)

        self._lock = asyncio.Lock()
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._logger = structlog.get_logger("streaming_engine")

        # Metrics
        self._total_sessions = 0
        self._total_bytes = 0
        self._start_time: Optional[datetime] = None

    async def start(self) -> None:
        """Start the streaming engine."""
        self._running = True
        self._start_time = datetime.utcnow()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._logger.info("Streaming engine started")

    async def stop(self) -> None:
        """Stop the streaming engine."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all sessions
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.close_session(session_id)

        self._logger.info("Streaming engine stopped")

    async def create_session(
        self,
        config: Optional[StreamConfig] = None,
        **kwargs,
    ) -> StreamSession:
        """
        Create a new streaming session.

        Args:
            config: Stream configuration
            **kwargs: Additional config parameters

        Returns:
            Created StreamSession
        """
        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError(f"Maximum sessions ({self._max_sessions}) reached")

        if config is None:
            config = StreamConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        session = StreamSession(config)

        async with self._lock:
            self._sessions[session.id] = session
            if config.call_id:
                self._sessions_by_call[config.call_id].add(session.id)
            if config.organization_id:
                self._sessions_by_org[config.organization_id].add(session.id)
            self._total_sessions += 1

        self._logger.info(
            f"Created session: {session.id}",
            stream_type=config.stream_type.value,
            call_id=config.call_id,
        )

        self._emit_event("session_created", session)
        return session

    async def get_session(self, session_id: str) -> Optional[StreamSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def close_session(self, session_id: str) -> bool:
        """Close and remove a session."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        await session.stop()

        async with self._lock:
            del self._sessions[session_id]
            if session.config.call_id:
                self._sessions_by_call[session.config.call_id].discard(session_id)
            if session.config.organization_id:
                self._sessions_by_org[session.config.organization_id].discard(session_id)

            self._total_bytes += session.metrics.bytes_received + session.metrics.bytes_sent

        self._emit_event("session_closed", session)
        return True

    async def get_sessions_for_call(self, call_id: str) -> List[StreamSession]:
        """Get all sessions for a call."""
        session_ids = self._sessions_by_call.get(call_id, set())
        return [self._sessions[sid] for sid in session_ids if sid in self._sessions]

    async def get_sessions_for_org(self, organization_id: str) -> List[StreamSession]:
        """Get all sessions for an organization."""
        session_ids = self._sessions_by_org.get(organization_id, set())
        return [self._sessions[sid] for sid in session_ids if sid in self._sessions]

    async def broadcast_to_call(self, call_id: str, data: bytes) -> int:
        """Broadcast data to all sessions in a call."""
        sessions = await self.get_sessions_for_call(call_id)
        sent = 0
        for session in sessions:
            if session.config.stream_type in (StreamType.AUDIO_OUT, StreamType.DATA):
                await session.write(data)
                sent += 1
        return sent

    def on_event(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        self._event_handlers[event].append(handler)

    def _emit_event(self, event: str, *args, **kwargs) -> None:
        """Emit an event."""
        for handler in self._event_handlers.get(event, []):
            try:
                handler(*args, **kwargs)
            except Exception as e:
                self._logger.error(f"Event handler error: {e}")

    async def _cleanup_loop(self) -> None:
        """Background loop to clean up idle sessions."""
        while self._running:
            await asyncio.sleep(self._cleanup_interval)

            idle_sessions = [
                sid for sid, session in self._sessions.items()
                if session.is_idle or session.state in (StreamState.CLOSED, StreamState.ERROR)
            ]

            for session_id in idle_sessions:
                self._logger.info(f"Cleaning up idle session: {session_id}")
                await self.close_session(session_id)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics."""
        active_sessions = sum(1 for s in self._sessions.values() if s.is_active)
        streaming_sessions = sum(
            1 for s in self._sessions.values()
            if s.state == StreamState.STREAMING
        )

        current_bytes = sum(
            s.metrics.bytes_received + s.metrics.bytes_sent
            for s in self._sessions.values()
        )

        uptime = (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0

        return {
            "uptime_seconds": uptime,
            "total_sessions_created": self._total_sessions,
            "current_sessions": len(self._sessions),
            "active_sessions": active_sessions,
            "streaming_sessions": streaming_sessions,
            "max_sessions": self._max_sessions,
            "total_bytes_processed": self._total_bytes + current_bytes,
            "sessions_by_type": {
                st.value: sum(
                    1 for s in self._sessions.values()
                    if s.config.stream_type == st
                )
                for st in StreamType
            },
        }

    async def get_session_metrics(self, session_id: str) -> Optional[StreamMetrics]:
        """Get metrics for a specific session."""
        session = self._sessions.get(session_id)
        if session:
            session.metrics.duration_seconds = (
                datetime.utcnow() - session.metrics.start_time
            ).total_seconds()
            return session.metrics
        return None


class StreamMultiplexer:
    """
    Multiplexes multiple streams for efficient processing.

    Useful for handling multiple audio streams from a conference call
    or combining/splitting streams.
    """

    def __init__(self):
        self._streams: Dict[str, StreamSession] = {}
        self._combined_output: Deque[bytes] = deque(maxlen=1000)
        self._lock = asyncio.Lock()

    async def add_stream(self, stream: StreamSession) -> None:
        """Add a stream to the multiplexer."""
        async with self._lock:
            self._streams[stream.id] = stream
            stream.on_data(lambda data: self._on_stream_data(stream.id, data))

    async def remove_stream(self, stream_id: str) -> None:
        """Remove a stream from the multiplexer."""
        async with self._lock:
            self._streams.pop(stream_id, None)

    def _on_stream_data(self, stream_id: str, data: bytes) -> None:
        """Handle data from a stream."""
        # In a real implementation, this would mix audio
        # For now, just pass through
        self._combined_output.append(data)

    async def read_combined(self) -> Optional[bytes]:
        """Read combined/mixed output."""
        if self._combined_output:
            return self._combined_output.popleft()
        return None

    @property
    def stream_count(self) -> int:
        """Get number of active streams."""
        return len(self._streams)
