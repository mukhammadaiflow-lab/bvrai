"""
Voice Engine Core

Central voice processing engine:
- Session management
- Audio stream handling
- Real-time processing
- Multi-protocol support
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import uuid
import time
import json

logger = logging.getLogger(__name__)


class StreamState(str, Enum):
    """Audio stream states."""
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ACTIVE = "active"
    PAUSED = "paused"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


class SessionType(str, Enum):
    """Voice session types."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    WEBRTC = "webrtc"
    SIP = "sip"
    PSTN = "pstn"
    INTERNAL = "internal"


class AudioFormat(str, Enum):
    """Audio formats."""
    PCM_16 = "pcm_16"
    PCM_32 = "pcm_32"
    MULAW = "mulaw"
    ALAW = "alaw"
    OPUS = "opus"
    MP3 = "mp3"
    AAC = "aac"
    WAV = "wav"


@dataclass
class VoiceEngineConfig:
    """Voice engine configuration."""
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16
    frame_duration_ms: int = 20

    # Buffer settings
    input_buffer_size: int = 10
    output_buffer_size: int = 10
    jitter_buffer_ms: int = 50

    # Processing
    enable_vad: bool = True
    enable_noise_reduction: bool = True
    enable_echo_cancellation: bool = True
    enable_auto_gain: bool = True

    # Timeouts
    connection_timeout_seconds: float = 30.0
    idle_timeout_seconds: float = 300.0
    max_session_duration_seconds: float = 3600.0

    # Limits
    max_concurrent_sessions: int = 1000
    max_streams_per_session: int = 10

    # Quality
    min_audio_quality: float = 0.7
    target_latency_ms: int = 150
    max_latency_ms: int = 500

    def get_frame_size(self) -> int:
        """Calculate frame size in samples."""
        return int(self.sample_rate * self.frame_duration_ms / 1000)

    def get_bytes_per_frame(self) -> int:
        """Calculate bytes per frame."""
        return self.get_frame_size() * self.channels * (self.bits_per_sample // 8)


@dataclass
class AudioStream:
    """Audio stream representation."""
    stream_id: str
    session_id: str
    direction: str  # "inbound" or "outbound"
    format: AudioFormat = AudioFormat.PCM_16
    sample_rate: int = 16000
    channels: int = 1
    state: StreamState = StreamState.INITIALIZING
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Statistics
    bytes_received: int = 0
    bytes_sent: int = 0
    packets_received: int = 0
    packets_sent: int = 0
    packets_lost: int = 0

    # Quality metrics
    jitter_ms: float = 0.0
    latency_ms: float = 0.0
    packet_loss_rate: float = 0.0

    # Buffers
    _input_buffer: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=100))
    _output_buffer: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=100))

    async def write(self, data: bytes) -> None:
        """Write audio data to stream."""
        try:
            await asyncio.wait_for(
                self._output_buffer.put(data),
                timeout=0.1,
            )
            self.bytes_sent += len(data)
            self.packets_sent += 1
        except asyncio.TimeoutError:
            logger.warning(f"Stream {self.stream_id} output buffer full")

    async def read(self, timeout: float = 0.1) -> Optional[bytes]:
        """Read audio data from stream."""
        try:
            data = await asyncio.wait_for(
                self._input_buffer.get(),
                timeout=timeout,
            )
            self.bytes_received += len(data)
            self.packets_received += 1
            return data
        except asyncio.TimeoutError:
            return None

    async def receive(self, data: bytes) -> None:
        """Receive audio data into stream."""
        try:
            await asyncio.wait_for(
                self._input_buffer.put(data),
                timeout=0.1,
            )
        except asyncio.TimeoutError:
            self.packets_lost += 1
            logger.warning(f"Stream {self.stream_id} input buffer full, packet lost")

    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        total_packets = self.packets_received + self.packets_lost
        self.packet_loss_rate = self.packets_lost / max(1, total_packets)

        return {
            "stream_id": self.stream_id,
            "state": self.state.value,
            "bytes_received": self.bytes_received,
            "bytes_sent": self.bytes_sent,
            "packets_received": self.packets_received,
            "packets_sent": self.packets_sent,
            "packets_lost": self.packets_lost,
            "packet_loss_rate": self.packet_loss_rate,
            "jitter_ms": self.jitter_ms,
            "latency_ms": self.latency_ms,
        }


@dataclass
class VoiceSession:
    """Voice session representation."""
    session_id: str
    tenant_id: str
    session_type: SessionType
    call_id: Optional[str] = None
    agent_id: Optional[str] = None

    # Endpoints
    caller_id: str = ""
    callee_id: str = ""
    from_number: str = ""
    to_number: str = ""

    # State
    state: StreamState = StreamState.INITIALIZING
    created_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Streams
    streams: Dict[str, AudioStream] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Quality
    quality_score: float = 1.0

    def add_stream(self, direction: str, **kwargs) -> AudioStream:
        """Add audio stream to session."""
        stream = AudioStream(
            stream_id=str(uuid.uuid4()),
            session_id=self.session_id,
            direction=direction,
            **kwargs,
        )
        self.streams[stream.stream_id] = stream
        return stream

    def get_inbound_stream(self) -> Optional[AudioStream]:
        """Get inbound audio stream."""
        for stream in self.streams.values():
            if stream.direction == "inbound":
                return stream
        return None

    def get_outbound_stream(self) -> Optional[AudioStream]:
        """Get outbound audio stream."""
        for stream in self.streams.values():
            if stream.direction == "outbound":
                return stream
        return None

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end = self.ended_at or datetime.utcnow()
        start = self.connected_at or self.created_at
        return (end - start).total_seconds()

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "session_type": self.session_type.value,
            "state": self.state.value,
            "duration_seconds": self.duration_seconds,
            "quality_score": self.quality_score,
            "streams": {
                sid: stream.get_stats()
                for sid, stream in self.streams.items()
            },
        }


class SessionEventHandler(ABC):
    """Abstract session event handler."""

    @abstractmethod
    async def on_session_started(self, session: VoiceSession) -> None:
        """Handle session start."""
        pass

    @abstractmethod
    async def on_session_connected(self, session: VoiceSession) -> None:
        """Handle session connected."""
        pass

    @abstractmethod
    async def on_audio_received(
        self,
        session: VoiceSession,
        stream: AudioStream,
        audio: bytes,
    ) -> None:
        """Handle received audio."""
        pass

    @abstractmethod
    async def on_session_ended(self, session: VoiceSession) -> None:
        """Handle session end."""
        pass


class DefaultSessionHandler(SessionEventHandler):
    """Default session event handler."""

    async def on_session_started(self, session: VoiceSession) -> None:
        """Log session start."""
        logger.info(f"Session started: {session.session_id}")

    async def on_session_connected(self, session: VoiceSession) -> None:
        """Log session connected."""
        logger.info(f"Session connected: {session.session_id}")

    async def on_audio_received(
        self,
        session: VoiceSession,
        stream: AudioStream,
        audio: bytes,
    ) -> None:
        """Process received audio."""
        pass

    async def on_session_ended(self, session: VoiceSession) -> None:
        """Log session end."""
        logger.info(
            f"Session ended: {session.session_id}, "
            f"duration: {session.duration_seconds:.2f}s"
        )


class SessionManager:
    """
    Manages voice sessions.

    Handles:
    - Session lifecycle
    - Resource allocation
    - Session lookup and routing
    """

    def __init__(
        self,
        config: Optional[VoiceEngineConfig] = None,
        event_handler: Optional[SessionEventHandler] = None,
    ):
        self.config = config or VoiceEngineConfig()
        self.event_handler = event_handler or DefaultSessionHandler()

        self._sessions: Dict[str, VoiceSession] = {}
        self._sessions_by_call: Dict[str, str] = {}
        self._sessions_by_tenant: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()

        # Metrics
        self._total_sessions = 0
        self._active_sessions = 0

    async def create_session(
        self,
        tenant_id: str,
        session_type: SessionType,
        call_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs,
    ) -> VoiceSession:
        """Create new voice session."""
        async with self._lock:
            # Check limits
            tenant_sessions = self._sessions_by_tenant.get(tenant_id, set())
            if len(tenant_sessions) >= self.config.max_concurrent_sessions:
                raise ValueError(f"Max sessions exceeded for tenant {tenant_id}")

            session = VoiceSession(
                session_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                session_type=session_type,
                call_id=call_id,
                agent_id=agent_id,
                **kwargs,
            )

            self._sessions[session.session_id] = session

            if call_id:
                self._sessions_by_call[call_id] = session.session_id

            if tenant_id not in self._sessions_by_tenant:
                self._sessions_by_tenant[tenant_id] = set()
            self._sessions_by_tenant[tenant_id].add(session.session_id)

            self._total_sessions += 1
            self._active_sessions += 1

        await self.event_handler.on_session_started(session)
        return session

    async def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    async def get_session_by_call(self, call_id: str) -> Optional[VoiceSession]:
        """Get session by call ID."""
        session_id = self._sessions_by_call.get(call_id)
        if session_id:
            return self._sessions.get(session_id)
        return None

    async def connect_session(self, session_id: str) -> bool:
        """Mark session as connected."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.state = StreamState.CONNECTED
        session.connected_at = datetime.utcnow()

        await self.event_handler.on_session_connected(session)
        return True

    async def end_session(self, session_id: str, reason: str = "") -> bool:
        """End voice session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            session.state = StreamState.CLOSED
            session.ended_at = datetime.utcnow()
            session.metadata["end_reason"] = reason

            # Update streams
            for stream in session.streams.values():
                stream.state = StreamState.CLOSED

            self._active_sessions -= 1

        await self.event_handler.on_session_ended(session)
        return True

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        now = datetime.utcnow()
        expired = []

        async with self._lock:
            for session_id, session in self._sessions.items():
                # Check idle timeout
                if session.state == StreamState.CONNECTED:
                    idle_time = (now - (session.connected_at or session.created_at)).total_seconds()
                    if idle_time > self.config.idle_timeout_seconds:
                        expired.append(session_id)

                # Check max duration
                if session.duration_seconds > self.config.max_session_duration_seconds:
                    expired.append(session_id)

        for session_id in expired:
            await self.end_session(session_id, "timeout")

        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        return {
            "total_sessions": self._total_sessions,
            "active_sessions": self._active_sessions,
            "sessions_by_tenant": {
                tid: len(sids) for tid, sids in self._sessions_by_tenant.items()
            },
        }


class AudioPipeline:
    """
    Audio processing pipeline.

    Processes audio through configurable stages:
    - Input processing
    - VAD
    - Noise reduction
    - Echo cancellation
    - Output processing
    """

    def __init__(self, config: Optional[VoiceEngineConfig] = None):
        self.config = config or VoiceEngineConfig()
        self._processors: List[Callable[[bytes], bytes]] = []
        self._async_processors: List[Callable[[bytes], Awaitable[bytes]]] = []

    def add_processor(
        self,
        processor: Callable[[bytes], bytes],
    ) -> "AudioPipeline":
        """Add synchronous processor."""
        self._processors.append(processor)
        return self

    def add_async_processor(
        self,
        processor: Callable[[bytes], Awaitable[bytes]],
    ) -> "AudioPipeline":
        """Add asynchronous processor."""
        self._async_processors.append(processor)
        return self

    async def process(self, audio: bytes) -> bytes:
        """Process audio through pipeline."""
        result = audio

        # Sync processors
        for processor in self._processors:
            result = processor(result)

        # Async processors
        for processor in self._async_processors:
            result = await processor(result)

        return result


class VoiceEngine:
    """
    Core voice engine.

    Manages:
    - Voice sessions
    - Audio streams
    - Real-time processing
    - Protocol handling
    """

    def __init__(
        self,
        config: Optional[VoiceEngineConfig] = None,
    ):
        self.config = config or VoiceEngineConfig()
        self._session_manager = SessionManager(self.config)
        self._audio_pipeline = AudioPipeline(self.config)

        # Protocol handlers
        self._protocol_handlers: Dict[str, Any] = {}

        # Event callbacks
        self._on_audio_callbacks: List[Callable[[VoiceSession, bytes], Awaitable[None]]] = []
        self._on_speech_callbacks: List[Callable[[VoiceSession, str], Awaitable[None]]] = []

        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Metrics
        self._audio_frames_processed = 0
        self._total_audio_duration_ms = 0

    async def start(self) -> None:
        """Start voice engine."""
        if self._running:
            return

        self._running = True

        # Start background tasks
        self._tasks.append(
            asyncio.create_task(self._cleanup_loop())
        )
        self._tasks.append(
            asyncio.create_task(self._metrics_loop())
        )

        logger.info("Voice engine started")

    async def stop(self) -> None:
        """Stop voice engine."""
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        logger.info("Voice engine stopped")

    async def create_session(
        self,
        tenant_id: str,
        session_type: SessionType,
        **kwargs,
    ) -> VoiceSession:
        """Create voice session."""
        session = await self._session_manager.create_session(
            tenant_id=tenant_id,
            session_type=session_type,
            **kwargs,
        )

        # Create default streams
        session.add_stream(
            direction="inbound",
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
        )
        session.add_stream(
            direction="outbound",
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
        )

        return session

    async def process_audio(
        self,
        session_id: str,
        audio: bytes,
        direction: str = "inbound",
    ) -> Optional[bytes]:
        """Process audio for session."""
        session = await self._session_manager.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        # Find stream
        stream = None
        for s in session.streams.values():
            if s.direction == direction:
                stream = s
                break

        if not stream:
            logger.warning(f"Stream not found for direction: {direction}")
            return None

        # Process through pipeline
        processed = await self._audio_pipeline.process(audio)

        # Store in stream
        await stream.receive(processed)

        # Update metrics
        self._audio_frames_processed += 1
        self._total_audio_duration_ms += self.config.frame_duration_ms

        # Trigger callbacks
        for callback in self._on_audio_callbacks:
            try:
                await callback(session, processed)
            except Exception as e:
                logger.error(f"Audio callback error: {e}")

        return processed

    async def send_audio(
        self,
        session_id: str,
        audio: bytes,
    ) -> bool:
        """Send audio to session."""
        session = await self._session_manager.get_session(session_id)
        if not session:
            return False

        stream = session.get_outbound_stream()
        if not stream:
            return False

        await stream.write(audio)
        return True

    async def end_session(self, session_id: str, reason: str = "") -> bool:
        """End voice session."""
        return await self._session_manager.end_session(session_id, reason)

    def on_audio(
        self,
        callback: Callable[[VoiceSession, bytes], Awaitable[None]],
    ) -> None:
        """Register audio callback."""
        self._on_audio_callbacks.append(callback)

    def on_speech(
        self,
        callback: Callable[[VoiceSession, str], Awaitable[None]],
    ) -> None:
        """Register speech callback."""
        self._on_speech_callbacks.append(callback)

    def register_protocol_handler(self, protocol: str, handler: Any) -> None:
        """Register protocol handler."""
        self._protocol_handlers[protocol] = handler

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(60)
                expired = await self._session_manager.cleanup_expired_sessions()
                if expired:
                    logger.info(f"Cleaned up {expired} expired sessions")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _metrics_loop(self) -> None:
        """Background metrics loop."""
        while self._running:
            try:
                await asyncio.sleep(30)
                stats = self.get_stats()
                logger.debug(f"Voice engine stats: {json.dumps(stats)}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get voice engine statistics."""
        return {
            "running": self._running,
            "audio_frames_processed": self._audio_frames_processed,
            "total_audio_duration_seconds": self._total_audio_duration_ms / 1000,
            "sessions": self._session_manager.get_stats(),
        }


class VoiceEngineFactory:
    """Factory for creating voice engines."""

    _instances: Dict[str, VoiceEngine] = {}

    @classmethod
    def create(
        cls,
        name: str = "default",
        config: Optional[VoiceEngineConfig] = None,
    ) -> VoiceEngine:
        """Create or get voice engine instance."""
        if name not in cls._instances:
            cls._instances[name] = VoiceEngine(config)
        return cls._instances[name]

    @classmethod
    def get(cls, name: str = "default") -> Optional[VoiceEngine]:
        """Get voice engine instance."""
        return cls._instances.get(name)

    @classmethod
    async def shutdown_all(cls) -> None:
        """Shutdown all engines."""
        for engine in cls._instances.values():
            await engine.stop()
        cls._instances.clear()
