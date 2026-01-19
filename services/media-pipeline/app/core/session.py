"""Media session - represents a single call's media handling."""

import asyncio
import structlog
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

from app.core.pipeline import MediaPipeline, PipelineConfig


logger = structlog.get_logger()


class SessionState(str, Enum):
    """Session lifecycle state."""
    PENDING = "pending"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ACTIVE = "active"
    HOLD = "hold"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


class MediaType(str, Enum):
    """Media type."""
    AUDIO = "audio"
    VIDEO = "video"  # Future support


@dataclass
class SessionConfig:
    """Session configuration."""
    call_id: str
    agent_id: str
    source: str = "telephony"  # telephony, webrtc
    direction: str = "inbound"  # inbound, outbound
    codec: str = "pcmu"
    sample_rate: int = 8000
    caller_number: Optional[str] = None
    callee_number: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionStats:
    """Session statistics."""
    created_at: float = field(default_factory=time.time)
    connected_at: Optional[float] = None
    disconnected_at: Optional[float] = None
    total_audio_received_bytes: int = 0
    total_audio_sent_bytes: int = 0
    total_speech_duration_ms: float = 0
    interruptions: int = 0
    asr_requests: int = 0
    tts_requests: int = 0
    llm_requests: int = 0

    @property
    def duration_seconds(self) -> float:
        """Get session duration."""
        end = self.disconnected_at or time.time()
        start = self.connected_at or self.created_at
        return end - start

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "connected_at": self.connected_at,
            "disconnected_at": self.disconnected_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_audio_received_bytes": self.total_audio_received_bytes,
            "total_audio_sent_bytes": self.total_audio_sent_bytes,
            "total_speech_duration_ms": round(self.total_speech_duration_ms, 2),
            "interruptions": self.interruptions,
            "asr_requests": self.asr_requests,
            "tts_requests": self.tts_requests,
            "llm_requests": self.llm_requests,
        }


class MediaSession:
    """
    Represents a single call's media session.

    Manages:
    - Media pipeline lifecycle
    - Audio routing configuration
    - Session state and statistics
    - Event handling
    """

    def __init__(self, config: SessionConfig):
        self.id = str(uuid.uuid4())
        self.config = config
        self.call_id = config.call_id
        self.agent_id = config.agent_id
        self.state = SessionState.PENDING
        self.stats = SessionStats()

        # Create pipeline
        pipeline_config = PipelineConfig(
            session_id=self.id,
            source_codec=config.codec,
            sample_rate=config.sample_rate,
        )
        self.pipeline = MediaPipeline(pipeline_config)

        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_speech_detected: Optional[Callable] = None
        self._on_transcript: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

        # Locks
        self._state_lock = asyncio.Lock()

        logger.info(
            "session_created",
            session_id=self.id,
            call_id=self.call_id,
            agent_id=self.agent_id,
            source=config.source,
        )

    async def connect(self) -> None:
        """Establish media connection."""
        async with self._state_lock:
            if self.state != SessionState.PENDING:
                raise RuntimeError(f"Cannot connect in state {self.state}")

            self.state = SessionState.CONNECTING

        try:
            # Setup pipeline callbacks
            self.pipeline.set_callbacks(
                on_speech_start=self._handle_speech_start,
                on_speech_end=self._handle_speech_end,
                on_vad_event=self._handle_vad_event,
            )

            # Start pipeline
            await self.pipeline.start()

            async with self._state_lock:
                self.state = SessionState.CONNECTED
                self.stats.connected_at = time.time()

            await self._notify_state_change()

            logger.info("session_connected", session_id=self.id)

        except Exception as e:
            async with self._state_lock:
                self.state = SessionState.FAILED

            logger.error(
                "session_connect_failed",
                session_id=self.id,
                error=str(e),
            )
            raise

    async def disconnect(self) -> None:
        """Disconnect media session."""
        async with self._state_lock:
            if self.state in (SessionState.DISCONNECTED, SessionState.DISCONNECTING):
                return

            self.state = SessionState.DISCONNECTING

        try:
            await self.pipeline.stop()

            async with self._state_lock:
                self.state = SessionState.DISCONNECTED
                self.stats.disconnected_at = time.time()

            await self._notify_state_change()

            logger.info(
                "session_disconnected",
                session_id=self.id,
                stats=self.stats.to_dict(),
            )

        except Exception as e:
            logger.error(
                "session_disconnect_error",
                session_id=self.id,
                error=str(e),
            )

    async def process_audio(self, audio_data: bytes, timestamp: int = 0) -> None:
        """
        Process incoming audio.

        Args:
            audio_data: Raw audio bytes
            timestamp: RTP timestamp (optional)
        """
        if self.state not in (SessionState.CONNECTED, SessionState.ACTIVE):
            return

        async with self._state_lock:
            if self.state == SessionState.CONNECTED:
                self.state = SessionState.ACTIVE

        await self.pipeline.process_inbound(audio_data, timestamp)
        self.stats.total_audio_received_bytes += len(audio_data)

    async def send_audio(self, audio_data: bytes, interrupt: bool = False) -> None:
        """
        Send audio to caller.

        Args:
            audio_data: PCM audio to send
            interrupt: If True, interrupt current playback
        """
        if self.state not in (SessionState.CONNECTED, SessionState.ACTIVE):
            return

        if interrupt:
            self.stats.interruptions += 1

        await self.pipeline.inject_audio(audio_data, priority=interrupt)
        self.stats.total_audio_sent_bytes += len(audio_data)

    async def hold(self) -> None:
        """Put session on hold."""
        async with self._state_lock:
            if self.state == SessionState.ACTIVE:
                self.state = SessionState.HOLD
                await self._notify_state_change()
                logger.info("session_on_hold", session_id=self.id)

    async def resume(self) -> None:
        """Resume session from hold."""
        async with self._state_lock:
            if self.state == SessionState.HOLD:
                self.state = SessionState.ACTIVE
                await self._notify_state_change()
                logger.info("session_resumed", session_id=self.id)

    def set_callbacks(
        self,
        on_state_change: Optional[Callable] = None,
        on_speech_detected: Optional[Callable] = None,
        on_transcript: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        """Set session callbacks."""
        self._on_state_change = on_state_change
        self._on_speech_detected = on_speech_detected
        self._on_transcript = on_transcript
        self._on_error = on_error

    async def _handle_speech_start(self) -> None:
        """Handle speech start event."""
        if self._on_speech_detected:
            await self._on_speech_detected("start")

    async def _handle_speech_end(self, duration: float) -> None:
        """Handle speech end event."""
        self.stats.total_speech_duration_ms += duration * 1000

        if self._on_speech_detected:
            await self._on_speech_detected("end", duration)

    async def _handle_vad_event(self, event_type: str, value: Any) -> None:
        """Handle VAD events."""
        logger.debug(
            "vad_event",
            session_id=self.id,
            event_type=event_type,
            value=value,
        )

    async def _notify_state_change(self) -> None:
        """Notify state change."""
        if self._on_state_change:
            await self._on_state_change(self.state)

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": self.id,
            "call_id": self.call_id,
            "agent_id": self.agent_id,
            "state": self.state.value,
            "pipeline": self.pipeline.get_metrics(),
            **self.stats.to_dict(),
        }

    def __repr__(self) -> str:
        return f"MediaSession(id={self.id}, call_id={self.call_id}, state={self.state})"
