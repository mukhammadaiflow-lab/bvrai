"""Core media pipeline - orchestrates audio flow."""

import asyncio
import structlog
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import time

from app.audio.processor import AudioProcessor
from app.buffer.jitter import JitterBuffer
from app.routing.router import AudioRouter
from app.codec.manager import CodecManager


logger = structlog.get_logger()


class PipelineState(str, Enum):
    """Pipeline state."""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    DRAINING = "draining"
    CLOSED = "closed"


class AudioDirection(str, Enum):
    """Audio flow direction."""
    INBOUND = "inbound"   # From caller to system
    OUTBOUND = "outbound"  # From system to caller
    BOTH = "both"


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    packets_received: int = 0
    packets_sent: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    packets_dropped: int = 0
    buffer_underruns: int = 0
    buffer_overruns: int = 0
    avg_latency_ms: float = 0.0
    peak_latency_ms: float = 0.0
    jitter_ms: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packets_received": self.packets_received,
            "packets_sent": self.packets_sent,
            "bytes_received": self.bytes_received,
            "bytes_sent": self.bytes_sent,
            "packets_dropped": self.packets_dropped,
            "buffer_underruns": self.buffer_underruns,
            "buffer_overruns": self.buffer_overruns,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "peak_latency_ms": round(self.peak_latency_ms, 2),
            "jitter_ms": round(self.jitter_ms, 2),
            "uptime_seconds": round(time.time() - self.created_at, 2),
        }


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    session_id: str
    source_codec: str = "pcmu"
    target_codec: str = "pcm16"
    sample_rate: int = 8000
    channels: int = 1
    jitter_buffer_ms: int = 100
    enable_agc: bool = True
    enable_noise_suppression: bool = True
    enable_echo_cancellation: bool = False
    vad_enabled: bool = True
    vad_threshold: float = 0.5


class MediaPipeline:
    """
    Core media pipeline for real-time audio processing.

    Handles:
    - Inbound audio from telephony/WebRTC
    - Codec transcoding
    - Jitter buffering
    - Audio processing (AGC, noise suppression)
    - VAD (Voice Activity Detection)
    - Routing to ASR/TTS services
    - Outbound audio to caller
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.session_id = config.session_id
        self.state = PipelineState.INITIALIZING
        self.metrics = PipelineMetrics()

        # Components
        self.codec_manager = CodecManager()
        self.audio_processor = AudioProcessor(
            sample_rate=config.sample_rate,
            channels=config.channels,
            enable_agc=config.enable_agc,
            enable_noise_suppression=config.enable_noise_suppression,
        )
        self.jitter_buffer = JitterBuffer(
            buffer_ms=config.jitter_buffer_ms,
            sample_rate=config.sample_rate,
        )
        self.router = AudioRouter(session_id=config.session_id)

        # Callbacks
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None
        self._on_audio_ready: Optional[Callable] = None
        self._on_vad_event: Optional[Callable] = None

        # State
        self._is_speaking = False
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._latency_samples: list[float] = []

        # Tasks
        self._process_task: Optional[asyncio.Task] = None
        self._output_task: Optional[asyncio.Task] = None

        logger.info("pipeline_created", session_id=self.session_id)

    async def start(self) -> None:
        """Start the pipeline."""
        if self.state != PipelineState.INITIALIZING:
            raise RuntimeError(f"Cannot start pipeline in state {self.state}")

        await self.router.connect()

        # Start processing tasks
        self._process_task = asyncio.create_task(self._process_loop())
        self._output_task = asyncio.create_task(self._output_loop())

        self.state = PipelineState.READY
        logger.info("pipeline_started", session_id=self.session_id)

    async def stop(self) -> None:
        """Stop the pipeline."""
        self.state = PipelineState.DRAINING

        # Cancel tasks
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        if self._output_task:
            self._output_task.cancel()
            try:
                await self._output_task
            except asyncio.CancelledError:
                pass

        await self.router.disconnect()
        self.state = PipelineState.CLOSED

        logger.info(
            "pipeline_stopped",
            session_id=self.session_id,
            metrics=self.metrics.to_dict(),
        )

    async def process_inbound(self, audio_data: bytes, timestamp: int) -> None:
        """
        Process inbound audio from caller.

        Args:
            audio_data: Raw audio bytes (in source codec)
            timestamp: RTP timestamp
        """
        if self.state not in (PipelineState.READY, PipelineState.ACTIVE):
            return

        self.state = PipelineState.ACTIVE
        start_time = time.perf_counter()

        try:
            # Decode from source codec
            pcm_data = self.codec_manager.decode(
                audio_data,
                self.config.source_codec,
            )

            # Add to jitter buffer
            self.jitter_buffer.push(pcm_data, timestamp)

            # Update metrics
            self.metrics.packets_received += 1
            self.metrics.bytes_received += len(audio_data)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_latency(latency_ms)

        except Exception as e:
            logger.error(
                "inbound_processing_error",
                session_id=self.session_id,
                error=str(e),
            )
            self.metrics.packets_dropped += 1

    async def send_outbound(self, audio_data: bytes) -> None:
        """
        Send audio to caller.

        Args:
            audio_data: PCM audio to send
        """
        if self.state not in (PipelineState.READY, PipelineState.ACTIVE):
            return

        try:
            # Encode to target codec
            encoded = self.codec_manager.encode(
                audio_data,
                self.config.source_codec,  # Same codec as source for now
            )

            # Send via router
            await self.router.send_audio(encoded)

            # Update metrics
            self.metrics.packets_sent += 1
            self.metrics.bytes_sent += len(encoded)

        except Exception as e:
            logger.error(
                "outbound_processing_error",
                session_id=self.session_id,
                error=str(e),
            )

    async def inject_audio(self, audio_data: bytes, priority: bool = False) -> None:
        """
        Inject audio into outbound stream (for TTS output).

        Args:
            audio_data: PCM audio to inject
            priority: If True, interrupt current playback
        """
        if priority:
            # Clear any pending audio
            self.router.clear_outbound_queue()

        await self.send_outbound(audio_data)

    def set_callbacks(
        self,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
        on_audio_ready: Optional[Callable] = None,
        on_vad_event: Optional[Callable] = None,
    ) -> None:
        """Set event callbacks."""
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._on_audio_ready = on_audio_ready
        self._on_vad_event = on_vad_event

    async def _process_loop(self) -> None:
        """Main processing loop."""
        frame_duration = self.config.jitter_buffer_ms / 1000 / 5  # Process 5x per buffer

        while self.state in (PipelineState.READY, PipelineState.ACTIVE):
            try:
                # Get audio from jitter buffer
                audio_frame = self.jitter_buffer.pop()

                if audio_frame is None:
                    self.metrics.buffer_underruns += 1
                    await asyncio.sleep(frame_duration)
                    continue

                # Process audio (AGC, noise suppression)
                processed = self.audio_processor.process(audio_frame)

                # VAD detection
                if self.config.vad_enabled:
                    is_speech = self.audio_processor.detect_voice(processed)
                    await self._handle_vad(is_speech)

                # Route to ASR
                await self.router.route_to_asr(processed)

                # Notify audio ready
                if self._on_audio_ready:
                    await self._on_audio_ready(processed)

                await asyncio.sleep(frame_duration)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "process_loop_error",
                    session_id=self.session_id,
                    error=str(e),
                )
                await asyncio.sleep(0.01)

    async def _output_loop(self) -> None:
        """Output processing loop for TTS audio."""
        while self.state in (PipelineState.READY, PipelineState.ACTIVE):
            try:
                # Get TTS audio from router
                tts_audio = await self.router.get_tts_audio()

                if tts_audio:
                    await self.send_outbound(tts_audio)

                await asyncio.sleep(0.02)  # 20ms

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "output_loop_error",
                    session_id=self.session_id,
                    error=str(e),
                )
                await asyncio.sleep(0.01)

    async def _handle_vad(self, is_speech: bool) -> None:
        """Handle VAD events."""
        now = time.time()

        if is_speech and not self._is_speaking:
            # Speech started
            self._is_speaking = True
            self._speech_start_time = now
            self._silence_start_time = None

            if self._on_speech_start:
                await self._on_speech_start()

            if self._on_vad_event:
                await self._on_vad_event("speech_start", now)

            logger.debug("speech_started", session_id=self.session_id)

        elif not is_speech and self._is_speaking:
            # Potential speech end - wait for silence duration
            if self._silence_start_time is None:
                self._silence_start_time = now

            silence_duration = now - self._silence_start_time

            # End of speech after 500ms silence
            if silence_duration >= 0.5:
                self._is_speaking = False
                speech_duration = now - (self._speech_start_time or now)

                if self._on_speech_end:
                    await self._on_speech_end(speech_duration)

                if self._on_vad_event:
                    await self._on_vad_event("speech_end", speech_duration)

                logger.debug(
                    "speech_ended",
                    session_id=self.session_id,
                    duration=speech_duration,
                )

        elif is_speech:
            # Reset silence timer if still speaking
            self._silence_start_time = None

    def _update_latency(self, latency_ms: float) -> None:
        """Update latency metrics."""
        self._latency_samples.append(latency_ms)

        # Keep last 100 samples
        if len(self._latency_samples) > 100:
            self._latency_samples = self._latency_samples[-100:]

        self.metrics.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)
        self.metrics.peak_latency_ms = max(self.metrics.peak_latency_ms, latency_ms)

        # Calculate jitter (standard deviation)
        if len(self._latency_samples) > 1:
            mean = self.metrics.avg_latency_ms
            variance = sum((x - mean) ** 2 for x in self._latency_samples) / len(self._latency_samples)
            self.metrics.jitter_ms = variance ** 0.5

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        return self.metrics.to_dict()

    @property
    def is_active(self) -> bool:
        """Check if pipeline is active."""
        return self.state == PipelineState.ACTIVE
