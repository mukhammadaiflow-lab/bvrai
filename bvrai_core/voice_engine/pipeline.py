"""
Voice Pipeline - Main orchestrator for real-time voice conversations.

This module provides the VoicePipeline class that coordinates all voice
processing components into a cohesive, real-time conversational system.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from .audio import (
    AudioBuffer,
    AudioChunk,
    AudioCodec,
    AudioDecoder,
    AudioEncoder,
    AudioFormat,
    AudioResampler,
)
from .vad import (
    VADConfig,
    VADEvent,
    VADEventType,
    VADFactory,
    VoiceActivityDetector,
)
from .stt import (
    STTConfig,
    STTProviderFactory,
    STTProvider,
    TranscriptionResult,
    TranscriptionSegment,
)
from .tts import (
    TTSConfig,
    TTSProviderFactory,
    TTSProvider,
    TTSVoice,
    VoiceStyle,
)
from .interruption import (
    InterruptionConfig,
    InterruptionEvent,
    InterruptionHandler,
    InterruptionStrategy,
)


logger = logging.getLogger(__name__)


class PipelineEventType(str, Enum):
    """Types of events emitted by the voice pipeline."""

    # Lifecycle events
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_STOPPED = "pipeline_stopped"
    PIPELINE_ERROR = "pipeline_error"
    PIPELINE_STATE_CHANGED = "pipeline_state_changed"

    # Audio events
    AUDIO_RECEIVED = "audio_received"
    AUDIO_SENT = "audio_sent"
    AUDIO_LEVEL = "audio_level"

    # VAD events
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    SILENCE_DETECTED = "silence_detected"

    # Transcription events
    TRANSCRIPTION_PARTIAL = "transcription_partial"
    TRANSCRIPTION_FINAL = "transcription_final"
    TRANSCRIPTION_ERROR = "transcription_error"

    # Response events
    RESPONSE_STARTED = "response_started"
    RESPONSE_TEXT = "response_text"
    RESPONSE_COMPLETED = "response_completed"
    RESPONSE_ERROR = "response_error"

    # TTS events
    TTS_STARTED = "tts_started"
    TTS_AUDIO = "tts_audio"
    TTS_COMPLETED = "tts_completed"
    TTS_ERROR = "tts_error"

    # Interruption events
    INTERRUPTION_DETECTED = "interruption_detected"
    INTERRUPTION_HANDLED = "interruption_handled"

    # Turn events
    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"

    # Conversation events
    CONVERSATION_STARTED = "conversation_started"
    CONVERSATION_ENDED = "conversation_ended"


class VoicePipelineState(str, Enum):
    """State of the voice pipeline."""

    IDLE = "idle"                       # Pipeline created but not started
    STARTING = "starting"               # Pipeline is initializing
    LISTENING = "listening"             # Waiting for user speech
    PROCESSING_SPEECH = "processing_speech"  # Processing user speech (VAD + STT)
    GENERATING_RESPONSE = "generating_response"  # LLM is generating response
    SPEAKING = "speaking"               # TTS is playing response
    INTERRUPTED = "interrupted"         # User interrupted AI speech
    PAUSED = "paused"                   # Pipeline temporarily paused
    STOPPING = "stopping"               # Pipeline is shutting down
    STOPPED = "stopped"                 # Pipeline fully stopped
    ERROR = "error"                     # Pipeline in error state


@dataclass
class PipelineEvent:
    """Event emitted by the voice pipeline."""

    type: PipelineEventType
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    # Optional context
    turn_id: Optional[str] = None
    conversation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
        }


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""

    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # User input
    user_audio_duration_ms: float = 0.0
    user_transcript: str = ""
    user_transcript_segments: List[TranscriptionSegment] = field(default_factory=list)

    # AI response
    ai_response_text: str = ""
    ai_audio_duration_ms: float = 0.0

    # Timing
    started_at: float = field(default_factory=time.time)
    speech_started_at: Optional[float] = None
    speech_ended_at: Optional[float] = None
    response_started_at: Optional[float] = None
    response_ended_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Interruption info
    was_interrupted: bool = False
    interruption_point_ms: Optional[float] = None

    # Metrics
    stt_latency_ms: Optional[float] = None
    llm_first_token_ms: Optional[float] = None
    tts_first_audio_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None

    def calculate_metrics(self) -> None:
        """Calculate turn metrics."""
        if self.speech_ended_at and self.response_started_at:
            self.stt_latency_ms = (self.response_started_at - self.speech_ended_at) * 1000

        if self.speech_ended_at and self.completed_at:
            self.total_latency_ms = (self.completed_at - self.speech_ended_at) * 1000


@dataclass
class ConversationContext:
    """Context for the current conversation."""

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Conversation history
    turns: List[ConversationTurn] = field(default_factory=list)

    # Current state
    current_turn: Optional[ConversationTurn] = None

    # Metadata
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    # User info (for personalization)
    user_id: Optional[str] = None
    user_metadata: Dict[str, Any] = field(default_factory=dict)

    # Agent info
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None

    # Session data (persisted across turns)
    session_data: Dict[str, Any] = field(default_factory=dict)

    # Extracted entities and intents
    entities: Dict[str, Any] = field(default_factory=dict)
    intents: List[str] = field(default_factory=list)

    def get_transcript_history(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """Get conversation history as list of messages."""
        history = []
        for turn in self.turns[-max_turns:]:
            if turn.user_transcript:
                history.append({"role": "user", "content": turn.user_transcript})
            if turn.ai_response_text:
                history.append({"role": "assistant", "content": turn.ai_response_text})
        return history

    def start_new_turn(self) -> ConversationTurn:
        """Start a new conversation turn."""
        self.current_turn = ConversationTurn()
        return self.current_turn

    def complete_current_turn(self) -> Optional[ConversationTurn]:
        """Complete and archive the current turn."""
        if self.current_turn:
            self.current_turn.completed_at = time.time()
            self.current_turn.calculate_metrics()
            self.turns.append(self.current_turn)
            completed = self.current_turn
            self.current_turn = None
            return completed
        return None


@dataclass
class VoicePipelineConfig:
    """Configuration for the voice pipeline."""

    # Audio format
    input_format: AudioFormat = field(default_factory=lambda: AudioFormat.telephony())
    output_format: AudioFormat = field(default_factory=lambda: AudioFormat.telephony())

    # Provider selection
    stt_provider: str = "deepgram"
    tts_provider: str = "elevenlabs"
    vad_type: str = "adaptive"

    # Provider configs
    stt_config: Optional[STTConfig] = None
    tts_config: Optional[TTSConfig] = None
    vad_config: Optional[VADConfig] = None
    interruption_config: Optional[InterruptionConfig] = None

    # API keys (will be loaded from environment if not provided)
    deepgram_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    google_credentials_path: Optional[str] = None
    azure_speech_key: Optional[str] = None
    azure_speech_region: Optional[str] = None
    assemblyai_api_key: Optional[str] = None
    playht_api_key: Optional[str] = None
    playht_user_id: Optional[str] = None

    # Voice settings
    voice_id: Optional[str] = None
    voice_name: Optional[str] = None
    speaking_rate: float = 1.0
    voice_style: VoiceStyle = VoiceStyle.FRIENDLY

    # Timing settings
    end_of_speech_silence_ms: int = 700  # Silence duration to consider speech ended
    max_speech_duration_ms: int = 60000  # Maximum single utterance duration
    min_speech_duration_ms: int = 100    # Minimum speech to process
    response_delay_ms: int = 0           # Delay before starting response

    # Buffer settings
    audio_buffer_duration_ms: int = 30000  # Max audio buffer size

    # Behavior settings
    enable_interruption: bool = True
    interruption_strategy: InterruptionStrategy = InterruptionStrategy.SMART
    enable_backchanneling: bool = True    # "mm-hmm", "I see", etc.
    enable_filler_words: bool = False     # "um", "let me think", etc.

    # Streaming settings
    stream_stt: bool = True               # Use streaming STT
    stream_tts: bool = True               # Use streaming TTS

    # Debug settings
    log_audio_levels: bool = False
    save_audio_recordings: bool = False
    recordings_path: Optional[str] = None


class ResponseGenerator(ABC):
    """Abstract base class for generating AI responses."""

    @abstractmethod
    async def generate(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> AsyncIterator[str]:
        """
        Generate response text given user transcript and context.

        Yields response text chunks for streaming.
        """
        pass

    @abstractmethod
    async def generate_backchannel(
        self,
        context: ConversationContext,
    ) -> Optional[str]:
        """Generate a backchannel response (e.g., "mm-hmm")."""
        pass


class SimpleResponseGenerator(ResponseGenerator):
    """Simple response generator for testing."""

    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or [
            "I understand. Please tell me more.",
            "That's interesting. How can I help you with that?",
            "I see. Let me help you with that.",
        ]
        self._response_index = 0

    async def generate(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> AsyncIterator[str]:
        """Generate a simple response."""
        response = self.responses[self._response_index % len(self.responses)]
        self._response_index += 1

        # Simulate streaming
        words = response.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.05)

    async def generate_backchannel(
        self,
        context: ConversationContext,
    ) -> Optional[str]:
        """Generate a backchannel."""
        import random
        backchannels = ["mm-hmm", "I see", "okay", "right", "got it"]
        return random.choice(backchannels)


class VoicePipeline:
    """
    Main voice pipeline orchestrator.

    Coordinates audio input/output, VAD, STT, response generation, TTS,
    and interruption handling into a cohesive real-time conversation system.
    """

    def __init__(
        self,
        config: VoicePipelineConfig,
        response_generator: ResponseGenerator,
    ):
        self.config = config
        self.response_generator = response_generator

        # State
        self._state = VoicePipelineState.IDLE
        self._previous_state = VoicePipelineState.IDLE

        # Components (initialized in start())
        self._vad: Optional[VoiceActivityDetector] = None
        self._stt: Optional[STTProvider] = None
        self._tts: Optional[TTSProvider] = None
        self._interruption_handler: Optional[InterruptionHandler] = None

        # Audio processing
        self._input_resampler: Optional[AudioResampler] = None
        self._output_resampler: Optional[AudioResampler] = None
        self._input_buffer: Optional[AudioBuffer] = None
        self._output_buffer: Optional[AudioBuffer] = None

        # Conversation
        self._context: Optional[ConversationContext] = None

        # Event handling
        self._event_handlers: List[Callable[[PipelineEvent], None]] = []
        self._async_event_handlers: List[Callable[[PipelineEvent], asyncio.Future]] = []

        # Tasks
        self._tasks: Set[asyncio.Task] = set()
        self._main_loop_task: Optional[asyncio.Task] = None

        # Audio queues
        self._input_audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        self._output_audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()

        # Control flags
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially

        # Current speech tracking
        self._speech_buffer: List[AudioChunk] = []
        self._speech_start_time: Optional[float] = None
        self._is_speaking: bool = False

        # TTS playback tracking
        self._tts_playing: bool = False
        self._tts_text_position: int = 0
        self._tts_audio_position_ms: float = 0.0

        # Metrics
        self._total_input_audio_ms: float = 0.0
        self._total_output_audio_ms: float = 0.0

    @property
    def state(self) -> VoicePipelineState:
        """Get current pipeline state."""
        return self._state

    @property
    def context(self) -> Optional[ConversationContext]:
        """Get current conversation context."""
        return self._context

    def add_event_handler(
        self,
        handler: Callable[[PipelineEvent], None],
        async_handler: bool = False,
    ) -> None:
        """Add an event handler."""
        if async_handler:
            self._async_event_handlers.append(handler)
        else:
            self._event_handlers.append(handler)

    def remove_event_handler(
        self,
        handler: Callable[[PipelineEvent], None],
    ) -> None:
        """Remove an event handler."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)
        if handler in self._async_event_handlers:
            self._async_event_handlers.remove(handler)

    async def _emit_event(self, event: PipelineEvent) -> None:
        """Emit an event to all handlers."""
        # Add context info
        if self._context:
            event.conversation_id = self._context.conversation_id
            if self._context.current_turn:
                event.turn_id = self._context.current_turn.turn_id

        # Sync handlers
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

        # Async handlers
        for handler in self._async_event_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Async event handler error: {e}")

    async def _set_state(self, new_state: VoicePipelineState) -> None:
        """Set pipeline state and emit event."""
        if new_state != self._state:
            self._previous_state = self._state
            self._state = new_state

            await self._emit_event(PipelineEvent(
                type=PipelineEventType.PIPELINE_STATE_CHANGED,
                data={
                    "previous_state": self._previous_state.value,
                    "new_state": new_state.value,
                },
            ))

    async def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        # Initialize VAD
        vad_config = self.config.vad_config or VADConfig(
            speech_threshold=0.5,
            silence_threshold=0.3,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.end_of_speech_silence_ms,
        )
        self._vad = VADFactory.create(self.config.vad_type, vad_config)

        # Initialize STT
        stt_config = self.config.stt_config or STTConfig(
            language="en-US",
            enable_punctuation=True,
            enable_word_timestamps=True,
        )

        stt_kwargs = {"config": stt_config}
        if self.config.stt_provider == "deepgram" and self.config.deepgram_api_key:
            stt_kwargs["api_key"] = self.config.deepgram_api_key
        elif self.config.stt_provider == "whisper" and self.config.openai_api_key:
            stt_kwargs["api_key"] = self.config.openai_api_key
        elif self.config.stt_provider == "google" and self.config.google_credentials_path:
            stt_kwargs["credentials_path"] = self.config.google_credentials_path
        elif self.config.stt_provider == "azure":
            stt_kwargs["subscription_key"] = self.config.azure_speech_key
            stt_kwargs["region"] = self.config.azure_speech_region
        elif self.config.stt_provider == "assemblyai" and self.config.assemblyai_api_key:
            stt_kwargs["api_key"] = self.config.assemblyai_api_key

        self._stt = STTProviderFactory.create(self.config.stt_provider, **stt_kwargs)

        # Initialize TTS
        tts_config = self.config.tts_config or TTSConfig(
            speaking_rate=self.config.speaking_rate,
            style=self.config.voice_style,
        )

        tts_kwargs = {"config": tts_config}
        if self.config.tts_provider == "elevenlabs" and self.config.elevenlabs_api_key:
            tts_kwargs["api_key"] = self.config.elevenlabs_api_key
        elif self.config.tts_provider == "openai" and self.config.openai_api_key:
            tts_kwargs["api_key"] = self.config.openai_api_key
        elif self.config.tts_provider == "google" and self.config.google_credentials_path:
            tts_kwargs["credentials_path"] = self.config.google_credentials_path
        elif self.config.tts_provider == "azure":
            tts_kwargs["subscription_key"] = self.config.azure_speech_key
            tts_kwargs["region"] = self.config.azure_speech_region
        elif self.config.tts_provider == "playht":
            tts_kwargs["api_key"] = self.config.playht_api_key
            tts_kwargs["user_id"] = self.config.playht_user_id

        self._tts = TTSProviderFactory.create(self.config.tts_provider, **tts_kwargs)

        # Initialize interruption handler
        interruption_config = self.config.interruption_config or InterruptionConfig(
            strategy=self.config.interruption_strategy,
            min_speech_for_interrupt_ms=200,
            interrupt_energy_threshold=0.6,
        )
        self._interruption_handler = InterruptionHandler(interruption_config)

        # Initialize audio buffers
        self._input_buffer = AudioBuffer(
            format=self.config.input_format,
            max_duration_ms=self.config.audio_buffer_duration_ms,
        )
        self._output_buffer = AudioBuffer(
            format=self.config.output_format,
            max_duration_ms=self.config.audio_buffer_duration_ms,
        )

        # Initialize resamplers if needed
        internal_format = AudioFormat(sample_rate=16000, channels=1, sample_width=2)

        if self.config.input_format.sample_rate != internal_format.sample_rate:
            self._input_resampler = AudioResampler(
                input_rate=self.config.input_format.sample_rate,
                output_rate=internal_format.sample_rate,
                channels=self.config.input_format.channels,
                sample_width=self.config.input_format.sample_width,
            )

        if self.config.output_format.sample_rate != internal_format.sample_rate:
            self._output_resampler = AudioResampler(
                input_rate=internal_format.sample_rate,
                output_rate=self.config.output_format.sample_rate,
                channels=internal_format.channels,
                sample_width=internal_format.sample_width,
            )

    async def _cleanup_components(self) -> None:
        """Cleanup all pipeline components."""
        if self._stt:
            await self._stt.disconnect()
            self._stt = None

        if self._tts:
            self._tts = None

        self._vad = None
        self._interruption_handler = None
        self._input_buffer = None
        self._output_buffer = None
        self._input_resampler = None
        self._output_resampler = None

    async def start(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Start the voice pipeline.

        Args:
            user_id: Optional user identifier
            agent_id: Optional agent identifier
            agent_name: Optional agent name for the conversation
            initial_context: Optional initial session data
        """
        if self._state != VoicePipelineState.IDLE:
            raise RuntimeError(f"Cannot start pipeline in state: {self._state}")

        await self._set_state(VoicePipelineState.STARTING)

        try:
            # Initialize components
            await self._initialize_components()

            # Connect STT if streaming
            if self.config.stream_stt and self._stt:
                await self._stt.connect()

            # Create conversation context
            self._context = ConversationContext(
                user_id=user_id,
                agent_id=agent_id,
                agent_name=agent_name,
            )
            if initial_context:
                self._context.session_data.update(initial_context)

            # Reset control flags
            self._stop_event.clear()
            self._pause_event.set()

            # Clear queues
            while not self._input_audio_queue.empty():
                self._input_audio_queue.get_nowait()
            while not self._output_audio_queue.empty():
                self._output_audio_queue.get_nowait()

            # Start main processing loop
            self._main_loop_task = asyncio.create_task(self._main_loop())
            self._tasks.add(self._main_loop_task)

            await self._set_state(VoicePipelineState.LISTENING)

            await self._emit_event(PipelineEvent(
                type=PipelineEventType.PIPELINE_STARTED,
                data={
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                },
            ))

            await self._emit_event(PipelineEvent(
                type=PipelineEventType.CONVERSATION_STARTED,
            ))

            logger.info(f"Voice pipeline started - conversation_id={self._context.conversation_id}")

        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            await self._set_state(VoicePipelineState.ERROR)
            await self._cleanup_components()
            raise

    async def stop(self) -> None:
        """Stop the voice pipeline."""
        if self._state in (VoicePipelineState.IDLE, VoicePipelineState.STOPPED):
            return

        await self._set_state(VoicePipelineState.STOPPING)

        # Signal stop
        self._stop_event.set()

        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Complete current turn if any
        if self._context and self._context.current_turn:
            self._context.complete_current_turn()

        # Mark conversation ended
        if self._context:
            self._context.ended_at = time.time()

        # Cleanup
        await self._cleanup_components()

        await self._set_state(VoicePipelineState.STOPPED)

        await self._emit_event(PipelineEvent(
            type=PipelineEventType.CONVERSATION_ENDED,
        ))

        await self._emit_event(PipelineEvent(
            type=PipelineEventType.PIPELINE_STOPPED,
        ))

        logger.info("Voice pipeline stopped")

    async def pause(self) -> None:
        """Pause the pipeline."""
        if self._state == VoicePipelineState.PAUSED:
            return

        self._pause_event.clear()
        await self._set_state(VoicePipelineState.PAUSED)
        logger.info("Voice pipeline paused")

    async def resume(self) -> None:
        """Resume the pipeline."""
        if self._state != VoicePipelineState.PAUSED:
            return

        self._pause_event.set()
        await self._set_state(VoicePipelineState.LISTENING)
        logger.info("Voice pipeline resumed")

    async def push_audio(self, audio: Union[bytes, AudioChunk]) -> None:
        """
        Push audio data into the pipeline.

        Args:
            audio: Raw audio bytes or AudioChunk
        """
        if self._state in (VoicePipelineState.IDLE, VoicePipelineState.STOPPED):
            return

        # Convert bytes to AudioChunk if needed
        if isinstance(audio, bytes):
            chunk = AudioChunk(
                data=audio,
                format=self.config.input_format,
                timestamp=time.time(),
            )
        else:
            chunk = audio

        # Resample if needed
        if self._input_resampler:
            chunk = self._input_resampler.resample_chunk(chunk)

        # Add to queue
        await self._input_audio_queue.put(chunk)

        self._total_input_audio_ms += chunk.duration_ms

    async def get_output_audio(self) -> AsyncIterator[AudioChunk]:
        """
        Get output audio from the pipeline.

        Yields AudioChunks as they become available.
        """
        while not self._stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(
                    self._output_audio_queue.get(),
                    timeout=0.1,
                )
                yield chunk
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _main_loop(self) -> None:
        """Main processing loop."""
        try:
            while not self._stop_event.is_set():
                # Wait if paused
                await self._pause_event.wait()

                if self._stop_event.is_set():
                    break

                # Get audio from queue
                try:
                    chunk = await asyncio.wait_for(
                        self._input_audio_queue.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    continue

                # Process based on current state
                if self._state == VoicePipelineState.LISTENING:
                    await self._process_listening(chunk)

                elif self._state == VoicePipelineState.PROCESSING_SPEECH:
                    await self._process_speech(chunk)

                elif self._state == VoicePipelineState.SPEAKING:
                    await self._process_during_speech(chunk)

                elif self._state == VoicePipelineState.GENERATING_RESPONSE:
                    # Buffer audio while generating
                    await self._process_during_response_generation(chunk)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            await self._set_state(VoicePipelineState.ERROR)
            await self._emit_event(PipelineEvent(
                type=PipelineEventType.PIPELINE_ERROR,
                data={"error": str(e)},
            ))

    async def _process_listening(self, chunk: AudioChunk) -> None:
        """Process audio while in listening state."""
        if not self._vad:
            return

        # Emit audio level if configured
        if self.config.log_audio_levels:
            await self._emit_event(PipelineEvent(
                type=PipelineEventType.AUDIO_LEVEL,
                data={"level_db": chunk.level_db, "rms": chunk.rms},
            ))

        # Run VAD
        vad_event = await self._vad.process(chunk)

        if vad_event and vad_event.event_type == VADEventType.SPEECH_START:
            # Speech started
            await self._set_state(VoicePipelineState.PROCESSING_SPEECH)

            # Start new turn
            if self._context:
                turn = self._context.start_new_turn()
                turn.speech_started_at = time.time()

            # Initialize speech buffer
            self._speech_buffer = [chunk]
            self._speech_start_time = time.time()
            self._is_speaking = True

            await self._emit_event(PipelineEvent(
                type=PipelineEventType.SPEECH_STARTED,
            ))

            await self._emit_event(PipelineEvent(
                type=PipelineEventType.TURN_STARTED,
            ))

            # Start streaming STT if enabled
            if self.config.stream_stt and self._stt:
                self._tasks.add(asyncio.create_task(
                    self._stream_stt_processing()
                ))

    async def _process_speech(self, chunk: AudioChunk) -> None:
        """Process audio while user is speaking."""
        if not self._vad:
            return

        # Add to speech buffer
        self._speech_buffer.append(chunk)

        # Check max duration
        if self._speech_start_time:
            speech_duration = (time.time() - self._speech_start_time) * 1000
            if speech_duration > self.config.max_speech_duration_ms:
                logger.warning("Max speech duration reached, processing...")
                await self._finalize_speech()
                return

        # Run VAD
        vad_event = await self._vad.process(chunk)

        if vad_event and vad_event.event_type == VADEventType.SPEECH_END:
            # Speech ended
            await self._finalize_speech()

    async def _finalize_speech(self) -> None:
        """Finalize speech and start response generation."""
        self._is_speaking = False

        # Update turn timing
        if self._context and self._context.current_turn:
            self._context.current_turn.speech_ended_at = time.time()

        await self._emit_event(PipelineEvent(
            type=PipelineEventType.SPEECH_ENDED,
            data={
                "duration_ms": len(self._speech_buffer) * 20,  # Approximate
            },
        ))

        # Calculate total audio duration
        total_duration_ms = sum(chunk.duration_ms for chunk in self._speech_buffer)

        # Skip if too short
        if total_duration_ms < self.config.min_speech_duration_ms:
            logger.debug("Speech too short, ignoring")
            await self._set_state(VoicePipelineState.LISTENING)
            self._speech_buffer.clear()
            if self._context:
                self._context.current_turn = None
            return

        # Transition to generating response
        await self._set_state(VoicePipelineState.GENERATING_RESPONSE)

        # Transcribe if not already done via streaming
        if not self.config.stream_stt:
            await self._transcribe_speech()

        # Generate and speak response
        await self._generate_and_speak_response()

    async def _stream_stt_processing(self) -> None:
        """Process speech with streaming STT."""
        if not self._stt or not self._speech_buffer:
            return

        try:
            # Stream audio chunks to STT
            async def audio_generator():
                # Yield buffered chunks
                for chunk in self._speech_buffer:
                    yield chunk

                # Continue yielding new chunks while speaking
                while self._is_speaking and not self._stop_event.is_set():
                    if self._speech_buffer:
                        # Wait a bit for more audio
                        await asyncio.sleep(0.02)
                        if len(self._speech_buffer) > 0:
                            yield self._speech_buffer[-1]

            # Get transcription results
            async for result in self._stt.transcribe_stream(audio_generator()):
                if result.segments:
                    segment = result.segments[-1]

                    if segment.is_final:
                        await self._emit_event(PipelineEvent(
                            type=PipelineEventType.TRANSCRIPTION_FINAL,
                            data={
                                "text": segment.text,
                                "confidence": segment.confidence,
                            },
                        ))

                        # Update turn
                        if self._context and self._context.current_turn:
                            self._context.current_turn.user_transcript = segment.text
                            self._context.current_turn.user_transcript_segments.append(segment)
                    else:
                        await self._emit_event(PipelineEvent(
                            type=PipelineEventType.TRANSCRIPTION_PARTIAL,
                            data={"text": segment.text},
                        ))

        except Exception as e:
            logger.error(f"Streaming STT error: {e}")
            await self._emit_event(PipelineEvent(
                type=PipelineEventType.TRANSCRIPTION_ERROR,
                data={"error": str(e)},
            ))

    async def _transcribe_speech(self) -> None:
        """Transcribe buffered speech (non-streaming)."""
        if not self._stt or not self._speech_buffer:
            return

        try:
            # Concatenate audio
            audio_data = b"".join(chunk.data for chunk in self._speech_buffer)
            audio_format = self._speech_buffer[0].format if self._speech_buffer else self.config.input_format

            # Transcribe
            result = await self._stt.transcribe(audio_data, audio_format)

            if result.segments:
                transcript = " ".join(s.text for s in result.segments)

                await self._emit_event(PipelineEvent(
                    type=PipelineEventType.TRANSCRIPTION_FINAL,
                    data={
                        "text": transcript,
                        "segments": [s.text for s in result.segments],
                    },
                ))

                # Update turn
                if self._context and self._context.current_turn:
                    self._context.current_turn.user_transcript = transcript
                    self._context.current_turn.user_transcript_segments = result.segments

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            await self._emit_event(PipelineEvent(
                type=PipelineEventType.TRANSCRIPTION_ERROR,
                data={"error": str(e)},
            ))

    async def _generate_and_speak_response(self) -> None:
        """Generate AI response and speak it."""
        if not self._context or not self._context.current_turn:
            await self._set_state(VoicePipelineState.LISTENING)
            return

        transcript = self._context.current_turn.user_transcript
        if not transcript:
            logger.warning("No transcript available")
            await self._set_state(VoicePipelineState.LISTENING)
            return

        # Optional delay before responding
        if self.config.response_delay_ms > 0:
            await asyncio.sleep(self.config.response_delay_ms / 1000)

        try:
            await self._emit_event(PipelineEvent(
                type=PipelineEventType.RESPONSE_STARTED,
            ))

            self._context.current_turn.response_started_at = time.time()

            # Generate response
            response_text = ""
            first_token_time = None

            async for chunk in self.response_generator.generate(transcript, self._context):
                if first_token_time is None:
                    first_token_time = time.time()
                    if self._context.current_turn.speech_ended_at:
                        self._context.current_turn.llm_first_token_ms = (
                            first_token_time - self._context.current_turn.speech_ended_at
                        ) * 1000

                response_text += chunk

                await self._emit_event(PipelineEvent(
                    type=PipelineEventType.RESPONSE_TEXT,
                    data={"chunk": chunk, "full_text": response_text},
                ))

            self._context.current_turn.ai_response_text = response_text

            await self._emit_event(PipelineEvent(
                type=PipelineEventType.RESPONSE_COMPLETED,
                data={"text": response_text},
            ))

            # Speak the response
            if response_text and self._tts:
                await self._speak_response(response_text)

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            await self._emit_event(PipelineEvent(
                type=PipelineEventType.RESPONSE_ERROR,
                data={"error": str(e)},
            ))

        # Complete turn
        if self._context:
            completed_turn = self._context.complete_current_turn()
            if completed_turn:
                await self._emit_event(PipelineEvent(
                    type=PipelineEventType.TURN_COMPLETED,
                    data={
                        "user_transcript": completed_turn.user_transcript,
                        "ai_response": completed_turn.ai_response_text,
                        "total_latency_ms": completed_turn.total_latency_ms,
                    },
                ))

        # Clear speech buffer
        self._speech_buffer.clear()

        # Return to listening
        await self._set_state(VoicePipelineState.LISTENING)

    async def _speak_response(self, text: str) -> None:
        """Speak response text using TTS."""
        if not self._tts:
            return

        await self._set_state(VoicePipelineState.SPEAKING)
        self._tts_playing = True
        self._tts_text_position = 0
        self._tts_audio_position_ms = 0.0

        await self._emit_event(PipelineEvent(
            type=PipelineEventType.TTS_STARTED,
            data={"text": text},
        ))

        # Update interruption handler
        if self._interruption_handler:
            self._interruption_handler.set_ai_speaking(True, text)

        try:
            first_audio_time = None

            if self.config.stream_tts:
                # Streaming TTS
                async for audio_chunk in self._tts.synthesize_stream(text):
                    if self._stop_event.is_set():
                        break

                    # Check for interruption
                    if not self._tts_playing:
                        break

                    if first_audio_time is None:
                        first_audio_time = time.time()
                        if self._context and self._context.current_turn:
                            if self._context.current_turn.speech_ended_at:
                                self._context.current_turn.tts_first_audio_ms = (
                                    first_audio_time - self._context.current_turn.speech_ended_at
                                ) * 1000

                    # Resample if needed
                    if self._output_resampler:
                        audio_chunk = self._output_resampler.resample_chunk(audio_chunk)

                    # Send to output queue
                    await self._output_audio_queue.put(audio_chunk)

                    # Track position
                    self._tts_audio_position_ms += audio_chunk.duration_ms
                    self._total_output_audio_ms += audio_chunk.duration_ms

                    await self._emit_event(PipelineEvent(
                        type=PipelineEventType.TTS_AUDIO,
                        data={
                            "duration_ms": audio_chunk.duration_ms,
                            "total_ms": self._tts_audio_position_ms,
                        },
                    ))
            else:
                # Non-streaming TTS
                audio_data = await self._tts.synthesize(text)

                if audio_data:
                    # Create chunk
                    chunk = AudioChunk(
                        data=audio_data,
                        format=self.config.output_format,
                        timestamp=time.time(),
                    )

                    # Resample if needed
                    if self._output_resampler:
                        chunk = self._output_resampler.resample_chunk(chunk)

                    # Send to output queue
                    await self._output_audio_queue.put(chunk)

                    self._tts_audio_position_ms += chunk.duration_ms
                    self._total_output_audio_ms += chunk.duration_ms

            await self._emit_event(PipelineEvent(
                type=PipelineEventType.TTS_COMPLETED,
                data={
                    "total_duration_ms": self._tts_audio_position_ms,
                },
            ))

        except Exception as e:
            logger.error(f"TTS error: {e}")
            await self._emit_event(PipelineEvent(
                type=PipelineEventType.TTS_ERROR,
                data={"error": str(e)},
            ))

        finally:
            self._tts_playing = False
            if self._interruption_handler:
                self._interruption_handler.set_ai_speaking(False, "")

            # Update turn audio duration
            if self._context and self._context.current_turn:
                self._context.current_turn.ai_audio_duration_ms = self._tts_audio_position_ms

    async def _process_during_speech(self, chunk: AudioChunk) -> None:
        """Process audio while AI is speaking (for interruption detection)."""
        if not self.config.enable_interruption or not self._interruption_handler:
            return

        # Check for interruption
        interruption = await self._interruption_handler.process_audio(
            chunk,
            self._tts_audio_position_ms,
        )

        if interruption and interruption.should_stop:
            await self._handle_interruption(interruption)

    async def _process_during_response_generation(self, chunk: AudioChunk) -> None:
        """Process audio while generating response."""
        # Could buffer for next turn or check for cancellation intent
        pass

    async def _handle_interruption(self, interruption: InterruptionEvent) -> None:
        """Handle user interruption."""
        await self._set_state(VoicePipelineState.INTERRUPTED)

        # Stop TTS playback
        self._tts_playing = False

        # Update turn
        if self._context and self._context.current_turn:
            self._context.current_turn.was_interrupted = True
            self._context.current_turn.interruption_point_ms = interruption.ai_speech_position_ms

        await self._emit_event(PipelineEvent(
            type=PipelineEventType.INTERRUPTION_DETECTED,
            data={
                "position_ms": interruption.ai_speech_position_ms,
                "confidence": interruption.confidence,
            },
        ))

        await self._emit_event(PipelineEvent(
            type=PipelineEventType.INTERRUPTION_HANDLED,
            data={
                "action": interruption.action_taken,
            },
        ))

        logger.info(f"Interruption handled at {interruption.ai_speech_position_ms}ms")

        # Complete current turn
        if self._context:
            self._context.complete_current_turn()

        # Start new turn for interruption
        self._speech_buffer.clear()
        await self._set_state(VoicePipelineState.LISTENING)

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        metrics = {
            "state": self._state.value,
            "total_input_audio_ms": self._total_input_audio_ms,
            "total_output_audio_ms": self._total_output_audio_ms,
        }

        if self._context:
            metrics["conversation_id"] = self._context.conversation_id
            metrics["total_turns"] = len(self._context.turns)

            if self._context.turns:
                # Calculate average metrics
                latencies = [
                    t.total_latency_ms for t in self._context.turns
                    if t.total_latency_ms is not None
                ]
                if latencies:
                    metrics["avg_latency_ms"] = sum(latencies) / len(latencies)

                interruptions = sum(1 for t in self._context.turns if t.was_interrupted)
                metrics["interruption_count"] = interruptions

        return metrics


# Convenience functions for creating pipelines

def create_pipeline(
    response_generator: ResponseGenerator,
    stt_provider: str = "deepgram",
    tts_provider: str = "elevenlabs",
    **kwargs,
) -> VoicePipeline:
    """
    Create a voice pipeline with common defaults.

    Args:
        response_generator: The response generator to use
        stt_provider: STT provider name
        tts_provider: TTS provider name
        **kwargs: Additional config options

    Returns:
        Configured VoicePipeline instance
    """
    config = VoicePipelineConfig(
        stt_provider=stt_provider,
        tts_provider=tts_provider,
        **kwargs,
    )

    return VoicePipeline(config=config, response_generator=response_generator)


def create_test_pipeline(
    responses: Optional[List[str]] = None,
) -> VoicePipeline:
    """
    Create a test pipeline with simple response generator.

    Args:
        responses: Optional list of canned responses

    Returns:
        VoicePipeline configured for testing
    """
    generator = SimpleResponseGenerator(responses)
    config = VoicePipelineConfig(
        stt_provider="whisper",  # Good for testing
        tts_provider="openai",   # Good for testing
        stream_stt=False,        # Simpler for testing
        stream_tts=False,        # Simpler for testing
    )

    return VoicePipeline(config=config, response_generator=generator)


__all__ = [
    "VoicePipeline",
    "VoicePipelineConfig",
    "VoicePipelineState",
    "PipelineEvent",
    "PipelineEventType",
    "ConversationTurn",
    "ConversationContext",
    "ResponseGenerator",
    "SimpleResponseGenerator",
    "create_pipeline",
    "create_test_pipeline",
]
