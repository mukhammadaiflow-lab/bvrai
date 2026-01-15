"""
Streaming Orchestrator - Core Pipeline Implementation.

This module implements the main orchestration logic for the ultra-low
latency voice pipeline, coordinating ASR, LLM, and TTS streaming with
speculative execution and parallel processing.

Target: <300ms end-to-end latency from speech end to first audio byte.
"""

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple

from ..config import Settings, get_settings
from ..models import (
    AudioChunk,
    AudioBuffer,
    AudioFormat,
    EventType,
    LLMMessage,
    LLMStreamToken,
    SessionConfig,
    SessionMetrics,
    SessionState,
    StreamEvent,
    TranscriptResult,
    TranscriptSegment,
    TranscriptStatus,
    Turn,
    TTSChunk,
)
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerRegistry
from .latency import LatencyComponent, LatencyTracker, LatencyContext
from .speculative import SpeculativeConfig, SpeculativeExecutor, SpeculativeResult

logger = logging.getLogger(__name__)


class OrchestratorState(str, Enum):
    """State of the streaming orchestrator."""

    IDLE = "idle"
    STARTING = "starting"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    INTERRUPTING = "interrupting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    # Session
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    agent_id: str = ""

    # Audio formats
    input_format: AudioFormat = AudioFormat.PCM_16KHZ_16BIT
    input_sample_rate: int = 16000
    output_format: AudioFormat = AudioFormat.PCM_16KHZ_16BIT
    output_sample_rate: int = 16000

    # Provider selection
    asr_provider: str = "deepgram"
    llm_provider: str = "groq"
    llm_model: str = "llama-3.1-70b-versatile"
    tts_provider: str = "cartesia"
    tts_voice: str = ""

    # Behavior
    enable_speculative_execution: bool = True
    enable_interruption: bool = True
    enable_parallel_tts: bool = True  # Start TTS on first LLM tokens

    # Timing
    silence_timeout_ms: int = 700
    max_turn_duration_ms: int = 60000
    min_speech_duration_ms: int = 100

    # Context
    system_prompt: str = ""
    max_conversation_turns: int = 50

    # Buffer sizes
    input_buffer_size_ms: int = 100
    output_buffer_size_ms: int = 150


class StreamingOrchestrator:
    """
    Main orchestrator for ultra-low latency voice streaming.

    Architecture:
    ```
    Audio Input
         │
         ▼
    ┌────────────────┐
    │   VAD + ASR    │──────► Partial Transcripts
    │   (Streaming)  │              │
    └────────────────┘              │
         │                          ▼
         │              ┌────────────────────┐
         │              │ Speculative        │
         │              │ Execution Engine   │
         │              └────────────────────┘
         │                          │
         ▼                          ▼
    ┌────────────────┐    ┌────────────────┐
    │ Final          │───►│ LLM Streaming  │
    │ Transcript     │    │ (Parallel)     │
    └────────────────┘    └────────────────┘
                                   │
                                   ▼
                          ┌────────────────┐
                          │ TTS Streaming  │
                          │ (Parallel)     │
                          └────────────────┘
                                   │
                                   ▼
                            Audio Output
    ```

    Key Optimizations:
    1. Parallel ASR → LLM → TTS streaming
    2. Speculative LLM execution on stable partials
    3. Sentence-level TTS for faster first audio
    4. Adaptive buffering based on network conditions
    5. Circuit breakers for provider resilience
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize the streaming orchestrator.

        Args:
            config: Orchestrator configuration
            settings: Optional application settings
        """
        self.config = config
        self.settings = settings or get_settings()

        # State
        self._state = OrchestratorState.IDLE
        self._session_id = config.session_id

        # Conversation
        self._conversation_history: List[LLMMessage] = []
        self._current_turn: Optional[Turn] = None
        self._turns: List[Turn] = []

        # Metrics
        self._metrics = SessionMetrics()
        self._latency_tracker = LatencyTracker()
        self._latencies: List[float] = []  # E2E latencies for percentile calculation

        # Buffers
        self._input_buffer = AudioBuffer(max_duration_ms=config.input_buffer_size_ms)
        self._speech_buffer = AudioBuffer(max_duration_ms=config.max_turn_duration_ms)

        # Circuit breakers
        self._circuit_registry = CircuitBreakerRegistry()

        # Speculative execution
        self._speculative_executor: Optional[SpeculativeExecutor] = None
        if config.enable_speculative_execution:
            spec_config = SpeculativeConfig(
                enabled=True,
                min_stability_score=0.7,
                min_words=3,
                max_speculative_tokens=50,
            )
            self._speculative_executor = SpeculativeExecutor(
                spec_config,
                self._generate_llm_tokens,
            )

        # Event handling
        self._event_handlers: List[Callable[[StreamEvent], asyncio.Future]] = []
        self._event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

        # Control
        self._stop_event = asyncio.Event()
        self._tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

        # Output queue
        self._output_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()

        # Provider clients (to be initialized)
        self._asr_client = None
        self._llm_client = None
        self._tts_client = None

        logger.info(
            f"StreamingOrchestrator initialized: session={self._session_id}, "
            f"providers=({config.asr_provider}, {config.llm_provider}, {config.tts_provider})"
        )

    @property
    def state(self) -> OrchestratorState:
        """Get current state."""
        return self._state

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    @property
    def metrics(self) -> SessionMetrics:
        """Get session metrics."""
        return self._metrics

    def add_event_handler(
        self,
        handler: Callable[[StreamEvent], asyncio.Future],
    ) -> None:
        """Add an event handler."""
        self._event_handlers.append(handler)

    async def _emit_event(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit an event to all handlers."""
        event = StreamEvent(
            event_type=event_type,
            session_id=self._session_id,
            data=data or {},
            turn_id=self._current_turn.turn_id if self._current_turn else None,
        )

        # Add to queue for async processing
        await self._event_queue.put(event)

        # Also call handlers directly for real-time events
        for handler in self._event_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    async def _set_state(self, new_state: OrchestratorState) -> None:
        """Set orchestrator state."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            logger.debug(f"State change: {old_state.value} -> {new_state.value}")

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> None:
        """Start the orchestrator."""
        if self._state != OrchestratorState.IDLE:
            raise RuntimeError(f"Cannot start in state: {self._state}")

        await self._set_state(OrchestratorState.STARTING)

        try:
            # Initialize provider clients
            await self._initialize_providers()

            # Initialize circuit breakers
            await self._initialize_circuit_breakers()

            # Add system prompt to conversation
            if self.config.system_prompt:
                self._conversation_history.append(
                    LLMMessage(role="system", content=self.config.system_prompt)
                )

            # Start background tasks
            self._tasks.add(asyncio.create_task(self._event_processor()))

            await self._set_state(OrchestratorState.LISTENING)

            await self._emit_event(EventType.SESSION_STARTED, {
                "session_id": self._session_id,
                "config": {
                    "asr_provider": self.config.asr_provider,
                    "llm_provider": self.config.llm_provider,
                    "tts_provider": self.config.tts_provider,
                },
            })

            logger.info(f"Orchestrator started: session={self._session_id}")

        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}")
            await self._set_state(OrchestratorState.ERROR)
            raise

    async def stop(self) -> None:
        """Stop the orchestrator."""
        if self._state == OrchestratorState.STOPPED:
            return

        await self._set_state(OrchestratorState.STOPPING)

        # Signal stop
        self._stop_event.set()

        # Cancel speculative execution
        if self._speculative_executor:
            await self._speculative_executor.cancel()

        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Complete current turn
        if self._current_turn:
            await self._complete_turn()

        # Cleanup providers
        await self._cleanup_providers()

        await self._set_state(OrchestratorState.STOPPED)

        await self._emit_event(EventType.SESSION_ENDED, {
            "session_id": self._session_id,
            "total_turns": len(self._turns),
            "metrics": {
                "avg_e2e_latency_ms": self._metrics.avg_e2e_latency_ms,
                "total_audio_received_ms": self._metrics.total_audio_received_ms,
                "total_audio_sent_ms": self._metrics.total_audio_sent_ms,
            },
        })

        logger.info(f"Orchestrator stopped: session={self._session_id}")

    # =========================================================================
    # Provider Management
    # =========================================================================

    async def _initialize_providers(self) -> None:
        """Initialize all provider clients."""
        # ASR client
        # In production, instantiate actual provider clients
        logger.debug(f"Initializing ASR provider: {self.config.asr_provider}")

        # LLM client
        logger.debug(f"Initializing LLM provider: {self.config.llm_provider}")

        # TTS client
        logger.debug(f"Initializing TTS provider: {self.config.tts_provider}")

    async def _cleanup_providers(self) -> None:
        """Cleanup provider connections."""
        if self._asr_client:
            # Close ASR connection
            pass
        if self._llm_client:
            # Close LLM connection
            pass
        if self._tts_client:
            # Close TTS connection
            pass

    async def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for each provider."""
        # ASR circuit breaker
        await self._circuit_registry.get_or_create(
            name=f"asr_{self.config.asr_provider}",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_s=30.0,
                timeout_threshold_ms=5000,
            ),
        )

        # LLM circuit breaker
        await self._circuit_registry.get_or_create(
            name=f"llm_{self.config.llm_provider}",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_s=30.0,
                timeout_threshold_ms=10000,
            ),
        )

        # TTS circuit breaker
        await self._circuit_registry.get_or_create(
            name=f"tts_{self.config.tts_provider}",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_s=30.0,
                timeout_threshold_ms=5000,
            ),
        )

    # =========================================================================
    # Audio Processing
    # =========================================================================

    async def process_audio(self, audio_data: bytes, timestamp_ms: float = 0) -> None:
        """
        Process incoming audio data.

        This is the main entry point for audio from the client.

        Args:
            audio_data: Raw audio bytes
            timestamp_ms: Optional timestamp in milliseconds
        """
        if self._state not in (OrchestratorState.LISTENING, OrchestratorState.PROCESSING):
            return

        # Create audio chunk
        chunk = AudioChunk(
            data=audio_data,
            timestamp_ms=timestamp_ms or time.time() * 1000,
            duration_ms=len(audio_data) / (self.config.input_sample_rate * 2) * 1000,
            format=self.config.input_format,
            sample_rate=self.config.input_sample_rate,
        )

        # Track metrics
        self._metrics.total_audio_received_ms += chunk.duration_ms

        # Add to input buffer
        self._input_buffer.add(chunk)

        # Process through ASR
        await self._process_audio_chunk(chunk)

    async def _process_audio_chunk(self, chunk: AudioChunk) -> None:
        """Process a single audio chunk through the pipeline."""
        # In production, this would:
        # 1. Run VAD to detect speech
        # 2. Stream audio to ASR
        # 3. Handle transcripts as they arrive

        # Start new turn if needed
        if self._state == OrchestratorState.LISTENING:
            await self._start_turn()

        # Add to speech buffer
        self._speech_buffer.add(chunk)

        await self._emit_event(EventType.AUDIO_RECEIVED, {
            "duration_ms": chunk.duration_ms,
            "total_ms": self._speech_buffer.total_duration_ms,
        })

    async def get_output_audio(self) -> AsyncIterator[AudioChunk]:
        """
        Get output audio chunks.

        Yields:
            Audio chunks as they become available
        """
        while not self._stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(
                    self._output_queue.get(),
                    timeout=0.1,
                )
                yield chunk
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    # =========================================================================
    # Turn Management
    # =========================================================================

    async def _start_turn(self) -> None:
        """Start a new conversation turn."""
        async with self._lock:
            if self._current_turn:
                return  # Turn already in progress

            await self._set_state(OrchestratorState.PROCESSING)

            self._current_turn = Turn()
            self._speech_buffer.clear()

            await self._emit_event(EventType.TURN_START, {
                "turn_id": self._current_turn.turn_id,
            })

            logger.debug(f"Turn started: {self._current_turn.turn_id}")

    async def _complete_turn(self) -> None:
        """Complete the current turn."""
        if not self._current_turn:
            return

        self._current_turn.ended_at = time.time()

        # Calculate metrics
        if self._current_turn.e2e_latency_ms:
            self._latencies.append(self._current_turn.e2e_latency_ms)
            self._metrics.update_latency(
                self._current_turn.e2e_latency_ms,
                self._latencies,
            )
            await self._latency_tracker.record(
                LatencyComponent.E2E,
                self._current_turn.e2e_latency_ms,
                session_id=self._session_id,
                turn_id=self._current_turn.turn_id,
            )

        self._metrics.total_turns += 1

        # Archive turn
        self._turns.append(self._current_turn)

        # Trim old turns
        if len(self._turns) > self.config.max_conversation_turns:
            self._turns.pop(0)

        await self._emit_event(EventType.TURN_END, {
            "turn_id": self._current_turn.turn_id,
            "user_transcript": self._current_turn.user_transcript,
            "assistant_response": self._current_turn.assistant_response,
            "e2e_latency_ms": self._current_turn.e2e_latency_ms,
        })

        self._current_turn = None
        await self._set_state(OrchestratorState.LISTENING)

    # =========================================================================
    # Transcript Processing
    # =========================================================================

    async def process_transcript(self, segment: TranscriptSegment) -> None:
        """
        Process an incoming transcript segment.

        This handles both partial and final transcripts, triggering
        speculative execution when appropriate.

        Args:
            segment: The transcript segment from ASR
        """
        if not self._current_turn:
            await self._start_turn()

        # Update turn
        if self._current_turn:
            self._current_turn.transcript_segments.append(segment)

        # Emit event
        event_type = (
            EventType.TRANSCRIPT_FINAL
            if segment.status == TranscriptStatus.FINAL
            else EventType.TRANSCRIPT_PARTIAL
        )
        await self._emit_event(event_type, {
            "text": segment.text,
            "confidence": segment.confidence,
            "is_final": segment.status == TranscriptStatus.FINAL,
        })

        # Process through speculative executor
        if self._speculative_executor:
            result = await self._speculative_executor.process_transcript(
                segment,
                self._get_llm_context(),
            )

            if result and result.state.value == "validated":
                # Speculation was successful!
                self._metrics.speculations_validated += 1
                await self._emit_event(EventType.SPECULATION_VALIDATED, {
                    "tokens_reusable": result.tokens_reusable,
                    "latency_saved_ms": result.latency_saved_ms,
                })

        # Handle final transcript
        if segment.status == TranscriptStatus.FINAL:
            await self._handle_final_transcript(segment)

    async def _handle_final_transcript(self, segment: TranscriptSegment) -> None:
        """Handle a final transcript and generate response."""
        if not self._current_turn:
            return

        self._current_turn.user_transcript = segment.text
        self._current_turn.speech_end_at = time.time()

        # Add to conversation history
        self._conversation_history.append(
            LLMMessage(role="user", content=segment.text)
        )

        # Generate and speak response
        await self._generate_response()

    # =========================================================================
    # Response Generation
    # =========================================================================

    async def _generate_response(self) -> None:
        """Generate LLM response and stream to TTS."""
        if not self._current_turn:
            return

        await self._set_state(OrchestratorState.RESPONDING)

        # Check for reusable speculative tokens
        reusable_tokens: List[LLMStreamToken] = []
        if self._speculative_executor:
            reusable_tokens = await self._speculative_executor.get_reusable_tokens()
            if reusable_tokens:
                self._metrics.speculations_validated += 1
                logger.debug(f"Reusing {len(reusable_tokens)} speculative tokens")

        try:
            response_text = ""
            first_token_time: Optional[float] = None
            first_audio_time: Optional[float] = None

            # TTS streaming task
            tts_queue: asyncio.Queue[str] = asyncio.Queue()
            tts_task = asyncio.create_task(
                self._stream_tts(tts_queue)
            )
            self._tasks.add(tts_task)

            # Stream reusable tokens first
            for token in reusable_tokens:
                if first_token_time is None:
                    first_token_time = time.time()
                    self._current_turn.llm_first_token_at = first_token_time

                response_text += token.token
                await tts_queue.put(token.token)

                await self._emit_event(EventType.LLM_TOKEN, {
                    "token": token.token,
                    "is_reused": True,
                })

            # Generate remaining response
            context = self._get_llm_context()

            async with LatencyContext(
                self._latency_tracker,
                LatencyComponent.LLM_FIRST_TOKEN,
                session_id=self._session_id,
            ):
                async for token in self._generate_llm_tokens(
                    self._current_turn.user_transcript,
                    context,
                ):
                    if first_token_time is None:
                        first_token_time = time.time()
                        self._current_turn.llm_first_token_at = first_token_time

                    response_text += token.token
                    await tts_queue.put(token.token)

                    await self._emit_event(EventType.LLM_TOKEN, {
                        "token": token.token,
                        "is_reused": False,
                    })

            # Signal TTS completion
            await tts_queue.put("")  # Empty string signals end

            # Wait for TTS to complete
            await tts_task

            # Update turn
            self._current_turn.assistant_response = response_text
            self._current_turn.llm_complete_at = time.time()

            # Add to conversation history
            self._conversation_history.append(
                LLMMessage(role="assistant", content=response_text)
            )

            await self._emit_event(EventType.LLM_COMPLETE, {
                "text": response_text,
                "tokens": len(response_text.split()),
            })

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            await self._emit_event(EventType.LLM_ERROR, {"error": str(e)})
            self._metrics.llm_errors += 1

        finally:
            await self._complete_turn()

    async def _generate_llm_tokens(
        self,
        prompt: str,
        context: Dict[str, Any],
    ) -> AsyncIterator[LLMStreamToken]:
        """
        Generate LLM response tokens.

        This is the interface for the speculative executor and main
        response generation.

        Args:
            prompt: User prompt/transcript
            context: Additional context (system prompt, history, etc.)

        Yields:
            LLM tokens as they're generated
        """
        # In production, this would call the actual LLM provider
        # For now, simulate streaming response

        # Simulated response
        response = f"I understand you said: '{prompt}'. Let me help you with that."

        words = response.split()
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")

            yield LLMStreamToken(
                token=token,
                is_first=(i == 0),
                is_last=(i == len(words) - 1),
                cumulative_text=" ".join(words[:i + 1]),
            )

            # Simulate LLM latency
            await asyncio.sleep(0.02)  # ~50 tokens/sec

    def _get_llm_context(self) -> Dict[str, Any]:
        """Get context for LLM generation."""
        return {
            "system_prompt": self.config.system_prompt,
            "conversation_history": [
                {"role": msg.role, "content": msg.content}
                for msg in self._conversation_history[-10:]  # Last 10 messages
            ],
            "model": self.config.llm_model,
        }

    # =========================================================================
    # TTS Streaming
    # =========================================================================

    async def _stream_tts(self, text_queue: asyncio.Queue[str]) -> None:
        """
        Stream text to TTS and output audio.

        This implements sentence-level chunking for faster first audio.

        Args:
            text_queue: Queue of text chunks to synthesize
        """
        buffer = ""
        sentence_delimiters = {'.', '!', '?', '\n'}
        first_audio_sent = False

        try:
            while True:
                try:
                    text = await asyncio.wait_for(text_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if text == "":  # End signal
                    # Synthesize remaining buffer
                    if buffer.strip():
                        await self._synthesize_and_output(buffer.strip())
                    break

                buffer += text

                # Check for sentence boundaries
                for i, char in enumerate(buffer):
                    if char in sentence_delimiters:
                        sentence = buffer[:i + 1].strip()
                        if sentence:
                            async with LatencyContext(
                                self._latency_tracker,
                                LatencyComponent.TTS_FIRST_AUDIO,
                                session_id=self._session_id,
                            ) as ctx:
                                await self._synthesize_and_output(sentence)

                                if not first_audio_sent and self._current_turn:
                                    self._current_turn.tts_first_audio_at = time.time()
                                    first_audio_sent = True

                        buffer = buffer[i + 1:]
                        break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            self._metrics.tts_errors += 1

    async def _synthesize_and_output(self, text: str) -> None:
        """
        Synthesize text and add to output queue.

        Args:
            text: Text to synthesize
        """
        # In production, call actual TTS provider
        # For now, simulate audio generation

        await self._emit_event(EventType.TTS_START, {"text": text})

        # Simulate TTS audio (20ms chunks)
        chunk_duration_ms = 20
        total_duration_ms = len(text) * 50  # ~50ms per character

        for offset in range(0, int(total_duration_ms), chunk_duration_ms):
            # Create dummy audio chunk
            samples_per_chunk = int(self.config.output_sample_rate * chunk_duration_ms / 1000)
            audio_data = bytes(samples_per_chunk * 2)  # 16-bit silence

            chunk = AudioChunk(
                data=audio_data,
                timestamp_ms=time.time() * 1000,
                duration_ms=chunk_duration_ms,
                format=self.config.output_format,
                sample_rate=self.config.output_sample_rate,
            )

            await self._output_queue.put(chunk)

            if self._current_turn:
                self._current_turn.assistant_audio_duration_ms += chunk_duration_ms

            self._metrics.total_audio_sent_ms += chunk_duration_ms

            await self._emit_event(EventType.TTS_AUDIO, {
                "duration_ms": chunk_duration_ms,
            })

            # Small delay to simulate real streaming
            await asyncio.sleep(0.01)

        await self._emit_event(EventType.TTS_COMPLETE, {
            "text": text,
            "duration_ms": total_duration_ms,
        })

    # =========================================================================
    # Interruption Handling
    # =========================================================================

    async def handle_interruption(self) -> None:
        """Handle user interruption during AI response."""
        if not self.config.enable_interruption:
            return

        if self._state != OrchestratorState.RESPONDING:
            return

        logger.debug("Handling interruption")

        await self._set_state(OrchestratorState.INTERRUPTING)

        # Update metrics
        self._metrics.interruptions_detected += 1

        # Mark turn as interrupted
        if self._current_turn:
            self._current_turn.was_interrupted = True
            self._current_turn.interruption_at_ms = time.time() * 1000

        # Clear output queue
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Cancel speculative execution
        if self._speculative_executor:
            await self._speculative_executor.cancel()

        await self._emit_event(EventType.INTERRUPTION_HANDLED, {
            "turn_id": self._current_turn.turn_id if self._current_turn else None,
        })

        # Complete turn and return to listening
        await self._complete_turn()

    # =========================================================================
    # Background Tasks
    # =========================================================================

    async def _event_processor(self) -> None:
        """Background task for processing events."""
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.1,
                )
                # Events are already emitted to handlers in _emit_event
                # This task could be used for logging, persistence, etc.
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        return {
            "session_id": self._session_id,
            "state": self._state.value,
            "metrics": {
                "total_turns": self._metrics.total_turns,
                "total_audio_received_ms": round(self._metrics.total_audio_received_ms, 2),
                "total_audio_sent_ms": round(self._metrics.total_audio_sent_ms, 2),
                "avg_e2e_latency_ms": round(self._metrics.avg_e2e_latency_ms, 2),
                "p50_e2e_latency_ms": round(self._metrics.p50_e2e_latency_ms, 2),
                "p95_e2e_latency_ms": round(self._metrics.p95_e2e_latency_ms, 2),
                "p99_e2e_latency_ms": round(self._metrics.p99_e2e_latency_ms, 2),
            },
            "speculative_execution": (
                self._speculative_executor.get_statistics()
                if self._speculative_executor
                else None
            ),
            "errors": {
                "asr_errors": self._metrics.asr_errors,
                "llm_errors": self._metrics.llm_errors,
                "tts_errors": self._metrics.tts_errors,
            },
            "interruptions": {
                "detected": self._metrics.interruptions_detected,
                "handled": self._metrics.interruptions_handled,
            },
        }


# Factory function
def create_orchestrator(
    tenant_id: str,
    agent_id: str,
    system_prompt: str = "",
    asr_provider: str = "deepgram",
    llm_provider: str = "groq",
    llm_model: str = "llama-3.1-70b-versatile",
    tts_provider: str = "cartesia",
    enable_speculation: bool = True,
) -> StreamingOrchestrator:
    """
    Create a configured streaming orchestrator.

    Args:
        tenant_id: Tenant identifier
        agent_id: Agent identifier
        system_prompt: System prompt for the agent
        asr_provider: ASR provider name
        llm_provider: LLM provider name
        llm_model: LLM model name
        tts_provider: TTS provider name
        enable_speculation: Enable speculative execution

    Returns:
        Configured StreamingOrchestrator instance
    """
    config = OrchestratorConfig(
        tenant_id=tenant_id,
        agent_id=agent_id,
        system_prompt=system_prompt,
        asr_provider=asr_provider,
        llm_provider=llm_provider,
        llm_model=llm_model,
        tts_provider=tts_provider,
        enable_speculative_execution=enable_speculation,
    )

    return StreamingOrchestrator(config)
