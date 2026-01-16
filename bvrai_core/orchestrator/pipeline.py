"""
Call Processing Pipeline Module

This module provides the real-time processing pipeline for voice calls,
coordinating STT, LLM, and TTS components.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
)

from .base import (
    CallSession,
    CallState,
    PipelineContext,
    ParticipantRole,
    ConversationTurn,
    EventType,
)


logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name."""
        pass

    @abstractmethod
    async def process(self, context: PipelineContext) -> PipelineContext:
        """Process the context through this stage."""
        pass


class SpeechToTextStage(PipelineStage):
    """
    Speech-to-text processing stage.

    Converts incoming audio to text transcription.
    """

    @property
    def name(self) -> str:
        return "stt"

    def __init__(
        self,
        stt_provider: Optional[Any] = None,
    ):
        """
        Initialize STT stage.

        Args:
            stt_provider: STT provider instance
        """
        self.stt_provider = stt_provider
        self._buffer = bytearray()
        self._last_speech_time = 0.0

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Process audio chunk through STT."""
        start_time = time.time()

        if not context.audio_chunk:
            return context

        # In production, this would stream to STT provider
        # For now, simulate processing
        # context.partial_transcript = await self.stt_provider.transcribe(context.audio_chunk)

        # Detect speech activity
        context.speech_detected = self._detect_speech(context.audio_chunk)

        if context.speech_detected:
            self._last_speech_time = time.time()
            context.silence_detected = False
        else:
            # Check for silence timeout
            silence_duration = time.time() - self._last_speech_time
            if silence_duration > 1.0:  # 1 second silence threshold
                context.silence_detected = True

        # Calculate latency
        context.stt_latency_ms = (time.time() - start_time) * 1000

        return context

    def _detect_speech(self, audio: bytes) -> bool:
        """Simple speech detection based on audio energy."""
        if len(audio) < 2:
            return False

        # Calculate RMS energy
        samples = [
            int.from_bytes(audio[i:i+2], byteorder='little', signed=True)
            for i in range(0, len(audio) - 1, 2)
        ]

        if not samples:
            return False

        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5

        # Threshold for speech detection
        return rms > 500


class InterruptionDetectorStage(PipelineStage):
    """
    Detects customer interruptions during agent speech.
    """

    @property
    def name(self) -> str:
        return "interruption_detector"

    def __init__(
        self,
        sensitivity: float = 0.5,
        min_speech_duration_ms: int = 300,
    ):
        """
        Initialize interruption detector.

        Args:
            sensitivity: Interruption sensitivity (0-1)
            min_speech_duration_ms: Min speech duration to trigger interruption
        """
        self.sensitivity = sensitivity
        self.min_speech_duration_ms = min_speech_duration_ms
        self._speech_start_time: Optional[float] = None

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Detect interruptions."""
        session = context.session

        # Only check for interruptions when agent is speaking
        if session.state != CallState.AGENT_SPEAKING:
            self._speech_start_time = None
            return context

        # Check if customer started speaking
        if context.speech_detected:
            if self._speech_start_time is None:
                self._speech_start_time = time.time()
            else:
                # Check if speech duration exceeds threshold
                speech_duration_ms = (time.time() - self._speech_start_time) * 1000
                threshold = self.min_speech_duration_ms * (1 - self.sensitivity)

                if speech_duration_ms > threshold:
                    context.interruption_detected = True
                    logger.debug(
                        f"Interruption detected after {speech_duration_ms:.0f}ms "
                        f"(threshold: {threshold:.0f}ms)"
                    )
        else:
            self._speech_start_time = None

        return context


class TurnDetectorStage(PipelineStage):
    """
    Detects end of customer turn based on silence and context.
    """

    @property
    def name(self) -> str:
        return "turn_detector"

    def __init__(
        self,
        silence_threshold_ms: int = 700,
        max_turn_duration_ms: int = 30000,
    ):
        """
        Initialize turn detector.

        Args:
            silence_threshold_ms: Silence duration to end turn
            max_turn_duration_ms: Maximum turn duration
        """
        self.silence_threshold_ms = silence_threshold_ms
        self.max_turn_duration_ms = max_turn_duration_ms
        self._silence_start: Optional[float] = None
        self._turn_start: Optional[float] = None

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Detect end of turn."""
        session = context.session

        # Only detect turns when customer is speaking
        if session.state != CallState.CUSTOMER_SPEAKING:
            self._silence_start = None
            self._turn_start = None
            return context

        # Initialize turn start
        if self._turn_start is None:
            self._turn_start = time.time()

        # Track silence
        if context.silence_detected:
            if self._silence_start is None:
                self._silence_start = time.time()
            else:
                silence_duration_ms = (time.time() - self._silence_start) * 1000

                if silence_duration_ms >= self.silence_threshold_ms:
                    context.end_of_turn = True
                    logger.debug(
                        f"End of turn detected after {silence_duration_ms:.0f}ms silence"
                    )
        else:
            self._silence_start = None

        # Check max turn duration
        if self._turn_start:
            turn_duration_ms = (time.time() - self._turn_start) * 1000
            if turn_duration_ms >= self.max_turn_duration_ms:
                context.end_of_turn = True
                logger.debug("End of turn due to max duration")

        return context


class LLMProcessingStage(PipelineStage):
    """
    LLM processing stage for generating agent responses.
    """

    @property
    def name(self) -> str:
        return "llm"

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
    ):
        """
        Initialize LLM stage.

        Args:
            llm_provider: LLM provider instance
        """
        self.llm_provider = llm_provider

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Generate agent response using LLM."""
        start_time = time.time()

        # Only process when we have a complete customer turn
        if not context.end_of_turn or not context.final_transcript:
            return context

        session = context.session

        # Build messages for LLM
        messages = session.get_conversation_context()

        # Add current customer message
        messages.append({
            "role": "user",
            "content": context.final_transcript,
        })

        # Add function definitions if available
        tools = None
        if session.config.functions:
            tools = [
                {
                    "type": "function",
                    "function": func,
                }
                for func in session.config.functions
            ]

        # In production, this would call the LLM
        # response = await self.llm_provider.generate(
        #     messages=messages,
        #     tools=tools,
        #     temperature=session.config.temperature,
        #     max_tokens=session.config.max_tokens,
        # )
        # context.agent_response = response.content
        # context.pending_function_calls = response.function_calls

        # Simulate response for now
        context.agent_response = "This is a placeholder response from the LLM."

        # Calculate latency
        context.llm_latency_ms = (time.time() - start_time) * 1000

        return context


class FunctionExecutorStage(PipelineStage):
    """
    Executes function calls requested by the agent.
    """

    @property
    def name(self) -> str:
        return "function_executor"

    def __init__(
        self,
        function_handlers: Optional[Dict[str, Callable]] = None,
        default_timeout_ms: int = 10000,
    ):
        """
        Initialize function executor.

        Args:
            function_handlers: Map of function names to handlers
            default_timeout_ms: Default timeout for function calls
        """
        self.function_handlers = function_handlers or {}
        self.default_timeout_ms = default_timeout_ms

    def register_handler(
        self,
        name: str,
        handler: Callable[[Dict[str, Any]], Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a function handler."""
        self.function_handlers[name] = handler

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Execute pending function calls."""
        if not context.pending_function_calls:
            return context

        session = context.session

        for call in context.pending_function_calls:
            func_name = call.get("name", "")
            func_args = call.get("arguments", {})
            call_id = call.get("id", "")

            # Log function call event
            session.add_event(EventType.FUNCTION_CALLED, {
                "function": func_name,
                "arguments": func_args,
                "call_id": call_id,
            })

            handler = self.function_handlers.get(func_name)
            if not handler:
                logger.warning(f"No handler for function: {func_name}")
                context.function_results.append({
                    "call_id": call_id,
                    "success": False,
                    "error": f"Unknown function: {func_name}",
                })
                continue

            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    handler(func_args),
                    timeout=self.default_timeout_ms / 1000,
                )

                context.function_results.append({
                    "call_id": call_id,
                    "success": True,
                    "result": result,
                })

                session.add_event(EventType.FUNCTION_COMPLETED, {
                    "function": func_name,
                    "call_id": call_id,
                    "result": result,
                })

            except asyncio.TimeoutError:
                logger.error(f"Function {func_name} timed out")
                context.function_results.append({
                    "call_id": call_id,
                    "success": False,
                    "error": "Function timed out",
                })

                session.add_event(EventType.FUNCTION_FAILED, {
                    "function": func_name,
                    "call_id": call_id,
                    "error": "Timeout",
                })

            except Exception as e:
                logger.exception(f"Function {func_name} failed: {e}")
                context.function_results.append({
                    "call_id": call_id,
                    "success": False,
                    "error": str(e),
                })

                session.add_event(EventType.FUNCTION_FAILED, {
                    "function": func_name,
                    "call_id": call_id,
                    "error": str(e),
                })

        return context


class TextToSpeechStage(PipelineStage):
    """
    Text-to-speech processing stage.

    Converts agent text response to audio.
    """

    @property
    def name(self) -> str:
        return "tts"

    def __init__(
        self,
        tts_provider: Optional[Any] = None,
    ):
        """
        Initialize TTS stage.

        Args:
            tts_provider: TTS provider instance
        """
        self.tts_provider = tts_provider

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Convert agent response to audio."""
        start_time = time.time()

        if not context.agent_response:
            return context

        # In production, this would stream from TTS provider
        # async for audio_chunk in self.tts_provider.synthesize_stream(
        #     text=context.agent_response,
        #     voice_id=context.session.config.voice_id,
        #     speed=context.session.config.voice_speed,
        # ):
        #     yield audio_chunk

        # Simulate TTS output
        context.agent_audio = b"\x00" * 1600  # Placeholder audio

        # Calculate latency
        context.tts_latency_ms = (time.time() - start_time) * 1000

        return context


class CallPipeline:
    """
    Real-time call processing pipeline.

    Coordinates all stages of call processing:
    1. Speech-to-text
    2. Interruption detection
    3. Turn detection
    4. LLM processing
    5. Function execution
    6. Text-to-speech
    """

    def __init__(self):
        """Initialize the pipeline."""
        self.stages: List[PipelineStage] = []
        self._stage_metrics: Dict[str, List[float]] = {}

    def add_stage(self, stage: PipelineStage) -> "CallPipeline":
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        self._stage_metrics[stage.name] = []
        return self

    def create_default_stages(
        self,
        stt_provider: Optional[Any] = None,
        llm_provider: Optional[Any] = None,
        tts_provider: Optional[Any] = None,
        function_handlers: Optional[Dict[str, Callable]] = None,
    ) -> "CallPipeline":
        """Create default pipeline stages."""
        self.add_stage(SpeechToTextStage(stt_provider))
        self.add_stage(InterruptionDetectorStage())
        self.add_stage(TurnDetectorStage())
        self.add_stage(LLMProcessingStage(llm_provider))
        self.add_stage(FunctionExecutorStage(function_handlers))
        self.add_stage(TextToSpeechStage(tts_provider))
        return self

    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Process context through all pipeline stages.

        Args:
            context: Pipeline context

        Returns:
            Processed context
        """
        context.pipeline_start_time = time.time()

        for stage in self.stages:
            stage_start = time.time()

            try:
                context = await stage.process(context)

                # Track stage latency
                stage_latency = (time.time() - stage_start) * 1000
                self._stage_metrics[stage.name].append(stage_latency)

                # Keep only last 100 measurements
                if len(self._stage_metrics[stage.name]) > 100:
                    self._stage_metrics[stage.name].pop(0)

            except Exception as e:
                logger.exception(f"Pipeline stage {stage.name} failed: {e}")
                # Continue to next stage or break based on error type

        return context

    async def process_stream(
        self,
        session: CallSession,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[Tuple[PipelineContext, Optional[bytes]]]:
        """
        Process audio stream through pipeline.

        Args:
            session: Call session
            audio_stream: Async iterator of audio chunks

        Yields:
            Tuple of (context, response_audio)
        """
        async for audio_chunk in audio_stream:
            context = PipelineContext(
                session=session,
                audio_chunk=audio_chunk,
            )

            context = await self.process(context)

            yield context, context.agent_audio

    def get_stage_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get average latency metrics for each stage."""
        metrics = {}

        for stage_name, latencies in self._stage_metrics.items():
            if latencies:
                metrics[stage_name] = {
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies),
                    "samples": len(latencies),
                }
            else:
                metrics[stage_name] = {
                    "avg_latency_ms": 0,
                    "min_latency_ms": 0,
                    "max_latency_ms": 0,
                    "samples": 0,
                }

        return metrics


def create_call_pipeline(
    stt_provider: Optional[Any] = None,
    llm_provider: Optional[Any] = None,
    tts_provider: Optional[Any] = None,
    function_handlers: Optional[Dict[str, Callable]] = None,
) -> CallPipeline:
    """
    Create a configured call processing pipeline.

    Args:
        stt_provider: Speech-to-text provider
        llm_provider: LLM provider
        tts_provider: Text-to-speech provider
        function_handlers: Function handlers

    Returns:
        Configured pipeline
    """
    pipeline = CallPipeline()
    pipeline.create_default_stages(
        stt_provider=stt_provider,
        llm_provider=llm_provider,
        tts_provider=tts_provider,
        function_handlers=function_handlers,
    )
    return pipeline


__all__ = [
    "PipelineStage",
    "SpeechToTextStage",
    "InterruptionDetectorStage",
    "TurnDetectorStage",
    "LLMProcessingStage",
    "FunctionExecutorStage",
    "TextToSpeechStage",
    "CallPipeline",
    "create_call_pipeline",
]
