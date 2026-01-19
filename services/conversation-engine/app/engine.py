"""Main Conversation Engine orchestrator."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import httpx
import structlog

from app.config import get_settings
from app.state.machine import (
    ConversationStateMachine,
    ConversationContext,
    ConversationState,
    ConversationEvent,
)
from app.turn.detector import TurnDetector, TurnConfig, TurnInfo
from app.interrupt.handler import (
    InterruptHandler,
    InterruptConfig,
    InterruptEvent,
    InterruptType,
)

logger = structlog.get_logger()


@dataclass
class EngineConfig:
    """Configuration for conversation engine."""
    session_id: str
    agent_id: str
    greeting: str = ""
    voice_id: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: list[dict] = field(default_factory=list)


class ConversationEngine:
    """
    Main orchestrator for voice conversations.

    Coordinates:
    - State machine for conversation flow
    - Turn detection for knowing when user finishes
    - Interrupt handling for barge-in
    - ASR/TTS/LLM service communication
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        settings = get_settings()

        # Create context
        self.context = ConversationContext(
            session_id=config.session_id,
            agent_id=config.agent_id,
        )

        # Create state machine
        self.state_machine = ConversationStateMachine(self.context)

        # Create turn detector
        turn_config = TurnConfig(
            silence_threshold_ms=settings.silence_threshold_ms,
            min_speech_ms=settings.min_speech_ms,
            no_speech_timeout_ms=settings.no_speech_timeout_ms,
        )
        self.turn_detector = TurnDetector(
            config=turn_config,
            on_turn_complete=self._on_turn_complete,
        )

        # Create interrupt handler
        interrupt_config = InterruptConfig(
            enabled=settings.interrupt_enabled,
            min_speech_ms=settings.interrupt_min_speech_ms,
            overlap_tolerance_ms=settings.interrupt_overlap_tolerance_ms,
        )
        self.interrupt_handler = InterruptHandler(
            config=interrupt_config,
            on_interrupt=self._on_interrupt,
        )

        # HTTP client for service communication
        self._http_client: Optional[httpx.AsyncClient] = None

        # Callbacks
        self._on_speak: Optional[Callable[[str], asyncio.Future]] = None
        self._on_clear_audio: Optional[Callable[[], asyncio.Future]] = None
        self._on_call_ended: Optional[Callable[[dict], None]] = None

        self.settings = settings
        self.logger = logger.bind(
            session_id=config.session_id,
            agent_id=config.agent_id,
        )

    async def start(self) -> None:
        """Start the conversation engine."""
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Start turn detector
        await self.turn_detector.start()

        # Register state handlers
        self._register_state_handlers()

        self.logger.info("Conversation engine started")

    async def stop(self) -> None:
        """Stop the conversation engine."""
        await self.turn_detector.stop()

        if self._http_client:
            await self._http_client.aclose()

        self.logger.info("Conversation engine stopped")

    def _register_state_handlers(self) -> None:
        """Register handlers for each state."""
        self.state_machine.register_handler(
            ConversationState.GREETING,
            self._handle_greeting_state,
        )
        self.state_machine.register_handler(
            ConversationState.LISTENING,
            self._handle_listening_state,
        )
        self.state_machine.register_handler(
            ConversationState.PROCESSING,
            self._handle_processing_state,
        )
        self.state_machine.register_handler(
            ConversationState.SPEAKING,
            self._handle_speaking_state,
        )
        self.state_machine.register_handler(
            ConversationState.ENDING,
            self._handle_ending_state,
        )

    # Public API

    def on_speak(self, callback: Callable[[str], asyncio.Future]) -> None:
        """Set callback for when engine wants to speak text."""
        self._on_speak = callback

    def on_clear_audio(self, callback: Callable[[], asyncio.Future]) -> None:
        """Set callback for clearing audio buffer."""
        self._on_clear_audio = callback

    def on_call_ended(self, callback: Callable[[dict], None]) -> None:
        """Set callback for when call ends."""
        self._on_call_ended = callback

    async def handle_call_start(self) -> None:
        """Handle incoming call start."""
        self.logger.info("Call started")
        await self.state_machine.handle_event(ConversationEvent.CALL_STARTED)

    async def handle_call_end(self, reason: str = "completed") -> None:
        """Handle call end."""
        self.logger.info("Call ended", reason=reason)
        await self.state_machine.handle_event(
            ConversationEvent.CALL_ENDED,
            {"reason": reason},
        )

        if self._on_call_ended:
            self._on_call_ended({
                "session_id": self.config.session_id,
                "reason": reason,
                "duration_sec": time.time() - self.context.state_history[0].timestamp,
                "turn_count": self.context.turn_count,
                "messages": self.context.messages,
            })

    async def handle_speech_start(self) -> None:
        """Handle VAD speech start event."""
        self.turn_detector.on_speech_start()

        # Check for interrupt
        if self.state_machine.is_agent_turn():
            event = self.interrupt_handler.on_user_speech_start()
            if event:
                await self._handle_interrupt_event(event)

    async def handle_speech_end(self) -> None:
        """Handle VAD speech end event."""
        self.turn_detector.on_speech_end()
        self.interrupt_handler.on_user_speech_end()

    async def handle_transcript(
        self,
        text: str,
        is_final: bool = False,
        speech_final: bool = False,
    ) -> None:
        """Handle transcript from ASR."""
        self.logger.debug(
            "Transcript received",
            text=text[:100],
            is_final=is_final,
            speech_final=speech_final,
        )

        # Check for interrupt if agent is speaking
        if self.state_machine.is_agent_turn():
            event = self.interrupt_handler.on_user_transcript(text, is_final)
            if event:
                await self._handle_interrupt_event(event)
                return

        # Update turn detector
        turn_info = self.turn_detector.on_transcript(text, is_final, speech_final)

        # Update context
        self.context.current_transcript = text

        if turn_info and turn_info.state.value == "turn_complete":
            # Turn is complete, process it
            await self._process_user_turn(text)

    async def handle_tts_complete(self) -> None:
        """Handle TTS playback complete."""
        self.interrupt_handler.agent_stopped_speaking()
        await self.state_machine.handle_event(ConversationEvent.TTS_COMPLETED)

    # State handlers

    async def _handle_greeting_state(self, context: ConversationContext, transition) -> None:
        """Handle entering greeting state."""
        greeting = self.config.greeting or self.settings.default_greeting

        self.logger.info("Playing greeting", greeting=greeting[:50])

        if self._on_speak:
            await self._on_speak(greeting)

        self.interrupt_handler.agent_started_speaking()
        await self.state_machine.handle_event(ConversationEvent.GREETING_STARTED)

    async def _handle_listening_state(self, context: ConversationContext, transition) -> None:
        """Handle entering listening state."""
        self.turn_detector.reset()
        context.turn_start_time = time.time()
        self.logger.debug("Now listening for user speech")

    async def _handle_processing_state(self, context: ConversationContext, transition) -> None:
        """Handle entering processing state."""
        # Get LLM response
        transcript = context.current_transcript

        if not transcript.strip():
            # No transcript, send timeout message
            response = self.settings.timeout_message
        else:
            # Get response from AI orchestrator
            response = await self._get_llm_response(transcript)

        if response:
            context.current_response = response
            await self.state_machine.handle_event(ConversationEvent.LLM_RESPONSE_STARTED)

    async def _handle_speaking_state(self, context: ConversationContext, transition) -> None:
        """Handle entering speaking state."""
        response = context.current_response

        if response and self._on_speak:
            self.interrupt_handler.agent_started_speaking()
            await self._on_speak(response)

    async def _handle_ending_state(self, context: ConversationContext, transition) -> None:
        """Handle entering ending state."""
        goodbye = self.settings.goodbye_message

        if self._on_speak:
            await self._on_speak(goodbye)

    # Internal methods

    def _on_turn_complete(self, turn_info: TurnInfo) -> None:
        """Callback when turn detector determines turn is complete."""
        asyncio.create_task(self._process_user_turn(turn_info.transcript))

    async def _process_user_turn(self, transcript: str) -> None:
        """Process a completed user turn."""
        if not transcript.strip():
            return

        self.logger.info("Processing user turn", transcript=transcript[:100])

        # Add to conversation history
        self.state_machine.add_user_message(transcript)

        # Trigger processing state
        await self.state_machine.handle_event(ConversationEvent.TRANSCRIPT_FINAL)

    def _on_interrupt(self, event: InterruptEvent) -> None:
        """Callback when interrupt is detected."""
        asyncio.create_task(self._handle_interrupt_event(event))

    async def _handle_interrupt_event(self, event: InterruptEvent) -> None:
        """Handle an interrupt event."""
        self.logger.info(
            "Handling interrupt",
            type=event.type.value,
            transcript=event.transcript[:50],
        )

        if event.type == InterruptType.HARD:
            # Stop speaking
            if self._on_clear_audio:
                await self._on_clear_audio()

            self.interrupt_handler.agent_stopped_speaking()

            # Transition to listening
            await self.state_machine.handle_event(ConversationEvent.BARGE_IN)

    async def _get_llm_response(self, transcript: str) -> str:
        """Get response from AI orchestrator."""
        try:
            # Build conversation history
            messages = self.state_machine.get_conversation_history()

            payload = {
                "session_id": self.config.session_id,
                "agent_id": self.config.agent_id,
                "transcript": transcript,
                "messages": messages,
                "system_prompt": self.config.system_prompt,
                "tools": self.config.tools,
            }

            response = await self._http_client.post(
                f"{self.settings.ai_orchestrator_url}/generate",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            text = data.get("text", "")

            # Add to history
            if text:
                self.state_machine.add_agent_message(text)

            return text

        except Exception as e:
            self.logger.error("Failed to get LLM response", error=str(e))
            return self.settings.error_message

    def get_state(self) -> str:
        """Get current conversation state."""
        return self.context.state.value

    def get_context(self) -> dict:
        """Get conversation context as dict."""
        return {
            "session_id": self.context.session_id,
            "agent_id": self.context.agent_id,
            "state": self.context.state.value,
            "turn_count": self.context.turn_count,
            "messages": self.context.messages,
            "entities": self.context.entities,
        }
