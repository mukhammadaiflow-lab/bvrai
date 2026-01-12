"""Media bridge between WebRTC and Conversation Engine."""

import asyncio
from typing import Optional, Callable, Awaitable

import httpx
import structlog

from app.config import get_settings
from app.sessions.session import Session
from app.media.processor import AudioProcessor

logger = structlog.get_logger()
settings = get_settings()


class MediaBridge:
    """Bridges media between WebRTC session and Conversation Engine.

    Handles:
    - Audio streaming to/from conversation engine
    - Transcript forwarding
    - Agent response handling
    """

    def __init__(self, session: Session):
        self.session = session
        self.logger = logger.bind(
            session_id=session.session_id,
            component="media_bridge",
        )

        # Audio processor
        self.audio_processor = AudioProcessor(session.session_id)

        # HTTP client for conversation engine
        self._http_client: Optional[httpx.AsyncClient] = None

        # WebSocket to conversation engine
        self._engine_ws = None

        # Processing tasks
        self._input_task: Optional[asyncio.Task] = None
        self._output_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_transcript: Optional[Callable[[str, str, bool], Awaitable[None]]] = None
        self._on_agent_audio: Optional[Callable[[bytes], Awaitable[None]]] = None

    async def start(self) -> None:
        """Start the media bridge."""
        self._http_client = httpx.AsyncClient(timeout=30.0)

        await self.audio_processor.start()

        # Connect to conversation engine
        await self._connect_engine()

        self.logger.info("Media bridge started")

    async def stop(self) -> None:
        """Stop the media bridge."""
        # Cancel tasks
        if self._input_task:
            self._input_task.cancel()
        if self._output_task:
            self._output_task.cancel()

        await self.audio_processor.stop()

        if self._http_client:
            await self._http_client.aclose()

        self.logger.info("Media bridge stopped")

    async def handle_user_audio(self, audio_data: bytes) -> None:
        """Handle incoming audio from user."""
        processed = await self.audio_processor.process_input(audio_data)
        if processed:
            await self._send_to_engine(processed)

    async def handle_agent_response(self, text: str) -> None:
        """Handle agent text response.

        This would:
        1. Send text to TTS service
        2. Stream audio back to client
        """
        try:
            # Request TTS synthesis
            response = await self._http_client.post(
                f"{settings.conversation_engine_url}/tts/synthesize",
                json={
                    "text": text,
                    "session_id": self.session.session_id,
                },
            )

            if response.status_code == 200:
                audio_data = response.content
                processed = await self.audio_processor.process_output(audio_data)
                if processed and self._on_agent_audio:
                    await self._on_agent_audio(processed)

        except Exception as e:
            self.logger.error("TTS synthesis failed", error=str(e))

    def on_transcript(
        self,
        callback: Callable[[str, str, bool], Awaitable[None]],
    ) -> None:
        """Set callback for transcript updates.

        callback(speaker: str, text: str, is_final: bool)
        """
        self._on_transcript = callback

    def on_agent_audio(
        self,
        callback: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Set callback for agent audio."""
        self._on_agent_audio = callback

    async def _connect_engine(self) -> None:
        """Connect to conversation engine via WebSocket."""
        try:
            # Get agent config from session
            agent_config = self.session.metadata.get("agent_config", {})

            # Initialize session with conversation engine
            response = await self._http_client.post(
                f"{settings.conversation_engine_url}/session/init",
                json={
                    "session_id": self.session.session_id,
                    "agent_id": str(self.session.agent_id),
                    "agent_config": agent_config,
                },
            )

            if response.status_code == 200:
                self.logger.info("Connected to conversation engine")
            else:
                self.logger.warning(
                    "Failed to connect to engine",
                    status=response.status_code,
                )

        except Exception as e:
            self.logger.error("Engine connection error", error=str(e))

    async def _send_to_engine(self, audio_data: bytes) -> None:
        """Send audio to conversation engine."""
        try:
            # In production, this would stream audio via WebSocket
            # For now, we use HTTP for simplicity

            response = await self._http_client.post(
                f"{settings.conversation_engine_url}/audio/stream",
                content=audio_data,
                headers={
                    "Content-Type": "audio/pcm",
                    "X-Session-Id": self.session.session_id,
                },
            )

            # Handle any transcript in response
            if response.status_code == 200:
                data = response.json()
                if "transcript" in data and self._on_transcript:
                    await self._on_transcript(
                        data.get("speaker", "user"),
                        data["transcript"],
                        data.get("is_final", False),
                    )

        except Exception as e:
            self.logger.debug("Send to engine failed", error=str(e))

    async def send_greeting(self) -> None:
        """Send agent greeting message."""
        agent_config = self.session.metadata.get("agent_config", {})
        greeting = agent_config.get("greeting_message")

        if greeting:
            await self.handle_agent_response(greeting)

            if self._on_transcript:
                await self._on_transcript("agent", greeting, True)


class ConversationEngineClient:
    """HTTP client for Conversation Engine."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.logger = logger.bind(component="engine_client")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def init_session(
        self,
        session_id: str,
        agent_id: str,
        config: dict,
    ) -> bool:
        """Initialize a conversation session."""
        try:
            response = await self.client.post(
                f"{self.base_url}/session/init",
                json={
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "config": config,
                },
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error("Session init failed", error=str(e))
            return False

    async def end_session(self, session_id: str) -> bool:
        """End a conversation session."""
        try:
            response = await self.client.post(
                f"{self.base_url}/session/end",
                json={"session_id": session_id},
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error("Session end failed", error=str(e))
            return False

    async def send_audio(
        self,
        session_id: str,
        audio_data: bytes,
    ) -> Optional[dict]:
        """Send audio to conversation engine."""
        try:
            response = await self.client.post(
                f"{self.base_url}/audio/process",
                content=audio_data,
                headers={
                    "Content-Type": "audio/pcm",
                    "X-Session-Id": session_id,
                },
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            self.logger.debug("Send audio failed", error=str(e))
            return None

    async def get_response(
        self,
        session_id: str,
        transcript: str,
    ) -> Optional[dict]:
        """Get agent response for transcript."""
        try:
            response = await self.client.post(
                f"{self.base_url}/turn",
                json={
                    "session_id": session_id,
                    "transcript": transcript,
                },
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            self.logger.error("Get response failed", error=str(e))
            return None
