"""Audio router for directing audio streams."""

import asyncio
import structlog
from typing import Optional, Dict, Callable, Any
from dataclasses import dataclass
from enum import Enum
import httpx
import time

from app.config import settings


logger = structlog.get_logger()


class RouteTarget(str, Enum):
    """Audio route targets."""
    ASR = "asr"
    TTS = "tts"
    CALLER = "caller"
    RECORDING = "recording"
    MONITOR = "monitor"


@dataclass
class RouteConfig:
    """Route configuration."""
    target: RouteTarget
    url: str
    enabled: bool = True
    priority: int = 0


class AudioRouter:
    """
    Routes audio between services.

    Handles:
    - Sending audio to ASR service
    - Receiving audio from TTS service
    - Routing to recording/monitoring
    - Multi-destination routing
    """

    def __init__(self, session_id: str):
        self.session_id = session_id

        # HTTP client for service communication
        self._client: Optional[httpx.AsyncClient] = None

        # Routes
        self._routes: Dict[RouteTarget, RouteConfig] = {
            RouteTarget.ASR: RouteConfig(
                target=RouteTarget.ASR,
                url=settings.asr_service_url,
            ),
            RouteTarget.TTS: RouteConfig(
                target=RouteTarget.TTS,
                url=settings.tts_service_url,
            ),
        }

        # TTS audio queue
        self._tts_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Outbound audio queue (to caller)
        self._outbound_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # WebSocket connections
        self._ws_connections: Dict[str, Any] = {}

        # Callbacks
        self._on_asr_result: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

        # Statistics
        self._audio_sent_to_asr = 0
        self._audio_received_from_tts = 0
        self._errors = 0

        logger.debug("audio_router_created", session_id=session_id)

    async def connect(self) -> None:
        """Initialize connections to services."""
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info("audio_router_connected", session_id=self.session_id)

    async def disconnect(self) -> None:
        """Close all connections."""
        if self._client:
            await self._client.aclose()
            self._client = None

        # Clear queues
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self._outbound_queue.empty():
            try:
                self._outbound_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("audio_router_disconnected", session_id=self.session_id)

    async def route_to_asr(self, audio_data: bytes) -> None:
        """
        Send audio to ASR service.

        Args:
            audio_data: PCM audio to transcribe
        """
        route = self._routes.get(RouteTarget.ASR)
        if not route or not route.enabled:
            return

        try:
            # In production: use WebSocket for streaming
            # For now: HTTP POST
            if self._client:
                await self._client.post(
                    f"{route.url}/v1/audio",
                    content=audio_data,
                    headers={
                        "Content-Type": "audio/pcm",
                        "X-Session-ID": self.session_id,
                    },
                )
                self._audio_sent_to_asr += len(audio_data)

        except Exception as e:
            self._errors += 1
            logger.error(
                "asr_route_error",
                session_id=self.session_id,
                error=str(e),
            )

    async def route_to_tts(self, text: str, voice_id: Optional[str] = None) -> None:
        """
        Request TTS synthesis.

        Args:
            text: Text to synthesize
            voice_id: Optional voice ID
        """
        route = self._routes.get(RouteTarget.TTS)
        if not route or not route.enabled:
            return

        try:
            if self._client:
                response = await self._client.post(
                    f"{route.url}/v1/synthesize",
                    json={
                        "text": text,
                        "voice_id": voice_id,
                        "session_id": self.session_id,
                    },
                )

                if response.status_code == 200:
                    audio_data = response.content
                    await self._tts_queue.put(audio_data)
                    self._audio_received_from_tts += len(audio_data)

        except Exception as e:
            self._errors += 1
            logger.error(
                "tts_route_error",
                session_id=self.session_id,
                error=str(e),
            )

    async def get_tts_audio(self) -> Optional[bytes]:
        """
        Get TTS audio from queue.

        Returns:
            Audio bytes or None if queue empty
        """
        try:
            return self._tts_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def send_audio(self, audio_data: bytes) -> None:
        """
        Queue audio for sending to caller.

        Args:
            audio_data: Encoded audio to send
        """
        try:
            self._outbound_queue.put_nowait(audio_data)
        except asyncio.QueueFull:
            # Drop oldest
            try:
                self._outbound_queue.get_nowait()
                self._outbound_queue.put_nowait(audio_data)
            except asyncio.QueueEmpty:
                pass

    async def get_outbound_audio(self) -> Optional[bytes]:
        """
        Get audio for sending to caller.

        Returns:
            Audio bytes or None if queue empty
        """
        try:
            return self._outbound_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def clear_outbound_queue(self) -> int:
        """
        Clear outbound queue (for interrupts).

        Returns:
            Number of packets cleared
        """
        count = 0
        while not self._outbound_queue.empty():
            try:
                self._outbound_queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        return count

    def set_route_enabled(self, target: RouteTarget, enabled: bool) -> None:
        """Enable or disable a route."""
        if target in self._routes:
            self._routes[target].enabled = enabled

    def set_callbacks(
        self,
        on_asr_result: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        """Set router callbacks."""
        self._on_asr_result = on_asr_result
        self._on_error = on_error

    def get_statistics(self) -> dict:
        """Get router statistics."""
        return {
            "session_id": self.session_id,
            "audio_sent_to_asr_bytes": self._audio_sent_to_asr,
            "audio_received_from_tts_bytes": self._audio_received_from_tts,
            "tts_queue_size": self._tts_queue.qsize(),
            "outbound_queue_size": self._outbound_queue.qsize(),
            "errors": self._errors,
        }
