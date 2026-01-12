"""WebSocket audio streaming."""

import asyncio
import structlog
from typing import Optional, Callable, Any
from dataclasses import dataclass
import json
import base64
import time

from fastapi import WebSocket, WebSocketDisconnect


logger = structlog.get_logger()


@dataclass
class WebSocketConfig:
    """WebSocket stream configuration."""
    session_id: str
    audio_format: str = "pcmu"  # pcmu, pcm16, opus
    sample_rate: int = 8000
    encoding: str = "base64"  # base64, binary


class WebSocketStream:
    """
    WebSocket audio stream handler.

    Handles bidirectional audio streaming over WebSocket,
    compatible with Twilio Media Streams and browser WebRTC.
    """

    def __init__(
        self,
        websocket: WebSocket,
        config: WebSocketConfig,
    ):
        self.websocket = websocket
        self.config = config
        self.session_id = config.session_id

        # Callbacks
        self._on_audio_received: Optional[Callable] = None
        self._on_connected: Optional[Callable] = None
        self._on_disconnected: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

        # State
        self._connected = False
        self._receive_task: Optional[asyncio.Task] = None

        # Statistics
        self._frames_received = 0
        self._frames_sent = 0
        self._bytes_received = 0
        self._bytes_sent = 0
        self._connected_at: Optional[float] = None

        logger.debug(
            "websocket_stream_created",
            session_id=self.session_id,
        )

    async def connect(self) -> None:
        """Accept WebSocket connection and start receiving."""
        await self.websocket.accept()
        self._connected = True
        self._connected_at = time.time()

        # Start receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())

        if self._on_connected:
            await self._on_connected()

        logger.info("websocket_stream_connected", session_id=self.session_id)

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._connected = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        try:
            await self.websocket.close()
        except Exception:
            pass

        if self._on_disconnected:
            await self._on_disconnected()

        logger.info(
            "websocket_stream_disconnected",
            session_id=self.session_id,
            stats=self.get_statistics(),
        )

    async def send_audio(self, audio_data: bytes) -> bool:
        """
        Send audio to client.

        Args:
            audio_data: Audio bytes to send

        Returns:
            True if sent successfully
        """
        if not self._connected:
            return False

        try:
            # Encode based on config
            if self.config.encoding == "base64":
                encoded = base64.b64encode(audio_data).decode("utf-8")
                message = {
                    "event": "media",
                    "streamSid": self.session_id,
                    "media": {
                        "payload": encoded,
                    },
                }
                await self.websocket.send_json(message)
            else:
                await self.websocket.send_bytes(audio_data)

            self._frames_sent += 1
            self._bytes_sent += len(audio_data)
            return True

        except Exception as e:
            logger.error(
                "websocket_send_error",
                session_id=self.session_id,
                error=str(e),
            )
            return False

    async def send_event(self, event_type: str, data: dict) -> bool:
        """
        Send event message to client.

        Args:
            event_type: Event type
            data: Event data

        Returns:
            True if sent successfully
        """
        if not self._connected:
            return False

        try:
            message = {
                "event": event_type,
                "streamSid": self.session_id,
                **data,
            }
            await self.websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(
                "websocket_event_error",
                session_id=self.session_id,
                error=str(e),
            )
            return False

    def set_callbacks(
        self,
        on_audio_received: Optional[Callable] = None,
        on_connected: Optional[Callable] = None,
        on_disconnected: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        """Set stream callbacks."""
        self._on_audio_received = on_audio_received
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._on_error = on_error

    async def _receive_loop(self) -> None:
        """Main receive loop."""
        while self._connected:
            try:
                # Receive message
                message = await self.websocket.receive()

                if message["type"] == "websocket.disconnect":
                    break

                # Handle based on message type
                if "text" in message:
                    await self._handle_json_message(message["text"])
                elif "bytes" in message:
                    await self._handle_binary_message(message["bytes"])

            except WebSocketDisconnect:
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "websocket_receive_error",
                    session_id=self.session_id,
                    error=str(e),
                )
                if self._on_error:
                    await self._on_error(e)
                break

        await self.disconnect()

    async def _handle_json_message(self, text: str) -> None:
        """Handle JSON message (Twilio format)."""
        try:
            data = json.loads(text)
            event = data.get("event")

            if event == "media":
                # Audio data
                media = data.get("media", {})
                payload = media.get("payload", "")

                if payload:
                    audio_data = base64.b64decode(payload)
                    await self._process_audio(audio_data, data)

            elif event == "start":
                # Stream started
                logger.info(
                    "websocket_stream_started",
                    session_id=self.session_id,
                    data=data,
                )

            elif event == "stop":
                # Stream stopped
                self._connected = False

            elif event == "mark":
                # Mark event (Twilio)
                logger.debug(
                    "websocket_mark_event",
                    session_id=self.session_id,
                    name=data.get("mark", {}).get("name"),
                )

        except json.JSONDecodeError as e:
            logger.error(
                "websocket_json_error",
                session_id=self.session_id,
                error=str(e),
            )

    async def _handle_binary_message(self, data: bytes) -> None:
        """Handle binary audio message."""
        await self._process_audio(data, {})

    async def _process_audio(self, audio_data: bytes, metadata: dict) -> None:
        """Process received audio."""
        self._frames_received += 1
        self._bytes_received += len(audio_data)

        if self._on_audio_received:
            timestamp = metadata.get("media", {}).get("timestamp", 0)
            await self._on_audio_received(audio_data, timestamp)

    def get_statistics(self) -> dict:
        """Get stream statistics."""
        duration = 0
        if self._connected_at:
            duration = time.time() - self._connected_at

        return {
            "session_id": self.session_id,
            "connected": self._connected,
            "duration_seconds": round(duration, 2),
            "frames_received": self._frames_received,
            "frames_sent": self._frames_sent,
            "bytes_received": self._bytes_received,
            "bytes_sent": self._bytes_sent,
        }

    @property
    def is_connected(self) -> bool:
        """Check if stream is connected."""
        return self._connected
