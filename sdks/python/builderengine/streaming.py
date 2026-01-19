"""
Builder Engine Python SDK - Streaming Client

This module provides WebSocket streaming capabilities for real-time events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional, Dict, Any, Callable, AsyncIterator, Union
from dataclasses import dataclass
from enum import Enum

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

from builderengine.exceptions import WebSocketError, AuthenticationError

logger = logging.getLogger("builderengine.streaming")


class StreamEventType(str, Enum):
    """Types of streaming events."""
    # Connection events
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

    # Call events
    CALL_STARTED = "call.started"
    CALL_RINGING = "call.ringing"
    CALL_ANSWERED = "call.answered"
    CALL_ENDED = "call.ended"
    CALL_FAILED = "call.failed"

    # Audio events
    AUDIO_START = "audio.start"
    AUDIO_DATA = "audio.data"
    AUDIO_END = "audio.end"

    # Transcription events
    TRANSCRIPTION_PARTIAL = "transcription.partial"
    TRANSCRIPTION_FINAL = "transcription.final"

    # Conversation events
    USER_SPEECH_START = "user.speech.start"
    USER_SPEECH_END = "user.speech.end"
    AGENT_SPEECH_START = "agent.speech.start"
    AGENT_SPEECH_END = "agent.speech.end"

    # Function events
    FUNCTION_CALL = "function.call"
    FUNCTION_RESULT = "function.result"

    # DTMF events
    DTMF_RECEIVED = "dtmf.received"


@dataclass
class StreamEvent:
    """A streaming event from the WebSocket connection."""
    type: StreamEventType
    data: Dict[str, Any]
    call_id: Optional[str] = None
    timestamp: Optional[str] = None

    @classmethod
    def from_message(cls, message: Dict[str, Any]) -> "StreamEvent":
        """Create a StreamEvent from a WebSocket message."""
        return cls(
            type=StreamEventType(message.get("type", "error")),
            data=message.get("data", {}),
            call_id=message.get("call_id"),
            timestamp=message.get("timestamp"),
        )


# Type alias for event handlers
EventHandler = Callable[[StreamEvent], None]
AsyncEventHandler = Callable[[StreamEvent], Any]


class StreamingClient:
    """
    WebSocket client for real-time streaming events.

    Provides real-time access to call events, transcriptions,
    and audio streams via WebSocket connections.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> streaming = client.streaming()
        >>>
        >>> @streaming.on(StreamEventType.TRANSCRIPTION_FINAL)
        ... async def handle_transcription(event):
        ...     print(f"User said: {event.data['text']}")
        >>>
        >>> async with streaming.connect(call_id="call_abc123"):
        ...     await streaming.listen()
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "wss://api.builderengine.io",
    ) -> None:
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets package is required for streaming. "
                "Install it with: pip install websockets"
            )

        self._api_key = api_key
        self._base_url = base_url
        self._websocket: Optional[WebSocketClientProtocol] = None
        self._handlers: Dict[StreamEventType, list] = {}
        self._running = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 1.0

    def on(
        self,
        event_type: Union[StreamEventType, str],
    ) -> Callable[[AsyncEventHandler], AsyncEventHandler]:
        """
        Decorator to register an event handler.

        Args:
            event_type: Type of event to handle

        Returns:
            Decorator function

        Example:
            >>> @streaming.on(StreamEventType.CALL_STARTED)
            ... async def on_call_started(event):
            ...     print(f"Call started: {event.call_id}")
        """
        if isinstance(event_type, str):
            event_type = StreamEventType(event_type)

        def decorator(handler: AsyncEventHandler) -> AsyncEventHandler:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
            return handler

        return decorator

    def add_handler(
        self,
        event_type: Union[StreamEventType, str],
        handler: AsyncEventHandler,
    ) -> None:
        """
        Add an event handler.

        Args:
            event_type: Type of event to handle
            handler: Async function to handle the event
        """
        if isinstance(event_type, str):
            event_type = StreamEventType(event_type)

        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def remove_handler(
        self,
        event_type: Union[StreamEventType, str],
        handler: AsyncEventHandler,
    ) -> None:
        """
        Remove an event handler.

        Args:
            event_type: Type of event
            handler: Handler to remove
        """
        if isinstance(event_type, str):
            event_type = StreamEventType(event_type)

        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)

    async def connect(
        self,
        call_id: Optional[str] = None,
        subscribe_all: bool = False,
    ) -> "StreamingClient":
        """
        Connect to the WebSocket server.

        Args:
            call_id: Specific call to subscribe to
            subscribe_all: Subscribe to all events (requires permission)

        Returns:
            Self for chaining
        """
        url = f"{self._base_url}/ws/v1/stream"
        params = []

        if call_id:
            params.append(f"call_id={call_id}")
        if subscribe_all:
            params.append("subscribe_all=true")

        if params:
            url = f"{url}?{'&'.join(params)}"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            self._websocket = await websockets.connect(
                url,
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=10,
            )
            self._running = True
            self._reconnect_attempts = 0

            logger.info(f"Connected to WebSocket: {url}")

            # Emit connected event
            await self._dispatch_event(StreamEvent(
                type=StreamEventType.CONNECTED,
                data={"url": url, "call_id": call_id},
            ))

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise WebSocketError(f"Failed to connect: {e}")

        return self

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        self._running = False

        if self._websocket:
            await self._websocket.close()
            self._websocket = None

            logger.info("Disconnected from WebSocket")

            await self._dispatch_event(StreamEvent(
                type=StreamEventType.DISCONNECTED,
                data={},
            ))

    async def send(self, message: Dict[str, Any]) -> None:
        """
        Send a message to the server.

        Args:
            message: Message to send
        """
        if not self._websocket:
            raise WebSocketError("Not connected")

        await self._websocket.send(json.dumps(message))
        logger.debug(f"Sent message: {message}")

    async def subscribe(self, call_id: str) -> None:
        """
        Subscribe to events for a specific call.

        Args:
            call_id: Call to subscribe to
        """
        await self.send({
            "action": "subscribe",
            "call_id": call_id,
        })

    async def unsubscribe(self, call_id: str) -> None:
        """
        Unsubscribe from events for a specific call.

        Args:
            call_id: Call to unsubscribe from
        """
        await self.send({
            "action": "unsubscribe",
            "call_id": call_id,
        })

    async def send_audio(self, call_id: str, audio_data: bytes) -> None:
        """
        Send audio data to a call.

        Args:
            call_id: Call to send audio to
            audio_data: Raw audio bytes (PCM 16-bit, 16kHz)
        """
        import base64

        await self.send({
            "action": "audio",
            "call_id": call_id,
            "data": base64.b64encode(audio_data).decode("utf-8"),
        })

    async def inject_text(self, call_id: str, text: str, interrupt: bool = False) -> None:
        """
        Inject text for the agent to speak.

        Args:
            call_id: Call to inject into
            text: Text for the agent to speak
            interrupt: Whether to interrupt current speech
        """
        await self.send({
            "action": "inject",
            "call_id": call_id,
            "text": text,
            "interrupt": interrupt,
        })

    async def listen(self) -> None:
        """
        Start listening for events.

        This method blocks until disconnected.
        """
        if not self._websocket:
            raise WebSocketError("Not connected")

        try:
            async for message in self._websocket:
                if not self._running:
                    break

                try:
                    data = json.loads(message)
                    event = StreamEvent.from_message(data)
                    await self._dispatch_event(event)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except websockets.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            await self._handle_disconnect()
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            raise WebSocketError(str(e))

    async def events(self) -> AsyncIterator[StreamEvent]:
        """
        Async iterator for events.

        Yields:
            StreamEvent objects

        Example:
            >>> async for event in streaming.events():
            ...     print(f"Event: {event.type}")
        """
        if not self._websocket:
            raise WebSocketError("Not connected")

        try:
            async for message in self._websocket:
                if not self._running:
                    break

                try:
                    data = json.loads(message)
                    yield StreamEvent.from_message(data)
                except json.JSONDecodeError:
                    continue
        except websockets.ConnectionClosed:
            pass

    async def _dispatch_event(self, event: StreamEvent) -> None:
        """Dispatch an event to registered handlers."""
        handlers = self._handlers.get(event.type, [])

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    async def _handle_disconnect(self) -> None:
        """Handle unexpected disconnection."""
        if self._running and self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))

            logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
            await asyncio.sleep(delay)

            try:
                await self.connect()
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                await self._handle_disconnect()

    async def __aenter__(self) -> "StreamingClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


class CallEventHandler:
    """
    Handler for call-specific events.

    Provides a simplified interface for handling events
    for a specific call.

    Example:
        >>> async def handle_call(call_id):
        ...     async with CallEventHandler(streaming, call_id) as handler:
        ...         async for event in handler.events():
        ...             if event.type == StreamEventType.TRANSCRIPTION_FINAL:
        ...                 print(f"User: {event.data['text']}")
    """

    def __init__(
        self,
        streaming: StreamingClient,
        call_id: str,
    ) -> None:
        self._streaming = streaming
        self._call_id = call_id
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

    async def __aenter__(self) -> "CallEventHandler":
        """Subscribe to call events."""
        await self._streaming.subscribe(self._call_id)

        # Add handler for this call
        @self._streaming.on(StreamEventType.TRANSCRIPTION_FINAL)
        async def handler(event: StreamEvent):
            if event.call_id == self._call_id:
                await self._queue.put(event)

        self._handler = handler
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Unsubscribe from call events."""
        await self._streaming.unsubscribe(self._call_id)

    async def events(self) -> AsyncIterator[StreamEvent]:
        """Iterate over events for this call."""
        while True:
            event = await self._queue.get()
            yield event


class TranscriptionHandler:
    """
    Handler for real-time transcription.

    Buffers partial transcriptions and yields complete utterances.

    Example:
        >>> handler = TranscriptionHandler()
        >>> async for text in handler.transcribe(streaming, call_id):
        ...     print(f"User said: {text}")
    """

    def __init__(self, min_confidence: float = 0.8) -> None:
        self._min_confidence = min_confidence
        self._buffer = ""

    async def transcribe(
        self,
        streaming: StreamingClient,
        call_id: str,
    ) -> AsyncIterator[str]:
        """
        Stream transcriptions for a call.

        Args:
            streaming: StreamingClient instance
            call_id: Call to transcribe

        Yields:
            Complete utterance strings
        """
        await streaming.subscribe(call_id)

        async for event in streaming.events():
            if event.call_id != call_id:
                continue

            if event.type == StreamEventType.TRANSCRIPTION_PARTIAL:
                self._buffer = event.data.get("text", "")

            elif event.type == StreamEventType.TRANSCRIPTION_FINAL:
                text = event.data.get("text", "")
                confidence = event.data.get("confidence", 1.0)

                if confidence >= self._min_confidence and text:
                    yield text
                    self._buffer = ""

            elif event.type == StreamEventType.CALL_ENDED:
                break
