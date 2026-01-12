"""Streaming API for real-time events via WebSocket."""

from typing import Optional, Dict, Any, Callable, List, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import json
import logging

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketClientProtocol = Any

logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """Types of streaming events."""
    # Connection events
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"

    # Call events
    CALL_STARTED = "call.started"
    CALL_RINGING = "call.ringing"
    CALL_ANSWERED = "call.answered"
    CALL_ENDED = "call.ended"
    CALL_FAILED = "call.failed"

    # Audio events
    AUDIO_INPUT = "audio.input"
    AUDIO_OUTPUT = "audio.output"

    # Transcription events
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"

    # Conversation events
    TURN_STARTED = "turn.started"
    TURN_ENDED = "turn.ended"
    AGENT_THINKING = "agent.thinking"
    AGENT_SPEAKING = "agent.speaking"

    # Function events
    FUNCTION_CALL = "function.call"
    FUNCTION_RESULT = "function.result"

    # State events
    STATE_CHANGED = "state.changed"
    VAD_SPEECH_START = "vad.speech_start"
    VAD_SPEECH_END = "vad.speech_end"

    # Metrics events
    LATENCY_REPORT = "latency.report"


@dataclass
class StreamEvent:
    """A streaming event."""
    type: StreamEventType
    timestamp: datetime
    data: Dict[str, Any]
    call_id: Optional[str] = None
    session_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamEvent":
        return cls(
            type=StreamEventType(data["type"]),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            data=data.get("data", {}),
            call_id=data.get("call_id"),
            session_id=data.get("session_id"),
        )


@dataclass
class TranscriptEvent:
    """A transcription event."""
    text: str
    is_final: bool
    confidence: float
    speaker: str  # "user" or "agent"
    start_time: float
    end_time: float
    words: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_stream_event(cls, event: StreamEvent) -> "TranscriptEvent":
        data = event.data
        return cls(
            text=data.get("text", ""),
            is_final=event.type == StreamEventType.TRANSCRIPT_FINAL,
            confidence=data.get("confidence", 0.0),
            speaker=data.get("speaker", "user"),
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
            words=data.get("words", []),
        )


@dataclass
class AudioFrame:
    """An audio frame for streaming."""
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "pcm_s16le"
    timestamp_ms: int = 0

    def to_message(self) -> Dict[str, Any]:
        """Convert to WebSocket message."""
        import base64
        return {
            "type": "audio.input",
            "data": {
                "audio": base64.b64encode(self.data).decode("utf-8"),
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "encoding": self.encoding,
                "timestamp_ms": self.timestamp_ms,
            }
        }


EventHandler = Callable[[StreamEvent], Any]


class StreamingConnection:
    """
    WebSocket connection for real-time streaming.

    Example:
        async with StreamingConnection(api_key, call_id) as stream:
            async for event in stream:
                if event.type == StreamEventType.TRANSCRIPT_FINAL:
                    print(f"Transcript: {event.data['text']}")
    """

    def __init__(
        self,
        api_key: str,
        call_id: Optional[str] = None,
        session_id: Optional[str] = None,
        base_url: str = "wss://api.bvrai.com",
        auto_reconnect: bool = True,
        reconnect_interval: float = 1.0,
        max_reconnect_attempts: int = 5,
    ):
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required for streaming. Install with: pip install websockets")

        self.api_key = api_key
        self.call_id = call_id
        self.session_id = session_id
        self.base_url = base_url
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts

        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._handlers: Dict[StreamEventType, List[EventHandler]] = {}
        self._reconnect_count = 0
        self._should_stop = False
        self._event_queue: asyncio.Queue = asyncio.Queue()

    @property
    def connected(self) -> bool:
        """Check if connected."""
        return self._connected and self._ws is not None

    def _get_url(self) -> str:
        """Build the WebSocket URL."""
        url = f"{self.base_url}/v1/stream"
        if self.call_id:
            url += f"?call_id={self.call_id}"
        elif self.session_id:
            url += f"?session_id={self.session_id}"
        return url

    async def connect(self) -> None:
        """Connect to the streaming endpoint."""
        url = self._get_url()
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            self._ws = await websockets.connect(
                url,
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=10,
            )
            self._connected = True
            self._reconnect_count = 0
            logger.info(f"Connected to streaming endpoint: {url}")

            # Send connected event
            await self._event_queue.put(StreamEvent(
                type=StreamEventType.CONNECTED,
                timestamp=datetime.now(),
                data={"url": url},
                call_id=self.call_id,
                session_id=self.session_id,
            ))

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Disconnect from the streaming endpoint."""
        self._should_stop = True
        self._connected = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        await self._event_queue.put(StreamEvent(
            type=StreamEventType.DISCONNECTED,
            timestamp=datetime.now(),
            data={},
            call_id=self.call_id,
            session_id=self.session_id,
        ))

    async def _reconnect(self) -> bool:
        """Attempt to reconnect."""
        if not self.auto_reconnect or self._should_stop:
            return False

        if self._reconnect_count >= self.max_reconnect_attempts:
            logger.error("Max reconnect attempts reached")
            return False

        self._reconnect_count += 1
        wait_time = self.reconnect_interval * (2 ** (self._reconnect_count - 1))
        logger.info(f"Reconnecting in {wait_time}s (attempt {self._reconnect_count})")

        await asyncio.sleep(wait_time)

        try:
            await self.connect()
            return True
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")
            return await self._reconnect()

    async def _receive_loop(self) -> None:
        """Main receive loop."""
        while not self._should_stop:
            try:
                if not self._ws:
                    if not await self._reconnect():
                        break
                    continue

                message = await self._ws.recv()

                if isinstance(message, bytes):
                    # Binary message (audio)
                    event = StreamEvent(
                        type=StreamEventType.AUDIO_OUTPUT,
                        timestamp=datetime.now(),
                        data={"audio": message},
                        call_id=self.call_id,
                        session_id=self.session_id,
                    )
                else:
                    # JSON message
                    data = json.loads(message)
                    event = StreamEvent.from_dict(data)
                    event.call_id = event.call_id or self.call_id
                    event.session_id = event.session_id or self.session_id

                await self._event_queue.put(event)

                # Call registered handlers
                await self._dispatch_event(event)

            except websockets.ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")
                self._connected = False
                self._ws = None

                if not await self._reconnect():
                    break

            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                await self._event_queue.put(StreamEvent(
                    type=StreamEventType.ERROR,
                    timestamp=datetime.now(),
                    data={"error": str(e)},
                    call_id=self.call_id,
                    session_id=self.session_id,
                ))

    async def _dispatch_event(self, event: StreamEvent) -> None:
        """Dispatch event to registered handlers."""
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Handler error: {e}")

    def on(self, event_type: StreamEventType, handler: EventHandler) -> None:
        """Register an event handler."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def off(self, event_type: StreamEventType, handler: EventHandler) -> None:
        """Unregister an event handler."""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def send(self, message: Dict[str, Any]) -> None:
        """Send a message to the server."""
        if not self._ws:
            raise RuntimeError("Not connected")

        await self._ws.send(json.dumps(message))

    async def send_audio(self, audio_frame: AudioFrame) -> None:
        """Send an audio frame."""
        await self.send(audio_frame.to_message())

    async def send_text(self, text: str) -> None:
        """Send text input (simulated speech)."""
        await self.send({
            "type": "text.input",
            "data": {"text": text},
        })

    async def send_dtmf(self, digits: str) -> None:
        """Send DTMF tones."""
        await self.send({
            "type": "dtmf.input",
            "data": {"digits": digits},
        })

    async def interrupt(self) -> None:
        """Interrupt the current agent response."""
        await self.send({"type": "interrupt"})

    async def end_turn(self) -> None:
        """Signal end of user turn."""
        await self.send({"type": "turn.end"})

    async def __aenter__(self) -> "StreamingConnection":
        """Enter async context."""
        await self.connect()
        asyncio.create_task(self._receive_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        await self.disconnect()

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """Iterate over events."""
        return self

    async def __anext__(self) -> StreamEvent:
        """Get next event."""
        if self._should_stop and self._event_queue.empty():
            raise StopAsyncIteration

        try:
            event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
            return event
        except asyncio.TimeoutError:
            if self._should_stop:
                raise StopAsyncIteration
            return await self.__anext__()


class StreamingSession:
    """
    High-level streaming session for voice conversations.

    Example:
        async with StreamingSession(api_key, agent_id) as session:
            await session.start_call(to_number="+1234567890")

            @session.on_transcript
            def handle_transcript(text: str, is_final: bool, speaker: str):
                print(f"[{speaker}]: {text}")

            @session.on_function_call
            async def handle_function(name: str, args: dict) -> dict:
                if name == "get_weather":
                    return {"temperature": 72, "condition": "sunny"}
                return {}

            await session.wait_for_end()
    """

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        base_url: str = "wss://api.bvrai.com",
    ):
        self.api_key = api_key
        self.agent_id = agent_id
        self.base_url = base_url

        self._connection: Optional[StreamingConnection] = None
        self._call_id: Optional[str] = None
        self._ended = asyncio.Event()

        self._transcript_handler: Optional[Callable] = None
        self._function_handler: Optional[Callable] = None
        self._state_handler: Optional[Callable] = None
        self._audio_handler: Optional[Callable] = None

    @property
    def call_id(self) -> Optional[str]:
        """Get the current call ID."""
        return self._call_id

    @property
    def connected(self) -> bool:
        """Check if connected."""
        return self._connection is not None and self._connection.connected

    async def connect(self) -> None:
        """Connect to streaming endpoint."""
        self._connection = StreamingConnection(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        await self._connection.connect()
        asyncio.create_task(self._event_loop())

    async def disconnect(self) -> None:
        """Disconnect from streaming endpoint."""
        if self._connection:
            await self._connection.disconnect()
            self._connection = None

    async def start_call(
        self,
        to_number: Optional[str] = None,
        from_number: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new call."""
        if not self._connection:
            raise RuntimeError("Not connected")

        await self._connection.send({
            "type": "call.start",
            "data": {
                "agent_id": self.agent_id,
                "to_number": to_number,
                "from_number": from_number,
                "metadata": metadata or {},
            }
        })

        # Wait for call_id
        while not self._call_id:
            await asyncio.sleep(0.1)

        return self._call_id

    async def end_call(self) -> None:
        """End the current call."""
        if not self._connection:
            return

        await self._connection.send({
            "type": "call.end",
            "data": {"call_id": self._call_id},
        })

    async def wait_for_end(self) -> None:
        """Wait for the call to end."""
        await self._ended.wait()

    async def _event_loop(self) -> None:
        """Process incoming events."""
        if not self._connection:
            return

        async for event in self._connection:
            await self._handle_event(event)

    async def _handle_event(self, event: StreamEvent) -> None:
        """Handle a streaming event."""
        if event.type == StreamEventType.CALL_STARTED:
            self._call_id = event.data.get("call_id")

        elif event.type in (StreamEventType.CALL_ENDED, StreamEventType.CALL_FAILED):
            self._ended.set()

        elif event.type in (StreamEventType.TRANSCRIPT_PARTIAL, StreamEventType.TRANSCRIPT_FINAL):
            if self._transcript_handler:
                transcript = TranscriptEvent.from_stream_event(event)
                result = self._transcript_handler(
                    transcript.text,
                    transcript.is_final,
                    transcript.speaker,
                )
                if asyncio.iscoroutine(result):
                    await result

        elif event.type == StreamEventType.FUNCTION_CALL:
            if self._function_handler:
                name = event.data.get("name", "")
                args = event.data.get("arguments", {})
                call_id = event.data.get("call_id", "")

                result = self._function_handler(name, args)
                if asyncio.iscoroutine(result):
                    result = await result

                # Send result back
                if self._connection:
                    await self._connection.send({
                        "type": "function.result",
                        "data": {
                            "call_id": call_id,
                            "result": result,
                        }
                    })

        elif event.type == StreamEventType.STATE_CHANGED:
            if self._state_handler:
                state = event.data.get("state", "")
                result = self._state_handler(state)
                if asyncio.iscoroutine(result):
                    await result

        elif event.type == StreamEventType.AUDIO_OUTPUT:
            if self._audio_handler:
                audio_data = event.data.get("audio", b"")
                result = self._audio_handler(audio_data)
                if asyncio.iscoroutine(result):
                    await result

    def on_transcript(self, handler: Callable) -> Callable:
        """Register transcript handler."""
        self._transcript_handler = handler
        return handler

    def on_function_call(self, handler: Callable) -> Callable:
        """Register function call handler."""
        self._function_handler = handler
        return handler

    def on_state_change(self, handler: Callable) -> Callable:
        """Register state change handler."""
        self._state_handler = handler
        return handler

    def on_audio(self, handler: Callable) -> Callable:
        """Register audio handler."""
        self._audio_handler = handler
        return handler

    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio data."""
        if self._connection:
            frame = AudioFrame(data=audio_data)
            await self._connection.send_audio(frame)

    async def send_text(self, text: str) -> None:
        """Send text input."""
        if self._connection:
            await self._connection.send_text(text)

    async def interrupt(self) -> None:
        """Interrupt the agent."""
        if self._connection:
            await self._connection.interrupt()

    async def __aenter__(self) -> "StreamingSession":
        """Enter async context."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        await self.disconnect()


class AudioStreamPlayer:
    """
    Utility for playing streaming audio output.

    Requires pyaudio: pip install pyaudio
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        try:
            import pyaudio
            self._pyaudio = pyaudio
        except ImportError:
            raise ImportError("pyaudio required. Install with: pip install pyaudio")

        self.sample_rate = sample_rate
        self.channels = channels
        self._pa = pyaudio.PyAudio()
        self._stream = None

    def start(self) -> None:
        """Start audio playback."""
        self._stream = self._pa.open(
            format=self._pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
        )

    def play(self, audio_data: bytes) -> None:
        """Play audio data."""
        if self._stream:
            self._stream.write(audio_data)

    def stop(self) -> None:
        """Stop audio playback."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "AudioStreamPlayer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class AudioStreamRecorder:
    """
    Utility for recording audio input for streaming.

    Requires pyaudio: pip install pyaudio
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
    ):
        try:
            import pyaudio
            self._pyaudio = pyaudio
        except ImportError:
            raise ImportError("pyaudio required. Install with: pip install pyaudio")

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._recording = False

    def start(self) -> None:
        """Start recording."""
        self._stream = self._pa.open(
            format=self._pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        self._recording = True

    def read(self) -> bytes:
        """Read a chunk of audio."""
        if not self._stream:
            raise RuntimeError("Not recording")
        return self._stream.read(self.chunk_size, exception_on_overflow=False)

    def stop(self) -> None:
        """Stop recording."""
        self._recording = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    async def stream_to(
        self,
        connection: StreamingConnection,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Stream recorded audio to a connection."""
        self.start()
        start_time = asyncio.get_event_loop().time()
        timestamp_ms = 0
        chunk_duration_ms = int(self.chunk_size / self.sample_rate * 1000)

        try:
            while self._recording:
                if duration_seconds:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed >= duration_seconds:
                        break

                audio_data = self.read()
                frame = AudioFrame(
                    data=audio_data,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    timestamp_ms=timestamp_ms,
                )
                await connection.send_audio(frame)
                timestamp_ms += chunk_duration_ms

                # Small delay to prevent flooding
                await asyncio.sleep(0.01)

        finally:
            self.stop()

    def __enter__(self) -> "AudioStreamRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
