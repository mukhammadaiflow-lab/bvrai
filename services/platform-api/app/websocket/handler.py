"""WebSocket connection handler."""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import json
import logging
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    STREAMING = "streaming"
    PAUSED = "paused"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


class MessageType(str, Enum):
    """Types of WebSocket messages."""
    # Control messages
    PING = "ping"
    PONG = "pong"
    AUTH = "auth"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILED = "auth_failed"
    ERROR = "error"

    # Call control
    CALL_START = "call_start"
    CALL_END = "call_end"
    CALL_UPDATE = "call_update"

    # Audio messages
    AUDIO_START = "audio_start"
    AUDIO_CHUNK = "audio_chunk"
    AUDIO_END = "audio_end"
    AUDIO_CONFIG = "audio_config"

    # Transcript messages
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"

    # Speech events
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"

    # Agent messages
    AGENT_RESPONSE = "agent_response"
    AGENT_THINKING = "agent_thinking"
    AGENT_FUNCTION_CALL = "agent_function_call"
    AGENT_FUNCTION_RESULT = "agent_function_result"

    # Media control
    MUTE = "mute"
    UNMUTE = "unmute"
    HOLD = "hold"
    RESUME = "resume"

    # Transfer
    TRANSFER_START = "transfer_start"
    TRANSFER_COMPLETE = "transfer_complete"

    # Custom
    CUSTOM = "custom"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: MessageType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebSocketMessage":
        """Create from dictionary."""
        return cls(
            type=MessageType(data.get("type", "custom")),
            data=data.get("data", {}),
            message_id=data.get("message_id", str(uuid.uuid4())),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "WebSocketMessage":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection."""
    connection_id: str
    websocket: WebSocket
    state: ConnectionState = ConnectionState.CONNECTING
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    message_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "connection_id": self.connection_id,
            "state": self.state.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "call_id": self.call_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": self.message_count,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
        }


MessageHandler = Callable[[WebSocketConnection, WebSocketMessage], Awaitable[None]]


class WebSocketHandler:
    """
    Handler for WebSocket connections.

    Manages connection lifecycle, message routing, and protocol handling.

    Usage:
        handler = WebSocketHandler()

        @handler.on(MessageType.AUDIO_CHUNK)
        async def handle_audio(conn, msg):
            await process_audio(msg.data["audio"])

        await handler.handle_connection(websocket)
    """

    def __init__(
        self,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0,
        max_message_size: int = 1024 * 1024,  # 1MB
        auth_timeout: float = 30.0,
        require_auth: bool = True,
    ):
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.max_message_size = max_message_size
        self.auth_timeout = auth_timeout
        self.require_auth = require_auth

        self._handlers: Dict[MessageType, List[MessageHandler]] = {}
        self._middleware: List[Callable] = []
        self._auth_handler: Optional[Callable] = None
        self._on_connect: Optional[Callable] = None
        self._on_disconnect: Optional[Callable] = None

    def on(self, message_type: MessageType) -> Callable:
        """
        Decorator to register a message handler.

        Usage:
            @handler.on(MessageType.AUDIO_CHUNK)
            async def handle_audio(conn, msg):
                ...
        """
        def decorator(func: MessageHandler) -> MessageHandler:
            if message_type not in self._handlers:
                self._handlers[message_type] = []
            self._handlers[message_type].append(func)
            return func
        return decorator

    def use(self, middleware: Callable) -> None:
        """Add middleware."""
        self._middleware.append(middleware)

    def set_auth_handler(
        self,
        handler: Callable[[WebSocketConnection, Dict[str, Any]], Awaitable[bool]],
    ) -> None:
        """Set authentication handler."""
        self._auth_handler = handler

    def on_connect(self, handler: Callable[[WebSocketConnection], Awaitable[None]]) -> None:
        """Set connection handler."""
        self._on_connect = handler

    def on_disconnect(self, handler: Callable[[WebSocketConnection], Awaitable[None]]) -> None:
        """Set disconnection handler."""
        self._on_disconnect = handler

    async def handle_connection(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> WebSocketConnection:
        """
        Handle a WebSocket connection.

        Main entry point for WebSocket connections.
        """
        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(
            connection_id=connection_id,
            websocket=websocket,
            user_id=user_id,
            session_id=session_id,
        )

        try:
            # Accept connection
            await websocket.accept()
            connection.state = ConnectionState.CONNECTED
            logger.info(f"WebSocket connected: {connection_id}")

            # Call connect handler
            if self._on_connect:
                await self._on_connect(connection)

            # Handle authentication if required
            if self.require_auth and not user_id:
                authenticated = await self._handle_auth(connection)
                if not authenticated:
                    await self._send_error(connection, "Authentication failed")
                    await websocket.close(code=4001, reason="Authentication failed")
                    return connection

            connection.state = ConnectionState.AUTHENTICATED

            # Start ping/pong task
            ping_task = asyncio.create_task(self._ping_loop(connection))

            try:
                # Main message loop
                await self._message_loop(connection)
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass

        except WebSocketDisconnect as e:
            logger.info(f"WebSocket disconnected: {connection_id}, code={e.code}")
            connection.state = ConnectionState.CLOSED

        except Exception as e:
            logger.exception(f"WebSocket error: {connection_id}")
            connection.state = ConnectionState.ERROR
            try:
                await websocket.close(code=1011, reason=str(e))
            except:
                pass

        finally:
            # Call disconnect handler
            if self._on_disconnect:
                try:
                    await self._on_disconnect(connection)
                except Exception as e:
                    logger.error(f"Disconnect handler error: {e}")

        return connection

    async def _handle_auth(self, connection: WebSocketConnection) -> bool:
        """Handle authentication."""
        try:
            # Wait for auth message
            message = await asyncio.wait_for(
                self._receive_message(connection),
                timeout=self.auth_timeout,
            )

            if message.type != MessageType.AUTH:
                return False

            # Call auth handler
            if self._auth_handler:
                authenticated = await self._auth_handler(connection, message.data)
                if authenticated:
                    await self._send_message(
                        connection,
                        WebSocketMessage(type=MessageType.AUTH_SUCCESS),
                    )
                    return True
                else:
                    await self._send_message(
                        connection,
                        WebSocketMessage(
                            type=MessageType.AUTH_FAILED,
                            data={"reason": "Invalid credentials"},
                        ),
                    )
                    return False

            # No auth handler, accept any auth
            connection.user_id = message.data.get("user_id")
            await self._send_message(
                connection,
                WebSocketMessage(type=MessageType.AUTH_SUCCESS),
            )
            return True

        except asyncio.TimeoutError:
            logger.warning(f"Auth timeout: {connection.connection_id}")
            return False

    async def _message_loop(self, connection: WebSocketConnection) -> None:
        """Main message processing loop."""
        while connection.state not in [ConnectionState.CLOSING, ConnectionState.CLOSED]:
            try:
                message = await self._receive_message(connection)
                connection.update_activity()
                connection.message_count += 1

                # Run middleware
                for middleware in self._middleware:
                    should_continue = await middleware(connection, message)
                    if not should_continue:
                        break
                else:
                    # Dispatch to handlers
                    await self._dispatch_message(connection, message)

            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await self._send_error(connection, str(e))

    async def _receive_message(self, connection: WebSocketConnection) -> WebSocketMessage:
        """Receive and parse a message."""
        websocket = connection.websocket

        if websocket.client_state != WebSocketState.CONNECTED:
            raise WebSocketDisconnect(code=1000)

        data = await websocket.receive()

        if "text" in data:
            text = data["text"]
            connection.bytes_received += len(text)
            return WebSocketMessage.from_json(text)

        elif "bytes" in data:
            # Binary message (audio data)
            audio_data = data["bytes"]
            connection.bytes_received += len(audio_data)
            return WebSocketMessage(
                type=MessageType.AUDIO_CHUNK,
                data={"audio": audio_data, "binary": True},
            )

        elif data.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect(code=data.get("code", 1000))

        raise ValueError(f"Unknown message format: {data}")

    async def _dispatch_message(
        self,
        connection: WebSocketConnection,
        message: WebSocketMessage,
    ) -> None:
        """Dispatch message to handlers."""
        # Handle ping/pong
        if message.type == MessageType.PING:
            await self._send_message(
                connection,
                WebSocketMessage(type=MessageType.PONG),
            )
            return

        # Call registered handlers
        handlers = self._handlers.get(message.type, [])
        for handler in handlers:
            try:
                await handler(connection, message)
            except Exception as e:
                logger.error(f"Handler error: {e}")

    async def _send_message(
        self,
        connection: WebSocketConnection,
        message: WebSocketMessage,
    ) -> None:
        """Send a message."""
        websocket = connection.websocket

        if websocket.client_state != WebSocketState.CONNECTED:
            return

        json_data = message.to_json()
        connection.bytes_sent += len(json_data)

        await websocket.send_text(json_data)

    async def _send_binary(
        self,
        connection: WebSocketConnection,
        data: bytes,
    ) -> None:
        """Send binary data."""
        websocket = connection.websocket

        if websocket.client_state != WebSocketState.CONNECTED:
            return

        connection.bytes_sent += len(data)
        await websocket.send_bytes(data)

    async def _send_error(
        self,
        connection: WebSocketConnection,
        error: str,
        code: Optional[str] = None,
    ) -> None:
        """Send error message."""
        await self._send_message(
            connection,
            WebSocketMessage(
                type=MessageType.ERROR,
                data={"error": error, "code": code},
            ),
        )

    async def _ping_loop(self, connection: WebSocketConnection) -> None:
        """Send periodic pings to keep connection alive."""
        while connection.state not in [ConnectionState.CLOSING, ConnectionState.CLOSED]:
            await asyncio.sleep(self.ping_interval)

            try:
                await self._send_message(
                    connection,
                    WebSocketMessage(type=MessageType.PING),
                )
            except Exception as e:
                logger.warning(f"Ping failed: {e}")
                break

    # Public methods for sending messages

    async def send(
        self,
        connection: WebSocketConnection,
        message_type: MessageType,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a message to connection."""
        await self._send_message(
            connection,
            WebSocketMessage(type=message_type, data=data or {}),
        )

    async def send_audio(
        self,
        connection: WebSocketConnection,
        audio_data: bytes,
    ) -> None:
        """Send audio data to connection."""
        await self._send_binary(connection, audio_data)

    async def send_transcript(
        self,
        connection: WebSocketConnection,
        text: str,
        is_final: bool = False,
    ) -> None:
        """Send transcript to connection."""
        message_type = (
            MessageType.TRANSCRIPT_FINAL if is_final
            else MessageType.TRANSCRIPT_PARTIAL
        )
        await self.send(
            connection,
            message_type,
            {"text": text, "is_final": is_final},
        )

    async def send_agent_response(
        self,
        connection: WebSocketConnection,
        text: str,
        audio: Optional[bytes] = None,
    ) -> None:
        """Send agent response to connection."""
        await self.send(
            connection,
            MessageType.AGENT_RESPONSE,
            {"text": text, "has_audio": audio is not None},
        )
        if audio:
            await self.send_audio(connection, audio)

    async def close(
        self,
        connection: WebSocketConnection,
        code: int = 1000,
        reason: str = "Normal closure",
    ) -> None:
        """Close a connection."""
        connection.state = ConnectionState.CLOSING
        try:
            await connection.websocket.close(code=code, reason=reason)
        except Exception as e:
            logger.warning(f"Close error: {e}")
        connection.state = ConnectionState.CLOSED
