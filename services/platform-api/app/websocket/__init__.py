"""WebSocket module for real-time communication."""

from app.websocket.handler import (
    WebSocketHandler,
    WebSocketConnection,
    ConnectionState,
    MessageType,
    WebSocketMessage,
)

from app.websocket.manager import (
    ConnectionManager,
    RoomManager,
    BroadcastManager,
    get_connection_manager,
)

from app.websocket.audio import (
    AudioStreamHandler,
    AudioCodec,
    AudioFormat,
    AudioChunk,
    AudioBuffer,
)

from app.websocket.events import (
    WebSocketEvent,
    EventType,
    EventDispatcher,
    EventHandler,
)

__all__ = [
    # Handler
    "WebSocketHandler",
    "WebSocketConnection",
    "ConnectionState",
    "MessageType",
    "WebSocketMessage",
    # Manager
    "ConnectionManager",
    "RoomManager",
    "BroadcastManager",
    "get_connection_manager",
    # Audio
    "AudioStreamHandler",
    "AudioCodec",
    "AudioFormat",
    "AudioChunk",
    "AudioBuffer",
    # Events
    "WebSocketEvent",
    "EventType",
    "EventDispatcher",
    "EventHandler",
]
