"""
Real-time Communication Module

This module provides WebSocket-based real-time communication,
audio streaming, and event handling for voice agents.
"""

from .base import (
    # Connection types
    Connection,
    ConnectionState,
    ConnectionType,
    ConnectionConfig,
    # Message types
    Message,
    MessageType,
    MessagePriority,
    # Event types
    Event,
    EventType,
    EventPriority,
    # Session
    RealtimeSession,
    SessionState,
)

from .websocket import (
    WebSocketServer,
    WebSocketConnection,
    WebSocketConfig,
    ConnectionManager,
)

from .audio_stream import (
    AudioStream,
    AudioStreamConfig,
    AudioBuffer,
    AudioFormat,
    StreamDirection,
    AudioStreamManager,
    convert_mulaw_to_pcm16,
    convert_pcm16_to_mulaw,
    resample_audio,
)

from .events import (
    EventBus,
    EventHandler,
    EventSubscription,
    create_event_bus,
)

from .protocol import (
    ProtocolVersion,
    FrameType,
    CompressionType,
    ProtocolFrame,
    HandshakeMessage,
    TranscriptMessage,
    AudioMessage,
    AgentStateMessage,
    FunctionCallMessage,
    FunctionResultMessage,
    ProtocolHandler,
    VoiceAgentProtocol,
    MessageRouter,
    create_protocol,
)

from .manager import (
    RealtimeManager,
    RealtimeConfig,
    SessionContext,
    RealtimeSessionManager,
    DefaultProtocolHandler,
    create_realtime_manager,
)


__all__ = [
    # Base types
    "Connection",
    "ConnectionState",
    "ConnectionType",
    "ConnectionConfig",
    "Message",
    "MessageType",
    "MessagePriority",
    "Event",
    "EventType",
    "EventPriority",
    "RealtimeSession",
    "SessionState",
    # WebSocket
    "WebSocketServer",
    "WebSocketConnection",
    "WebSocketConfig",
    "ConnectionManager",
    # Audio
    "AudioStream",
    "AudioStreamConfig",
    "AudioBuffer",
    "AudioFormat",
    "StreamDirection",
    "AudioStreamManager",
    "convert_mulaw_to_pcm16",
    "convert_pcm16_to_mulaw",
    "resample_audio",
    # Events
    "EventBus",
    "EventHandler",
    "EventSubscription",
    "create_event_bus",
    # Protocol
    "ProtocolVersion",
    "FrameType",
    "CompressionType",
    "ProtocolFrame",
    "HandshakeMessage",
    "TranscriptMessage",
    "AudioMessage",
    "AgentStateMessage",
    "FunctionCallMessage",
    "FunctionResultMessage",
    "ProtocolHandler",
    "VoiceAgentProtocol",
    "MessageRouter",
    "create_protocol",
    # Manager
    "RealtimeManager",
    "RealtimeConfig",
    "SessionContext",
    "RealtimeSessionManager",
    "DefaultProtocolHandler",
    "create_realtime_manager",
]
