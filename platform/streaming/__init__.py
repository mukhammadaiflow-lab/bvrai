"""
Real-time Streaming Engine
==========================

High-performance streaming infrastructure for real-time voice processing,
WebSocket communication, and audio stream handling.

Author: Platform Engineering Team
Version: 2.0.0
"""

from platform.streaming.engine import (
    StreamingEngine,
    StreamSession,
    StreamConfig,
    StreamState,
    StreamMetrics,
)
from platform.streaming.audio import (
    AudioProcessor,
    AudioBuffer,
    AudioChunk,
    AudioFormat,
    AudioCodec,
    VoiceActivityDetector,
    AudioResampler,
)
from platform.streaming.websocket import (
    WebSocketManager,
    WebSocketSession,
    WebSocketMessage,
    MessageType,
)
from platform.streaming.protocols import (
    StreamProtocol,
    RTCProtocol,
    SIPBridge,
    TwilioMediaStream,
)

__all__ = [
    # Engine
    "StreamingEngine",
    "StreamSession",
    "StreamConfig",
    "StreamState",
    "StreamMetrics",
    # Audio
    "AudioProcessor",
    "AudioBuffer",
    "AudioChunk",
    "AudioFormat",
    "AudioCodec",
    "VoiceActivityDetector",
    "AudioResampler",
    # WebSocket
    "WebSocketManager",
    "WebSocketSession",
    "WebSocketMessage",
    "MessageType",
    # Protocols
    "StreamProtocol",
    "RTCProtocol",
    "SIPBridge",
    "TwilioMediaStream",
]
