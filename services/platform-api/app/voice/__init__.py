"""
Voice Engine System

Enterprise voice processing with:
- WebRTC support for browser-based calls
- SIP integration for telephony
- Real-time audio streaming
- Voice activity detection
- Audio codec handling
- Media server integration
"""

from app.voice.engine import (
    VoiceEngine,
    VoiceEngineConfig,
    AudioStream,
    StreamState,
    VoiceSession,
    SessionManager,
)

from app.voice.webrtc import (
    WebRTCHandler,
    WebRTCConfig,
    PeerConnection,
    MediaTrack,
    ICECandidate,
    SDPOffer,
    SDPAnswer,
    SignalingServer,
)

from app.voice.sip import (
    SIPClient,
    SIPConfig,
    SIPSession,
    SIPMessage,
    SIPDialog,
    SIPRegistrar,
    SIPProxy,
)

from app.voice.audio import (
    AudioProcessor,
    AudioBuffer,
    AudioCodec,
    CodecType,
    VADDetector,
    NoiseReducer,
    AudioMixer,
    AudioResampler,
)

from app.voice.streaming import (
    StreamingPipeline,
    StreamProcessor,
    AudioChunk,
    StreamConfig,
    BiDirectionalStream,
    StreamMultiplexer,
)

__all__ = [
    # Engine
    "VoiceEngine",
    "VoiceEngineConfig",
    "AudioStream",
    "StreamState",
    "VoiceSession",
    "SessionManager",
    # WebRTC
    "WebRTCHandler",
    "WebRTCConfig",
    "PeerConnection",
    "MediaTrack",
    "ICECandidate",
    "SDPOffer",
    "SDPAnswer",
    "SignalingServer",
    # SIP
    "SIPClient",
    "SIPConfig",
    "SIPSession",
    "SIPMessage",
    "SIPDialog",
    "SIPRegistrar",
    "SIPProxy",
    # Audio
    "AudioProcessor",
    "AudioBuffer",
    "AudioCodec",
    "CodecType",
    "VADDetector",
    "NoiseReducer",
    "AudioMixer",
    "AudioResampler",
    # Streaming
    "StreamingPipeline",
    "StreamProcessor",
    "AudioChunk",
    "StreamConfig",
    "BiDirectionalStream",
    "StreamMultiplexer",
]
