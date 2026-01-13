"""
Builder Engine Voice Pipeline

A high-performance, real-time voice processing engine that provides:
- Multi-provider Speech-to-Text (STT) integration
- Multi-provider Text-to-Speech (TTS) integration
- Voice Activity Detection (VAD)
- Audio streaming and buffering
- Interruption handling
- Turn-taking management

Designed for ultra-low latency conversational AI applications.
"""

from .audio import (
    AudioFormat,
    AudioChunk,
    AudioBuffer,
    AudioResampler,
    AudioEncoder,
    AudioDecoder,
)
from .vad import (
    VoiceActivityDetector,
    VADConfig,
    VADEvent,
    VADState,
    SileroVAD,
    WebRTCVAD,
    EnergyVAD,
)
from .stt import (
    STTProvider,
    STTConfig,
    STTResult,
    TranscriptionSegment,
    DeepgramSTT,
    WhisperSTT,
    GoogleSTT,
    AzureSTT,
    AssemblyAISTT,
    STTProviderFactory,
)
from .tts import (
    TTSProvider,
    TTSConfig,
    TTSVoice,
    ElevenLabsTTS,
    PlayHTTTS,
    AzureTTS,
    GoogleTTS,
    OpenAITTS,
    TTSProviderFactory,
)
from .pipeline import (
    VoicePipeline,
    VoicePipelineConfig,
    VoicePipelineState,
    PipelineEvent,
    PipelineEventType,
    ConversationTurn,
    ConversationContext,
    ResponseGenerator,
    SimpleResponseGenerator,
    create_pipeline,
    create_test_pipeline,
)
from .interruption import (
    InterruptionHandler,
    InterruptionConfig,
    InterruptionEvent,
    InterruptionStrategy,
)

__all__ = [
    # Audio
    "AudioFormat",
    "AudioChunk",
    "AudioBuffer",
    "AudioResampler",
    "AudioEncoder",
    "AudioDecoder",
    # VAD
    "VoiceActivityDetector",
    "VADConfig",
    "VADEvent",
    "VADState",
    "SileroVAD",
    "WebRTCVAD",
    "EnergyVAD",
    # STT
    "STTProvider",
    "STTConfig",
    "STTResult",
    "TranscriptionSegment",
    "DeepgramSTT",
    "WhisperSTT",
    "GoogleSTT",
    "AzureSTT",
    "AssemblyAISTT",
    "STTProviderFactory",
    # TTS
    "TTSProvider",
    "TTSConfig",
    "TTSVoice",
    "ElevenLabsTTS",
    "PlayHTTTS",
    "AzureTTS",
    "GoogleTTS",
    "OpenAITTS",
    "TTSProviderFactory",
    # Pipeline
    "VoicePipeline",
    "VoicePipelineConfig",
    "VoicePipelineState",
    "PipelineEvent",
    "PipelineEventType",
    "ConversationTurn",
    "ConversationContext",
    "ResponseGenerator",
    "SimpleResponseGenerator",
    "create_pipeline",
    "create_test_pipeline",
    # Interruption
    "InterruptionHandler",
    "InterruptionConfig",
    "InterruptionEvent",
    "InterruptionStrategy",
]
