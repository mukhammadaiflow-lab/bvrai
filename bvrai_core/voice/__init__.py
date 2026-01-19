"""
BVRAI Voice Engine

Production-ready voice processing module with multi-provider STT/TTS support,
automatic failover, health monitoring, and real-time streaming.

Quick Start:
    from bvrai_core.voice import get_voice_engine, STTConfig, TTSConfig

    # Initialize engine
    engine = await initialize_voice_engine()

    # Transcribe audio
    result = await engine.transcribe(audio_data)
    print(result.text)

    # Synthesize speech
    result = await engine.synthesize("Hello, world!")
    audio_bytes = result.audio_data

Features:
    - Multi-provider support (Deepgram, ElevenLabs, OpenAI, AssemblyAI, PlayHT)
    - Automatic failover with configurable fallback chains
    - Real-time streaming for low-latency conversational AI
    - Circuit breaker pattern for provider health management
    - Comprehensive metrics and logging
    - Response caching for TTS
    - Voice profiles for agent customization
"""

from .base import (
    # Enums
    AudioFormat,
    SampleRate,
    STTProvider,
    TTSProvider,
    TranscriptionStatus,
    SynthesisStatus,
    VoiceGender,
    VoiceStyle,
    # Configuration
    AudioConfig,
    STTConfig,
    TTSConfig,
    VoiceProfile,
    # Results
    TranscriptionWord,
    TranscriptionResult,
    SynthesisResult,
    AudioChunk,
    # Health & Metrics
    ProviderHealth,
    VoiceMetrics,
    # Interfaces
    STTProviderInterface,
    TTSProviderInterface,
    # Exceptions
    VoiceError,
    STTError,
    TTSError,
    ProviderConnectionError,
    ProviderTimeoutError,
    ProviderRateLimitError,
    InvalidAudioError,
    UnsupportedLanguageError,
    VoiceNotFoundError,
    QuotaExceededError,
    # Utilities
    generate_request_id,
    calculate_audio_hash,
    estimate_audio_duration,
    format_duration,
)

from .engine import (
    VoiceEngine,
    VoiceEngineConfig,
    get_voice_engine,
    initialize_voice_engine,
)

from .stt_providers import (
    DeepgramSTTProvider,
    AssemblyAISTTProvider,
    OpenAIWhisperSTTProvider,
    MockSTTProvider,
    create_stt_provider,
)

from .tts_providers import (
    ElevenLabsTTSProvider,
    OpenAITTSProvider,
    PlayHTTTSProvider,
    MockTTSProvider,
    create_tts_provider,
)

__all__ = [
    # Enums
    "AudioFormat",
    "SampleRate",
    "STTProvider",
    "TTSProvider",
    "TranscriptionStatus",
    "SynthesisStatus",
    "VoiceGender",
    "VoiceStyle",
    # Configuration
    "AudioConfig",
    "STTConfig",
    "TTSConfig",
    "VoiceProfile",
    "VoiceEngineConfig",
    # Results
    "TranscriptionWord",
    "TranscriptionResult",
    "SynthesisResult",
    "AudioChunk",
    # Health & Metrics
    "ProviderHealth",
    "VoiceMetrics",
    # Interfaces
    "STTProviderInterface",
    "TTSProviderInterface",
    # Engine
    "VoiceEngine",
    "get_voice_engine",
    "initialize_voice_engine",
    # STT Providers
    "DeepgramSTTProvider",
    "AssemblyAISTTProvider",
    "OpenAIWhisperSTTProvider",
    "MockSTTProvider",
    "create_stt_provider",
    # TTS Providers
    "ElevenLabsTTSProvider",
    "OpenAITTSProvider",
    "PlayHTTTSProvider",
    "MockTTSProvider",
    "create_tts_provider",
    # Exceptions
    "VoiceError",
    "STTError",
    "TTSError",
    "ProviderConnectionError",
    "ProviderTimeoutError",
    "ProviderRateLimitError",
    "InvalidAudioError",
    "UnsupportedLanguageError",
    "VoiceNotFoundError",
    "QuotaExceededError",
    # Utilities
    "generate_request_id",
    "calculate_audio_hash",
    "estimate_audio_duration",
    "format_duration",
]

__version__ = "1.0.0"
