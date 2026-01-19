"""
Configuration for Streaming Orchestrator Service.

This module defines all configuration options for the ultra-low latency
streaming pipeline, including provider settings, latency targets, and
optimization parameters.
"""

from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LatencyProfile(str, Enum):
    """Predefined latency optimization profiles."""

    ULTRA_LOW = "ultra_low"      # <200ms target, aggressive optimization
    LOW = "low"                   # <300ms target, balanced optimization
    BALANCED = "balanced"         # <500ms target, quality-focused
    HIGH_QUALITY = "high_quality" # <800ms target, maximum quality


class StreamingMode(str, Enum):
    """Streaming pipeline modes."""

    FULL_DUPLEX = "full_duplex"           # Simultaneous send/receive
    HALF_DUPLEX = "half_duplex"           # Turn-based with interruption
    PUSH_TO_TALK = "push_to_talk"         # Manual speech start/end
    CONTINUOUS = "continuous"              # Always-on listening


class ASRProvider(str, Enum):
    """Supported ASR providers for streaming."""

    DEEPGRAM = "deepgram"
    ASSEMBLY_AI = "assemblyai"
    GOOGLE = "google"
    AZURE = "azure"
    WHISPER_STREAMING = "whisper_streaming"
    GROQ_WHISPER = "groq_whisper"


class LLMProvider(str, Enum):
    """Supported LLM providers for streaming."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    CEREBRAS = "cerebras"


class TTSProvider(str, Enum):
    """Supported TTS providers for streaming."""

    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    CARTESIA = "cartesia"
    PLAYHT = "playht"
    DEEPGRAM = "deepgram"
    AZURE = "azure"
    RIME = "rime"


class LatencyTargets(BaseSettings):
    """Latency target configuration."""

    model_config = SettingsConfigDict(env_prefix="LATENCY_")

    # End-to-end targets (from speech end to first audio byte)
    e2e_target_ms: int = Field(
        default=300,
        ge=100,
        le=2000,
        description="Target end-to-end latency in milliseconds",
    )
    e2e_max_ms: int = Field(
        default=500,
        ge=200,
        le=5000,
        description="Maximum acceptable end-to-end latency",
    )

    # Component targets
    asr_target_ms: int = Field(
        default=100,
        description="Target ASR streaming latency",
    )
    llm_first_token_target_ms: int = Field(
        default=150,
        description="Target time to first LLM token",
    )
    tts_first_audio_target_ms: int = Field(
        default=75,
        description="Target time to first TTS audio byte",
    )

    # Monitoring thresholds
    warning_threshold_multiplier: float = Field(
        default=1.5,
        description="Multiplier for warning threshold",
    )
    critical_threshold_multiplier: float = Field(
        default=2.0,
        description="Multiplier for critical threshold",
    )


class SpeculativeExecutionConfig(BaseSettings):
    """Configuration for speculative LLM execution."""

    model_config = SettingsConfigDict(env_prefix="SPECULATIVE_")

    enabled: bool = Field(
        default=True,
        description="Enable speculative execution on partial transcripts",
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum transcript confidence to trigger speculation",
    )
    min_words: int = Field(
        default=3,
        ge=1,
        description="Minimum words in partial transcript to speculate",
    )
    stability_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Stability score threshold for speculation",
    )
    max_speculative_tokens: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximum tokens to generate speculatively",
    )
    abandon_on_transcript_change: bool = Field(
        default=True,
        description="Abandon speculative results if transcript changes significantly",
    )
    transcript_change_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Edit distance ratio to trigger abandonment",
    )


class AudioBufferConfig(BaseSettings):
    """Configuration for audio buffering and jitter handling."""

    model_config = SettingsConfigDict(env_prefix="AUDIO_BUFFER_")

    # Input buffering
    input_buffer_ms: int = Field(
        default=100,
        ge=20,
        le=500,
        description="Input audio buffer size in milliseconds",
    )
    input_chunk_ms: int = Field(
        default=20,
        ge=10,
        le=100,
        description="Input audio chunk size for processing",
    )

    # Output buffering
    output_buffer_ms: int = Field(
        default=150,
        ge=50,
        le=500,
        description="Output audio buffer for smooth playback",
    )
    output_chunk_ms: int = Field(
        default=20,
        ge=10,
        le=100,
        description="Output audio chunk size",
    )

    # Jitter compensation
    jitter_buffer_enabled: bool = Field(
        default=True,
        description="Enable adaptive jitter buffer",
    )
    jitter_buffer_min_ms: int = Field(
        default=20,
        description="Minimum jitter buffer size",
    )
    jitter_buffer_max_ms: int = Field(
        default=200,
        description="Maximum jitter buffer size",
    )
    jitter_adaptation_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Rate of jitter buffer adaptation",
    )


class CircuitBreakerConfig(BaseSettings):
    """Configuration for circuit breaker patterns."""

    model_config = SettingsConfigDict(env_prefix="CIRCUIT_BREAKER_")

    enabled: bool = Field(
        default=True,
        description="Enable circuit breaker for external services",
    )
    failure_threshold: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Failures before opening circuit",
    )
    recovery_timeout_s: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Seconds before attempting recovery",
    )
    half_open_max_calls: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max calls in half-open state",
    )
    success_threshold: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Successes needed to close circuit",
    )


class ProviderConfig(BaseSettings):
    """Provider-specific configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ASR Providers
    deepgram_api_key: str = Field(default="", description="Deepgram API key")
    assemblyai_api_key: str = Field(default="", description="AssemblyAI API key")
    google_credentials_path: str = Field(default="", description="Google credentials path")
    azure_speech_key: str = Field(default="", description="Azure Speech key")
    azure_speech_region: str = Field(default="eastus", description="Azure region")

    # LLM Providers
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    together_api_key: str = Field(default="", description="Together AI API key")
    fireworks_api_key: str = Field(default="", description="Fireworks AI API key")
    cerebras_api_key: str = Field(default="", description="Cerebras API key")

    # TTS Providers
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key")
    cartesia_api_key: str = Field(default="", description="Cartesia API key")
    playht_api_key: str = Field(default="", description="PlayHT API key")
    playht_user_id: str = Field(default="", description="PlayHT user ID")
    rime_api_key: str = Field(default="", description="Rime API key")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Service configuration
    service_name: str = Field(
        default="streaming-orchestrator",
        description="Service name for identification",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind to",
    )
    port: int = Field(
        default=8088,
        ge=1024,
        le=65535,
        description="Port to listen on",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    log_level: str = Field(
        default="info",
        description="Logging level",
    )

    # Pipeline configuration
    latency_profile: LatencyProfile = Field(
        default=LatencyProfile.LOW,
        description="Latency optimization profile",
    )
    streaming_mode: StreamingMode = Field(
        default=StreamingMode.HALF_DUPLEX,
        description="Streaming pipeline mode",
    )

    # Provider selection
    default_asr_provider: ASRProvider = Field(
        default=ASRProvider.DEEPGRAM,
        description="Default ASR provider",
    )
    default_llm_provider: LLMProvider = Field(
        default=LLMProvider.GROQ,
        description="Default LLM provider (Groq for lowest latency)",
    )
    default_llm_model: str = Field(
        default="llama-3.1-70b-versatile",
        description="Default LLM model",
    )
    default_tts_provider: TTSProvider = Field(
        default=TTSProvider.CARTESIA,
        description="Default TTS provider (Cartesia for lowest latency)",
    )
    default_tts_voice: str = Field(
        default="",
        description="Default TTS voice ID",
    )

    # Fallback providers (for resilience)
    fallback_asr_provider: Optional[ASRProvider] = Field(
        default=ASRProvider.GOOGLE,
        description="Fallback ASR provider",
    )
    fallback_llm_provider: Optional[LLMProvider] = Field(
        default=LLMProvider.OPENAI,
        description="Fallback LLM provider",
    )
    fallback_tts_provider: Optional[TTSProvider] = Field(
        default=TTSProvider.ELEVENLABS,
        description="Fallback TTS provider",
    )

    # Concurrency limits
    max_concurrent_sessions: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Maximum concurrent streaming sessions",
    )
    max_concurrent_per_tenant: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum concurrent sessions per tenant",
    )

    # Internal service URLs
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    platform_api_url: str = Field(
        default="http://localhost:8086",
        description="Platform API URL",
    )

    # Metrics and monitoring
    metrics_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )
    tracing_enabled: bool = Field(
        default=True,
        description="Enable distributed tracing",
    )
    latency_histogram_buckets: List[float] = Field(
        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0],
        description="Histogram buckets for latency metrics (seconds)",
    )

    # Sub-configurations
    latency_targets: LatencyTargets = Field(
        default_factory=LatencyTargets,
        description="Latency target configuration",
    )
    speculative_execution: SpeculativeExecutionConfig = Field(
        default_factory=SpeculativeExecutionConfig,
        description="Speculative execution configuration",
    )
    audio_buffer: AudioBufferConfig = Field(
        default_factory=AudioBufferConfig,
        description="Audio buffer configuration",
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration",
    )
    providers: ProviderConfig = Field(
        default_factory=ProviderConfig,
        description="Provider credentials and configuration",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.lower()

    def get_provider_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        key_map = {
            "deepgram": self.providers.deepgram_api_key,
            "assemblyai": self.providers.assemblyai_api_key,
            "openai": self.providers.openai_api_key,
            "anthropic": self.providers.anthropic_api_key,
            "groq": self.providers.groq_api_key,
            "together": self.providers.together_api_key,
            "fireworks": self.providers.fireworks_api_key,
            "cerebras": self.providers.cerebras_api_key,
            "elevenlabs": self.providers.elevenlabs_api_key,
            "cartesia": self.providers.cartesia_api_key,
            "playht": self.providers.playht_api_key,
            "rime": self.providers.rime_api_key,
        }
        return key_map.get(provider.lower())

    def apply_latency_profile(self) -> None:
        """Apply settings based on latency profile."""
        profiles = {
            LatencyProfile.ULTRA_LOW: {
                "e2e_target_ms": 200,
                "asr_target_ms": 75,
                "llm_first_token_target_ms": 100,
                "tts_first_audio_target_ms": 50,
                "input_buffer_ms": 60,
                "output_buffer_ms": 100,
            },
            LatencyProfile.LOW: {
                "e2e_target_ms": 300,
                "asr_target_ms": 100,
                "llm_first_token_target_ms": 150,
                "tts_first_audio_target_ms": 75,
                "input_buffer_ms": 100,
                "output_buffer_ms": 150,
            },
            LatencyProfile.BALANCED: {
                "e2e_target_ms": 500,
                "asr_target_ms": 150,
                "llm_first_token_target_ms": 200,
                "tts_first_audio_target_ms": 100,
                "input_buffer_ms": 150,
                "output_buffer_ms": 200,
            },
            LatencyProfile.HIGH_QUALITY: {
                "e2e_target_ms": 800,
                "asr_target_ms": 250,
                "llm_first_token_target_ms": 300,
                "tts_first_audio_target_ms": 150,
                "input_buffer_ms": 200,
                "output_buffer_ms": 300,
            },
        }

        if self.latency_profile in profiles:
            profile = profiles[self.latency_profile]
            self.latency_targets.e2e_target_ms = profile["e2e_target_ms"]
            self.latency_targets.asr_target_ms = profile["asr_target_ms"]
            self.latency_targets.llm_first_token_target_ms = profile["llm_first_token_target_ms"]
            self.latency_targets.tts_first_audio_target_ms = profile["tts_first_audio_target_ms"]
            self.audio_buffer.input_buffer_ms = profile["input_buffer_ms"]
            self.audio_buffer.output_buffer_ms = profile["output_buffer_ms"]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.apply_latency_profile()
    return settings


# Export settings instance
settings = get_settings()
