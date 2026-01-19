"""Configuration for ASR Service."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Service settings
    service_name: str = "asr-service"
    host: str = "0.0.0.0"
    port: int = 8082
    debug: bool = False
    log_level: str = "info"

    # ASR Provider settings
    asr_provider: Literal["deepgram", "whisper", "mock"] = "deepgram"

    # Deepgram settings
    deepgram_api_key: str = Field(default="", description="Deepgram API key")
    deepgram_model: str = "nova-2"  # nova-2 is fastest and most accurate
    deepgram_language: str = "en-US"
    deepgram_smart_format: bool = True
    deepgram_punctuate: bool = True
    deepgram_profanity_filter: bool = False
    deepgram_diarize: bool = False
    deepgram_filler_words: bool = False
    deepgram_interim_results: bool = True
    deepgram_utterance_end_ms: int = 1000  # Silence to end utterance
    deepgram_vad_events: bool = True  # Voice activity detection events
    deepgram_endpointing: int = 300  # End of speech detection (ms)

    # Whisper settings (for self-hosted)
    whisper_model: str = "base.en"  # tiny, base, small, medium, large
    whisper_device: str = "cpu"  # cpu or cuda
    whisper_compute_type: str = "int8"  # int8, float16, float32

    # Audio settings
    sample_rate: int = 16000  # Target sample rate
    input_sample_rate: int = 8000  # Twilio sends 8kHz
    channels: int = 1
    encoding: str = "linear16"  # linear16, mulaw, opus

    # VAD settings (Voice Activity Detection)
    vad_enabled: bool = True
    vad_threshold: float = 0.5  # Speech probability threshold
    vad_min_speech_ms: int = 250  # Minimum speech duration
    vad_min_silence_ms: int = 300  # Minimum silence to end speech
    vad_padding_ms: int = 300  # Padding around speech

    # Streaming settings
    max_connections: int = 100
    connection_timeout: int = 300  # 5 minutes
    audio_chunk_ms: int = 20  # Audio chunk size in milliseconds

    # Redis settings (for caching/pub-sub)
    redis_url: str = "redis://localhost:6379/0"

    # Media Pipeline settings
    media_pipeline_url: str = "localhost:8081"
    media_pipeline_grpc: str = "localhost:50051"

    # Conversation Engine settings
    conversation_engine_url: str = "http://localhost:8084"

    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9092

    @property
    def deepgram_options(self) -> dict:
        """Get Deepgram transcription options."""
        return {
            "model": self.deepgram_model,
            "language": self.deepgram_language,
            "smart_format": self.deepgram_smart_format,
            "punctuate": self.deepgram_punctuate,
            "profanity_filter": self.deepgram_profanity_filter,
            "diarize": self.deepgram_diarize,
            "filler_words": self.deepgram_filler_words,
            "interim_results": self.deepgram_interim_results,
            "utterance_end_ms": self.deepgram_utterance_end_ms,
            "vad_events": self.deepgram_vad_events,
            "endpointing": self.deepgram_endpointing,
            "encoding": "mulaw" if self.input_sample_rate == 8000 else "linear16",
            "sample_rate": self.input_sample_rate,
            "channels": self.channels,
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
