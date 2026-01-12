"""Configuration for TTS Service."""

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
    service_name: str = "tts-service"
    host: str = "0.0.0.0"
    port: int = 8083
    debug: bool = False
    log_level: str = "info"

    # TTS Provider settings
    tts_provider: Literal["elevenlabs", "playht", "mock"] = "elevenlabs"

    # ElevenLabs settings
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key")
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel - default voice
    elevenlabs_model_id: str = "eleven_turbo_v2"  # Fastest model
    elevenlabs_stability: float = 0.5
    elevenlabs_similarity_boost: float = 0.75
    elevenlabs_style: float = 0.0
    elevenlabs_use_speaker_boost: bool = True

    # PlayHT settings (alternative)
    playht_api_key: str = ""
    playht_user_id: str = ""
    playht_voice: str = "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json"

    # Audio output settings
    output_format: str = "mp3_44100_128"  # For ElevenLabs
    output_sample_rate: int = 8000  # For Twilio (8kHz mulaw)
    output_encoding: str = "mulaw"  # For Twilio compatibility

    # Streaming settings
    chunk_size: int = 1024  # Bytes per chunk
    latency_optimization: int = 3  # 1-4, higher = lower latency but lower quality

    # Caching
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour

    # Rate limiting
    max_concurrent_requests: int = 50
    max_text_length: int = 5000

    # Voice presets
    voices: dict = {
        "default": "21m00Tcm4TlvDq8ikWAM",  # Rachel
        "male": "VR6AewLTigWG4xSOukaG",  # Arnold
        "female": "EXAVITQu4vr4xnSDxMaL",  # Bella
        "professional": "ThT5KcBeYPX3keUQqHPh",  # Dorothy
        "friendly": "pNInz6obpgDQGcFmaJgB",  # Adam
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
