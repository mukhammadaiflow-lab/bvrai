"""WebRTC Gateway configuration."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8087
    debug: bool = False
    log_level: str = "info"

    # Redis for session management
    redis_url: str = "redis://localhost:6379/0"

    # Service URLs
    conversation_engine_url: str = "http://localhost:8084"
    platform_api_url: str = "http://localhost:8086"

    # TURN/STUN Configuration
    turn_enabled: bool = True
    turn_url: str = "turn:localhost:3478"
    turn_secret: str = Field(default="dev_turn_secret")
    stun_url: str = "stun:stun.l.google.com:19302"

    # ICE Configuration
    ice_candidate_pool_size: int = 10
    ice_gathering_timeout: int = 5000  # ms

    # Audio Configuration
    audio_sample_rate: int = 16000  # 16kHz for speech
    audio_channels: int = 1  # Mono
    audio_codec: str = "opus"  # Opus codec for WebRTC

    # Session Configuration
    session_timeout: int = 86400  # 24 hours (was 1 hour - too aggressive for conversation data)
    max_concurrent_sessions: int = 1000

    # WebSocket Configuration
    ws_heartbeat_interval: int = 30
    ws_max_message_size: int = 65536  # 64KB


@lru_cache
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()
