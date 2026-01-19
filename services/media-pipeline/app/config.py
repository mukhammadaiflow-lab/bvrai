"""Media Pipeline configuration."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Media Pipeline settings."""

    # Server
    port: int = 8081
    host: str = "0.0.0.0"
    log_level: str = "info"

    # Audio settings
    sample_rate: int = 8000  # 8kHz for telephony
    sample_rate_webrtc: int = 48000  # 48kHz for WebRTC
    channels: int = 1  # Mono
    bit_depth: int = 16
    frame_duration_ms: int = 20  # 20ms frames

    # Buffer settings
    jitter_buffer_ms: int = 100  # Jitter buffer size
    max_buffer_ms: int = 500  # Maximum buffer before dropping
    min_buffer_ms: int = 40  # Minimum buffer before underrun

    # Codec settings
    default_codec: str = "pcmu"  # μ-law for Twilio
    opus_bitrate: int = 24000  # Opus bitrate for WebRTC

    # Service URLs
    asr_service_url: str = "http://asr-service:8082"
    tts_service_url: str = "http://tts-service:8083"
    conversation_engine_url: str = "http://conversation-engine:8084"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Performance
    max_concurrent_sessions: int = 500
    worker_threads: int = 4

    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9091

    class Config:
        env_prefix = ""
        case_sensitive = False


settings = Settings()
