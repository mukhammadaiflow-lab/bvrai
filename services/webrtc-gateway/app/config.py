"""WebRTC Gateway configuration.

Security Note:
    - TURN secrets MUST be provided via environment variables in production
    - No default secrets are provided for production safety
"""

import os
import secrets
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _is_production() -> bool:
    """Check if running in production environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    return env in ("production", "prod", "staging")


def _generate_dev_turn_secret() -> str:
    """Generate a random TURN secret for development only."""
    if _is_production():
        raise ValueError(
            "SECURITY ERROR: TURN_SECRET environment variable is required in production."
        )
    return f"dev_turn_{secrets.token_hex(16)}"


class Settings(BaseSettings):
    """Application settings with secure defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Environment
    environment: str = Field(default="development", description="Environment")

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
    turn_secret: str = Field(
        default_factory=_generate_dev_turn_secret,
        description="TURN server shared secret (REQUIRED in production)",
    )
    stun_url: str = "stun:stun.l.google.com:19302"

    @field_validator("turn_secret")
    @classmethod
    def validate_turn_secret(cls, v: str) -> str:
        """Validate TURN secret in production."""
        if _is_production():
            if not v or v.startswith("dev_turn_"):
                raise ValueError("TURN_SECRET is required in production")
            if len(v) < 16:
                raise ValueError("TURN_SECRET must be at least 16 characters")
        return v

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
