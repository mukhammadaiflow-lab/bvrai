"""Configuration for Conversation Engine."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Service settings
    service_name: str = "conversation-engine"
    host: str = "0.0.0.0"
    port: int = 8084
    debug: bool = False
    log_level: str = "info"

    # Turn detection
    silence_threshold_ms: int = 700
    min_speech_ms: int = 200
    no_speech_timeout_ms: int = 10000

    # Interrupt handling
    interrupt_enabled: bool = True
    interrupt_min_speech_ms: int = 200
    interrupt_overlap_tolerance_ms: int = 300

    # Conversation settings
    max_turns: int = 100
    max_conversation_duration_sec: int = 3600  # 1 hour
    history_max_messages: int = 20

    # Service URLs
    asr_service_url: str = "http://localhost:8082"
    tts_service_url: str = "http://localhost:8083"
    ai_orchestrator_url: str = "http://localhost:8085"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Greeting
    default_greeting: str = "Hello! How can I help you today?"

    # Error messages
    error_message: str = "I'm sorry, I encountered an error. Please try again."
    timeout_message: str = "I didn't hear anything. Are you still there?"
    goodbye_message: str = "Thank you for calling. Goodbye!"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()
