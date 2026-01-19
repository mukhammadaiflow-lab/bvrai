"""Configuration for Platform API."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Service settings
    service_name: str = "platform-api"
    host: str = "0.0.0.0"
    port: int = 8086
    debug: bool = False
    log_level: str = "info"

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://bvrai:devpassword@localhost:5432/bvrai",
        description="PostgreSQL connection URL",
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # JWT Authentication
    jwt_secret: str = Field(
        default="dev_secret_key_change_in_production",
        description="JWT signing secret",
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24 hours

    # API Keys
    api_key_header: str = "X-API-Key"

    # Service URLs
    telephony_gateway_url: str = "http://localhost:8080"
    conversation_engine_url: str = "http://localhost:8084"
    ai_orchestrator_url: str = "http://localhost:8085"

    # Twilio Configuration
    twilio_account_sid: str = Field(
        default="",
        description="Twilio Account SID",
    )
    twilio_auth_token: str = Field(
        default="",
        description="Twilio Auth Token",
    )
    twilio_api_key_sid: str = Field(
        default="",
        description="Twilio API Key SID (alternative to Account SID/Auth Token)",
    )
    twilio_api_key_secret: str = Field(
        default="",
        description="Twilio API Key Secret",
    )
    twilio_webhook_base_url: str = Field(
        default="",
        description="Base URL for Twilio webhooks (e.g., https://your-domain.com)",
    )
    twilio_default_caller_id: str = Field(
        default="",
        description="Default Twilio phone number for outbound calls",
    )
    twilio_status_callback_url: str = Field(
        default="",
        description="URL for Twilio status callbacks",
    )
    twilio_recording_enabled: bool = Field(
        default=True,
        description="Enable call recording by default",
    )
    twilio_transcription_enabled: bool = Field(
        default=False,
        description="Enable call transcription",
    )

    # Pagination
    default_page_size: int = 20
    max_page_size: int = 100


@lru_cache
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()
