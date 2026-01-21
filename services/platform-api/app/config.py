"""Configuration for Platform API.

Security Note:
    - All secrets MUST be provided via environment variables
    - No default secrets are provided for production safety
    - Application will fail fast if required secrets are missing in production
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


def _generate_dev_secret() -> str:
    """Generate a random secret for development only."""
    if _is_production():
        raise ValueError(
            "SECURITY ERROR: JWT_SECRET environment variable is required in production. "
            "Please set JWT_SECRET to a secure random value (minimum 32 characters)."
        )
    return f"dev_only_{secrets.token_hex(32)}"


class Settings(BaseSettings):
    """Application settings with secure defaults.

    Security Features:
        - No hardcoded secrets (generated dynamically for dev, required for prod)
        - Fail-fast validation for missing required secrets in production
        - Minimum secret length validation
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Environment
    environment: str = Field(
        default="development",
        description="Environment: development, staging, production",
    )

    # Service settings
    service_name: str = "platform-api"
    host: str = "0.0.0.0"
    port: int = 8086
    debug: bool = False
    log_level: str = "info"

    # Database - Required, no default in production
    database_url: str = Field(
        default="postgresql+asyncpg://bvrai:devpassword@localhost:5432/bvrai",
        description="PostgreSQL connection URL (REQUIRED in production)",
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # JWT Authentication - NEVER store default secrets
    jwt_secret: str = Field(
        default_factory=_generate_dev_secret,
        description="JWT signing secret (REQUIRED - minimum 32 characters)",
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24 hours

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        """Validate JWT secret meets security requirements."""
        if _is_production():
            if not v or v.startswith("dev_only_"):
                raise ValueError(
                    "JWT_SECRET is required in production and must not use development defaults"
                )
            if len(v) < 32:
                raise ValueError(
                    "JWT_SECRET must be at least 32 characters for security"
                )
        return v

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL in production."""
        if _is_production() and "devpassword" in v:
            raise ValueError(
                "DATABASE_URL contains development credentials. "
                "Use proper credentials in production."
            )
        return v

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
