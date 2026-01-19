"""
Configuration for Voice Lab Service.
"""

from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VoiceProvider(str, Enum):
    """Supported voice cloning providers."""

    ELEVENLABS = "elevenlabs"
    PLAYHT = "playht"
    CARTESIA = "cartesia"
    RESEMBLE = "resemble"
    AZURE = "azure"
    INTERNAL = "internal"  # Future: self-hosted model


class VoiceQuality(str, Enum):
    """Voice quality tiers."""

    INSTANT = "instant"         # Quick clone, 15-30s audio
    STANDARD = "standard"       # Better quality, 1-3 min audio
    PROFESSIONAL = "professional"  # Studio quality, 5+ min audio
    ULTRA = "ultra"             # Highest quality, extensive training


class VoiceGender(str, Enum):
    """Voice gender classification."""

    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceAge(str, Enum):
    """Voice age classification."""

    CHILD = "child"
    YOUNG = "young"
    MIDDLE = "middle"
    SENIOR = "senior"


class ConsentStatus(str, Enum):
    """Voice consent status for compliance."""

    PENDING = "pending"
    VERIFIED = "verified"
    EXPIRED = "expired"
    REVOKED = "revoked"


class StorageConfig(BaseSettings):
    """Storage configuration for voice files."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_")

    provider: str = Field(default="s3", description="Storage provider (s3, gcs, azure, local)")
    bucket_name: str = Field(default="voice-lab-samples", description="Storage bucket name")
    region: str = Field(default="us-east-1", description="Storage region")

    # S3 specific
    aws_access_key_id: str = Field(default="", description="AWS access key")
    aws_secret_access_key: str = Field(default="", description="AWS secret key")

    # GCS specific
    gcs_credentials_path: str = Field(default="", description="GCS credentials path")

    # Local storage
    local_path: str = Field(default="/data/voices", description="Local storage path")

    # Retention
    sample_retention_days: int = Field(default=365, description="Days to retain samples")
    temp_retention_hours: int = Field(default=24, description="Hours to retain temp files")


class ProviderCredentials(BaseSettings):
    """API credentials for voice providers."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ElevenLabs
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs API key")
    elevenlabs_model_id: str = Field(
        default="eleven_multilingual_v2",
        description="Default ElevenLabs model",
    )

    # PlayHT
    playht_api_key: str = Field(default="", description="PlayHT API key")
    playht_user_id: str = Field(default="", description="PlayHT user ID")

    # Cartesia
    cartesia_api_key: str = Field(default="", description="Cartesia API key")

    # Resemble
    resemble_api_key: str = Field(default="", description="Resemble API key")

    # Azure
    azure_speech_key: str = Field(default="", description="Azure Speech key")
    azure_speech_region: str = Field(default="eastus", description="Azure region")


class AudioRequirements(BaseSettings):
    """Audio quality requirements for voice cloning."""

    model_config = SettingsConfigDict(env_prefix="AUDIO_")

    # Duration requirements (seconds)
    min_instant_duration_s: float = Field(default=15.0, description="Min for instant clone")
    min_standard_duration_s: float = Field(default=60.0, description="Min for standard clone")
    min_professional_duration_s: float = Field(default=300.0, description="Min for professional")

    # Quality requirements
    min_sample_rate: int = Field(default=16000, description="Minimum sample rate Hz")
    preferred_sample_rate: int = Field(default=44100, description="Preferred sample rate Hz")
    min_bit_depth: int = Field(default=16, description="Minimum bit depth")

    # Content requirements
    max_silence_ratio: float = Field(default=0.3, description="Max silence in sample")
    min_snr_db: float = Field(default=15.0, description="Minimum signal-to-noise ratio")

    # File limits
    max_file_size_mb: int = Field(default=100, description="Maximum upload file size MB")
    supported_formats: List[str] = Field(
        default=["wav", "mp3", "m4a", "flac", "ogg", "webm"],
        description="Supported audio formats",
    )


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Service configuration
    service_name: str = Field(default="voice-lab", description="Service name")
    host: str = Field(default="0.0.0.0", description="Host to bind")
    port: int = Field(default=8089, ge=1024, le=65535, description="Port to listen on")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="info", description="Logging level")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://bvrai:devpassword@localhost:5432/bvrai",
        description="Database connection URL",
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")

    # Default provider
    default_provider: VoiceProvider = Field(
        default=VoiceProvider.ELEVENLABS,
        description="Default voice cloning provider",
    )

    # Quality defaults
    default_quality: VoiceQuality = Field(
        default=VoiceQuality.STANDARD,
        description="Default voice quality tier",
    )

    # Concurrency
    max_concurrent_clones: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent clone operations",
    )
    max_voices_per_tenant: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum voices per tenant",
    )

    # Preview
    preview_max_length_s: float = Field(
        default=30.0,
        description="Maximum preview audio length in seconds",
    )
    preview_default_text: str = Field(
        default="Hello! This is a preview of your cloned voice. How does it sound?",
        description="Default text for voice preview",
    )

    # Compliance
    require_consent: bool = Field(
        default=True,
        description="Require consent verification for voice cloning",
    )
    consent_expiry_days: int = Field(
        default=365,
        description="Days until consent expires",
    )

    # Sub-configurations
    storage: StorageConfig = Field(default_factory=StorageConfig)
    credentials: ProviderCredentials = Field(default_factory=ProviderCredentials)
    audio: AudioRequirements = Field(default_factory=AudioRequirements)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid log level: {v}")
        return v.lower()

    def get_provider_key(self, provider: VoiceProvider) -> Optional[str]:
        """Get API key for a provider."""
        key_map = {
            VoiceProvider.ELEVENLABS: self.credentials.elevenlabs_api_key,
            VoiceProvider.PLAYHT: self.credentials.playht_api_key,
            VoiceProvider.CARTESIA: self.credentials.cartesia_api_key,
            VoiceProvider.RESEMBLE: self.credentials.resemble_api_key,
            VoiceProvider.AZURE: self.credentials.azure_speech_key,
        }
        return key_map.get(provider)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
