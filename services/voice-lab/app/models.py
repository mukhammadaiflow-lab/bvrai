"""
Data Models for Voice Lab Service.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field, field_validator

from .config import (
    VoiceProvider,
    VoiceQuality,
    VoiceGender,
    VoiceAge,
    ConsentStatus,
)


# =============================================================================
# Enums
# =============================================================================


class VoiceStatus(str, Enum):
    """Status of a cloned voice."""

    PENDING = "pending"           # Awaiting processing
    PROCESSING = "processing"     # Clone in progress
    READY = "ready"              # Ready for use
    FAILED = "failed"            # Cloning failed
    DISABLED = "disabled"        # Temporarily disabled
    ARCHIVED = "archived"        # Archived (soft delete)


class CloneJobStatus(str, Enum):
    """Status of a cloning job."""

    QUEUED = "queued"
    ANALYZING = "analyzing"
    UPLOADING = "uploading"
    CLONING = "cloning"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SampleType(str, Enum):
    """Type of audio sample."""

    UPLOADED = "uploaded"         # Direct file upload
    RECORDED = "recorded"         # Recorded in browser
    EXTRACTED = "extracted"       # Extracted from video/call
    GENERATED = "generated"       # Synthetic sample


# =============================================================================
# Voice Style Models
# =============================================================================


class VoiceStyleSettings(BaseModel):
    """Customizable voice style settings."""

    # Speed and pacing
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speaking speed multiplier")
    stability: float = Field(default=0.75, ge=0.0, le=1.0, description="Voice stability (0=varied, 1=stable)")
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0, description="Similarity to original voice")

    # Emotional expression
    expressiveness: float = Field(default=0.5, ge=0.0, le=1.0, description="Emotional expressiveness")
    emotion: Optional[str] = Field(default=None, description="Target emotion (happy, sad, angry, etc.)")

    # Technical adjustments
    pitch_shift: float = Field(default=0.0, ge=-12.0, le=12.0, description="Pitch shift in semitones")
    clarity: float = Field(default=0.5, ge=0.0, le=1.0, description="Clarity enhancement")

    # Provider-specific settings
    provider_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific settings",
    )


class VoiceCharacteristics(BaseModel):
    """Detected voice characteristics."""

    # Demographics
    detected_gender: Optional[VoiceGender] = None
    detected_age: Optional[VoiceAge] = None
    detected_accent: Optional[str] = None
    detected_language: Optional[str] = None

    # Voice qualities
    pitch_mean_hz: Optional[float] = None
    pitch_range_hz: Optional[float] = None
    speaking_rate_wpm: Optional[float] = None

    # Technical metrics
    clarity_score: Optional[float] = None
    naturalness_score: Optional[float] = None

    # Embedding (for similarity search)
    embedding: Optional[List[float]] = None


# =============================================================================
# Consent Models
# =============================================================================


class VoiceConsent(BaseModel):
    """Voice consent record for compliance."""

    consent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    voice_owner_name: str = Field(..., description="Name of the voice owner")
    voice_owner_email: Optional[str] = Field(None, description="Email of voice owner")

    # Consent details
    status: ConsentStatus = Field(default=ConsentStatus.PENDING)
    consent_type: str = Field(default="voice_clone", description="Type of consent")
    purpose: str = Field(default="ai_agent", description="Purpose of voice use")

    # Verification
    verification_method: str = Field(default="email", description="How consent was verified")
    verification_code: Optional[str] = Field(None, description="Verification code sent")
    verified_at: Optional[datetime] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

    # Legal
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    terms_version: str = Field(default="1.0", description="Version of terms accepted")


# =============================================================================
# Audio Sample Models
# =============================================================================


class AudioSampleMetadata(BaseModel):
    """Metadata for an audio sample."""

    # File info
    filename: str
    file_size_bytes: int
    mime_type: str
    format: str  # wav, mp3, etc.

    # Audio properties
    duration_s: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int] = None

    # Quality metrics
    silence_ratio: float = 0.0
    snr_db: Optional[float] = None
    clipping_ratio: float = 0.0

    # Content analysis
    speech_ratio: float = 0.0  # Ratio of speech vs non-speech
    detected_language: Optional[str] = None
    word_count_estimate: Optional[int] = None


class AudioSample(BaseModel):
    """An audio sample for voice cloning."""

    sample_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    voice_id: Optional[str] = None

    # Sample info
    sample_type: SampleType = SampleType.UPLOADED
    metadata: Optional[AudioSampleMetadata] = None

    # Storage
    storage_path: str = ""
    storage_url: Optional[str] = None

    # Status
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    analyzed_at: Optional[datetime] = None
    is_valid: bool = True
    validation_errors: List[str] = Field(default_factory=list)


# =============================================================================
# Voice Models
# =============================================================================


class Voice(BaseModel):
    """A cloned voice profile."""

    # Identifiers
    voice_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    created_by: str  # User ID

    # Basic info
    name: str = Field(..., min_length=1, max_length=100, description="Voice name")
    description: Optional[str] = Field(None, max_length=500, description="Voice description")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    # Voice configuration
    provider: VoiceProvider
    provider_voice_id: Optional[str] = None  # ID in the provider's system
    quality: VoiceQuality = VoiceQuality.STANDARD

    # Style settings
    style: VoiceStyleSettings = Field(default_factory=VoiceStyleSettings)
    characteristics: Optional[VoiceCharacteristics] = None

    # Samples used for cloning
    sample_ids: List[str] = Field(default_factory=list)
    total_sample_duration_s: float = 0.0

    # Status
    status: VoiceStatus = VoiceStatus.PENDING
    status_message: Optional[str] = None

    # Consent
    consent_id: Optional[str] = None
    consent: Optional[VoiceConsent] = None

    # Sharing
    is_public: bool = False
    is_template: bool = False  # Can be used as a starting point

    # Usage stats
    usage_count: int = 0
    last_used_at: Optional[datetime] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    archived_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Clone Job Models
# =============================================================================


class CloneJobProgress(BaseModel):
    """Progress tracking for a clone job."""

    current_step: str = "queued"
    steps_completed: List[str] = Field(default_factory=list)
    total_steps: int = 5
    progress_percent: float = 0.0
    estimated_remaining_s: Optional[float] = None
    message: str = "Waiting to start..."


class CloneJob(BaseModel):
    """A voice cloning job."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    voice_id: str
    tenant_id: str
    created_by: str

    # Configuration
    provider: VoiceProvider
    quality: VoiceQuality
    sample_ids: List[str]

    # Status
    status: CloneJobStatus = CloneJobStatus.QUEUED
    progress: CloneJobProgress = Field(default_factory=CloneJobProgress)
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    provider_voice_id: Optional[str] = None
    processing_time_s: Optional[float] = None


# =============================================================================
# API Request/Response Models
# =============================================================================


class CreateVoiceRequest(BaseModel):
    """Request to create a new voice clone."""

    name: str = Field(..., min_length=1, max_length=100, description="Voice name")
    description: Optional[str] = Field(None, max_length=500)
    tags: List[str] = Field(default_factory=list)

    # Provider selection
    provider: Optional[VoiceProvider] = None
    quality: VoiceQuality = VoiceQuality.STANDARD

    # Style customization
    style: Optional[VoiceStyleSettings] = None

    # Consent
    consent: Optional[VoiceConsent] = None

    # Sharing
    is_public: bool = False


class UpdateVoiceRequest(BaseModel):
    """Request to update a voice."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = None
    style: Optional[VoiceStyleSettings] = None
    is_public: Optional[bool] = None


class VoiceResponse(BaseModel):
    """Voice response with full details."""

    voice: Voice
    samples: List[AudioSample] = Field(default_factory=list)
    preview_url: Optional[str] = None


class VoiceListResponse(BaseModel):
    """Paginated list of voices."""

    voices: List[Voice]
    total: int
    page: int
    page_size: int
    has_more: bool


class PreviewRequest(BaseModel):
    """Request to generate a voice preview."""

    text: str = Field(..., min_length=1, max_length=1000, description="Text to synthesize")
    style: Optional[VoiceStyleSettings] = None
    output_format: str = Field(default="mp3", description="Output audio format")


class PreviewResponse(BaseModel):
    """Voice preview response."""

    audio_url: str
    audio_base64: Optional[str] = None
    duration_s: float
    format: str
    sample_rate: int


class UploadSampleRequest(BaseModel):
    """Request metadata for sample upload."""

    voice_id: Optional[str] = None
    sample_type: SampleType = SampleType.UPLOADED
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeAudioResponse(BaseModel):
    """Response from audio analysis."""

    is_valid: bool
    metadata: AudioSampleMetadata
    quality_score: float = Field(ge=0.0, le=1.0)
    recommendations: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)


class CloneStatusResponse(BaseModel):
    """Status of a clone operation."""

    job: CloneJob
    voice: Optional[Voice] = None


# =============================================================================
# Library Models
# =============================================================================


class VoiceCategory(BaseModel):
    """A category for organizing voices."""

    category_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    voice_count: int = 0


class VoiceLibrary(BaseModel):
    """Voice library for a tenant."""

    tenant_id: str
    categories: List[VoiceCategory] = Field(default_factory=list)
    total_voices: int = 0
    total_storage_bytes: int = 0
    quota_voices: int = 50
    quota_storage_bytes: int = 5 * 1024 * 1024 * 1024  # 5GB


# =============================================================================
# Health & Stats Models
# =============================================================================


class ProviderStatus(BaseModel):
    """Status of a voice provider."""

    provider: VoiceProvider
    is_available: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    last_checked: datetime = Field(default_factory=datetime.utcnow)


class ServiceStats(BaseModel):
    """Service statistics."""

    total_voices: int
    total_samples: int
    total_storage_bytes: int
    active_jobs: int
    completed_jobs_24h: int
    failed_jobs_24h: int
    providers: List[ProviderStatus]


# =============================================================================
# Export
# =============================================================================


__all__ = [
    # Enums
    "VoiceStatus",
    "CloneJobStatus",
    "SampleType",
    # Style
    "VoiceStyleSettings",
    "VoiceCharacteristics",
    # Consent
    "VoiceConsent",
    # Samples
    "AudioSampleMetadata",
    "AudioSample",
    # Voice
    "Voice",
    # Jobs
    "CloneJobProgress",
    "CloneJob",
    # API
    "CreateVoiceRequest",
    "UpdateVoiceRequest",
    "VoiceResponse",
    "VoiceListResponse",
    "PreviewRequest",
    "PreviewResponse",
    "UploadSampleRequest",
    "AnalyzeAudioResponse",
    "CloneStatusResponse",
    # Library
    "VoiceCategory",
    "VoiceLibrary",
    # Stats
    "ProviderStatus",
    "ServiceStats",
]
