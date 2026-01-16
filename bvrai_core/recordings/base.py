"""
Call Recording & Storage Base Types Module

This module defines core types and data structures for managing call recordings,
transcriptions, retention policies, and storage backends.
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)


# =============================================================================
# Enums
# =============================================================================


class RecordingStatus(str, Enum):
    """Status of a recording."""

    RECORDING = "recording"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"
    DELETED = "deleted"
    REDACTED = "redacted"


class RecordingFormat(str, Enum):
    """Audio format of recordings."""

    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    WEBM = "webm"
    M4A = "m4a"


class TranscriptionStatus(str, Enum):
    """Status of a transcription."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REDACTED = "redacted"


class TranscriptionProvider(str, Enum):
    """Transcription service provider."""

    DEEPGRAM = "deepgram"
    ASSEMBLY_AI = "assembly_ai"
    OPENAI_WHISPER = "openai_whisper"
    GOOGLE = "google"
    AWS_TRANSCRIBE = "aws_transcribe"
    AZURE = "azure"


class StorageBackend(str, Enum):
    """Storage backend type."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    MINIO = "minio"


class RetentionAction(str, Enum):
    """Action to take when retention period expires."""

    DELETE = "delete"
    ARCHIVE = "archive"
    ANONYMIZE = "anonymize"
    NOTIFY = "notify"


class RedactionType(str, Enum):
    """Type of content redaction."""

    PII = "pii"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    ADDRESS = "address"
    CUSTOM_PATTERN = "custom_pattern"
    AUDIO_SEGMENT = "audio_segment"


class AccessLevel(str, Enum):
    """Access level for recordings."""

    PRIVATE = "private"
    ORGANIZATION = "organization"
    SHARED = "shared"
    PUBLIC = "public"


class ChannelType(str, Enum):
    """Audio channel type."""

    MONO = "mono"
    STEREO = "stereo"
    SPLIT = "split"  # Separate files for each party


# =============================================================================
# Storage Configuration
# =============================================================================


@dataclass
class StorageConfig:
    """Configuration for a storage backend."""

    backend: StorageBackend
    bucket_name: str = ""
    region: str = ""
    endpoint_url: Optional[str] = None

    # Authentication
    access_key: Optional[str] = None
    secret_key: Optional[str] = None

    # Storage options
    storage_class: str = "STANDARD"
    encryption_enabled: bool = True
    encryption_key_id: Optional[str] = None

    # Path configuration
    base_path: str = "recordings"
    path_template: str = "{org_id}/{year}/{month}/{day}/{call_id}"

    # Transfer settings
    multipart_threshold_mb: int = 100
    multipart_chunk_size_mb: int = 10
    max_concurrent_transfers: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend.value,
            "bucket_name": self.bucket_name,
            "region": self.region,
            "endpoint_url": self.endpoint_url,
            "storage_class": self.storage_class,
            "encryption_enabled": self.encryption_enabled,
            "encryption_key_id": self.encryption_key_id,
            "base_path": self.base_path,
            "path_template": self.path_template,
            "multipart_threshold_mb": self.multipart_threshold_mb,
            "multipart_chunk_size_mb": self.multipart_chunk_size_mb,
            "max_concurrent_transfers": self.max_concurrent_transfers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageConfig":
        """Create from dictionary."""
        return cls(
            backend=StorageBackend(data["backend"]),
            bucket_name=data.get("bucket_name", ""),
            region=data.get("region", ""),
            endpoint_url=data.get("endpoint_url"),
            access_key=data.get("access_key"),
            secret_key=data.get("secret_key"),
            storage_class=data.get("storage_class", "STANDARD"),
            encryption_enabled=data.get("encryption_enabled", True),
            encryption_key_id=data.get("encryption_key_id"),
            base_path=data.get("base_path", "recordings"),
            path_template=data.get("path_template", "{org_id}/{year}/{month}/{day}/{call_id}"),
            multipart_threshold_mb=data.get("multipart_threshold_mb", 100),
            multipart_chunk_size_mb=data.get("multipart_chunk_size_mb", 10),
            max_concurrent_transfers=data.get("max_concurrent_transfers", 10),
        )


# =============================================================================
# Recording Types
# =============================================================================


@dataclass
class RecordingMetadata:
    """Metadata for a recording."""

    # Call information
    call_id: str
    organization_id: str
    agent_id: Optional[str] = None

    # Participant info
    caller_phone: Optional[str] = None
    agent_phone: Optional[str] = None
    caller_name: Optional[str] = None

    # Call context
    campaign_id: Optional[str] = None
    direction: str = "inbound"  # inbound, outbound
    disposition: Optional[str] = None

    # Tags and custom data
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "call_id": self.call_id,
            "organization_id": self.organization_id,
            "agent_id": self.agent_id,
            "caller_phone": self.caller_phone,
            "agent_phone": self.agent_phone,
            "caller_name": self.caller_name,
            "campaign_id": self.campaign_id,
            "direction": self.direction,
            "disposition": self.disposition,
            "tags": self.tags,
            "custom_fields": self.custom_fields,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordingMetadata":
        """Create from dictionary."""
        return cls(
            call_id=data["call_id"],
            organization_id=data["organization_id"],
            agent_id=data.get("agent_id"),
            caller_phone=data.get("caller_phone"),
            agent_phone=data.get("agent_phone"),
            caller_name=data.get("caller_name"),
            campaign_id=data.get("campaign_id"),
            direction=data.get("direction", "inbound"),
            disposition=data.get("disposition"),
            tags=data.get("tags", []),
            custom_fields=data.get("custom_fields", {}),
        )


@dataclass
class AudioProperties:
    """Technical properties of an audio recording."""

    format: RecordingFormat = RecordingFormat.WAV
    channels: ChannelType = ChannelType.MONO
    sample_rate: int = 16000
    bit_depth: int = 16
    bitrate: Optional[int] = None  # For compressed formats

    # Duration and size
    duration_seconds: float = 0.0
    file_size_bytes: int = 0

    # Quality metrics
    average_volume_db: Optional[float] = None
    silence_percentage: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format": self.format.value,
            "channels": self.channels.value,
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "bitrate": self.bitrate,
            "duration_seconds": self.duration_seconds,
            "file_size_bytes": self.file_size_bytes,
            "average_volume_db": self.average_volume_db,
            "silence_percentage": self.silence_percentage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioProperties":
        """Create from dictionary."""
        return cls(
            format=RecordingFormat(data.get("format", "wav")),
            channels=ChannelType(data.get("channels", "mono")),
            sample_rate=data.get("sample_rate", 16000),
            bit_depth=data.get("bit_depth", 16),
            bitrate=data.get("bitrate"),
            duration_seconds=data.get("duration_seconds", 0.0),
            file_size_bytes=data.get("file_size_bytes", 0),
            average_volume_db=data.get("average_volume_db"),
            silence_percentage=data.get("silence_percentage"),
        )


@dataclass
class Recording:
    """A call recording."""

    id: str
    call_id: str
    organization_id: str

    # Status
    status: RecordingStatus = RecordingStatus.RECORDING

    # Storage
    storage_backend: StorageBackend = StorageBackend.S3
    storage_path: str = ""
    storage_bucket: str = ""

    # Audio properties
    audio: AudioProperties = field(default_factory=AudioProperties)

    # Metadata
    metadata: RecordingMetadata = field(default_factory=lambda: RecordingMetadata(
        call_id="", organization_id=""
    ))

    # Access control
    access_level: AccessLevel = AccessLevel.ORGANIZATION
    allowed_users: List[str] = field(default_factory=list)
    allowed_roles: List[str] = field(default_factory=list)

    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Retention
    retention_policy_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None

    # Content hash for integrity
    content_hash: Optional[str] = None

    # Related records
    transcription_id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"rec_{uuid.uuid4().hex[:24]}"
        if self.metadata.call_id == "":
            self.metadata.call_id = self.call_id
        if self.metadata.organization_id == "":
            self.metadata.organization_id = self.organization_id

    @property
    def duration(self) -> timedelta:
        """Get recording duration."""
        return timedelta(seconds=self.audio.duration_seconds)

    @property
    def file_size_mb(self) -> float:
        """Get file size in MB."""
        return self.audio.file_size_bytes / (1024 * 1024)

    def get_storage_url(self) -> str:
        """Get storage URL for the recording."""
        if self.storage_backend == StorageBackend.S3:
            return f"s3://{self.storage_bucket}/{self.storage_path}"
        elif self.storage_backend == StorageBackend.GCS:
            return f"gs://{self.storage_bucket}/{self.storage_path}"
        else:
            return self.storage_path

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "call_id": self.call_id,
            "organization_id": self.organization_id,
            "status": self.status.value,
            "storage_backend": self.storage_backend.value,
            "storage_path": self.storage_path,
            "storage_bucket": self.storage_bucket,
            "audio": self.audio.to_dict(),
            "metadata": self.metadata.to_dict(),
            "access_level": self.access_level.value,
            "allowed_users": self.allowed_users,
            "allowed_roles": self.allowed_roles,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "retention_policy_id": self.retention_policy_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "content_hash": self.content_hash,
            "transcription_id": self.transcription_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Recording":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            call_id=data["call_id"],
            organization_id=data["organization_id"],
            status=RecordingStatus(data.get("status", "recording")),
            storage_backend=StorageBackend(data.get("storage_backend", "s3")),
            storage_path=data.get("storage_path", ""),
            storage_bucket=data.get("storage_bucket", ""),
            audio=AudioProperties.from_dict(data.get("audio", {})),
            metadata=RecordingMetadata.from_dict(data.get("metadata", {"call_id": data["call_id"], "organization_id": data["organization_id"]})),
            access_level=AccessLevel(data.get("access_level", "organization")),
            allowed_users=data.get("allowed_users", []),
            allowed_roles=data.get("allowed_roles", []),
            started_at=datetime.fromisoformat(data["started_at"]) if "started_at" in data else datetime.utcnow(),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            retention_policy_id=data.get("retention_policy_id"),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            archived_at=datetime.fromisoformat(data["archived_at"]) if data.get("archived_at") else None,
            content_hash=data.get("content_hash"),
            transcription_id=data.get("transcription_id"),
        )


# =============================================================================
# Transcription Types
# =============================================================================


@dataclass
class TranscriptSegment:
    """A segment of a transcription with timing and speaker info."""

    id: str
    text: str
    start_time: float  # seconds
    end_time: float  # seconds

    # Speaker identification
    speaker: str = "unknown"  # "agent", "caller", "unknown"
    speaker_id: Optional[str] = None

    # Confidence
    confidence: float = 1.0

    # Word-level timing
    words: List[Dict[str, Any]] = field(default_factory=list)

    # Sentiment (if analyzed)
    sentiment: Optional[str] = None  # positive, negative, neutral
    sentiment_score: Optional[float] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"seg_{uuid.uuid4().hex[:12]}"

    @property
    def duration(self) -> float:
        """Get segment duration."""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker": self.speaker,
            "speaker_id": self.speaker_id,
            "confidence": self.confidence,
            "words": self.words,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptSegment":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            text=data["text"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            speaker=data.get("speaker", "unknown"),
            speaker_id=data.get("speaker_id"),
            confidence=data.get("confidence", 1.0),
            words=data.get("words", []),
            sentiment=data.get("sentiment"),
            sentiment_score=data.get("sentiment_score"),
        )


@dataclass
class Transcription:
    """A transcription of a recording."""

    id: str
    recording_id: str
    organization_id: str

    # Status
    status: TranscriptionStatus = TranscriptionStatus.PENDING

    # Provider
    provider: TranscriptionProvider = TranscriptionProvider.DEEPGRAM
    provider_job_id: Optional[str] = None

    # Language
    language: str = "en-US"
    detected_language: Optional[str] = None

    # Content
    full_text: str = ""
    segments: List[TranscriptSegment] = field(default_factory=list)

    # Statistics
    word_count: int = 0
    speaker_count: int = 0
    average_confidence: float = 0.0

    # Analysis results
    summary: Optional[str] = None
    key_topics: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Error info
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"trx_{uuid.uuid4().hex[:24]}"

    @property
    def duration(self) -> float:
        """Get total duration from segments."""
        if not self.segments:
            return 0.0
        return max(s.end_time for s in self.segments)

    def get_speaker_segments(self, speaker: str) -> List[TranscriptSegment]:
        """Get segments for a specific speaker."""
        return [s for s in self.segments if s.speaker == speaker]

    def get_text_by_speaker(self, speaker: str) -> str:
        """Get combined text for a speaker."""
        segments = self.get_speaker_segments(speaker)
        return " ".join(s.text for s in segments)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "recording_id": self.recording_id,
            "organization_id": self.organization_id,
            "status": self.status.value,
            "provider": self.provider.value,
            "provider_job_id": self.provider_job_id,
            "language": self.language,
            "detected_language": self.detected_language,
            "full_text": self.full_text,
            "segments": [s.to_dict() for s in self.segments],
            "word_count": self.word_count,
            "speaker_count": self.speaker_count,
            "average_confidence": self.average_confidence,
            "summary": self.summary,
            "key_topics": self.key_topics,
            "action_items": self.action_items,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "updated_at": self.updated_at.isoformat(),
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transcription":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            recording_id=data["recording_id"],
            organization_id=data["organization_id"],
            status=TranscriptionStatus(data.get("status", "pending")),
            provider=TranscriptionProvider(data.get("provider", "deepgram")),
            provider_job_id=data.get("provider_job_id"),
            language=data.get("language", "en-US"),
            detected_language=data.get("detected_language"),
            full_text=data.get("full_text", ""),
            segments=[TranscriptSegment.from_dict(s) for s in data.get("segments", [])],
            word_count=data.get("word_count", 0),
            speaker_count=data.get("speaker_count", 0),
            average_confidence=data.get("average_confidence", 0.0),
            summary=data.get("summary"),
            key_topics=data.get("key_topics", []),
            action_items=data.get("action_items", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            error_message=data.get("error_message"),
        )


# =============================================================================
# Retention Policy
# =============================================================================


@dataclass
class RetentionPolicy:
    """Policy for recording retention and lifecycle management."""

    id: str
    name: str
    organization_id: str

    # Retention periods
    retention_days: int = 90
    archive_after_days: Optional[int] = None

    # Actions
    action_on_expiry: RetentionAction = RetentionAction.DELETE
    notify_before_days: int = 7

    # Scope
    applies_to_campaigns: List[str] = field(default_factory=list)
    applies_to_agents: List[str] = field(default_factory=list)
    applies_to_tags: List[str] = field(default_factory=list)

    # Compliance
    compliance_required: bool = False
    legal_hold: bool = False

    # Automatic processing
    auto_redact_pii: bool = False
    redaction_types: List[RedactionType] = field(default_factory=list)

    # Status
    is_active: bool = True
    is_default: bool = False

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"rp_{uuid.uuid4().hex[:24]}"

    def calculate_expiry_date(self, start_date: datetime) -> datetime:
        """Calculate expiry date from start date."""
        return start_date + timedelta(days=self.retention_days)

    def calculate_archive_date(self, start_date: datetime) -> Optional[datetime]:
        """Calculate archive date if applicable."""
        if self.archive_after_days:
            return start_date + timedelta(days=self.archive_after_days)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "organization_id": self.organization_id,
            "retention_days": self.retention_days,
            "archive_after_days": self.archive_after_days,
            "action_on_expiry": self.action_on_expiry.value,
            "notify_before_days": self.notify_before_days,
            "applies_to_campaigns": self.applies_to_campaigns,
            "applies_to_agents": self.applies_to_agents,
            "applies_to_tags": self.applies_to_tags,
            "compliance_required": self.compliance_required,
            "legal_hold": self.legal_hold,
            "auto_redact_pii": self.auto_redact_pii,
            "redaction_types": [r.value for r in self.redaction_types],
            "is_active": self.is_active,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetentionPolicy":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            organization_id=data["organization_id"],
            retention_days=data.get("retention_days", 90),
            archive_after_days=data.get("archive_after_days"),
            action_on_expiry=RetentionAction(data.get("action_on_expiry", "delete")),
            notify_before_days=data.get("notify_before_days", 7),
            applies_to_campaigns=data.get("applies_to_campaigns", []),
            applies_to_agents=data.get("applies_to_agents", []),
            applies_to_tags=data.get("applies_to_tags", []),
            compliance_required=data.get("compliance_required", False),
            legal_hold=data.get("legal_hold", False),
            auto_redact_pii=data.get("auto_redact_pii", False),
            redaction_types=[RedactionType(r) for r in data.get("redaction_types", [])],
            is_active=data.get("is_active", True),
            is_default=data.get("is_default", False),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
        )


# =============================================================================
# Redaction
# =============================================================================


@dataclass
class RedactionRule:
    """A rule for redacting sensitive content."""

    id: str
    name: str
    redaction_type: RedactionType

    # Pattern matching
    pattern: Optional[str] = None  # Regex pattern
    keywords: List[str] = field(default_factory=list)

    # Replacement
    replacement_text: str = "[REDACTED]"
    replacement_audio: str = "beep"  # beep, silence, mute

    # Scope
    apply_to_transcript: bool = True
    apply_to_audio: bool = False

    # Status
    is_active: bool = True

    def __post_init__(self):
        if not self.id:
            self.id = f"rr_{uuid.uuid4().hex[:16]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "redaction_type": self.redaction_type.value,
            "pattern": self.pattern,
            "keywords": self.keywords,
            "replacement_text": self.replacement_text,
            "replacement_audio": self.replacement_audio,
            "apply_to_transcript": self.apply_to_transcript,
            "apply_to_audio": self.apply_to_audio,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedactionRule":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data["name"],
            redaction_type=RedactionType(data["redaction_type"]),
            pattern=data.get("pattern"),
            keywords=data.get("keywords", []),
            replacement_text=data.get("replacement_text", "[REDACTED]"),
            replacement_audio=data.get("replacement_audio", "beep"),
            apply_to_transcript=data.get("apply_to_transcript", True),
            apply_to_audio=data.get("apply_to_audio", False),
            is_active=data.get("is_active", True),
        )


@dataclass
class RedactionResult:
    """Result of a redaction operation."""

    recording_id: str
    redaction_rule_ids: List[str] = field(default_factory=list)

    # Statistics
    text_redactions_count: int = 0
    audio_redactions_count: int = 0
    redacted_segments: List[Dict[str, Any]] = field(default_factory=list)

    # Status
    transcript_redacted: bool = False
    audio_redacted: bool = False

    # Timestamps
    performed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recording_id": self.recording_id,
            "redaction_rule_ids": self.redaction_rule_ids,
            "text_redactions_count": self.text_redactions_count,
            "audio_redactions_count": self.audio_redactions_count,
            "redacted_segments": self.redacted_segments,
            "transcript_redacted": self.transcript_redacted,
            "audio_redacted": self.audio_redacted,
            "performed_at": self.performed_at.isoformat(),
        }


# =============================================================================
# Exceptions
# =============================================================================


class RecordingError(Exception):
    """Base exception for recording errors."""
    pass


class StorageError(RecordingError):
    """Error with storage operations."""
    pass


class TranscriptionError(RecordingError):
    """Error with transcription operations."""
    pass


class RetentionError(RecordingError):
    """Error with retention operations."""
    pass


class RedactionError(RecordingError):
    """Error with redaction operations."""
    pass


class AccessDeniedError(RecordingError):
    """Access denied to recording."""
    pass


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "RecordingStatus",
    "RecordingFormat",
    "TranscriptionStatus",
    "TranscriptionProvider",
    "StorageBackend",
    "RetentionAction",
    "RedactionType",
    "AccessLevel",
    "ChannelType",
    # Storage
    "StorageConfig",
    # Recording types
    "RecordingMetadata",
    "AudioProperties",
    "Recording",
    # Transcription types
    "TranscriptSegment",
    "Transcription",
    # Retention
    "RetentionPolicy",
    # Redaction
    "RedactionRule",
    "RedactionResult",
    # Exceptions
    "RecordingError",
    "StorageError",
    "TranscriptionError",
    "RetentionError",
    "RedactionError",
    "AccessDeniedError",
]
