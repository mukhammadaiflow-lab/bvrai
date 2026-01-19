"""
Call Recording & Storage Module

This module provides comprehensive recording management for voice agents,
including storage, transcription, retention policies, and content redaction.

Features:
- Multi-backend storage (Local, S3, GCS)
- Transcription with multiple providers (Deepgram, AssemblyAI, Whisper)
- Speaker diarization and sentiment analysis
- Retention policies with automatic lifecycle management
- PII detection and redaction
- Access control and compliance support

Example usage:

    from bvrai_core.recordings import (
        RecordingService,
        StorageConfig,
        StorageBackend,
        TranscriptionProvider,
        RetentionAction,
        RedactionType,
    )

    # Initialize service with S3 storage
    storage_config = StorageConfig(
        backend=StorageBackend.S3,
        bucket_name="my-recordings",
        region="us-east-1",
    )

    service = RecordingService()
    service.storage.add_provider("primary", storage_config, is_default=True)

    # Start recording
    recording = await service.start_recording(
        call_id="call_123",
        organization_id="org_456",
        agent_id="agent_789",
        caller_phone="+15551234567",
    )

    # Finish and upload recording
    recording = await service.finish_recording(
        recording_id=recording.id,
        audio_data=audio_bytes,
        duration_seconds=120.5,
    )

    # Transcribe
    transcription = await service.transcribe_recording(
        recording_id=recording.id,
        provider=TranscriptionProvider.DEEPGRAM,
        diarization=True,
    )

    # Search recordings
    results = await service.search_recordings(
        organization_id="org_456",
        query="appointment",
    )

    # Create retention policy
    policy = await service.retention.create_policy(
        name="90 Day Retention",
        organization_id="org_456",
        retention_days=90,
        action_on_expiry=RetentionAction.ARCHIVE,
    )

    # Set up PII redaction
    await service.redaction.add_rule(
        name="Credit Card",
        redaction_type=RedactionType.CREDIT_CARD,
        organization_id="org_456",
    )

    # Get download URL
    url = await service.get_download_url(recording.id)

    # Get statistics
    stats = await service.get_recording_stats("org_456")
"""

# Base types and enums
from .base import (
    # Enums
    RecordingStatus,
    RecordingFormat,
    TranscriptionStatus,
    TranscriptionProvider,
    StorageBackend,
    RetentionAction,
    RedactionType,
    AccessLevel,
    ChannelType,
    # Storage
    StorageConfig,
    # Recording types
    RecordingMetadata,
    AudioProperties,
    Recording,
    # Transcription types
    TranscriptSegment,
    Transcription,
    # Retention
    RetentionPolicy,
    # Redaction
    RedactionRule,
    RedactionResult,
    # Exceptions
    RecordingError,
    StorageError,
    TranscriptionError,
    RetentionError,
    RedactionError,
    AccessDeniedError,
)

# Storage providers
from .storage import (
    StorageProvider,
    LocalStorageProvider,
    S3StorageProvider,
    GCSStorageProvider,
    StorageProviderFactory,
    StorageManager,
)

# Transcription services
from .transcription import (
    TranscriptionService,
    DeepgramTranscriptionService,
    AssemblyAITranscriptionService,
    WhisperTranscriptionService,
    TranscriptionManager,
    TranscriptionPostProcessor,
)

# Service layer
from .service import (
    RecordingStorage,
    RetentionManager,
    RedactionEngine,
    RecordingService,
)


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
    # Storage config
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
    # Storage
    "StorageProvider",
    "LocalStorageProvider",
    "S3StorageProvider",
    "GCSStorageProvider",
    "StorageProviderFactory",
    "StorageManager",
    # Transcription
    "TranscriptionService",
    "DeepgramTranscriptionService",
    "AssemblyAITranscriptionService",
    "WhisperTranscriptionService",
    "TranscriptionManager",
    "TranscriptionPostProcessor",
    # Service
    "RecordingStorage",
    "RetentionManager",
    "RedactionEngine",
    "RecordingService",
    # Exceptions
    "RecordingError",
    "StorageError",
    "TranscriptionError",
    "RetentionError",
    "RedactionError",
    "AccessDeniedError",
]
