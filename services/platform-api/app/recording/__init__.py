"""Call recording and storage module."""

from app.recording.recorder import (
    RecordingConfig,
    RecordingStatus,
    Recording,
    AudioRecorder,
    CallRecorder,
)

from app.recording.storage import (
    StorageBackend,
    LocalStorage,
    S3Storage,
    GCSStorage,
    StorageManager,
    get_storage_manager,
)

from app.recording.transcription import (
    TranscriptSegment,
    Transcript,
    TranscriptionConfig,
    TranscriptionService,
    get_transcription_service,
)

__all__ = [
    # Recorder
    "RecordingConfig",
    "RecordingStatus",
    "Recording",
    "AudioRecorder",
    "CallRecorder",
    # Storage
    "StorageBackend",
    "LocalStorage",
    "S3Storage",
    "GCSStorage",
    "StorageManager",
    "get_storage_manager",
    # Transcription
    "TranscriptSegment",
    "Transcript",
    "TranscriptionConfig",
    "TranscriptionService",
    "get_transcription_service",
]
