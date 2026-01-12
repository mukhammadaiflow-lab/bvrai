"""
Call Recording and Transcription Module

Complete recording system:
- Multi-track audio recording
- Storage backends (Local, S3, GCS)
- Transcription providers (Deepgram, AssemblyAI, Whisper)
- Recording lifecycle management
"""

from app.recording.recorder import (
    RecordingConfig,
    RecordingStatus,
    Recording,
    AudioRecorder,
    CallRecorder,
    MultiTrackRecorder,
)

from app.recording.storage import (
    StorageBackend,
    StorageObject,
    LocalStorage,
    S3Storage,
    GCSStorage,
    StorageManager,
    get_storage_manager,
    setup_storage,
)

from app.recording.transcription import (
    TranscriptSegment,
    Transcript,
    TranscriptionConfig,
    TranscriptionProvider,
    DeepgramProvider,
    AssemblyAIProvider,
    WhisperProvider,
    TranscriptionService,
    get_transcription_service,
)

from app.recording.service import (
    RecordingFormat,
    TranscriptionStatus,
    TranscriptionProvider as TranscriptionProviderEnum,
    StorageProvider,
    RecordingConfig as RecordingServiceConfig,
    Recording as RecordingEntity,
    TranscriptSegment as TranscriptSegmentEntity,
    Transcription,
    BaseTranscriptionProvider,
    DeepgramProvider as DeepgramTranscriber,
    AssemblyAIProvider as AssemblyAITranscriber,
    RecordingService,
    get_recording_service,
)

__all__ = [
    # Recorder
    "RecordingConfig",
    "RecordingStatus",
    "Recording",
    "AudioRecorder",
    "CallRecorder",
    "MultiTrackRecorder",
    # Storage
    "StorageBackend",
    "StorageObject",
    "LocalStorage",
    "S3Storage",
    "GCSStorage",
    "StorageManager",
    "get_storage_manager",
    "setup_storage",
    # Transcription
    "TranscriptSegment",
    "Transcript",
    "TranscriptionConfig",
    "TranscriptionProvider",
    "DeepgramProvider",
    "AssemblyAIProvider",
    "WhisperProvider",
    "TranscriptionService",
    "get_transcription_service",
    # Service
    "RecordingFormat",
    "TranscriptionStatus",
    "TranscriptionProviderEnum",
    "StorageProvider",
    "RecordingServiceConfig",
    "RecordingEntity",
    "TranscriptSegmentEntity",
    "Transcription",
    "BaseTranscriptionProvider",
    "DeepgramTranscriber",
    "AssemblyAITranscriber",
    "RecordingService",
    "get_recording_service",
]
