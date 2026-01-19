"""
BVRAI Voice Engine - Base Types and Interfaces

This module provides the foundational types, interfaces, and abstractions
for the voice processing pipeline including STT (Speech-to-Text) and
TTS (Text-to-Speech) operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
)
import asyncio
import hashlib
import time
import uuid


# =============================================================================
# Enums and Constants
# =============================================================================

class AudioFormat(str, Enum):
    """Supported audio formats."""
    PCM_16 = "pcm_16"
    PCM_24 = "pcm_24"
    PCM_32 = "pcm_32"
    MULAW = "mulaw"
    ALAW = "alaw"
    MP3 = "mp3"
    OGG_OPUS = "ogg_opus"
    OGG_VORBIS = "ogg_vorbis"
    FLAC = "flac"
    WAV = "wav"
    WEBM = "webm"
    AAC = "aac"


class SampleRate(int, Enum):
    """Common audio sample rates."""
    RATE_8000 = 8000    # Telephony
    RATE_16000 = 16000  # Wideband telephony
    RATE_22050 = 22050  # Common audio
    RATE_24000 = 24000  # High quality speech
    RATE_44100 = 44100  # CD quality
    RATE_48000 = 48000  # Professional audio


class STTProvider(str, Enum):
    """Supported Speech-to-Text providers."""
    DEEPGRAM = "deepgram"
    ASSEMBLYAI = "assemblyai"
    OPENAI_WHISPER = "openai_whisper"
    GOOGLE_SPEECH = "google_speech"
    AZURE_SPEECH = "azure_speech"
    AWS_TRANSCRIBE = "aws_transcribe"
    MOCK = "mock"


class TTSProvider(str, Enum):
    """Supported Text-to-Speech providers."""
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    PLAYHT = "playht"
    GOOGLE_TTS = "google_tts"
    AZURE_TTS = "azure_tts"
    AWS_POLLY = "aws_polly"
    CARTESIA = "cartesia"
    RIME = "rime"
    MOCK = "mock"


class TranscriptionStatus(str, Enum):
    """Status of a transcription operation."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class SynthesisStatus(str, Enum):
    """Status of a synthesis operation."""
    PENDING = "pending"
    PROCESSING = "processing"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"


class VoiceGender(str, Enum):
    """Voice gender classification."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceStyle(str, Enum):
    """Voice style/emotion."""
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CHEERFUL = "cheerful"
    EMPATHETIC = "empathetic"
    SERIOUS = "serious"
    CALM = "calm"
    EXCITED = "excited"


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class AudioConfig:
    """Audio configuration settings."""
    format: AudioFormat = AudioFormat.PCM_16
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16

    @property
    def bytes_per_sample(self) -> int:
        return self.bits_per_sample // 8

    @property
    def frame_size(self) -> int:
        return self.bytes_per_sample * self.channels

    def duration_from_bytes(self, num_bytes: int) -> float:
        """Calculate duration in seconds from byte count."""
        return num_bytes / (self.sample_rate * self.frame_size)

    def bytes_from_duration(self, duration: float) -> int:
        """Calculate byte count from duration in seconds."""
        return int(duration * self.sample_rate * self.frame_size)


@dataclass
class STTConfig:
    """Speech-to-Text configuration."""
    provider: STTProvider = STTProvider.DEEPGRAM
    model: str = "nova-2"
    language: str = "en-US"
    audio_config: AudioConfig = field(default_factory=AudioConfig)

    # Features
    punctuate: bool = True
    profanity_filter: bool = False
    diarize: bool = False
    smart_format: bool = True
    interim_results: bool = True
    endpointing: int = 500  # ms of silence to end utterance

    # Advanced
    keywords: List[str] = field(default_factory=list)
    custom_vocabulary: List[str] = field(default_factory=list)
    boost_keywords: float = 1.5

    # Timeouts
    connection_timeout: float = 5.0
    utterance_timeout: float = 30.0

    # Provider-specific options
    provider_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TTSConfig:
    """Text-to-Speech configuration."""
    provider: TTSProvider = TTSProvider.ELEVENLABS
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Default ElevenLabs voice
    model: str = "eleven_turbo_v2"

    # Voice settings
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True

    # Audio output
    audio_config: AudioConfig = field(default_factory=lambda: AudioConfig(
        format=AudioFormat.MP3,
        sample_rate=24000
    ))

    # Speed and pitch
    speed: float = 1.0
    pitch: float = 0.0

    # Streaming
    optimize_streaming_latency: int = 3  # 0-4, higher = lower latency
    enable_ssml: bool = False

    # Provider-specific options
    provider_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceProfile:
    """Complete voice profile for an agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Provider settings
    tts_provider: TTSProvider = TTSProvider.ELEVENLABS
    voice_id: str = ""
    model: str = ""

    # Voice characteristics
    gender: VoiceGender = VoiceGender.NEUTRAL
    style: VoiceStyle = VoiceStyle.PROFESSIONAL
    language: str = "en-US"

    # Tuning parameters
    stability: float = 0.5
    similarity_boost: float = 0.75
    speed: float = 1.0
    pitch: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_tts_config(self) -> TTSConfig:
        """Convert profile to TTS configuration."""
        return TTSConfig(
            provider=self.tts_provider,
            voice_id=self.voice_id,
            model=self.model,
            stability=self.stability,
            similarity_boost=self.similarity_boost,
            speed=self.speed,
            pitch=self.pitch,
        )


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class TranscriptionWord:
    """Individual word in a transcription."""
    word: str
    start: float  # seconds
    end: float    # seconds
    confidence: float
    speaker: Optional[int] = None  # For diarization


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    words: List[TranscriptionWord] = field(default_factory=list)

    # Status
    status: TranscriptionStatus = TranscriptionStatus.COMPLETED
    is_final: bool = True

    # Metadata
    confidence: float = 0.0
    language: str = "en"
    duration: float = 0.0

    # Provider info
    provider: STTProvider = STTProvider.DEEPGRAM
    model: str = ""

    # Timing
    latency_ms: float = 0.0
    processing_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Error handling
    error: Optional[str] = None
    error_code: Optional[str] = None


@dataclass
class SynthesisResult:
    """Result of a synthesis operation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    audio_data: bytes = b""

    # Status
    status: SynthesisStatus = SynthesisStatus.COMPLETED

    # Metadata
    text_length: int = 0
    audio_duration: float = 0.0
    audio_format: AudioFormat = AudioFormat.MP3
    sample_rate: int = 24000

    # Provider info
    provider: TTSProvider = TTSProvider.ELEVENLABS
    voice_id: str = ""
    model: str = ""

    # Timing and cost
    latency_ms: float = 0.0
    processing_time_ms: float = 0.0
    characters_billed: int = 0
    estimated_cost: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Error handling
    error: Optional[str] = None
    error_code: Optional[str] = None


@dataclass
class AudioChunk:
    """Chunk of audio data for streaming."""
    data: bytes
    sequence: int
    timestamp: float
    is_final: bool = False
    duration_ms: float = 0.0

    @property
    def size(self) -> int:
        return len(self.data)


# =============================================================================
# Provider Health and Metrics
# =============================================================================

@dataclass
class ProviderHealth:
    """Health status of a provider."""
    provider: Union[STTProvider, TTSProvider]
    is_healthy: bool = True
    last_check: datetime = field(default_factory=datetime.utcnow)

    # Metrics
    success_count: int = 0
    error_count: int = 0
    total_requests: int = 0

    # Performance
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Error tracking
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    consecutive_errors: int = 0

    # Circuit breaker
    circuit_open: bool = False
    circuit_open_until: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.success_count / self.total_requests

    @property
    def error_rate(self) -> float:
        return 1.0 - self.success_rate

    def record_success(self, latency_ms: float) -> None:
        """Record a successful request."""
        self.success_count += 1
        self.total_requests += 1
        self.consecutive_errors = 0
        self._update_latency(latency_ms)

    def record_error(self, error: str) -> None:
        """Record a failed request."""
        self.error_count += 1
        self.total_requests += 1
        self.consecutive_errors += 1
        self.last_error = error
        self.last_error_time = datetime.utcnow()

        # Open circuit breaker after 5 consecutive errors
        if self.consecutive_errors >= 5:
            self.circuit_open = True
            self.circuit_open_until = datetime.utcnow()

    def _update_latency(self, latency_ms: float) -> None:
        """Update latency metrics using exponential moving average."""
        alpha = 0.1
        self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms


@dataclass
class VoiceMetrics:
    """Aggregated metrics for voice operations."""
    # STT metrics
    stt_requests: int = 0
    stt_characters_transcribed: int = 0
    stt_audio_seconds_processed: float = 0.0
    stt_avg_latency_ms: float = 0.0

    # TTS metrics
    tts_requests: int = 0
    tts_characters_synthesized: int = 0
    tts_audio_seconds_generated: float = 0.0
    tts_avg_latency_ms: float = 0.0

    # Error metrics
    stt_errors: int = 0
    tts_errors: int = 0

    # Cost tracking
    estimated_stt_cost: float = 0.0
    estimated_tts_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.estimated_stt_cost + self.estimated_tts_cost

    @property
    def stt_error_rate(self) -> float:
        if self.stt_requests == 0:
            return 0.0
        return self.stt_errors / self.stt_requests

    @property
    def tts_error_rate(self) -> float:
        if self.tts_requests == 0:
            return 0.0
        return self.tts_errors / self.tts_requests


# =============================================================================
# Abstract Base Classes (Protocols)
# =============================================================================

class STTProviderInterface(ABC):
    """Abstract interface for STT providers."""

    @property
    @abstractmethod
    def provider_name(self) -> STTProvider:
        """Get the provider identifier."""
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        config: STTConfig
    ) -> TranscriptionResult:
        """Transcribe audio data to text."""
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        config: STTConfig
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """Stream transcription results as audio is received."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up provider resources."""
        pass


class TTSProviderInterface(ABC):
    """Abstract interface for TTS providers."""

    @property
    @abstractmethod
    def provider_name(self) -> TTSProvider:
        """Get the provider identifier."""
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        config: TTSConfig
    ) -> SynthesisResult:
        """Synthesize text to audio."""
        pass

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        config: TTSConfig
    ) -> AsyncGenerator[AudioChunk, None]:
        """Stream audio chunks as they are generated."""
        pass

    @abstractmethod
    async def get_voices(self) -> List[Dict[str, Any]]:
        """Get available voices for this provider."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up provider resources."""
        pass


# =============================================================================
# Exceptions
# =============================================================================

class VoiceError(Exception):
    """Base exception for voice operations."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code or "VOICE_ERROR"
        self.provider = provider
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "provider": self.provider,
            "details": self.details,
        }


class STTError(VoiceError):
    """Error during speech-to-text operation."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code="STT_ERROR", **kwargs)


class TTSError(VoiceError):
    """Error during text-to-speech operation."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code="TTS_ERROR", **kwargs)


class ProviderConnectionError(VoiceError):
    """Failed to connect to provider."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code="PROVIDER_CONNECTION_ERROR", **kwargs)


class ProviderTimeoutError(VoiceError):
    """Provider operation timed out."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code="PROVIDER_TIMEOUT", **kwargs)


class ProviderRateLimitError(VoiceError):
    """Provider rate limit exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, code="PROVIDER_RATE_LIMIT", **kwargs)
        self.retry_after = retry_after


class InvalidAudioError(VoiceError):
    """Invalid audio data provided."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code="INVALID_AUDIO", **kwargs)


class UnsupportedLanguageError(VoiceError):
    """Language not supported by provider."""

    def __init__(self, message: str, language: str, **kwargs):
        super().__init__(message, code="UNSUPPORTED_LANGUAGE", **kwargs)
        self.language = language


class VoiceNotFoundError(VoiceError):
    """Requested voice not found."""

    def __init__(self, message: str, voice_id: str, **kwargs):
        super().__init__(message, code="VOICE_NOT_FOUND", **kwargs)
        self.voice_id = voice_id


class QuotaExceededError(VoiceError):
    """Provider quota exceeded."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code="QUOTA_EXCEEDED", **kwargs)


# =============================================================================
# Utility Functions
# =============================================================================

def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"vr_{uuid.uuid4().hex[:16]}"


def calculate_audio_hash(audio_data: bytes) -> str:
    """Calculate SHA-256 hash of audio data."""
    return hashlib.sha256(audio_data).hexdigest()


def estimate_audio_duration(
    audio_bytes: int,
    sample_rate: int = 16000,
    channels: int = 1,
    bits_per_sample: int = 16
) -> float:
    """Estimate audio duration from byte count."""
    bytes_per_sample = bits_per_sample // 8
    samples = audio_bytes / (bytes_per_sample * channels)
    return samples / sample_rate


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.1f}s"


# =============================================================================
# Type Aliases
# =============================================================================

AudioStream = AsyncGenerator[bytes, None]
TranscriptionStream = AsyncGenerator[TranscriptionResult, None]
SynthesisStream = AsyncGenerator[AudioChunk, None]

# Callback types
OnTranscriptionCallback = Callable[[TranscriptionResult], None]
OnAudioChunkCallback = Callable[[AudioChunk], None]
OnErrorCallback = Callable[[VoiceError], None]
