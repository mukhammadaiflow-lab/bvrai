"""Base ASR adapter interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable, Optional
import asyncio


class TranscriptType(str, Enum):
    """Type of transcript result."""
    INTERIM = "interim"  # Partial result, may change
    FINAL = "final"  # Final result for utterance
    ENDPOINT = "endpoint"  # End of speech detected


class VADEvent(str, Enum):
    """Voice Activity Detection events."""
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class Word:
    """Individual word in transcript."""
    word: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float = 1.0
    punctuated_word: Optional[str] = None  # Word with punctuation

    @property
    def display_word(self) -> str:
        """Get word for display (with punctuation if available)."""
        return self.punctuated_word or self.word


@dataclass
class TranscriptResult:
    """A transcript result from ASR."""
    text: str
    is_final: bool
    confidence: float = 1.0
    words: list[Word] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    speech_final: bool = False  # True if this is end of speech segment
    channel: int = 0

    # Metadata
    language: Optional[str] = None
    duration: float = 0.0

    @property
    def transcript_type(self) -> TranscriptType:
        """Get the type of this transcript."""
        if self.speech_final:
            return TranscriptType.ENDPOINT
        return TranscriptType.FINAL if self.is_final else TranscriptType.INTERIM


@dataclass
class TranscriptEvent:
    """An event from the ASR stream."""
    type: str  # "transcript", "vad", "error", "metadata"
    transcript: Optional[TranscriptResult] = None
    vad_event: Optional[VADEvent] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None
    timestamp: float = 0.0


class ASRAdapter(ABC):
    """Abstract base class for ASR adapters."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[[TranscriptEvent], None]] = []
        self._async_callbacks: list[Callable[[TranscriptEvent], asyncio.Future]] = []
        self._is_connected: bool = False

    @abstractmethod
    async def connect(self, session_id: str, options: Optional[dict] = None) -> None:
        """
        Connect to the ASR service.

        Args:
            session_id: Unique session identifier
            options: Provider-specific options
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the ASR service."""
        pass

    @abstractmethod
    async def send_audio(self, audio: bytes) -> None:
        """
        Send audio data to the ASR service.

        Args:
            audio: Raw audio bytes (format depends on configuration)
        """
        pass

    @abstractmethod
    async def stream_transcripts(self) -> AsyncIterator[TranscriptEvent]:
        """
        Stream transcript events from the ASR service.

        Yields:
            TranscriptEvent objects as they arrive
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Check if connected to ASR service."""
        return self._is_connected

    def on_transcript(self, callback: Callable[[TranscriptEvent], None]) -> None:
        """
        Register a callback for transcript events.

        Args:
            callback: Function to call with transcript events
        """
        self._callbacks.append(callback)

    def on_transcript_async(
        self, callback: Callable[[TranscriptEvent], asyncio.Future]
    ) -> None:
        """
        Register an async callback for transcript events.

        Args:
            callback: Async function to call with transcript events
        """
        self._async_callbacks.append(callback)

    async def _emit_event(self, event: TranscriptEvent) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Don't let callback errors break the stream

        for callback in self._async_callbacks:
            try:
                await callback(event)
            except Exception:
                pass

    @abstractmethod
    async def finalize(self) -> Optional[TranscriptResult]:
        """
        Signal end of audio and get final transcript.

        Returns:
            Final transcript result if available
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the adapter name."""
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if adapter supports real-time streaming."""
        pass

    @property
    @abstractmethod
    def supports_interim_results(self) -> bool:
        """Check if adapter supports interim (partial) results."""
        pass
