"""Base TTS adapter interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional
import time


@dataclass
class AudioChunk:
    """A chunk of audio data from TTS."""
    data: bytes
    sample_rate: int
    encoding: str  # pcm, mulaw, mp3, opus
    sequence: int = 0
    is_final: bool = False
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class TTSRequest:
    """Request to synthesize speech."""
    text: str
    voice_id: Optional[str] = None
    model_id: Optional[str] = None
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    output_format: str = "mp3_44100_128"


@dataclass
class VoiceInfo:
    """Information about a voice."""
    voice_id: str
    name: str
    description: Optional[str] = None
    labels: Optional[dict] = None
    preview_url: Optional[str] = None


class TTSAdapter(ABC):
    """Abstract base class for TTS adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the adapter name."""
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if adapter supports streaming audio output."""
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            **kwargs: Additional provider-specific options

        Returns:
            Complete audio data as bytes
        """
        pass

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[AudioChunk]:
        """
        Stream synthesized audio chunks.

        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            **kwargs: Additional provider-specific options

        Yields:
            AudioChunk objects as they become available
        """
        pass

    @abstractmethod
    async def get_voices(self) -> list[VoiceInfo]:
        """
        Get available voices.

        Returns:
            List of available voices
        """
        pass

    async def close(self) -> None:
        """Clean up resources."""
        pass
