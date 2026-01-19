"""Transcription service for recordings."""

from typing import Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""
    speaker: str  # "user" or "agent"
    text: str
    start_time: float  # seconds
    end_time: float  # seconds
    confidence: float = 1.0
    words: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "speaker": self.speaker,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "words": self.words,
            "language": self.language,
            "metadata": self.metadata,
        }


@dataclass
class Transcript:
    """Complete transcript of a recording."""
    recording_id: str
    segments: List[TranscriptSegment] = field(default_factory=list)
    full_text: str = ""
    duration: float = 0.0
    language: str = "en"
    word_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recording_id": self.recording_id,
            "segments": [s.to_dict() for s in self.segments],
            "full_text": self.full_text,
            "duration": self.duration,
            "language": self.language,
            "word_count": self.word_count,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    def get_speaker_segments(self, speaker: str) -> List[TranscriptSegment]:
        """Get segments for a specific speaker."""
        return [s for s in self.segments if s.speaker == speaker]

    def get_speaker_text(self, speaker: str) -> str:
        """Get concatenated text for a speaker."""
        return " ".join(s.text for s in self.get_speaker_segments(speaker))

    def to_srt(self) -> str:
        """Convert to SRT subtitle format."""
        lines = []
        for i, segment in enumerate(self.segments, 1):
            start = self._format_srt_time(segment.start_time)
            end = self._format_srt_time(segment.end_time)
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(f"[{segment.speaker}] {segment.text}")
            lines.append("")
        return "\n".join(lines)

    def to_vtt(self) -> str:
        """Convert to WebVTT format."""
        lines = ["WEBVTT", ""]
        for segment in self.segments:
            start = self._format_vtt_time(segment.start_time)
            end = self._format_vtt_time(segment.end_time)
            lines.append(f"{start} --> {end}")
            lines.append(f"<v {segment.speaker}>{segment.text}")
            lines.append("")
        return "\n".join(lines)

    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_vtt_time(self, seconds: float) -> str:
        """Format time for VTT."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


@dataclass
class TranscriptionConfig:
    """Configuration for transcription."""
    language: str = "en"
    model: str = "default"
    enable_diarization: bool = True
    enable_punctuation: bool = True
    enable_profanity_filter: bool = False
    max_speakers: int = 2
    word_timestamps: bool = True


class TranscriptionProvider(ABC):
    """Abstract base class for transcription providers."""

    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        config: TranscriptionConfig,
    ) -> Transcript:
        """Transcribe audio data."""
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: TranscriptionConfig,
    ) -> AsyncIterator[TranscriptSegment]:
        """Transcribe streaming audio."""
        pass


class DeepgramProvider(TranscriptionProvider):
    """
    Deepgram transcription provider.

    Usage:
        provider = DeepgramProvider(api_key="...")
        transcript = await provider.transcribe(audio_data, config)
    """

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")

    async def transcribe(
        self,
        audio_data: bytes,
        config: TranscriptionConfig,
    ) -> Transcript:
        """Transcribe using Deepgram."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.deepgram.com/v1/listen",
                    params={
                        "model": config.model if config.model != "default" else "nova-2",
                        "language": config.language,
                        "punctuate": str(config.enable_punctuation).lower(),
                        "diarize": str(config.enable_diarization).lower(),
                        "utterances": "true",
                        "smart_format": "true",
                    },
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type": "audio/wav",
                    },
                    content=audio_data,
                    timeout=300.0,
                )
                response.raise_for_status()

                data = response.json()
                return self._parse_response(data, "recording")

        except Exception as e:
            logger.error(f"Deepgram transcription error: {e}")
            raise

    def _parse_response(self, data: Dict[str, Any], recording_id: str) -> Transcript:
        """Parse Deepgram response."""
        segments = []
        full_text_parts = []

        results = data.get("results", {})
        channels = results.get("channels", [{}])

        if channels:
            alternatives = channels[0].get("alternatives", [{}])
            if alternatives:
                # Get utterances if available
                utterances = results.get("utterances", [])

                if utterances:
                    for utt in utterances:
                        speaker = f"speaker_{utt.get('speaker', 0)}"
                        segments.append(TranscriptSegment(
                            speaker=speaker,
                            text=utt.get("transcript", ""),
                            start_time=utt.get("start", 0),
                            end_time=utt.get("end", 0),
                            confidence=utt.get("confidence", 1.0),
                        ))
                        full_text_parts.append(utt.get("transcript", ""))
                else:
                    # Fall back to full transcript
                    alt = alternatives[0]
                    segments.append(TranscriptSegment(
                        speaker="unknown",
                        text=alt.get("transcript", ""),
                        start_time=0,
                        end_time=0,
                        confidence=alt.get("confidence", 1.0),
                    ))
                    full_text_parts.append(alt.get("transcript", ""))

        return Transcript(
            recording_id=recording_id,
            segments=segments,
            full_text=" ".join(full_text_parts),
            word_count=len(" ".join(full_text_parts).split()),
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: TranscriptionConfig,
    ) -> AsyncIterator[TranscriptSegment]:
        """Stream transcription using Deepgram."""
        # Simplified streaming implementation
        buffer = bytearray()

        async for chunk in audio_stream:
            buffer.extend(chunk)

            # Process every 1 second of audio (simplified)
            if len(buffer) >= 32000:  # ~1 second at 16kHz 16-bit
                transcript = await self.transcribe(bytes(buffer), config)
                for segment in transcript.segments:
                    yield segment
                buffer.clear()

        # Process remaining audio
        if buffer:
            transcript = await self.transcribe(bytes(buffer), config)
            for segment in transcript.segments:
                yield segment


class AssemblyAIProvider(TranscriptionProvider):
    """
    AssemblyAI transcription provider.

    Usage:
        provider = AssemblyAIProvider(api_key="...")
        transcript = await provider.transcribe(audio_data, config)
    """

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        self._base_url = "https://api.assemblyai.com/v2"

    async def transcribe(
        self,
        audio_data: bytes,
        config: TranscriptionConfig,
    ) -> Transcript:
        """Transcribe using AssemblyAI."""
        try:
            import httpx

            headers = {"authorization": self.api_key}

            async with httpx.AsyncClient() as client:
                # Upload audio
                upload_response = await client.post(
                    f"{self._base_url}/upload",
                    headers=headers,
                    content=audio_data,
                    timeout=300.0,
                )
                upload_response.raise_for_status()
                upload_url = upload_response.json()["upload_url"]

                # Start transcription
                transcript_request = {
                    "audio_url": upload_url,
                    "language_code": config.language,
                    "punctuate": config.enable_punctuation,
                    "speaker_labels": config.enable_diarization,
                }

                start_response = await client.post(
                    f"{self._base_url}/transcript",
                    headers=headers,
                    json=transcript_request,
                    timeout=30.0,
                )
                start_response.raise_for_status()
                transcript_id = start_response.json()["id"]

                # Poll for completion
                while True:
                    status_response = await client.get(
                        f"{self._base_url}/transcript/{transcript_id}",
                        headers=headers,
                        timeout=30.0,
                    )
                    status_response.raise_for_status()
                    status_data = status_response.json()

                    if status_data["status"] == "completed":
                        return self._parse_response(status_data, "recording")
                    elif status_data["status"] == "error":
                        raise Exception(f"Transcription failed: {status_data.get('error')}")

                    await asyncio.sleep(3)

        except Exception as e:
            logger.error(f"AssemblyAI transcription error: {e}")
            raise

    def _parse_response(self, data: Dict[str, Any], recording_id: str) -> Transcript:
        """Parse AssemblyAI response."""
        segments = []

        # Parse utterances if available
        utterances = data.get("utterances", [])
        for utt in utterances:
            speaker = f"speaker_{utt.get('speaker', 'A')}"
            segments.append(TranscriptSegment(
                speaker=speaker,
                text=utt.get("text", ""),
                start_time=utt.get("start", 0) / 1000,  # Convert ms to seconds
                end_time=utt.get("end", 0) / 1000,
                confidence=utt.get("confidence", 1.0),
            ))

        return Transcript(
            recording_id=recording_id,
            segments=segments,
            full_text=data.get("text", ""),
            word_count=len(data.get("text", "").split()),
            duration=data.get("audio_duration", 0),
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: TranscriptionConfig,
    ) -> AsyncIterator[TranscriptSegment]:
        """Stream transcription (not fully supported by AssemblyAI)."""
        # Collect all audio and transcribe
        audio_data = bytearray()
        async for chunk in audio_stream:
            audio_data.extend(chunk)

        transcript = await self.transcribe(bytes(audio_data), config)
        for segment in transcript.segments:
            yield segment


class WhisperProvider(TranscriptionProvider):
    """
    OpenAI Whisper transcription provider.

    Usage:
        provider = WhisperProvider(api_key="...")
        transcript = await provider.transcribe(audio_data, config)
    """

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    async def transcribe(
        self,
        audio_data: bytes,
        config: TranscriptionConfig,
    ) -> Transcript:
        """Transcribe using OpenAI Whisper."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                files = {
                    "file": ("audio.wav", audio_data, "audio/wav"),
                    "model": (None, "whisper-1"),
                    "language": (None, config.language),
                    "response_format": (None, "verbose_json"),
                }

                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    timeout=300.0,
                )
                response.raise_for_status()

                data = response.json()
                return self._parse_response(data, "recording")

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            raise

    def _parse_response(self, data: Dict[str, Any], recording_id: str) -> Transcript:
        """Parse Whisper response."""
        segments = []

        # Parse segments
        for seg in data.get("segments", []):
            segments.append(TranscriptSegment(
                speaker="unknown",  # Whisper doesn't do diarization
                text=seg.get("text", ""),
                start_time=seg.get("start", 0),
                end_time=seg.get("end", 0),
                confidence=1.0 - seg.get("no_speech_prob", 0),
            ))

        return Transcript(
            recording_id=recording_id,
            segments=segments,
            full_text=data.get("text", ""),
            word_count=len(data.get("text", "").split()),
            duration=data.get("duration", 0),
            language=data.get("language", "en"),
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: TranscriptionConfig,
    ) -> AsyncIterator[TranscriptSegment]:
        """Stream transcription (not supported by Whisper API)."""
        # Collect all audio and transcribe
        audio_data = bytearray()
        async for chunk in audio_stream:
            audio_data.extend(chunk)

        transcript = await self.transcribe(bytes(audio_data), config)
        for segment in transcript.segments:
            yield segment


class TranscriptionService:
    """
    Transcription service with provider abstraction.

    Usage:
        service = TranscriptionService(provider="deepgram")
        transcript = await service.transcribe_recording(recording)
    """

    PROVIDERS = {
        "deepgram": DeepgramProvider,
        "assemblyai": AssemblyAIProvider,
        "whisper": WhisperProvider,
    }

    def __init__(
        self,
        provider: str = "deepgram",
        api_key: Optional[str] = None,
        config: Optional[TranscriptionConfig] = None,
    ):
        self.provider_name = provider
        self.config = config or TranscriptionConfig()

        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        self._provider = self.PROVIDERS[provider](api_key=api_key)

    async def transcribe(
        self,
        audio_data: bytes,
        config: Optional[TranscriptionConfig] = None,
    ) -> Transcript:
        """Transcribe audio data."""
        cfg = config or self.config
        return await self._provider.transcribe(audio_data, cfg)

    async def transcribe_file(
        self,
        file_path: str,
        config: Optional[TranscriptionConfig] = None,
    ) -> Transcript:
        """Transcribe audio file."""
        import aiofiles
        async with aiofiles.open(file_path, 'rb') as f:
            audio_data = await f.read()
        return await self.transcribe(audio_data, config)

    async def transcribe_url(
        self,
        url: str,
        config: Optional[TranscriptionConfig] = None,
    ) -> Transcript:
        """Transcribe audio from URL."""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=300.0)
            response.raise_for_status()
            audio_data = response.content
        return await self.transcribe(audio_data, config)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: Optional[TranscriptionConfig] = None,
    ) -> AsyncIterator[TranscriptSegment]:
        """Transcribe streaming audio."""
        cfg = config or self.config
        async for segment in self._provider.transcribe_stream(audio_stream, cfg):
            yield segment


# Global transcription service
_transcription_service: Optional[TranscriptionService] = None


def get_transcription_service(
    provider: str = "deepgram",
) -> TranscriptionService:
    """Get or create the global transcription service."""
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService(provider=provider)
    return _transcription_service
