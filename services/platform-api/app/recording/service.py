"""
Recording and Transcription Service

Complete call recording system:
- Recording capture and storage
- Real-time transcription
- Post-call processing
- Storage management
"""

from typing import Optional, Dict, Any, List, AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import asyncio
import aiohttp
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class RecordingStatus(str, Enum):
    """Recording status."""
    PENDING = "pending"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class RecordingFormat(str, Enum):
    """Recording formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    WEBM = "webm"


class TranscriptionStatus(str, Enum):
    """Transcription status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptionProvider(str, Enum):
    """Transcription providers."""
    DEEPGRAM = "deepgram"
    ASSEMBLY_AI = "assembly_ai"
    WHISPER = "whisper"
    GOOGLE = "google"
    AWS = "aws"


class StorageProvider(str, Enum):
    """Storage providers."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"


@dataclass
class RecordingConfig:
    """Recording configuration."""
    format: RecordingFormat = RecordingFormat.WAV
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16

    # Splitting
    split_on_silence: bool = False
    silence_threshold_db: int = -40
    silence_duration_ms: int = 2000

    # Storage
    storage_provider: StorageProvider = StorageProvider.LOCAL
    storage_path: str = "/recordings"
    retention_days: int = 30

    # Processing
    auto_transcribe: bool = True
    transcription_provider: TranscriptionProvider = TranscriptionProvider.DEEPGRAM

    # Quality
    noise_reduction: bool = True
    normalize_audio: bool = True


@dataclass
class Recording:
    """Recording entity."""
    recording_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Context
    call_id: str = ""
    conversation_id: str = ""
    tenant_id: str = ""
    agent_id: str = ""

    # Status
    status: RecordingStatus = RecordingStatus.PENDING

    # Audio
    format: RecordingFormat = RecordingFormat.WAV
    duration_seconds: float = 0.0
    file_size_bytes: int = 0
    sample_rate: int = 16000
    channels: int = 1

    # Storage
    storage_url: str = ""
    storage_provider: StorageProvider = StorageProvider.LOCAL
    storage_path: str = ""

    # Transcription
    transcription_id: Optional[str] = None
    transcription_status: TranscriptionStatus = TranscriptionStatus.PENDING

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Timing
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recording_id": self.recording_id,
            "call_id": self.call_id,
            "status": self.status.value,
            "format": self.format.value,
            "duration_seconds": self.duration_seconds,
            "file_size_bytes": self.file_size_bytes,
            "storage_url": self.storage_url,
            "transcription_status": self.transcription_status.value,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TranscriptSegment:
    """Segment of transcription."""
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    speaker: str = ""  # "agent", "user", "unknown"
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0
    words: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Transcription:
    """Complete transcription."""
    transcription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    recording_id: str = ""

    # Status
    status: TranscriptionStatus = TranscriptionStatus.PENDING

    # Content
    full_text: str = ""
    segments: List[TranscriptSegment] = field(default_factory=list)

    # Processing
    provider: TranscriptionProvider = TranscriptionProvider.DEEPGRAM
    language: str = "en"
    model: str = ""

    # Analytics
    word_count: int = 0
    speaker_count: int = 0
    duration_seconds: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transcription_id": self.transcription_id,
            "recording_id": self.recording_id,
            "status": self.status.value,
            "full_text": self.full_text,
            "word_count": self.word_count,
            "speaker_count": self.speaker_count,
            "language": self.language,
            "created_at": self.created_at.isoformat(),
        }


class BaseTranscriptionProvider(ABC):
    """Abstract transcription provider."""

    @property
    @abstractmethod
    def provider_name(self) -> TranscriptionProvider:
        """Provider name."""
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio_url: str,
        language: str = "en",
        **kwargs,
    ) -> Transcription:
        """Transcribe audio file."""
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: str = "en",
        **kwargs,
    ) -> AsyncGenerator[TranscriptSegment, None]:
        """Transcribe audio stream in real-time."""
        pass


class DeepgramProvider(BaseTranscriptionProvider):
    """Deepgram transcription provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "nova-2",
    ):
        self.api_key = api_key
        self.model = model
        self._base_url = "https://api.deepgram.com/v1"

    @property
    def provider_name(self) -> TranscriptionProvider:
        return TranscriptionProvider.DEEPGRAM

    async def transcribe(
        self,
        audio_url: str,
        language: str = "en",
        **kwargs,
    ) -> Transcription:
        """Transcribe audio file."""
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

        params = {
            "model": self.model,
            "language": language,
            "punctuate": True,
            "diarize": True,
            "utterances": True,
            "smart_format": True,
        }
        params.update(kwargs)

        payload = {"url": audio_url}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/listen",
                    headers=headers,
                    params=params,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise Exception(f"Deepgram error: {error}")

                    data = await response.json()
                    return self._parse_response(data, language)

        except Exception as e:
            logger.error(f"Deepgram transcription error: {e}")
            raise

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: str = "en",
        **kwargs,
    ) -> AsyncGenerator[TranscriptSegment, None]:
        """Stream transcription via WebSocket."""
        import websockets

        url = f"wss://api.deepgram.com/v1/listen"
        params = f"?model={self.model}&language={language}&punctuate=true&diarize=true"

        headers = {"Authorization": f"Token {self.api_key}"}

        try:
            async with websockets.connect(
                url + params,
                extra_headers=headers,
            ) as ws:
                # Start receiving task
                async def receive():
                    async for message in ws:
                        data = json.loads(message)
                        if data.get("is_final"):
                            channel = data.get("channel", {})
                            alternatives = channel.get("alternatives", [{}])
                            if alternatives:
                                alt = alternatives[0]
                                yield TranscriptSegment(
                                    text=alt.get("transcript", ""),
                                    confidence=alt.get("confidence", 0),
                                    start_time=data.get("start", 0),
                                    end_time=data.get("start", 0) + data.get("duration", 0),
                                    words=alt.get("words", []),
                                )

                # Send audio
                async def send():
                    async for chunk in audio_stream:
                        await ws.send(chunk)
                    await ws.send(json.dumps({"type": "CloseStream"}))

                # Run concurrently
                send_task = asyncio.create_task(send())

                async for segment in receive():
                    yield segment

                await send_task

        except Exception as e:
            logger.error(f"Deepgram stream error: {e}")
            raise

    def _parse_response(self, data: Dict[str, Any], language: str) -> Transcription:
        """Parse Deepgram response."""
        results = data.get("results", {})
        channels = results.get("channels", [{}])
        channel = channels[0] if channels else {}
        alternatives = channel.get("alternatives", [{}])
        alt = alternatives[0] if alternatives else {}

        segments = []
        utterances = results.get("utterances", [])

        for utt in utterances:
            segments.append(TranscriptSegment(
                text=utt.get("transcript", ""),
                speaker=f"speaker_{utt.get('speaker', 0)}",
                start_time=utt.get("start", 0),
                end_time=utt.get("end", 0),
                confidence=utt.get("confidence", 0),
                words=utt.get("words", []),
            ))

        # Calculate speakers
        speakers = set(s.speaker for s in segments)

        return Transcription(
            status=TranscriptionStatus.COMPLETED,
            full_text=alt.get("transcript", ""),
            segments=segments,
            provider=self.provider_name,
            language=language,
            model=self.model,
            word_count=len(alt.get("transcript", "").split()),
            speaker_count=len(speakers),
            duration_seconds=results.get("metadata", {}).get("duration", 0),
            completed_at=datetime.utcnow(),
        )


class AssemblyAIProvider(BaseTranscriptionProvider):
    """AssemblyAI transcription provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._base_url = "https://api.assemblyai.com/v2"

    @property
    def provider_name(self) -> TranscriptionProvider:
        return TranscriptionProvider.ASSEMBLY_AI

    async def transcribe(
        self,
        audio_url: str,
        language: str = "en",
        **kwargs,
    ) -> Transcription:
        """Transcribe audio file."""
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "audio_url": audio_url,
            "language_code": language,
            "punctuate": True,
            "format_text": True,
            "speaker_labels": True,
        }
        payload.update(kwargs)

        try:
            async with aiohttp.ClientSession() as session:
                # Submit transcription
                async with session.post(
                    f"{self._base_url}/transcript",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise Exception(f"AssemblyAI error: {error}")

                    data = await response.json()
                    transcript_id = data["id"]

                # Poll for completion
                while True:
                    async with session.get(
                        f"{self._base_url}/transcript/{transcript_id}",
                        headers=headers,
                    ) as response:
                        data = await response.json()
                        status = data.get("status")

                        if status == "completed":
                            return self._parse_response(data, language)
                        elif status == "error":
                            raise Exception(f"Transcription failed: {data.get('error')}")

                        await asyncio.sleep(3)

        except Exception as e:
            logger.error(f"AssemblyAI transcription error: {e}")
            raise

    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        language: str = "en",
        **kwargs,
    ) -> AsyncGenerator[TranscriptSegment, None]:
        """Stream transcription."""
        # AssemblyAI real-time would use WebSocket
        raise NotImplementedError("AssemblyAI streaming not implemented")

    def _parse_response(self, data: Dict[str, Any], language: str) -> Transcription:
        """Parse AssemblyAI response."""
        segments = []
        utterances = data.get("utterances", [])

        for utt in utterances:
            segments.append(TranscriptSegment(
                text=utt.get("text", ""),
                speaker=utt.get("speaker", "unknown"),
                start_time=utt.get("start", 0) / 1000,
                end_time=utt.get("end", 0) / 1000,
                confidence=utt.get("confidence", 0),
            ))

        speakers = set(s.speaker for s in segments)

        return Transcription(
            status=TranscriptionStatus.COMPLETED,
            full_text=data.get("text", ""),
            segments=segments,
            provider=self.provider_name,
            language=language,
            word_count=data.get("words", 0) or len(data.get("text", "").split()),
            speaker_count=len(speakers),
            duration_seconds=data.get("audio_duration", 0),
            completed_at=datetime.utcnow(),
        )


class RecordingService:
    """
    Recording management service.

    Features:
    - Recording lifecycle
    - Storage management
    - Transcription integration
    """

    def __init__(self, config: Optional[RecordingConfig] = None):
        self.config = config or RecordingConfig()
        self._recordings: Dict[str, Recording] = {}
        self._transcriptions: Dict[str, Transcription] = {}
        self._transcription_providers: Dict[TranscriptionProvider, BaseTranscriptionProvider] = {}
        self._by_call: Dict[str, List[str]] = {}
        self._by_tenant: Dict[str, List[str]] = {}

    def register_transcription_provider(
        self,
        provider: BaseTranscriptionProvider,
    ) -> None:
        """Register transcription provider."""
        self._transcription_providers[provider.provider_name] = provider

    async def start_recording(
        self,
        call_id: str,
        tenant_id: str,
        agent_id: str = "",
        conversation_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Recording:
        """Start a new recording."""
        recording = Recording(
            call_id=call_id,
            tenant_id=tenant_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            status=RecordingStatus.RECORDING,
            format=self.config.format,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            storage_provider=self.config.storage_provider,
            started_at=datetime.utcnow(),
            metadata=metadata or {},
        )

        self._recordings[recording.recording_id] = recording

        if call_id not in self._by_call:
            self._by_call[call_id] = []
        self._by_call[call_id].append(recording.recording_id)

        if tenant_id not in self._by_tenant:
            self._by_tenant[tenant_id] = []
        self._by_tenant[tenant_id].append(recording.recording_id)

        logger.info(f"Started recording: {recording.recording_id}")
        return recording

    async def stop_recording(
        self,
        recording_id: str,
        audio_data: Optional[bytes] = None,
        audio_url: Optional[str] = None,
    ) -> Recording:
        """Stop recording and save."""
        recording = self._recordings.get(recording_id)
        if not recording:
            raise ValueError(f"Recording not found: {recording_id}")

        recording.status = RecordingStatus.STOPPED
        recording.stopped_at = datetime.utcnow()

        if recording.started_at:
            recording.duration_seconds = (
                recording.stopped_at - recording.started_at
            ).total_seconds()

        # Store audio
        if audio_data:
            await self._store_audio(recording, audio_data)
        elif audio_url:
            recording.storage_url = audio_url

        # Mark completed
        recording.status = RecordingStatus.COMPLETED

        # Auto-transcribe if enabled
        if self.config.auto_transcribe and recording.storage_url:
            asyncio.create_task(self._transcribe_recording(recording))

        logger.info(f"Stopped recording: {recording_id}")
        return recording

    async def _store_audio(
        self,
        recording: Recording,
        audio_data: bytes,
    ) -> None:
        """Store audio data."""
        recording.file_size_bytes = len(audio_data)

        if self.config.storage_provider == StorageProvider.LOCAL:
            # Store locally
            import os
            path = f"{self.config.storage_path}/{recording.tenant_id}"
            os.makedirs(path, exist_ok=True)
            file_path = f"{path}/{recording.recording_id}.{recording.format.value}"

            with open(file_path, "wb") as f:
                f.write(audio_data)

            recording.storage_path = file_path
            recording.storage_url = f"file://{file_path}"

        elif self.config.storage_provider == StorageProvider.S3:
            # Store to S3 (simplified)
            recording.storage_path = f"recordings/{recording.tenant_id}/{recording.recording_id}.{recording.format.value}"

    async def _transcribe_recording(self, recording: Recording) -> None:
        """Transcribe recording asynchronously."""
        provider = self._transcription_providers.get(self.config.transcription_provider)
        if not provider:
            logger.warning(f"Transcription provider not registered: {self.config.transcription_provider}")
            return

        try:
            recording.transcription_status = TranscriptionStatus.IN_PROGRESS

            transcription = await provider.transcribe(recording.storage_url)
            transcription.recording_id = recording.recording_id

            self._transcriptions[transcription.transcription_id] = transcription
            recording.transcription_id = transcription.transcription_id
            recording.transcription_status = TranscriptionStatus.COMPLETED

            logger.info(f"Transcription completed: {transcription.transcription_id}")

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            recording.transcription_status = TranscriptionStatus.FAILED

    async def transcribe(
        self,
        recording_id: str,
        provider: Optional[TranscriptionProvider] = None,
    ) -> Transcription:
        """Manually transcribe a recording."""
        recording = self._recordings.get(recording_id)
        if not recording:
            raise ValueError(f"Recording not found: {recording_id}")

        if not recording.storage_url:
            raise ValueError("Recording has no audio URL")

        provider_type = provider or self.config.transcription_provider
        transcription_provider = self._transcription_providers.get(provider_type)

        if not transcription_provider:
            raise ValueError(f"Transcription provider not registered: {provider_type}")

        recording.transcription_status = TranscriptionStatus.IN_PROGRESS

        transcription = await transcription_provider.transcribe(recording.storage_url)
        transcription.recording_id = recording_id

        self._transcriptions[transcription.transcription_id] = transcription
        recording.transcription_id = transcription.transcription_id
        recording.transcription_status = TranscriptionStatus.COMPLETED

        return transcription

    def get_recording(self, recording_id: str) -> Optional[Recording]:
        """Get recording by ID."""
        return self._recordings.get(recording_id)

    def get_transcription(self, transcription_id: str) -> Optional[Transcription]:
        """Get transcription by ID."""
        return self._transcriptions.get(transcription_id)

    def get_recording_transcription(
        self,
        recording_id: str,
    ) -> Optional[Transcription]:
        """Get transcription for a recording."""
        recording = self._recordings.get(recording_id)
        if recording and recording.transcription_id:
            return self._transcriptions.get(recording.transcription_id)
        return None

    def list_by_call(self, call_id: str) -> List[Recording]:
        """List recordings by call."""
        recording_ids = self._by_call.get(call_id, [])
        return [
            self._recordings[rid]
            for rid in recording_ids
            if rid in self._recordings
        ]

    def list_by_tenant(
        self,
        tenant_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Recording]:
        """List recordings by tenant."""
        recording_ids = self._by_tenant.get(tenant_id, [])
        return [
            self._recordings[rid]
            for rid in recording_ids[offset:offset + limit]
            if rid in self._recordings
        ]

    async def delete_recording(self, recording_id: str) -> bool:
        """Delete a recording."""
        recording = self._recordings.get(recording_id)
        if not recording:
            return False

        # Delete file if local
        if recording.storage_provider == StorageProvider.LOCAL and recording.storage_path:
            import os
            if os.path.exists(recording.storage_path):
                os.remove(recording.storage_path)

        # Remove from indices
        if recording.call_id in self._by_call:
            self._by_call[recording.call_id].remove(recording_id)
        if recording.tenant_id in self._by_tenant:
            self._by_tenant[recording.tenant_id].remove(recording_id)

        # Delete transcription
        if recording.transcription_id:
            self._transcriptions.pop(recording.transcription_id, None)

        del self._recordings[recording_id]
        logger.info(f"Deleted recording: {recording_id}")
        return True

    async def cleanup_old_recordings(self) -> int:
        """Clean up recordings past retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.config.retention_days)
        to_delete = []

        for recording_id, recording in self._recordings.items():
            if recording.created_at < cutoff:
                to_delete.append(recording_id)

        deleted = 0
        for recording_id in to_delete:
            if await self.delete_recording(recording_id):
                deleted += 1

        logger.info(f"Cleaned up {deleted} old recordings")
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        recordings = list(self._recordings.values())
        transcriptions = list(self._transcriptions.values())

        total_duration = sum(r.duration_seconds for r in recordings)
        total_size = sum(r.file_size_bytes for r in recordings)

        return {
            "total_recordings": len(recordings),
            "total_transcriptions": len(transcriptions),
            "total_duration_seconds": total_duration,
            "total_size_bytes": total_size,
            "by_status": {
                status.value: sum(1 for r in recordings if r.status == status)
                for status in RecordingStatus
            },
            "transcription_rate": len(transcriptions) / len(recordings) if recordings else 0,
        }


# Singleton service
_service_instance: Optional[RecordingService] = None


def get_recording_service(
    config: Optional[RecordingConfig] = None,
) -> RecordingService:
    """Get singleton recording service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = RecordingService(config)
    return _service_instance
