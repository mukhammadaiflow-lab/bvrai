"""Audio recording for calls."""

from typing import Optional, Dict, Any, List, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import asyncio
import logging
import struct
import uuid
import io

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


@dataclass
class RecordingConfig:
    """Configuration for recording."""
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16
    format: str = "wav"  # wav, mp3, ogg
    stereo_recording: bool = True  # Separate channels for user/agent
    max_duration_seconds: int = 3600  # 1 hour max
    silence_detection: bool = True
    silence_threshold_db: float = -40.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bits_per_sample": self.bits_per_sample,
            "format": self.format,
            "stereo_recording": self.stereo_recording,
            "max_duration_seconds": self.max_duration_seconds,
        }


@dataclass
class Recording:
    """Recording metadata."""
    recording_id: str
    call_id: str
    status: RecordingStatus = RecordingStatus.PENDING
    config: RecordingConfig = field(default_factory=RecordingConfig)
    file_path: Optional[str] = None
    file_size: int = 0
    duration_seconds: float = 0.0
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recording_id": self.recording_id,
            "call_id": self.call_id,
            "status": self.status.value,
            "config": self.config.to_dict(),
            "file_path": self.file_path,
            "file_size": self.file_size,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "metadata": self.metadata,
        }


class AudioRecorder:
    """
    Audio recorder for capturing audio streams.

    Handles buffering, format conversion, and writing to file.

    Usage:
        recorder = AudioRecorder(config)
        await recorder.start()

        await recorder.write_audio(audio_chunk, channel="user")
        await recorder.write_audio(audio_chunk, channel="agent")

        recording = await recorder.stop()
    """

    def __init__(
        self,
        config: Optional[RecordingConfig] = None,
        output_path: Optional[str] = None,
    ):
        self.config = config or RecordingConfig()
        self.output_path = output_path

        self._recording_id = str(uuid.uuid4())
        self._status = RecordingStatus.PENDING
        self._started_at: Optional[datetime] = None
        self._stopped_at: Optional[datetime] = None

        # Audio buffers for each channel
        self._user_buffer = bytearray()
        self._agent_buffer = bytearray()
        self._mixed_buffer = bytearray()

        self._total_samples = 0
        self._lock = asyncio.Lock()

    @property
    def status(self) -> RecordingStatus:
        """Get current status."""
        return self._status

    @property
    def duration_seconds(self) -> float:
        """Get current duration in seconds."""
        bytes_per_sample = self.config.bits_per_sample // 8
        return self._total_samples / (self.config.sample_rate * bytes_per_sample)

    async def start(self) -> None:
        """Start recording."""
        if self._status != RecordingStatus.PENDING:
            raise ValueError(f"Cannot start recording in {self._status} state")

        self._status = RecordingStatus.RECORDING
        self._started_at = datetime.utcnow()
        logger.info(f"Recording started: {self._recording_id}")

    async def stop(self) -> Recording:
        """Stop recording and return metadata."""
        if self._status not in [RecordingStatus.RECORDING, RecordingStatus.PAUSED]:
            raise ValueError(f"Cannot stop recording in {self._status} state")

        self._status = RecordingStatus.STOPPED
        self._stopped_at = datetime.utcnow()

        # Generate output file
        file_path = await self._write_output()

        recording = Recording(
            recording_id=self._recording_id,
            call_id="",  # Set by caller
            status=RecordingStatus.COMPLETED,
            config=self.config,
            file_path=file_path,
            file_size=len(self._mixed_buffer),
            duration_seconds=self.duration_seconds,
            started_at=self._started_at,
            stopped_at=self._stopped_at,
        )

        logger.info(
            f"Recording stopped: {self._recording_id}, "
            f"duration={recording.duration_seconds:.2f}s"
        )

        return recording

    async def pause(self) -> None:
        """Pause recording."""
        if self._status != RecordingStatus.RECORDING:
            return
        self._status = RecordingStatus.PAUSED
        logger.debug(f"Recording paused: {self._recording_id}")

    async def resume(self) -> None:
        """Resume recording."""
        if self._status != RecordingStatus.PAUSED:
            return
        self._status = RecordingStatus.RECORDING
        logger.debug(f"Recording resumed: {self._recording_id}")

    async def write_audio(
        self,
        audio_data: bytes,
        channel: str = "mixed",
    ) -> None:
        """
        Write audio data.

        Args:
            audio_data: Raw PCM audio data
            channel: "user", "agent", or "mixed"
        """
        if self._status != RecordingStatus.RECORDING:
            return

        # Check max duration
        if self.duration_seconds >= self.config.max_duration_seconds:
            logger.warning(f"Recording max duration reached: {self._recording_id}")
            return

        async with self._lock:
            if channel == "user":
                self._user_buffer.extend(audio_data)
            elif channel == "agent":
                self._agent_buffer.extend(audio_data)
            else:
                self._mixed_buffer.extend(audio_data)

            self._total_samples += len(audio_data)

    async def write_stereo(
        self,
        user_audio: bytes,
        agent_audio: bytes,
    ) -> None:
        """Write stereo audio with user on left, agent on right."""
        if self._status != RecordingStatus.RECORDING:
            return

        async with self._lock:
            self._user_buffer.extend(user_audio)
            self._agent_buffer.extend(agent_audio)

            # Mix to stereo
            stereo = self._interleave_stereo(user_audio, agent_audio)
            self._mixed_buffer.extend(stereo)

            self._total_samples += len(user_audio)

    def _interleave_stereo(self, left: bytes, right: bytes) -> bytes:
        """Interleave two mono channels into stereo."""
        bytes_per_sample = self.config.bits_per_sample // 8
        output = bytearray()

        # Pad shorter buffer
        max_len = max(len(left), len(right))
        left = left.ljust(max_len, b'\x00')
        right = right.ljust(max_len, b'\x00')

        for i in range(0, max_len, bytes_per_sample):
            output.extend(left[i:i + bytes_per_sample])
            output.extend(right[i:i + bytes_per_sample])

        return bytes(output)

    async def _write_output(self) -> str:
        """Write output file."""
        # Determine output path
        if self.output_path:
            file_path = self.output_path
        else:
            file_path = f"/tmp/recordings/{self._recording_id}.{self.config.format}"

        # Create directory
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Prepare audio data
        if self.config.stereo_recording and self._user_buffer and self._agent_buffer:
            audio_data = self._interleave_stereo(
                bytes(self._user_buffer),
                bytes(self._agent_buffer),
            )
            channels = 2
        else:
            audio_data = bytes(self._mixed_buffer)
            channels = self.config.channels

        # Write file based on format
        if self.config.format == "wav":
            await self._write_wav(file_path, audio_data, channels)
        else:
            # For other formats, write WAV and convert (simplified)
            await self._write_wav(file_path, audio_data, channels)

        return file_path

    async def _write_wav(
        self,
        file_path: str,
        audio_data: bytes,
        channels: int,
    ) -> None:
        """Write WAV file."""
        sample_rate = self.config.sample_rate
        bits_per_sample = self.config.bits_per_sample
        byte_rate = sample_rate * channels * (bits_per_sample // 8)
        block_align = channels * (bits_per_sample // 8)

        with open(file_path, 'wb') as f:
            # RIFF header
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + len(audio_data)))
            f.write(b'WAVE')

            # fmt chunk
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))  # Chunk size
            f.write(struct.pack('<H', 1))   # Audio format (PCM)
            f.write(struct.pack('<H', channels))
            f.write(struct.pack('<I', sample_rate))
            f.write(struct.pack('<I', byte_rate))
            f.write(struct.pack('<H', block_align))
            f.write(struct.pack('<H', bits_per_sample))

            # data chunk
            f.write(b'data')
            f.write(struct.pack('<I', len(audio_data)))
            f.write(audio_data)

    def get_buffer(self, channel: str = "mixed") -> bytes:
        """Get audio buffer for a channel."""
        if channel == "user":
            return bytes(self._user_buffer)
        elif channel == "agent":
            return bytes(self._agent_buffer)
        return bytes(self._mixed_buffer)


class CallRecorder:
    """
    High-level call recorder.

    Manages recording lifecycle for a call.

    Usage:
        recorder = CallRecorder(call_id="call-123")
        await recorder.start()

        # During call
        await recorder.record_user_audio(audio)
        await recorder.record_agent_audio(audio)

        # End call
        recording = await recorder.finalize()
    """

    def __init__(
        self,
        call_id: str,
        config: Optional[RecordingConfig] = None,
        storage_path: Optional[str] = None,
    ):
        self.call_id = call_id
        self.config = config or RecordingConfig()
        self.storage_path = storage_path or f"/tmp/recordings/{call_id}"

        self._recorder = AudioRecorder(
            config=self.config,
            output_path=f"{self.storage_path}/recording.wav",
        )
        self._recording: Optional[Recording] = None
        self._segments: List[Dict[str, Any]] = []

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recorder.status == RecordingStatus.RECORDING

    async def start(self) -> None:
        """Start recording the call."""
        await self._recorder.start()
        logger.info(f"Call recording started: {self.call_id}")

    async def stop(self) -> Recording:
        """Stop recording and return metadata."""
        recording = await self._recorder.stop()
        recording.call_id = self.call_id
        recording.metadata["segments"] = self._segments
        self._recording = recording
        return recording

    async def record_user_audio(self, audio_data: bytes) -> None:
        """Record user audio."""
        await self._recorder.write_audio(audio_data, channel="user")

    async def record_agent_audio(self, audio_data: bytes) -> None:
        """Record agent audio."""
        await self._recorder.write_audio(audio_data, channel="agent")

    async def record_mixed_audio(self, audio_data: bytes) -> None:
        """Record mixed audio."""
        await self._recorder.write_audio(audio_data, channel="mixed")

    def add_segment(
        self,
        speaker: str,
        text: str,
        start_time: float,
        end_time: float,
        confidence: float = 1.0,
    ) -> None:
        """Add a transcript segment."""
        self._segments.append({
            "speaker": speaker,
            "text": text,
            "start_time": start_time,
            "end_time": end_time,
            "confidence": confidence,
        })

    async def pause(self) -> None:
        """Pause recording."""
        await self._recorder.pause()

    async def resume(self) -> None:
        """Resume recording."""
        await self._recorder.resume()

    def get_recording(self) -> Optional[Recording]:
        """Get the recording metadata."""
        return self._recording

    def get_segments(self) -> List[Dict[str, Any]]:
        """Get transcript segments."""
        return self._segments.copy()

    async def finalize(
        self,
        upload: bool = True,
        storage_manager: Optional[Any] = None,
    ) -> Recording:
        """Finalize recording and optionally upload to storage."""
        if self._recorder.status == RecordingStatus.RECORDING:
            recording = await self.stop()
        elif self._recording:
            recording = self._recording
        else:
            raise ValueError("No recording to finalize")

        # Upload to storage if requested
        if upload and storage_manager and recording.file_path:
            try:
                remote_path = await storage_manager.upload(
                    recording.file_path,
                    f"recordings/{self.call_id}/{recording.recording_id}.wav",
                )
                recording.metadata["remote_path"] = remote_path
                recording.metadata["uploaded"] = True
            except Exception as e:
                logger.error(f"Failed to upload recording: {e}")
                recording.metadata["upload_error"] = str(e)

        return recording


class MultiTrackRecorder:
    """
    Multi-track recorder for complex recording scenarios.

    Supports multiple audio tracks with synchronization.
    """

    def __init__(
        self,
        recording_id: str,
        config: Optional[RecordingConfig] = None,
    ):
        self.recording_id = recording_id
        self.config = config or RecordingConfig()

        self._tracks: Dict[str, AudioRecorder] = {}
        self._started_at: Optional[datetime] = None
        self._status = RecordingStatus.PENDING
        self._lock = asyncio.Lock()

    async def add_track(self, track_id: str) -> None:
        """Add a new track."""
        async with self._lock:
            if track_id not in self._tracks:
                self._tracks[track_id] = AudioRecorder(config=self.config)
                if self._status == RecordingStatus.RECORDING:
                    await self._tracks[track_id].start()

    async def remove_track(self, track_id: str) -> Optional[bytes]:
        """Remove a track and return its audio."""
        async with self._lock:
            if track_id in self._tracks:
                recorder = self._tracks.pop(track_id)
                return recorder.get_buffer()
            return None

    async def start(self) -> None:
        """Start all tracks."""
        self._status = RecordingStatus.RECORDING
        self._started_at = datetime.utcnow()

        for recorder in self._tracks.values():
            await recorder.start()

    async def stop(self) -> Dict[str, Recording]:
        """Stop all tracks and return recordings."""
        self._status = RecordingStatus.STOPPED
        recordings = {}

        for track_id, recorder in self._tracks.items():
            try:
                recordings[track_id] = await recorder.stop()
            except Exception as e:
                logger.error(f"Failed to stop track {track_id}: {e}")

        return recordings

    async def write_to_track(
        self,
        track_id: str,
        audio_data: bytes,
    ) -> None:
        """Write audio to a specific track."""
        if track_id in self._tracks:
            await self._tracks[track_id].write_audio(audio_data)

    def get_track_ids(self) -> List[str]:
        """Get all track IDs."""
        return list(self._tracks.keys())
