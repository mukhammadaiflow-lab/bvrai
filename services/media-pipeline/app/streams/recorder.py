"""Audio stream recorder for call recording."""

import asyncio
import struct
import time
from typing import Optional, BinaryIO
from pathlib import Path
from dataclasses import dataclass
import structlog
import os


logger = structlog.get_logger()


@dataclass
class RecordingConfig:
    """Recording configuration."""
    session_id: str
    output_dir: str = "/recordings"
    format: str = "wav"  # wav, raw
    sample_rate: int = 8000
    channels: int = 1
    bits_per_sample: int = 16
    max_duration_seconds: int = 3600  # 1 hour max
    buffer_size: int = 32000  # Buffer before writing


class StreamRecorder:
    """
    Records audio streams to file.

    Features:
    - WAV format output
    - Separate tracks for caller/agent
    - Duration limiting
    - Async buffered writing
    """

    def __init__(self, config: RecordingConfig):
        self.config = config
        self.session_id = config.session_id

        # File handles
        self._caller_file: Optional[BinaryIO] = None
        self._agent_file: Optional[BinaryIO] = None
        self._mixed_file: Optional[BinaryIO] = None

        # Paths
        self._caller_path: Optional[Path] = None
        self._agent_path: Optional[Path] = None
        self._mixed_path: Optional[Path] = None

        # Buffers
        self._caller_buffer = bytearray()
        self._agent_buffer = bytearray()
        self._mixed_buffer = bytearray()

        # State
        self._recording = False
        self._start_time: Optional[float] = None
        self._flush_task: Optional[asyncio.Task] = None

        # Statistics
        self._caller_bytes = 0
        self._agent_bytes = 0
        self._mixed_bytes = 0

    async def start(self) -> dict:
        """
        Start recording.

        Returns:
            Dict with file paths
        """
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())

        # Create file paths
        self._caller_path = output_dir / f"{self.session_id}_{timestamp}_caller.wav"
        self._agent_path = output_dir / f"{self.session_id}_{timestamp}_agent.wav"
        self._mixed_path = output_dir / f"{self.session_id}_{timestamp}_mixed.wav"

        # Open files and write WAV headers
        self._caller_file = open(self._caller_path, "wb")
        self._agent_file = open(self._agent_path, "wb")
        self._mixed_file = open(self._mixed_path, "wb")

        # Write placeholder headers (will update on close)
        for f in [self._caller_file, self._agent_file, self._mixed_file]:
            self._write_wav_header(f, 0)

        self._recording = True
        self._start_time = time.time()

        # Start flush task
        self._flush_task = asyncio.create_task(self._flush_loop())

        logger.info(
            "recording_started",
            session_id=self.session_id,
            caller_path=str(self._caller_path),
            agent_path=str(self._agent_path),
            mixed_path=str(self._mixed_path),
        )

        return {
            "caller": str(self._caller_path),
            "agent": str(self._agent_path),
            "mixed": str(self._mixed_path),
        }

    async def stop(self) -> dict:
        """
        Stop recording and finalize files.

        Returns:
            Recording summary
        """
        self._recording = False

        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_buffers()

        # Update WAV headers with final sizes
        duration = time.time() - (self._start_time or time.time())

        for f, bytes_written in [
            (self._caller_file, self._caller_bytes),
            (self._agent_file, self._agent_bytes),
            (self._mixed_file, self._mixed_bytes),
        ]:
            if f:
                self._finalize_wav(f, bytes_written)
                f.close()

        logger.info(
            "recording_stopped",
            session_id=self.session_id,
            duration_seconds=round(duration, 2),
            caller_bytes=self._caller_bytes,
            agent_bytes=self._agent_bytes,
            mixed_bytes=self._mixed_bytes,
        )

        return {
            "session_id": self.session_id,
            "duration_seconds": round(duration, 2),
            "files": {
                "caller": str(self._caller_path) if self._caller_path else None,
                "agent": str(self._agent_path) if self._agent_path else None,
                "mixed": str(self._mixed_path) if self._mixed_path else None,
            },
            "sizes": {
                "caller_bytes": self._caller_bytes,
                "agent_bytes": self._agent_bytes,
                "mixed_bytes": self._mixed_bytes,
            },
        }

    async def record_caller(self, audio_data: bytes) -> None:
        """Record caller audio."""
        if not self._recording:
            return

        self._caller_buffer.extend(audio_data)
        self._mixed_buffer.extend(audio_data)  # Add to mixed

        # Check duration limit
        if self._check_duration_limit():
            await self.stop()

    async def record_agent(self, audio_data: bytes) -> None:
        """Record agent audio."""
        if not self._recording:
            return

        self._agent_buffer.extend(audio_data)
        # For mixed: would need proper mixing with caller audio
        # Simplified: just track separately

    async def _flush_loop(self) -> None:
        """Periodic buffer flush."""
        while self._recording:
            try:
                await asyncio.sleep(1)  # Flush every second
                await self._flush_buffers()
            except asyncio.CancelledError:
                break

    async def _flush_buffers(self) -> None:
        """Flush all buffers to disk."""
        if self._caller_buffer and self._caller_file:
            self._caller_file.write(bytes(self._caller_buffer))
            self._caller_bytes += len(self._caller_buffer)
            self._caller_buffer.clear()

        if self._agent_buffer and self._agent_file:
            self._agent_file.write(bytes(self._agent_buffer))
            self._agent_bytes += len(self._agent_buffer)
            self._agent_buffer.clear()

        if self._mixed_buffer and self._mixed_file:
            self._mixed_file.write(bytes(self._mixed_buffer))
            self._mixed_bytes += len(self._mixed_buffer)
            self._mixed_buffer.clear()

    def _check_duration_limit(self) -> bool:
        """Check if recording has reached duration limit."""
        if self._start_time is None:
            return False

        duration = time.time() - self._start_time
        return duration >= self.config.max_duration_seconds

    def _write_wav_header(self, f: BinaryIO, data_size: int) -> None:
        """Write WAV file header."""
        sample_rate = self.config.sample_rate
        channels = self.config.channels
        bits_per_sample = self.config.bits_per_sample
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8

        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))  # File size - 8
        f.write(b"WAVE")

        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # Chunk size
        f.write(struct.pack("<H", 1))  # Audio format (PCM)
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))

    def _finalize_wav(self, f: BinaryIO, data_size: int) -> None:
        """Update WAV header with final size."""
        f.seek(0)
        self._write_wav_header(f, data_size)

    def get_statistics(self) -> dict:
        """Get recording statistics."""
        duration = 0
        if self._start_time:
            duration = time.time() - self._start_time

        return {
            "session_id": self.session_id,
            "recording": self._recording,
            "duration_seconds": round(duration, 2),
            "caller_bytes": self._caller_bytes + len(self._caller_buffer),
            "agent_bytes": self._agent_bytes + len(self._agent_buffer),
            "mixed_bytes": self._mixed_bytes + len(self._mixed_buffer),
        }

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording
