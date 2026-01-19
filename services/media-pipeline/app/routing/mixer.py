"""Audio mixer for combining multiple streams."""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass, field
import threading


@dataclass
class MixerChannel:
    """Audio mixer channel."""
    id: str
    volume: float = 1.0
    pan: float = 0.0  # -1.0 (left) to 1.0 (right)
    muted: bool = False
    solo: bool = False


class AudioMixer:
    """
    Real-time audio mixer.

    Features:
    - Multiple input channels
    - Volume control per channel
    - Panning (for stereo)
    - Solo/mute
    - Clipping prevention
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        channels: int = 1,
        max_inputs: int = 8,
    ):
        self.sample_rate = sample_rate
        self.output_channels = channels
        self.max_inputs = max_inputs

        # Mixer channels
        self._channels: Dict[str, MixerChannel] = {}
        self._channel_buffers: Dict[str, List[bytes]] = {}
        self._lock = threading.Lock()

        # Master settings
        self._master_volume = 1.0
        self._limiter_threshold = 0.95

        # Statistics
        self._frames_mixed = 0
        self._peak_level = 0.0

    def add_channel(
        self,
        channel_id: str,
        volume: float = 1.0,
        pan: float = 0.0,
    ) -> bool:
        """
        Add a mixer channel.

        Args:
            channel_id: Unique channel identifier
            volume: Initial volume (0.0-1.0)
            pan: Pan position (-1.0 to 1.0)

        Returns:
            True if channel added
        """
        with self._lock:
            if len(self._channels) >= self.max_inputs:
                return False

            if channel_id in self._channels:
                return False

            self._channels[channel_id] = MixerChannel(
                id=channel_id,
                volume=volume,
                pan=pan,
            )
            self._channel_buffers[channel_id] = []
            return True

    def remove_channel(self, channel_id: str) -> bool:
        """Remove a mixer channel."""
        with self._lock:
            if channel_id in self._channels:
                del self._channels[channel_id]
                del self._channel_buffers[channel_id]
                return True
            return False

    def push_audio(self, channel_id: str, audio_data: bytes) -> bool:
        """
        Push audio data to a channel.

        Args:
            channel_id: Channel to push to
            audio_data: PCM audio bytes

        Returns:
            True if audio pushed
        """
        with self._lock:
            if channel_id not in self._channels:
                return False

            self._channel_buffers[channel_id].append(audio_data)
            return True

    def mix(self, frame_samples: int = 160) -> bytes:
        """
        Mix all channels and produce output frame.

        Args:
            frame_samples: Number of samples per frame

        Returns:
            Mixed PCM audio bytes
        """
        with self._lock:
            # Check for solo channels
            solo_active = any(ch.solo for ch in self._channels.values())

            # Collect audio from all active channels
            channel_audio: List[np.ndarray] = []
            channel_volumes: List[float] = []

            for channel_id, channel in self._channels.items():
                # Skip muted or non-solo channels when solo active
                if channel.muted:
                    continue
                if solo_active and not channel.solo:
                    continue

                # Get audio from buffer
                audio = self._get_channel_audio(channel_id, frame_samples)
                if audio is not None:
                    channel_audio.append(audio)
                    channel_volumes.append(channel.volume)

            # No audio to mix
            if not channel_audio:
                return bytes(frame_samples * 2)  # Silence

            # Mix channels
            mixed = self._mix_channels(channel_audio, channel_volumes)

            # Apply master volume
            mixed = mixed * self._master_volume

            # Apply limiter
            mixed = self._soft_limit(mixed)

            # Update statistics
            self._frames_mixed += 1
            peak = np.max(np.abs(mixed))
            self._peak_level = max(self._peak_level * 0.99, peak / 32768.0)

            return mixed.astype(np.int16).tobytes()

    def _get_channel_audio(
        self,
        channel_id: str,
        frame_samples: int,
    ) -> Optional[np.ndarray]:
        """Get audio from channel buffer."""
        buffer = self._channel_buffers.get(channel_id, [])

        if not buffer:
            return None

        # Collect enough samples
        collected = bytearray()
        needed_bytes = frame_samples * 2

        while len(collected) < needed_bytes and buffer:
            chunk = buffer.pop(0)
            collected.extend(chunk)

        if len(collected) < needed_bytes:
            # Pad with silence
            collected.extend(bytes(needed_bytes - len(collected)))

        # Convert to numpy
        return np.frombuffer(bytes(collected[:needed_bytes]), dtype=np.int16).astype(np.float32)

    def _mix_channels(
        self,
        channels: List[np.ndarray],
        volumes: List[float],
    ) -> np.ndarray:
        """Mix multiple channels with volume weighting."""
        if len(channels) == 1:
            return channels[0] * volumes[0]

        # Sum with volume
        mixed = np.zeros_like(channels[0])
        for audio, volume in zip(channels, volumes):
            mixed += audio * volume

        # Normalize to prevent clipping
        # Use equal-power mixing
        mixed = mixed / np.sqrt(len(channels))

        return mixed

    def _soft_limit(self, samples: np.ndarray) -> np.ndarray:
        """Apply soft limiting to prevent clipping."""
        threshold = 32768 * self._limiter_threshold

        # Find samples above threshold
        abs_samples = np.abs(samples)
        mask = abs_samples > threshold

        if np.any(mask):
            # Soft knee compression
            excess = abs_samples[mask] - threshold
            compressed = threshold + excess * 0.25
            samples[mask] = np.sign(samples[mask]) * compressed

        return samples

    def set_channel_volume(self, channel_id: str, volume: float) -> bool:
        """Set channel volume."""
        with self._lock:
            if channel_id in self._channels:
                self._channels[channel_id].volume = max(0.0, min(2.0, volume))
                return True
            return False

    def set_channel_mute(self, channel_id: str, muted: bool) -> bool:
        """Set channel mute state."""
        with self._lock:
            if channel_id in self._channels:
                self._channels[channel_id].muted = muted
                return True
            return False

    def set_channel_solo(self, channel_id: str, solo: bool) -> bool:
        """Set channel solo state."""
        with self._lock:
            if channel_id in self._channels:
                self._channels[channel_id].solo = solo
                return True
            return False

    def set_master_volume(self, volume: float) -> None:
        """Set master output volume."""
        self._master_volume = max(0.0, min(2.0, volume))

    def get_statistics(self) -> dict:
        """Get mixer statistics."""
        with self._lock:
            return {
                "active_channels": len(self._channels),
                "frames_mixed": self._frames_mixed,
                "peak_level": round(self._peak_level, 3),
                "master_volume": self._master_volume,
                "channels": {
                    ch_id: {
                        "volume": ch.volume,
                        "muted": ch.muted,
                        "solo": ch.solo,
                        "buffer_chunks": len(self._channel_buffers.get(ch_id, [])),
                    }
                    for ch_id, ch in self._channels.items()
                },
            }

    def clear_all(self) -> None:
        """Clear all channel buffers."""
        with self._lock:
            for buffer in self._channel_buffers.values():
                buffer.clear()
