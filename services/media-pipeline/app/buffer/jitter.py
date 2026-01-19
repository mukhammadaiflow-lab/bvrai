"""Jitter buffer for real-time audio streaming."""

import heapq
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import threading


@dataclass(order=True)
class AudioPacket:
    """Audio packet with timestamp for ordering."""
    timestamp: int
    data: bytes = field(compare=False)
    received_at: float = field(default_factory=time.time, compare=False)


class JitterBuffer:
    """
    Adaptive jitter buffer for handling network timing variations.

    Features:
    - Packet reordering
    - Gap filling with silence
    - Adaptive buffer sizing
    - Late packet handling
    - Statistics tracking
    """

    def __init__(
        self,
        buffer_ms: int = 100,
        min_buffer_ms: int = 40,
        max_buffer_ms: int = 500,
        sample_rate: int = 8000,
        samples_per_packet: int = 160,  # 20ms at 8kHz
    ):
        self.target_buffer_ms = buffer_ms
        self.min_buffer_ms = min_buffer_ms
        self.max_buffer_ms = max_buffer_ms
        self.sample_rate = sample_rate
        self.samples_per_packet = samples_per_packet

        # Buffer storage (min-heap by timestamp)
        self._buffer: List[AudioPacket] = []
        self._lock = threading.Lock()

        # Playback state
        self._playback_timestamp: Optional[int] = None
        self._last_played_timestamp: Optional[int] = None

        # Adaptive buffering
        self._current_buffer_ms = buffer_ms
        self._jitter_samples: deque = deque(maxlen=100)

        # Statistics
        self._packets_received = 0
        self._packets_played = 0
        self._packets_dropped_late = 0
        self._packets_dropped_early = 0
        self._gaps_filled = 0
        self._reorders = 0

        # Silence frame for gap filling
        self._silence_frame = bytes(samples_per_packet * 2)  # 16-bit silence

    def push(self, audio_data: bytes, timestamp: int) -> None:
        """
        Add audio packet to buffer.

        Args:
            audio_data: PCM audio bytes
            timestamp: RTP timestamp
        """
        packet = AudioPacket(timestamp=timestamp, data=audio_data)

        with self._lock:
            self._packets_received += 1

            # Check for late packet
            if self._playback_timestamp is not None:
                if timestamp < self._playback_timestamp - self.samples_per_packet:
                    self._packets_dropped_late += 1
                    return

            # Check for duplicate
            for existing in self._buffer:
                if existing.timestamp == timestamp:
                    return

            # Add to heap
            heapq.heappush(self._buffer, packet)

            # Track reordering
            if self._last_played_timestamp is not None:
                if timestamp < self._last_played_timestamp:
                    self._reorders += 1

            # Update jitter estimate
            self._update_jitter(packet)

            # Limit buffer size
            self._trim_buffer()

    def pop(self) -> Optional[bytes]:
        """
        Get next audio frame for playback.

        Returns:
            Audio bytes or None if buffer underrun
        """
        with self._lock:
            # Initialize playback timestamp
            if self._playback_timestamp is None:
                if len(self._buffer) < self._min_packets():
                    return None
                # Start playback from oldest packet
                self._playback_timestamp = self._buffer[0].timestamp

            # Look for packet at playback timestamp
            packet = self._find_packet(self._playback_timestamp)

            if packet:
                # Found packet at expected timestamp
                self._packets_played += 1
                self._last_played_timestamp = packet.timestamp
            else:
                # Gap - generate silence
                self._gaps_filled += 1
                packet = AudioPacket(
                    timestamp=self._playback_timestamp,
                    data=self._silence_frame,
                )

            # Advance playback timestamp
            self._playback_timestamp += self.samples_per_packet

            return packet.data

    def _find_packet(self, timestamp: int) -> Optional[AudioPacket]:
        """Find and remove packet at timestamp."""
        tolerance = self.samples_per_packet // 2

        for i, packet in enumerate(self._buffer):
            if abs(packet.timestamp - timestamp) <= tolerance:
                # Remove from heap (swap with last, pop, re-heapify)
                self._buffer[i] = self._buffer[-1]
                self._buffer.pop()
                if self._buffer:
                    heapq.heapify(self._buffer)
                return packet

        return None

    def _min_packets(self) -> int:
        """Calculate minimum packets before playback starts."""
        ms_per_packet = self.samples_per_packet * 1000 / self.sample_rate
        return max(2, int(self._current_buffer_ms / ms_per_packet))

    def _update_jitter(self, packet: AudioPacket) -> None:
        """Update jitter estimate for adaptive buffering."""
        if len(self._buffer) < 2:
            return

        # Calculate inter-arrival jitter
        prev_packet = self._buffer[-2] if len(self._buffer) >= 2 else None
        if prev_packet:
            expected_delta = (packet.timestamp - prev_packet.timestamp) / self.sample_rate
            actual_delta = packet.received_at - prev_packet.received_at
            jitter = abs(expected_delta - actual_delta) * 1000  # ms

            self._jitter_samples.append(jitter)

            # Adjust buffer based on jitter
            if len(self._jitter_samples) >= 10:
                avg_jitter = sum(self._jitter_samples) / len(self._jitter_samples)
                max_jitter = max(self._jitter_samples)

                # Target buffer = 2x average jitter + margin
                target = avg_jitter * 2 + 20

                # Clamp to limits
                target = max(self.min_buffer_ms, min(self.max_buffer_ms, target))

                # Smooth adjustment
                self._current_buffer_ms = (
                    self._current_buffer_ms * 0.9 + target * 0.1
                )

    def _trim_buffer(self) -> None:
        """Remove packets that are too old."""
        if self._playback_timestamp is None:
            return

        max_age = self.max_buffer_ms * self.sample_rate / 1000

        while self._buffer:
            oldest = self._buffer[0]
            if oldest.timestamp < self._playback_timestamp - max_age:
                heapq.heappop(self._buffer)
                self._packets_dropped_early += 1
            else:
                break

    def get_buffer_level(self) -> int:
        """Get current buffer level in packets."""
        with self._lock:
            return len(self._buffer)

    def get_buffer_ms(self) -> float:
        """Get current buffer level in milliseconds."""
        packets = self.get_buffer_level()
        ms_per_packet = self.samples_per_packet * 1000 / self.sample_rate
        return packets * ms_per_packet

    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                "packets_received": self._packets_received,
                "packets_played": self._packets_played,
                "packets_dropped_late": self._packets_dropped_late,
                "packets_dropped_early": self._packets_dropped_early,
                "gaps_filled": self._gaps_filled,
                "reorders": self._reorders,
                "buffer_level_packets": len(self._buffer),
                "buffer_level_ms": round(self.get_buffer_ms(), 2),
                "adaptive_buffer_ms": round(self._current_buffer_ms, 2),
                "avg_jitter_ms": round(
                    sum(self._jitter_samples) / max(1, len(self._jitter_samples)), 2
                ),
            }

    def reset(self) -> None:
        """Reset buffer state."""
        with self._lock:
            self._buffer.clear()
            self._playback_timestamp = None
            self._last_played_timestamp = None
            self._jitter_samples.clear()
            self._current_buffer_ms = self.target_buffer_ms
