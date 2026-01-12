"""Adaptive buffer with dynamic sizing."""

import threading
from typing import Optional, List
from collections import deque
import time
import numpy as np


class AdaptiveBuffer:
    """
    Adaptive audio buffer that adjusts size based on conditions.

    Features:
    - Dynamic buffer sizing based on network conditions
    - Smooth transitions to avoid audio artifacts
    - Underrun prediction and prevention
    - Latency vs quality tradeoff management
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        initial_ms: int = 100,
        min_ms: int = 20,
        max_ms: int = 500,
        target_latency_ms: int = 150,
    ):
        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_ms / 1000)
        self.max_samples = int(sample_rate * max_ms / 1000)
        self.target_latency_samples = int(sample_rate * target_latency_ms / 1000)

        # Current buffer size target
        self._target_samples = int(sample_rate * initial_ms / 1000)

        # Data storage
        self._buffer: deque = deque()
        self._total_samples = 0
        self._lock = threading.Lock()

        # Adaptation metrics
        self._arrival_times: deque = deque(maxlen=50)
        self._read_times: deque = deque(maxlen=50)
        self._underrun_count = 0
        self._overrun_count = 0
        self._last_adjustment = time.time()

        # Statistics
        self._total_written = 0
        self._total_read = 0

    def write(self, samples: bytes) -> int:
        """
        Write audio data to buffer.

        Args:
            samples: PCM audio bytes

        Returns:
            Number of bytes written
        """
        with self._lock:
            now = time.time()
            self._arrival_times.append(now)

            sample_count = len(samples) // 2  # 16-bit samples
            self._total_samples += sample_count

            # Check for overrun
            if self._total_samples > self.max_samples:
                # Drop oldest data
                excess = self._total_samples - self.max_samples
                self._drop_samples(excess)
                self._overrun_count += 1

            self._buffer.append(samples)
            self._total_written += len(samples)

            # Trigger adaptation check
            self._maybe_adapt(now)

            return len(samples)

    def read(self, byte_count: int) -> Optional[bytes]:
        """
        Read audio data from buffer.

        Args:
            byte_count: Number of bytes to read

        Returns:
            Audio bytes or None if insufficient data
        """
        with self._lock:
            now = time.time()
            self._read_times.append(now)

            sample_count = byte_count // 2

            # Check for underrun
            if self._total_samples < sample_count:
                self._underrun_count += 1
                return None

            # Collect data from buffer
            result = bytearray()
            remaining = byte_count

            while remaining > 0 and self._buffer:
                chunk = self._buffer[0]

                if len(chunk) <= remaining:
                    result.extend(chunk)
                    remaining -= len(chunk)
                    self._buffer.popleft()
                else:
                    result.extend(chunk[:remaining])
                    self._buffer[0] = chunk[remaining:]
                    remaining = 0

            self._total_samples -= len(result) // 2
            self._total_read += len(result)

            return bytes(result)

    def read_available(self) -> bytes:
        """Read all available data."""
        with self._lock:
            if not self._buffer:
                return b""

            result = b"".join(self._buffer)
            self._buffer.clear()
            self._total_samples = 0
            self._total_read += len(result)

            return result

    def _drop_samples(self, count: int) -> None:
        """Drop oldest samples."""
        bytes_to_drop = count * 2
        dropped = 0

        while dropped < bytes_to_drop and self._buffer:
            chunk = self._buffer[0]

            if len(chunk) <= bytes_to_drop - dropped:
                dropped += len(chunk)
                self._buffer.popleft()
            else:
                remaining = bytes_to_drop - dropped
                self._buffer[0] = chunk[remaining:]
                dropped = bytes_to_drop

        self._total_samples = sum(len(c) // 2 for c in self._buffer)

    def _maybe_adapt(self, now: float) -> None:
        """Check and perform buffer adaptation."""
        # Adapt every 500ms
        if now - self._last_adjustment < 0.5:
            return

        self._last_adjustment = now

        # Calculate metrics
        if len(self._arrival_times) < 10:
            return

        # Estimate jitter from arrival times
        intervals = []
        times = list(self._arrival_times)
        for i in range(1, len(times)):
            intervals.append(times[i] - times[i-1])

        if not intervals:
            return

        avg_interval = sum(intervals) / len(intervals)
        jitter = sum(abs(i - avg_interval) for i in intervals) / len(intervals)

        # Adjust target based on conditions
        jitter_ms = jitter * 1000

        if self._underrun_count > 0:
            # Increase buffer on underruns
            increase = min(self._target_samples * 0.2, self.max_samples - self._target_samples)
            self._target_samples = int(self._target_samples + increase)
            self._underrun_count = 0

        elif jitter_ms > 50:
            # High jitter - increase buffer
            self._target_samples = min(
                self._target_samples + int(self.sample_rate * 0.02),
                self.max_samples,
            )

        elif jitter_ms < 10 and self._total_samples > self._target_samples * 1.5:
            # Low jitter and excess buffer - decrease
            self._target_samples = max(
                self._target_samples - int(self.sample_rate * 0.01),
                self.min_samples,
            )

    @property
    def available(self) -> int:
        """Get number of samples available."""
        with self._lock:
            return self._total_samples

    @property
    def available_bytes(self) -> int:
        """Get number of bytes available."""
        return self.available * 2

    @property
    def target_size(self) -> int:
        """Get current target buffer size in samples."""
        return self._target_samples

    @property
    def latency_ms(self) -> float:
        """Get current buffer latency in milliseconds."""
        return self.available * 1000 / self.sample_rate

    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                "available_samples": self._total_samples,
                "available_ms": round(self._total_samples * 1000 / self.sample_rate, 2),
                "target_samples": self._target_samples,
                "target_ms": round(self._target_samples * 1000 / self.sample_rate, 2),
                "total_written_bytes": self._total_written,
                "total_read_bytes": self._total_read,
                "underruns": self._underrun_count,
                "overruns": self._overrun_count,
                "chunks": len(self._buffer),
            }

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._total_samples = 0
