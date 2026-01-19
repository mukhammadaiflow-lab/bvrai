"""Circular buffer for audio streaming."""

import threading
from typing import Optional
import numpy as np


class CircularBuffer:
    """
    Lock-free circular buffer for audio samples.

    Optimized for single producer, single consumer pattern
    common in real-time audio processing.
    """

    def __init__(self, capacity_samples: int = 16000):
        """
        Initialize circular buffer.

        Args:
            capacity_samples: Buffer capacity in samples
        """
        self.capacity = capacity_samples
        self._buffer = np.zeros(capacity_samples, dtype=np.int16)

        # Atomic indices
        self._write_idx = 0
        self._read_idx = 0

        # Statistics
        self._total_written = 0
        self._total_read = 0
        self._overruns = 0
        self._underruns = 0

        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> int:
        """
        Write samples to buffer.

        Args:
            samples: Audio samples to write (int16)

        Returns:
            Number of samples written
        """
        if len(samples) == 0:
            return 0

        with self._lock:
            available = self._available_write()

            if available == 0:
                self._overruns += 1
                return 0

            # Write as much as possible
            to_write = min(len(samples), available)

            # Handle wrap-around
            end_idx = self._write_idx + to_write

            if end_idx <= self.capacity:
                # No wrap
                self._buffer[self._write_idx:end_idx] = samples[:to_write]
            else:
                # Wrap around
                first_part = self.capacity - self._write_idx
                self._buffer[self._write_idx:] = samples[:first_part]
                self._buffer[:end_idx - self.capacity] = samples[first_part:to_write]

            self._write_idx = end_idx % self.capacity
            self._total_written += to_write

            return to_write

    def read(self, count: int) -> Optional[np.ndarray]:
        """
        Read samples from buffer.

        Args:
            count: Number of samples to read

        Returns:
            Audio samples or None if not enough data
        """
        with self._lock:
            available = self._available_read()

            if available == 0:
                self._underruns += 1
                return None

            # Read as much as available
            to_read = min(count, available)
            output = np.zeros(to_read, dtype=np.int16)

            # Handle wrap-around
            end_idx = self._read_idx + to_read

            if end_idx <= self.capacity:
                # No wrap
                output[:] = self._buffer[self._read_idx:end_idx]
            else:
                # Wrap around
                first_part = self.capacity - self._read_idx
                output[:first_part] = self._buffer[self._read_idx:]
                output[first_part:] = self._buffer[:end_idx - self.capacity]

            self._read_idx = end_idx % self.capacity
            self._total_read += to_read

            return output

    def peek(self, count: int) -> Optional[np.ndarray]:
        """
        Read samples without consuming them.

        Args:
            count: Number of samples to peek

        Returns:
            Audio samples or None if not enough data
        """
        with self._lock:
            available = self._available_read()

            if available == 0:
                return None

            to_read = min(count, available)
            output = np.zeros(to_read, dtype=np.int16)

            end_idx = self._read_idx + to_read

            if end_idx <= self.capacity:
                output[:] = self._buffer[self._read_idx:end_idx]
            else:
                first_part = self.capacity - self._read_idx
                output[:first_part] = self._buffer[self._read_idx:]
                output[first_part:] = self._buffer[:end_idx - self.capacity]

            return output

    def skip(self, count: int) -> int:
        """
        Skip samples without reading them.

        Args:
            count: Number of samples to skip

        Returns:
            Number of samples skipped
        """
        with self._lock:
            available = self._available_read()
            to_skip = min(count, available)
            self._read_idx = (self._read_idx + to_skip) % self.capacity
            return to_skip

    def _available_write(self) -> int:
        """Get number of samples that can be written."""
        if self._write_idx >= self._read_idx:
            return self.capacity - (self._write_idx - self._read_idx) - 1
        else:
            return self._read_idx - self._write_idx - 1

    def _available_read(self) -> int:
        """Get number of samples available to read."""
        if self._write_idx >= self._read_idx:
            return self._write_idx - self._read_idx
        else:
            return self.capacity - self._read_idx + self._write_idx

    @property
    def available(self) -> int:
        """Get number of samples available to read."""
        with self._lock:
            return self._available_read()

    @property
    def free_space(self) -> int:
        """Get number of samples that can be written."""
        with self._lock:
            return self._available_write()

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._write_idx = 0
            self._read_idx = 0

    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                "capacity": self.capacity,
                "available": self._available_read(),
                "free_space": self._available_write(),
                "total_written": self._total_written,
                "total_read": self._total_read,
                "overruns": self._overruns,
                "underruns": self._underruns,
                "fill_ratio": round(
                    self._available_read() / self.capacity, 3
                ),
            }
