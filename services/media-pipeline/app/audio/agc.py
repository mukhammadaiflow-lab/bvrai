"""Automatic Gain Control (AGC) implementation."""

import numpy as np
from typing import Optional
from collections import deque
import math


class AutomaticGainControl:
    """
    Automatic Gain Control for voice audio.

    Features:
    - Target level normalization
    - Smooth gain transitions
    - Peak limiting
    - Noise gate
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        target_level_dbfs: float = -20.0,
        max_gain_db: float = 30.0,
        min_gain_db: float = -10.0,
        attack_time_ms: float = 5.0,
        release_time_ms: float = 50.0,
        noise_gate_db: float = -50.0,
    ):
        self.sample_rate = sample_rate
        self.target_level = 10 ** (target_level_dbfs / 20) * 32768
        self.max_gain = 10 ** (max_gain_db / 20)
        self.min_gain = 10 ** (min_gain_db / 20)
        self.noise_gate = 10 ** (noise_gate_db / 20) * 32768

        # Time constants
        self.attack_coeff = self._time_to_coeff(attack_time_ms)
        self.release_coeff = self._time_to_coeff(release_time_ms)

        # State
        self._current_gain = 1.0
        self._envelope = 0.0
        self._level_history: deque = deque(maxlen=50)

    def _time_to_coeff(self, time_ms: float) -> float:
        """Convert time constant to smoothing coefficient."""
        if time_ms <= 0:
            return 0.0
        samples = self.sample_rate * time_ms / 1000
        return math.exp(-1.0 / samples)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply AGC to audio samples.

        Args:
            samples: Input audio (int16)

        Returns:
            Gain-adjusted audio (int16)
        """
        # Work with float
        audio = samples.astype(np.float32)

        # Calculate envelope (RMS)
        rms = np.sqrt(np.mean(audio ** 2))
        self._level_history.append(rms)

        # Update envelope with smoothing
        if rms > self._envelope:
            # Attack (fast)
            self._envelope = self.attack_coeff * self._envelope + (1 - self.attack_coeff) * rms
        else:
            # Release (slow)
            self._envelope = self.release_coeff * self._envelope + (1 - self.release_coeff) * rms

        # Noise gate check
        if self._envelope < self.noise_gate:
            # Below noise gate, apply minimal processing
            return samples

        # Calculate desired gain
        if self._envelope > 0:
            desired_gain = self.target_level / self._envelope
        else:
            desired_gain = 1.0

        # Limit gain range
        desired_gain = np.clip(desired_gain, self.min_gain, self.max_gain)

        # Smooth gain transition
        gain_diff = desired_gain - self._current_gain
        if abs(gain_diff) > 0.01:
            # Gradual adjustment
            self._current_gain += gain_diff * 0.1
        else:
            self._current_gain = desired_gain

        # Apply gain
        output = audio * self._current_gain

        # Soft limiter to prevent clipping
        output = self._soft_limit(output)

        return output.astype(np.int16)

    def _soft_limit(self, samples: np.ndarray) -> np.ndarray:
        """Apply soft limiting to prevent hard clipping."""
        threshold = 28000  # Below max int16
        ratio = 4.0  # Compression ratio above threshold

        output = samples.copy()

        # Positive values
        mask_pos = samples > threshold
        if np.any(mask_pos):
            excess = samples[mask_pos] - threshold
            output[mask_pos] = threshold + excess / ratio

        # Negative values
        mask_neg = samples < -threshold
        if np.any(mask_neg):
            excess = -samples[mask_neg] - threshold
            output[mask_neg] = -threshold - excess / ratio

        # Final hard clip just in case
        return np.clip(output, -32767, 32767)

    def get_current_gain_db(self) -> float:
        """Get current gain in dB."""
        if self._current_gain > 0:
            return 20 * math.log10(self._current_gain)
        return -100.0

    def get_statistics(self) -> dict:
        """Get AGC statistics."""
        avg_level = np.mean(list(self._level_history)) if self._level_history else 0

        return {
            "current_gain_db": round(self.get_current_gain_db(), 2),
            "envelope": round(self._envelope, 2),
            "average_level": round(avg_level, 2),
        }

    def reset(self) -> None:
        """Reset AGC state."""
        self._current_gain = 1.0
        self._envelope = 0.0
        self._level_history.clear()


class PeakLimiter:
    """
    Peak limiter to prevent clipping.

    Uses lookahead for transparent limiting.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        threshold_db: float = -1.0,
        attack_ms: float = 0.1,
        release_ms: float = 50.0,
        lookahead_ms: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.threshold = 10 ** (threshold_db / 20) * 32768

        # Lookahead buffer
        lookahead_samples = int(sample_rate * lookahead_ms / 1000)
        self._buffer: deque = deque(maxlen=lookahead_samples)

        # Gain reduction state
        self._gain_reduction = 1.0
        self._attack_coeff = math.exp(-1.0 / (sample_rate * attack_ms / 1000))
        self._release_coeff = math.exp(-1.0 / (sample_rate * release_ms / 1000))

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply peak limiting."""
        output = np.zeros_like(samples, dtype=np.float32)

        for i, sample in enumerate(samples.astype(np.float32)):
            # Add to lookahead buffer
            self._buffer.append(sample)

            # Check peak in buffer
            if len(self._buffer) > 0:
                peak = max(abs(s) for s in self._buffer)

                # Calculate required gain reduction
                if peak > self.threshold:
                    target_gr = self.threshold / peak
                else:
                    target_gr = 1.0

                # Smooth gain reduction
                if target_gr < self._gain_reduction:
                    # Attack
                    self._gain_reduction = (
                        self._attack_coeff * self._gain_reduction +
                        (1 - self._attack_coeff) * target_gr
                    )
                else:
                    # Release
                    self._gain_reduction = (
                        self._release_coeff * self._gain_reduction +
                        (1 - self._release_coeff) * target_gr
                    )

            # Output oldest sample with gain reduction
            if len(self._buffer) == self._buffer.maxlen:
                output[i] = self._buffer[0] * self._gain_reduction

        return output.astype(np.int16)

    def reset(self) -> None:
        """Reset limiter state."""
        self._buffer.clear()
        self._gain_reduction = 1.0
