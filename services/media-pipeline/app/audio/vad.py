"""Voice Activity Detection (VAD) implementation."""

import numpy as np
from typing import Optional, List
from collections import deque
from enum import Enum


class VADMode(str, Enum):
    """VAD aggressiveness mode."""
    VERY_AGGRESSIVE = "very_aggressive"  # Fewer false positives
    AGGRESSIVE = "aggressive"
    NORMAL = "normal"
    QUALITY = "quality"  # Fewer false negatives


class VoiceActivityDetector:
    """
    Voice Activity Detector using energy and zero-crossing rate.

    This is a simple but effective VAD implementation that combines:
    - Short-term energy
    - Zero-crossing rate
    - Spectral features (optional)
    - Adaptive thresholding
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        frame_duration_ms: int = 20,
        mode: VADMode = VADMode.NORMAL,
        energy_threshold: float = 0.001,
        zcr_threshold: float = 0.1,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.mode = mode

        # Thresholds based on mode
        self._mode_config = {
            VADMode.VERY_AGGRESSIVE: {"energy_mult": 2.0, "hangover_frames": 3},
            VADMode.AGGRESSIVE: {"energy_mult": 1.5, "hangover_frames": 5},
            VADMode.NORMAL: {"energy_mult": 1.0, "hangover_frames": 10},
            VADMode.QUALITY: {"energy_mult": 0.7, "hangover_frames": 15},
        }

        config = self._mode_config[mode]
        self.energy_threshold = energy_threshold * config["energy_mult"]
        self.zcr_threshold = zcr_threshold
        self.hangover_frames = config["hangover_frames"]

        # Adaptive threshold state
        self._noise_floor = 0.0
        self._noise_floor_samples: deque = deque(maxlen=100)
        self._speech_samples: deque = deque(maxlen=50)

        # State tracking
        self._is_speech = False
        self._hangover_counter = 0
        self._speech_frames = 0
        self._silence_frames = 0

        # Statistics
        self._total_frames = 0
        self._speech_frame_count = 0

    def is_speech(self, samples: np.ndarray) -> bool:
        """
        Determine if audio frame contains speech.

        Args:
            samples: Audio samples (int16)

        Returns:
            True if speech detected
        """
        self._total_frames += 1

        # Normalize samples
        normalized = samples.astype(np.float32) / 32768.0

        # Calculate features
        energy = self._calculate_energy(normalized)
        zcr = self._calculate_zcr(normalized)

        # Update noise floor estimate during silence
        if not self._is_speech:
            self._update_noise_floor(energy)

        # Decision logic
        speech_detected = self._make_decision(energy, zcr)

        # Apply hangover to prevent choppy detection
        if speech_detected:
            self._is_speech = True
            self._hangover_counter = self.hangover_frames
            self._speech_frames += 1
            self._speech_frame_count += 1
        elif self._hangover_counter > 0:
            self._hangover_counter -= 1
            # Still considered speech during hangover
        else:
            self._is_speech = False
            self._silence_frames += 1

        return self._is_speech

    def _calculate_energy(self, samples: np.ndarray) -> float:
        """Calculate short-term energy."""
        return np.mean(samples ** 2)

    def _calculate_zcr(self, samples: np.ndarray) -> float:
        """Calculate zero-crossing rate."""
        if len(samples) < 2:
            return 0.0

        signs = np.sign(samples)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        return crossings / len(samples)

    def _update_noise_floor(self, energy: float) -> None:
        """Update adaptive noise floor estimate."""
        self._noise_floor_samples.append(energy)

        if len(self._noise_floor_samples) >= 10:
            # Use 10th percentile as noise floor
            sorted_samples = sorted(self._noise_floor_samples)
            percentile_idx = len(sorted_samples) // 10
            self._noise_floor = sorted_samples[percentile_idx]

    def _make_decision(self, energy: float, zcr: float) -> bool:
        """Make speech/non-speech decision."""
        # Dynamic threshold based on noise floor
        adaptive_threshold = max(
            self.energy_threshold,
            self._noise_floor * 3.0,
        )

        # Energy-based decision
        energy_decision = energy > adaptive_threshold

        # ZCR helps distinguish between speech and noise
        # Speech typically has moderate ZCR, noise can be high or low
        zcr_reasonable = 0.02 < zcr < 0.4

        # Combined decision
        return energy_decision and zcr_reasonable

    def get_speech_probability(self, samples: np.ndarray) -> float:
        """
        Get probability that frame contains speech.

        Args:
            samples: Audio samples

        Returns:
            Probability 0.0 to 1.0
        """
        normalized = samples.astype(np.float32) / 32768.0
        energy = self._calculate_energy(normalized)
        zcr = self._calculate_zcr(normalized)

        # Energy score
        if self._noise_floor > 0:
            energy_ratio = energy / (self._noise_floor + 1e-10)
            energy_score = min(1.0, energy_ratio / 10.0)
        else:
            energy_score = min(1.0, energy / (self.energy_threshold * 10))

        # ZCR score (penalize very low or very high)
        if 0.05 < zcr < 0.3:
            zcr_score = 1.0
        elif 0.02 < zcr < 0.4:
            zcr_score = 0.7
        else:
            zcr_score = 0.3

        # Combined probability
        probability = energy_score * zcr_score

        # Boost if currently in speech state
        if self._is_speech:
            probability = min(1.0, probability * 1.2)

        return probability

    def get_statistics(self) -> dict:
        """Get VAD statistics."""
        total = self._speech_frames + self._silence_frames
        speech_ratio = self._speech_frames / max(1, total)

        return {
            "total_frames": self._total_frames,
            "speech_frames": self._speech_frame_count,
            "speech_ratio": round(speech_ratio, 3),
            "noise_floor": round(self._noise_floor, 6),
            "current_state": "speech" if self._is_speech else "silence",
        }

    def reset(self) -> None:
        """Reset VAD state."""
        self._noise_floor = 0.0
        self._noise_floor_samples.clear()
        self._speech_samples.clear()
        self._is_speech = False
        self._hangover_counter = 0
        self._speech_frames = 0
        self._silence_frames = 0
