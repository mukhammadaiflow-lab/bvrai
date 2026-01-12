"""Noise suppression implementation."""

import numpy as np
from typing import Optional
from collections import deque
import math


class NoiseSuppressor:
    """
    Simple spectral subtraction noise suppressor.

    Uses spectral subtraction with noise floor estimation
    to reduce background noise while preserving speech.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        frame_size: int = 256,
        noise_estimation_frames: int = 20,
        suppression_factor: float = 1.5,
        floor_db: float = -40.0,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.fft_size = frame_size * 2
        self.suppression_factor = suppression_factor
        self.floor = 10 ** (floor_db / 20)

        # Noise estimation
        self._noise_spectrum: Optional[np.ndarray] = None
        self._noise_frames: deque = deque(maxlen=noise_estimation_frames)
        self._frames_processed = 0

        # Overlap-add state
        self._overlap_buffer = np.zeros(frame_size)

        # Window function
        self._window = np.hanning(self.fft_size)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply noise suppression.

        Args:
            samples: Input audio (int16)

        Returns:
            Noise-suppressed audio (int16)
        """
        # Convert to float
        audio = samples.astype(np.float32) / 32768.0

        # Process in frames
        output = np.zeros_like(audio)
        hop_size = self.frame_size

        for i in range(0, len(audio) - self.fft_size + 1, hop_size):
            frame = audio[i:i + self.fft_size]

            # Apply window
            windowed = frame * self._window

            # FFT
            spectrum = np.fft.rfft(windowed)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Update noise estimate during first frames
            if self._frames_processed < 20:
                self._update_noise_estimate(magnitude)
            else:
                # Check if this frame is likely noise (low energy)
                frame_energy = np.mean(magnitude ** 2)
                if self._noise_spectrum is not None:
                    noise_energy = np.mean(self._noise_spectrum ** 2)
                    if frame_energy < noise_energy * 1.5:
                        self._update_noise_estimate(magnitude)

            # Spectral subtraction
            if self._noise_spectrum is not None:
                # Calculate gain
                noise_estimate = self._noise_spectrum * self.suppression_factor
                gain = np.maximum(
                    (magnitude - noise_estimate) / (magnitude + 1e-10),
                    self.floor,
                )

                # Smooth gain
                gain = self._smooth_gain(gain)

                # Apply gain
                magnitude = magnitude * gain

            # Reconstruct
            spectrum_out = magnitude * np.exp(1j * phase)
            frame_out = np.fft.irfft(spectrum_out)

            # Overlap-add
            output[i:i + self.fft_size] += frame_out * self._window

            self._frames_processed += 1

        # Normalize overlap-add
        output = output / 1.5  # Approximate normalization for Hann window

        # Convert back to int16
        return (output * 32768).astype(np.int16)

    def _update_noise_estimate(self, magnitude: np.ndarray) -> None:
        """Update noise floor estimate."""
        self._noise_frames.append(magnitude.copy())

        if len(self._noise_frames) >= 5:
            # Average of noise frames
            self._noise_spectrum = np.mean(list(self._noise_frames), axis=0)

    def _smooth_gain(self, gain: np.ndarray) -> np.ndarray:
        """Smooth gain curve to reduce musical noise."""
        # Simple moving average smoothing
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size

        # Pad for convolution
        padded = np.pad(gain, kernel_size // 2, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')

        return smoothed[:len(gain)]

    def get_statistics(self) -> dict:
        """Get noise suppressor statistics."""
        noise_floor_db = -100.0
        if self._noise_spectrum is not None:
            rms = np.sqrt(np.mean(self._noise_spectrum ** 2))
            if rms > 0:
                noise_floor_db = 20 * math.log10(rms)

        return {
            "frames_processed": self._frames_processed,
            "noise_floor_db": round(noise_floor_db, 2),
            "noise_frames_collected": len(self._noise_frames),
        }

    def reset(self) -> None:
        """Reset noise suppressor state."""
        self._noise_spectrum = None
        self._noise_frames.clear()
        self._frames_processed = 0
        self._overlap_buffer = np.zeros(self.frame_size)


class WienerFilter:
    """
    Wiener filter-based noise reduction.

    More sophisticated than spectral subtraction,
    uses statistical estimation for optimal filtering.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        frame_size: int = 256,
        alpha: float = 0.98,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.fft_size = frame_size * 2
        self.alpha = alpha  # Smoothing factor

        # State
        self._noise_psd: Optional[np.ndarray] = None
        self._prev_gain: Optional[np.ndarray] = None
        self._window = np.hanning(self.fft_size)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply Wiener filtering."""
        audio = samples.astype(np.float32) / 32768.0
        output = np.zeros_like(audio)

        for i in range(0, len(audio) - self.fft_size + 1, self.frame_size):
            frame = audio[i:i + self.fft_size] * self._window

            spectrum = np.fft.rfft(frame)
            power = np.abs(spectrum) ** 2

            # Initialize noise PSD
            if self._noise_psd is None:
                self._noise_psd = power.copy()

            # Estimate SNR
            snr = power / (self._noise_psd + 1e-10)

            # Wiener gain
            gain = snr / (snr + 1)

            # Smooth gain
            if self._prev_gain is not None:
                gain = self.alpha * self._prev_gain + (1 - self.alpha) * gain
            self._prev_gain = gain.copy()

            # Apply gain
            spectrum_out = spectrum * gain
            frame_out = np.fft.irfft(spectrum_out)

            # Overlap-add
            output[i:i + self.fft_size] += frame_out * self._window

        return (output * 32768 / 1.5).astype(np.int16)

    def update_noise(self, samples: np.ndarray) -> None:
        """Update noise estimate from known noise segment."""
        audio = samples.astype(np.float32) / 32768.0

        if len(audio) >= self.fft_size:
            frame = audio[:self.fft_size] * self._window
            spectrum = np.fft.rfft(frame)
            self._noise_psd = np.abs(spectrum) ** 2

    def reset(self) -> None:
        """Reset filter state."""
        self._noise_psd = None
        self._prev_gain = None
