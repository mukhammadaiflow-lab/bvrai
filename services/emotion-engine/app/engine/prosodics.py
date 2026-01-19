"""
Prosodic Feature Analyzer.

Extracts prosodic features (pitch, energy, rhythm) from audio
for emotion detection.
"""

import logging
from typing import Optional, Tuple
import numpy as np

from ..config import ProsodicsConfig, get_settings
from ..models import ProsodicsFeatures

logger = logging.getLogger(__name__)


class ProsodicsAnalyzer:
    """
    Analyzes prosodic features from audio signals.

    Extracts:
    - Pitch (F0) statistics
    - Energy/intensity features
    - Rhythm and timing
    - Spectral characteristics
    - Voice quality metrics
    """

    def __init__(self, config: Optional[ProsodicsConfig] = None):
        """Initialize analyzer."""
        self.config = config or get_settings().prosodics

        # Derived parameters
        self.frame_samples = int(
            self.config.frame_size_ms * get_settings().sample_rate / 1000
        )
        self.hop_samples = int(
            self.config.hop_size_ms * get_settings().sample_rate / 1000
        )

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
        timestamp_ms: float = 0.0,
    ) -> ProsodicsFeatures:
        """
        Analyze prosodic features from audio.

        Args:
            audio: Audio samples (normalized float32, -1 to 1)
            sample_rate: Sample rate in Hz
            timestamp_ms: Timestamp of the audio segment

        Returns:
            ProsodicsFeatures with all extracted features
        """
        # Ensure audio is the right format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0  # Convert from int16 range

        # Extract features
        pitch_features = self._extract_pitch_features(audio, sample_rate)
        energy_features = self._extract_energy_features(audio)
        rhythm_features = self._extract_rhythm_features(audio, sample_rate)
        spectral_features = self._extract_spectral_features(audio, sample_rate)
        quality_features = self._extract_voice_quality(audio, sample_rate)

        return ProsodicsFeatures(
            timestamp_ms=timestamp_ms,
            # Pitch
            pitch_mean_hz=pitch_features.get("mean", 0.0),
            pitch_std_hz=pitch_features.get("std", 0.0),
            pitch_min_hz=pitch_features.get("min", 0.0),
            pitch_max_hz=pitch_features.get("max", 0.0),
            pitch_range_hz=pitch_features.get("range", 0.0),
            pitch_slope=pitch_features.get("slope", 0.0),
            # Energy
            energy_mean=energy_features.get("mean", 0.0),
            energy_std=energy_features.get("std", 0.0),
            energy_max=energy_features.get("max", 0.0),
            loudness_db=energy_features.get("loudness_db", -60.0),
            # Rhythm
            speaking_rate_sps=rhythm_features.get("rate", 0.0),
            pause_ratio=rhythm_features.get("pause_ratio", 0.0),
            voiced_ratio=rhythm_features.get("voiced_ratio", 0.0),
            # Spectral
            spectral_centroid=spectral_features.get("centroid", 0.0),
            spectral_bandwidth=spectral_features.get("bandwidth", 0.0),
            spectral_rolloff=spectral_features.get("rolloff", 0.0),
            zero_crossing_rate=spectral_features.get("zcr", 0.0),
            # Voice quality
            jitter=quality_features.get("jitter", 0.0),
            shimmer=quality_features.get("shimmer", 0.0),
            hnr_db=quality_features.get("hnr", 0.0),
        )

    def _extract_pitch_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict:
        """Extract pitch (F0) features using autocorrelation."""
        if len(audio) < self.frame_samples * 2:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0, "slope": 0.0}

        pitches = []
        n_frames = (len(audio) - self.frame_samples) // self.hop_samples

        for i in range(max(1, n_frames)):
            start = i * self.hop_samples
            end = start + self.frame_samples
            frame = audio[start:end]

            pitch = self._estimate_pitch_autocorr(frame, sample_rate)
            if pitch > 0:
                pitches.append(pitch)

        if not pitches:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0, "slope": 0.0}

        pitches = np.array(pitches)

        # Calculate slope (rising/falling intonation)
        if len(pitches) > 1:
            slope = np.polyfit(np.arange(len(pitches)), pitches, 1)[0]
        else:
            slope = 0.0

        return {
            "mean": float(np.mean(pitches)),
            "std": float(np.std(pitches)),
            "min": float(np.min(pitches)),
            "max": float(np.max(pitches)),
            "range": float(np.max(pitches) - np.min(pitches)),
            "slope": float(slope),
        }

    def _estimate_pitch_autocorr(
        self,
        frame: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Estimate pitch using autocorrelation."""
        # Autocorrelation
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]

        # Find pitch period in expected range
        min_period = int(sample_rate / self.config.pitch_max_hz)
        max_period = int(sample_rate / self.config.pitch_min_hz)

        if max_period > len(corr) - 1:
            max_period = len(corr) - 1

        if min_period >= max_period:
            return 0.0

        # Find peak in correlation
        corr_segment = corr[min_period:max_period]
        if len(corr_segment) == 0 or np.max(corr_segment) < self.config.pitch_threshold * corr[0]:
            return 0.0  # Unvoiced

        peak_idx = np.argmax(corr_segment) + min_period
        pitch = sample_rate / peak_idx

        return float(pitch)

    def _extract_energy_features(self, audio: np.ndarray) -> dict:
        """Extract energy/intensity features."""
        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))

        # Frame-based energy
        n_frames = len(audio) // self.frame_samples
        if n_frames == 0:
            return {
                "mean": float(rms),
                "std": 0.0,
                "max": float(rms),
                "loudness_db": 20 * np.log10(rms + 1e-10),
            }

        frame_energies = []
        for i in range(n_frames):
            start = i * self.frame_samples
            end = start + self.frame_samples
            frame = audio[start:end]
            frame_energies.append(np.sqrt(np.mean(frame ** 2)))

        frame_energies = np.array(frame_energies)

        # Loudness in dB
        loudness_db = 20 * np.log10(rms + 1e-10)

        return {
            "mean": float(np.mean(frame_energies)),
            "std": float(np.std(frame_energies)),
            "max": float(np.max(frame_energies)),
            "loudness_db": float(loudness_db),
        }

    def _extract_rhythm_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict:
        """Extract rhythm and timing features."""
        # Energy-based voice activity detection
        n_frames = len(audio) // self.hop_samples
        if n_frames == 0:
            return {"rate": 0.0, "pause_ratio": 0.0, "voiced_ratio": 0.0}

        energies = []
        for i in range(n_frames):
            start = i * self.hop_samples
            end = min(start + self.frame_samples, len(audio))
            frame = audio[start:end]
            energies.append(np.sqrt(np.mean(frame ** 2)))

        energies = np.array(energies)

        # Voice activity
        threshold = np.mean(energies) * 0.3
        voiced = energies > threshold
        voiced_ratio = np.mean(voiced)

        # Pause ratio
        pause_ratio = 1.0 - voiced_ratio

        # Estimate speaking rate from energy envelope transitions
        transitions = np.abs(np.diff(voiced.astype(float)))
        syllable_estimate = np.sum(transitions) / 2  # Pairs of transitions
        duration_s = len(audio) / sample_rate
        speaking_rate = syllable_estimate / duration_s if duration_s > 0 else 0.0

        return {
            "rate": float(speaking_rate),
            "pause_ratio": float(pause_ratio),
            "voiced_ratio": float(voiced_ratio),
        }

    def _extract_spectral_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict:
        """Extract spectral features."""
        # FFT
        n_fft = min(2048, len(audio))
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))

        spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
        freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)

        # Normalize spectrum
        spectrum_norm = spectrum / (np.sum(spectrum) + 1e-10)

        # Spectral centroid
        centroid = np.sum(freqs * spectrum_norm)

        # Spectral bandwidth
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum_norm))

        # Spectral rolloff (85%)
        cumsum = np.cumsum(spectrum_norm)
        rolloff_idx = np.searchsorted(cumsum, 0.85)
        rolloff = freqs[rolloff_idx] if rolloff_idx < len(freqs) else freqs[-1]

        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)

        return {
            "centroid": float(centroid),
            "bandwidth": float(bandwidth),
            "rolloff": float(rolloff),
            "zcr": float(zcr),
        }

    def _extract_voice_quality(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> dict:
        """Extract voice quality features (jitter, shimmer, HNR)."""
        # Simplified voice quality estimation
        # In production, use dedicated libraries like parselmouth/praat

        # Jitter: variation in pitch period
        # Estimated from energy envelope variations
        n_frames = len(audio) // self.frame_samples
        if n_frames < 2:
            return {"jitter": 0.0, "shimmer": 0.0, "hnr": 0.0}

        frame_peaks = []
        for i in range(n_frames):
            start = i * self.frame_samples
            end = start + self.frame_samples
            frame = audio[start:end]
            frame_peaks.append(np.max(np.abs(frame)))

        frame_peaks = np.array(frame_peaks)

        # Shimmer: variation in amplitude
        shimmer = np.mean(np.abs(np.diff(frame_peaks))) / (np.mean(frame_peaks) + 1e-10)

        # Approximate jitter from shimmer correlation
        jitter = shimmer * 0.3  # Simplified approximation

        # HNR: Harmonics-to-noise ratio (simplified)
        # Using spectral periodicity as proxy
        spectrum = np.abs(np.fft.rfft(audio))
        harmonic_energy = np.sum(spectrum[:len(spectrum)//4] ** 2)
        noise_energy = np.sum(spectrum[len(spectrum)//4:] ** 2)
        hnr = 10 * np.log10((harmonic_energy + 1e-10) / (noise_energy + 1e-10))

        return {
            "jitter": float(np.clip(jitter, 0, 0.1)),
            "shimmer": float(np.clip(shimmer, 0, 0.3)),
            "hnr": float(np.clip(hnr, -10, 30)),
        }
