"""
Audio Analyzer Service.

Analyzes audio samples for quality, content, and suitability
for voice cloning.
"""

import asyncio
import io
import logging
import struct
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import get_settings, AudioRequirements
from ..models import AudioSampleMetadata

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of audio analysis."""

    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    metadata: AudioSampleMetadata
    recommendations: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Detailed metrics
    speech_segments: List[Tuple[float, float]] = field(default_factory=list)  # (start_s, end_s)
    noise_profile: Optional[Dict[str, float]] = None


class AudioAnalyzer:
    """
    Analyzes audio for voice cloning suitability.

    Features:
    - Audio format detection and validation
    - Speech detection and segmentation
    - Signal-to-noise ratio estimation
    - Quality scoring
    - Recommendations for improvement
    """

    def __init__(self, requirements: Optional[AudioRequirements] = None):
        """Initialize analyzer with requirements."""
        self.requirements = requirements or get_settings().audio

    async def analyze(
        self,
        audio_data: bytes,
        filename: str,
        mime_type: str,
    ) -> AnalysisResult:
        """
        Analyze audio data for voice cloning suitability.

        Args:
            audio_data: Raw audio bytes
            filename: Original filename
            mime_type: MIME type of the audio

        Returns:
            AnalysisResult with quality assessment
        """
        validation_errors = []
        warnings = []
        recommendations = []

        # Check file size
        file_size = len(audio_data)
        max_size = self.requirements.max_file_size_mb * 1024 * 1024

        if file_size > max_size:
            validation_errors.append(
                f"File too large: {file_size / 1024 / 1024:.1f}MB > {self.requirements.max_file_size_mb}MB"
            )

        # Detect format
        audio_format = self._detect_format(filename, mime_type, audio_data)
        if audio_format not in self.requirements.supported_formats:
            validation_errors.append(
                f"Unsupported format: {audio_format}. Supported: {', '.join(self.requirements.supported_formats)}"
            )

        # Parse audio properties
        try:
            properties = await self._parse_audio_properties(audio_data, audio_format)
        except Exception as e:
            logger.error(f"Failed to parse audio: {e}")
            validation_errors.append(f"Failed to parse audio: {str(e)}")
            properties = self._default_properties()

        # Create metadata
        metadata = AudioSampleMetadata(
            filename=filename,
            file_size_bytes=file_size,
            mime_type=mime_type,
            format=audio_format,
            duration_s=properties.get("duration_s", 0.0),
            sample_rate=properties.get("sample_rate", 0),
            channels=properties.get("channels", 1),
            bit_depth=properties.get("bit_depth"),
            silence_ratio=properties.get("silence_ratio", 0.0),
            snr_db=properties.get("snr_db"),
            clipping_ratio=properties.get("clipping_ratio", 0.0),
            speech_ratio=properties.get("speech_ratio", 0.0),
            detected_language=properties.get("detected_language"),
        )

        # Validate sample rate
        if metadata.sample_rate < self.requirements.min_sample_rate:
            validation_errors.append(
                f"Sample rate too low: {metadata.sample_rate}Hz < {self.requirements.min_sample_rate}Hz"
            )
        elif metadata.sample_rate < self.requirements.preferred_sample_rate:
            recommendations.append(
                f"Consider using {self.requirements.preferred_sample_rate}Hz for better quality"
            )

        # Validate duration
        if metadata.duration_s < self.requirements.min_instant_duration_s:
            validation_errors.append(
                f"Audio too short: {metadata.duration_s:.1f}s < {self.requirements.min_instant_duration_s}s minimum"
            )

        # Check silence ratio
        if metadata.silence_ratio > self.requirements.max_silence_ratio:
            warnings.append(
                f"High silence ratio: {metadata.silence_ratio:.0%}. Consider trimming silence."
            )
            recommendations.append("Trim silence from the beginning and end of the recording")

        # Check SNR
        if metadata.snr_db and metadata.snr_db < self.requirements.min_snr_db:
            warnings.append(
                f"Low signal-to-noise ratio: {metadata.snr_db:.1f}dB < {self.requirements.min_snr_db}dB"
            )
            recommendations.append("Record in a quieter environment or use noise reduction")

        # Check clipping
        if metadata.clipping_ratio > 0.01:  # More than 1% clipping
            warnings.append(f"Audio clipping detected: {metadata.clipping_ratio:.1%} of samples")
            recommendations.append("Reduce input volume to avoid distortion")

        # Calculate quality score
        quality_score = self._calculate_quality_score(metadata, validation_errors, warnings)

        # Generate additional recommendations
        if metadata.duration_s < self.requirements.min_standard_duration_s:
            recommendations.append(
                f"For better voice quality, provide at least {self.requirements.min_standard_duration_s}s of audio"
            )

        if not metadata.speech_ratio or metadata.speech_ratio < 0.5:
            recommendations.append("Ensure the sample contains clear speech throughout")

        is_valid = len(validation_errors) == 0

        return AnalysisResult(
            is_valid=is_valid,
            quality_score=quality_score,
            metadata=metadata,
            recommendations=recommendations,
            validation_errors=validation_errors,
            warnings=warnings,
        )

    def _detect_format(
        self,
        filename: str,
        mime_type: str,
        data: bytes,
    ) -> str:
        """Detect audio format from filename, mime type, or magic bytes."""
        # Check magic bytes first
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return "wav"
        if data[:3] == b"ID3" or data[:2] == b"\xff\xfb":
            return "mp3"
        if data[:4] == b"fLaC":
            return "flac"
        if data[:4] == b"OggS":
            return "ogg"

        # Check filename extension
        if "." in filename:
            ext = filename.rsplit(".", 1)[1].lower()
            if ext in self.requirements.supported_formats:
                return ext

        # Check mime type
        mime_map = {
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/flac": "flac",
            "audio/ogg": "ogg",
            "audio/webm": "webm",
            "audio/m4a": "m4a",
            "audio/mp4": "m4a",
        }
        return mime_map.get(mime_type, "unknown")

    async def _parse_audio_properties(
        self,
        data: bytes,
        audio_format: str,
    ) -> Dict[str, Any]:
        """Parse audio properties from data."""
        if audio_format == "wav":
            return await self._parse_wav(data)
        else:
            # For other formats, use default parsing
            # In production, use librosa or soundfile
            return await self._parse_generic(data, audio_format)

    async def _parse_wav(self, data: bytes) -> Dict[str, Any]:
        """Parse WAV file properties."""
        try:
            with io.BytesIO(data) as f:
                with wave.open(f, "rb") as wav:
                    channels = wav.getnchannels()
                    sample_rate = wav.getframerate()
                    sample_width = wav.getsampwidth()
                    n_frames = wav.getnframes()
                    duration_s = n_frames / sample_rate

                    # Read samples for analysis
                    wav.rewind()
                    frames = wav.readframes(n_frames)

            # Convert to numpy for analysis
            if sample_width == 2:
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 1:
                samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
            else:
                samples = np.zeros(n_frames * channels)

            # Mono conversion for analysis
            if channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)

            # Calculate metrics
            silence_ratio = self._calculate_silence_ratio(samples)
            snr_db = self._estimate_snr(samples)
            clipping_ratio = self._calculate_clipping_ratio(samples)
            speech_ratio = 1.0 - silence_ratio  # Simple approximation

            return {
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "channels": channels,
                "bit_depth": sample_width * 8,
                "silence_ratio": silence_ratio,
                "snr_db": snr_db,
                "clipping_ratio": clipping_ratio,
                "speech_ratio": speech_ratio,
            }

        except Exception as e:
            logger.error(f"WAV parsing error: {e}")
            raise

    async def _parse_generic(
        self,
        data: bytes,
        audio_format: str,
    ) -> Dict[str, Any]:
        """Generic audio parsing (placeholder for librosa integration)."""
        # In production, use librosa or soundfile for format detection
        return {
            "duration_s": len(data) / (16000 * 2),  # Rough estimate
            "sample_rate": 16000,
            "channels": 1,
            "bit_depth": 16,
            "silence_ratio": 0.1,
            "snr_db": 20.0,
            "clipping_ratio": 0.0,
            "speech_ratio": 0.9,
        }

    def _default_properties(self) -> Dict[str, Any]:
        """Return default properties on parse failure."""
        return {
            "duration_s": 0.0,
            "sample_rate": 0,
            "channels": 1,
            "bit_depth": None,
            "silence_ratio": 0.0,
            "snr_db": None,
            "clipping_ratio": 0.0,
            "speech_ratio": 0.0,
        }

    def _calculate_silence_ratio(
        self,
        samples: np.ndarray,
        threshold: float = 0.01,
        frame_size: int = 1024,
    ) -> float:
        """Calculate ratio of silence in audio."""
        if len(samples) == 0:
            return 0.0

        # Calculate RMS for each frame
        n_frames = len(samples) // frame_size
        if n_frames == 0:
            return 0.0

        silence_frames = 0
        for i in range(n_frames):
            frame = samples[i * frame_size:(i + 1) * frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < threshold:
                silence_frames += 1

        return silence_frames / n_frames

    def _estimate_snr(self, samples: np.ndarray) -> Optional[float]:
        """Estimate signal-to-noise ratio."""
        if len(samples) < 1000:
            return None

        # Simple SNR estimation using signal vs. quiet sections
        rms_values = []
        frame_size = 1024

        for i in range(len(samples) // frame_size):
            frame = samples[i * frame_size:(i + 1) * frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            rms_values.append(rms)

        if not rms_values:
            return None

        rms_values = np.array(rms_values)

        # Signal: top 10% of frames
        signal_rms = np.percentile(rms_values, 90)

        # Noise: bottom 10% of frames
        noise_rms = np.percentile(rms_values, 10)

        if noise_rms < 1e-10:
            return 60.0  # Very clean signal

        snr = 20 * np.log10(signal_rms / noise_rms)
        return float(min(60.0, max(0.0, snr)))

    def _calculate_clipping_ratio(
        self,
        samples: np.ndarray,
        threshold: float = 0.99,
    ) -> float:
        """Calculate ratio of clipped samples."""
        if len(samples) == 0:
            return 0.0

        clipped = np.sum(np.abs(samples) > threshold)
        return float(clipped / len(samples))

    def _calculate_quality_score(
        self,
        metadata: AudioSampleMetadata,
        errors: List[str],
        warnings: List[str],
    ) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        if errors:
            return 0.0

        score = 1.0

        # Duration factor
        duration_s = metadata.duration_s
        if duration_s < 15:
            score *= 0.3
        elif duration_s < 30:
            score *= 0.6
        elif duration_s < 60:
            score *= 0.8
        elif duration_s < 120:
            score *= 0.9

        # Sample rate factor
        if metadata.sample_rate < 16000:
            score *= 0.5
        elif metadata.sample_rate < 22050:
            score *= 0.8
        elif metadata.sample_rate < 44100:
            score *= 0.9

        # Silence factor
        if metadata.silence_ratio > 0.5:
            score *= 0.5
        elif metadata.silence_ratio > 0.3:
            score *= 0.7

        # SNR factor
        if metadata.snr_db:
            if metadata.snr_db < 10:
                score *= 0.5
            elif metadata.snr_db < 15:
                score *= 0.7
            elif metadata.snr_db < 20:
                score *= 0.9

        # Clipping factor
        if metadata.clipping_ratio > 0.05:
            score *= 0.5
        elif metadata.clipping_ratio > 0.01:
            score *= 0.8

        # Warnings factor
        score *= (1.0 - 0.05 * len(warnings))

        return max(0.0, min(1.0, score))


# Factory function
def create_analyzer(requirements: Optional[AudioRequirements] = None) -> AudioAnalyzer:
    """Create an audio analyzer with optional custom requirements."""
    return AudioAnalyzer(requirements)
