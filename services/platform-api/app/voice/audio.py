"""
Audio Processing

Audio processing utilities:
- Audio codecs
- Voice activity detection
- Noise reduction
- Echo cancellation
- Audio mixing
- Resampling
"""

from typing import Optional, Dict, Any, List, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import struct
import array
import math

logger = logging.getLogger(__name__)


class CodecType(str, Enum):
    """Audio codec types."""
    PCM = "pcm"
    PCMU = "pcmu"  # G.711 mu-law
    PCMA = "pcma"  # G.711 a-law
    OPUS = "opus"
    G722 = "g722"
    G729 = "g729"
    SPEEX = "speex"
    AAC = "aac"
    MP3 = "mp3"


@dataclass
class AudioFormat:
    """Audio format specification."""
    codec: CodecType = CodecType.PCM
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16

    @property
    def bytes_per_sample(self) -> int:
        """Get bytes per sample."""
        return self.bits_per_sample // 8

    @property
    def frame_size(self) -> int:
        """Get frame size for 20ms."""
        return int(self.sample_rate * 0.02) * self.channels * self.bytes_per_sample

    def duration_to_samples(self, duration_ms: float) -> int:
        """Convert duration to samples."""
        return int(self.sample_rate * duration_ms / 1000)

    def samples_to_duration(self, samples: int) -> float:
        """Convert samples to duration in ms."""
        return samples * 1000 / self.sample_rate


@dataclass
class AudioBuffer:
    """Audio buffer for storing samples."""
    format: AudioFormat = field(default_factory=AudioFormat)
    data: bytes = b""
    timestamp_ms: float = 0.0

    @property
    def duration_ms(self) -> float:
        """Get buffer duration in milliseconds."""
        samples = len(self.data) // (self.format.channels * self.format.bytes_per_sample)
        return self.format.samples_to_duration(samples)

    @property
    def sample_count(self) -> int:
        """Get number of samples."""
        return len(self.data) // (self.format.channels * self.format.bytes_per_sample)

    def get_samples(self) -> List[int]:
        """Get samples as list of integers."""
        if self.format.bits_per_sample == 16:
            return list(struct.unpack(f"<{self.sample_count}h", self.data))
        elif self.format.bits_per_sample == 8:
            return list(self.data)
        else:
            raise ValueError(f"Unsupported bits per sample: {self.format.bits_per_sample}")

    def from_samples(self, samples: List[int]) -> "AudioBuffer":
        """Create buffer from samples."""
        if self.format.bits_per_sample == 16:
            data = struct.pack(f"<{len(samples)}h", *samples)
        elif self.format.bits_per_sample == 8:
            data = bytes(samples)
        else:
            raise ValueError(f"Unsupported bits per sample: {self.format.bits_per_sample}")

        return AudioBuffer(
            format=self.format,
            data=data,
            timestamp_ms=self.timestamp_ms,
        )

    def append(self, other: "AudioBuffer") -> "AudioBuffer":
        """Append another buffer."""
        return AudioBuffer(
            format=self.format,
            data=self.data + other.data,
            timestamp_ms=self.timestamp_ms,
        )

    def slice(self, start_ms: float, end_ms: float) -> "AudioBuffer":
        """Slice buffer by time."""
        start_samples = self.format.duration_to_samples(start_ms)
        end_samples = self.format.duration_to_samples(end_ms)

        bytes_per_sample = self.format.channels * self.format.bytes_per_sample
        start_byte = start_samples * bytes_per_sample
        end_byte = end_samples * bytes_per_sample

        return AudioBuffer(
            format=self.format,
            data=self.data[start_byte:end_byte],
            timestamp_ms=self.timestamp_ms + start_ms,
        )


class AudioCodec(ABC):
    """Abstract audio codec."""

    @property
    @abstractmethod
    def codec_type(self) -> CodecType:
        """Get codec type."""
        pass

    @abstractmethod
    def encode(self, pcm_data: bytes) -> bytes:
        """Encode PCM to codec format."""
        pass

    @abstractmethod
    def decode(self, encoded_data: bytes) -> bytes:
        """Decode from codec format to PCM."""
        pass


class PCMCodec(AudioCodec):
    """PCM (passthrough) codec."""

    @property
    def codec_type(self) -> CodecType:
        return CodecType.PCM

    def encode(self, pcm_data: bytes) -> bytes:
        """Passthrough encoding."""
        return pcm_data

    def decode(self, encoded_data: bytes) -> bytes:
        """Passthrough decoding."""
        return encoded_data


class MuLawCodec(AudioCodec):
    """G.711 mu-law codec."""

    MULAW_MAX = 0x1FFF
    MULAW_BIAS = 33

    # Lookup tables
    _encode_table: List[int] = []
    _decode_table: List[int] = []

    def __init__(self):
        if not self._encode_table:
            self._build_tables()

    @classmethod
    def _build_tables(cls):
        """Build encoding/decoding lookup tables."""
        # Build encode table
        cls._encode_table = [0] * 65536
        for i in range(65536):
            sample = i - 32768
            cls._encode_table[i] = cls._linear_to_mulaw(sample)

        # Build decode table
        cls._decode_table = [0] * 256
        for i in range(256):
            cls._decode_table[i] = cls._mulaw_to_linear(i)

    @classmethod
    def _linear_to_mulaw(cls, sample: int) -> int:
        """Convert linear sample to mu-law."""
        sign = 0
        if sample < 0:
            sign = 0x80
            sample = -sample

        sample = min(sample, cls.MULAW_MAX)
        sample += cls.MULAW_BIAS

        # Find segment
        segment = 0
        for i in range(8):
            if sample >= (1 << (i + 6)):
                segment = i

        # Calculate quantization
        if segment >= 8:
            return sign | 0x7F

        mantissa = (sample >> (segment + 3)) & 0x0F
        return sign | ((segment << 4) | mantissa) ^ 0xFF

    @classmethod
    def _mulaw_to_linear(cls, mulaw: int) -> int:
        """Convert mu-law to linear sample."""
        mulaw = ~mulaw
        sign = mulaw & 0x80
        segment = (mulaw >> 4) & 0x07
        mantissa = mulaw & 0x0F

        sample = ((mantissa << 3) + cls.MULAW_BIAS) << segment
        sample -= cls.MULAW_BIAS

        if sign:
            return -sample
        return sample

    @property
    def codec_type(self) -> CodecType:
        return CodecType.PCMU

    def encode(self, pcm_data: bytes) -> bytes:
        """Encode PCM to mu-law."""
        samples = struct.unpack(f"<{len(pcm_data)//2}h", pcm_data)
        encoded = bytes(self._encode_table[s + 32768] for s in samples)
        return encoded

    def decode(self, encoded_data: bytes) -> bytes:
        """Decode mu-law to PCM."""
        samples = [self._decode_table[b] for b in encoded_data]
        return struct.pack(f"<{len(samples)}h", *samples)


class ALawCodec(AudioCodec):
    """G.711 A-law codec."""

    _encode_table: List[int] = []
    _decode_table: List[int] = []

    def __init__(self):
        if not self._encode_table:
            self._build_tables()

    @classmethod
    def _build_tables(cls):
        """Build encoding/decoding lookup tables."""
        # Build encode table
        cls._encode_table = [0] * 65536
        for i in range(65536):
            sample = i - 32768
            cls._encode_table[i] = cls._linear_to_alaw(sample)

        # Build decode table
        cls._decode_table = [0] * 256
        for i in range(256):
            cls._decode_table[i] = cls._alaw_to_linear(i)

    @classmethod
    def _linear_to_alaw(cls, sample: int) -> int:
        """Convert linear sample to A-law."""
        sign = 0
        if sample < 0:
            sign = 0x80
            sample = -sample

        if sample > 32767:
            sample = 32767

        if sample >= 256:
            segment = 0
            for i in range(8):
                if sample >= (1 << (i + 8)):
                    segment = i + 1

            mantissa = (sample >> (segment + 3)) & 0x0F
            return sign | ((segment << 4) | mantissa) ^ 0x55
        else:
            return sign | (sample >> 4) ^ 0x55

    @classmethod
    def _alaw_to_linear(cls, alaw: int) -> int:
        """Convert A-law to linear sample."""
        alaw ^= 0x55
        sign = alaw & 0x80
        segment = (alaw >> 4) & 0x07
        mantissa = alaw & 0x0F

        if segment == 0:
            sample = (mantissa << 4) + 8
        else:
            sample = ((mantissa << 3) + 0x84) << (segment - 1)

        if sign:
            return -sample
        return sample

    @property
    def codec_type(self) -> CodecType:
        return CodecType.PCMA

    def encode(self, pcm_data: bytes) -> bytes:
        """Encode PCM to A-law."""
        samples = struct.unpack(f"<{len(pcm_data)//2}h", pcm_data)
        encoded = bytes(self._encode_table[s + 32768] for s in samples)
        return encoded

    def decode(self, encoded_data: bytes) -> bytes:
        """Decode A-law to PCM."""
        samples = [self._decode_table[b] for b in encoded_data]
        return struct.pack(f"<{len(samples)}h", *samples)


class CodecRegistry:
    """Registry of audio codecs."""

    _codecs: Dict[CodecType, AudioCodec] = {}

    @classmethod
    def register(cls, codec: AudioCodec) -> None:
        """Register codec."""
        cls._codecs[codec.codec_type] = codec

    @classmethod
    def get(cls, codec_type: CodecType) -> Optional[AudioCodec]:
        """Get codec by type."""
        return cls._codecs.get(codec_type)

    @classmethod
    def transcode(
        cls,
        data: bytes,
        from_codec: CodecType,
        to_codec: CodecType,
    ) -> bytes:
        """Transcode between codecs."""
        if from_codec == to_codec:
            return data

        from_codec_obj = cls.get(from_codec)
        to_codec_obj = cls.get(to_codec)

        if not from_codec_obj or not to_codec_obj:
            raise ValueError(f"Codec not found")

        # Decode to PCM
        pcm = from_codec_obj.decode(data)

        # Encode to target
        return to_codec_obj.encode(pcm)


# Register default codecs
CodecRegistry.register(PCMCodec())
CodecRegistry.register(MuLawCodec())
CodecRegistry.register(ALawCodec())


class VADDetector:
    """
    Voice Activity Detection.

    Detects presence of speech in audio.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        energy_threshold: float = 0.01,
        zero_crossing_threshold: float = 0.1,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold = energy_threshold
        self.zero_crossing_threshold = zero_crossing_threshold

        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        # Adaptive threshold
        self._noise_floor = energy_threshold
        self._adaptation_rate = 0.01

        # State
        self._speech_frames = 0
        self._silence_frames = 0
        self._is_speaking = False

    def process(self, audio: bytes) -> bool:
        """
        Process audio frame and detect voice activity.

        Returns True if speech is detected.
        """
        # Convert to samples
        samples = struct.unpack(f"<{len(audio)//2}h", audio)

        # Calculate energy
        energy = self._calculate_energy(samples)

        # Calculate zero-crossing rate
        zcr = self._calculate_zcr(samples)

        # Adapt noise floor
        if not self._is_speaking:
            self._noise_floor = (
                self._noise_floor * (1 - self._adaptation_rate) +
                energy * self._adaptation_rate
            )

        # Dynamic threshold
        threshold = max(self.energy_threshold, self._noise_floor * 2)

        # Detect speech
        is_speech = energy > threshold and zcr < self.zero_crossing_threshold

        # Hangover logic
        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            if self._speech_frames >= 3:
                self._is_speaking = True
        else:
            self._silence_frames += 1
            if self._silence_frames >= 10:
                self._is_speaking = False
                self._speech_frames = 0

        return self._is_speaking

    def _calculate_energy(self, samples: Tuple[int, ...]) -> float:
        """Calculate frame energy."""
        if not samples:
            return 0.0

        # Normalize and calculate RMS
        max_val = 32768.0
        normalized = [s / max_val for s in samples]
        sum_sq = sum(s * s for s in normalized)
        return math.sqrt(sum_sq / len(samples))

    def _calculate_zcr(self, samples: Tuple[int, ...]) -> float:
        """Calculate zero-crossing rate."""
        if len(samples) < 2:
            return 0.0

        crossings = sum(
            1 for i in range(1, len(samples))
            if (samples[i-1] >= 0) != (samples[i] >= 0)
        )

        return crossings / len(samples)

    def reset(self) -> None:
        """Reset detector state."""
        self._speech_frames = 0
        self._silence_frames = 0
        self._is_speaking = False


class NoiseReducer:
    """
    Noise reduction processor.

    Reduces background noise in audio.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 320,
        noise_gate_threshold: float = 0.02,
        reduction_amount: float = 0.7,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.noise_gate_threshold = noise_gate_threshold
        self.reduction_amount = reduction_amount

        # Noise profile
        self._noise_profile: List[float] = []
        self._profile_frames = 0
        self._max_profile_frames = 50

        # Smoothing
        self._prev_gain = 1.0
        self._smoothing = 0.1

    def process(self, audio: bytes) -> bytes:
        """Process audio and reduce noise."""
        samples = list(struct.unpack(f"<{len(audio)//2}h", audio))

        # Calculate frame energy
        energy = self._calculate_energy(samples)

        # Update noise profile during silence
        if energy < self.noise_gate_threshold and self._profile_frames < self._max_profile_frames:
            self._update_noise_profile(samples)
            self._profile_frames += 1

        # Apply noise reduction
        if energy < self.noise_gate_threshold:
            # Apply noise gate
            gain = 1.0 - self.reduction_amount
        else:
            gain = 1.0

        # Smooth gain changes
        gain = self._prev_gain * (1 - self._smoothing) + gain * self._smoothing
        self._prev_gain = gain

        # Apply gain
        processed = [int(s * gain) for s in samples]

        # Clip to valid range
        processed = [max(-32768, min(32767, s)) for s in processed]

        return struct.pack(f"<{len(processed)}h", *processed)

    def _calculate_energy(self, samples: List[int]) -> float:
        """Calculate frame energy."""
        if not samples:
            return 0.0

        max_val = 32768.0
        normalized = [s / max_val for s in samples]
        sum_sq = sum(s * s for s in normalized)
        return math.sqrt(sum_sq / len(samples))

    def _update_noise_profile(self, samples: List[int]) -> None:
        """Update noise profile from samples."""
        if not self._noise_profile:
            self._noise_profile = [abs(s) / 32768.0 for s in samples]
        else:
            alpha = 0.1
            self._noise_profile = [
                p * (1 - alpha) + abs(s) / 32768.0 * alpha
                for p, s in zip(self._noise_profile, samples)
            ]

    def reset(self) -> None:
        """Reset noise profile."""
        self._noise_profile = []
        self._profile_frames = 0


class AudioMixer:
    """
    Audio mixer.

    Mixes multiple audio streams together.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

        # Input streams
        self._streams: Dict[str, List[bytes]] = {}
        self._volumes: Dict[str, float] = {}
        self._muted: Dict[str, bool] = {}

        # Master settings
        self._master_volume = 1.0

    def add_stream(self, stream_id: str, volume: float = 1.0) -> None:
        """Add audio stream."""
        self._streams[stream_id] = []
        self._volumes[stream_id] = volume
        self._muted[stream_id] = False

    def remove_stream(self, stream_id: str) -> None:
        """Remove audio stream."""
        self._streams.pop(stream_id, None)
        self._volumes.pop(stream_id, None)
        self._muted.pop(stream_id, None)

    def push_audio(self, stream_id: str, audio: bytes) -> None:
        """Push audio to stream."""
        if stream_id in self._streams:
            self._streams[stream_id].append(audio)

    def set_volume(self, stream_id: str, volume: float) -> None:
        """Set stream volume (0.0 to 1.0)."""
        if stream_id in self._volumes:
            self._volumes[stream_id] = max(0.0, min(1.0, volume))

    def set_muted(self, stream_id: str, muted: bool) -> None:
        """Mute/unmute stream."""
        if stream_id in self._muted:
            self._muted[stream_id] = muted

    def set_master_volume(self, volume: float) -> None:
        """Set master volume."""
        self._master_volume = max(0.0, min(1.0, volume))

    def mix(self) -> Optional[bytes]:
        """Mix all streams and return combined audio."""
        # Get audio from each stream
        stream_audio = {}
        min_length = float('inf')

        for stream_id, buffer in self._streams.items():
            if buffer and not self._muted.get(stream_id, False):
                audio = buffer.pop(0)
                stream_audio[stream_id] = audio
                min_length = min(min_length, len(audio))

        if not stream_audio:
            return None

        # Convert to samples
        mixed_samples = [0.0] * (int(min_length) // 2)

        for stream_id, audio in stream_audio.items():
            volume = self._volumes.get(stream_id, 1.0)
            samples = struct.unpack(f"<{len(audio)//2}h", audio[:int(min_length)])

            for i, sample in enumerate(samples):
                mixed_samples[i] += sample * volume

        # Apply master volume and normalize
        max_sample = max(abs(s) for s in mixed_samples) if mixed_samples else 1
        if max_sample > 32767:
            normalize = 32767 / max_sample
        else:
            normalize = 1.0

        # Convert back to bytes
        output_samples = [
            int(s * self._master_volume * normalize)
            for s in mixed_samples
        ]

        # Clip
        output_samples = [max(-32768, min(32767, s)) for s in output_samples]

        return struct.pack(f"<{len(output_samples)}h", *output_samples)


class AudioResampler:
    """
    Audio resampler.

    Converts audio between sample rates.
    """

    def __init__(self, from_rate: int, to_rate: int, channels: int = 1):
        self.from_rate = from_rate
        self.to_rate = to_rate
        self.channels = channels

        # Calculate ratio
        self.ratio = to_rate / from_rate

        # Filter coefficients for anti-aliasing
        self._filter_length = 16
        self._filter = self._create_sinc_filter()

    def _create_sinc_filter(self) -> List[float]:
        """Create sinc filter for interpolation."""
        cutoff = min(self.from_rate, self.to_rate) / max(self.from_rate, self.to_rate)

        coeffs = []
        for i in range(self._filter_length):
            x = i - self._filter_length // 2
            if x == 0:
                coeffs.append(cutoff)
            else:
                coeffs.append(math.sin(math.pi * cutoff * x) / (math.pi * x))

        # Apply Hamming window
        for i in range(len(coeffs)):
            window = 0.54 - 0.46 * math.cos(2 * math.pi * i / (len(coeffs) - 1))
            coeffs[i] *= window

        # Normalize
        total = sum(coeffs)
        coeffs = [c / total for c in coeffs]

        return coeffs

    def resample(self, audio: bytes) -> bytes:
        """Resample audio to target rate."""
        if self.from_rate == self.to_rate:
            return audio

        # Convert to samples
        samples = list(struct.unpack(f"<{len(audio)//2}h", audio))

        # Calculate output length
        output_length = int(len(samples) * self.ratio)
        output = []

        for i in range(output_length):
            # Source position
            src_pos = i / self.ratio
            src_idx = int(src_pos)
            frac = src_pos - src_idx

            # Linear interpolation (simple but effective)
            if src_idx + 1 < len(samples):
                sample = samples[src_idx] * (1 - frac) + samples[src_idx + 1] * frac
            else:
                sample = samples[min(src_idx, len(samples) - 1)]

            output.append(int(sample))

        return struct.pack(f"<{len(output)}h", *output)


class AudioProcessor:
    """
    Complete audio processor.

    Combines VAD, noise reduction, and other processing.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        enable_vad: bool = True,
        enable_noise_reduction: bool = True,
    ):
        self.sample_rate = sample_rate
        self.channels = channels

        # Processors
        self.vad = VADDetector(sample_rate) if enable_vad else None
        self.noise_reducer = NoiseReducer(sample_rate) if enable_noise_reduction else None

        # Processing chain
        self._processors: List[Callable[[bytes], bytes]] = []

        # Statistics
        self._frames_processed = 0
        self._speech_frames = 0

    def add_processor(self, processor: Callable[[bytes], bytes]) -> "AudioProcessor":
        """Add custom processor to chain."""
        self._processors.append(processor)
        return self

    def process(self, audio: bytes) -> Tuple[bytes, bool]:
        """
        Process audio frame.

        Returns (processed_audio, is_speech)
        """
        result = audio

        # Run VAD
        is_speech = True
        if self.vad:
            is_speech = self.vad.process(audio)

        # Apply noise reduction
        if self.noise_reducer:
            result = self.noise_reducer.process(result)

        # Apply custom processors
        for processor in self._processors:
            result = processor(result)

        # Update stats
        self._frames_processed += 1
        if is_speech:
            self._speech_frames += 1

        return result, is_speech

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "frames_processed": self._frames_processed,
            "speech_frames": self._speech_frames,
            "speech_ratio": self._speech_frames / max(1, self._frames_processed),
        }

    def reset(self) -> None:
        """Reset processor state."""
        if self.vad:
            self.vad.reset()
        if self.noise_reducer:
            self.noise_reducer.reset()
        self._frames_processed = 0
        self._speech_frames = 0
