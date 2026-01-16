"""
Voice Activity Detection (VAD) module.

Provides multiple VAD implementations:
- SileroVAD: High-accuracy neural network-based VAD
- WebRTCVAD: Google's WebRTC VAD (fast, good accuracy)
- EnergyVAD: Simple energy-based VAD (fastest, basic accuracy)

Features:
- Speech start/end detection with configurable thresholds
- Minimum speech/silence duration requirements
- Padding for natural speech boundaries
- Event-driven architecture for real-time processing
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading
import struct

from .audio import AudioFormat, AudioChunk, calculate_rms, calculate_db

logger = logging.getLogger(__name__)


class VADState(str, Enum):
    """Voice Activity Detection states."""
    SILENCE = "silence"           # No speech detected
    SPEECH_START = "speech_start" # Speech just started
    SPEECH = "speech"             # Ongoing speech
    SPEECH_END = "speech_end"     # Speech just ended
    UNKNOWN = "unknown"           # Initial/unknown state


class VADEventType(str, Enum):
    """Types of VAD events."""
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    SPEECH_CHUNK = "speech_chunk"
    SILENCE_CHUNK = "silence_chunk"
    VAD_READY = "vad_ready"
    VAD_ERROR = "vad_error"


@dataclass
class VADEvent:
    """
    Voice Activity Detection event.

    Attributes:
        event_type: Type of VAD event
        timestamp_ms: Timestamp when event occurred
        duration_ms: Duration of speech/silence segment
        audio_chunk: Associated audio chunk (if any)
        confidence: Detection confidence (0.0 - 1.0)
        metadata: Additional event metadata
    """
    event_type: VADEventType
    timestamp_ms: float = 0.0
    duration_ms: float = 0.0
    audio_chunk: Optional[AudioChunk] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VADConfig:
    """
    Configuration for Voice Activity Detection.

    Attributes:
        mode: Aggressiveness mode (0-3, higher = more aggressive filtering)
        sample_rate: Expected audio sample rate
        frame_duration_ms: Frame size for processing (10, 20, or 30ms)

        speech_threshold: Probability/energy threshold for speech detection
        silence_threshold: Threshold for silence detection

        min_speech_duration_ms: Minimum duration to confirm speech started
        min_silence_duration_ms: Minimum silence duration to confirm speech ended

        speech_pad_ms: Padding added before speech start
        silence_pad_ms: Padding added after speech end

        max_speech_duration_ms: Maximum continuous speech duration
    """
    mode: int = 2
    sample_rate: int = 16000
    frame_duration_ms: int = 20

    speech_threshold: float = 0.5
    silence_threshold: float = 0.3

    min_speech_duration_ms: float = 100
    min_silence_duration_ms: float = 300

    speech_pad_ms: float = 50
    silence_pad_ms: float = 100

    max_speech_duration_ms: float = 60000  # 60 seconds

    # Energy-based VAD specific
    energy_threshold_db: float = -35.0
    energy_smoothing: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "sample_rate": self.sample_rate,
            "frame_duration_ms": self.frame_duration_ms,
            "speech_threshold": self.speech_threshold,
            "silence_threshold": self.silence_threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "speech_pad_ms": self.speech_pad_ms,
            "silence_pad_ms": self.silence_pad_ms,
            "max_speech_duration_ms": self.max_speech_duration_ms,
        }


class VoiceActivityDetector(ABC):
    """Abstract base class for Voice Activity Detectors."""

    def __init__(self, config: VADConfig):
        self.config = config
        self._state = VADState.SILENCE
        self._speech_start_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._current_speech_duration: float = 0.0
        self._current_silence_duration: float = 0.0
        self._is_speaking = False

        # Buffer for padding
        self._pre_speech_buffer: List[AudioChunk] = []
        self._max_pre_buffer_chunks = int(config.speech_pad_ms / config.frame_duration_ms) + 1

        # Event callbacks
        self._event_callbacks: List[Callable[[VADEvent], None]] = []

        self._lock = threading.Lock()

    @property
    def state(self) -> VADState:
        """Current VAD state."""
        return self._state

    @property
    def is_speaking(self) -> bool:
        """Whether speech is currently detected."""
        return self._is_speaking

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the VAD (load models, etc.)."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the VAD and release resources."""
        pass

    @abstractmethod
    def process_frame(self, audio_chunk: AudioChunk) -> Tuple[bool, float]:
        """
        Process a single audio frame.

        Args:
            audio_chunk: Audio frame to process

        Returns:
            Tuple of (is_speech, confidence)
        """
        pass

    async def process(self, audio_chunk: AudioChunk) -> Optional[VADEvent]:
        """
        Process an audio chunk and return VAD event if state changed.

        Args:
            audio_chunk: Audio chunk to process

        Returns:
            VADEvent if state changed, None otherwise
        """
        # Detect speech in this frame
        is_speech, confidence = self.process_frame(audio_chunk)

        current_time = audio_chunk.timestamp_ms
        audio_chunk.is_speech = is_speech

        with self._lock:
            event = self._update_state(audio_chunk, is_speech, confidence, current_time)

        if event:
            await self._emit_event(event)

        return event

    def _update_state(
        self,
        chunk: AudioChunk,
        is_speech: bool,
        confidence: float,
        current_time: float,
    ) -> Optional[VADEvent]:
        """Update internal state and return event if state changed."""
        event = None

        if is_speech:
            self._current_silence_duration = 0.0

            if not self._is_speaking:
                # Potential speech start
                if self._speech_start_time is None:
                    self._speech_start_time = current_time
                    self._current_speech_duration = chunk.duration_ms
                else:
                    self._current_speech_duration += chunk.duration_ms

                # Check if we've reached minimum speech duration
                if self._current_speech_duration >= self.config.min_speech_duration_ms:
                    self._is_speaking = True
                    self._state = VADState.SPEECH_START
                    self._silence_start_time = None

                    # Include pre-speech buffer
                    event = VADEvent(
                        event_type=VADEventType.SPEECH_START,
                        timestamp_ms=self._speech_start_time,
                        confidence=confidence,
                        metadata={"pre_buffer_chunks": len(self._pre_speech_buffer)},
                    )

                    self._pre_speech_buffer.clear()
                    self._state = VADState.SPEECH

            else:
                # Ongoing speech
                self._current_speech_duration += chunk.duration_ms

                # Check for max speech duration
                if self._current_speech_duration >= self.config.max_speech_duration_ms:
                    event = VADEvent(
                        event_type=VADEventType.SPEECH_END,
                        timestamp_ms=current_time,
                        duration_ms=self._current_speech_duration,
                        confidence=confidence,
                        metadata={"reason": "max_duration"},
                    )
                    self._reset_state()

        else:
            self._current_speech_duration = 0.0
            self._speech_start_time = None

            if self._is_speaking:
                # Potential speech end
                if self._silence_start_time is None:
                    self._silence_start_time = current_time
                    self._current_silence_duration = chunk.duration_ms
                else:
                    self._current_silence_duration += chunk.duration_ms

                # Check if we've reached minimum silence duration
                if self._current_silence_duration >= self.config.min_silence_duration_ms:
                    speech_duration = current_time - (self._speech_start_time or 0)

                    event = VADEvent(
                        event_type=VADEventType.SPEECH_END,
                        timestamp_ms=current_time,
                        duration_ms=speech_duration,
                        confidence=confidence,
                    )
                    self._reset_state()

            else:
                # Continuous silence - maintain pre-speech buffer
                self._pre_speech_buffer.append(chunk)
                if len(self._pre_speech_buffer) > self._max_pre_buffer_chunks:
                    self._pre_speech_buffer.pop(0)

        return event

    def _reset_state(self) -> None:
        """Reset internal state after speech ends."""
        self._is_speaking = False
        self._state = VADState.SILENCE
        self._speech_start_time = None
        self._silence_start_time = None
        self._current_speech_duration = 0.0
        self._current_silence_duration = 0.0

    def add_event_callback(self, callback: Callable[[VADEvent], None]) -> None:
        """Add a callback for VAD events."""
        self._event_callbacks.append(callback)

    def remove_event_callback(self, callback: Callable[[VADEvent], None]) -> None:
        """Remove an event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    async def _emit_event(self, event: VADEvent) -> None:
        """Emit event to all registered callbacks."""
        for callback in self._event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in VAD event callback: {e}")

    def reset(self) -> None:
        """Reset VAD state."""
        with self._lock:
            self._reset_state()
            self._pre_speech_buffer.clear()

    def get_pre_speech_buffer(self) -> List[AudioChunk]:
        """Get the pre-speech audio buffer."""
        with self._lock:
            return list(self._pre_speech_buffer)


class EnergyVAD(VoiceActivityDetector):
    """
    Simple energy-based Voice Activity Detector.

    Uses audio RMS/dB levels to detect speech.
    Fast and lightweight, suitable for basic use cases.
    """

    def __init__(self, config: VADConfig):
        super().__init__(config)
        self._smoothed_energy: float = 0.0
        self._energy_history: List[float] = []
        self._max_history = 50

    async def initialize(self) -> None:
        """Initialize energy VAD."""
        logger.info("EnergyVAD initialized")

    async def shutdown(self) -> None:
        """Shutdown energy VAD."""
        pass

    def process_frame(self, audio_chunk: AudioChunk) -> Tuple[bool, float]:
        """
        Process frame using energy-based detection.

        Returns:
            Tuple of (is_speech, confidence)
        """
        db = calculate_db(audio_chunk.data, audio_chunk.format.sample_width)

        # Apply smoothing
        self._smoothed_energy = (
            self.config.energy_smoothing * self._smoothed_energy +
            (1 - self.config.energy_smoothing) * db
        )

        # Track history for adaptive thresholding
        self._energy_history.append(db)
        if len(self._energy_history) > self._max_history:
            self._energy_history.pop(0)

        # Detect speech based on threshold
        is_speech = self._smoothed_energy > self.config.energy_threshold_db

        # Calculate confidence based on how far above threshold
        if is_speech:
            # Higher energy = higher confidence
            confidence = min(1.0, (self._smoothed_energy - self.config.energy_threshold_db) / 20.0)
        else:
            confidence = 0.0

        return is_speech, confidence


class WebRTCVAD(VoiceActivityDetector):
    """
    WebRTC-based Voice Activity Detector.

    Uses Google's WebRTC VAD which is fast and accurate.
    Requires the 'webrtcvad' package.
    """

    def __init__(self, config: VADConfig):
        super().__init__(config)
        self._vad = None
        self._supported_rates = [8000, 16000, 32000, 48000]
        self._resampler = None

    async def initialize(self) -> None:
        """Initialize WebRTC VAD."""
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self.config.mode)
            logger.info(f"WebRTCVAD initialized with mode {self.config.mode}")
        except ImportError:
            logger.warning("webrtcvad not installed, falling back to EnergyVAD")
            raise

    async def shutdown(self) -> None:
        """Shutdown WebRTC VAD."""
        self._vad = None

    def process_frame(self, audio_chunk: AudioChunk) -> Tuple[bool, float]:
        """
        Process frame using WebRTC VAD.

        Returns:
            Tuple of (is_speech, confidence)
        """
        if not self._vad:
            return False, 0.0

        # WebRTC VAD requires specific sample rates and frame durations
        if audio_chunk.format.sample_rate not in self._supported_rates:
            logger.warning(f"Unsupported sample rate: {audio_chunk.format.sample_rate}")
            return False, 0.0

        # Frame must be 10, 20, or 30ms
        expected_samples = int(
            audio_chunk.format.sample_rate * self.config.frame_duration_ms / 1000
        )
        expected_bytes = expected_samples * audio_chunk.format.sample_width

        if len(audio_chunk.data) != expected_bytes:
            # Pad or truncate
            if len(audio_chunk.data) < expected_bytes:
                data = audio_chunk.data + bytes(expected_bytes - len(audio_chunk.data))
            else:
                data = audio_chunk.data[:expected_bytes]
        else:
            data = audio_chunk.data

        try:
            is_speech = self._vad.is_speech(
                data,
                audio_chunk.format.sample_rate,
            )
            # WebRTC VAD is binary, use energy for confidence
            if is_speech:
                confidence = min(1.0, calculate_rms(data, 2) / 5000)
            else:
                confidence = 0.0

            return is_speech, confidence

        except Exception as e:
            logger.error(f"WebRTC VAD error: {e}")
            return False, 0.0


class SileroVAD(VoiceActivityDetector):
    """
    Silero VAD - High accuracy neural network-based VAD.

    Uses the Silero VAD model from snakers4/silero-vad.
    Requires torch and torchaudio.
    """

    def __init__(self, config: VADConfig):
        super().__init__(config)
        self._model = None
        self._utils = None
        self._get_speech_timestamps = None
        self._collect_chunks = None
        self._device = "cpu"

    async def initialize(self) -> None:
        """Initialize Silero VAD model."""
        try:
            import torch

            # Load Silero VAD model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                verbose=False,
            )

            self._model = model
            self._utils = utils
            (
                self._get_speech_timestamps,
                _,  # save_audio
                _,  # read_audio
                _,  # VADIterator
                self._collect_chunks,
            ) = utils

            # Check for GPU
            if torch.cuda.is_available():
                self._device = "cuda"
                self._model = self._model.to(self._device)

            logger.info(f"SileroVAD initialized on {self._device}")

        except ImportError:
            logger.warning("torch not installed, Silero VAD unavailable")
            raise

    async def shutdown(self) -> None:
        """Shutdown Silero VAD."""
        self._model = None

    def process_frame(self, audio_chunk: AudioChunk) -> Tuple[bool, float]:
        """
        Process frame using Silero VAD.

        Returns:
            Tuple of (is_speech, probability)
        """
        if not self._model:
            return False, 0.0

        try:
            import torch
            import numpy as np

            # Convert audio to tensor
            audio_array = np.frombuffer(audio_chunk.data, dtype=np.int16)
            audio_tensor = torch.from_numpy(audio_array).float() / 32768.0

            # Ensure correct sample rate (Silero expects 16kHz)
            if audio_chunk.format.sample_rate != 16000:
                # Would need resampling here
                logger.warning("Silero VAD requires 16kHz audio")
                return False, 0.0

            # Get speech probability
            if self._device == "cuda":
                audio_tensor = audio_tensor.to(self._device)

            speech_prob = self._model(audio_tensor, 16000).item()

            is_speech = speech_prob > self.config.speech_threshold

            return is_speech, speech_prob

        except Exception as e:
            logger.error(f"Silero VAD error: {e}")
            return False, 0.0


class AdaptiveVAD(VoiceActivityDetector):
    """
    Adaptive VAD that adjusts thresholds based on ambient noise.

    Combines multiple detection methods for robust performance.
    """

    def __init__(self, config: VADConfig):
        super().__init__(config)
        self._energy_vad = EnergyVAD(config)
        self._primary_vad: Optional[VoiceActivityDetector] = None

        # Adaptive threshold parameters
        self._noise_floor_db: float = -60.0
        self._noise_samples: List[float] = []
        self._max_noise_samples = 100
        self._adaptation_rate = 0.01

        # Calibration
        self._is_calibrating = False
        self._calibration_samples: List[float] = []
        self._calibration_duration_ms = 2000

    async def initialize(self) -> None:
        """Initialize adaptive VAD with fallback strategy."""
        await self._energy_vad.initialize()

        # Try to initialize advanced VADs
        try:
            self._primary_vad = WebRTCVAD(self.config)
            await self._primary_vad.initialize()
            logger.info("AdaptiveVAD using WebRTCVAD as primary")
        except Exception:
            try:
                self._primary_vad = SileroVAD(self.config)
                await self._primary_vad.initialize()
                logger.info("AdaptiveVAD using SileroVAD as primary")
            except Exception:
                logger.info("AdaptiveVAD using EnergyVAD only")
                self._primary_vad = None

    async def shutdown(self) -> None:
        """Shutdown all VAD components."""
        await self._energy_vad.shutdown()
        if self._primary_vad:
            await self._primary_vad.shutdown()

    def process_frame(self, audio_chunk: AudioChunk) -> Tuple[bool, float]:
        """
        Process frame using adaptive multi-VAD approach.

        Returns:
            Tuple of (is_speech, confidence)
        """
        # Update noise floor estimation
        self._update_noise_estimation(audio_chunk)

        # Get energy-based detection
        energy_speech, energy_conf = self._energy_vad.process_frame(audio_chunk)

        if self._primary_vad:
            # Get primary VAD detection
            primary_speech, primary_conf = self._primary_vad.process_frame(audio_chunk)

            # Combine results (use primary if confident, otherwise combine)
            if primary_conf > 0.7:
                return primary_speech, primary_conf
            elif primary_speech and energy_speech:
                # Both agree it's speech
                return True, max(primary_conf, energy_conf)
            elif primary_speech or energy_speech:
                # One detects speech - use average confidence
                avg_conf = (primary_conf + energy_conf) / 2
                return avg_conf > 0.4, avg_conf
            else:
                return False, 0.0
        else:
            return energy_speech, energy_conf

    def _update_noise_estimation(self, chunk: AudioChunk) -> None:
        """Update ambient noise floor estimation."""
        db = calculate_db(chunk.data, chunk.format.sample_width)

        if not self._is_speaking:
            # Only update noise floor during silence
            self._noise_samples.append(db)
            if len(self._noise_samples) > self._max_noise_samples:
                self._noise_samples.pop(0)

            # Calculate adaptive noise floor
            if len(self._noise_samples) >= 10:
                sorted_samples = sorted(self._noise_samples)
                # Use 10th percentile as noise floor
                idx = len(sorted_samples) // 10
                new_floor = sorted_samples[idx]

                # Smooth update
                self._noise_floor_db = (
                    (1 - self._adaptation_rate) * self._noise_floor_db +
                    self._adaptation_rate * new_floor
                )

    async def calibrate(self, audio_chunks: List[AudioChunk]) -> None:
        """
        Calibrate VAD using ambient noise samples.

        Args:
            audio_chunks: List of audio chunks recorded during silence
        """
        self._is_calibrating = True

        try:
            noise_levels = []
            for chunk in audio_chunks:
                db = calculate_db(chunk.data, chunk.format.sample_width)
                noise_levels.append(db)

            if noise_levels:
                # Set noise floor to 95th percentile + margin
                sorted_levels = sorted(noise_levels)
                idx = int(len(sorted_levels) * 0.95)
                self._noise_floor_db = sorted_levels[idx]

                # Adjust energy threshold based on noise floor
                self._energy_vad.config.energy_threshold_db = self._noise_floor_db + 10

                logger.info(
                    f"VAD calibrated: noise_floor={self._noise_floor_db:.1f}dB, "
                    f"threshold={self._energy_vad.config.energy_threshold_db:.1f}dB"
                )

        finally:
            self._is_calibrating = False


class VADFactory:
    """Factory for creating VAD instances."""

    @staticmethod
    async def create(
        vad_type: str = "adaptive",
        config: Optional[VADConfig] = None,
    ) -> VoiceActivityDetector:
        """
        Create a VAD instance.

        Args:
            vad_type: Type of VAD ('energy', 'webrtc', 'silero', 'adaptive')
            config: VAD configuration

        Returns:
            Initialized VoiceActivityDetector
        """
        config = config or VADConfig()

        if vad_type == "energy":
            vad = EnergyVAD(config)
        elif vad_type == "webrtc":
            vad = WebRTCVAD(config)
        elif vad_type == "silero":
            vad = SileroVAD(config)
        elif vad_type == "adaptive":
            vad = AdaptiveVAD(config)
        else:
            raise ValueError(f"Unknown VAD type: {vad_type}")

        await vad.initialize()
        return vad
