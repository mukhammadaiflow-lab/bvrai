"""TTS adapters for different text-to-speech providers."""

from app.adapters.base import TTSAdapter, AudioChunk
from app.adapters.elevenlabs import ElevenLabsAdapter
from app.adapters.mock import MockTTSAdapter

__all__ = [
    "TTSAdapter",
    "AudioChunk",
    "ElevenLabsAdapter",
    "MockTTSAdapter",
]
