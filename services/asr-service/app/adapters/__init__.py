"""ASR adapters for different speech-to-text providers."""

from app.adapters.base import ASRAdapter, TranscriptResult, TranscriptEvent
from app.adapters.deepgram import DeepgramAdapter
from app.adapters.mock import MockASRAdapter

__all__ = [
    "ASRAdapter",
    "TranscriptResult",
    "TranscriptEvent",
    "DeepgramAdapter",
    "MockASRAdapter",
]
