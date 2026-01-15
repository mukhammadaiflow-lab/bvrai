"""
Emotion Engine Core Components.

This module provides the core emotion analysis pipeline:
- ProsodicsAnalyzer: Extracts pitch, energy, rhythm features from audio
- EmotionClassifier: Maps prosodic features to emotion categories
- EmotionalStateTracker: Tracks emotional state over time with smoothing
- ResponseAdvisor: Generates response recommendations based on emotional state
- EmotionEngine: Main orchestrator combining all components
"""

from .prosodics import ProsodicsAnalyzer
from .classifier import EmotionClassifier
from .state_tracker import EmotionalStateTracker
from .advisor import ResponseAdvisor
from .engine import EmotionEngine, get_engine

__all__ = [
    "ProsodicsAnalyzer",
    "EmotionClassifier",
    "EmotionalStateTracker",
    "ResponseAdvisor",
    "EmotionEngine",
    "get_engine",
]
