"""
Voice Lab Service - Voice Cloning & Voice Management.

This service provides comprehensive voice cloning, customization,
and management capabilities for the Builder Engine platform.

Key Features:
- Instant voice cloning from audio samples (15-60 seconds)
- Professional voice cloning with studio-quality training
- Voice style customization (emotion, pace, pitch, etc.)
- Multi-provider support (ElevenLabs, PlayHT, Cartesia, Resemble)
- Voice library management
- Voice sharing and marketplace
- Compliance and consent management

Architecture:
    Audio Sample(s)
         │
         ▼
    ┌────────────────┐
    │ Audio Analyzer │
    │ (Quality/VAD)  │
    └────────────────┘
         │
         ▼
    ┌────────────────┐
    │ Voice Cloner   │
    │ (Multi-Provider)│
    └────────────────┘
         │
         ▼
    ┌────────────────┐
    │ Voice Registry │
    │ (DB + Storage) │
    └────────────────┘
         │
         ▼
    ┌────────────────┐
    │ Voice Preview  │
    │ (TTS Synthesis)│
    └────────────────┘

Supported Providers:
- ElevenLabs: Industry-leading voice quality
- PlayHT: Fast cloning, good quality
- Cartesia: Ultra-low latency
- Resemble: Real-time voice cloning
- Azure Neural Voices: Enterprise-grade
"""

__version__ = "1.0.0"
__author__ = "Builder Engine Team"
