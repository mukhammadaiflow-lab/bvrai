"""
Streaming Orchestrator Service - Ultra-Low Latency Voice AI Pipeline.

This service provides sub-300ms end-to-end latency for voice conversations
by implementing parallel streaming ASR → LLM → TTS processing with
speculative execution and intelligent buffering.

Key Features:
- Parallel streaming pipeline (ASR, LLM, TTS run concurrently)
- Speculative LLM execution on partial transcripts
- Adaptive audio buffering with jitter compensation
- Real-time latency monitoring and optimization
- Automatic fallback and circuit breaker patterns

Architecture:
    Audio Input
         ↓
    [VAD + ASR Streaming] ─────→ [Partial Transcript]
         ↓                              ↓
    [Speech End Detection]    [Speculative LLM Start]
         ↓                              ↓
    [Final Transcript] ──────→ [LLM Streaming Response]
         ↓                              ↓
    [Token Aggregation] ←─────── [TTS Streaming]
         ↓
    Audio Output

Target Latency Breakdown:
- ASR Processing: 50-100ms (streaming with partial results)
- LLM First Token: 100-150ms (with speculative execution)
- TTS First Audio: 50-75ms (streaming synthesis)
- Total End-to-End: <300ms (from speech end to first audio)
"""

__version__ = "1.0.0"
__author__ = "Builder Engine Team"
