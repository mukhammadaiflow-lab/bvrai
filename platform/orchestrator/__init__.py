"""
Call Orchestrator Package

This package provides the complete call orchestration system for
real-time voice call processing, including:

- Session management with lifecycle tracking
- Audio processing pipeline (STT, LLM, TTS)
- Event handling and dispatch
- Provider integration
- Metrics and recovery

Example usage:

    from platform.orchestrator import (
        CallOrchestrator,
        OrchestratorConfig,
        CallDirection,
    )

    # Create orchestrator
    config = OrchestratorConfig(
        max_concurrent_calls=100,
        enable_recovery=True,
    )
    orchestrator = CallOrchestrator(config)

    # Start orchestrator
    await orchestrator.start()

    # Start a call
    session = await orchestrator.start_call(
        organization_id="org_123",
        agent_id="agent_456",
        direction=CallDirection.INBOUND,
        from_number="+15551234567",
        to_number="+15559876543",
    )

    # Process audio
    response_audio = await orchestrator.process_audio(
        session.id,
        audio_chunk=audio_data,
    )

    # End call
    await orchestrator.end_call(session.id)

    # Stop orchestrator
    await orchestrator.stop()
"""

# Base types
from .base import (
    CallState,
    CallDirection,
    ParticipantRole,
    EventType,
    CallConfig,
    CallMetrics,
    TranscriptEntry,
    ConversationTurn,
    OrchestratorEvent,
    CallSession,
    PipelineContext,
)

# Pipeline stages
from .pipeline import (
    PipelineStage,
    SpeechToTextStage,
    InterruptionDetectorStage,
    TurnDetectorStage,
    LLMProcessingStage,
    FunctionExecutorStage,
    TextToSpeechStage,
    CallPipeline,
    create_call_pipeline,
)

# Session management
from .session import (
    SessionStorageBackend,
    InMemorySessionStorage,
    RedisSessionStorage,
    SessionPoolConfig,
    SessionPool,
    SessionLifecycleManager,
    session_context,
    SessionRecoveryManager,
)

# Event handlers
from .handlers import (
    HandlerPriority,
    HandlerResult,
    EventHandler,
    CallConnectedHandler,
    AudioReceivedHandler,
    TranscriptHandler,
    InterruptionHandler,
    EndOfTurnHandler,
    FunctionCallHandler,
    TransferHandler,
    CallEndedHandler,
    LoggingHandler,
    MetricsHandler,
    HandlerRegistry,
    create_default_handlers,
)

# Main orchestrator
from .orchestrator import (
    OrchestratorState,
    OrchestratorConfig,
    OrchestratorMetrics,
    CallOrchestrator,
    orchestrator_context,
    OrchestratorBuilder,
    create_orchestrator,
)


__all__ = [
    # Base types
    "CallState",
    "CallDirection",
    "ParticipantRole",
    "EventType",
    "CallConfig",
    "CallMetrics",
    "TranscriptEntry",
    "ConversationTurn",
    "OrchestratorEvent",
    "CallSession",
    "PipelineContext",
    # Pipeline
    "PipelineStage",
    "SpeechToTextStage",
    "InterruptionDetectorStage",
    "TurnDetectorStage",
    "LLMProcessingStage",
    "FunctionExecutorStage",
    "TextToSpeechStage",
    "CallPipeline",
    "create_call_pipeline",
    # Session
    "SessionStorageBackend",
    "InMemorySessionStorage",
    "RedisSessionStorage",
    "SessionPoolConfig",
    "SessionPool",
    "SessionLifecycleManager",
    "session_context",
    "SessionRecoveryManager",
    # Handlers
    "HandlerPriority",
    "HandlerResult",
    "EventHandler",
    "CallConnectedHandler",
    "AudioReceivedHandler",
    "TranscriptHandler",
    "InterruptionHandler",
    "EndOfTurnHandler",
    "FunctionCallHandler",
    "TransferHandler",
    "CallEndedHandler",
    "LoggingHandler",
    "MetricsHandler",
    "HandlerRegistry",
    "create_default_handlers",
    # Orchestrator
    "OrchestratorState",
    "OrchestratorConfig",
    "OrchestratorMetrics",
    "CallOrchestrator",
    "orchestrator_context",
    "OrchestratorBuilder",
    "create_orchestrator",
]
