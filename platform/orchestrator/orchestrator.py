"""
Call Orchestrator Module

This module provides the main orchestrator that coordinates all components
of the voice call processing system.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from .base import (
    CallSession,
    CallState,
    CallDirection,
    CallConfig,
    CallMetrics,
    EventType,
    OrchestratorEvent,
    TranscriptEntry,
    ConversationTurn,
    ParticipantRole,
    PipelineContext,
)
from .pipeline import (
    CallPipeline,
    PipelineStage,
    create_call_pipeline,
)
from .session import (
    SessionPool,
    SessionPoolConfig,
    SessionLifecycleManager,
    SessionRecoveryManager,
    SessionStorageBackend,
    InMemorySessionStorage,
    session_context,
)
from .handlers import (
    HandlerRegistry,
    EventHandler,
    HandlerResult,
    create_default_handlers,
)


logger = logging.getLogger(__name__)


class OrchestratorState(str, Enum):
    """Orchestrator operational state."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class OrchestratorConfig:
    """Configuration for the call orchestrator."""

    # Session pool settings
    max_concurrent_calls: int = 1000
    max_calls_per_org: int = 100
    session_timeout_seconds: int = 3600

    # Pipeline settings
    pipeline_timeout_ms: int = 5000
    enable_interruption_detection: bool = True
    enable_turn_detection: bool = True

    # Audio settings
    sample_rate: int = 16000
    audio_format: str = "pcm16"
    chunk_size_ms: int = 20

    # Recovery settings
    enable_recovery: bool = True
    checkpoint_interval_seconds: int = 5

    # Metrics settings
    enable_metrics: bool = True
    metrics_flush_interval_seconds: int = 10

    # Provider settings
    stt_provider: Optional[str] = None
    llm_provider: Optional[str] = None
    tts_provider: Optional[str] = None


@dataclass
class OrchestratorMetrics:
    """Runtime metrics for the orchestrator."""

    # Call counts
    total_calls_started: int = 0
    total_calls_completed: int = 0
    total_calls_failed: int = 0
    active_calls: int = 0

    # Duration stats
    total_call_duration_seconds: float = 0.0
    average_call_duration_seconds: float = 0.0

    # Latency stats
    average_stt_latency_ms: float = 0.0
    average_llm_latency_ms: float = 0.0
    average_tts_latency_ms: float = 0.0
    average_pipeline_latency_ms: float = 0.0

    # Error counts
    stt_errors: int = 0
    llm_errors: int = 0
    tts_errors: int = 0
    pipeline_errors: int = 0

    # Uptime
    start_time: Optional[datetime] = None
    uptime_seconds: float = 0.0


class CallOrchestrator:
    """
    Main call orchestrator.

    Coordinates all components of the voice call processing system:
    - Session management
    - Audio pipeline processing
    - Event handling
    - Provider integration
    - Metrics collection
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        storage: Optional[SessionStorageBackend] = None,
    ):
        """
        Initialize the call orchestrator.

        Args:
            config: Orchestrator configuration
            storage: Session storage backend
        """
        self.config = config or OrchestratorConfig()
        self.state = OrchestratorState.STOPPED
        self.metrics = OrchestratorMetrics()

        # Initialize storage
        self._storage = storage or InMemorySessionStorage()

        # Initialize session pool
        pool_config = SessionPoolConfig(
            max_sessions_per_org=self.config.max_calls_per_org,
            max_total_sessions=self.config.max_concurrent_calls,
            session_timeout_seconds=self.config.session_timeout_seconds,
        )
        self._session_pool = SessionPool(self._storage, pool_config)

        # Initialize lifecycle manager
        self._lifecycle = SessionLifecycleManager(
            self._session_pool,
            event_callback=self._on_session_event,
        )

        # Initialize recovery manager
        self._recovery = SessionRecoveryManager(
            self._storage,
            checkpoint_interval_seconds=self.config.checkpoint_interval_seconds,
        ) if self.config.enable_recovery else None

        # Initialize handler registry
        self._handlers = create_default_handlers(self._lifecycle)

        # Initialize pipeline (will be configured with providers)
        self._pipeline: Optional[CallPipeline] = None

        # Provider instances
        self._stt_provider: Optional[Any] = None
        self._llm_provider: Optional[Any] = None
        self._tts_provider: Optional[Any] = None

        # Function handlers for agent tools
        self._function_handlers: Dict[str, Callable] = {}

        # Active call tasks
        self._call_tasks: Dict[str, asyncio.Task] = {}

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []

        # Event callbacks
        self._event_callbacks: List[
            Callable[[OrchestratorEvent], Coroutine[Any, Any, None]]
        ] = []

        # Lock for state changes
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the orchestrator."""
        async with self._lock:
            if self.state != OrchestratorState.STOPPED:
                logger.warning(f"Cannot start orchestrator in state {self.state}")
                return

            self.state = OrchestratorState.STARTING
            logger.info("Starting call orchestrator...")

        try:
            # Start session pool
            await self._session_pool.start()

            # Initialize pipeline
            self._pipeline = create_call_pipeline(
                stt_provider=self._stt_provider,
                llm_provider=self._llm_provider,
                tts_provider=self._tts_provider,
                function_handlers=self._function_handlers,
            )

            # Start metrics collection if enabled
            if self.config.enable_metrics:
                self._background_tasks.append(
                    asyncio.create_task(self._metrics_loop())
                )

            # Recover any interrupted sessions
            if self._recovery:
                await self._recover_sessions()

            # Update state
            self.metrics.start_time = datetime.utcnow()
            self.state = OrchestratorState.RUNNING
            logger.info("Call orchestrator started successfully")

        except Exception as e:
            logger.exception(f"Failed to start orchestrator: {e}")
            self.state = OrchestratorState.ERROR
            raise

    async def stop(self) -> None:
        """Stop the orchestrator."""
        async with self._lock:
            if self.state not in (OrchestratorState.RUNNING, OrchestratorState.ERROR):
                logger.warning(f"Cannot stop orchestrator in state {self.state}")
                return

            self.state = OrchestratorState.STOPPING
            logger.info("Stopping call orchestrator...")

        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # End all active calls
            for session_id, task in list(self._call_tasks.items()):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Stop session pool
            await self._session_pool.stop()

            self.state = OrchestratorState.STOPPED
            logger.info("Call orchestrator stopped")

        except Exception as e:
            logger.exception(f"Error stopping orchestrator: {e}")
            self.state = OrchestratorState.ERROR

    async def _metrics_loop(self) -> None:
        """Background task to collect and report metrics."""
        while True:
            try:
                await asyncio.sleep(self.config.metrics_flush_interval_seconds)
                await self._update_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Metrics collection error: {e}")

    async def _update_metrics(self) -> None:
        """Update orchestrator metrics."""
        # Update uptime
        if self.metrics.start_time:
            self.metrics.uptime_seconds = (
                datetime.utcnow() - self.metrics.start_time
            ).total_seconds()

        # Update active calls
        self.metrics.active_calls = len(self._call_tasks)

        # Calculate averages
        if self.metrics.total_calls_completed > 0:
            self.metrics.average_call_duration_seconds = (
                self.metrics.total_call_duration_seconds /
                self.metrics.total_calls_completed
            )

        # Get pipeline metrics
        if self._pipeline:
            stage_metrics = self._pipeline.get_stage_metrics()
            if "stt" in stage_metrics:
                self.metrics.average_stt_latency_ms = (
                    stage_metrics["stt"]["avg_latency_ms"]
                )
            if "llm" in stage_metrics:
                self.metrics.average_llm_latency_ms = (
                    stage_metrics["llm"]["avg_latency_ms"]
                )
            if "tts" in stage_metrics:
                self.metrics.average_tts_latency_ms = (
                    stage_metrics["tts"]["avg_latency_ms"]
                )

    async def _recover_sessions(self) -> None:
        """Recover interrupted sessions on startup."""
        if not self._recovery:
            return

        sessions = await self._recovery.list_recoverable_sessions()
        logger.info(f"Found {len(sessions)} sessions to recover")

        for session in sessions:
            try:
                await self._recovery.recover_session(session.id)
                logger.info(f"Recovered session {session.id}")
            except Exception as e:
                logger.exception(f"Failed to recover session {session.id}: {e}")

    async def _on_session_event(self, event: OrchestratorEvent) -> None:
        """Handle session events."""
        # Dispatch to handlers
        session = await self._session_pool.get(event.session_id)
        if session:
            await self._handlers.dispatch(event, session)

        # Notify external callbacks
        for callback in self._event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.exception(f"Event callback failed: {e}")

    def configure_providers(
        self,
        stt_provider: Optional[Any] = None,
        llm_provider: Optional[Any] = None,
        tts_provider: Optional[Any] = None,
    ) -> None:
        """
        Configure AI providers.

        Args:
            stt_provider: Speech-to-text provider
            llm_provider: LLM provider
            tts_provider: Text-to-speech provider
        """
        self._stt_provider = stt_provider
        self._llm_provider = llm_provider
        self._tts_provider = tts_provider

        # Reconfigure pipeline if already running
        if self._pipeline:
            self._pipeline = create_call_pipeline(
                stt_provider=stt_provider,
                llm_provider=llm_provider,
                tts_provider=tts_provider,
                function_handlers=self._function_handlers,
            )

    def register_handler(self, handler: EventHandler) -> None:
        """
        Register an event handler.

        Args:
            handler: Event handler to register
        """
        self._handlers.register(handler)

    def register_function(
        self,
        name: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """
        Register a function handler for agent tools.

        Args:
            name: Function name
            handler: Async function handler
        """
        self._function_handlers[name] = handler

    def add_event_callback(
        self,
        callback: Callable[[OrchestratorEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Add an event callback.

        Args:
            callback: Callback function
        """
        self._event_callbacks.append(callback)

    async def start_call(
        self,
        organization_id: str,
        agent_id: str,
        direction: CallDirection,
        from_number: str,
        to_number: str,
        config: Optional[CallConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CallSession:
        """
        Start a new call.

        Args:
            organization_id: Organization ID
            agent_id: Agent ID
            direction: Call direction
            from_number: From phone number
            to_number: To phone number
            config: Call configuration
            metadata: Additional metadata

        Returns:
            Call session
        """
        if self.state != OrchestratorState.RUNNING:
            raise RuntimeError(f"Orchestrator not running (state: {self.state})")

        # Create session
        session = await self._lifecycle.create_session(
            organization_id=organization_id,
            agent_id=agent_id,
            direction=direction,
            from_number=from_number,
            to_number=to_number,
            config=config,
            metadata=metadata,
        )

        # Start checkpointing if recovery enabled
        if self._recovery:
            await self._recovery.start_checkpointing(session)

        # Update metrics
        self.metrics.total_calls_started += 1

        logger.info(
            f"Started call {session.id} ({direction.value}) "
            f"from {from_number} to {to_number}"
        )

        return session

    async def process_audio(
        self,
        session_id: str,
        audio_chunk: bytes,
    ) -> Optional[bytes]:
        """
        Process an audio chunk for a call.

        Args:
            session_id: Session ID
            audio_chunk: Audio data

        Returns:
            Response audio or None
        """
        session = await self._session_pool.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        if not self._pipeline:
            logger.error("Pipeline not initialized")
            return None

        # Create pipeline context
        context = PipelineContext(
            session=session,
            audio_chunk=audio_chunk,
        )

        # Process through pipeline
        try:
            context = await asyncio.wait_for(
                self._pipeline.process(context),
                timeout=self.config.pipeline_timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Pipeline timeout for session {session_id}")
            self.metrics.pipeline_errors += 1
            return None

        # Handle state transitions based on pipeline results
        await self._handle_pipeline_results(session, context)

        # Update session
        await self._session_pool.update(session)

        return context.agent_audio

    async def _handle_pipeline_results(
        self,
        session: CallSession,
        context: PipelineContext,
    ) -> None:
        """Handle pipeline processing results."""
        # Handle interruption
        if context.interruption_detected:
            await self._handlers.dispatch(
                OrchestratorEvent(
                    id=f"evt_{uuid.uuid4().hex[:12]}",
                    session_id=session.id,
                    event_type=EventType.INTERRUPTION_DETECTED,
                    data={"partial_transcript": context.partial_transcript},
                ),
                session,
            )

        # Handle end of turn
        if context.end_of_turn:
            await self._handlers.dispatch(
                OrchestratorEvent(
                    id=f"evt_{uuid.uuid4().hex[:12]}",
                    session_id=session.id,
                    event_type=EventType.END_OF_TURN,
                    data={"text": context.final_transcript},
                ),
                session,
            )

        # Handle function calls
        for func_call in context.pending_function_calls:
            await self._handlers.dispatch(
                OrchestratorEvent(
                    id=f"evt_{uuid.uuid4().hex[:12]}",
                    session_id=session.id,
                    event_type=EventType.FUNCTION_CALLED,
                    data=func_call,
                ),
                session,
            )

        # Handle agent response
        if context.agent_response:
            session.add_transcript_entry(
                ParticipantRole.AGENT,
                context.agent_response,
            )

        # Handle transcript updates
        if context.final_transcript:
            session.add_transcript_entry(
                ParticipantRole.CUSTOMER,
                context.final_transcript,
                context.stt_latency_ms / 1000 if context.stt_latency_ms else 1.0,
            )

    async def end_call(
        self,
        session_id: str,
        reason: Optional[str] = None,
    ) -> None:
        """
        End a call.

        Args:
            session_id: Session ID
            reason: Reason for ending
        """
        session = await self._session_pool.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return

        # Stop checkpointing
        if self._recovery:
            await self._recovery.stop_checkpointing(session_id)

        # Cancel any active tasks
        task = self._call_tasks.pop(session_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # End session
        await self._lifecycle.end_session(
            session,
            state=CallState.COMPLETED,
            reason=reason or "Call ended normally",
        )

        # Update metrics
        self.metrics.total_calls_completed += 1
        self.metrics.total_call_duration_seconds += (
            session.metrics.total_duration_seconds
        )

        logger.info(
            f"Ended call {session_id} (duration: "
            f"{session.metrics.total_duration_seconds:.1f}s)"
        )

    async def handle_call_event(
        self,
        session_id: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Handle an external call event.

        Args:
            session_id: Session ID
            event_type: Event type string
            data: Event data
        """
        session = await self._session_pool.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return

        # Map string event type to enum
        try:
            event_type_enum = EventType(event_type)
        except ValueError:
            logger.warning(f"Unknown event type: {event_type}")
            return

        # Create and dispatch event
        event = OrchestratorEvent(
            id=f"evt_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            event_type=event_type_enum,
            data=data,
        )

        await self._handlers.dispatch(event, session)

    async def transfer_call(
        self,
        session_id: str,
        target: str,
        transfer_type: str = "warm",
    ) -> bool:
        """
        Transfer a call to another destination.

        Args:
            session_id: Session ID
            target: Transfer target (phone number or SIP URI)
            transfer_type: Transfer type (warm or cold)

        Returns:
            True if transfer initiated successfully
        """
        session = await self._session_pool.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False

        # Dispatch transfer event
        await self._handlers.dispatch(
            OrchestratorEvent(
                id=f"evt_{uuid.uuid4().hex[:12]}",
                session_id=session_id,
                event_type=EventType.CALL_TRANSFERRED,
                data={
                    "target": target,
                    "type": transfer_type,
                },
            ),
            session,
        )

        return True

    async def get_session(self, session_id: str) -> Optional[CallSession]:
        """
        Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Call session or None
        """
        return await self._session_pool.get(session_id)

    async def list_active_calls(
        self,
        organization_id: Optional[str] = None,
    ) -> List[CallSession]:
        """
        List active calls.

        Args:
            organization_id: Filter by organization

        Returns:
            List of active call sessions
        """
        return await self._session_pool.list_active(organization_id)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get orchestrator metrics.

        Returns:
            Metrics dictionary
        """
        return {
            "state": self.state.value,
            "uptime_seconds": self.metrics.uptime_seconds,
            "calls": {
                "total_started": self.metrics.total_calls_started,
                "total_completed": self.metrics.total_calls_completed,
                "total_failed": self.metrics.total_calls_failed,
                "active": self.metrics.active_calls,
            },
            "duration": {
                "total_seconds": self.metrics.total_call_duration_seconds,
                "average_seconds": self.metrics.average_call_duration_seconds,
            },
            "latency": {
                "stt_ms": self.metrics.average_stt_latency_ms,
                "llm_ms": self.metrics.average_llm_latency_ms,
                "tts_ms": self.metrics.average_tts_latency_ms,
                "pipeline_ms": self.metrics.average_pipeline_latency_ms,
            },
            "errors": {
                "stt": self.metrics.stt_errors,
                "llm": self.metrics.llm_errors,
                "tts": self.metrics.tts_errors,
                "pipeline": self.metrics.pipeline_errors,
            },
            "pool": self._session_pool.get_metrics(),
        }


@asynccontextmanager
async def orchestrator_context(
    config: Optional[OrchestratorConfig] = None,
    storage: Optional[SessionStorageBackend] = None,
) -> AsyncIterator[CallOrchestrator]:
    """
    Context manager for call orchestrator.

    Handles orchestrator lifecycle automatically.

    Args:
        config: Orchestrator configuration
        storage: Session storage backend

    Yields:
        Running call orchestrator

    Example:
        async with orchestrator_context() as orchestrator:
            session = await orchestrator.start_call(...)
            # Process call
    """
    orchestrator = CallOrchestrator(config, storage)
    await orchestrator.start()

    try:
        yield orchestrator
    finally:
        await orchestrator.stop()


class OrchestratorBuilder:
    """
    Builder for configuring and creating call orchestrators.

    Provides a fluent API for orchestrator configuration.
    """

    def __init__(self):
        """Initialize builder."""
        self._config = OrchestratorConfig()
        self._storage: Optional[SessionStorageBackend] = None
        self._stt_provider: Optional[Any] = None
        self._llm_provider: Optional[Any] = None
        self._tts_provider: Optional[Any] = None
        self._handlers: List[EventHandler] = []
        self._functions: Dict[str, Callable] = {}
        self._event_callbacks: List[
            Callable[[OrchestratorEvent], Coroutine[Any, Any, None]]
        ] = []

    def with_config(self, config: OrchestratorConfig) -> "OrchestratorBuilder":
        """Set orchestrator configuration."""
        self._config = config
        return self

    def with_max_concurrent_calls(self, max_calls: int) -> "OrchestratorBuilder":
        """Set maximum concurrent calls."""
        self._config.max_concurrent_calls = max_calls
        return self

    def with_session_timeout(self, timeout_seconds: int) -> "OrchestratorBuilder":
        """Set session timeout."""
        self._config.session_timeout_seconds = timeout_seconds
        return self

    def with_storage(self, storage: SessionStorageBackend) -> "OrchestratorBuilder":
        """Set storage backend."""
        self._storage = storage
        return self

    def with_stt_provider(self, provider: Any) -> "OrchestratorBuilder":
        """Set STT provider."""
        self._stt_provider = provider
        return self

    def with_llm_provider(self, provider: Any) -> "OrchestratorBuilder":
        """Set LLM provider."""
        self._llm_provider = provider
        return self

    def with_tts_provider(self, provider: Any) -> "OrchestratorBuilder":
        """Set TTS provider."""
        self._tts_provider = provider
        return self

    def with_handler(self, handler: EventHandler) -> "OrchestratorBuilder":
        """Add an event handler."""
        self._handlers.append(handler)
        return self

    def with_function(
        self,
        name: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
    ) -> "OrchestratorBuilder":
        """Register a function handler."""
        self._functions[name] = handler
        return self

    def with_event_callback(
        self,
        callback: Callable[[OrchestratorEvent], Coroutine[Any, Any, None]],
    ) -> "OrchestratorBuilder":
        """Add an event callback."""
        self._event_callbacks.append(callback)
        return self

    def enable_recovery(self, enabled: bool = True) -> "OrchestratorBuilder":
        """Enable or disable session recovery."""
        self._config.enable_recovery = enabled
        return self

    def enable_metrics(self, enabled: bool = True) -> "OrchestratorBuilder":
        """Enable or disable metrics collection."""
        self._config.enable_metrics = enabled
        return self

    def build(self) -> CallOrchestrator:
        """
        Build the configured orchestrator.

        Returns:
            Configured call orchestrator (not started)
        """
        orchestrator = CallOrchestrator(self._config, self._storage)

        # Configure providers
        orchestrator.configure_providers(
            stt_provider=self._stt_provider,
            llm_provider=self._llm_provider,
            tts_provider=self._tts_provider,
        )

        # Register handlers
        for handler in self._handlers:
            orchestrator.register_handler(handler)

        # Register functions
        for name, handler in self._functions.items():
            orchestrator.register_function(name, handler)

        # Add callbacks
        for callback in self._event_callbacks:
            orchestrator.add_event_callback(callback)

        return orchestrator


def create_orchestrator(
    config: Optional[OrchestratorConfig] = None,
    storage: Optional[SessionStorageBackend] = None,
    stt_provider: Optional[Any] = None,
    llm_provider: Optional[Any] = None,
    tts_provider: Optional[Any] = None,
) -> CallOrchestrator:
    """
    Create a call orchestrator with default configuration.

    Args:
        config: Orchestrator configuration
        storage: Session storage backend
        stt_provider: STT provider
        llm_provider: LLM provider
        tts_provider: TTS provider

    Returns:
        Configured call orchestrator
    """
    orchestrator = CallOrchestrator(config, storage)
    orchestrator.configure_providers(
        stt_provider=stt_provider,
        llm_provider=llm_provider,
        tts_provider=tts_provider,
    )
    return orchestrator


__all__ = [
    "OrchestratorState",
    "OrchestratorConfig",
    "OrchestratorMetrics",
    "CallOrchestrator",
    "orchestrator_context",
    "OrchestratorBuilder",
    "create_orchestrator",
]
