"""
Call Event Handlers Module

This module provides event handlers for call lifecycle events,
including connection, audio processing, and state management.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
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
    Type,
    TypeVar,
    Union,
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
from .session import SessionLifecycleManager


logger = logging.getLogger(__name__)


class HandlerPriority(int, Enum):
    """Handler execution priority."""
    CRITICAL = 0  # Must run first (e.g., authentication)
    HIGH = 10
    NORMAL = 50
    LOW = 90
    LOGGING = 100  # Must run last (e.g., logging)


@dataclass
class HandlerResult:
    """Result from a handler execution."""

    success: bool
    handler_name: str
    duration_ms: float
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    stop_propagation: bool = False  # If True, no more handlers will run


class EventHandler(ABC):
    """Abstract base class for event handlers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Handler name."""
        pass

    @property
    def priority(self) -> HandlerPriority:
        """Handler priority."""
        return HandlerPriority.NORMAL

    @property
    def handled_events(self) -> Set[EventType]:
        """Events this handler processes."""
        return set(EventType)  # All events by default

    @abstractmethod
    async def handle(
        self,
        event: OrchestratorEvent,
        session: CallSession,
    ) -> HandlerResult:
        """
        Handle an event.

        Args:
            event: The event to handle
            session: The call session

        Returns:
            Handler result
        """
        pass


class CallConnectedHandler(EventHandler):
    """Handles call connected events."""

    @property
    def name(self) -> str:
        return "call_connected"

    @property
    def handled_events(self) -> Set[EventType]:
        return {EventType.CALL_ANSWERED}

    def __init__(
        self,
        lifecycle: SessionLifecycleManager,
        initial_greeting_callback: Optional[
            Callable[[CallSession], Coroutine[Any, Any, str]]
        ] = None,
    ):
        """
        Initialize handler.

        Args:
            lifecycle: Session lifecycle manager
            initial_greeting_callback: Optional callback to get initial greeting
        """
        self.lifecycle = lifecycle
        self.initial_greeting_callback = initial_greeting_callback

    async def handle(
        self,
        event: OrchestratorEvent,
        session: CallSession,
    ) -> HandlerResult:
        """Handle call connected event."""
        start_time = time.time()

        try:
            # Transition to connected state
            await self.lifecycle.transition_state(
                session,
                CallState.CONNECTED,
                reason="Call answered",
            )

            result_data = {
                "connected_at": datetime.utcnow().isoformat(),
            }

            # Generate initial greeting if callback provided
            if self.initial_greeting_callback:
                greeting = await self.initial_greeting_callback(session)
                result_data["initial_greeting"] = greeting

            return HandlerResult(
                success=True,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                data=result_data,
            )

        except Exception as e:
            logger.exception(f"Call connected handler failed: {e}")
            return HandlerResult(
                success=False,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


class AudioReceivedHandler(EventHandler):
    """Handles incoming audio data."""

    @property
    def name(self) -> str:
        return "audio_received"

    @property
    def handled_events(self) -> Set[EventType]:
        return {EventType.AUDIO_RECEIVED}

    @property
    def priority(self) -> HandlerPriority:
        return HandlerPriority.HIGH

    def __init__(
        self,
        audio_processor: Optional[
            Callable[[bytes, CallSession], Coroutine[Any, Any, Dict[str, Any]]]
        ] = None,
        buffer_size: int = 32000,  # ~1 second at 16kHz 16-bit
    ):
        """
        Initialize handler.

        Args:
            audio_processor: Optional audio processing callback
            buffer_size: Audio buffer size in bytes
        """
        self.audio_processor = audio_processor
        self.buffer_size = buffer_size
        self._buffers: Dict[str, bytearray] = {}

    async def handle(
        self,
        event: OrchestratorEvent,
        session: CallSession,
    ) -> HandlerResult:
        """Handle audio received event."""
        start_time = time.time()

        try:
            audio_data = event.data.get("audio", b"")

            # Buffer audio
            if session.id not in self._buffers:
                self._buffers[session.id] = bytearray()

            self._buffers[session.id].extend(audio_data)

            result_data = {
                "bytes_received": len(audio_data),
                "buffer_size": len(self._buffers[session.id]),
            }

            # Process if buffer is full
            if len(self._buffers[session.id]) >= self.buffer_size:
                if self.audio_processor:
                    buffer = bytes(self._buffers[session.id])
                    self._buffers[session.id].clear()

                    process_result = await self.audio_processor(buffer, session)
                    result_data["processing_result"] = process_result

            return HandlerResult(
                success=True,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                data=result_data,
            )

        except Exception as e:
            logger.exception(f"Audio handler failed: {e}")
            return HandlerResult(
                success=False,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def cleanup_session(self, session_id: str) -> None:
        """Clean up buffers for a session."""
        self._buffers.pop(session_id, None)


class TranscriptHandler(EventHandler):
    """Handles transcript updates."""

    @property
    def name(self) -> str:
        return "transcript"

    @property
    def handled_events(self) -> Set[EventType]:
        return {EventType.TRANSCRIPT_PARTIAL, EventType.TRANSCRIPT_FINAL}

    def __init__(
        self,
        lifecycle: SessionLifecycleManager,
        transcript_callback: Optional[
            Callable[[CallSession, str, bool], Coroutine[Any, Any, None]]
        ] = None,
    ):
        """
        Initialize handler.

        Args:
            lifecycle: Session lifecycle manager
            transcript_callback: Callback for transcript updates (session, text, is_final)
        """
        self.lifecycle = lifecycle
        self.transcript_callback = transcript_callback

    async def handle(
        self,
        event: OrchestratorEvent,
        session: CallSession,
    ) -> HandlerResult:
        """Handle transcript event."""
        start_time = time.time()

        try:
            text = event.data.get("text", "")
            is_final = event.event_type == EventType.TRANSCRIPT_FINAL
            role = ParticipantRole(event.data.get("role", "customer"))
            confidence = event.data.get("confidence", 1.0)

            result_data = {
                "text": text,
                "is_final": is_final,
                "role": role.value,
            }

            # Add to session transcript if final
            if is_final and text.strip():
                await self.lifecycle.add_transcript_entry(
                    session, role, text, confidence
                )

                # Update state to customer speaking if customer is talking
                if role == ParticipantRole.CUSTOMER:
                    if session.state not in (
                        CallState.CUSTOMER_SPEAKING,
                        CallState.PROCESSING,
                    ):
                        await self.lifecycle.transition_state(
                            session,
                            CallState.CUSTOMER_SPEAKING,
                        )

            # Call transcript callback if provided
            if self.transcript_callback:
                await self.transcript_callback(session, text, is_final)

            return HandlerResult(
                success=True,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                data=result_data,
            )

        except Exception as e:
            logger.exception(f"Transcript handler failed: {e}")
            return HandlerResult(
                success=False,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


class InterruptionHandler(EventHandler):
    """Handles customer interruptions during agent speech."""

    @property
    def name(self) -> str:
        return "interruption"

    @property
    def handled_events(self) -> Set[EventType]:
        return {EventType.INTERRUPTION_DETECTED}

    @property
    def priority(self) -> HandlerPriority:
        return HandlerPriority.HIGH

    def __init__(
        self,
        lifecycle: SessionLifecycleManager,
        stop_speech_callback: Optional[
            Callable[[CallSession], Coroutine[Any, Any, None]]
        ] = None,
    ):
        """
        Initialize handler.

        Args:
            lifecycle: Session lifecycle manager
            stop_speech_callback: Callback to stop agent speech
        """
        self.lifecycle = lifecycle
        self.stop_speech_callback = stop_speech_callback

    async def handle(
        self,
        event: OrchestratorEvent,
        session: CallSession,
    ) -> HandlerResult:
        """Handle interruption event."""
        start_time = time.time()

        try:
            # Only handle if agent is speaking
            if session.state != CallState.AGENT_SPEAKING:
                return HandlerResult(
                    success=True,
                    handler_name=self.name,
                    duration_ms=(time.time() - start_time) * 1000,
                    data={"action": "ignored", "reason": "agent_not_speaking"},
                )

            # Stop agent speech
            if self.stop_speech_callback:
                await self.stop_speech_callback(session)

            # Transition to customer speaking
            await self.lifecycle.transition_state(
                session,
                CallState.CUSTOMER_SPEAKING,
                reason="Customer interruption",
            )

            # Track interruption metric
            session.metrics.interruption_count += 1

            return HandlerResult(
                success=True,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                data={
                    "action": "stopped_speech",
                    "interruption_count": session.metrics.interruption_count,
                },
            )

        except Exception as e:
            logger.exception(f"Interruption handler failed: {e}")
            return HandlerResult(
                success=False,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


class EndOfTurnHandler(EventHandler):
    """Handles end of customer turn detection."""

    @property
    def name(self) -> str:
        return "end_of_turn"

    @property
    def handled_events(self) -> Set[EventType]:
        return {EventType.END_OF_TURN}

    def __init__(
        self,
        lifecycle: SessionLifecycleManager,
        generate_response_callback: Optional[
            Callable[[CallSession, str], Coroutine[Any, Any, str]]
        ] = None,
    ):
        """
        Initialize handler.

        Args:
            lifecycle: Session lifecycle manager
            generate_response_callback: Callback to generate agent response
        """
        self.lifecycle = lifecycle
        self.generate_response_callback = generate_response_callback

    async def handle(
        self,
        event: OrchestratorEvent,
        session: CallSession,
    ) -> HandlerResult:
        """Handle end of turn event."""
        start_time = time.time()

        try:
            # Get the customer's utterance
            customer_text = event.data.get("text", "")

            # Transition to processing
            await self.lifecycle.transition_state(
                session,
                CallState.PROCESSING,
                reason="Processing customer input",
            )

            result_data = {
                "customer_text": customer_text,
            }

            # Generate response if callback provided
            if self.generate_response_callback and customer_text.strip():
                response = await self.generate_response_callback(session, customer_text)
                result_data["agent_response"] = response

                # Transition to agent speaking
                await self.lifecycle.transition_state(
                    session,
                    CallState.AGENT_SPEAKING,
                )

                # Add agent response to transcript
                await self.lifecycle.add_transcript_entry(
                    session,
                    ParticipantRole.AGENT,
                    response,
                )

            return HandlerResult(
                success=True,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                data=result_data,
            )

        except Exception as e:
            logger.exception(f"End of turn handler failed: {e}")
            return HandlerResult(
                success=False,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


class FunctionCallHandler(EventHandler):
    """Handles function call requests from the agent."""

    @property
    def name(self) -> str:
        return "function_call"

    @property
    def handled_events(self) -> Set[EventType]:
        return {EventType.FUNCTION_CALLED}

    def __init__(
        self,
        function_registry: Optional[Dict[str, Callable]] = None,
        default_timeout_seconds: float = 30.0,
    ):
        """
        Initialize handler.

        Args:
            function_registry: Registry of callable functions
            default_timeout_seconds: Default timeout for function execution
        """
        self.function_registry = function_registry or {}
        self.default_timeout = default_timeout_seconds

    def register_function(
        self,
        name: str,
        func: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register a function."""
        self.function_registry[name] = func

    async def handle(
        self,
        event: OrchestratorEvent,
        session: CallSession,
    ) -> HandlerResult:
        """Handle function call event."""
        start_time = time.time()

        try:
            func_name = event.data.get("function", "")
            func_args = event.data.get("arguments", {})
            call_id = event.data.get("call_id", str(uuid.uuid4()))

            result_data = {
                "function": func_name,
                "call_id": call_id,
            }

            # Find function in registry
            func = self.function_registry.get(func_name)
            if not func:
                return HandlerResult(
                    success=False,
                    handler_name=self.name,
                    duration_ms=(time.time() - start_time) * 1000,
                    error=f"Unknown function: {func_name}",
                    data=result_data,
                )

            # Execute function with timeout
            try:
                result = await asyncio.wait_for(
                    func(**func_args),
                    timeout=self.default_timeout,
                )
                result_data["result"] = result
                result_data["success"] = True

                # Track function call metric
                session.metrics.function_calls += 1

            except asyncio.TimeoutError:
                result_data["error"] = "Function execution timed out"
                result_data["success"] = False

            except Exception as e:
                result_data["error"] = str(e)
                result_data["success"] = False

            return HandlerResult(
                success=result_data.get("success", False),
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                data=result_data,
            )

        except Exception as e:
            logger.exception(f"Function call handler failed: {e}")
            return HandlerResult(
                success=False,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


class TransferHandler(EventHandler):
    """Handles call transfer requests."""

    @property
    def name(self) -> str:
        return "transfer"

    @property
    def handled_events(self) -> Set[EventType]:
        return {EventType.CALL_TRANSFERRED}

    def __init__(
        self,
        lifecycle: SessionLifecycleManager,
        transfer_callback: Optional[
            Callable[[CallSession, str], Coroutine[Any, Any, bool]]
        ] = None,
    ):
        """
        Initialize handler.

        Args:
            lifecycle: Session lifecycle manager
            transfer_callback: Callback to execute transfer
        """
        self.lifecycle = lifecycle
        self.transfer_callback = transfer_callback

    async def handle(
        self,
        event: OrchestratorEvent,
        session: CallSession,
    ) -> HandlerResult:
        """Handle transfer event."""
        start_time = time.time()

        try:
            target = event.data.get("target", "")
            transfer_type = event.data.get("type", "warm")  # warm or cold

            result_data = {
                "target": target,
                "type": transfer_type,
            }

            # Transition to transferring state
            await self.lifecycle.transition_state(
                session,
                CallState.TRANSFERRING,
                reason=f"Transfer to {target}",
            )

            # Execute transfer
            if self.transfer_callback:
                success = await self.transfer_callback(session, target)
                result_data["transfer_success"] = success

                if success:
                    session.metrics.transfer_count += 1
                else:
                    # Return to connected state on failure
                    await self.lifecycle.transition_state(
                        session,
                        CallState.CONNECTED,
                        reason="Transfer failed",
                    )

            return HandlerResult(
                success=True,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                data=result_data,
            )

        except Exception as e:
            logger.exception(f"Transfer handler failed: {e}")
            return HandlerResult(
                success=False,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


class CallEndedHandler(EventHandler):
    """Handles call ended events."""

    @property
    def name(self) -> str:
        return "call_ended"

    @property
    def handled_events(self) -> Set[EventType]:
        return {EventType.CALL_ENDED, EventType.CALL_FAILED}

    @property
    def priority(self) -> HandlerPriority:
        return HandlerPriority.LOW

    def __init__(
        self,
        lifecycle: SessionLifecycleManager,
        post_call_callback: Optional[
            Callable[[CallSession], Coroutine[Any, Any, None]]
        ] = None,
    ):
        """
        Initialize handler.

        Args:
            lifecycle: Session lifecycle manager
            post_call_callback: Callback for post-call processing
        """
        self.lifecycle = lifecycle
        self.post_call_callback = post_call_callback

    async def handle(
        self,
        event: OrchestratorEvent,
        session: CallSession,
    ) -> HandlerResult:
        """Handle call ended event."""
        start_time = time.time()

        try:
            reason = event.data.get("reason", "unknown")
            error = event.data.get("error")

            # Determine terminal state
            if event.event_type == EventType.CALL_FAILED:
                terminal_state = CallState.FAILED
            else:
                terminal_state = CallState.COMPLETED

            # End session
            await self.lifecycle.end_session(
                session,
                state=terminal_state,
                reason=reason,
                error=error,
            )

            # Execute post-call callback
            if self.post_call_callback:
                await self.post_call_callback(session)

            return HandlerResult(
                success=True,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                data={
                    "terminal_state": terminal_state.value,
                    "reason": reason,
                    "duration_seconds": session.metrics.total_duration_seconds,
                },
            )

        except Exception as e:
            logger.exception(f"Call ended handler failed: {e}")
            return HandlerResult(
                success=False,
                handler_name=self.name,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


class LoggingHandler(EventHandler):
    """Logs all events for debugging and auditing."""

    @property
    def name(self) -> str:
        return "logging"

    @property
    def priority(self) -> HandlerPriority:
        return HandlerPriority.LOGGING

    def __init__(
        self,
        log_level: int = logging.DEBUG,
        include_data: bool = True,
    ):
        """
        Initialize handler.

        Args:
            log_level: Logging level
            include_data: Include event data in logs
        """
        self.log_level = log_level
        self.include_data = include_data

    async def handle(
        self,
        event: OrchestratorEvent,
        session: CallSession,
    ) -> HandlerResult:
        """Log event."""
        start_time = time.time()

        log_msg = (
            f"Event: {event.event_type.value} | "
            f"Session: {session.id} | "
            f"State: {session.state.value}"
        )

        if self.include_data and event.data:
            # Truncate large data
            data_str = str(event.data)
            if len(data_str) > 500:
                data_str = data_str[:500] + "..."
            log_msg += f" | Data: {data_str}"

        logger.log(self.log_level, log_msg)

        return HandlerResult(
            success=True,
            handler_name=self.name,
            duration_ms=(time.time() - start_time) * 1000,
        )


class MetricsHandler(EventHandler):
    """Collects and reports metrics for events."""

    @property
    def name(self) -> str:
        return "metrics"

    @property
    def priority(self) -> HandlerPriority:
        return HandlerPriority.LOGGING

    def __init__(
        self,
        metrics_callback: Optional[
            Callable[[str, Dict[str, Any]], Coroutine[Any, Any, None]]
        ] = None,
    ):
        """
        Initialize handler.

        Args:
            metrics_callback: Callback to report metrics
        """
        self.metrics_callback = metrics_callback
        self._event_counts: Dict[str, int] = {}
        self._latency_samples: Dict[str, List[float]] = {}

    async def handle(
        self,
        event: OrchestratorEvent,
        session: CallSession,
    ) -> HandlerResult:
        """Collect metrics for event."""
        start_time = time.time()

        event_name = event.event_type.value

        # Count events
        self._event_counts[event_name] = self._event_counts.get(event_name, 0) + 1

        # Report metrics
        if self.metrics_callback:
            await self.metrics_callback(event_name, {
                "session_id": session.id,
                "organization_id": session.organization_id,
                "state": session.state.value,
                "timestamp": event.timestamp.isoformat(),
            })

        return HandlerResult(
            success=True,
            handler_name=self.name,
            duration_ms=(time.time() - start_time) * 1000,
            data={"event_count": self._event_counts.get(event_name, 0)},
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return {
            "event_counts": dict(self._event_counts),
        }


class HandlerRegistry:
    """
    Registry and dispatcher for event handlers.

    Manages handler registration and event dispatch with
    priority-based execution order.
    """

    def __init__(self):
        """Initialize handler registry."""
        self._handlers: Dict[EventType, List[Tuple[HandlerPriority, EventHandler]]] = {}
        self._global_handlers: List[Tuple[HandlerPriority, EventHandler]] = []

    def register(
        self,
        handler: EventHandler,
        events: Optional[Set[EventType]] = None,
    ) -> None:
        """
        Register a handler.

        Args:
            handler: Event handler
            events: Events to handle (uses handler's handled_events if None)
        """
        target_events = events or handler.handled_events

        # If handler handles all events, add to global handlers
        if target_events == set(EventType):
            self._global_handlers.append((handler.priority, handler))
            self._global_handlers.sort(key=lambda x: x[0].value)
        else:
            # Add to specific event handlers
            for event_type in target_events:
                if event_type not in self._handlers:
                    self._handlers[event_type] = []

                self._handlers[event_type].append((handler.priority, handler))
                self._handlers[event_type].sort(key=lambda x: x[0].value)

        logger.debug(
            f"Registered handler {handler.name} with priority "
            f"{handler.priority.name} for {len(target_events)} events"
        )

    def unregister(self, handler_name: str) -> bool:
        """
        Unregister a handler by name.

        Args:
            handler_name: Handler name

        Returns:
            True if handler was found and removed
        """
        removed = False

        # Remove from global handlers
        self._global_handlers = [
            (p, h) for p, h in self._global_handlers
            if h.name != handler_name
        ]

        # Remove from event-specific handlers
        for event_type in self._handlers:
            original_len = len(self._handlers[event_type])
            self._handlers[event_type] = [
                (p, h) for p, h in self._handlers[event_type]
                if h.name != handler_name
            ]
            if len(self._handlers[event_type]) < original_len:
                removed = True

        return removed

    async def dispatch(
        self,
        event: OrchestratorEvent,
        session: CallSession,
    ) -> List[HandlerResult]:
        """
        Dispatch an event to all registered handlers.

        Args:
            event: Event to dispatch
            session: Call session

        Returns:
            List of handler results
        """
        results = []

        # Collect all handlers for this event
        handlers: List[Tuple[HandlerPriority, EventHandler]] = []

        # Add global handlers
        handlers.extend(self._global_handlers)

        # Add event-specific handlers
        if event.event_type in self._handlers:
            handlers.extend(self._handlers[event.event_type])

        # Sort by priority
        handlers.sort(key=lambda x: x[0].value)

        # Execute handlers in order
        for _, handler in handlers:
            try:
                result = await handler.handle(event, session)
                results.append(result)

                # Check if propagation should stop
                if result.stop_propagation:
                    logger.debug(
                        f"Handler {handler.name} stopped event propagation"
                    )
                    break

            except Exception as e:
                logger.exception(f"Handler {handler.name} raised exception: {e}")
                results.append(HandlerResult(
                    success=False,
                    handler_name=handler.name,
                    duration_ms=0,
                    error=str(e),
                ))

        return results

    def get_handlers(self, event_type: EventType) -> List[str]:
        """Get handler names for an event type."""
        handlers = []

        # Add global handlers
        handlers.extend([h.name for _, h in self._global_handlers])

        # Add event-specific handlers
        if event_type in self._handlers:
            handlers.extend([h.name for _, h in self._handlers[event_type]])

        return handlers


def create_default_handlers(
    lifecycle: SessionLifecycleManager,
) -> HandlerRegistry:
    """
    Create a handler registry with default handlers.

    Args:
        lifecycle: Session lifecycle manager

    Returns:
        Configured handler registry
    """
    registry = HandlerRegistry()

    # Register default handlers
    registry.register(CallConnectedHandler(lifecycle))
    registry.register(AudioReceivedHandler())
    registry.register(TranscriptHandler(lifecycle))
    registry.register(InterruptionHandler(lifecycle))
    registry.register(EndOfTurnHandler(lifecycle))
    registry.register(FunctionCallHandler())
    registry.register(TransferHandler(lifecycle))
    registry.register(CallEndedHandler(lifecycle))
    registry.register(LoggingHandler())
    registry.register(MetricsHandler())

    return registry


__all__ = [
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
]
