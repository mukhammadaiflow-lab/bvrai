"""
Call Manager

Central call management:
- Call lifecycle management
- Event handling
- Integration with other systems
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import uuid

from app.calls.state_machine import (
    CallStateMachine,
    CallState,
    CallEvent,
    TransitionResult,
    StateMachineManager,
)
from app.calls.router import (
    RouterManager,
    RoutingContext,
    RoutingResult,
    RoutingTarget,
    RoutingRule,
)
from app.calls.queue import (
    QueueManager,
    CallQueue,
    QueueConfig,
    QueuePriority,
    QueuedCall,
)
from app.calls.transfer import (
    TransferManager,
    TransferRequest,
    TransferResult,
    TransferType,
)

logger = logging.getLogger(__name__)


class CallDirection(str, Enum):
    """Call direction."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"


@dataclass
class CallManagerConfig:
    """Call manager configuration."""
    # Limits
    max_concurrent_calls: int = 1000
    max_call_duration_seconds: int = 3600
    max_ring_duration_seconds: int = 30

    # Timeouts
    answer_timeout_seconds: int = 30
    idle_timeout_seconds: int = 300

    # Features
    enable_recording: bool = True
    enable_transcription: bool = True
    enable_analytics: bool = True

    # Defaults
    default_queue_id: Optional[str] = None
    default_agent_id: Optional[str] = None


@dataclass
class CallInfo:
    """Call information."""
    call_id: str
    tenant_id: str
    direction: CallDirection

    # Endpoints
    from_number: str = ""
    to_number: str = ""
    caller_name: Optional[str] = None

    # Agent/Queue
    agent_id: Optional[str] = None
    queue_id: Optional[str] = None

    # Session
    session_id: str = ""
    voice_session_id: Optional[str] = None

    # State
    state: CallState = CallState.INITIALIZING
    created_at: datetime = field(default_factory=datetime.utcnow)
    answered_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Recording
    recording_enabled: bool = False
    recording_url: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Get call duration."""
        if self.answered_at:
            end = self.ended_at or datetime.utcnow()
            return (end - self.answered_at).total_seconds()
        return 0.0

    @property
    def ring_duration_seconds(self) -> float:
        """Get ring duration."""
        if self.answered_at:
            return (self.answered_at - self.created_at).total_seconds()
        return 0.0


class CallEventType(str, Enum):
    """Call event types."""
    CALL_INITIATED = "call_initiated"
    CALL_RINGING = "call_ringing"
    CALL_ANSWERED = "call_answered"
    CALL_CONNECTED = "call_connected"
    CALL_ENDED = "call_ended"
    CALL_FAILED = "call_failed"
    CALL_TRANSFERRED = "call_transferred"
    CALL_HELD = "call_held"
    CALL_RESUMED = "call_resumed"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    RECORDING_STARTED = "recording_started"
    RECORDING_STOPPED = "recording_stopped"


@dataclass
class CallEventData:
    """Call event data."""
    event_type: CallEventType
    call_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)


class CallEventHandler(ABC):
    """Abstract call event handler."""

    @abstractmethod
    async def handle(self, event: CallEventData) -> None:
        """Handle call event."""
        pass


class DefaultCallEventHandler(CallEventHandler):
    """Default event handler with logging."""

    async def handle(self, event: CallEventData) -> None:
        """Log call event."""
        logger.info(
            f"Call event: {event.event_type.value} "
            f"call_id={event.call_id} data={event.data}"
        )


class CallManager:
    """
    Central call manager.

    Coordinates:
    - Call lifecycle
    - State management
    - Routing
    - Queuing
    - Transfers
    """

    def __init__(
        self,
        config: Optional[CallManagerConfig] = None,
        event_handler: Optional[CallEventHandler] = None,
    ):
        self.config = config or CallManagerConfig()
        self.event_handler = event_handler or DefaultCallEventHandler()

        # Sub-managers
        self._state_manager = StateMachineManager()
        self._router_manager = RouterManager()
        self._queue_manager = QueueManager()
        self._transfer_manager = TransferManager()

        # Call storage
        self._calls: Dict[str, CallInfo] = {}
        self._calls_by_session: Dict[str, str] = {}
        self._lock = asyncio.Lock()

        # Running state
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Statistics
        self._total_calls = 0
        self._active_calls = 0
        self._completed_calls = 0
        self._failed_calls = 0

    async def start(self) -> None:
        """Start call manager."""
        if self._running:
            return

        self._running = True

        # Start sub-managers
        await self._queue_manager.start()

        # Start background tasks
        self._tasks.append(asyncio.create_task(self._monitor_loop()))
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))

        logger.info("Call manager started")

    async def stop(self) -> None:
        """Stop call manager."""
        self._running = False

        # Stop background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Stop sub-managers
        await self._queue_manager.stop()

        logger.info("Call manager stopped")

    # Call Lifecycle

    async def create_call(
        self,
        tenant_id: str,
        direction: CallDirection,
        from_number: str,
        to_number: str,
        agent_id: Optional[str] = None,
        queue_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CallInfo:
        """Create new call."""
        async with self._lock:
            # Check limits
            if self._active_calls >= self.config.max_concurrent_calls:
                raise ValueError("Max concurrent calls reached")

            call_id = str(uuid.uuid4())
            session_id = str(uuid.uuid4())

            call = CallInfo(
                call_id=call_id,
                tenant_id=tenant_id,
                direction=direction,
                from_number=from_number,
                to_number=to_number,
                agent_id=agent_id,
                queue_id=queue_id or self.config.default_queue_id,
                session_id=session_id,
                recording_enabled=self.config.enable_recording,
                metadata=metadata or {},
            )

            self._calls[call_id] = call
            self._calls_by_session[session_id] = call_id

            # Create state machine
            await self._state_manager.create(call_id)

            self._total_calls += 1
            self._active_calls += 1

        # Emit event
        await self._emit_event(CallEventType.CALL_INITIATED, call_id, {
            "direction": direction.value,
            "from_number": from_number,
            "to_number": to_number,
        })

        logger.info(f"Call created: {call_id}")

        return call

    async def initiate_outbound(
        self,
        tenant_id: str,
        to_number: str,
        from_number: str,
        agent_id: str,
        **kwargs,
    ) -> CallInfo:
        """Initiate outbound call."""
        call = await self.create_call(
            tenant_id=tenant_id,
            direction=CallDirection.OUTBOUND,
            from_number=from_number,
            to_number=to_number,
            agent_id=agent_id,
            **kwargs,
        )

        # Transition to dialing
        await self._state_manager.trigger(call.call_id, CallEvent.DIAL)
        call.state = CallState.DIALING

        return call

    async def handle_inbound(
        self,
        tenant_id: str,
        from_number: str,
        to_number: str,
        **kwargs,
    ) -> CallInfo:
        """Handle inbound call."""
        call = await self.create_call(
            tenant_id=tenant_id,
            direction=CallDirection.INBOUND,
            from_number=from_number,
            to_number=to_number,
            **kwargs,
        )

        # Route the call
        context = RoutingContext(
            call_id=call.call_id,
            caller_number=from_number,
            called_number=to_number,
            tenant_id=tenant_id,
            call_type="inbound",
        )

        routing_result = await self._router_manager.route(context)

        if routing_result.success and routing_result.target:
            target = routing_result.target

            if target.target_type == "agent":
                call.agent_id = target.target_id
                await self._state_manager.trigger(call.call_id, CallEvent.RING)
                call.state = CallState.RINGING

            elif target.target_type == "queue":
                call.queue_id = target.target_id
                await self._queue_call(call)

        else:
            # Use default queue
            if self.config.default_queue_id:
                call.queue_id = self.config.default_queue_id
                await self._queue_call(call)

        await self._emit_event(CallEventType.CALL_RINGING, call.call_id, {
            "agent_id": call.agent_id,
            "queue_id": call.queue_id,
        })

        return call

    async def answer_call(self, call_id: str, agent_id: str) -> bool:
        """Answer a call."""
        call = await self.get_call(call_id)
        if not call:
            return False

        if call.state not in [CallState.RINGING, CallState.QUEUED]:
            return False

        # Update state
        result = await self._state_manager.trigger(call_id, CallEvent.ANSWER)
        if not result or not result.success:
            return False

        call.state = CallState.CONNECTING
        call.agent_id = agent_id
        call.answered_at = datetime.utcnow()

        # Remove from queue if queued
        if call.queue_id:
            queue = await self._queue_manager.get_queue(call.queue_id)
            if queue:
                await queue.remove(call_id)

        # Connect
        await self._state_manager.trigger(call_id, CallEvent.CONNECT)
        call.state = CallState.CONNECTED

        await self._emit_event(CallEventType.CALL_ANSWERED, call_id, {
            "agent_id": agent_id,
            "ring_duration": call.ring_duration_seconds,
        })

        return True

    async def end_call(
        self,
        call_id: str,
        reason: str = "normal",
    ) -> bool:
        """End a call."""
        call = await self.get_call(call_id)
        if not call:
            return False

        if call.state.is_ended:
            return False

        # Determine end event
        if reason == "normal":
            event = CallEvent.COMPLETE
            call.state = CallState.COMPLETED
        elif reason == "failed":
            event = CallEvent.FAIL
            call.state = CallState.FAILED
            self._failed_calls += 1
        elif reason == "cancelled":
            event = CallEvent.CANCEL
            call.state = CallState.CANCELLED
        elif reason == "no_answer":
            event = CallEvent.TIMEOUT
            call.state = CallState.NO_ANSWER
        else:
            event = CallEvent.HANGUP
            call.state = CallState.COMPLETED

        await self._state_manager.trigger(call_id, event)
        call.ended_at = datetime.utcnow()

        async with self._lock:
            self._active_calls -= 1
            if call.state == CallState.COMPLETED:
                self._completed_calls += 1

        await self._emit_event(CallEventType.CALL_ENDED, call_id, {
            "reason": reason,
            "duration": call.duration_seconds,
        })

        logger.info(f"Call ended: {call_id}, reason: {reason}")

        return True

    # Call Operations

    async def hold_call(self, call_id: str) -> bool:
        """Put call on hold."""
        call = await self.get_call(call_id)
        if not call or not call.state.is_active:
            return False

        result = await self._state_manager.trigger(call_id, CallEvent.HOLD)
        if result and result.success:
            call.state = CallState.ON_HOLD
            await self._emit_event(CallEventType.CALL_HELD, call_id)
            return True

        return False

    async def resume_call(self, call_id: str) -> bool:
        """Resume call from hold."""
        call = await self.get_call(call_id)
        if not call or call.state != CallState.ON_HOLD:
            return False

        result = await self._state_manager.trigger(call_id, CallEvent.RESUME)
        if result and result.success:
            call.state = CallState.IN_PROGRESS
            await self._emit_event(CallEventType.CALL_RESUMED, call_id)
            return True

        return False

    async def transfer_call(
        self,
        call_id: str,
        transfer_type: TransferType,
        to_agent_id: Optional[str] = None,
        to_queue_id: Optional[str] = None,
        to_number: Optional[str] = None,
        **kwargs,
    ) -> TransferResult:
        """Transfer a call."""
        call = await self.get_call(call_id)
        if not call or not call.state.is_active:
            return TransferResult(
                transfer_id="",
                success=False,
                status="failed",
                transfer_type=transfer_type,
                error="Call not active",
            )

        request = TransferRequest(
            call_id=call_id,
            transfer_type=transfer_type,
            from_agent_id=call.agent_id,
            from_queue_id=call.queue_id,
            to_agent_id=to_agent_id,
            to_queue_id=to_queue_id,
            to_number=to_number,
            **kwargs,
        )

        # Update state
        await self._state_manager.trigger(call_id, CallEvent.TRANSFER_INITIATE)
        call.state = CallState.TRANSFERRING

        # Execute transfer
        result = await self._transfer_manager.transfer(request)

        if result.success:
            await self._state_manager.trigger(call_id, CallEvent.TRANSFER_COMPLETE)
            call.state = CallState.TRANSFERRED
            call.agent_id = to_agent_id
            call.queue_id = to_queue_id

            await self._emit_event(CallEventType.CALL_TRANSFERRED, call_id, {
                "transfer_type": transfer_type.value,
                "to_agent_id": to_agent_id,
                "to_queue_id": to_queue_id,
            })
        else:
            await self._state_manager.trigger(call_id, CallEvent.TRANSFER_FAIL)
            call.state = CallState.IN_PROGRESS

        return result

    # Recording

    async def start_recording(self, call_id: str) -> bool:
        """Start call recording."""
        call = await self.get_call(call_id)
        if not call or not call.state.is_active:
            return False

        result = await self._state_manager.trigger(call_id, CallEvent.START_RECORDING)
        if result and result.success:
            call.recording_enabled = True
            await self._emit_event(CallEventType.RECORDING_STARTED, call_id)
            return True

        return False

    async def stop_recording(self, call_id: str) -> Optional[str]:
        """Stop call recording."""
        call = await self.get_call(call_id)
        if not call:
            return None

        result = await self._state_manager.trigger(call_id, CallEvent.STOP_RECORDING)
        if result and result.success:
            call.recording_enabled = False
            # In production: finalize recording and get URL
            call.recording_url = f"https://recordings.example.com/{call_id}.wav"
            await self._emit_event(CallEventType.RECORDING_STOPPED, call_id, {
                "recording_url": call.recording_url,
            })
            return call.recording_url

        return None

    # Queries

    async def get_call(self, call_id: str) -> Optional[CallInfo]:
        """Get call by ID."""
        return self._calls.get(call_id)

    async def get_call_by_session(self, session_id: str) -> Optional[CallInfo]:
        """Get call by session ID."""
        call_id = self._calls_by_session.get(session_id)
        if call_id:
            return self._calls.get(call_id)
        return None

    async def list_active_calls(
        self,
        tenant_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[CallInfo]:
        """List active calls."""
        calls = []

        for call in self._calls.values():
            if call.state.is_ended:
                continue

            if tenant_id and call.tenant_id != tenant_id:
                continue

            if agent_id and call.agent_id != agent_id:
                continue

            calls.append(call)

        return calls

    # Queue Management

    async def _queue_call(self, call: CallInfo) -> bool:
        """Queue a call."""
        if not call.queue_id:
            return False

        priority = QueuePriority.NORMAL

        queued = await self._queue_manager.enqueue(
            queue_id=call.queue_id,
            call_id=call.call_id,
            caller_number=call.from_number,
            priority=priority,
            caller_name=call.caller_name,
        )

        if queued:
            await self._state_manager.trigger(call.call_id, CallEvent.QUEUE)
            call.state = CallState.QUEUED
            return True

        return False

    async def create_queue(self, config: QueueConfig) -> CallQueue:
        """Create call queue."""
        return await self._queue_manager.create_queue(config)

    async def get_queue(self, queue_id: str) -> Optional[CallQueue]:
        """Get queue by ID."""
        return await self._queue_manager.get_queue(queue_id)

    # Routing

    def add_routing_rule(self, rule: RoutingRule) -> None:
        """Add routing rule."""
        self._router_manager.add_rule(rule)

    def register_routing_target(self, target: RoutingTarget) -> None:
        """Register routing target."""
        self._router_manager.register_target(target)

    # Event Handling

    async def _emit_event(
        self,
        event_type: CallEventType,
        call_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit call event."""
        event = CallEventData(
            event_type=event_type,
            call_id=call_id,
            data=data or {},
        )

        try:
            await self.event_handler.handle(event)
        except Exception as e:
            logger.error(f"Event handler error: {e}")

    # Background Tasks

    async def _monitor_loop(self) -> None:
        """Monitor calls for timeouts."""
        while self._running:
            try:
                await asyncio.sleep(5)

                now = datetime.utcnow()

                for call in list(self._calls.values()):
                    # Check ring timeout
                    if call.state == CallState.RINGING:
                        ring_duration = (now - call.created_at).total_seconds()
                        if ring_duration > self.config.max_ring_duration_seconds:
                            await self.end_call(call.call_id, "no_answer")

                    # Check max duration
                    if call.state.is_active and call.answered_at:
                        duration = (now - call.answered_at).total_seconds()
                        if duration > self.config.max_call_duration_seconds:
                            await self.end_call(call.call_id, "timeout")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

    async def _cleanup_loop(self) -> None:
        """Clean up ended calls."""
        while self._running:
            try:
                await asyncio.sleep(60)

                # Remove old ended calls from memory
                now = datetime.utcnow()
                to_remove = []

                for call_id, call in self._calls.items():
                    if call.state.is_ended and call.ended_at:
                        age = (now - call.ended_at).total_seconds()
                        if age > 3600:  # 1 hour
                            to_remove.append(call_id)

                async with self._lock:
                    for call_id in to_remove:
                        call = self._calls.pop(call_id, None)
                        if call:
                            self._calls_by_session.pop(call.session_id, None)

                # Cleanup state machines
                await self._state_manager.cleanup_ended()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    # Statistics

    def get_stats(self) -> Dict[str, Any]:
        """Get call manager statistics."""
        return {
            "total_calls": self._total_calls,
            "active_calls": self._active_calls,
            "completed_calls": self._completed_calls,
            "failed_calls": self._failed_calls,
            "queues": self._queue_manager.get_stats(),
            "routing": self._router_manager.get_stats(),
            "transfers": self._transfer_manager.get_stats(),
            "state_machines": self._state_manager.get_stats(),
        }
