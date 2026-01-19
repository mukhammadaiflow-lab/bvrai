"""
Call State Machine

Manages call lifecycle with:
- State transitions
- Event handling
- Transition hooks
- State persistence
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Set, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging

logger = logging.getLogger(__name__)


class CallState(str, Enum):
    """Call states."""
    # Initial states
    INITIALIZING = "initializing"
    QUEUED = "queued"

    # Connection states
    DIALING = "dialing"
    RINGING = "ringing"
    CONNECTING = "connecting"

    # Active states
    CONNECTED = "connected"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    MUTED = "muted"

    # Transfer states
    TRANSFERRING = "transferring"
    TRANSFERRED = "transferred"

    # Recording states
    RECORDING = "recording"

    # End states
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    REJECTED = "rejected"

    @property
    def is_active(self) -> bool:
        """Check if call is in active state."""
        return self in {
            CallState.CONNECTED,
            CallState.IN_PROGRESS,
            CallState.ON_HOLD,
            CallState.MUTED,
            CallState.RECORDING,
        }

    @property
    def is_ended(self) -> bool:
        """Check if call has ended."""
        return self in {
            CallState.COMPLETED,
            CallState.FAILED,
            CallState.CANCELLED,
            CallState.NO_ANSWER,
            CallState.BUSY,
            CallState.REJECTED,
        }


class CallEvent(str, Enum):
    """Call events that trigger state transitions."""
    # Lifecycle events
    INITIATE = "initiate"
    QUEUE = "queue"
    DIAL = "dial"
    RING = "ring"
    ANSWER = "answer"
    CONNECT = "connect"

    # Active call events
    START_SPEECH = "start_speech"
    END_SPEECH = "end_speech"
    HOLD = "hold"
    RESUME = "resume"
    MUTE = "mute"
    UNMUTE = "unmute"

    # Transfer events
    TRANSFER_INITIATE = "transfer_initiate"
    TRANSFER_COMPLETE = "transfer_complete"
    TRANSFER_FAIL = "transfer_fail"

    # Recording events
    START_RECORDING = "start_recording"
    STOP_RECORDING = "stop_recording"

    # End events
    HANGUP = "hangup"
    COMPLETE = "complete"
    FAIL = "fail"
    CANCEL = "cancel"
    TIMEOUT = "timeout"
    REJECT = "reject"


@dataclass
class StateTransition:
    """Represents a valid state transition."""
    from_state: CallState
    event: CallEvent
    to_state: CallState
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    priority: int = 0

    def is_valid(self, context: Dict[str, Any]) -> bool:
        """Check if transition is valid given context."""
        if self.condition is None:
            return True
        return self.condition(context)


@dataclass
class TransitionResult:
    """Result of a state transition."""
    success: bool
    from_state: CallState
    to_state: CallState
    event: CallEvent
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TransitionHandler(ABC):
    """Abstract transition handler."""

    @abstractmethod
    async def on_enter(self, state: CallState, context: Dict[str, Any]) -> None:
        """Called when entering a state."""
        pass

    @abstractmethod
    async def on_exit(self, state: CallState, context: Dict[str, Any]) -> None:
        """Called when exiting a state."""
        pass

    @abstractmethod
    async def on_transition(
        self,
        from_state: CallState,
        to_state: CallState,
        event: CallEvent,
        context: Dict[str, Any],
    ) -> None:
        """Called during state transition."""
        pass


class DefaultTransitionHandler(TransitionHandler):
    """Default transition handler with logging."""

    async def on_enter(self, state: CallState, context: Dict[str, Any]) -> None:
        """Log state entry."""
        call_id = context.get("call_id", "unknown")
        logger.info(f"Call {call_id} entered state: {state.value}")

    async def on_exit(self, state: CallState, context: Dict[str, Any]) -> None:
        """Log state exit."""
        call_id = context.get("call_id", "unknown")
        logger.debug(f"Call {call_id} exiting state: {state.value}")

    async def on_transition(
        self,
        from_state: CallState,
        to_state: CallState,
        event: CallEvent,
        context: Dict[str, Any],
    ) -> None:
        """Log transition."""
        call_id = context.get("call_id", "unknown")
        logger.info(
            f"Call {call_id} transition: {from_state.value} -> {to_state.value} "
            f"(event: {event.value})"
        )


@dataclass
class StateMachineConfig:
    """State machine configuration."""
    allow_self_transitions: bool = False
    strict_transitions: bool = True
    max_history: int = 100
    timeout_seconds: float = 30.0


class CallStateMachine:
    """
    Call state machine.

    Manages call state transitions with validation and hooks.
    """

    # Default valid transitions
    DEFAULT_TRANSITIONS: List[StateTransition] = [
        # Initialization
        StateTransition(CallState.INITIALIZING, CallEvent.QUEUE, CallState.QUEUED),
        StateTransition(CallState.INITIALIZING, CallEvent.DIAL, CallState.DIALING),

        # Outbound flow
        StateTransition(CallState.QUEUED, CallEvent.DIAL, CallState.DIALING),
        StateTransition(CallState.DIALING, CallEvent.RING, CallState.RINGING),
        StateTransition(CallState.RINGING, CallEvent.ANSWER, CallState.CONNECTING),
        StateTransition(CallState.CONNECTING, CallEvent.CONNECT, CallState.CONNECTED),

        # Inbound flow
        StateTransition(CallState.QUEUED, CallEvent.RING, CallState.RINGING),
        StateTransition(CallState.RINGING, CallEvent.CONNECT, CallState.CONNECTED),

        # Active call
        StateTransition(CallState.CONNECTED, CallEvent.START_SPEECH, CallState.IN_PROGRESS),
        StateTransition(CallState.IN_PROGRESS, CallEvent.HOLD, CallState.ON_HOLD),
        StateTransition(CallState.ON_HOLD, CallEvent.RESUME, CallState.IN_PROGRESS),
        StateTransition(CallState.IN_PROGRESS, CallEvent.MUTE, CallState.MUTED),
        StateTransition(CallState.MUTED, CallEvent.UNMUTE, CallState.IN_PROGRESS),

        # Recording
        StateTransition(CallState.IN_PROGRESS, CallEvent.START_RECORDING, CallState.RECORDING),
        StateTransition(CallState.RECORDING, CallEvent.STOP_RECORDING, CallState.IN_PROGRESS),

        # Transfer
        StateTransition(CallState.IN_PROGRESS, CallEvent.TRANSFER_INITIATE, CallState.TRANSFERRING),
        StateTransition(CallState.TRANSFERRING, CallEvent.TRANSFER_COMPLETE, CallState.TRANSFERRED),
        StateTransition(CallState.TRANSFERRING, CallEvent.TRANSFER_FAIL, CallState.IN_PROGRESS),

        # End states from connected
        StateTransition(CallState.CONNECTED, CallEvent.HANGUP, CallState.COMPLETED),
        StateTransition(CallState.CONNECTED, CallEvent.FAIL, CallState.FAILED),

        # End states from in_progress
        StateTransition(CallState.IN_PROGRESS, CallEvent.HANGUP, CallState.COMPLETED),
        StateTransition(CallState.IN_PROGRESS, CallEvent.COMPLETE, CallState.COMPLETED),
        StateTransition(CallState.IN_PROGRESS, CallEvent.FAIL, CallState.FAILED),

        # End states from on_hold
        StateTransition(CallState.ON_HOLD, CallEvent.HANGUP, CallState.COMPLETED),

        # End states from ringing
        StateTransition(CallState.RINGING, CallEvent.TIMEOUT, CallState.NO_ANSWER),
        StateTransition(CallState.RINGING, CallEvent.REJECT, CallState.REJECTED),
        StateTransition(CallState.RINGING, CallEvent.CANCEL, CallState.CANCELLED),

        # End states from dialing
        StateTransition(CallState.DIALING, CallEvent.TIMEOUT, CallState.NO_ANSWER),
        StateTransition(CallState.DIALING, CallEvent.FAIL, CallState.FAILED),
        StateTransition(CallState.DIALING, CallEvent.CANCEL, CallState.CANCELLED),

        # End states from queued
        StateTransition(CallState.QUEUED, CallEvent.CANCEL, CallState.CANCELLED),
        StateTransition(CallState.QUEUED, CallEvent.TIMEOUT, CallState.FAILED),

        # End states from transferring
        StateTransition(CallState.TRANSFERRING, CallEvent.HANGUP, CallState.COMPLETED),
    ]

    def __init__(
        self,
        call_id: str,
        initial_state: CallState = CallState.INITIALIZING,
        config: Optional[StateMachineConfig] = None,
        handler: Optional[TransitionHandler] = None,
    ):
        self.call_id = call_id
        self._state = initial_state
        self.config = config or StateMachineConfig()
        self.handler = handler or DefaultTransitionHandler()

        # Transitions
        self._transitions: List[StateTransition] = list(self.DEFAULT_TRANSITIONS)

        # History
        self._history: List[TransitionResult] = []

        # Context
        self._context: Dict[str, Any] = {"call_id": call_id}

        # Callbacks
        self._on_state_change: List[Callable[[CallState, CallState], Awaitable[None]]] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CallState:
        """Get current state."""
        return self._state

    @property
    def context(self) -> Dict[str, Any]:
        """Get state context."""
        return self._context.copy()

    @property
    def history(self) -> List[TransitionResult]:
        """Get transition history."""
        return list(self._history)

    def add_transition(self, transition: StateTransition) -> None:
        """Add custom transition."""
        self._transitions.append(transition)

    def remove_transition(self, from_state: CallState, event: CallEvent) -> bool:
        """Remove transition."""
        before = len(self._transitions)
        self._transitions = [
            t for t in self._transitions
            if not (t.from_state == from_state and t.event == event)
        ]
        return len(self._transitions) < before

    def set_context(self, key: str, value: Any) -> None:
        """Set context value."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context value."""
        return self._context.get(key, default)

    def get_valid_events(self) -> List[CallEvent]:
        """Get list of valid events from current state."""
        valid = set()
        for transition in self._transitions:
            if transition.from_state == self._state:
                if transition.is_valid(self._context):
                    valid.add(transition.event)
        return list(valid)

    def can_transition(self, event: CallEvent) -> bool:
        """Check if event can trigger transition."""
        for transition in self._transitions:
            if (
                transition.from_state == self._state
                and transition.event == event
                and transition.is_valid(self._context)
            ):
                return True
        return False

    async def trigger(
        self,
        event: CallEvent,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TransitionResult:
        """Trigger state transition."""
        async with self._lock:
            return await self._do_transition(event, metadata or {})

    async def _do_transition(
        self,
        event: CallEvent,
        metadata: Dict[str, Any],
    ) -> TransitionResult:
        """Internal transition logic."""
        # Find valid transition
        transition = self._find_transition(event)

        if not transition:
            if self.config.strict_transitions:
                error = f"No valid transition for event {event.value} from state {self._state.value}"
                logger.warning(error)
                return TransitionResult(
                    success=False,
                    from_state=self._state,
                    to_state=self._state,
                    event=event,
                    error=error,
                    metadata=metadata,
                )
            else:
                # Non-strict mode: stay in current state
                return TransitionResult(
                    success=True,
                    from_state=self._state,
                    to_state=self._state,
                    event=event,
                    metadata=metadata,
                )

        from_state = self._state
        to_state = transition.to_state

        # Check for self-transition
        if from_state == to_state and not self.config.allow_self_transitions:
            return TransitionResult(
                success=True,
                from_state=from_state,
                to_state=to_state,
                event=event,
                metadata=metadata,
            )

        try:
            # Exit current state
            await self.handler.on_exit(from_state, self._context)

            # Perform transition
            await self.handler.on_transition(from_state, to_state, event, self._context)

            # Update state
            self._state = to_state
            self._context["last_event"] = event.value
            self._context["last_transition"] = datetime.utcnow().isoformat()

            # Enter new state
            await self.handler.on_enter(to_state, self._context)

            # Create result
            result = TransitionResult(
                success=True,
                from_state=from_state,
                to_state=to_state,
                event=event,
                metadata=metadata,
            )

            # Add to history
            self._add_to_history(result)

            # Trigger callbacks
            for callback in self._on_state_change:
                try:
                    await callback(from_state, to_state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")

            return result

        except Exception as e:
            logger.error(f"Transition error: {e}")
            return TransitionResult(
                success=False,
                from_state=from_state,
                to_state=from_state,
                event=event,
                error=str(e),
                metadata=metadata,
            )

    def _find_transition(self, event: CallEvent) -> Optional[StateTransition]:
        """Find matching transition."""
        candidates = []

        for transition in self._transitions:
            if (
                transition.from_state == self._state
                and transition.event == event
                and transition.is_valid(self._context)
            ):
                candidates.append(transition)

        if not candidates:
            return None

        # Return highest priority
        return max(candidates, key=lambda t: t.priority)

    def _add_to_history(self, result: TransitionResult) -> None:
        """Add result to history."""
        self._history.append(result)

        # Trim history
        if len(self._history) > self.config.max_history:
            self._history = self._history[-self.config.max_history:]

    def on_state_change(
        self,
        callback: Callable[[CallState, CallState], Awaitable[None]],
    ) -> None:
        """Register state change callback."""
        self._on_state_change.append(callback)

    def get_duration_in_state(self) -> float:
        """Get duration in current state in seconds."""
        if not self._history:
            return 0.0

        last = self._history[-1]
        return (datetime.utcnow() - last.timestamp).total_seconds()

    def get_stats(self) -> Dict[str, Any]:
        """Get state machine statistics."""
        state_durations: Dict[str, float] = {}

        for i, result in enumerate(self._history):
            if i > 0:
                prev = self._history[i - 1]
                duration = (result.timestamp - prev.timestamp).total_seconds()
                state = prev.to_state.value
                state_durations[state] = state_durations.get(state, 0) + duration

        return {
            "call_id": self.call_id,
            "current_state": self._state.value,
            "transition_count": len(self._history),
            "state_durations": state_durations,
            "duration_in_current_state": self.get_duration_in_state(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state machine."""
        return {
            "call_id": self.call_id,
            "state": self._state.value,
            "context": self._context,
            "history": [
                {
                    "from_state": r.from_state.value,
                    "to_state": r.to_state.value,
                    "event": r.event.value,
                    "timestamp": r.timestamp.isoformat(),
                    "success": r.success,
                }
                for r in self._history
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        config: Optional[StateMachineConfig] = None,
        handler: Optional[TransitionHandler] = None,
    ) -> "CallStateMachine":
        """Deserialize state machine."""
        machine = cls(
            call_id=data["call_id"],
            initial_state=CallState(data["state"]),
            config=config,
            handler=handler,
        )
        machine._context = data.get("context", {"call_id": data["call_id"]})

        # Restore history
        for h in data.get("history", []):
            result = TransitionResult(
                success=h.get("success", True),
                from_state=CallState(h["from_state"]),
                to_state=CallState(h["to_state"]),
                event=CallEvent(h["event"]),
                timestamp=datetime.fromisoformat(h["timestamp"]),
            )
            machine._history.append(result)

        return machine


class StateMachineManager:
    """
    Manages multiple call state machines.
    """

    def __init__(
        self,
        config: Optional[StateMachineConfig] = None,
        handler: Optional[TransitionHandler] = None,
    ):
        self.config = config or StateMachineConfig()
        self.handler = handler or DefaultTransitionHandler()

        self._machines: Dict[str, CallStateMachine] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        call_id: str,
        initial_state: CallState = CallState.INITIALIZING,
    ) -> CallStateMachine:
        """Create new state machine."""
        async with self._lock:
            if call_id in self._machines:
                return self._machines[call_id]

            machine = CallStateMachine(
                call_id=call_id,
                initial_state=initial_state,
                config=self.config,
                handler=self.handler,
            )

            self._machines[call_id] = machine
            return machine

    async def get(self, call_id: str) -> Optional[CallStateMachine]:
        """Get state machine by call ID."""
        return self._machines.get(call_id)

    async def remove(self, call_id: str) -> bool:
        """Remove state machine."""
        async with self._lock:
            return self._machines.pop(call_id, None) is not None

    async def trigger(
        self,
        call_id: str,
        event: CallEvent,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TransitionResult]:
        """Trigger event on state machine."""
        machine = await self.get(call_id)
        if machine:
            return await machine.trigger(event, metadata)
        return None

    async def cleanup_ended(self) -> int:
        """Remove ended call state machines."""
        async with self._lock:
            ended = [
                call_id for call_id, machine in self._machines.items()
                if machine.state.is_ended
            ]

            for call_id in ended:
                del self._machines[call_id]

            return len(ended)

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        states: Dict[str, int] = {}
        for machine in self._machines.values():
            state = machine.state.value
            states[state] = states.get(state, 0) + 1

        return {
            "total_machines": len(self._machines),
            "states": states,
        }
