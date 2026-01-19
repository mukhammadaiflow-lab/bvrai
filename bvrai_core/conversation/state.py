"""
Conversation State Machine

This module provides state machine functionality for managing
conversation states and transitions.
"""

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .base import (
    ConversationState,
    ConversationEvent,
    ConversationEventType,
)
from .context import ConversationContext


logger = logging.getLogger(__name__)


@dataclass
class StateTransition:
    """Definition of a state transition."""

    from_state: ConversationState
    to_state: ConversationState

    # Conditions
    on_event: Optional[ConversationEventType] = None
    condition: Optional[Callable[[ConversationContext], bool]] = None

    # Actions
    actions: List[Callable[[ConversationContext], None]] = field(default_factory=list)
    before_transition: Optional[Callable[[ConversationContext], bool]] = None
    after_transition: Optional[Callable[[ConversationContext], None]] = None

    # Priority
    priority: int = 0

    def can_transition(
        self,
        event: Optional[ConversationEvent],
        context: ConversationContext,
    ) -> bool:
        """Check if transition is valid."""
        # Check event match
        if self.on_event and event:
            if event.type != self.on_event:
                return False

        # Check custom condition
        if self.condition:
            if not self.condition(context):
                return False

        # Check before_transition guard
        if self.before_transition:
            if not self.before_transition(context):
                return False

        return True


@dataclass
class StateHandler:
    """Handler for a specific state."""

    state: ConversationState

    # Entry/exit hooks
    on_enter: Optional[Callable[[ConversationContext], None]] = None
    on_exit: Optional[Callable[[ConversationContext], None]] = None

    # Event handler
    on_event: Optional[Callable[[ConversationEvent, ConversationContext], Optional[ConversationState]]] = None

    # Timeout handling
    timeout_seconds: Optional[int] = None
    on_timeout: Optional[Callable[[ConversationContext], Optional[ConversationState]]] = None


class StateMachine:
    """
    State machine for conversation flow management.
    """

    def __init__(
        self,
        initial_state: ConversationState = ConversationState.INITIALIZED,
    ):
        self._current_state = initial_state
        self._initial_state = initial_state

        # State handlers
        self._handlers: Dict[ConversationState, StateHandler] = {}

        # Transitions
        self._transitions: List[StateTransition] = []

        # Global transitions (apply from any state)
        self._global_transitions: List[StateTransition] = []

        # Valid state transitions
        self._valid_transitions: Dict[ConversationState, Set[ConversationState]] = {}

        # Event listeners
        self._listeners: List[Callable[[ConversationEvent], None]] = []

        # History
        self._history: List[Tuple[ConversationState, float]] = []
        self._max_history = 100

    @property
    def current_state(self) -> ConversationState:
        """Get current state."""
        return self._current_state

    @property
    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        return self._current_state in (
            ConversationState.COMPLETED,
            ConversationState.TRANSFERRED,
            ConversationState.ABANDONED,
            ConversationState.ERROR,
        )

    def register_handler(self, handler: StateHandler) -> None:
        """Register a state handler."""
        self._handlers[handler.state] = handler

    def add_transition(
        self,
        from_state: ConversationState,
        to_state: ConversationState,
        on_event: Optional[ConversationEventType] = None,
        condition: Optional[Callable[[ConversationContext], bool]] = None,
        actions: Optional[List[Callable]] = None,
        is_global: bool = False,
    ) -> None:
        """Add a state transition."""
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            on_event=on_event,
            condition=condition,
            actions=actions or [],
        )

        if is_global:
            self._global_transitions.append(transition)
        else:
            self._transitions.append(transition)

        # Track valid transitions
        if from_state not in self._valid_transitions:
            self._valid_transitions[from_state] = set()
        self._valid_transitions[from_state].add(to_state)

    def add_listener(
        self,
        listener: Callable[[ConversationEvent], None],
    ) -> None:
        """Add an event listener."""
        self._listeners.append(listener)

    def can_transition_to(self, target_state: ConversationState) -> bool:
        """Check if transition to target state is valid."""
        valid = self._valid_transitions.get(self._current_state, set())
        return target_state in valid

    async def transition(
        self,
        to_state: ConversationState,
        context: ConversationContext,
        event: Optional[ConversationEvent] = None,
    ) -> bool:
        """
        Attempt to transition to a new state.

        Returns True if transition was successful.
        """
        from_state = self._current_state

        # Check if transition is valid
        if not self.can_transition_to(to_state):
            logger.warning(
                f"Invalid transition: {from_state.value} -> {to_state.value}"
            )
            return False

        # Find matching transition
        transition = self._find_transition(from_state, to_state, event, context)

        if not transition:
            # Allow direct transition if in valid set
            transition = StateTransition(
                from_state=from_state,
                to_state=to_state,
            )

        # Exit current state
        current_handler = self._handlers.get(from_state)
        if current_handler and current_handler.on_exit:
            try:
                result = current_handler.on_exit(context)
                if hasattr(result, '__await__'):
                    await result
            except Exception as e:
                logger.error(f"Error in state exit handler: {e}")

        # Execute transition actions
        for action in transition.actions:
            try:
                result = action(context)
                if hasattr(result, '__await__'):
                    await result
            except Exception as e:
                logger.error(f"Error in transition action: {e}")

        # Update state
        self._current_state = to_state

        # Record history
        import time
        self._history.append((to_state, time.time()))
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Enter new state
        new_handler = self._handlers.get(to_state)
        if new_handler and new_handler.on_enter:
            try:
                result = new_handler.on_enter(context)
                if hasattr(result, '__await__'):
                    await result
            except Exception as e:
                logger.error(f"Error in state entry handler: {e}")

        # Execute after_transition
        if transition.after_transition:
            try:
                result = transition.after_transition(context)
                if hasattr(result, '__await__'):
                    await result
            except Exception as e:
                logger.error(f"Error in after_transition: {e}")

        # Emit state change event
        state_event = ConversationEvent(
            type=ConversationEventType.STATE_CHANGED,
            data={
                "from_state": from_state.value,
                "to_state": to_state.value,
            },
            session_id=context.session_id,
        )
        self._emit_event(state_event)

        logger.debug(f"State transition: {from_state.value} -> {to_state.value}")
        return True

    async def process_event(
        self,
        event: ConversationEvent,
        context: ConversationContext,
    ) -> Optional[ConversationState]:
        """
        Process an event and potentially trigger state transition.

        Returns the new state if a transition occurred, None otherwise.
        """
        # Emit event to listeners
        self._emit_event(event)

        # Let current state handler process event
        handler = self._handlers.get(self._current_state)
        if handler and handler.on_event:
            try:
                result = handler.on_event(event, context)
                if hasattr(result, '__await__'):
                    result = await result

                if result and isinstance(result, ConversationState):
                    if await self.transition(result, context, event):
                        return result
            except Exception as e:
                logger.error(f"Error in state event handler: {e}")

        # Check for automatic transitions
        for transition in self._transitions + self._global_transitions:
            if transition.from_state != self._current_state:
                if transition not in self._global_transitions:
                    continue

            if transition.can_transition(event, context):
                if await self.transition(transition.to_state, context, event):
                    return transition.to_state

        return None

    def _find_transition(
        self,
        from_state: ConversationState,
        to_state: ConversationState,
        event: Optional[ConversationEvent],
        context: ConversationContext,
    ) -> Optional[StateTransition]:
        """Find a matching transition."""
        candidates = []

        for transition in self._transitions + self._global_transitions:
            if transition.from_state == from_state and transition.to_state == to_state:
                if transition.can_transition(event, context):
                    candidates.append(transition)

        if not candidates:
            return None

        # Return highest priority
        candidates.sort(key=lambda t: t.priority, reverse=True)
        return candidates[0]

    def _emit_event(self, event: ConversationEvent) -> None:
        """Emit event to listeners."""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")

    def reset(self) -> None:
        """Reset to initial state."""
        self._current_state = self._initial_state
        self._history.clear()

    def get_history(self) -> List[Tuple[ConversationState, float]]:
        """Get state history."""
        return self._history.copy()


def create_default_state_machine() -> StateMachine:
    """Create a state machine with default conversation states and transitions."""
    sm = StateMachine(initial_state=ConversationState.INITIALIZED)

    # Define valid transitions
    transitions = [
        # From INITIALIZED
        (ConversationState.INITIALIZED, ConversationState.GREETING),
        (ConversationState.INITIALIZED, ConversationState.LISTENING),

        # From GREETING
        (ConversationState.GREETING, ConversationState.LISTENING),
        (ConversationState.GREETING, ConversationState.COMPLETED),

        # From LISTENING
        (ConversationState.LISTENING, ConversationState.PROCESSING),
        (ConversationState.LISTENING, ConversationState.COLLECTING_SLOT),
        (ConversationState.LISTENING, ConversationState.COMPLETED),
        (ConversationState.LISTENING, ConversationState.ABANDONED),

        # From PROCESSING
        (ConversationState.PROCESSING, ConversationState.RESPONDING),
        (ConversationState.PROCESSING, ConversationState.EXECUTING_ACTION),
        (ConversationState.PROCESSING, ConversationState.BRANCHING),
        (ConversationState.PROCESSING, ConversationState.ERROR),

        # From RESPONDING
        (ConversationState.RESPONDING, ConversationState.LISTENING),
        (ConversationState.RESPONDING, ConversationState.WAITING_INPUT),
        (ConversationState.RESPONDING, ConversationState.COMPLETED),
        (ConversationState.RESPONDING, ConversationState.TRANSFERRING),

        # From WAITING_INPUT
        (ConversationState.WAITING_INPUT, ConversationState.PROCESSING),
        (ConversationState.WAITING_INPUT, ConversationState.LISTENING),
        (ConversationState.WAITING_INPUT, ConversationState.ABANDONED),

        # From COLLECTING_SLOT
        (ConversationState.COLLECTING_SLOT, ConversationState.VALIDATING_SLOT),
        (ConversationState.COLLECTING_SLOT, ConversationState.LISTENING),
        (ConversationState.COLLECTING_SLOT, ConversationState.ABANDONED),

        # From VALIDATING_SLOT
        (ConversationState.VALIDATING_SLOT, ConversationState.COLLECTING_SLOT),
        (ConversationState.VALIDATING_SLOT, ConversationState.PROCESSING),
        (ConversationState.VALIDATING_SLOT, ConversationState.RESPONDING),

        # From EXECUTING_ACTION
        (ConversationState.EXECUTING_ACTION, ConversationState.RESPONDING),
        (ConversationState.EXECUTING_ACTION, ConversationState.PROCESSING),
        (ConversationState.EXECUTING_ACTION, ConversationState.ERROR),

        # From BRANCHING
        (ConversationState.BRANCHING, ConversationState.PROCESSING),
        (ConversationState.BRANCHING, ConversationState.RESPONDING),
        (ConversationState.BRANCHING, ConversationState.TRANSFERRING),

        # From TRANSFERRING
        (ConversationState.TRANSFERRING, ConversationState.ON_HOLD),
        (ConversationState.TRANSFERRING, ConversationState.TRANSFERRED),
        (ConversationState.TRANSFERRING, ConversationState.ERROR),

        # From ON_HOLD
        (ConversationState.ON_HOLD, ConversationState.LISTENING),
        (ConversationState.ON_HOLD, ConversationState.TRANSFERRED),
        (ConversationState.ON_HOLD, ConversationState.ABANDONED),

        # From ERROR
        (ConversationState.ERROR, ConversationState.LISTENING),
        (ConversationState.ERROR, ConversationState.COMPLETED),
        (ConversationState.ERROR, ConversationState.TRANSFERRING),
    ]

    for from_state, to_state in transitions:
        sm.add_transition(from_state, to_state)

    return sm


__all__ = [
    "StateTransition",
    "StateHandler",
    "StateMachine",
    "create_default_state_machine",
]
