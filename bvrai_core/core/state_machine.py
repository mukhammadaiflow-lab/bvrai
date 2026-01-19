"""
Advanced State Machine Engine
=============================

Enterprise-grade hierarchical state machine for managing complex
state transitions in conversations, calls, and business processes.

Features:
- Hierarchical states (nested state machines)
- Guard conditions
- Entry/exit actions
- Transition actions
- Event-driven transitions
- History states
- Parallel states
- State persistence

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

T = TypeVar("T")
StateT = TypeVar("StateT", bound="State")


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

Guard = Callable[["StateContext"], bool]
Action = Callable[["StateContext"], Awaitable[None]]
AsyncGuard = Callable[["StateContext"], Awaitable[bool]]


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class StateContext:
    """
    Context passed through state transitions.

    Contains all data needed for guards and actions to execute.
    """

    machine_id: str = field(default_factory=lambda: str(uuid4()))
    current_state: str = ""
    previous_state: Optional[str] = None
    event: Optional[str] = None
    event_data: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def get(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.variables[key] = value
        self.updated_at = datetime.utcnow()

    def get_event_data(self, key: str, default: Any = None) -> Any:
        return self.event_data.get(key, default)

    def record_transition(
        self,
        from_state: str,
        to_state: str,
        event: str
    ) -> None:
        self.history.append({
            "from": from_state,
            "to": to_state,
            "event": event,
            "timestamp": datetime.utcnow().isoformat()
        })

    def clone(self) -> "StateContext":
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "current_state": self.current_state,
            "previous_state": self.previous_state,
            "event": self.event,
            "event_data": self.event_data,
            "variables": self.variables,
            "history": self.history,
            "metadata": self.metadata,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class TransitionResult:
    """Result of a state transition"""

    success: bool
    from_state: str
    to_state: Optional[str] = None
    event: str = ""
    error: Optional[str] = None
    duration_ms: float = 0.0


# =============================================================================
# STATE DEFINITION
# =============================================================================


class State:
    """
    Represents a state in the state machine.

    States can have:
    - Entry actions (executed when entering the state)
    - Exit actions (executed when leaving the state)
    - Internal transitions (no state change)
    - Child states (for hierarchical machines)
    """

    def __init__(
        self,
        name: str,
        is_initial: bool = False,
        is_final: bool = False,
        is_history: bool = False,
        parent: Optional["State"] = None,
        entry_actions: Optional[List[Action]] = None,
        exit_actions: Optional[List[Action]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.is_initial = is_initial
        self.is_final = is_final
        self.is_history = is_history
        self.parent = parent
        self.entry_actions = entry_actions or []
        self.exit_actions = exit_actions or []
        self.metadata = metadata or {}

        # Child states for hierarchical machines
        self.children: Dict[str, "State"] = {}
        self.initial_child: Optional[str] = None

        # Transitions from this state
        self.transitions: List["Transition"] = []

        self._logger = structlog.get_logger(f"state.{name}")

    @property
    def full_name(self) -> str:
        """Get fully qualified state name"""
        if self.parent:
            return f"{self.parent.full_name}.{self.name}"
        return self.name

    @property
    def is_composite(self) -> bool:
        """Check if this is a composite state"""
        return len(self.children) > 0

    def add_child(self, state: "State") -> "State":
        """Add a child state"""
        state.parent = self
        self.children[state.name] = state
        if state.is_initial:
            self.initial_child = state.name
        return self

    def add_transition(self, transition: "Transition") -> "State":
        """Add a transition from this state"""
        self.transitions.append(transition)
        return self

    def on_entry(self, action: Action) -> "State":
        """Add an entry action"""
        self.entry_actions.append(action)
        return self

    def on_exit(self, action: Action) -> "State":
        """Add an exit action"""
        self.exit_actions.append(action)
        return self

    async def enter(self, context: StateContext) -> None:
        """Execute entry actions"""
        self._logger.debug("entering_state", state=self.name)
        for action in self.entry_actions:
            await action(context)

    async def exit(self, context: StateContext) -> None:
        """Execute exit actions"""
        self._logger.debug("exiting_state", state=self.name)
        for action in self.exit_actions:
            await action(context)

    def get_transition(
        self,
        event: str,
        context: StateContext
    ) -> Optional["Transition"]:
        """Find a valid transition for an event"""
        for transition in self.transitions:
            if transition.event == event and transition.is_valid(context):
                return transition
        return None


# =============================================================================
# TRANSITION DEFINITION
# =============================================================================


class Transition:
    """
    Represents a transition between states.

    Transitions can have:
    - Guards (conditions that must be true)
    - Actions (executed during transition)
    - Target state
    """

    def __init__(
        self,
        event: str,
        target: Optional[str] = None,
        guards: Optional[List[Guard]] = None,
        actions: Optional[List[Action]] = None,
        internal: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.event = event
        self.target = target
        self.guards = guards or []
        self.actions = actions or []
        self.internal = internal  # Internal transitions don't exit/enter states
        self.metadata = metadata or {}

        self._logger = structlog.get_logger("transition")

    def is_valid(self, context: StateContext) -> bool:
        """Check if all guards pass"""
        for guard in self.guards:
            if not guard(context):
                return False
        return True

    async def execute(self, context: StateContext) -> None:
        """Execute transition actions"""
        for action in self.actions:
            await action(context)


# =============================================================================
# STATE MACHINE
# =============================================================================


class StateMachine:
    """
    Hierarchical State Machine.

    Manages states, transitions, and state persistence.

    Usage:
        machine = StateMachine("conversation")

        # Define states
        machine.add_state(State("idle", is_initial=True))
        machine.add_state(State("active"))
        machine.add_state(State("ended", is_final=True))

        # Define transitions
        machine.add_transition("idle", "active", "start_call")
        machine.add_transition("active", "ended", "end_call")

        # Start and process events
        await machine.start()
        await machine.send("start_call")
        await machine.send("end_call")
    """

    def __init__(
        self,
        name: str,
        context: Optional[StateContext] = None
    ):
        self.name = name
        self._states: Dict[str, State] = {}
        self._initial_state: Optional[str] = None
        self._current_state: Optional[State] = None
        self._context = context or StateContext()
        self._context.machine_id = f"{name}-{uuid4().hex[:8]}"
        self._running = False
        self._history: Dict[str, str] = {}  # For history states
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger(f"state_machine.{name}")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def current_state(self) -> Optional[str]:
        return self._current_state.name if self._current_state else None

    @property
    def context(self) -> StateContext:
        return self._context

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_final(self) -> bool:
        return self._current_state is not None and self._current_state.is_final

    # -------------------------------------------------------------------------
    # State Definition
    # -------------------------------------------------------------------------

    def add_state(self, state: State) -> "StateMachine":
        """Add a state to the machine"""
        self._states[state.name] = state
        if state.is_initial:
            self._initial_state = state.name
        return self

    def get_state(self, name: str) -> Optional[State]:
        """Get a state by name"""
        if "." in name:
            parts = name.split(".")
            state = self._states.get(parts[0])
            for part in parts[1:]:
                if state and part in state.children:
                    state = state.children[part]
                else:
                    return None
            return state
        return self._states.get(name)

    def add_transition(
        self,
        from_state: str,
        to_state: Optional[str],
        event: str,
        guards: Optional[List[Guard]] = None,
        actions: Optional[List[Action]] = None,
        internal: bool = False
    ) -> "StateMachine":
        """Add a transition between states"""
        state = self.get_state(from_state)
        if state:
            transition = Transition(
                event=event,
                target=to_state,
                guards=guards,
                actions=actions,
                internal=internal
            )
            state.add_transition(transition)
        return self

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def on_transition(
        self,
        handler: Callable[[str, str, str], Awaitable[None]]
    ) -> "StateMachine":
        """Register a transition handler"""
        self._event_handlers["transition"].append(handler)
        return self

    def on_state_enter(
        self,
        state_name: str,
        handler: Callable[[StateContext], Awaitable[None]]
    ) -> "StateMachine":
        """Register a state entry handler"""
        self._event_handlers[f"enter:{state_name}"].append(handler)
        return self

    def on_state_exit(
        self,
        state_name: str,
        handler: Callable[[StateContext], Awaitable[None]]
    ) -> "StateMachine":
        """Register a state exit handler"""
        self._event_handlers[f"exit:{state_name}"].append(handler)
        return self

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self, initial_data: Optional[Dict[str, Any]] = None) -> None:
        """Start the state machine"""
        if self._running:
            return

        if not self._initial_state:
            raise ValueError("No initial state defined")

        self._running = True
        self._context.variables.update(initial_data or {})

        # Enter initial state
        await self._enter_state(self._initial_state)

        self._logger.info(
            "state_machine_started",
            initial_state=self._initial_state
        )

    async def stop(self) -> None:
        """Stop the state machine"""
        if not self._running:
            return

        # Exit current state
        if self._current_state:
            await self._exit_state()

        self._running = False
        self._logger.info("state_machine_stopped")

    async def reset(self) -> None:
        """Reset the state machine"""
        await self.stop()
        self._context = StateContext()
        self._history.clear()
        await self.start()

    # -------------------------------------------------------------------------
    # Event Processing
    # -------------------------------------------------------------------------

    async def send(
        self,
        event: str,
        data: Optional[Dict[str, Any]] = None
    ) -> TransitionResult:
        """
        Send an event to the state machine.

        Args:
            event: Event name
            data: Event data

        Returns:
            TransitionResult indicating success/failure
        """
        async with self._lock:
            if not self._running:
                return TransitionResult(
                    success=False,
                    from_state="",
                    error="State machine is not running"
                )

            if not self._current_state:
                return TransitionResult(
                    success=False,
                    from_state="",
                    error="No current state"
                )

            import time
            start_time = time.time()

            from_state = self._current_state.name
            self._context.event = event
            self._context.event_data = data or {}

            # Find valid transition
            transition = self._find_transition(event)

            if not transition:
                return TransitionResult(
                    success=False,
                    from_state=from_state,
                    event=event,
                    error=f"No valid transition for event '{event}'"
                )

            try:
                # Execute transition
                if transition.internal:
                    # Internal transition - no state change
                    await transition.execute(self._context)
                    to_state = from_state
                else:
                    # Regular transition
                    to_state = transition.target

                    if to_state:
                        # Exit current state
                        await self._exit_state()

                        # Execute transition actions
                        await transition.execute(self._context)

                        # Enter new state
                        await self._enter_state(to_state)

                # Record transition
                self._context.record_transition(from_state, to_state or from_state, event)

                # Notify handlers
                await self._notify_transition(from_state, to_state or from_state, event)

                duration_ms = (time.time() - start_time) * 1000

                self._logger.info(
                    "transition_completed",
                    from_state=from_state,
                    to_state=to_state,
                    event=event,
                    duration_ms=duration_ms
                )

                return TransitionResult(
                    success=True,
                    from_state=from_state,
                    to_state=to_state,
                    event=event,
                    duration_ms=duration_ms
                )

            except Exception as e:
                self._logger.error(
                    "transition_failed",
                    from_state=from_state,
                    event=event,
                    error=str(e)
                )

                return TransitionResult(
                    success=False,
                    from_state=from_state,
                    event=event,
                    error=str(e)
                )

    def _find_transition(self, event: str) -> Optional[Transition]:
        """Find a valid transition for an event"""
        if not self._current_state:
            return None

        # Check current state first
        transition = self._current_state.get_transition(event, self._context)
        if transition:
            return transition

        # Check parent states (for hierarchical machines)
        parent = self._current_state.parent
        while parent:
            transition = parent.get_transition(event, self._context)
            if transition:
                return transition
            parent = parent.parent

        return None

    async def _enter_state(self, state_name: str) -> None:
        """Enter a state"""
        state = self.get_state(state_name)
        if not state:
            raise ValueError(f"State '{state_name}' not found")

        self._current_state = state
        self._context.previous_state = self._context.current_state
        self._context.current_state = state_name

        # Execute entry actions
        await state.enter(self._context)

        # Notify handlers
        for handler in self._event_handlers.get(f"enter:{state_name}", []):
            await handler(self._context)

        # Enter initial child for composite states
        if state.is_composite and state.initial_child:
            await self._enter_state(f"{state_name}.{state.initial_child}")

    async def _exit_state(self) -> None:
        """Exit the current state"""
        if not self._current_state:
            return

        state_name = self._current_state.name

        # Store history for history states
        if self._current_state.parent:
            self._history[self._current_state.parent.name] = state_name

        # Execute exit actions
        await self._current_state.exit(self._context)

        # Notify handlers
        for handler in self._event_handlers.get(f"exit:{state_name}", []):
            await handler(self._context)

    async def _notify_transition(
        self,
        from_state: str,
        to_state: str,
        event: str
    ) -> None:
        """Notify transition handlers"""
        for handler in self._event_handlers.get("transition", []):
            await handler(from_state, to_state, event)

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def can_send(self, event: str) -> bool:
        """Check if an event can trigger a transition"""
        return self._find_transition(event) is not None

    def get_available_events(self) -> List[str]:
        """Get list of events that can trigger transitions"""
        if not self._current_state:
            return []

        events = []
        for transition in self._current_state.transitions:
            if transition.is_valid(self._context):
                events.append(transition.event)

        return events

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_snapshot(self) -> Dict[str, Any]:
        """Get machine state snapshot for persistence"""
        return {
            "name": self.name,
            "current_state": self.current_state,
            "context": self._context.to_dict(),
            "history": self._history
        }

    async def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore machine state from snapshot"""
        self._context = StateContext(**{
            k: v for k, v in snapshot.get("context", {}).items()
            if k not in ("started_at", "updated_at")
        })
        self._history = snapshot.get("history", {})

        state_name = snapshot.get("current_state")
        if state_name:
            self._current_state = self.get_state(state_name)
            self._running = True


# =============================================================================
# STATE MACHINE BUILDER
# =============================================================================


class StateMachineBuilder:
    """
    Fluent builder for creating state machines.

    Usage:
        machine = (StateMachineBuilder("call")
            .state("idle", initial=True)
            .state("ringing")
            .state("connected")
            .state("ended", final=True)
            .transition("idle", "ringing", "incoming_call")
            .transition("ringing", "connected", "answer")
            .transition("connected", "ended", "hangup")
            .build())
    """

    def __init__(self, name: str):
        self._name = name
        self._states: Dict[str, State] = {}
        self._transitions: List[Tuple[str, Optional[str], str, List[Guard], List[Action]]] = []
        self._initial: Optional[str] = None

    def state(
        self,
        name: str,
        initial: bool = False,
        final: bool = False,
        entry: Optional[Action] = None,
        exit: Optional[Action] = None
    ) -> "StateMachineBuilder":
        """Add a state"""
        state = State(name, is_initial=initial, is_final=final)
        if entry:
            state.on_entry(entry)
        if exit:
            state.on_exit(exit)
        self._states[name] = state
        if initial:
            self._initial = name
        return self

    def transition(
        self,
        from_state: str,
        to_state: Optional[str],
        event: str,
        guard: Optional[Guard] = None,
        action: Optional[Action] = None
    ) -> "StateMachineBuilder":
        """Add a transition"""
        guards = [guard] if guard else []
        actions = [action] if action else []
        self._transitions.append((from_state, to_state, event, guards, actions))
        return self

    def internal(
        self,
        state: str,
        event: str,
        action: Action
    ) -> "StateMachineBuilder":
        """Add an internal transition"""
        self._transitions.append((state, None, event, [], [action]))
        return self

    def build(self) -> StateMachine:
        """Build the state machine"""
        machine = StateMachine(self._name)

        for state in self._states.values():
            machine.add_state(state)

        for from_state, to_state, event, guards, actions in self._transitions:
            machine.add_transition(
                from_state, to_state, event,
                guards=guards, actions=actions,
                internal=(to_state is None)
            )

        return machine


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def state_machine(name: str) -> StateMachineBuilder:
    """Create a state machine builder"""
    return StateMachineBuilder(name)


def guard(condition: Callable[[StateContext], bool]) -> Guard:
    """Create a guard from a condition function"""
    return condition


def action(func: Callable[[StateContext], Awaitable[None]]) -> Action:
    """Create an action from an async function"""
    return func


# Common guards
def has_variable(key: str) -> Guard:
    """Guard that checks if a variable exists"""
    return lambda ctx: key in ctx.variables


def variable_equals(key: str, value: Any) -> Guard:
    """Guard that checks if a variable equals a value"""
    return lambda ctx: ctx.get(key) == value


def variable_gt(key: str, value: Any) -> Guard:
    """Guard that checks if a variable is greater than a value"""
    return lambda ctx: ctx.get(key, 0) > value


def variable_lt(key: str, value: Any) -> Guard:
    """Guard that checks if a variable is less than a value"""
    return lambda ctx: ctx.get(key, 0) < value


def event_has_data(key: str) -> Guard:
    """Guard that checks if event data has a key"""
    return lambda ctx: key in ctx.event_data


# Common actions
def set_variable(key: str, value: Any) -> Action:
    """Action that sets a variable"""
    async def _action(ctx: StateContext) -> None:
        ctx.set(key, value)
    return _action


def increment_variable(key: str, amount: int = 1) -> Action:
    """Action that increments a variable"""
    async def _action(ctx: StateContext) -> None:
        ctx.set(key, ctx.get(key, 0) + amount)
    return _action


def copy_event_data(from_key: str, to_key: str) -> Action:
    """Action that copies event data to a variable"""
    async def _action(ctx: StateContext) -> None:
        ctx.set(to_key, ctx.get_event_data(from_key))
    return _action
