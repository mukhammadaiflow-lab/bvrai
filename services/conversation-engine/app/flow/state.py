"""Flow state management."""

import structlog
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


logger = structlog.get_logger()


class FlowStatus(str, Enum):
    """Status of flow execution."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"  # Waiting for user input
    COMPLETED = "completed"
    FAILED = "failed"
    TRANSFERRED = "transferred"


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_node: str
    to_node: str
    trigger: str  # What triggered the transition
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowState:
    """
    Current state of a conversation flow.

    Tracks:
    - Current position in flow
    - Variables and slots
    - Execution history
    - Error state
    """

    flow_id: str
    session_id: str

    # Current position
    current_node_id: Optional[str] = None
    status: FlowStatus = FlowStatus.PENDING

    # Variables (slots and computed values)
    variables: Dict[str, Any] = field(default_factory=dict)

    # Slot collection state
    pending_slot: Optional[str] = None
    slot_attempts: Dict[str, int] = field(default_factory=dict)

    # Execution history
    transitions: List[StateTransition] = field(default_factory=list)
    visited_nodes: List[str] = field(default_factory=list)

    # Loop tracking
    loop_counters: Dict[str, int] = field(default_factory=dict)
    max_loop_iterations: int = 10

    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Output buffer
    pending_messages: List[str] = field(default_factory=list)
    pending_actions: List[Dict[str, Any]] = field(default_factory=list)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable value."""
        self.variables[name] = value
        self.last_activity = datetime.utcnow()

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable value."""
        return self.variables.get(name, default)

    def has_variable(self, name: str) -> bool:
        """Check if variable exists."""
        return name in self.variables

    def transition_to(
        self,
        node_id: str,
        trigger: str = "auto",
    ) -> None:
        """Record transition to a new node."""
        if self.current_node_id:
            transition = StateTransition(
                from_node=self.current_node_id,
                to_node=node_id,
                trigger=trigger,
                context_snapshot=dict(self.variables),
            )
            self.transitions.append(transition)

        self.current_node_id = node_id
        self.visited_nodes.append(node_id)
        self.last_activity = datetime.utcnow()

        logger.debug(
            "flow_transition",
            session_id=self.session_id,
            from_node=self.transitions[-1].from_node if self.transitions else None,
            to_node=node_id,
            trigger=trigger,
        )

    def increment_loop(self, loop_id: str) -> int:
        """Increment loop counter and return new value."""
        self.loop_counters[loop_id] = self.loop_counters.get(loop_id, 0) + 1
        return self.loop_counters[loop_id]

    def reset_loop(self, loop_id: str) -> None:
        """Reset a loop counter."""
        self.loop_counters[loop_id] = 0

    def is_loop_exceeded(self, loop_id: str) -> bool:
        """Check if loop has exceeded max iterations."""
        return self.loop_counters.get(loop_id, 0) >= self.max_loop_iterations

    def add_message(self, message: str) -> None:
        """Add message to output buffer."""
        self.pending_messages.append(message)

    def add_action(self, action: Dict[str, Any]) -> None:
        """Add action to output buffer."""
        self.pending_actions.append(action)

    def flush_messages(self) -> List[str]:
        """Get and clear pending messages."""
        messages = self.pending_messages.copy()
        self.pending_messages.clear()
        return messages

    def flush_actions(self) -> List[Dict[str, Any]]:
        """Get and clear pending actions."""
        actions = self.pending_actions.copy()
        self.pending_actions.clear()
        return actions

    def record_slot_attempt(self, slot_name: str) -> int:
        """Record an attempt to collect a slot."""
        self.slot_attempts[slot_name] = self.slot_attempts.get(slot_name, 0) + 1
        return self.slot_attempts[slot_name]

    def get_slot_attempts(self, slot_name: str) -> int:
        """Get number of attempts for a slot."""
        return self.slot_attempts.get(slot_name, 0)

    def set_error(self, message: str) -> None:
        """Set error state."""
        self.error_message = message
        self.status = FlowStatus.FAILED
        logger.error(
            "flow_error",
            session_id=self.session_id,
            current_node=self.current_node_id,
            error=message,
        )

    def clear_error(self) -> None:
        """Clear error state."""
        self.error_message = None
        if self.status == FlowStatus.FAILED:
            self.status = FlowStatus.RUNNING

    def can_retry(self) -> bool:
        """Check if can retry after error."""
        return self.retry_count < self.max_retries

    def increment_retry(self) -> int:
        """Increment retry count."""
        self.retry_count += 1
        return self.retry_count

    def get_duration_seconds(self) -> float:
        """Get flow duration in seconds."""
        return (datetime.utcnow() - self.started_at).total_seconds()

    def get_context(self) -> Dict[str, Any]:
        """Get full context for condition evaluation."""
        return {
            "variables": self.variables,
            "slots": self.variables,  # Alias for convenience
            "current_node": self.current_node_id,
            "status": self.status.value,
            "turn_count": len(self.visited_nodes),
            "duration_seconds": self.get_duration_seconds(),
            **self.variables,  # Direct access to variables
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "flow_id": self.flow_id,
            "session_id": self.session_id,
            "current_node_id": self.current_node_id,
            "status": self.status.value,
            "variables": self.variables,
            "pending_slot": self.pending_slot,
            "slot_attempts": self.slot_attempts,
            "visited_nodes": self.visited_nodes,
            "loop_counters": self.loop_counters,
            "started_at": self.started_at.isoformat(),
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlowState":
        """Create from dictionary."""
        state = cls(
            flow_id=data["flow_id"],
            session_id=data["session_id"],
        )
        state.current_node_id = data.get("current_node_id")
        state.status = FlowStatus(data.get("status", "pending"))
        state.variables = data.get("variables", {})
        state.pending_slot = data.get("pending_slot")
        state.slot_attempts = data.get("slot_attempts", {})
        state.visited_nodes = data.get("visited_nodes", [])
        state.loop_counters = data.get("loop_counters", {})
        state.error_message = data.get("error_message")
        state.retry_count = data.get("retry_count", 0)

        if data.get("started_at"):
            state.started_at = datetime.fromisoformat(data["started_at"])

        return state


class StateStore:
    """
    Persistent storage for flow states.

    Supports in-memory and Redis backends.
    """

    def __init__(self, redis_client=None, ttl_seconds: int = 3600):
        self._states: Dict[str, FlowState] = {}
        self._redis = redis_client
        self._ttl = ttl_seconds

    async def save(self, state: FlowState) -> None:
        """Save state."""
        self._states[state.session_id] = state

        if self._redis:
            import json
            await self._redis.setex(
                f"flow_state:{state.session_id}",
                self._ttl,
                json.dumps(state.to_dict()),
            )

    async def load(self, session_id: str) -> Optional[FlowState]:
        """Load state."""
        if session_id in self._states:
            return self._states[session_id]

        if self._redis:
            import json
            data = await self._redis.get(f"flow_state:{session_id}")
            if data:
                state = FlowState.from_dict(json.loads(data))
                self._states[session_id] = state
                return state

        return None

    async def delete(self, session_id: str) -> bool:
        """Delete state."""
        if session_id in self._states:
            del self._states[session_id]

        if self._redis:
            await self._redis.delete(f"flow_state:{session_id}")

        return True

    async def exists(self, session_id: str) -> bool:
        """Check if state exists."""
        if session_id in self._states:
            return True

        if self._redis:
            return await self._redis.exists(f"flow_state:{session_id}")

        return False
