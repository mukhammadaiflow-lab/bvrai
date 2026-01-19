"""
Conversation Context Management

This module handles conversation context including variables,
scope management, and context persistence.
"""

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
)


logger = logging.getLogger(__name__)


class ContextScope(str, Enum):
    """Scope levels for context variables."""

    TURN = "turn"               # Single turn only
    FLOW = "flow"               # Within current flow
    SESSION = "session"         # Entire session
    USER = "user"               # Persisted across sessions
    GLOBAL = "global"           # System-wide


@dataclass
class ContextVariable:
    """A variable stored in context."""

    name: str
    value: Any
    scope: ContextScope = ContextScope.SESSION

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source: str = ""  # What set this variable

    # Expiration
    ttl_seconds: Optional[int] = None

    # Type info
    var_type: str = "any"  # For validation

    @property
    def is_expired(self) -> bool:
        """Check if variable has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.updated_at > self.ttl_seconds


@dataclass
class ConversationContext:
    """
    Context for a conversation session.

    Stores variables, entities, and state across turns.
    """

    # Identification
    session_id: str = ""
    user_id: Optional[str] = None
    agent_id: Optional[str] = None

    # Variables by scope
    _variables: Dict[ContextScope, Dict[str, ContextVariable]] = field(
        default_factory=lambda: {scope: {} for scope in ContextScope}
    )

    # Conversation history
    turns: List[Dict[str, Any]] = field(default_factory=list)
    max_turns: int = 50

    # Current flow state
    current_flow_id: Optional[str] = None
    current_node_id: Optional[str] = None
    flow_stack: List[Dict[str, str]] = field(default_factory=list)

    # Intent tracking
    current_intent: Optional[str] = None
    intent_history: List[str] = field(default_factory=list)

    # Slot values
    slots: Dict[str, Any] = field(default_factory=dict)
    pending_slots: List[str] = field(default_factory=list)

    # Entities extracted across turns
    entities: Dict[str, List[Any]] = field(default_factory=dict)

    # User profile (from persistent storage)
    user_profile: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    language: str = "en-US"
    channel: str = "voice"

    # Custom data
    custom: Dict[str, Any] = field(default_factory=dict)

    def set(
        self,
        name: str,
        value: Any,
        scope: ContextScope = ContextScope.SESSION,
        source: str = "",
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Set a context variable."""
        var = ContextVariable(
            name=name,
            value=value,
            scope=scope,
            source=source,
            ttl_seconds=ttl_seconds,
        )
        self._variables[scope][name] = var
        self.updated_at = time.time()

    def get(
        self,
        name: str,
        default: Any = None,
        scope: Optional[ContextScope] = None,
    ) -> Any:
        """
        Get a context variable.

        If scope is not specified, searches from narrowest to widest scope.
        """
        if scope:
            var = self._variables[scope].get(name)
            if var and not var.is_expired:
                return var.value
            return default

        # Search in order: TURN -> FLOW -> SESSION -> USER -> GLOBAL
        for s in [ContextScope.TURN, ContextScope.FLOW, ContextScope.SESSION,
                  ContextScope.USER, ContextScope.GLOBAL]:
            var = self._variables[s].get(name)
            if var and not var.is_expired:
                return var.value

        return default

    def delete(
        self,
        name: str,
        scope: Optional[ContextScope] = None,
    ) -> bool:
        """Delete a context variable."""
        if scope:
            if name in self._variables[scope]:
                del self._variables[scope][name]
                return True
            return False

        # Delete from all scopes
        deleted = False
        for s in ContextScope:
            if name in self._variables[s]:
                del self._variables[s][name]
                deleted = True
        return deleted

    def has(self, name: str, scope: Optional[ContextScope] = None) -> bool:
        """Check if a variable exists."""
        return self.get(name, scope=scope) is not None

    def clear_scope(self, scope: ContextScope) -> None:
        """Clear all variables in a scope."""
        self._variables[scope].clear()

    def get_all(
        self,
        scope: Optional[ContextScope] = None,
    ) -> Dict[str, Any]:
        """Get all variables, optionally filtered by scope."""
        result = {}

        scopes = [scope] if scope else list(ContextScope)

        for s in scopes:
            for name, var in self._variables[s].items():
                if not var.is_expired:
                    result[name] = var.value

        return result

    # Slot management
    def set_slot(self, name: str, value: Any) -> None:
        """Set a slot value."""
        self.slots[name] = value
        if name in self.pending_slots:
            self.pending_slots.remove(name)
        self.updated_at = time.time()

    def get_slot(self, name: str, default: Any = None) -> Any:
        """Get a slot value."""
        return self.slots.get(name, default)

    def clear_slot(self, name: str) -> None:
        """Clear a slot value."""
        if name in self.slots:
            del self.slots[name]

    def add_pending_slot(self, name: str) -> None:
        """Add a slot to pending list."""
        if name not in self.pending_slots:
            self.pending_slots.append(name)

    # Entity management
    def add_entity(
        self,
        entity_type: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an extracted entity."""
        if entity_type not in self.entities:
            self.entities[entity_type] = []

        entity = {
            "value": value,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self.entities[entity_type].append(entity)

    def get_entity(
        self,
        entity_type: str,
        index: int = -1,
    ) -> Optional[Any]:
        """Get an entity by type (default: most recent)."""
        entities = self.entities.get(entity_type, [])
        if entities:
            return entities[index]["value"]
        return None

    # Turn management
    def add_turn(
        self,
        user_text: str,
        agent_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a turn to history."""
        turn = {
            "user": user_text,
            "agent": agent_text,
            "timestamp": time.time(),
            "intent": self.current_intent,
            "metadata": metadata or {},
        }
        self.turns.append(turn)

        # Trim if needed
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_conversation_history(
        self,
        max_turns: int = 10,
        format: str = "messages",
    ) -> Union[List[Dict[str, str]], str]:
        """Get conversation history."""
        recent = self.turns[-max_turns:]

        if format == "messages":
            messages = []
            for turn in recent:
                if turn.get("user"):
                    messages.append({"role": "user", "content": turn["user"]})
                if turn.get("agent"):
                    messages.append({"role": "assistant", "content": turn["agent"]})
            return messages

        elif format == "text":
            lines = []
            for turn in recent:
                if turn.get("user"):
                    lines.append(f"User: {turn['user']}")
                if turn.get("agent"):
                    lines.append(f"Agent: {turn['agent']}")
            return "\n".join(lines)

        return recent

    # Flow management
    def push_flow(self, flow_id: str, node_id: str) -> None:
        """Push current flow to stack and set new one."""
        if self.current_flow_id:
            self.flow_stack.append({
                "flow_id": self.current_flow_id,
                "node_id": self.current_node_id or "",
            })
        self.current_flow_id = flow_id
        self.current_node_id = node_id

    def pop_flow(self) -> Optional[Dict[str, str]]:
        """Pop and restore previous flow from stack."""
        if self.flow_stack:
            prev = self.flow_stack.pop()
            self.current_flow_id = prev["flow_id"]
            self.current_node_id = prev["node_id"]
            return prev
        return None

    # Intent management
    def set_intent(self, intent: str) -> None:
        """Set current intent and add to history."""
        self.current_intent = intent
        self.intent_history.append(intent)
        # Keep last 20 intents
        if len(self.intent_history) > 20:
            self.intent_history = self.intent_history[-20:]

    # Serialization
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        variables = {}
        for scope in [ContextScope.SESSION, ContextScope.USER]:
            for name, var in self._variables[scope].items():
                if not var.is_expired:
                    variables[name] = {
                        "value": var.value,
                        "scope": var.scope.value,
                    }

        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "variables": variables,
            "slots": self.slots,
            "entities": self.entities,
            "turns": self.turns,
            "current_flow_id": self.current_flow_id,
            "current_node_id": self.current_node_id,
            "current_intent": self.current_intent,
            "user_profile": self.user_profile,
            "language": self.language,
            "custom": self.custom,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Create from dictionary."""
        ctx = cls(
            session_id=data.get("session_id", ""),
            user_id=data.get("user_id"),
            agent_id=data.get("agent_id"),
            language=data.get("language", "en-US"),
        )

        # Restore variables
        for name, var_data in data.get("variables", {}).items():
            ctx.set(
                name=name,
                value=var_data["value"],
                scope=ContextScope(var_data.get("scope", "session")),
            )

        ctx.slots = data.get("slots", {})
        ctx.entities = data.get("entities", {})
        ctx.turns = data.get("turns", [])
        ctx.current_flow_id = data.get("current_flow_id")
        ctx.current_node_id = data.get("current_node_id")
        ctx.current_intent = data.get("current_intent")
        ctx.user_profile = data.get("user_profile", {})
        ctx.custom = data.get("custom", {})
        ctx.created_at = data.get("created_at", time.time())
        ctx.updated_at = data.get("updated_at", time.time())

        return ctx

    def copy(self) -> "ConversationContext":
        """Create a deep copy."""
        return copy.deepcopy(self)


class ContextManager:
    """
    Manages context across multiple sessions with persistence support.
    """

    def __init__(
        self,
        storage_backend: Optional[Any] = None,
        user_context_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        self._contexts: Dict[str, ConversationContext] = {}
        self._storage = storage_backend
        self._user_loader = user_context_loader

    def create(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs,
    ) -> ConversationContext:
        """Create a new conversation context."""
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            **kwargs,
        )

        # Load user profile if available
        if user_id and self._user_loader:
            try:
                user_data = self._user_loader(user_id)
                context.user_profile = user_data.get("profile", {})
                # Restore user-scoped variables
                for name, value in user_data.get("variables", {}).items():
                    context.set(name, value, scope=ContextScope.USER)
            except Exception as e:
                logger.warning(f"Failed to load user context: {e}")

        self._contexts[session_id] = context
        return context

    def get(self, session_id: str) -> Optional[ConversationContext]:
        """Get context for a session."""
        context = self._contexts.get(session_id)

        # Try loading from storage if not in memory
        if context is None and self._storage:
            try:
                data = self._storage.load(session_id)
                if data:
                    context = ConversationContext.from_dict(data)
                    self._contexts[session_id] = context
            except Exception as e:
                logger.warning(f"Failed to load context from storage: {e}")

        return context

    def save(self, session_id: str) -> bool:
        """Save context to persistent storage."""
        context = self._contexts.get(session_id)
        if not context or not self._storage:
            return False

        try:
            data = context.to_dict()
            self._storage.save(session_id, data)
            return True
        except Exception as e:
            logger.error(f"Failed to save context: {e}")
            return False

    def delete(self, session_id: str) -> bool:
        """Delete a context."""
        if session_id in self._contexts:
            del self._contexts[session_id]

        if self._storage:
            try:
                self._storage.delete(session_id)
            except Exception:
                pass

        return True

    def cleanup_expired(self, max_age_seconds: int = 604800) -> int:
        """
        Clean up expired contexts from memory (NOT from persistent storage).

        This only removes contexts from the in-memory cache to free up resources.
        Data in persistent storage is preserved for historical access.

        Args:
            max_age_seconds: Maximum age before cleanup (default: 7 days)

        Returns:
            Number of contexts cleaned up from memory
        """
        now = time.time()
        expired = []

        for session_id, context in self._contexts.items():
            if now - context.updated_at > max_age_seconds:
                expired.append(session_id)

        # Only remove from memory, preserve in storage
        for session_id in expired:
            # Save to storage before removing from memory
            if self._storage:
                try:
                    context = self._contexts.get(session_id)
                    if context:
                        self._storage.save(session_id, context.to_dict())
                except Exception as e:
                    logger.warning(f"Failed to save context before cleanup: {e}")

            # Remove from memory only (not storage)
            if session_id in self._contexts:
                del self._contexts[session_id]

        if expired:
            logger.info(f"Cleaned up {len(expired)} contexts from memory (data preserved in storage)")

        return len(expired)


__all__ = [
    "ContextScope",
    "ContextVariable",
    "ConversationContext",
    "ContextManager",
]
