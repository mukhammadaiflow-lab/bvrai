"""Short-term memory for active conversations."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import uuid

from app.memory.store import MemoryStore, MemoryEntry, MemoryType


logger = structlog.get_logger()


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    turn_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    audio_duration_ms: Optional[int] = None
    confidence: Optional[float] = None
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    """
    Short-term memory for active conversations.

    Maintains:
    - Recent conversation turns
    - Current context/state
    - Temporary variables
    - Working slots
    """

    def __init__(
        self,
        session_id: str,
        max_turns: int = 50,
        ttl_seconds: int = 3600,  # 1 hour
        store: Optional[MemoryStore] = None,
    ):
        self.session_id = session_id
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._store = store

        # Conversation history
        self._turns: deque = deque(maxlen=max_turns)

        # Current state
        self._slots: Dict[str, Any] = {}
        self._context: Dict[str, Any] = {}
        self._temp_vars: Dict[str, Any] = {}

        # Timing
        self._created_at = datetime.utcnow()
        self._last_activity = datetime.utcnow()

    def add_turn(
        self,
        role: str,
        content: str,
        **metadata,
    ) -> ConversationTurn:
        """
        Add a conversation turn.

        Args:
            role: Speaker role (user/assistant)
            content: Message content
            **metadata: Additional metadata

        Returns:
            Created turn
        """
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            role=role,
            content=content,
            metadata=metadata,
        )

        if "audio_duration_ms" in metadata:
            turn.audio_duration_ms = metadata["audio_duration_ms"]
        if "confidence" in metadata:
            turn.confidence = metadata["confidence"]
        if "function_calls" in metadata:
            turn.function_calls = metadata["function_calls"]

        self._turns.append(turn)
        self._last_activity = datetime.utcnow()

        return turn

    def add_user_turn(self, content: str, **metadata) -> ConversationTurn:
        """Add user turn."""
        return self.add_turn("user", content, **metadata)

    def add_assistant_turn(self, content: str, **metadata) -> ConversationTurn:
        """Add assistant turn."""
        return self.add_turn("assistant", content, **metadata)

    def get_recent_turns(
        self,
        limit: Optional[int] = None,
        role: Optional[str] = None,
    ) -> List[ConversationTurn]:
        """
        Get recent conversation turns.

        Args:
            limit: Maximum turns to return
            role: Filter by role

        Returns:
            List of turns
        """
        turns = list(self._turns)

        if role:
            turns = [t for t in turns if t.role == role]

        if limit:
            turns = turns[-limit:]

        return turns

    def get_last_turn(self, role: Optional[str] = None) -> Optional[ConversationTurn]:
        """Get the most recent turn."""
        turns = self.get_recent_turns(limit=1, role=role)
        return turns[0] if turns else None

    def set_slot(self, name: str, value: Any) -> None:
        """Set a slot value."""
        self._slots[name] = value
        self._last_activity = datetime.utcnow()

    def get_slot(self, name: str, default: Any = None) -> Any:
        """Get a slot value."""
        return self._slots.get(name, default)

    def get_slots(self) -> Dict[str, Any]:
        """Get all slots."""
        return dict(self._slots)

    def clear_slot(self, name: str) -> bool:
        """Clear a slot."""
        if name in self._slots:
            del self._slots[name]
            return True
        return False

    def set_context(self, key: str, value: Any) -> None:
        """Set context value."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context value."""
        return self._context.get(key, default)

    def update_context(self, updates: Dict[str, Any]) -> None:
        """Update multiple context values."""
        self._context.update(updates)

    def get_full_context(self) -> Dict[str, Any]:
        """Get full context including slots."""
        return {
            **self._context,
            "slots": self._slots,
            "turn_count": len(self._turns),
            "session_id": self.session_id,
        }

    def set_temp(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set temporary variable with expiration."""
        self._temp_vars[key] = {
            "value": value,
            "expires_at": datetime.utcnow() + timedelta(seconds=ttl_seconds),
        }

    def get_temp(self, key: str, default: Any = None) -> Any:
        """Get temporary variable."""
        if key not in self._temp_vars:
            return default

        entry = self._temp_vars[key]
        if datetime.utcnow() > entry["expires_at"]:
            del self._temp_vars[key]
            return default

        return entry["value"]

    def format_for_llm(
        self,
        max_turns: Optional[int] = None,
        include_metadata: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Format conversation for LLM input.

        Args:
            max_turns: Maximum turns to include
            include_metadata: Include turn metadata

        Returns:
            List of message dicts
        """
        turns = self.get_recent_turns(limit=max_turns)
        messages = []

        for turn in turns:
            msg = {
                "role": turn.role,
                "content": turn.content,
            }

            if include_metadata and turn.metadata:
                msg["metadata"] = turn.metadata

            messages.append(msg)

        return messages

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation."""
        turns = list(self._turns)
        user_turns = [t for t in turns if t.role == "user"]
        assistant_turns = [t for t in turns if t.role == "assistant"]

        total_user_audio = sum(
            t.audio_duration_ms or 0 for t in user_turns
        )

        return {
            "session_id": self.session_id,
            "total_turns": len(turns),
            "user_turns": len(user_turns),
            "assistant_turns": len(assistant_turns),
            "slots_filled": len(self._slots),
            "total_user_audio_ms": total_user_audio,
            "duration_seconds": (datetime.utcnow() - self._created_at).total_seconds(),
        }

    async def persist(self) -> None:
        """Persist important memories to long-term store."""
        if not self._store:
            return

        # Persist filled slots
        for name, value in self._slots.items():
            entry = MemoryEntry(
                id=f"{self.session_id}:slot:{name}",
                type=MemoryType.ENTITY,
                content=f"{name}: {value}",
                metadata={"slot_name": name, "slot_value": value},
                session_id=self.session_id,
                importance=0.7,
            )
            await self._store.store(entry, ttl_seconds=self.ttl_seconds)

    def clear(self) -> None:
        """Clear all short-term memory."""
        self._turns.clear()
        self._slots.clear()
        self._context.clear()
        self._temp_vars.clear()

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        elapsed = (datetime.utcnow() - self._last_activity).total_seconds()
        return elapsed > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "role": t.role,
                    "content": t.content,
                    "timestamp": t.timestamp.isoformat(),
                }
                for t in self._turns
            ],
            "slots": self._slots,
            "context": self._context,
            "created_at": self._created_at.isoformat(),
            "last_activity": self._last_activity.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], store: Optional[MemoryStore] = None) -> "ShortTermMemory":
        """Create from dictionary."""
        memory = cls(
            session_id=data["session_id"],
            store=store,
        )

        memory._slots = data.get("slots", {})
        memory._context = data.get("context", {})

        if data.get("created_at"):
            memory._created_at = datetime.fromisoformat(data["created_at"])
        if data.get("last_activity"):
            memory._last_activity = datetime.fromisoformat(data["last_activity"])

        for turn_data in data.get("turns", []):
            turn = ConversationTurn(
                turn_id=turn_data["turn_id"],
                role=turn_data["role"],
                content=turn_data["content"],
            )
            if turn_data.get("timestamp"):
                turn.timestamp = datetime.fromisoformat(turn_data["timestamp"])
            memory._turns.append(turn)

        return memory
