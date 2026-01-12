"""Conversation context manager."""

import asyncio
import structlog
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import hashlib


logger = structlog.get_logger()


class MessageRole(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For function/tool calls
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

    # Audio metadata
    audio_duration_ms: Optional[int] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM."""
        result = {
            "role": self.role.value,
            "content": self.content,
        }

        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.function_call:
            result["function_call"] = self.function_call

        return result

    def token_estimate(self) -> int:
        """Estimate token count (rough approximation)."""
        # Roughly 4 chars per token for English
        return len(self.content) // 4 + 10  # +10 for role/structure overhead


@dataclass
class EntitySlot:
    """A slot for collecting entity information."""
    name: str
    type: str  # string, number, date, email, phone, etc.
    required: bool = False
    value: Optional[Any] = None
    confirmed: bool = False
    prompt: Optional[str] = None
    validation_regex: Optional[str] = None

    def is_filled(self) -> bool:
        """Check if slot has a value."""
        return self.value is not None

    def is_valid(self) -> bool:
        """Check if slot value is valid."""
        if not self.is_filled():
            return not self.required

        if self.validation_regex:
            import re
            return bool(re.match(self.validation_regex, str(self.value)))

        return True


@dataclass
class ConversationContext:
    """
    Full context for a conversation.

    Manages:
    - Message history
    - Extracted entities/slots
    - Conversation state
    - Agent configuration
    - Call metadata
    """

    session_id: str
    agent_id: str

    # Message history
    messages: List[Message] = field(default_factory=list)

    # System prompt and configuration
    system_prompt: str = ""
    agent_config: Dict[str, Any] = field(default_factory=dict)

    # Entity slots for data collection
    slots: Dict[str, EntitySlot] = field(default_factory=dict)

    # Conversation state
    current_intent: Optional[str] = None
    conversation_stage: str = "greeting"
    turn_count: int = 0

    # Call metadata
    caller_phone: Optional[str] = None
    caller_name: Optional[str] = None
    call_direction: str = "inbound"  # inbound or outbound

    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    # Summarization
    summary: Optional[str] = None
    summarized_up_to: int = 0

    # RAG context
    retrieved_context: List[Dict[str, Any]] = field(default_factory=list)

    # Function results
    pending_function_calls: List[Dict[str, Any]] = field(default_factory=list)
    function_results: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: MessageRole, content: str, **kwargs) -> Message:
        """Add a message to the conversation."""
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.last_activity = datetime.utcnow()
        self.turn_count += 1
        return message

    def add_user_message(self, content: str, **kwargs) -> Message:
        """Add user message (from caller)."""
        return self.add_message(MessageRole.USER, content, **kwargs)

    def add_assistant_message(self, content: str, **kwargs) -> Message:
        """Add assistant message (AI response)."""
        return self.add_message(MessageRole.ASSISTANT, content, **kwargs)

    def add_function_result(
        self,
        name: str,
        result: Any,
        tool_call_id: Optional[str] = None,
    ) -> Message:
        """Add function/tool result."""
        content = json.dumps(result) if not isinstance(result, str) else result
        return self.add_message(
            MessageRole.TOOL,
            content,
            name=name,
            tool_call_id=tool_call_id,
        )

    def get_messages_for_llm(
        self,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get messages formatted for LLM.

        Args:
            max_messages: Maximum number of messages to include
            max_tokens: Maximum token count
            include_system: Include system prompt

        Returns:
            List of message dicts
        """
        result = []

        # Add system prompt
        if include_system and self.system_prompt:
            system_content = self._build_system_prompt()
            result.append({
                "role": "system",
                "content": system_content,
            })

        # Get recent messages
        messages = self.messages
        if max_messages:
            messages = messages[-max_messages:]

        # Token limiting
        if max_tokens:
            total_tokens = sum(m.token_estimate() for m in messages)
            while total_tokens > max_tokens and len(messages) > 1:
                messages = messages[1:]
                total_tokens = sum(m.token_estimate() for m in messages)

        # Add summary if available and we truncated
        if self.summary and len(messages) < len(self.messages):
            result.append({
                "role": "system",
                "content": f"[Earlier conversation summary: {self.summary}]",
            })

        # Add messages
        for msg in messages:
            result.append(msg.to_dict())

        return result

    def _build_system_prompt(self) -> str:
        """Build full system prompt with context."""
        parts = [self.system_prompt]

        # Add slot information
        filled_slots = {k: v.value for k, v in self.slots.items() if v.is_filled()}
        if filled_slots:
            parts.append(f"\n\nCollected information: {json.dumps(filled_slots)}")

        # Add retrieved context
        if self.retrieved_context:
            context_text = "\n".join(
                f"- {ctx.get('content', '')}"
                for ctx in self.retrieved_context[:3]
            )
            parts.append(f"\n\nRelevant knowledge:\n{context_text}")

        # Add current stage
        parts.append(f"\n\nConversation stage: {self.conversation_stage}")

        return "".join(parts)

    def set_slot(self, name: str, value: Any, confirmed: bool = False) -> bool:
        """Set a slot value."""
        if name in self.slots:
            self.slots[name].value = value
            self.slots[name].confirmed = confirmed
            return True
        return False

    def get_slot(self, name: str) -> Optional[Any]:
        """Get a slot value."""
        if name in self.slots:
            return self.slots[name].value
        return None

    def get_unfilled_required_slots(self) -> List[EntitySlot]:
        """Get required slots that are not filled."""
        return [
            slot for slot in self.slots.values()
            if slot.required and not slot.is_filled()
        ]

    def get_duration_seconds(self) -> float:
        """Get conversation duration in seconds."""
        return (datetime.utcnow() - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "turn_count": self.turn_count,
            "conversation_stage": self.conversation_stage,
            "current_intent": self.current_intent,
            "slots": {k: {"value": v.value, "confirmed": v.confirmed}
                     for k, v in self.slots.items()},
            "started_at": self.started_at.isoformat(),
            "duration_seconds": self.get_duration_seconds(),
            "message_count": len(self.messages),
        }


class ContextManager:
    """
    Manages conversation contexts across sessions.

    Handles:
    - Context creation and retrieval
    - Context persistence (Redis)
    - Context expiration
    - Cross-session context sharing
    """

    def __init__(self, redis_client=None, ttl_seconds: int = 3600):
        self._contexts: Dict[str, ConversationContext] = {}
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    async def create_context(
        self,
        session_id: str,
        agent_id: str,
        system_prompt: str = "",
        agent_config: Optional[Dict[str, Any]] = None,
        slots: Optional[Dict[str, EntitySlot]] = None,
        **kwargs,
    ) -> ConversationContext:
        """Create a new conversation context."""
        context = ConversationContext(
            session_id=session_id,
            agent_id=agent_id,
            system_prompt=system_prompt,
            agent_config=agent_config or {},
            slots=slots or {},
            **kwargs,
        )

        async with self._lock:
            self._contexts[session_id] = context

        # Persist to Redis
        await self._save_to_redis(context)

        logger.info(
            "context_created",
            session_id=session_id,
            agent_id=agent_id,
        )

        return context

    async def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get context for a session."""
        # Check in-memory first
        if session_id in self._contexts:
            return self._contexts[session_id]

        # Try Redis
        context = await self._load_from_redis(session_id)
        if context:
            async with self._lock:
                self._contexts[session_id] = context
            return context

        return None

    async def update_context(self, context: ConversationContext) -> None:
        """Update context in storage."""
        await self._save_to_redis(context)

    async def delete_context(self, session_id: str) -> bool:
        """Delete a context."""
        async with self._lock:
            if session_id in self._contexts:
                del self._contexts[session_id]

        if self._redis:
            await self._redis.delete(f"context:{session_id}")

        return True

    async def _save_to_redis(self, context: ConversationContext) -> None:
        """Save context to Redis."""
        if not self._redis:
            return

        try:
            data = {
                "session_id": context.session_id,
                "agent_id": context.agent_id,
                "system_prompt": context.system_prompt,
                "agent_config": context.agent_config,
                "conversation_stage": context.conversation_stage,
                "current_intent": context.current_intent,
                "turn_count": context.turn_count,
                "caller_phone": context.caller_phone,
                "caller_name": context.caller_name,
                "call_direction": context.call_direction,
                "started_at": context.started_at.isoformat(),
                "summary": context.summary,
                "slots": {
                    k: {"value": v.value, "confirmed": v.confirmed, "name": v.name,
                        "type": v.type, "required": v.required}
                    for k, v in context.slots.items()
                },
                "messages": [
                    {
                        "role": m.role.value,
                        "content": m.content,
                        "timestamp": m.timestamp.isoformat(),
                        "name": m.name,
                        "tool_call_id": m.tool_call_id,
                    }
                    for m in context.messages[-50:]  # Keep last 50 messages
                ],
            }

            await self._redis.setex(
                f"context:{context.session_id}",
                self._ttl,
                json.dumps(data),
            )
        except Exception as e:
            logger.error("redis_save_error", error=str(e))

    async def _load_from_redis(self, session_id: str) -> Optional[ConversationContext]:
        """Load context from Redis."""
        if not self._redis:
            return None

        try:
            data = await self._redis.get(f"context:{session_id}")
            if not data:
                return None

            data = json.loads(data)

            # Reconstruct context
            context = ConversationContext(
                session_id=data["session_id"],
                agent_id=data["agent_id"],
                system_prompt=data.get("system_prompt", ""),
                agent_config=data.get("agent_config", {}),
                conversation_stage=data.get("conversation_stage", "greeting"),
                current_intent=data.get("current_intent"),
                turn_count=data.get("turn_count", 0),
                caller_phone=data.get("caller_phone"),
                caller_name=data.get("caller_name"),
                call_direction=data.get("call_direction", "inbound"),
                started_at=datetime.fromisoformat(data["started_at"]),
                summary=data.get("summary"),
            )

            # Reconstruct slots
            for name, slot_data in data.get("slots", {}).items():
                context.slots[name] = EntitySlot(
                    name=slot_data["name"],
                    type=slot_data["type"],
                    required=slot_data.get("required", False),
                    value=slot_data.get("value"),
                    confirmed=slot_data.get("confirmed", False),
                )

            # Reconstruct messages
            for msg_data in data.get("messages", []):
                context.messages.append(Message(
                    role=MessageRole(msg_data["role"]),
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    name=msg_data.get("name"),
                    tool_call_id=msg_data.get("tool_call_id"),
                ))

            return context

        except Exception as e:
            logger.error("redis_load_error", error=str(e))
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        return {
            "active_contexts": len(self._contexts),
            "redis_enabled": self._redis is not None,
            "ttl_seconds": self._ttl,
        }
