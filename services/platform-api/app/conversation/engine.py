"""
Conversation Engine Core

Central conversation management:
- Conversation lifecycle
- Message processing
- State management
- Event handling
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import uuid
import json

logger = logging.getLogger(__name__)


class ConversationState(str, Enum):
    """Conversation states."""
    CREATED = "created"
    STARTED = "started"
    ACTIVE = "active"
    WAITING_USER = "waiting_user"
    WAITING_AGENT = "waiting_agent"
    PROCESSING = "processing"
    PAUSED = "paused"
    ENDED = "ended"
    FAILED = "failed"


class MessageRole(str, Enum):
    """Message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


class MessageType(str, Enum):
    """Message types."""
    TEXT = "text"
    AUDIO = "audio"
    DTMF = "dtmf"
    ACTION = "action"
    EVENT = "event"
    FUNCTION_CALL = "function_call"
    FUNCTION_RESULT = "function_result"


@dataclass
class ConversationConfig:
    """Conversation configuration."""
    # Timing
    max_duration_seconds: int = 3600
    idle_timeout_seconds: int = 300
    response_timeout_seconds: int = 30

    # Messages
    max_messages: int = 1000
    max_message_length: int = 10000
    max_context_messages: int = 50

    # Features
    enable_interruption: bool = True
    enable_barge_in: bool = True
    enable_function_calling: bool = True

    # Processing
    parallel_processing: bool = False
    stream_responses: bool = True

    # Language
    default_language: str = "en"
    supported_languages: List[str] = field(default_factory=lambda: ["en"])


@dataclass
class Message:
    """Conversation message."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    role: MessageRole = MessageRole.USER
    message_type: MessageType = MessageType.TEXT

    # Content
    content: str = ""
    audio_url: Optional[str] = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    confidence: float = 1.0

    # Processing
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    sentiment: Optional[str] = None

    # Function calling
    function_name: Optional[str] = None
    function_args: Dict[str, Any] = field(default_factory=dict)
    function_result: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "role": self.role.value,
            "type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent,
            "entities": self.entities,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            role=MessageRole(data.get("role", "user")),
            message_type=MessageType(data.get("type", "text")),
            content=data.get("content", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            intent=data.get("intent"),
            entities=data.get("entities", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Conversation:
    """Conversation representation."""
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    agent_id: str = ""
    call_id: Optional[str] = None

    # State
    state: ConversationState = ConversationState.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Messages
    messages: List[Message] = field(default_factory=list)
    system_prompt: str = ""

    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    slots: Dict[str, Any] = field(default_factory=dict)
    current_intent: Optional[str] = None

    # Participant info
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    user_language: str = "en"

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Get conversation duration."""
        if self.started_at:
            end = self.ended_at or datetime.utcnow()
            return (end - self.started_at).total_seconds()
        return 0.0

    @property
    def message_count(self) -> int:
        """Get message count."""
        return len(self.messages)

    @property
    def user_message_count(self) -> int:
        """Get user message count."""
        return sum(1 for m in self.messages if m.role == MessageRole.USER)

    def add_message(self, message: Message) -> None:
        """Add message to conversation."""
        message.conversation_id = self.conversation_id
        self.messages.append(message)

    def get_last_message(self, role: Optional[MessageRole] = None) -> Optional[Message]:
        """Get last message, optionally filtered by role."""
        for message in reversed(self.messages):
            if role is None or message.role == role:
                return message
        return None

    def get_context_messages(self, max_messages: int = 50) -> List[Message]:
        """Get recent messages for context."""
        return self.messages[-max_messages:]

    def to_llm_messages(self, max_messages: int = 50) -> List[Dict[str, str]]:
        """Convert to LLM message format."""
        result = []

        if self.system_prompt:
            result.append({
                "role": "system",
                "content": self.system_prompt,
            })

        for message in self.get_context_messages(max_messages):
            if message.message_type == MessageType.TEXT:
                result.append({
                    "role": message.role.value,
                    "content": message.content,
                })
            elif message.message_type == MessageType.FUNCTION_CALL:
                result.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": message.function_name,
                        "arguments": json.dumps(message.function_args),
                    },
                })
            elif message.message_type == MessageType.FUNCTION_RESULT:
                result.append({
                    "role": "function",
                    "name": message.function_name,
                    "content": message.function_result or "",
                })

        return result


class ConversationEventType(str, Enum):
    """Conversation event types."""
    STARTED = "started"
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_SENT = "message_sent"
    INTENT_DETECTED = "intent_detected"
    SLOT_FILLED = "slot_filled"
    ACTION_EXECUTED = "action_executed"
    FUNCTION_CALLED = "function_called"
    STATE_CHANGED = "state_changed"
    ENDED = "ended"
    ERROR = "error"


@dataclass
class ConversationEvent:
    """Conversation event."""
    event_type: ConversationEventType
    conversation_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)


class ConversationEventHandler(ABC):
    """Abstract event handler."""

    @abstractmethod
    async def handle(self, event: ConversationEvent) -> None:
        """Handle conversation event."""
        pass


class ConversationProcessor(ABC):
    """Abstract conversation processor."""

    @abstractmethod
    async def process(
        self,
        conversation: Conversation,
        message: Message,
    ) -> Optional[Message]:
        """Process incoming message and generate response."""
        pass


class ConversationEngine:
    """
    Core conversation engine.

    Manages:
    - Message processing
    - Response generation
    - State management
    - Event handling
    """

    def __init__(
        self,
        config: Optional[ConversationConfig] = None,
        processor: Optional[ConversationProcessor] = None,
    ):
        self.config = config or ConversationConfig()
        self._processor = processor

        # Event handlers
        self._event_handlers: List[ConversationEventHandler] = []

        # Pre/post processors
        self._pre_processors: List[Callable[[Conversation, Message], Awaitable[Message]]] = []
        self._post_processors: List[Callable[[Conversation, Message], Awaitable[Message]]] = []

        # Function registry
        self._functions: Dict[str, Callable[..., Awaitable[Any]]] = {}

        # Statistics
        self._messages_processed = 0
        self._responses_generated = 0

    def set_processor(self, processor: ConversationProcessor) -> None:
        """Set conversation processor."""
        self._processor = processor

    def add_event_handler(self, handler: ConversationEventHandler) -> None:
        """Add event handler."""
        self._event_handlers.append(handler)

    def add_pre_processor(
        self,
        processor: Callable[[Conversation, Message], Awaitable[Message]],
    ) -> None:
        """Add pre-processor."""
        self._pre_processors.append(processor)

    def add_post_processor(
        self,
        processor: Callable[[Conversation, Message], Awaitable[Message]],
    ) -> None:
        """Add post-processor."""
        self._post_processors.append(processor)

    def register_function(
        self,
        name: str,
        func: Callable[..., Awaitable[Any]],
    ) -> None:
        """Register callable function."""
        self._functions[name] = func

    async def process_message(
        self,
        conversation: Conversation,
        message: Message,
    ) -> Optional[Message]:
        """
        Process incoming message and generate response.

        Steps:
        1. Pre-process message
        2. Add to conversation
        3. Emit event
        4. Generate response
        5. Post-process response
        6. Return response
        """
        # Update state
        if conversation.state == ConversationState.CREATED:
            conversation.state = ConversationState.STARTED
            conversation.started_at = datetime.utcnow()
            await self._emit_event(ConversationEventType.STARTED, conversation)

        conversation.state = ConversationState.PROCESSING

        # Pre-process
        for preprocessor in self._pre_processors:
            message = await preprocessor(conversation, message)

        # Add message
        conversation.add_message(message)
        self._messages_processed += 1

        # Emit message received event
        await self._emit_event(
            ConversationEventType.MESSAGE_RECEIVED,
            conversation,
            {"message_id": message.message_id, "content": message.content},
        )

        # Generate response
        response = None
        if self._processor:
            try:
                response = await self._processor.process(conversation, message)
            except Exception as e:
                logger.error(f"Processor error: {e}")
                await self._emit_event(
                    ConversationEventType.ERROR,
                    conversation,
                    {"error": str(e)},
                )

        if response:
            # Handle function calls
            if response.message_type == MessageType.FUNCTION_CALL:
                response = await self._handle_function_call(conversation, response)

            # Post-process
            for postprocessor in self._post_processors:
                response = await postprocessor(conversation, response)

            # Add response
            conversation.add_message(response)
            self._responses_generated += 1

            # Emit message sent event
            await self._emit_event(
                ConversationEventType.MESSAGE_SENT,
                conversation,
                {"message_id": response.message_id, "content": response.content},
            )

        # Update state
        conversation.state = ConversationState.WAITING_USER

        return response

    async def _handle_function_call(
        self,
        conversation: Conversation,
        message: Message,
    ) -> Message:
        """Handle function call in response."""
        func_name = message.function_name
        func_args = message.function_args

        if func_name not in self._functions:
            logger.warning(f"Function not found: {func_name}")
            return message

        try:
            # Execute function
            func = self._functions[func_name]
            result = await func(**func_args)

            # Emit event
            await self._emit_event(
                ConversationEventType.FUNCTION_CALLED,
                conversation,
                {"function": func_name, "args": func_args, "result": result},
            )

            # Create function result message
            result_message = Message(
                conversation_id=conversation.conversation_id,
                role=MessageRole.FUNCTION,
                message_type=MessageType.FUNCTION_RESULT,
                function_name=func_name,
                function_result=json.dumps(result) if not isinstance(result, str) else result,
            )

            conversation.add_message(result_message)

            # Process function result to get final response
            if self._processor:
                return await self._processor.process(conversation, result_message)

        except Exception as e:
            logger.error(f"Function call error: {e}")

        return message

    async def end_conversation(
        self,
        conversation: Conversation,
        reason: str = "normal",
    ) -> None:
        """End conversation."""
        conversation.state = ConversationState.ENDED
        conversation.ended_at = datetime.utcnow()
        conversation.metadata["end_reason"] = reason

        await self._emit_event(
            ConversationEventType.ENDED,
            conversation,
            {"reason": reason, "duration": conversation.duration_seconds},
        )

    async def _emit_event(
        self,
        event_type: ConversationEventType,
        conversation: Conversation,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit conversation event."""
        event = ConversationEvent(
            event_type=event_type,
            conversation_id=conversation.conversation_id,
            data=data or {},
        )

        for handler in self._event_handlers:
            try:
                await handler.handle(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "messages_processed": self._messages_processed,
            "responses_generated": self._responses_generated,
            "registered_functions": len(self._functions),
        }


class ConversationManager:
    """
    Manages multiple conversations.
    """

    def __init__(
        self,
        engine: Optional[ConversationEngine] = None,
        config: Optional[ConversationConfig] = None,
    ):
        self.engine = engine or ConversationEngine(config)
        self.config = config or ConversationConfig()

        # Storage
        self._conversations: Dict[str, Conversation] = {}
        self._by_call_id: Dict[str, str] = {}
        self._by_user_id: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()

        # Background tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start manager."""
        if self._running:
            return

        self._running = True
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))

    async def stop(self) -> None:
        """Stop manager."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

    async def create_conversation(
        self,
        tenant_id: str,
        agent_id: str,
        call_id: Optional[str] = None,
        user_id: Optional[str] = None,
        system_prompt: str = "",
        **kwargs,
    ) -> Conversation:
        """Create new conversation."""
        async with self._lock:
            conversation = Conversation(
                tenant_id=tenant_id,
                agent_id=agent_id,
                call_id=call_id,
                user_id=user_id,
                system_prompt=system_prompt,
                **kwargs,
            )

            self._conversations[conversation.conversation_id] = conversation

            if call_id:
                self._by_call_id[call_id] = conversation.conversation_id

            if user_id:
                if user_id not in self._by_user_id:
                    self._by_user_id[user_id] = []
                self._by_user_id[user_id].append(conversation.conversation_id)

            return conversation

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        return self._conversations.get(conversation_id)

    async def get_by_call_id(self, call_id: str) -> Optional[Conversation]:
        """Get conversation by call ID."""
        conv_id = self._by_call_id.get(call_id)
        if conv_id:
            return self._conversations.get(conv_id)
        return None

    async def get_user_conversations(self, user_id: str) -> List[Conversation]:
        """Get conversations for user."""
        conv_ids = self._by_user_id.get(user_id, [])
        return [
            self._conversations[cid]
            for cid in conv_ids
            if cid in self._conversations
        ]

    async def process_message(
        self,
        conversation_id: str,
        content: str,
        role: MessageRole = MessageRole.USER,
        **kwargs,
    ) -> Optional[Message]:
        """Process message in conversation."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return None

        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            **kwargs,
        )

        return await self.engine.process_message(conversation, message)

    async def end_conversation(
        self,
        conversation_id: str,
        reason: str = "normal",
    ) -> bool:
        """End conversation."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return False

        await self.engine.end_conversation(conversation, reason)
        return True

    async def _cleanup_loop(self) -> None:
        """Clean up old conversations."""
        while self._running:
            try:
                await asyncio.sleep(60)

                now = datetime.utcnow()
                to_remove = []

                async with self._lock:
                    for conv_id, conv in self._conversations.items():
                        # Check ended conversations
                        if conv.state == ConversationState.ENDED and conv.ended_at:
                            age = (now - conv.ended_at).total_seconds()
                            if age > 3600:  # 1 hour
                                to_remove.append(conv_id)

                        # Check idle timeout
                        if conv.state in [ConversationState.WAITING_USER, ConversationState.ACTIVE]:
                            last_message = conv.get_last_message()
                            if last_message:
                                idle_time = (now - last_message.timestamp).total_seconds()
                                if idle_time > self.config.idle_timeout_seconds:
                                    await self.engine.end_conversation(conv, "idle_timeout")

                    for conv_id in to_remove:
                        conv = self._conversations.pop(conv_id, None)
                        if conv:
                            if conv.call_id:
                                self._by_call_id.pop(conv.call_id, None)
                            if conv.user_id and conv.user_id in self._by_user_id:
                                if conv_id in self._by_user_id[conv.user_id]:
                                    self._by_user_id[conv.user_id].remove(conv_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        states: Dict[str, int] = {}
        for conv in self._conversations.values():
            state = conv.state.value
            states[state] = states.get(state, 0) + 1

        return {
            "total_conversations": len(self._conversations),
            "states": states,
            "engine": self.engine.get_stats(),
        }
