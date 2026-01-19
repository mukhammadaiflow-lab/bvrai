"""
Conversation Engine

The main orchestrator that combines all conversation components
into a cohesive conversation management system.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from .base import (
    AgentResponse,
    ConversationConfig,
    ConversationEvent,
    ConversationEventType,
    ConversationState,
    ConversationTurn,
    ResponseType,
    TurnState,
    UserInput,
)
from .context import ConversationContext, ContextManager, ContextScope
from .intents import Intent, IntentMatch, IntentRouter, StaticIntentMatcher
from .slots import Slot, SlotFiller, SlotValue
from .flows import DialogFlow, DialogNode, NodeType, FlowAction
from .state import StateMachine, create_default_state_machine


logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the conversation engine."""

    # Conversation settings
    conversation: ConversationConfig = field(default_factory=ConversationConfig)

    # LLM settings
    use_llm_fallback: bool = True
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7

    # Knowledge base
    use_knowledge_base: bool = False
    knowledge_top_k: int = 3

    # Response generation
    max_response_length: int = 500
    include_filler_responses: bool = True

    # Flow execution
    max_flow_depth: int = 5
    max_consecutive_actions: int = 10

    # Metrics
    track_metrics: bool = True


@dataclass
class TurnResult:
    """Result of processing a conversation turn."""

    # Response
    response: AgentResponse

    # State changes
    state_changed: bool = False
    new_state: Optional[ConversationState] = None

    # Flow changes
    flow_changed: bool = False
    current_flow_id: Optional[str] = None
    current_node_id: Optional[str] = None

    # Intent/Slot info
    detected_intent: Optional[str] = None
    filled_slots: Dict[str, Any] = field(default_factory=dict)

    # Timing
    processing_time_ms: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationSession:
    """
    Represents an active conversation session.
    """

    def __init__(
        self,
        session_id: str,
        context: ConversationContext,
        engine: "ConversationEngine",
    ):
        self.session_id = session_id
        self.context = context
        self._engine = engine

        # State machine
        self._state_machine = create_default_state_machine()

        # Current flow
        self._current_flow: Optional[DialogFlow] = None
        self._current_node: Optional[DialogNode] = None

        # Turn tracking
        self._current_turn: Optional[ConversationTurn] = None
        self._turn_count = 0

        # Timing
        self.created_at = time.time()
        self.last_activity = time.time()

    @property
    def state(self) -> ConversationState:
        """Get current state."""
        return self._state_machine.current_state

    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return not self._state_machine.is_terminal

    async def start(self) -> AgentResponse:
        """Start the conversation session."""
        # Transition to greeting state
        await self._state_machine.transition(
            ConversationState.GREETING,
            self.context,
        )

        # Get greeting
        config = self._engine.config.conversation
        if config.enable_greeting:
            greeting = config.greeting_message or f"Hello! I'm {config.agent_name}. How can I help you today?"
            response = AgentResponse(
                text=greeting,
                response_type=ResponseType.TEXT,
                expects_response=True,
            )
        else:
            response = AgentResponse(
                text="",
                response_type=ResponseType.SILENT,
                expects_response=True,
            )

        # Transition to listening
        await self._state_machine.transition(
            ConversationState.LISTENING,
            self.context,
        )

        # Emit event
        self._emit_event(ConversationEventType.SESSION_STARTED)

        return response

    async def process_input(self, user_input: UserInput) -> TurnResult:
        """Process user input and generate response."""
        start_time = time.time()
        self.last_activity = start_time

        # Create new turn
        self._turn_count += 1
        self._current_turn = ConversationTurn(user_input=user_input)

        # Emit turn started
        self._emit_event(ConversationEventType.TURN_STARTED, {
            "turn_id": self._current_turn.turn_id,
            "user_text": user_input.text,
        })

        # Transition to processing
        await self._state_machine.transition(
            ConversationState.PROCESSING,
            self.context,
        )

        try:
            # Process based on current state
            result = await self._process_turn(user_input)

            # Update turn
            self._current_turn.response = result.response
            self._current_turn.detected_intent = result.detected_intent
            self._current_turn.complete()

            # Add to context history
            self.context.add_turn(
                user_text=user_input.text,
                agent_text=result.response.text,
            )

            # Transition to appropriate state
            if result.response.response_type == ResponseType.END:
                await self._state_machine.transition(
                    ConversationState.COMPLETED,
                    self.context,
                )
            elif result.response.response_type == ResponseType.TRANSFER:
                await self._state_machine.transition(
                    ConversationState.TRANSFERRING,
                    self.context,
                )
            elif result.response.expects_response:
                await self._state_machine.transition(
                    ConversationState.LISTENING,
                    self.context,
                )
            else:
                await self._state_machine.transition(
                    ConversationState.WAITING_INPUT,
                    self.context,
                )

            # Calculate timing
            result.processing_time_ms = (time.time() - start_time) * 1000

            # Emit turn completed
            self._emit_event(ConversationEventType.TURN_COMPLETED, {
                "turn_id": self._current_turn.turn_id,
                "response_text": result.response.text,
                "processing_time_ms": result.processing_time_ms,
            })

            return result

        except Exception as e:
            logger.error(f"Error processing turn: {e}")

            # Transition to error state
            await self._state_machine.transition(
                ConversationState.ERROR,
                self.context,
            )

            # Return error response
            error_response = AgentResponse(
                text=self._engine.config.conversation.error_message,
                response_type=ResponseType.TEXT,
                expects_response=True,
            )

            return TurnResult(
                response=error_response,
                state_changed=True,
                new_state=ConversationState.ERROR,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def _process_turn(self, user_input: UserInput) -> TurnResult:
        """Process a turn through the conversation logic."""
        text = user_input.text

        # Check if we're in a flow
        if self._current_flow and self._current_node:
            return await self._process_flow_turn(text)

        # Try intent matching
        intent_match, handler_result = await self._engine._intent_router.route(
            text,
            self.context,
        )

        if intent_match:
            self._current_turn.detected_intent = intent_match.name
            self._current_turn.intent_confidence = intent_match.confidence

            # Check if intent triggers a flow
            flow = self._engine.get_flow_for_intent(intent_match.name)
            if flow:
                return await self._enter_flow(flow, text, intent_match)

            # Use handler result if available
            if handler_result and handler_result.get("response"):
                return TurnResult(
                    response=AgentResponse(
                        text=handler_result["response"],
                        response_type=ResponseType.TEXT,
                    ),
                    detected_intent=intent_match.name,
                )

        # Try slot filling if we have pending slots
        if self.context.pending_slots:
            return await self._process_slot_filling(text)

        # Fall back to LLM response
        if self._engine.config.use_llm_fallback:
            return await self._generate_llm_response(text, intent_match)

        # Default fallback response
        return TurnResult(
            response=AgentResponse(
                text=self._engine.config.conversation.fallback_message,
                response_type=ResponseType.TEXT,
            ),
        )

    async def _process_flow_turn(self, text: str) -> TurnResult:
        """Process turn within a flow."""
        node = self._current_node
        flow = self._current_flow

        if not node or not flow:
            return await self._generate_llm_response(text, None)

        # Handle based on node type
        if node.node_type == NodeType.QUESTION:
            # Try to match expected intents
            if node.expected_intents:
                match, _ = await self._engine._intent_router.route(
                    text,
                    self.context,
                    allowed_intents=node.expected_intents,
                )
                if match:
                    next_node_id = node.get_next_node(self.context, match.name)
                    return await self._navigate_to_node(next_node_id, match)

        elif node.node_type == NodeType.SLOT_FILL:
            if node.slot:
                # Try to fill the slot
                filled = self._engine._slot_filler.fill(
                    text,
                    [node.slot],
                    self.context,
                )

                if node.slot.name in filled:
                    slot_value = filled[node.slot.name]
                    if slot_value.is_valid:
                        self.context.set_slot(node.slot.name, slot_value.normalized_value)
                        next_node_id = node.get_next_node(self.context)
                        return await self._navigate_to_node(next_node_id)
                    else:
                        # Validation failed
                        return TurnResult(
                            response=AgentResponse(
                                text=slot_value.validation_error or node.slot.reprompt,
                                response_type=ResponseType.TEXT,
                            ),
                            current_flow_id=flow.flow_id,
                            current_node_id=node.node_id,
                        )

        # Check for global intents that can interrupt
        match, _ = await self._engine._intent_router.route(text, self.context)
        if match and match.intent.cancels_current_flow:
            # Exit flow
            await self._exit_flow()
            # Handle the intent
            handler_result = await self._engine._intent_router.route(text, self.context)
            if handler_result[1] and handler_result[1].get("response"):
                return TurnResult(
                    response=AgentResponse(
                        text=handler_result[1]["response"],
                        response_type=ResponseType.TEXT,
                    ),
                    detected_intent=match.name,
                    flow_changed=True,
                )

        # Default: move to next node
        next_node_id = node.get_next_node(self.context)
        return await self._navigate_to_node(next_node_id)

    async def _enter_flow(
        self,
        flow: DialogFlow,
        text: str,
        intent_match: Optional[IntentMatch] = None,
    ) -> TurnResult:
        """Enter a dialog flow."""
        self._current_flow = flow
        self.context.push_flow(flow.flow_id, flow.start_node_id)

        # Execute entry actions
        for action in flow.on_enter:
            await action.execute(self.context)

        # Emit flow started
        self._emit_event(ConversationEventType.FLOW_STARTED, {
            "flow_id": flow.flow_id,
            "trigger_intent": intent_match.name if intent_match else None,
        })

        # Get first node
        start_node = flow.get_start_node()
        if start_node:
            return await self._navigate_to_node(start_node.node_id)

        # Flow has no nodes
        await self._exit_flow()
        return TurnResult(
            response=AgentResponse(
                text="",
                response_type=ResponseType.SILENT,
            ),
        )

    async def _exit_flow(self) -> None:
        """Exit current flow."""
        if self._current_flow:
            # Execute exit actions
            for action in self._current_flow.on_exit:
                await action.execute(self.context)

            # Emit flow completed
            self._emit_event(ConversationEventType.FLOW_COMPLETED, {
                "flow_id": self._current_flow.flow_id,
            })

        # Pop from stack
        prev = self.context.pop_flow()

        if prev:
            # Return to previous flow
            self._current_flow = self._engine.get_flow(prev["flow_id"])
            if self._current_flow:
                self._current_node = self._current_flow.get_node(prev["node_id"])
        else:
            self._current_flow = None
            self._current_node = None

    async def _navigate_to_node(
        self,
        node_id: Optional[str],
        intent_match: Optional[IntentMatch] = None,
    ) -> TurnResult:
        """Navigate to a flow node and generate response."""
        if not node_id or not self._current_flow:
            await self._exit_flow()
            return TurnResult(
                response=AgentResponse(text="", response_type=ResponseType.SILENT),
                flow_changed=True,
            )

        node = self._current_flow.get_node(node_id)
        if not node:
            await self._exit_flow()
            return TurnResult(
                response=AgentResponse(text="", response_type=ResponseType.SILENT),
                flow_changed=True,
            )

        self._current_node = node
        self.context.current_node_id = node_id

        # Emit node entered
        self._emit_event(ConversationEventType.FLOW_NODE_ENTERED, {
            "flow_id": self._current_flow.flow_id,
            "node_id": node_id,
            "node_type": node.node_type.value,
        })

        # Process based on node type
        if node.node_type == NodeType.MESSAGE:
            response = AgentResponse(
                text=node.get_message(),
                ssml=node.ssml,
                response_type=ResponseType.TEXT,
                expects_response=node.wait_for_response,
            )

            if not node.wait_for_response:
                # Auto-advance to next node
                next_node_id = node.get_next_node(self.context)
                if next_node_id:
                    return await self._navigate_to_node(next_node_id)

            return TurnResult(
                response=response,
                current_flow_id=self._current_flow.flow_id,
                current_node_id=node_id,
            )

        elif node.node_type == NodeType.QUESTION:
            return TurnResult(
                response=AgentResponse(
                    text=node.get_message(),
                    ssml=node.ssml,
                    response_type=ResponseType.TEXT,
                    expects_response=True,
                ),
                current_flow_id=self._current_flow.flow_id,
                current_node_id=node_id,
            )

        elif node.node_type == NodeType.SLOT_FILL:
            if node.slot:
                self.context.add_pending_slot(node.slot.name)
                prompt = self._engine._slot_filler.get_prompt_for_slot(
                    node.slot,
                    self.context,
                )
                return TurnResult(
                    response=AgentResponse(
                        text=prompt,
                        response_type=ResponseType.TEXT,
                        expects_response=True,
                    ),
                    current_flow_id=self._current_flow.flow_id,
                    current_node_id=node_id,
                )

        elif node.node_type == NodeType.CONDITION:
            # Evaluate condition and move to appropriate node
            next_node_id = node.get_next_node(self.context)
            return await self._navigate_to_node(next_node_id)

        elif node.node_type == NodeType.ACTION:
            # Execute actions
            for action in node.actions:
                try:
                    result = await action.execute(self.context)
                    if action.result_variable:
                        self.context.set(action.result_variable, result)
                except Exception as e:
                    logger.error(f"Action execution error: {e}")

            # Move to next node
            next_node_id = node.get_next_node(self.context)
            return await self._navigate_to_node(next_node_id)

        elif node.node_type == NodeType.TRANSFER:
            return TurnResult(
                response=AgentResponse(
                    text=node.transfer_message or "Transferring you now...",
                    response_type=ResponseType.TRANSFER,
                    transfer_target=node.transfer_target,
                ),
                current_flow_id=self._current_flow.flow_id,
                current_node_id=node_id,
            )

        elif node.node_type == NodeType.END:
            message = node.get_message()
            await self._exit_flow()
            return TurnResult(
                response=AgentResponse(
                    text=message,
                    response_type=ResponseType.END if not self._current_flow else ResponseType.TEXT,
                ),
                flow_changed=True,
            )

        # Default
        return TurnResult(
            response=AgentResponse(text="", response_type=ResponseType.SILENT),
            current_flow_id=self._current_flow.flow_id,
            current_node_id=node_id,
        )

    async def _process_slot_filling(self, text: str) -> TurnResult:
        """Process slot filling."""
        # Get pending slots
        pending_slot_names = self.context.pending_slots

        # Get slot definitions
        slots = [
            self._engine.get_slot(name)
            for name in pending_slot_names
            if self._engine.get_slot(name)
        ]

        if not slots:
            self.context.pending_slots.clear()
            return await self._generate_llm_response(text, None)

        # Try to fill slots
        filled = self._engine._slot_filler.fill(text, slots, self.context)

        for slot_name, slot_value in filled.items():
            if slot_value.is_valid:
                self.context.set_slot(slot_name, slot_value.normalized_value)

        # Get next unfilled required slot
        next_slot = self._engine._slot_filler.get_next_required_slot(slots, self.context)

        if next_slot:
            attempts = self._engine._slot_filler.record_attempt(
                next_slot.name,
                self.context.session_id,
            )
            is_reprompt = attempts > 1

            prompt = self._engine._slot_filler.get_prompt_for_slot(
                next_slot,
                self.context,
                is_reprompt=is_reprompt,
            )

            return TurnResult(
                response=AgentResponse(
                    text=prompt,
                    response_type=ResponseType.TEXT,
                ),
                filled_slots={k: v.normalized_value for k, v in filled.items()},
            )

        # All slots filled
        self.context.pending_slots.clear()
        return TurnResult(
            response=AgentResponse(
                text="Thank you, I have all the information I need.",
                response_type=ResponseType.TEXT,
            ),
            filled_slots={k: v.normalized_value for k, v in filled.items()},
        )

    async def _generate_llm_response(
        self,
        text: str,
        intent_match: Optional[IntentMatch],
    ) -> TurnResult:
        """Generate response using LLM."""
        llm_provider = self._engine._llm_provider

        if not llm_provider:
            return TurnResult(
                response=AgentResponse(
                    text=self._engine.config.conversation.fallback_message,
                    response_type=ResponseType.TEXT,
                ),
            )

        try:
            # Build messages
            from ..llm.base import LLMMessage, LLMConfig

            messages = []

            # System prompt
            system_prompt = self._build_system_prompt()
            messages.append(LLMMessage.system(system_prompt))

            # Conversation history
            history = self.context.get_conversation_history(max_turns=10, format="messages")
            for msg in history:
                if msg["role"] == "user":
                    messages.append(LLMMessage.user(msg["content"]))
                else:
                    messages.append(LLMMessage.assistant(msg["content"]))

            # Current input
            messages.append(LLMMessage.user(text))

            # Generate response
            response = await llm_provider.complete(
                messages=messages,
                config=LLMConfig(
                    model=self._engine.config.llm_model,
                    temperature=self._engine.config.llm_temperature,
                    max_tokens=self._engine.config.max_response_length,
                ),
            )

            return TurnResult(
                response=AgentResponse(
                    text=response.content,
                    response_type=ResponseType.TEXT,
                    source="llm",
                ),
                detected_intent=intent_match.name if intent_match else None,
            )

        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return TurnResult(
                response=AgentResponse(
                    text=self._engine.config.conversation.fallback_message,
                    response_type=ResponseType.TEXT,
                ),
            )

    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM."""
        config = self._engine.config.conversation
        parts = []

        parts.append(f"You are {config.agent_name}, a helpful voice assistant.")

        if config.agent_persona:
            parts.append(config.agent_persona)

        # Add slot context if available
        if self.context.slots:
            parts.append(f"Known information about the user: {self.context.slots}")

        # Add any custom context
        custom = self.context.get_all(scope=ContextScope.SESSION)
        if custom:
            parts.append(f"Session context: {custom}")

        parts.append("Keep responses concise and conversational, suitable for voice.")

        return "\n\n".join(parts)

    async def end(self, reason: str = "completed") -> AgentResponse:
        """End the conversation session."""
        # Transition to appropriate terminal state
        if reason == "transferred":
            await self._state_machine.transition(
                ConversationState.TRANSFERRED,
                self.context,
            )
        elif reason == "abandoned":
            await self._state_machine.transition(
                ConversationState.ABANDONED,
                self.context,
            )
        else:
            await self._state_machine.transition(
                ConversationState.COMPLETED,
                self.context,
            )

        # Emit session ended
        self._emit_event(ConversationEventType.SESSION_ENDED, {
            "reason": reason,
            "turn_count": self._turn_count,
            "duration_seconds": time.time() - self.created_at,
        })

        return AgentResponse(
            text="Goodbye!",
            response_type=ResponseType.END,
            expects_response=False,
        )

    def _emit_event(
        self,
        event_type: ConversationEventType,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a conversation event."""
        event = ConversationEvent(
            type=event_type,
            data=data or {},
            session_id=self.session_id,
            turn_id=self._current_turn.turn_id if self._current_turn else None,
            flow_id=self._current_flow.flow_id if self._current_flow else None,
            node_id=self._current_node.node_id if self._current_node else None,
        )

        # Notify engine listeners
        for listener in self._engine._event_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")


class ConversationEngine:
    """
    Main conversation engine that orchestrates all components.
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        llm_provider: Optional[Any] = None,
        knowledge_retriever: Optional[Any] = None,
    ):
        self.config = config or EngineConfig()

        # Components
        self._llm_provider = llm_provider
        self._knowledge_retriever = knowledge_retriever

        # Intent handling
        self._intent_router = IntentRouter(matcher=StaticIntentMatcher())

        # Slot filling
        self._slot_filler = SlotFiller()
        self._slots: Dict[str, Slot] = {}

        # Flows
        self._flows: Dict[str, DialogFlow] = {}
        self._intent_to_flow: Dict[str, str] = {}

        # Context management
        self._context_manager = ContextManager()

        # Sessions
        self._sessions: Dict[str, ConversationSession] = {}

        # Event listeners
        self._event_listeners: List[Callable[[ConversationEvent], None]] = []

    def register_intent(
        self,
        intent: Intent,
        handler: Optional[Callable] = None,
        flow_id: Optional[str] = None,
        is_global: bool = False,
    ) -> None:
        """Register an intent with optional handler or flow."""
        self._intent_router.register(intent, handler, flow_id, is_global)

        if flow_id:
            self._intent_to_flow[intent.name] = flow_id

    def register_slot(self, slot: Slot) -> None:
        """Register a slot definition."""
        self._slots[slot.name] = slot

    def get_slot(self, name: str) -> Optional[Slot]:
        """Get a slot by name."""
        return self._slots.get(name)

    def register_flow(self, flow: DialogFlow) -> None:
        """Register a dialog flow."""
        self._flows[flow.flow_id] = flow

        # Map trigger intents to flow
        for intent_name in flow.trigger_intents:
            self._intent_to_flow[intent_name] = flow.flow_id

    def get_flow(self, flow_id: str) -> Optional[DialogFlow]:
        """Get a flow by ID."""
        return self._flows.get(flow_id)

    def get_flow_for_intent(self, intent_name: str) -> Optional[DialogFlow]:
        """Get the flow triggered by an intent."""
        flow_id = self._intent_to_flow.get(intent_name)
        if flow_id:
            return self._flows.get(flow_id)
        return None

    def add_event_listener(
        self,
        listener: Callable[[ConversationEvent], None],
    ) -> None:
        """Add an event listener."""
        self._event_listeners.append(listener)

    async def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        channel: str = "voice",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationSession:
        """Create a new conversation session."""
        session_id = session_id or str(uuid.uuid4())

        # Create context
        context = self._context_manager.create(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            channel=channel,
        )

        if metadata:
            context.custom.update(metadata)

        # Create session
        session = ConversationSession(
            session_id=session_id,
            context=context,
            engine=self,
        )

        self._sessions[session_id] = session

        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get an existing session."""
        return self._sessions.get(session_id)

    async def end_session(
        self,
        session_id: str,
        reason: str = "completed",
    ) -> None:
        """End a session."""
        session = self._sessions.get(session_id)
        if session:
            await session.end(reason)
            del self._sessions[session_id]

    def cleanup_sessions(self, max_idle_seconds: int = 1800) -> int:
        """Clean up idle sessions."""
        now = time.time()
        expired = []

        for session_id, session in self._sessions.items():
            if now - session.last_activity > max_idle_seconds:
                expired.append(session_id)

        for session_id in expired:
            del self._sessions[session_id]

        return len(expired)


__all__ = [
    "EngineConfig",
    "TurnResult",
    "ConversationSession",
    "ConversationEngine",
]
