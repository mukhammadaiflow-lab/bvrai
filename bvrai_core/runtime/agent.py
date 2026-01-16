"""
Agent Runtime Module

This module provides the main agent runtime that coordinates
all components for live agent execution during calls.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
)

from .base import (
    AgentState,
    ResponseType,
    IntentCategory,
    SentimentLevel,
    AgentPersona,
    AgentCapabilities,
    ConversationContext,
    Message,
    FunctionDefinition,
    FunctionCall,
    AgentResponse,
    AgentConfig,
    RuntimeMetrics,
    KnowledgeChunk,
    AgentExecutionError,
)
from .context import (
    ConversationMemory,
    ConversationTracker,
    TiktokenCounter,
)
from .executor import (
    ExecutionConfig,
    ExecutionResult,
    LLMExecutor,
)
from .tools import (
    ToolRegistry,
    ToolResult,
    create_default_tools,
)
from .knowledge import (
    KnowledgeRetriever,
    KnowledgeBaseConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class AgentRuntimeConfig:
    """Configuration for agent runtime."""

    # Agent configuration
    agent_config: AgentConfig

    # LLM settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 150

    # Response settings
    max_response_time_seconds: float = 30.0
    stream_response: bool = True

    # Knowledge settings
    enable_knowledge_retrieval: bool = True
    knowledge_base_ids: List[str] = field(default_factory=list)
    max_knowledge_chunks: int = 5

    # Tool settings
    enable_tools: bool = True
    max_tool_calls_per_turn: int = 5
    tool_timeout_seconds: float = 30.0

    # Behavior settings
    enable_intent_detection: bool = True
    enable_sentiment_analysis: bool = True
    enable_entity_extraction: bool = True


class AgentRuntime:
    """
    Main agent runtime for live call execution.

    Coordinates:
    - Conversation memory management
    - LLM execution with streaming
    - Tool execution
    - Knowledge retrieval
    - Intent and sentiment analysis
    """

    def __init__(
        self,
        config: AgentRuntimeConfig,
        tool_registry: Optional[ToolRegistry] = None,
        knowledge_retriever: Optional[KnowledgeRetriever] = None,
        llm_executor: Optional[LLMExecutor] = None,
    ):
        """
        Initialize agent runtime.

        Args:
            config: Runtime configuration
            tool_registry: Tool registry
            knowledge_retriever: Knowledge retriever
            llm_executor: LLM executor
        """
        self.config = config
        self.agent_config = config.agent_config

        # State
        self.state = AgentState.IDLE
        self.context: Optional[ConversationContext] = None

        # Components
        self.memory = ConversationMemory(
            context_window_tokens=8000,
            token_counter=TiktokenCounter(config.llm_model),
        )
        self.tracker = ConversationTracker()

        self.tools = tool_registry or create_default_tools(
            transfer_targets=config.agent_config.capabilities.transfer_targets
        )

        self.knowledge = knowledge_retriever or KnowledgeRetriever()
        self.executor = llm_executor or LLMExecutor(config.llm_provider)

        # Metrics
        self.metrics = RuntimeMetrics()

        # Callbacks
        self._response_callbacks: List[
            Callable[[str], Coroutine[Any, Any, None]]
        ] = []
        self._state_callbacks: List[
            Callable[[AgentState], Coroutine[Any, Any, None]]
        ] = []

    async def initialize(
        self,
        session_id: str,
        organization_id: str,
        caller_id: Optional[str] = None,
        context_variables: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize runtime for a new call.

        Args:
            session_id: Call session ID
            organization_id: Organization ID
            caller_id: Caller's phone number
            context_variables: Initial context variables
        """
        # Create conversation context
        self.context = ConversationContext(
            session_id=session_id,
            organization_id=organization_id,
            agent_id=self.agent_config.agent_id,
            caller_id=caller_id,
            variables=context_variables or {},
        )

        # Build and set system prompt
        system_prompt = self._build_system_prompt()
        self.memory.set_system_prompt(system_prompt)

        # Reset state
        self.state = AgentState.IDLE
        self.metrics = RuntimeMetrics()

        logger.info(
            f"Agent runtime initialized for session {session_id} "
            f"(agent: {self.agent_config.name})"
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        persona = self.agent_config.persona
        capabilities = self.agent_config.capabilities

        prompt_parts = []

        # Base identity
        prompt_parts.append(f"You are {persona.name}, a {persona.role} at {persona.company_name}.")

        # Personality
        prompt_parts.append(f"Your tone is {persona.tone} and {persona.speaking_style}.")

        # Industry context
        if persona.industry:
            prompt_parts.append(f"You specialize in {persona.industry}.")

        # Add custom system prompt
        if self.agent_config.system_prompt:
            prompt_parts.append(self.agent_config.system_prompt)

        # Behavioral guidelines
        guidelines = [
            "Keep responses concise and natural for voice conversation.",
            "Use a conversational tone as if speaking on the phone.",
            "Ask clarifying questions when needed.",
        ]

        if persona.acknowledge_emotions:
            guidelines.append("Acknowledge the customer's emotions and show empathy.")

        if capabilities.can_transfer:
            guidelines.append(
                "If you cannot help, offer to transfer to the appropriate department."
            )

        prompt_parts.append("\n".join(f"- {g}" for g in guidelines))

        return "\n\n".join(prompt_parts)

    async def get_initial_greeting(self) -> str:
        """Get the initial greeting message."""
        if self.agent_config.first_message:
            return self.agent_config.first_message

        persona = self.agent_config.persona

        if persona.greeting_style == "warm":
            return (
                f"Hi there! This is {persona.name} from {persona.company_name}. "
                f"How can I help you today?"
            )
        elif persona.greeting_style == "formal":
            return (
                f"Good day. You've reached {persona.company_name}. "
                f"My name is {persona.name}. How may I assist you?"
            )
        else:
            return (
                f"Hello, {persona.company_name}, this is {persona.name}. "
                f"What can I do for you?"
            )

    async def process_input(
        self,
        user_input: str,
    ) -> AgentResponse:
        """
        Process user input and generate response.

        Args:
            user_input: User's spoken text

        Returns:
            Agent response
        """
        start_time = time.time()
        await self._set_state(AgentState.THINKING)

        try:
            # Analyze input
            if self.config.enable_intent_detection:
                intent, _ = self.tracker.detect_intent(user_input)
                if self.context:
                    self.context.add_intent(intent)

            if self.config.enable_sentiment_analysis:
                sentiment, _ = self.tracker.analyze_sentiment(user_input)
                if self.context:
                    self.context.update_sentiment(sentiment)

            if self.config.enable_entity_extraction:
                entities = self.tracker.extract_entities(user_input)
                if self.context and entities:
                    for entity_type, values in entities.items():
                        self.context.collected_information[entity_type] = values

            # Add user message to memory
            self.memory.add_user_message(user_input)

            # Retrieve relevant knowledge
            knowledge_chunks = []
            if self.config.enable_knowledge_retrieval and self.config.knowledge_base_ids:
                knowledge_chunks = await self._retrieve_knowledge(user_input)

            # Build messages for LLM
            messages = self.memory.get_context_for_llm(knowledge_chunks)

            # Get tool definitions
            tool_definitions = None
            if self.config.enable_tools:
                tool_definitions = self.tools.get_definitions()

            # Execute LLM
            execution_config = ExecutionConfig(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout_seconds=self.config.max_response_time_seconds,
                stream=self.config.stream_response,
            )

            result = await self.executor.execute(
                messages=messages,
                config=execution_config,
                tools=tool_definitions,
            )

            # Handle function calls
            if result.function_calls:
                response = await self._handle_function_calls(
                    result.function_calls,
                    messages,
                    execution_config,
                    tool_definitions,
                )
            else:
                response = AgentResponse(
                    type=ResponseType.TEXT,
                    content=result.content,
                    thinking_time_ms=result.latency_ms,
                    tokens_used=result.total_tokens,
                    model_used=result.model,
                )

            # Add assistant response to memory
            if response.content:
                self.memory.add_assistant_message(response.content)

            # Update metrics
            thinking_time = (time.time() - start_time) * 1000
            self.metrics.add_turn_metrics(
                thinking_time_ms=thinking_time,
                response_time_ms=result.latency_ms,
                input_tokens=result.prompt_tokens,
                output_tokens=result.completion_tokens,
            )

            await self._set_state(AgentState.SPEAKING)
            return response

        except Exception as e:
            logger.exception(f"Error processing input: {e}")
            await self._set_state(AgentState.ERROR)
            raise AgentExecutionError(f"Failed to process input: {e}")

    async def stream_response(
        self,
        user_input: str,
    ) -> AsyncIterator[Tuple[str, Optional[AgentResponse]]]:
        """
        Stream response to user input.

        Args:
            user_input: User's spoken text

        Yields:
            Tuples of (text_chunk, final_response)
        """
        start_time = time.time()
        await self._set_state(AgentState.THINKING)

        try:
            # Analyze input
            if self.config.enable_intent_detection:
                intent, _ = self.tracker.detect_intent(user_input)
                if self.context:
                    self.context.add_intent(intent)

            if self.config.enable_sentiment_analysis:
                sentiment, _ = self.tracker.analyze_sentiment(user_input)
                if self.context:
                    self.context.update_sentiment(sentiment)

            # Add user message
            self.memory.add_user_message(user_input)

            # Retrieve knowledge
            knowledge_chunks = []
            if self.config.enable_knowledge_retrieval and self.config.knowledge_base_ids:
                knowledge_chunks = await self._retrieve_knowledge(user_input)

            # Build messages
            messages = self.memory.get_context_for_llm(knowledge_chunks)

            # Get tools
            tool_definitions = None
            if self.config.enable_tools:
                tool_definitions = self.tools.get_definitions()

            # Execute with streaming
            execution_config = ExecutionConfig(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout_seconds=self.config.max_response_time_seconds,
            )

            full_content = ""
            final_result: Optional[ExecutionResult] = None

            await self._set_state(AgentState.SPEAKING)

            async for chunk, result in self.executor.stream(
                messages=messages,
                config=execution_config,
                tools=tool_definitions,
            ):
                if chunk:
                    full_content += chunk
                    for callback in self._response_callbacks:
                        await callback(chunk)
                    yield chunk, None

                if result:
                    final_result = result

            # Handle function calls if any
            if final_result and final_result.function_calls:
                # For streaming with function calls, we need to handle them
                response = await self._handle_function_calls(
                    final_result.function_calls,
                    messages,
                    execution_config,
                    tool_definitions,
                )
                yield "", response
            else:
                response = AgentResponse(
                    type=ResponseType.TEXT,
                    content=full_content,
                    thinking_time_ms=(time.time() - start_time) * 1000,
                    tokens_used=final_result.total_tokens if final_result else 0,
                    model_used=final_result.model if final_result else "",
                )

                # Add to memory
                if full_content:
                    self.memory.add_assistant_message(full_content)

                yield "", response

        except Exception as e:
            logger.exception(f"Error streaming response: {e}")
            await self._set_state(AgentState.ERROR)
            raise AgentExecutionError(f"Failed to stream response: {e}")

    async def _retrieve_knowledge(
        self,
        query: str,
    ) -> List[KnowledgeChunk]:
        """Retrieve relevant knowledge for query."""
        try:
            chunks = await self.knowledge.retrieve(
                query=query,
                knowledge_base_ids=self.config.knowledge_base_ids,
                top_k=self.config.max_knowledge_chunks,
            )

            self.metrics.knowledge_queries += 1
            if chunks:
                self.metrics.knowledge_hits += 1

            return chunks

        except Exception as e:
            logger.exception(f"Knowledge retrieval failed: {e}")
            return []

    async def _handle_function_calls(
        self,
        function_calls: List[FunctionCall],
        messages: List[Dict[str, Any]],
        config: ExecutionConfig,
        tools: Optional[List[FunctionDefinition]],
    ) -> AgentResponse:
        """Handle function calls from LLM."""
        await self._set_state(AgentState.EXECUTING_TOOL)

        all_results = []
        final_response = None

        # Limit function calls per turn
        calls_to_process = function_calls[:self.config.max_tool_calls_per_turn]

        for call in calls_to_process:
            # Execute tool
            result = await self.tools.execute(
                call,
                self.context,
                self.config.tool_timeout_seconds,
            )
            all_results.append(result)

            # Update metrics
            self.metrics.total_function_calls += 1
            if result.success:
                self.metrics.successful_function_calls += 1
            else:
                self.metrics.failed_function_calls += 1

            # Check for special actions
            if result.success and isinstance(result.result, dict):
                action = result.result.get("action")

                if action == "end_call":
                    return AgentResponse(
                        type=ResponseType.END_CALL,
                        content=result.result.get("reason", "Call ended"),
                    )

                elif action == "transfer_call":
                    return AgentResponse(
                        type=ResponseType.TRANSFER,
                        transfer_target=result.result.get("target_number"),
                        transfer_message=result.result.get("reason"),
                    )

                elif action == "hold_call":
                    return AgentResponse(
                        type=ResponseType.HOLD,
                        content=f"Please hold for approximately {result.result.get('estimated_wait_seconds', 60)} seconds.",
                    )

                elif action == "collect_dtmf":
                    return AgentResponse(
                        type=ResponseType.COLLECT_DTMF,
                        content=result.result.get("prompt", "Please enter your response."),
                    )

            # Add result to messages for follow-up
            result_content = (
                str(result.result) if result.success
                else f"Error: {result.error}"
            )
            self.memory.add_function_result(
                result_content,
                call.name,
                call.id,
            )

        # Get follow-up response from LLM
        await self._set_state(AgentState.THINKING)

        follow_up_messages = self.memory.get_context_for_llm()

        follow_up_result = await self.executor.execute(
            messages=follow_up_messages,
            config=config,
            tools=tools,
        )

        # Handle nested function calls (recursion limit)
        if follow_up_result.function_calls:
            # For simplicity, don't recurse more than once
            return AgentResponse(
                type=ResponseType.TEXT,
                content="I'm having trouble completing that request. Let me try a different approach.",
                function_calls=follow_up_result.function_calls,
            )

        self.memory.add_assistant_message(follow_up_result.content)

        return AgentResponse(
            type=ResponseType.TEXT,
            content=follow_up_result.content,
            function_calls=[FunctionCall(
                id=r.call_id,
                name=r.tool_name,
                arguments={},
                status="completed" if r.success else "failed",
                result=r.result,
                error=r.error,
            ) for r in all_results],
            tokens_used=follow_up_result.total_tokens,
            model_used=follow_up_result.model,
        )

    async def _set_state(self, state: AgentState) -> None:
        """Set agent state and notify callbacks."""
        if self.state != state:
            self.state = state
            for callback in self._state_callbacks:
                try:
                    await callback(state)
                except Exception as e:
                    logger.exception(f"State callback failed: {e}")

    def add_response_callback(
        self,
        callback: Callable[[str], Coroutine[Any, Any, None]],
    ) -> None:
        """Add callback for response chunks."""
        self._response_callbacks.append(callback)

    def add_state_callback(
        self,
        callback: Callable[[AgentState], Coroutine[Any, Any, None]],
    ) -> None:
        """Add callback for state changes."""
        self._state_callbacks.append(callback)

    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a custom tool."""
        self.tools.register_function(name, description, handler, parameters)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation."""
        return {
            "session_id": self.context.session_id if self.context else None,
            "message_count": self.memory.get_message_count(),
            "token_count": self.memory.get_token_count(),
            "dominant_intent": self.tracker.get_dominant_intent(),
            "current_sentiment": self.tracker.get_current_sentiment().value,
            "collected_entities": self.tracker.get_all_entities(),
            "metrics": {
                "total_turns": self.metrics.total_turns,
                "avg_thinking_time_ms": self.metrics.avg_thinking_time_ms,
                "total_tokens": self.metrics.total_tokens,
                "function_calls": self.metrics.total_function_calls,
                "knowledge_hits": self.metrics.knowledge_hits,
            },
        }

    def reset(self) -> None:
        """Reset runtime state."""
        self.state = AgentState.IDLE
        self.context = None
        self.memory.clear()
        self.metrics = RuntimeMetrics()


class AgentRuntimeBuilder:
    """Builder for creating agent runtimes."""

    def __init__(self, agent_config: AgentConfig):
        """
        Initialize builder.

        Args:
            agent_config: Agent configuration
        """
        self._config = AgentRuntimeConfig(agent_config=agent_config)
        self._tool_registry: Optional[ToolRegistry] = None
        self._knowledge_retriever: Optional[KnowledgeRetriever] = None
        self._llm_executor: Optional[LLMExecutor] = None

    def with_llm(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
    ) -> "AgentRuntimeBuilder":
        """Configure LLM settings."""
        self._config.llm_provider = provider
        self._config.llm_model = model
        self._config.api_key = api_key
        self._config.temperature = temperature
        return self

    def with_tools(
        self,
        tool_registry: ToolRegistry,
    ) -> "AgentRuntimeBuilder":
        """Set tool registry."""
        self._tool_registry = tool_registry
        return self

    def with_knowledge(
        self,
        knowledge_retriever: KnowledgeRetriever,
        knowledge_base_ids: Optional[List[str]] = None,
    ) -> "AgentRuntimeBuilder":
        """Configure knowledge retrieval."""
        self._knowledge_retriever = knowledge_retriever
        if knowledge_base_ids:
            self._config.knowledge_base_ids = knowledge_base_ids
        return self

    def with_streaming(self, enabled: bool = True) -> "AgentRuntimeBuilder":
        """Enable/disable streaming."""
        self._config.stream_response = enabled
        return self

    def with_max_tokens(self, max_tokens: int) -> "AgentRuntimeBuilder":
        """Set max response tokens."""
        self._config.max_tokens = max_tokens
        return self

    def build(self) -> AgentRuntime:
        """Build the agent runtime."""
        return AgentRuntime(
            config=self._config,
            tool_registry=self._tool_registry,
            knowledge_retriever=self._knowledge_retriever,
            llm_executor=self._llm_executor,
        )


def create_agent_runtime(
    agent_config: AgentConfig,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4",
    api_key: Optional[str] = None,
) -> AgentRuntime:
    """
    Create an agent runtime with default configuration.

    Args:
        agent_config: Agent configuration
        llm_provider: LLM provider name
        llm_model: LLM model name
        api_key: API key for LLM provider

    Returns:
        Configured agent runtime
    """
    return (
        AgentRuntimeBuilder(agent_config)
        .with_llm(llm_provider, llm_model, api_key)
        .build()
    )


__all__ = [
    "AgentRuntimeConfig",
    "AgentRuntime",
    "AgentRuntimeBuilder",
    "create_agent_runtime",
]
