"""
Agent Runtime Package

This package provides the complete agent runtime system for live AI voice
agent execution during calls, including:

- Conversation memory and context management
- LLM execution with streaming and multi-provider support
- Tool/function execution
- Knowledge base retrieval
- Intent detection and sentiment analysis

Example usage:

    from bvrai_core.runtime import (
        AgentConfig,
        AgentPersona,
        AgentCapabilities,
        create_agent_runtime,
    )

    # Create agent configuration
    config = AgentConfig(
        agent_id="agent_123",
        name="Sarah",
        persona=AgentPersona(
            name="Sarah",
            role="Customer Service Representative",
            company_name="Acme Corp",
            tone="friendly",
        ),
        capabilities=AgentCapabilities(
            can_transfer=True,
            can_schedule=True,
        ),
        system_prompt="You are a helpful customer service agent.",
        first_message="Hi! How can I help you today?",
    )

    # Create runtime
    runtime = create_agent_runtime(
        agent_config=config,
        llm_provider="openai",
        llm_model="gpt-4",
    )

    # Initialize for a call
    await runtime.initialize(
        session_id="call_abc123",
        organization_id="org_xyz",
        caller_id="+15551234567",
    )

    # Process user input
    response = await runtime.process_input("I need to schedule an appointment")
    print(response.content)
"""

# Base types
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
    KnowledgeQuery,
    KnowledgeResult,
    RuntimeError,
    AgentExecutionError,
    FunctionExecutionError,
    KnowledgeRetrievalError,
    TokenLimitExceededError,
)

# Context management
from .context import (
    TokenCounter,
    SimpleTokenCounter,
    TiktokenCounter,
    ContextWindow,
    ContextCompressor,
    ConversationMemory,
    ConversationTracker,
)

# LLM execution
from .executor import (
    ExecutionConfig,
    ExecutionResult,
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    MockProvider,
    LLMExecutor,
)

# Tool execution
from .tools import (
    ToolResult,
    Tool,
    EndCallTool,
    TransferCallTool,
    HoldCallTool,
    CollectDTMFTool,
    ScheduleAppointmentTool,
    SendSMSTool,
    SendEmailTool,
    LookupAccountTool,
    ToolRegistry,
    create_default_tools,
)

# Knowledge retrieval
from .knowledge import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    MockEmbeddingProvider,
    VectorStore,
    InMemoryVectorStore,
    PineconeVectorStore,
    KnowledgeBaseConfig,
    KnowledgeBase,
    KnowledgeRetriever,
)

# Main runtime
from .agent import (
    AgentRuntimeConfig,
    AgentRuntime,
    AgentRuntimeBuilder,
    create_agent_runtime,
)


__all__ = [
    # Base types
    "AgentState",
    "ResponseType",
    "IntentCategory",
    "SentimentLevel",
    "AgentPersona",
    "AgentCapabilities",
    "ConversationContext",
    "Message",
    "FunctionDefinition",
    "FunctionCall",
    "AgentResponse",
    "AgentConfig",
    "RuntimeMetrics",
    "KnowledgeChunk",
    "KnowledgeQuery",
    "KnowledgeResult",
    "RuntimeError",
    "AgentExecutionError",
    "FunctionExecutionError",
    "KnowledgeRetrievalError",
    "TokenLimitExceededError",
    # Context
    "TokenCounter",
    "SimpleTokenCounter",
    "TiktokenCounter",
    "ContextWindow",
    "ContextCompressor",
    "ConversationMemory",
    "ConversationTracker",
    # Executor
    "ExecutionConfig",
    "ExecutionResult",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "MockProvider",
    "LLMExecutor",
    # Tools
    "ToolResult",
    "Tool",
    "EndCallTool",
    "TransferCallTool",
    "HoldCallTool",
    "CollectDTMFTool",
    "ScheduleAppointmentTool",
    "SendSMSTool",
    "SendEmailTool",
    "LookupAccountTool",
    "ToolRegistry",
    "create_default_tools",
    # Knowledge
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "MockEmbeddingProvider",
    "VectorStore",
    "InMemoryVectorStore",
    "PineconeVectorStore",
    "KnowledgeBaseConfig",
    "KnowledgeBase",
    "KnowledgeRetriever",
    # Runtime
    "AgentRuntimeConfig",
    "AgentRuntime",
    "AgentRuntimeBuilder",
    "create_agent_runtime",
]
