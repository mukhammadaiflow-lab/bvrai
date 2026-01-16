"""
Builder Engine LLM Orchestration Layer

A sophisticated multi-provider LLM orchestration system that provides:
- Multi-provider support (OpenAI, Anthropic, Google, Cohere, etc.)
- Streaming response handling
- Function/tool calling framework
- Context window management
- Automatic fallback and load balancing
- Rate limiting and retry handling
- Cost tracking and optimization

Designed for high-availability, production-grade AI agent systems.
"""

from .base import (
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMUsage,
    LLMRole,
    LLMProvider,
    ProviderConfig,
    ModelInfo,
    StreamChunk,
)
from .providers import (
    BaseLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    CohereProvider,
    TogetherProvider,
    GroqProvider,
    LLMProviderFactory,
)
from .tools import (
    Tool,
    ToolParameter,
    ToolResult,
    ToolCall,
    ToolRegistry,
    tool,
)
from .context import (
    ContextManager,
    ContextWindow,
    MessagePriority,
    ContextStrategy,
)
from .orchestrator import (
    LLMOrchestrator,
    OrchestratorConfig,
    RoutingStrategy,
    FallbackConfig,
)
from .streaming import (
    StreamProcessor,
    StreamBuffer,
    StreamEvent,
    StreamEventType,
)

__all__ = [
    # Base
    "LLMConfig",
    "LLMMessage",
    "LLMResponse",
    "LLMUsage",
    "LLMRole",
    "LLMProvider",
    "ProviderConfig",
    "ModelInfo",
    "StreamChunk",
    # Providers
    "BaseLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "CohereProvider",
    "TogetherProvider",
    "GroqProvider",
    "LLMProviderFactory",
    # Tools
    "Tool",
    "ToolParameter",
    "ToolResult",
    "ToolCall",
    "ToolRegistry",
    "tool",
    # Context
    "ContextManager",
    "ContextWindow",
    "MessagePriority",
    "ContextStrategy",
    # Orchestrator
    "LLMOrchestrator",
    "OrchestratorConfig",
    "RoutingStrategy",
    "FallbackConfig",
    # Streaming
    "StreamProcessor",
    "StreamBuffer",
    "StreamEvent",
    "StreamEventType",
]
