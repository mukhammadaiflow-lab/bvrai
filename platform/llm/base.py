"""
LLM Base Classes and Data Structures

This module defines the foundational types and configurations
for the LLM orchestration layer.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)


class LLMRole(str, Enum):
    """Message role in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"  # Deprecated, use TOOL


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    TOGETHER = "together"
    GROQ = "groq"
    MISTRAL = "mistral"
    PERPLEXITY = "perplexity"
    FIREWORKS = "fireworks"
    ANYSCALE = "anyscale"
    OLLAMA = "ollama"  # Local models
    CUSTOM = "custom"


@dataclass
class ModelInfo:
    """Information about an LLM model."""

    # Identification
    model_id: str
    provider: LLMProvider
    display_name: str = ""

    # Capabilities
    context_window: int = 4096
    max_output_tokens: int = 4096
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    supports_json_mode: bool = False

    # Pricing (per 1K tokens, in USD)
    input_price_per_1k: float = 0.0
    output_price_per_1k: float = 0.0

    # Performance characteristics
    avg_latency_ms: float = 500.0
    tokens_per_second: float = 50.0

    # Metadata
    description: str = ""
    release_date: Optional[str] = None
    deprecated: bool = False

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.model_id

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for given token counts."""
        input_cost = (input_tokens / 1000) * self.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.output_price_per_1k
        return input_cost + output_cost


# Pre-defined model information
MODELS = {
    # OpenAI Models
    "gpt-4o": ModelInfo(
        model_id="gpt-4o",
        provider=LLMProvider.OPENAI,
        display_name="GPT-4o",
        context_window=128000,
        max_output_tokens=16384,
        supports_vision=True,
        supports_json_mode=True,
        input_price_per_1k=0.005,
        output_price_per_1k=0.015,
        avg_latency_ms=400,
        tokens_per_second=80,
    ),
    "gpt-4o-mini": ModelInfo(
        model_id="gpt-4o-mini",
        provider=LLMProvider.OPENAI,
        display_name="GPT-4o Mini",
        context_window=128000,
        max_output_tokens=16384,
        supports_vision=True,
        supports_json_mode=True,
        input_price_per_1k=0.00015,
        output_price_per_1k=0.0006,
        avg_latency_ms=200,
        tokens_per_second=120,
    ),
    "gpt-4-turbo": ModelInfo(
        model_id="gpt-4-turbo",
        provider=LLMProvider.OPENAI,
        display_name="GPT-4 Turbo",
        context_window=128000,
        max_output_tokens=4096,
        supports_vision=True,
        supports_json_mode=True,
        input_price_per_1k=0.01,
        output_price_per_1k=0.03,
        avg_latency_ms=500,
        tokens_per_second=60,
    ),
    "gpt-3.5-turbo": ModelInfo(
        model_id="gpt-3.5-turbo",
        provider=LLMProvider.OPENAI,
        display_name="GPT-3.5 Turbo",
        context_window=16385,
        max_output_tokens=4096,
        supports_json_mode=True,
        input_price_per_1k=0.0005,
        output_price_per_1k=0.0015,
        avg_latency_ms=200,
        tokens_per_second=100,
    ),

    # Anthropic Models
    "claude-3-5-sonnet-20241022": ModelInfo(
        model_id="claude-3-5-sonnet-20241022",
        provider=LLMProvider.ANTHROPIC,
        display_name="Claude 3.5 Sonnet",
        context_window=200000,
        max_output_tokens=8192,
        supports_vision=True,
        input_price_per_1k=0.003,
        output_price_per_1k=0.015,
        avg_latency_ms=350,
        tokens_per_second=90,
    ),
    "claude-3-5-haiku-20241022": ModelInfo(
        model_id="claude-3-5-haiku-20241022",
        provider=LLMProvider.ANTHROPIC,
        display_name="Claude 3.5 Haiku",
        context_window=200000,
        max_output_tokens=8192,
        supports_vision=True,
        input_price_per_1k=0.001,
        output_price_per_1k=0.005,
        avg_latency_ms=200,
        tokens_per_second=150,
    ),
    "claude-3-opus-20240229": ModelInfo(
        model_id="claude-3-opus-20240229",
        provider=LLMProvider.ANTHROPIC,
        display_name="Claude 3 Opus",
        context_window=200000,
        max_output_tokens=4096,
        supports_vision=True,
        input_price_per_1k=0.015,
        output_price_per_1k=0.075,
        avg_latency_ms=800,
        tokens_per_second=40,
    ),

    # Google Models
    "gemini-1.5-pro": ModelInfo(
        model_id="gemini-1.5-pro",
        provider=LLMProvider.GOOGLE,
        display_name="Gemini 1.5 Pro",
        context_window=2097152,  # 2M tokens
        max_output_tokens=8192,
        supports_vision=True,
        supports_json_mode=True,
        input_price_per_1k=0.00125,
        output_price_per_1k=0.005,
        avg_latency_ms=400,
        tokens_per_second=70,
    ),
    "gemini-1.5-flash": ModelInfo(
        model_id="gemini-1.5-flash",
        provider=LLMProvider.GOOGLE,
        display_name="Gemini 1.5 Flash",
        context_window=1048576,  # 1M tokens
        max_output_tokens=8192,
        supports_vision=True,
        supports_json_mode=True,
        input_price_per_1k=0.000075,
        output_price_per_1k=0.0003,
        avg_latency_ms=150,
        tokens_per_second=200,
    ),

    # Groq Models (fast inference)
    "llama-3.1-70b-versatile": ModelInfo(
        model_id="llama-3.1-70b-versatile",
        provider=LLMProvider.GROQ,
        display_name="Llama 3.1 70B (Groq)",
        context_window=131072,
        max_output_tokens=8192,
        input_price_per_1k=0.00059,
        output_price_per_1k=0.00079,
        avg_latency_ms=100,
        tokens_per_second=300,
    ),
    "llama-3.1-8b-instant": ModelInfo(
        model_id="llama-3.1-8b-instant",
        provider=LLMProvider.GROQ,
        display_name="Llama 3.1 8B (Groq)",
        context_window=131072,
        max_output_tokens=8192,
        input_price_per_1k=0.00005,
        output_price_per_1k=0.00008,
        avg_latency_ms=50,
        tokens_per_second=500,
    ),
    "mixtral-8x7b-32768": ModelInfo(
        model_id="mixtral-8x7b-32768",
        provider=LLMProvider.GROQ,
        display_name="Mixtral 8x7B (Groq)",
        context_window=32768,
        max_output_tokens=8192,
        input_price_per_1k=0.00024,
        output_price_per_1k=0.00024,
        avg_latency_ms=80,
        tokens_per_second=400,
    ),
}


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Get model information by ID."""
    return MODELS.get(model_id)


def list_models(
    provider: Optional[LLMProvider] = None,
    supports_streaming: bool = False,
    supports_tools: bool = False,
    supports_vision: bool = False,
    max_price_per_1k: Optional[float] = None,
) -> List[ModelInfo]:
    """List models matching criteria."""
    models = list(MODELS.values())

    if provider:
        models = [m for m in models if m.provider == provider]

    if supports_streaming:
        models = [m for m in models if m.supports_streaming]

    if supports_tools:
        models = [m for m in models if m.supports_tools]

    if supports_vision:
        models = [m for m in models if m.supports_vision]

    if max_price_per_1k is not None:
        models = [m for m in models if m.output_price_per_1k <= max_price_per_1k]

    return models


@dataclass
class LLMMessage:
    """A message in the conversation."""

    role: LLMRole
    content: str

    # Optional fields
    name: Optional[str] = None  # For tool/function messages
    tool_call_id: Optional[str] = None  # For tool responses
    tool_calls: Optional[List[Dict[str, Any]]] = None  # For assistant tool calls

    # Metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For multimodal
    images: Optional[List[str]] = None  # Base64 or URLs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        msg = {
            "role": self.role.value,
            "content": self.content,
        }

        if self.name:
            msg["name"] = self.name

        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id

        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls

        if self.images:
            # Handle multimodal content
            content = []
            if self.content:
                content.append({"type": "text", "text": self.content})
            for image in self.images:
                if image.startswith("http"):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image}
                    })
                else:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                    })
            msg["content"] = content

        return msg

    @classmethod
    def system(cls, content: str, **kwargs) -> "LLMMessage":
        """Create a system message."""
        return cls(role=LLMRole.SYSTEM, content=content, **kwargs)

    @classmethod
    def user(cls, content: str, images: Optional[List[str]] = None, **kwargs) -> "LLMMessage":
        """Create a user message."""
        return cls(role=LLMRole.USER, content=content, images=images, **kwargs)

    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[List[Dict]] = None, **kwargs) -> "LLMMessage":
        """Create an assistant message."""
        return cls(role=LLMRole.ASSISTANT, content=content, tool_calls=tool_calls, **kwargs)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str, **kwargs) -> "LLMMessage":
        """Create a tool response message."""
        return cls(
            role=LLMRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            **kwargs
        )


@dataclass
class LLMUsage:
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Cost tracking
    cost_usd: float = 0.0

    # Timing
    latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    tokens_per_second: float = 0.0

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        """Add two usage objects."""
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_ms=self.latency_ms + other.latency_ms,
        )


@dataclass
class StreamChunk:
    """A chunk of streaming response."""

    content: str = ""
    role: Optional[LLMRole] = None

    # Tool calls (may be partial)
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # Metadata
    finish_reason: Optional[str] = None
    index: int = 0
    model: Optional[str] = None

    # Timing
    timestamp: float = field(default_factory=time.time)

    @property
    def is_finished(self) -> bool:
        """Check if this is the final chunk."""
        return self.finish_reason is not None


@dataclass
class LLMResponse:
    """Response from LLM."""

    # Content
    content: str = ""
    role: LLMRole = LLMRole.ASSISTANT

    # Tool calls
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # Usage
    usage: LLMUsage = field(default_factory=LLMUsage)

    # Metadata
    model: str = ""
    provider: LLMProvider = LLMProvider.OPENAI
    finish_reason: Optional[str] = None

    # Request tracking
    request_id: Optional[str] = None

    # Timing
    created_at: float = field(default_factory=time.time)

    def to_message(self) -> LLMMessage:
        """Convert response to message."""
        return LLMMessage(
            role=self.role,
            content=self.content,
            tool_calls=self.tool_calls,
        )

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    # Authentication
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None

    # Organization/Project
    organization_id: Optional[str] = None
    project_id: Optional[str] = None

    # Default model
    default_model: Optional[str] = None

    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 100000

    # Retry settings
    max_retries: int = 3
    retry_delay_ms: int = 1000
    retry_multiplier: float = 2.0

    # Timeouts
    connect_timeout_ms: int = 5000
    read_timeout_ms: int = 60000

    # Headers
    custom_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuration for LLM requests."""

    # Model selection
    model: str = "gpt-4o-mini"

    # Generation parameters
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Stop sequences
    stop: Optional[List[str]] = None

    # Output format
    response_format: Optional[Dict[str, Any]] = None  # For JSON mode

    # Tool/Function calling
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None  # "auto", "none", or specific

    # Streaming
    stream: bool = False

    # Seed for reproducibility
    seed: Optional[int] = None

    # Metadata
    user: Optional[str] = None  # End-user identifier

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

        if self.max_tokens:
            params["max_tokens"] = self.max_tokens

        if self.stop:
            params["stop"] = self.stop

        if self.response_format:
            params["response_format"] = self.response_format

        if self.tools:
            params["tools"] = self.tools

        if self.tool_choice:
            params["tool_choice"] = self.tool_choice

        if self.stream:
            params["stream"] = True

        if self.seed is not None:
            params["seed"] = self.seed

        if self.user:
            params["user"] = self.user

        return params


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    provider_type: LLMProvider = LLMProvider.CUSTOM

    def __init__(self, config: Optional[ProviderConfig] = None):
        self.config = config or ProviderConfig()
        self._client: Optional[Any] = None

    @abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a completion."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion."""
        pass

    async def close(self) -> None:
        """Close the provider and cleanup resources."""
        pass

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a model."""
        return get_model_info(model_id)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple estimation: ~4 characters per token for English
        return len(text) // 4


__all__ = [
    "LLMRole",
    "LLMProvider",
    "ModelInfo",
    "MODELS",
    "get_model_info",
    "list_models",
    "LLMMessage",
    "LLMUsage",
    "StreamChunk",
    "LLMResponse",
    "ProviderConfig",
    "LLMConfig",
    "BaseLLMProvider",
]
