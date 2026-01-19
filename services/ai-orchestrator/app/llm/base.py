"""Base LLM adapter interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional


@dataclass
class FunctionCall:
    """Represents a function/tool call from the LLM."""
    name: str
    arguments: dict[str, Any]
    id: Optional[str] = None


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    function_calls: list[FunctionCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Optional[dict] = None

    # For streaming
    is_partial: bool = False
    is_complete: bool = False


@dataclass
class Message:
    """A conversation message."""
    role: str  # system, user, assistant, function
    content: str
    name: Optional[str] = None  # For function messages
    function_call: Optional[FunctionCall] = None


@dataclass
class Tool:
    """A tool/function that the LLM can call."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get adapter name."""
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if adapter supports streaming."""
        pass

    @property
    @abstractmethod
    def supports_functions(self) -> bool:
        """Check if adapter supports function calling."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: Conversation history
            system_prompt: System prompt (overrides default)
            tools: Available tools/functions
            **kwargs: Provider-specific options

        Returns:
            LLMResponse with text and optional function calls
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        **kwargs,
    ) -> AsyncIterator[LLMResponse]:
        """
        Stream responses from the LLM.

        Args:
            messages: Conversation history
            system_prompt: System prompt
            tools: Available tools/functions
            **kwargs: Provider-specific options

        Yields:
            Partial LLMResponse objects as they arrive
        """
        pass

    async def close(self) -> None:
        """Clean up resources."""
        pass
