"""
LLM Executor Module

This module provides LLM execution capabilities with streaming support,
retry logic, and multi-provider abstraction.
"""

import asyncio
import logging
import time
import json
from abc import ABC, abstractmethod
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
    Union,
)

from .base import (
    AgentResponse,
    ResponseType,
    FunctionCall,
    FunctionDefinition,
    AgentExecutionError,
    TokenLimitExceededError,
)


logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for LLM execution."""

    # Provider settings
    provider: str = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Generation settings
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Timeout and retry
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Streaming
    stream: bool = True
    chunk_callback: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None


@dataclass
class ExecutionResult:
    """Result from LLM execution."""

    content: str = ""
    function_calls: List[FunctionCall] = field(default_factory=list)
    finish_reason: str = ""

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Timing
    latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0

    # Metadata
    model: str = ""
    provider: str = ""


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    async def execute(
        self,
        messages: List[Dict[str, Any]],
        config: ExecutionConfig,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ExecutionResult:
        """
        Execute LLM request.

        Args:
            messages: Conversation messages
            config: Execution configuration
            tools: Optional tool definitions

        Returns:
            Execution result
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        config: ExecutionConfig,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Tuple[str, Optional[ExecutionResult]]]:
        """
        Stream LLM response.

        Args:
            messages: Conversation messages
            config: Execution configuration
            tools: Optional tool definitions

        Yields:
            Tuples of (chunk, final_result)
            final_result is None until the last chunk
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    @property
    def name(self) -> str:
        return "openai"

    def __init__(self):
        """Initialize provider."""
        self._client: Optional[Any] = None

    async def _get_client(self, config: ExecutionConfig) -> Any:
        """Get or create OpenAI client."""
        try:
            from openai import AsyncOpenAI

            return AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout_seconds,
            )
        except ImportError:
            raise AgentExecutionError("openai package not installed")

    async def execute(
        self,
        messages: List[Dict[str, Any]],
        config: ExecutionConfig,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ExecutionResult:
        """Execute non-streaming request."""
        client = await self._get_client(config)
        start_time = time.time()

        try:
            kwargs = {
                "model": config.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty,
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await client.chat.completions.create(**kwargs)

            latency_ms = (time.time() - start_time) * 1000

            result = ExecutionResult(
                model=config.model,
                provider=self.name,
                latency_ms=latency_ms,
            )

            message = response.choices[0].message
            result.content = message.content or ""
            result.finish_reason = response.choices[0].finish_reason

            # Parse function calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    result.function_calls.append(FunctionCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments),
                    ))

            # Token usage
            if response.usage:
                result.prompt_tokens = response.usage.prompt_tokens
                result.completion_tokens = response.usage.completion_tokens
                result.total_tokens = response.usage.total_tokens

            return result

        except Exception as e:
            logger.exception(f"OpenAI execution failed: {e}")
            raise AgentExecutionError(f"OpenAI execution failed: {e}")

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        config: ExecutionConfig,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Tuple[str, Optional[ExecutionResult]]]:
        """Stream response."""
        client = await self._get_client(config)
        start_time = time.time()
        first_token_time: Optional[float] = None

        try:
            kwargs = {
                "model": config.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            stream = await client.chat.completions.create(**kwargs)

            full_content = ""
            tool_calls_data: Dict[int, Dict[str, Any]] = {}
            finish_reason = ""
            usage = None

            async for chunk in stream:
                if first_token_time is None:
                    first_token_time = time.time()

                delta = chunk.choices[0].delta if chunk.choices else None
                finish = chunk.choices[0].finish_reason if chunk.choices else None

                if finish:
                    finish_reason = finish

                if delta:
                    # Content
                    if delta.content:
                        full_content += delta.content
                        yield delta.content, None

                    # Tool calls
                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            idx = tool_call.index
                            if idx not in tool_calls_data:
                                tool_calls_data[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": "",
                                }
                            if tool_call.id:
                                tool_calls_data[idx]["id"] = tool_call.id
                            if tool_call.function:
                                if tool_call.function.name:
                                    tool_calls_data[idx]["name"] = tool_call.function.name
                                if tool_call.function.arguments:
                                    tool_calls_data[idx]["arguments"] += tool_call.function.arguments

                # Usage (comes in final chunk)
                if chunk.usage:
                    usage = chunk.usage

            # Build final result
            latency_ms = (time.time() - start_time) * 1000
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else latency_ms

            result = ExecutionResult(
                content=full_content,
                finish_reason=finish_reason,
                model=config.model,
                provider=self.name,
                latency_ms=latency_ms,
                time_to_first_token_ms=ttft_ms,
            )

            # Parse tool calls
            for data in tool_calls_data.values():
                if data["name"]:
                    try:
                        args = json.loads(data["arguments"]) if data["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}

                    result.function_calls.append(FunctionCall(
                        id=data["id"],
                        name=data["name"],
                        arguments=args,
                    ))

            if usage:
                result.prompt_tokens = usage.prompt_tokens
                result.completion_tokens = usage.completion_tokens
                result.total_tokens = usage.total_tokens

            yield "", result

        except Exception as e:
            logger.exception(f"OpenAI stream failed: {e}")
            raise AgentExecutionError(f"OpenAI stream failed: {e}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    @property
    def name(self) -> str:
        return "anthropic"

    async def _get_client(self, config: ExecutionConfig) -> Any:
        """Get or create Anthropic client."""
        try:
            from anthropic import AsyncAnthropic

            return AsyncAnthropic(
                api_key=config.api_key,
                timeout=config.timeout_seconds,
            )
        except ImportError:
            raise AgentExecutionError("anthropic package not installed")

    def _convert_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Convert messages to Anthropic format."""
        system_prompt = ""
        converted = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "user":
                converted.append({"role": "user", "content": content})
            elif role == "assistant":
                converted.append({"role": "assistant", "content": content})
            elif role == "function":
                # Convert function results to user messages with tool_result
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content,
                    }],
                })

        return system_prompt, converted

    def _convert_tools(
        self,
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert OpenAI tool format to Anthropic format."""
        if not tools:
            return None

        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                converted.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })

        return converted if converted else None

    async def execute(
        self,
        messages: List[Dict[str, Any]],
        config: ExecutionConfig,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ExecutionResult:
        """Execute non-streaming request."""
        client = await self._get_client(config)
        start_time = time.time()

        system_prompt, converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)

        try:
            kwargs = {
                "model": config.model,
                "messages": converted_messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if converted_tools:
                kwargs["tools"] = converted_tools

            response = await client.messages.create(**kwargs)

            latency_ms = (time.time() - start_time) * 1000

            result = ExecutionResult(
                model=config.model,
                provider=self.name,
                latency_ms=latency_ms,
            )

            # Parse content blocks
            for block in response.content:
                if block.type == "text":
                    result.content += block.text
                elif block.type == "tool_use":
                    result.function_calls.append(FunctionCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    ))

            result.finish_reason = response.stop_reason
            result.prompt_tokens = response.usage.input_tokens
            result.completion_tokens = response.usage.output_tokens
            result.total_tokens = result.prompt_tokens + result.completion_tokens

            return result

        except Exception as e:
            logger.exception(f"Anthropic execution failed: {e}")
            raise AgentExecutionError(f"Anthropic execution failed: {e}")

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        config: ExecutionConfig,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Tuple[str, Optional[ExecutionResult]]]:
        """Stream response."""
        client = await self._get_client(config)
        start_time = time.time()
        first_token_time: Optional[float] = None

        system_prompt, converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)

        try:
            kwargs = {
                "model": config.model,
                "messages": converted_messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if converted_tools:
                kwargs["tools"] = converted_tools

            full_content = ""
            tool_calls: List[FunctionCall] = []
            current_tool: Optional[Dict[str, Any]] = None
            usage_input = 0
            usage_output = 0
            stop_reason = ""

            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if first_token_time is None and hasattr(event, "type"):
                        first_token_time = time.time()

                    if hasattr(event, "type"):
                        if event.type == "content_block_start":
                            if event.content_block.type == "tool_use":
                                current_tool = {
                                    "id": event.content_block.id,
                                    "name": event.content_block.name,
                                    "input": "",
                                }
                        elif event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                full_content += event.delta.text
                                yield event.delta.text, None
                            elif hasattr(event.delta, "partial_json"):
                                if current_tool:
                                    current_tool["input"] += event.delta.partial_json
                        elif event.type == "content_block_stop":
                            if current_tool:
                                try:
                                    args = json.loads(current_tool["input"]) if current_tool["input"] else {}
                                except json.JSONDecodeError:
                                    args = {}
                                tool_calls.append(FunctionCall(
                                    id=current_tool["id"],
                                    name=current_tool["name"],
                                    arguments=args,
                                ))
                                current_tool = None
                        elif event.type == "message_delta":
                            if hasattr(event, "usage"):
                                usage_output = event.usage.output_tokens
                            if hasattr(event.delta, "stop_reason"):
                                stop_reason = event.delta.stop_reason
                        elif event.type == "message_start":
                            if hasattr(event.message, "usage"):
                                usage_input = event.message.usage.input_tokens

            latency_ms = (time.time() - start_time) * 1000
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else latency_ms

            result = ExecutionResult(
                content=full_content,
                function_calls=tool_calls,
                finish_reason=stop_reason,
                model=config.model,
                provider=self.name,
                latency_ms=latency_ms,
                time_to_first_token_ms=ttft_ms,
                prompt_tokens=usage_input,
                completion_tokens=usage_output,
                total_tokens=usage_input + usage_output,
            )

            yield "", result

        except Exception as e:
            logger.exception(f"Anthropic stream failed: {e}")
            raise AgentExecutionError(f"Anthropic stream failed: {e}")


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    @property
    def name(self) -> str:
        return "mock"

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        delay_ms: float = 100,
    ):
        """
        Initialize mock provider.

        Args:
            responses: List of responses to cycle through
            delay_ms: Simulated delay
        """
        self.responses = responses or ["This is a mock response."]
        self.delay_ms = delay_ms
        self._response_index = 0

    async def execute(
        self,
        messages: List[Dict[str, Any]],
        config: ExecutionConfig,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ExecutionResult:
        """Execute mock request."""
        await asyncio.sleep(self.delay_ms / 1000)

        response = self.responses[self._response_index % len(self.responses)]
        self._response_index += 1

        return ExecutionResult(
            content=response,
            finish_reason="stop",
            model="mock-model",
            provider=self.name,
            latency_ms=self.delay_ms,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        config: ExecutionConfig,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Tuple[str, Optional[ExecutionResult]]]:
        """Stream mock response."""
        response = self.responses[self._response_index % len(self.responses)]
        self._response_index += 1

        words = response.split()
        for i, word in enumerate(words):
            await asyncio.sleep(self.delay_ms / len(words) / 1000)
            chunk = word + " " if i < len(words) - 1 else word
            yield chunk, None

        result = ExecutionResult(
            content=response,
            finish_reason="stop",
            model="mock-model",
            provider=self.name,
            latency_ms=self.delay_ms,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        yield "", result


class LLMExecutor:
    """
    Main LLM executor with provider abstraction and retry logic.

    Supports:
    - Multiple providers (OpenAI, Anthropic)
    - Streaming and non-streaming
    - Automatic retries
    - Token counting
    """

    # Provider registry
    PROVIDERS: Dict[str, type] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "mock": MockProvider,
    }

    def __init__(
        self,
        default_provider: str = "openai",
    ):
        """
        Initialize executor.

        Args:
            default_provider: Default provider name
        """
        self.default_provider = default_provider
        self._providers: Dict[str, LLMProvider] = {}

    def _get_provider(self, name: str) -> LLMProvider:
        """Get or create provider instance."""
        if name not in self._providers:
            provider_class = self.PROVIDERS.get(name)
            if not provider_class:
                raise AgentExecutionError(f"Unknown provider: {name}")
            self._providers[name] = provider_class()
        return self._providers[name]

    def register_provider(
        self,
        name: str,
        provider: LLMProvider,
    ) -> None:
        """Register a custom provider."""
        self._providers[name] = provider

    async def execute(
        self,
        messages: List[Dict[str, Any]],
        config: Optional[ExecutionConfig] = None,
        tools: Optional[List[FunctionDefinition]] = None,
    ) -> ExecutionResult:
        """
        Execute LLM request with retry logic.

        Args:
            messages: Conversation messages
            config: Execution configuration
            tools: Optional tool definitions

        Returns:
            Execution result
        """
        config = config or ExecutionConfig()
        provider = self._get_provider(config.provider or self.default_provider)

        # Convert tools to API format
        tools_dict = [t.to_openai_format() for t in tools] if tools else None

        last_error: Optional[Exception] = None
        for attempt in range(config.max_retries + 1):
            try:
                if config.stream:
                    # Collect streaming response
                    full_result: Optional[ExecutionResult] = None
                    content_chunks = []

                    async for chunk, result in provider.stream(messages, config, tools_dict):
                        if chunk:
                            content_chunks.append(chunk)
                            if config.chunk_callback:
                                await config.chunk_callback(chunk)
                        if result:
                            full_result = result

                    return full_result or ExecutionResult(
                        content="".join(content_chunks),
                        provider=provider.name,
                    )
                else:
                    return await provider.execute(messages, config, tools_dict)

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM execution attempt {attempt + 1} failed: {e}"
                )
                if attempt < config.max_retries:
                    await asyncio.sleep(config.retry_delay_seconds * (attempt + 1))

        raise AgentExecutionError(
            f"LLM execution failed after {config.max_retries + 1} attempts: {last_error}"
        )

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        config: Optional[ExecutionConfig] = None,
        tools: Optional[List[FunctionDefinition]] = None,
    ) -> AsyncIterator[Tuple[str, Optional[ExecutionResult]]]:
        """
        Stream LLM response.

        Args:
            messages: Conversation messages
            config: Execution configuration
            tools: Optional tool definitions

        Yields:
            Tuples of (chunk, final_result)
        """
        config = config or ExecutionConfig()
        provider = self._get_provider(config.provider or self.default_provider)

        # Convert tools to API format
        tools_dict = [t.to_openai_format() for t in tools] if tools else None

        async for chunk, result in provider.stream(messages, config, tools_dict):
            yield chunk, result


__all__ = [
    "ExecutionConfig",
    "ExecutionResult",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "MockProvider",
    "LLMExecutor",
]
