"""
LLM Providers

Multi-provider LLM integration:
- OpenAI
- Anthropic
- Google (Gemini)
- Azure OpenAI
- Groq
- Together AI
- Custom/Local
"""

from typing import Optional, Dict, Any, List, AsyncGenerator, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import aiohttp
import json
import logging
import time

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    GROQ = "groq"
    TOGETHER = "together"
    MISTRAL = "mistral"
    COHERE = "cohere"
    CUSTOM = "custom"


class MessageRole(str, Enum):
    """Message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class Message:
    """Chat message."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        msg = {"role": self.role.value, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.function_call:
            msg["function_call"] = self.function_call
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


@dataclass
class CompletionConfig:
    """Configuration for completion requests."""
    model: str = "gpt-4-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False

    # Function calling
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # Response format
    response_format: Optional[Dict[str, str]] = None

    # Additional params
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResponse:
    """Response from completion request."""
    content: str = ""
    finish_reason: str = ""
    model: str = ""

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Function/tool calls
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # Metadata
    response_id: str = ""
    latency_ms: float = 0.0
    provider: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "finish_reason": self.finish_reason,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "function_call": self.function_call,
            "tool_calls": self.tool_calls,
            "latency_ms": self.latency_ms,
        }


@dataclass
class StreamChunk:
    """Streaming response chunk."""
    content: str = ""
    delta: str = ""
    finish_reason: Optional[str] = None
    function_call_delta: Optional[Dict[str, Any]] = None
    tool_call_delta: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """Abstract LLM provider."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> CompletionResponse:
        """Generate completion."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream completion."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider."""

    def __init__(
        self,
        api_key: str,
        organization: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
    ):
        self.api_key = api_key
        self.organization = organization
        self.base_url = base_url

    @property
    def provider_name(self) -> str:
        return "openai"

    async def complete(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> CompletionResponse:
        """Generate completion."""
        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        payload = self._build_payload(messages, config)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        logger.error(f"OpenAI API error: {error}")
                        raise Exception(f"OpenAI API error: {response.status}")

                    data = await response.json()
                    return self._parse_response(data, start_time)

        except Exception as e:
            logger.error(f"OpenAI completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream completion."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = self._build_payload(messages, config)
        payload["stream"] = True

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    content = ""
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
                                delta_content = delta.get("content", "")
                                content += delta_content

                                yield StreamChunk(
                                    content=content,
                                    delta=delta_content,
                                    finish_reason=data["choices"][0].get("finish_reason"),
                                    function_call_delta=delta.get("function_call"),
                                    tool_call_delta=delta.get("tool_calls"),
                                )
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"OpenAI stream error: {e}")
            raise

    def _build_payload(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> Dict[str, Any]:
        """Build API payload."""
        payload = {
            "model": config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
        }

        if config.stop:
            payload["stop"] = config.stop
        if config.functions:
            payload["functions"] = config.functions
        if config.function_call:
            payload["function_call"] = config.function_call
        if config.tools:
            payload["tools"] = config.tools
        if config.tool_choice:
            payload["tool_choice"] = config.tool_choice
        if config.response_format:
            payload["response_format"] = config.response_format

        payload.update(config.extra_params)
        return payload

    def _parse_response(
        self,
        data: Dict[str, Any],
        start_time: float,
    ) -> CompletionResponse:
        """Parse API response."""
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})

        return CompletionResponse(
            content=message.get("content", ""),
            finish_reason=choice.get("finish_reason", ""),
            model=data.get("model", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            function_call=message.get("function_call"),
            tool_calls=message.get("tool_calls"),
            response_id=data.get("id", ""),
            latency_ms=(time.time() - start_time) * 1000,
            provider=self.provider_name,
        )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
    ):
        self.api_key = api_key
        self.base_url = base_url

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def complete(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> CompletionResponse:
        """Generate completion."""
        start_time = time.time()

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = self._build_payload(messages, config)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        logger.error(f"Anthropic API error: {error}")
                        raise Exception(f"Anthropic API error: {response.status}")

                    data = await response.json()
                    return self._parse_response(data, start_time)

        except Exception as e:
            logger.error(f"Anthropic completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream completion."""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = self._build_payload(messages, config)
        payload["stream"] = True

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    content = ""
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                data = json.loads(data_str)
                                event_type = data.get("type")

                                if event_type == "content_block_delta":
                                    delta = data.get("delta", {})
                                    delta_text = delta.get("text", "")
                                    content += delta_text

                                    yield StreamChunk(
                                        content=content,
                                        delta=delta_text,
                                    )
                                elif event_type == "message_stop":
                                    yield StreamChunk(
                                        content=content,
                                        delta="",
                                        finish_reason="stop",
                                    )
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Anthropic stream error: {e}")
            raise

    def _build_payload(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> Dict[str, Any]:
        """Build API payload."""
        # Extract system message
        system = ""
        claude_messages = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system = msg.content
            else:
                claude_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        payload = {
            "model": config.model,
            "messages": claude_messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }

        if system:
            payload["system"] = system

        if config.stop:
            payload["stop_sequences"] = config.stop

        if config.tools:
            payload["tools"] = self._convert_tools(config.tools)

        return payload

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                converted.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
        return converted

    def _parse_response(
        self,
        data: Dict[str, Any],
        start_time: float,
    ) -> CompletionResponse:
        """Parse API response."""
        content_blocks = data.get("content", [])
        content = ""
        tool_use = None

        for block in content_blocks:
            if block["type"] == "text":
                content += block["text"]
            elif block["type"] == "tool_use":
                tool_use = {
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": json.dumps(block["input"]),
                }

        usage = data.get("usage", {})

        return CompletionResponse(
            content=content,
            finish_reason=data.get("stop_reason", ""),
            model=data.get("model", ""),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            tool_calls=[{"function": tool_use}] if tool_use else None,
            response_id=data.get("id", ""),
            latency_ms=(time.time() - start_time) * 1000,
            provider=self.provider_name,
        )


class GroqProvider(BaseLLMProvider):
    """Groq provider for fast inference."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
    ):
        self.api_key = api_key
        self.base_url = base_url
        # Groq uses OpenAI-compatible API
        self._openai = OpenAIProvider(api_key, base_url=base_url)

    @property
    def provider_name(self) -> str:
        return "groq"

    async def complete(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> CompletionResponse:
        """Generate completion."""
        response = await self._openai.complete(messages, config)
        response.provider = self.provider_name
        return response

    async def stream(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream completion."""
        async for chunk in self._openai.stream(messages, config):
            yield chunk


class TogetherProvider(BaseLLMProvider):
    """Together AI provider."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.together.xyz/v1",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._openai = OpenAIProvider(api_key, base_url=base_url)

    @property
    def provider_name(self) -> str:
        return "together"

    async def complete(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> CompletionResponse:
        """Generate completion."""
        response = await self._openai.complete(messages, config)
        response.provider = self.provider_name
        return response

    async def stream(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream completion."""
        async for chunk in self._openai.stream(messages, config):
            yield chunk


class MistralProvider(BaseLLMProvider):
    """Mistral AI provider."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.mistral.ai/v1",
    ):
        self.api_key = api_key
        self.base_url = base_url

    @property
    def provider_name(self) -> str:
        return "mistral"

    async def complete(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> CompletionResponse:
        """Generate completion."""
        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise Exception(f"Mistral API error: {error}")

                    data = await response.json()
                    choice = data["choices"][0]
                    usage = data.get("usage", {})

                    return CompletionResponse(
                        content=choice["message"]["content"],
                        finish_reason=choice.get("finish_reason", ""),
                        model=data.get("model", ""),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        response_id=data.get("id", ""),
                        latency_ms=(time.time() - start_time) * 1000,
                        provider=self.provider_name,
                    )

        except Exception as e:
            logger.error(f"Mistral completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream completion."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    content = ""
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
                                delta_content = delta.get("content", "")
                                content += delta_content

                                yield StreamChunk(
                                    content=content,
                                    delta=delta_content,
                                    finish_reason=data["choices"][0].get("finish_reason"),
                                )
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Mistral stream error: {e}")
            raise


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI provider."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment_name: str,
        api_version: str = "2024-02-01",
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version

    @property
    def provider_name(self) -> str:
        return "azure_openai"

    async def complete(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> CompletionResponse:
        """Generate completion."""
        start_time = time.time()

        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "messages": [m.to_dict() for m in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        }

        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment_name}"
            f"/chat/completions?api-version={self.api_version}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise Exception(f"Azure OpenAI API error: {error}")

                    data = await response.json()
                    choice = data["choices"][0]
                    usage = data.get("usage", {})

                    return CompletionResponse(
                        content=choice["message"]["content"],
                        finish_reason=choice.get("finish_reason", ""),
                        model=self.deployment_name,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        function_call=choice["message"].get("function_call"),
                        tool_calls=choice["message"].get("tool_calls"),
                        response_id=data.get("id", ""),
                        latency_ms=(time.time() - start_time) * 1000,
                        provider=self.provider_name,
                    )

        except Exception as e:
            logger.error(f"Azure OpenAI completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream completion."""
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "messages": [m.to_dict() for m in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": True,
        }

        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment_name}"
            f"/chat/completions?api-version={self.api_version}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    content = ""
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
                                delta_content = delta.get("content", "")
                                content += delta_content

                                yield StreamChunk(
                                    content=content,
                                    delta=delta_content,
                                    finish_reason=data["choices"][0].get("finish_reason"),
                                )
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Azure OpenAI stream error: {e}")
            raise


# Provider factory
def create_provider(
    provider: LLMProvider,
    api_key: str,
    **kwargs,
) -> BaseLLMProvider:
    """Create LLM provider."""
    providers = {
        LLMProvider.OPENAI: lambda: OpenAIProvider(api_key, **kwargs),
        LLMProvider.ANTHROPIC: lambda: AnthropicProvider(api_key, **kwargs),
        LLMProvider.GROQ: lambda: GroqProvider(api_key, **kwargs),
        LLMProvider.TOGETHER: lambda: TogetherProvider(api_key, **kwargs),
        LLMProvider.MISTRAL: lambda: MistralProvider(api_key, **kwargs),
        LLMProvider.AZURE_OPENAI: lambda: AzureOpenAIProvider(api_key, **kwargs),
    }

    factory = providers.get(provider)
    if not factory:
        raise ValueError(f"Unknown provider: {provider}")

    return factory()
