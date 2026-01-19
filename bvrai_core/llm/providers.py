"""
LLM Provider Implementations

This module contains implementations for various LLM providers
including OpenAI, Anthropic, Google, and others.
"""

import asyncio
import json
import logging
import os
import time
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Type,
)

from .base import (
    BaseLLMProvider,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMRole,
    LLMProvider,
    LLMUsage,
    ProviderConfig,
    StreamChunk,
    get_model_info,
)


logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (GPT-4, GPT-3.5, etc.)."""

    provider_type = LLMProvider.OPENAI

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config)

        self._api_key = (
            api_key or
            self.config.api_key or
            os.environ.get("OPENAI_API_KEY")
        )

        if not self._api_key:
            logger.warning("OpenAI API key not provided")

    async def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self.config.api_base,
                    organization=self.config.organization_id,
                    timeout=self.config.read_timeout_ms / 1000,
                    max_retries=self.config.max_retries,
                )
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client

    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format."""
        converted = []
        for msg in messages:
            converted.append(msg.to_dict())
        return converted

    async def complete(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a completion using OpenAI."""
        client = await self._get_client()
        config = config or LLMConfig()

        start_time = time.time()

        try:
            # Prepare request
            params = config.to_dict()
            params["messages"] = self._convert_messages(messages)

            # Make request
            response = await client.chat.completions.create(**params)

            latency_ms = (time.time() - start_time) * 1000

            # Extract response
            choice = response.choices[0]
            message = choice.message

            # Build tool calls if present
            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]

            # Calculate cost
            model_info = get_model_info(config.model)
            cost = 0.0
            if model_info and response.usage:
                cost = model_info.calculate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            return LLMResponse(
                content=message.content or "",
                role=LLMRole.ASSISTANT,
                tool_calls=tool_calls,
                usage=LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_tokens=response.usage.total_tokens if response.usage else 0,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                ),
                model=response.model,
                provider=self.provider_type,
                finish_reason=choice.finish_reason,
                request_id=response.id,
            )

        except Exception as e:
            logger.error(f"OpenAI completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion using OpenAI."""
        client = await self._get_client()
        config = config or LLMConfig()
        config.stream = True

        start_time = time.time()
        first_token_time = None

        try:
            # Prepare request
            params = config.to_dict()
            params["messages"] = self._convert_messages(messages)

            # Make streaming request
            stream = await client.chat.completions.create(**params)

            # Accumulate tool calls across chunks
            tool_call_accumulator: Dict[int, Dict[str, Any]] = {}

            async for chunk in stream:
                if first_token_time is None:
                    first_token_time = time.time()

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # Handle content
                content = delta.content or ""

                # Handle tool calls
                tool_calls = None
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_accumulator:
                            tool_call_accumulator[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }

                        if tc.id:
                            tool_call_accumulator[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_call_accumulator[idx]["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_call_accumulator[idx]["function"]["arguments"] += tc.function.arguments

                    # Include accumulated tool calls in final chunk
                    if finish_reason:
                        tool_calls = list(tool_call_accumulator.values())

                yield StreamChunk(
                    content=content,
                    role=LLMRole.ASSISTANT if delta.role else None,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    index=chunk.choices[0].index,
                    model=chunk.model,
                )

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

    async def close(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider (Claude models)."""

    provider_type = LLMProvider.ANTHROPIC

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config)

        self._api_key = (
            api_key or
            self.config.api_key or
            os.environ.get("ANTHROPIC_API_KEY")
        )

        if not self._api_key:
            logger.warning("Anthropic API key not provided")

    async def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(
                    api_key=self._api_key,
                    base_url=self.config.api_base,
                    timeout=self.config.read_timeout_ms / 1000,
                    max_retries=self.config.max_retries,
                )
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._client

    def _convert_messages(
        self,
        messages: List[LLMMessage],
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert messages to Anthropic format."""
        system_prompt = None
        converted = []

        for msg in messages:
            if msg.role == LLMRole.SYSTEM:
                system_prompt = msg.content
            elif msg.role == LLMRole.USER:
                content = msg.content
                if msg.images:
                    # Handle multimodal
                    content = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    for image in msg.images:
                        if image.startswith("http"):
                            content.append({
                                "type": "image",
                                "source": {"type": "url", "url": image}
                            })
                        else:
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image,
                                }
                            })
                converted.append({"role": "user", "content": content})
            elif msg.role == LLMRole.ASSISTANT:
                converted.append({"role": "assistant", "content": msg.content})
            elif msg.role == LLMRole.TOOL:
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }]
                })

        return system_prompt, converted

    def _convert_tools(self, tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
        """Convert OpenAI-style tools to Anthropic format."""
        if not tools:
            return None

        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })

        return anthropic_tools if anthropic_tools else None

    async def complete(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a completion using Anthropic."""
        client = await self._get_client()
        config = config or LLMConfig()

        start_time = time.time()

        try:
            system_prompt, converted_messages = self._convert_messages(messages)

            # Prepare request
            params = {
                "model": config.model,
                "messages": converted_messages,
                "max_tokens": config.max_tokens or 4096,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }

            if system_prompt:
                params["system"] = system_prompt

            if config.stop:
                params["stop_sequences"] = config.stop

            tools = self._convert_tools(config.tools)
            if tools:
                params["tools"] = tools

            # Make request
            response = await client.messages.create(**params)

            latency_ms = (time.time() - start_time) * 1000

            # Extract content and tool calls
            content = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        }
                    })

            # Calculate cost
            model_info = get_model_info(config.model)
            cost = 0.0
            if model_info:
                cost = model_info.calculate_cost(
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )

            return LLMResponse(
                content=content,
                role=LLMRole.ASSISTANT,
                tool_calls=tool_calls if tool_calls else None,
                usage=LLMUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                ),
                model=response.model,
                provider=self.provider_type,
                finish_reason=response.stop_reason,
                request_id=response.id,
            )

        except Exception as e:
            logger.error(f"Anthropic completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion using Anthropic."""
        client = await self._get_client()
        config = config or LLMConfig()

        start_time = time.time()
        first_token_time = None

        try:
            system_prompt, converted_messages = self._convert_messages(messages)

            # Prepare request
            params = {
                "model": config.model,
                "messages": converted_messages,
                "max_tokens": config.max_tokens or 4096,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }

            if system_prompt:
                params["system"] = system_prompt

            if config.stop:
                params["stop_sequences"] = config.stop

            tools = self._convert_tools(config.tools)
            if tools:
                params["tools"] = tools

            # Make streaming request
            async with client.messages.stream(**params) as stream:
                current_tool: Optional[Dict[str, Any]] = None
                tool_calls_accumulated = []

                async for event in stream:
                    if first_token_time is None:
                        first_token_time = time.time()

                    # Handle different event types
                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            current_tool = {
                                "id": event.content_block.id,
                                "type": "function",
                                "function": {
                                    "name": event.content_block.name,
                                    "arguments": "",
                                }
                            }

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield StreamChunk(
                                content=event.delta.text,
                                role=LLMRole.ASSISTANT,
                            )
                        elif event.delta.type == "input_json_delta":
                            if current_tool:
                                current_tool["function"]["arguments"] += event.delta.partial_json

                    elif event.type == "content_block_stop":
                        if current_tool:
                            tool_calls_accumulated.append(current_tool)
                            current_tool = None

                    elif event.type == "message_stop":
                        # Final chunk with tool calls
                        yield StreamChunk(
                            content="",
                            tool_calls=tool_calls_accumulated if tool_calls_accumulated else None,
                            finish_reason="end_turn",
                        )

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise

    async def close(self) -> None:
        """Close the Anthropic client."""
        if self._client:
            await self._client.close()
            self._client = None


class GoogleProvider(BaseLLMProvider):
    """Google AI (Gemini) provider."""

    provider_type = LLMProvider.GOOGLE

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config)

        self._api_key = (
            api_key or
            self.config.api_key or
            os.environ.get("GOOGLE_API_KEY")
        )

        if not self._api_key:
            logger.warning("Google API key not provided")

    async def _get_client(self):
        """Get or create Google AI client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._api_key)
                self._client = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package required. "
                    "Install with: pip install google-generativeai"
                )
        return self._client

    def _convert_messages(self, messages: List[LLMMessage]) -> tuple[Optional[str], List[Dict]]:
        """Convert messages to Google AI format."""
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == LLMRole.SYSTEM:
                system_instruction = msg.content
            elif msg.role == LLMRole.USER:
                parts = [{"text": msg.content}]
                if msg.images:
                    for image in msg.images:
                        if image.startswith("http"):
                            parts.append({"file_data": {"file_uri": image}})
                        else:
                            import base64
                            parts.append({
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image,
                                }
                            })
                contents.append({"role": "user", "parts": parts})
            elif msg.role == LLMRole.ASSISTANT:
                contents.append({"role": "model", "parts": [{"text": msg.content}]})

        return system_instruction, contents

    def _convert_tools(self, tools: Optional[List[Dict]]) -> Optional[List]:
        """Convert OpenAI-style tools to Google format."""
        if not tools:
            return None

        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                function_declarations.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })

        return [{"function_declarations": function_declarations}] if function_declarations else None

    async def complete(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a completion using Google AI."""
        genai = await self._get_client()
        config = config or LLMConfig()

        start_time = time.time()

        try:
            system_instruction, contents = self._convert_messages(messages)

            # Create model
            generation_config = {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_output_tokens": config.max_tokens or 4096,
            }

            if config.stop:
                generation_config["stop_sequences"] = config.stop

            model = genai.GenerativeModel(
                model_name=config.model,
                generation_config=generation_config,
                system_instruction=system_instruction,
            )

            # Make request
            response = await asyncio.to_thread(
                model.generate_content,
                contents,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract response
            content = response.text if response.text else ""

            # Handle tool calls
            tool_calls = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        if tool_calls is None:
                            tool_calls = []
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(dict(fc.args)),
                            }
                        })

            # Get usage
            usage_metadata = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0
            completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0

            # Calculate cost
            model_info = get_model_info(config.model)
            cost = model_info.calculate_cost(prompt_tokens, completion_tokens) if model_info else 0.0

            return LLMResponse(
                content=content,
                role=LLMRole.ASSISTANT,
                tool_calls=tool_calls,
                usage=LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                ),
                model=config.model,
                provider=self.provider_type,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
            )

        except Exception as e:
            logger.error(f"Google AI completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion using Google AI."""
        genai = await self._get_client()
        config = config or LLMConfig()

        try:
            system_instruction, contents = self._convert_messages(messages)

            # Create model
            generation_config = {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_output_tokens": config.max_tokens or 4096,
            }

            if config.stop:
                generation_config["stop_sequences"] = config.stop

            model = genai.GenerativeModel(
                model_name=config.model,
                generation_config=generation_config,
                system_instruction=system_instruction,
            )

            # Generate with streaming
            response = await asyncio.to_thread(
                model.generate_content,
                contents,
                stream=True,
            )

            for chunk in response:
                if chunk.text:
                    yield StreamChunk(
                        content=chunk.text,
                        role=LLMRole.ASSISTANT,
                    )

            # Final chunk
            yield StreamChunk(
                content="",
                finish_reason="stop",
            )

        except Exception as e:
            logger.error(f"Google AI streaming error: {e}")
            raise


class CohereProvider(BaseLLMProvider):
    """Cohere API provider."""

    provider_type = LLMProvider.COHERE

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config)

        self._api_key = (
            api_key or
            self.config.api_key or
            os.environ.get("COHERE_API_KEY")
        )

    async def _get_client(self):
        """Get or create Cohere client."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.AsyncClient(api_key=self._api_key)
            except ImportError:
                raise ImportError("cohere package required. Install with: pip install cohere")
        return self._client

    async def complete(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a completion using Cohere."""
        client = await self._get_client()
        config = config or LLMConfig()

        start_time = time.time()

        try:
            # Convert messages to Cohere format
            preamble = None
            chat_history = []
            message = ""

            for msg in messages:
                if msg.role == LLMRole.SYSTEM:
                    preamble = msg.content
                elif msg.role == LLMRole.USER:
                    if chat_history or preamble:
                        chat_history.append({"role": "USER", "message": msg.content})
                    else:
                        message = msg.content
                elif msg.role == LLMRole.ASSISTANT:
                    chat_history.append({"role": "CHATBOT", "message": msg.content})

            # Use last user message as the query
            if not message and chat_history:
                for i in range(len(chat_history) - 1, -1, -1):
                    if chat_history[i]["role"] == "USER":
                        message = chat_history[i]["message"]
                        chat_history = chat_history[:i]
                        break

            response = await client.chat(
                model=config.model,
                message=message,
                preamble=preamble,
                chat_history=chat_history if chat_history else None,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            latency_ms = (time.time() - start_time) * 1000

            return LLMResponse(
                content=response.text,
                role=LLMRole.ASSISTANT,
                usage=LLMUsage(
                    prompt_tokens=response.meta.tokens.input_tokens if response.meta else 0,
                    completion_tokens=response.meta.tokens.output_tokens if response.meta else 0,
                    total_tokens=(
                        (response.meta.tokens.input_tokens + response.meta.tokens.output_tokens)
                        if response.meta else 0
                    ),
                    latency_ms=latency_ms,
                ),
                model=config.model,
                provider=self.provider_type,
                finish_reason=response.finish_reason,
            )

        except Exception as e:
            logger.error(f"Cohere completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion using Cohere."""
        client = await self._get_client()
        config = config or LLMConfig()

        try:
            # Convert messages
            preamble = None
            chat_history = []
            message = ""

            for msg in messages:
                if msg.role == LLMRole.SYSTEM:
                    preamble = msg.content
                elif msg.role == LLMRole.USER:
                    if chat_history or preamble:
                        chat_history.append({"role": "USER", "message": msg.content})
                    else:
                        message = msg.content
                elif msg.role == LLMRole.ASSISTANT:
                    chat_history.append({"role": "CHATBOT", "message": msg.content})

            if not message and chat_history:
                for i in range(len(chat_history) - 1, -1, -1):
                    if chat_history[i]["role"] == "USER":
                        message = chat_history[i]["message"]
                        chat_history = chat_history[:i]
                        break

            async for event in client.chat_stream(
                model=config.model,
                message=message,
                preamble=preamble,
                chat_history=chat_history if chat_history else None,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            ):
                if event.event_type == "text-generation":
                    yield StreamChunk(
                        content=event.text,
                        role=LLMRole.ASSISTANT,
                    )
                elif event.event_type == "stream-end":
                    yield StreamChunk(
                        content="",
                        finish_reason=event.finish_reason,
                    )

        except Exception as e:
            logger.error(f"Cohere streaming error: {e}")
            raise


class GroqProvider(BaseLLMProvider):
    """Groq API provider (fast inference)."""

    provider_type = LLMProvider.GROQ

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config)

        self._api_key = (
            api_key or
            self.config.api_key or
            os.environ.get("GROQ_API_KEY")
        )

    async def _get_client(self):
        """Get or create Groq client."""
        if self._client is None:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(
                    api_key=self._api_key,
                    timeout=self.config.read_timeout_ms / 1000,
                    max_retries=self.config.max_retries,
                )
            except ImportError:
                raise ImportError("groq package required. Install with: pip install groq")
        return self._client

    async def complete(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a completion using Groq."""
        client = await self._get_client()
        config = config or LLMConfig()

        start_time = time.time()

        try:
            # Convert messages
            converted = [msg.to_dict() for msg in messages]

            # Prepare params
            params = {
                "model": config.model,
                "messages": converted,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens or 4096,
                "top_p": config.top_p,
            }

            if config.stop:
                params["stop"] = config.stop

            if config.tools:
                params["tools"] = config.tools

            response = await client.chat.completions.create(**params)

            latency_ms = (time.time() - start_time) * 1000

            choice = response.choices[0]
            message = choice.message

            # Build tool calls
            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]

            return LLMResponse(
                content=message.content or "",
                role=LLMRole.ASSISTANT,
                tool_calls=tool_calls,
                usage=LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_tokens=response.usage.total_tokens if response.usage else 0,
                    latency_ms=latency_ms,
                ),
                model=response.model,
                provider=self.provider_type,
                finish_reason=choice.finish_reason,
                request_id=response.id,
            )

        except Exception as e:
            logger.error(f"Groq completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion using Groq."""
        client = await self._get_client()
        config = config or LLMConfig()

        try:
            converted = [msg.to_dict() for msg in messages]

            params = {
                "model": config.model,
                "messages": converted,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens or 4096,
                "top_p": config.top_p,
                "stream": True,
            }

            if config.stop:
                params["stop"] = config.stop

            stream = await client.chat.completions.create(**params)

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                yield StreamChunk(
                    content=delta.content or "",
                    role=LLMRole.ASSISTANT if delta.role else None,
                    finish_reason=finish_reason,
                    model=chunk.model,
                )

        except Exception as e:
            logger.error(f"Groq streaming error: {e}")
            raise


class TogetherProvider(BaseLLMProvider):
    """Together AI provider."""

    provider_type = LLMProvider.TOGETHER

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config)

        self._api_key = (
            api_key or
            self.config.api_key or
            os.environ.get("TOGETHER_API_KEY")
        )

    async def _get_client(self):
        """Get or create Together client."""
        if self._client is None:
            try:
                from together import AsyncTogether
                self._client = AsyncTogether(api_key=self._api_key)
            except ImportError:
                raise ImportError("together package required. Install with: pip install together")
        return self._client

    async def complete(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a completion using Together."""
        client = await self._get_client()
        config = config or LLMConfig()

        start_time = time.time()

        try:
            converted = [msg.to_dict() for msg in messages]

            response = await client.chat.completions.create(
                model=config.model,
                messages=converted,
                temperature=config.temperature,
                max_tokens=config.max_tokens or 4096,
                top_p=config.top_p,
                stop=config.stop,
            )

            latency_ms = (time.time() - start_time) * 1000

            choice = response.choices[0]

            return LLMResponse(
                content=choice.message.content or "",
                role=LLMRole.ASSISTANT,
                usage=LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_tokens=response.usage.total_tokens if response.usage else 0,
                    latency_ms=latency_ms,
                ),
                model=response.model,
                provider=self.provider_type,
                finish_reason=choice.finish_reason,
                request_id=response.id,
            )

        except Exception as e:
            logger.error(f"Together completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion using Together."""
        client = await self._get_client()
        config = config or LLMConfig()

        try:
            converted = [msg.to_dict() for msg in messages]

            stream = await client.chat.completions.create(
                model=config.model,
                messages=converted,
                temperature=config.temperature,
                max_tokens=config.max_tokens or 4096,
                top_p=config.top_p,
                stop=config.stop,
                stream=True,
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                yield StreamChunk(
                    content=delta.content or "" if hasattr(delta, "content") else "",
                    role=LLMRole.ASSISTANT,
                    finish_reason=finish_reason,
                    model=chunk.model if hasattr(chunk, "model") else config.model,
                )

        except Exception as e:
            logger.error(f"Together streaming error: {e}")
            raise


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _providers: Dict[str, Type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "cohere": CohereProvider,
        "groq": GroqProvider,
        "together": TogetherProvider,
    }

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseLLMProvider]) -> None:
        """Register a new provider class."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def create(
        cls,
        provider: str,
        config: Optional[ProviderConfig] = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """Create a provider instance."""
        provider_lower = provider.lower()

        if provider_lower not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(cls._providers.keys())}")

        provider_class = cls._providers[provider_lower]
        return provider_class(config=config, **kwargs)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._providers.keys())


__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "CohereProvider",
    "GroqProvider",
    "TogetherProvider",
    "LLMProviderFactory",
]
