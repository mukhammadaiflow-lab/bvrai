"""Anthropic Claude LLM adapter with streaming and tool use."""

import json
from typing import AsyncIterator, Optional

import structlog
from anthropic import AsyncAnthropic

from app.llm.base import LLMAdapter, LLMResponse, Message, Tool, FunctionCall
from app.config import get_settings

logger = structlog.get_logger()


class AnthropicAdapter(LLMAdapter):
    """
    Anthropic Claude adapter.

    Supports:
    - Claude 3 models (Opus, Sonnet, Haiku)
    - Streaming responses
    - Tool use
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key

        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        self.client = AsyncAnthropic(api_key=self.api_key)
        self.model = settings.anthropic_model
        self.max_tokens = settings.anthropic_max_tokens
        self.default_system_prompt = settings.default_system_prompt

        self.logger = logger.bind(adapter="anthropic")

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_functions(self) -> bool:
        return True

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert our Message format to Anthropic format."""
        result = []

        for msg in messages:
            # Anthropic uses "user" and "assistant" roles
            role = msg.role
            if role == "function":
                role = "user"  # Function results come from user

            content = msg.content

            # Handle tool results
            if msg.name and msg.role == "function":
                content = f"[Function {msg.name} result]: {msg.content}"

            result.append({"role": role, "content": content})

        return result

    def _convert_tools(self, tools: list[Tool]) -> list[dict]:
        """Convert our Tool format to Anthropic format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    async def generate(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from Claude."""
        anthropic_messages = self._convert_messages(messages)
        system = system_prompt or self.default_system_prompt

        request_params = {
            "model": kwargs.get("model", self.model),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "system": system,
            "messages": anthropic_messages,
        }

        if tools:
            request_params["tools"] = self._convert_tools(tools)

        try:
            response = await self.client.messages.create(**request_params)

            # Extract text and tool use
            text = ""
            function_calls = []

            for block in response.content:
                if block.type == "text":
                    text += block.text
                elif block.type == "tool_use":
                    function_calls.append(
                        FunctionCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        )
                    )

            self.logger.info(
                "Generated response",
                model=self.model,
                tokens=response.usage.input_tokens + response.usage.output_tokens,
                has_tool_use=len(function_calls) > 0,
            )

            return LLMResponse(
                text=text,
                function_calls=function_calls,
                finish_reason=response.stop_reason,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                is_complete=True,
            )

        except Exception as e:
            self.logger.error("Anthropic API error", error=str(e))
            raise

    async def generate_stream(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        **kwargs,
    ) -> AsyncIterator[LLMResponse]:
        """Stream responses from Claude."""
        anthropic_messages = self._convert_messages(messages)
        system = system_prompt or self.default_system_prompt

        request_params = {
            "model": kwargs.get("model", self.model),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "system": system,
            "messages": anthropic_messages,
        }

        if tools:
            request_params["tools"] = self._convert_tools(tools)

        try:
            async with self.client.messages.stream(**request_params) as stream:
                full_text = ""
                function_calls = []

                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            text_delta = event.delta.text
                            full_text += text_delta
                            yield LLMResponse(
                                text=text_delta,
                                is_partial=True,
                                is_complete=False,
                            )

                    elif event.type == "content_block_stop":
                        pass  # Block finished

                    elif event.type == "message_stop":
                        yield LLMResponse(
                            text=full_text,
                            function_calls=function_calls,
                            is_partial=False,
                            is_complete=True,
                        )

            self.logger.info(
                "Streamed response",
                model=self.model,
                text_length=len(full_text),
            )

        except Exception as e:
            self.logger.error("Anthropic streaming error", error=str(e))
            raise

    async def close(self) -> None:
        """Close the Anthropic client."""
        await self.client.close()
