"""OpenAI LLM adapter with streaming and function calling."""

import json
from typing import AsyncIterator, Optional

import structlog
from openai import AsyncOpenAI

from app.llm.base import LLMAdapter, LLMResponse, Message, Tool, FunctionCall
from app.config import get_settings

logger = structlog.get_logger()


class OpenAIAdapter(LLMAdapter):
    """
    OpenAI adapter using the official SDK.

    Supports:
    - GPT-4, GPT-4o, GPT-3.5-turbo models
    - Streaming responses
    - Function/tool calling
    - JSON mode
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = settings.openai_model
        self.max_tokens = settings.openai_max_tokens
        self.temperature = settings.openai_temperature
        self.default_system_prompt = settings.default_system_prompt

        self.logger = logger.bind(adapter="openai")

    @property
    def name(self) -> str:
        return "openai"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_functions(self) -> bool:
        return True

    def _convert_messages(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """Convert our Message format to OpenAI format."""
        result = []

        # Add system prompt
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        elif self.default_system_prompt:
            result.append({"role": "system", "content": self.default_system_prompt})

        # Add conversation messages
        for msg in messages:
            message = {"role": msg.role, "content": msg.content}

            if msg.name:
                message["name"] = msg.name

            if msg.function_call:
                message["tool_calls"] = [{
                    "id": msg.function_call.id or "call_1",
                    "type": "function",
                    "function": {
                        "name": msg.function_call.name,
                        "arguments": json.dumps(msg.function_call.arguments),
                    },
                }]

            result.append(message)

        return result

    def _convert_tools(self, tools: list[Tool]) -> list[dict]:
        """Convert our Tool format to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
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
        """Generate a response from OpenAI."""
        openai_messages = self._convert_messages(messages, system_prompt)

        request_params = {
            "model": kwargs.get("model", self.model),
            "messages": openai_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if tools:
            request_params["tools"] = self._convert_tools(tools)
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        try:
            response = await self.client.chat.completions.create(**request_params)

            choice = response.choices[0]
            message = choice.message

            # Extract function calls
            function_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.type == "function":
                        function_calls.append(
                            FunctionCall(
                                id=tool_call.id,
                                name=tool_call.function.name,
                                arguments=json.loads(tool_call.function.arguments),
                            )
                        )

            self.logger.info(
                "Generated response",
                model=self.model,
                tokens=response.usage.total_tokens if response.usage else 0,
                has_function_calls=len(function_calls) > 0,
            )

            return LLMResponse(
                text=message.content or "",
                function_calls=function_calls,
                finish_reason=choice.finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
                is_complete=True,
            )

        except Exception as e:
            self.logger.error("OpenAI API error", error=str(e))
            raise

    async def generate_stream(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        **kwargs,
    ) -> AsyncIterator[LLMResponse]:
        """Stream responses from OpenAI."""
        openai_messages = self._convert_messages(messages, system_prompt)

        request_params = {
            "model": kwargs.get("model", self.model),
            "messages": openai_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
        }

        if tools:
            request_params["tools"] = self._convert_tools(tools)

        try:
            stream = await self.client.chat.completions.create(**request_params)

            full_text = ""
            function_calls = []
            current_tool_calls = {}  # id -> {name, arguments}

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # Handle text content
                if delta.content:
                    full_text += delta.content
                    yield LLMResponse(
                        text=delta.content,
                        is_partial=True,
                        is_complete=False,
                    )

                # Handle tool calls (streamed incrementally)
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        idx = tool_call.index
                        if idx not in current_tool_calls:
                            current_tool_calls[idx] = {
                                "id": tool_call.id or f"call_{idx}",
                                "name": "",
                                "arguments": "",
                            }

                        if tool_call.function:
                            if tool_call.function.name:
                                current_tool_calls[idx]["name"] = tool_call.function.name
                            if tool_call.function.arguments:
                                current_tool_calls[idx]["arguments"] += tool_call.function.arguments

                # Handle completion
                if finish_reason:
                    # Parse accumulated tool calls
                    for tc in current_tool_calls.values():
                        try:
                            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                            function_calls.append(
                                FunctionCall(
                                    id=tc["id"],
                                    name=tc["name"],
                                    arguments=args,
                                )
                            )
                        except json.JSONDecodeError:
                            pass

                    yield LLMResponse(
                        text=full_text,
                        function_calls=function_calls,
                        finish_reason=finish_reason,
                        is_partial=False,
                        is_complete=True,
                    )

            self.logger.info(
                "Streamed response",
                model=self.model,
                text_length=len(full_text),
                function_calls=len(function_calls),
            )

        except Exception as e:
            self.logger.error("OpenAI streaming error", error=str(e))
            raise

    async def close(self) -> None:
        """Close the OpenAI client."""
        await self.client.close()
