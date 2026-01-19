"""Groq LLM adapter for ultra-fast inference."""

import json
import time
import structlog
from typing import Optional, List, Dict, Any, AsyncIterator
import httpx

from app.llm.base import (
    LLMAdapter,
    LLMMessage,
    LLMResponse,
    StreamChunk,
    ToolCall,
    MessageRole,
)


logger = structlog.get_logger()


class GroqAdapter(LLMAdapter):
    """
    Groq adapter for ultra-fast LLM inference.

    Supports:
    - Llama 3.1, Mixtral, Gemma
    - Streaming
    - Tool calling
    - Very low latency
    """

    name = "groq"
    supports_streaming = True
    supports_tools = True
    supports_vision = False

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-70b-versatile",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

        self._client: Optional[httpx.AsyncClient] = None

        # Statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._total_latency = 0.0

    async def connect(self) -> None:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        logger.info("groq_adapter_connected", model=self.model)

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("groq_adapter_disconnected")

    async def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate response from Groq."""
        if not self._client:
            await self.connect()

        start_time = time.perf_counter()
        self._total_requests += 1

        # Build request (OpenAI-compatible API)
        request_body: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": self.format_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            request_body["tools"] = tools
            request_body["tool_choice"] = kwargs.get("tool_choice", "auto")

        try:
            response = await self._client.post(
                "/chat/completions",
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.perf_counter() - start_time) * 1000
            self._total_latency += latency_ms

            # Parse response
            choice = data["choices"][0]
            message = choice["message"]

            # Extract tool calls
            tool_calls = []
            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}

                    tool_calls.append(ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=args,
                    ))

            # Track usage
            usage = data.get("usage", {})
            self._total_tokens += usage.get("total_tokens", 0)

            # Groq provides timing info
            timing = data.get("x_groq", {}).get("usage", {})

            logger.debug(
                "groq_generate_complete",
                model=self.model,
                latency_ms=round(latency_ms, 2),
                tokens=usage.get("total_tokens"),
                groq_queue_time=timing.get("queue_time"),
                groq_prompt_time=timing.get("prompt_time"),
                groq_completion_time=timing.get("completion_time"),
            )

            return LLMResponse(
                content=message.get("content", "") or "",
                role=MessageRole.ASSISTANT,
                tool_calls=tool_calls,
                finish_reason=choice.get("finish_reason"),
                usage=usage,
                model=data.get("model"),
                latency_ms=latency_ms,
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                "groq_generate_error",
                status=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error("groq_generate_error", error=str(e))
            raise

    async def stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response from Groq."""
        if not self._client:
            await self.connect()

        self._total_requests += 1

        # Build request
        request_body: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": self.format_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if tools:
            request_body["tools"] = tools
            request_body["tool_choice"] = kwargs.get("tool_choice", "auto")

        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json=request_body,
            ) as response:
                response.raise_for_status()

                tool_call_buffer: Dict[int, Dict[str, Any]] = {}

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]

                    if data == "[DONE]":
                        yield StreamChunk(is_final=True)
                        break

                    try:
                        chunk = json.loads(data)
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})

                        content = delta.get("content", "")

                        tool_delta = None
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                if idx not in tool_call_buffer:
                                    tool_call_buffer[idx] = {
                                        "id": tc.get("id", ""),
                                        "name": "",
                                        "arguments": "",
                                    }

                                if "id" in tc:
                                    tool_call_buffer[idx]["id"] = tc["id"]
                                if "function" in tc:
                                    if "name" in tc["function"]:
                                        tool_call_buffer[idx]["name"] = tc["function"]["name"]
                                    if "arguments" in tc["function"]:
                                        tool_call_buffer[idx]["arguments"] += tc["function"]["arguments"]

                                tool_delta = tool_call_buffer[idx]

                        yield StreamChunk(
                            content=content,
                            tool_call_delta=tool_delta,
                            finish_reason=choice.get("finish_reason"),
                        )

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error("groq_stream_error", error=str(e))
            raise

    def format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Format messages for Groq (OpenAI-compatible)."""
        return [m.to_openai() for m in messages]

    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            "name": self.name,
            "model": self.model,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "average_latency_ms": round(
                self._total_latency / max(1, self._total_requests), 2
            ),
        }
