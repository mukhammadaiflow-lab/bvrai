"""Mock LLM adapter for testing."""

import asyncio
import random
from typing import AsyncIterator, Optional

import structlog

from app.llm.base import LLMAdapter, LLMResponse, Message, Tool, FunctionCall

logger = structlog.get_logger()


class MockLLMAdapter(LLMAdapter):
    """
    Mock LLM adapter for testing without API costs.

    Uses pattern matching to generate contextual responses.
    """

    RESPONSES = {
        "greeting": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Good day! How may I assist you?",
        ],
        "appointment": [
            "I'd be happy to help you schedule an appointment. What date works best for you?",
            "Sure, I can help with that. When would you like to come in?",
        ],
        "hours": [
            "We're open Monday through Friday, 9 AM to 5 PM.",
            "Our business hours are 9 to 5, Monday through Friday.",
        ],
        "thanks": [
            "You're welcome! Is there anything else I can help you with?",
            "Happy to help! Anything else?",
        ],
        "goodbye": [
            "Goodbye! Have a great day!",
            "Thank you for calling. Goodbye!",
        ],
        "default": [
            "I understand. Could you tell me more about what you need?",
            "Let me help you with that. What specifically would you like to know?",
            "I'm here to help. Can you provide more details?",
        ],
    }

    PATTERNS = [
        (["hello", "hi", "hey", "good morning", "good afternoon"], "greeting"),
        (["appointment", "schedule", "book", "reservation"], "appointment"),
        (["hours", "open", "close", "time"], "hours"),
        (["thank", "thanks", "appreciate"], "thanks"),
        (["bye", "goodbye", "see you", "that's all"], "goodbye"),
    ]

    def __init__(self, latency_ms: int = 100) -> None:
        self.latency_ms = latency_ms
        self.logger = logger.bind(adapter="mock")

    @property
    def name(self) -> str:
        return "mock"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_functions(self) -> bool:
        return True

    def _get_response(self, text: str) -> str:
        """Get a response based on the input text."""
        text_lower = text.lower()

        # Match patterns
        for keywords, category in self.PATTERNS:
            for keyword in keywords:
                if keyword in text_lower:
                    return random.choice(self.RESPONSES[category])

        return random.choice(self.RESPONSES["default"])

    async def generate(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a mock response."""
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Get last user message
        last_message = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_message = msg.content
                break

        response_text = self._get_response(last_message)

        # Check for function calling patterns
        function_calls = []
        if tools:
            text_lower = last_message.lower()
            for tool in tools:
                if tool.name.lower() in text_lower or any(
                    kw in text_lower for kw in ["book", "schedule", "appointment"]
                ):
                    if tool.name == "book_appointment":
                        function_calls.append(
                            FunctionCall(
                                id="mock_call_1",
                                name=tool.name,
                                arguments={"date": "tomorrow", "time": "10:00 AM"},
                            )
                        )
                        response_text = "I'll book that appointment for you."
                        break

        self.logger.info(
            "Generated mock response",
            input=last_message[:50],
            output=response_text[:50],
        )

        return LLMResponse(
            text=response_text,
            function_calls=function_calls,
            finish_reason="stop",
            is_complete=True,
        )

    async def generate_stream(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        **kwargs,
    ) -> AsyncIterator[LLMResponse]:
        """Stream mock responses word by word."""
        # Simulate initial latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Get last user message
        last_message = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_message = msg.content
                break

        response_text = self._get_response(last_message)

        # Stream word by word
        words = response_text.split()
        full_text = ""

        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            full_text += chunk

            yield LLMResponse(
                text=chunk,
                is_partial=True,
                is_complete=False,
            )

            # Simulate typing speed
            await asyncio.sleep(0.05)

        # Final response
        yield LLMResponse(
            text=full_text,
            finish_reason="stop",
            is_partial=False,
            is_complete=True,
        )

        self.logger.info(
            "Streamed mock response",
            input=last_message[:50],
            output=full_text[:50],
        )
