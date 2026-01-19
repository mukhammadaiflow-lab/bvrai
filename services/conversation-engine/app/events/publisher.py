"""Event publisher utilities."""

import asyncio
import structlog
from typing import Optional, Dict, Any
from datetime import datetime

from app.events.bus import Event, EventType, event_bus


logger = structlog.get_logger()


class EventPublisher:
    """
    Helper class for publishing events.

    Provides convenience methods for common event types
    and handles correlation tracking.
    """

    def __init__(
        self,
        source: str = "conversation-engine",
        default_session_id: Optional[str] = None,
    ):
        self.source = source
        self.default_session_id = default_session_id
        self._correlation_id: Optional[str] = None

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for event tracking."""
        self._correlation_id = correlation_id

    def set_session(self, session_id: str) -> None:
        """Set default session ID."""
        self.default_session_id = session_id

    async def publish(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Event:
        """Publish an event."""
        event = Event(
            type=event_type,
            data=data,
            session_id=session_id or self.default_session_id,
            source=self.source,
            correlation_id=self._correlation_id,
        )
        await event_bus.publish(event)
        return event

    # Call lifecycle events
    async def call_started(
        self,
        call_id: str,
        direction: str,
        caller_number: Optional[str] = None,
        agent_id: Optional[str] = None,
        **extra,
    ) -> Event:
        """Publish call started event."""
        return await self.publish(
            EventType.CALL_STARTED,
            {
                "call_id": call_id,
                "direction": direction,
                "caller_number": caller_number,
                "agent_id": agent_id,
                "started_at": datetime.utcnow().isoformat(),
                **extra,
            },
        )

    async def call_connected(
        self,
        call_id: str,
        connect_time_ms: Optional[int] = None,
        **extra,
    ) -> Event:
        """Publish call connected event."""
        return await self.publish(
            EventType.CALL_CONNECTED,
            {
                "call_id": call_id,
                "connect_time_ms": connect_time_ms,
                **extra,
            },
        )

    async def call_ended(
        self,
        call_id: str,
        reason: str,
        duration_seconds: float,
        **extra,
    ) -> Event:
        """Publish call ended event."""
        return await self.publish(
            EventType.CALL_ENDED,
            {
                "call_id": call_id,
                "reason": reason,
                "duration_seconds": duration_seconds,
                "ended_at": datetime.utcnow().isoformat(),
                **extra,
            },
        )

    # Conversation events
    async def turn_started(
        self,
        turn_id: str,
        speaker: str,  # "user" or "agent"
        **extra,
    ) -> Event:
        """Publish turn started event."""
        return await self.publish(
            EventType.TURN_STARTED,
            {
                "turn_id": turn_id,
                "speaker": speaker,
                "started_at": datetime.utcnow().isoformat(),
                **extra,
            },
        )

    async def turn_ended(
        self,
        turn_id: str,
        speaker: str,
        duration_ms: int,
        **extra,
    ) -> Event:
        """Publish turn ended event."""
        return await self.publish(
            EventType.TURN_ENDED,
            {
                "turn_id": turn_id,
                "speaker": speaker,
                "duration_ms": duration_ms,
                **extra,
            },
        )

    async def user_speaking(
        self,
        is_speaking: bool,
        confidence: Optional[float] = None,
        **extra,
    ) -> Event:
        """Publish user speaking status."""
        return await self.publish(
            EventType.USER_SPEAKING,
            {
                "is_speaking": is_speaking,
                "confidence": confidence,
                **extra,
            },
        )

    async def agent_speaking(
        self,
        is_speaking: bool,
        text: Optional[str] = None,
        **extra,
    ) -> Event:
        """Publish agent speaking status."""
        return await self.publish(
            EventType.AGENT_SPEAKING,
            {
                "is_speaking": is_speaking,
                "text": text,
                **extra,
            },
        )

    # Transcription events
    async def transcript_partial(
        self,
        text: str,
        confidence: float,
        **extra,
    ) -> Event:
        """Publish partial transcript."""
        return await self.publish(
            EventType.TRANSCRIPT_PARTIAL,
            {
                "text": text,
                "confidence": confidence,
                "is_final": False,
                **extra,
            },
        )

    async def transcript_final(
        self,
        text: str,
        confidence: float,
        duration_ms: int,
        **extra,
    ) -> Event:
        """Publish final transcript."""
        return await self.publish(
            EventType.TRANSCRIPT_FINAL,
            {
                "text": text,
                "confidence": confidence,
                "duration_ms": duration_ms,
                "is_final": True,
                **extra,
            },
        )

    # AI events
    async def llm_request(
        self,
        model: str,
        prompt_tokens: int,
        **extra,
    ) -> Event:
        """Publish LLM request event."""
        return await self.publish(
            EventType.LLM_REQUEST,
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "requested_at": datetime.utcnow().isoformat(),
                **extra,
            },
        )

    async def llm_response(
        self,
        model: str,
        response_text: str,
        latency_ms: int,
        completion_tokens: int,
        **extra,
    ) -> Event:
        """Publish LLM response event."""
        return await self.publish(
            EventType.LLM_RESPONSE,
            {
                "model": model,
                "response_text": response_text[:500],  # Truncate for logging
                "latency_ms": latency_ms,
                "completion_tokens": completion_tokens,
                **extra,
            },
        )

    async def function_called(
        self,
        function_name: str,
        arguments: Dict[str, Any],
        **extra,
    ) -> Event:
        """Publish function called event."""
        return await self.publish(
            EventType.FUNCTION_CALLED,
            {
                "function_name": function_name,
                "arguments": arguments,
                "called_at": datetime.utcnow().isoformat(),
                **extra,
            },
        )

    async def function_result(
        self,
        function_name: str,
        result: Any,
        latency_ms: int,
        success: bool = True,
        **extra,
    ) -> Event:
        """Publish function result event."""
        return await self.publish(
            EventType.FUNCTION_RESULT,
            {
                "function_name": function_name,
                "result": result,
                "latency_ms": latency_ms,
                "success": success,
                **extra,
            },
        )

    # Error events
    async def error_occurred(
        self,
        error_type: str,
        error_message: str,
        recoverable: bool = True,
        **extra,
    ) -> Event:
        """Publish error event."""
        return await self.publish(
            EventType.ERROR_OCCURRED,
            {
                "error_type": error_type,
                "error": error_message,
                "recoverable": recoverable,
                "occurred_at": datetime.utcnow().isoformat(),
                **extra,
            },
        )

    # Analytics events
    async def record_metric(
        self,
        metric_name: str,
        value: float,
        unit: Optional[str] = None,
        **extra,
    ) -> Event:
        """Record a metric."""
        return await self.publish(
            EventType.METRIC_RECORDED,
            {
                "metric_name": metric_name,
                "value": value,
                "unit": unit,
                **extra,
            },
        )

    async def record_latency(
        self,
        operation: str,
        latency_ms: float,
        **extra,
    ) -> Event:
        """Record latency measurement."""
        return await self.publish(
            EventType.LATENCY_RECORDED,
            {
                "operation": operation,
                "latency_ms": latency_ms,
                **extra,
            },
        )


# Default publisher
default_publisher = EventPublisher()
