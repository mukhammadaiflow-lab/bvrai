"""
Event Collector.

Collects, validates, and buffers analytics events.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional
import uuid

from ..config import EventType, get_settings
from ..models import AnalyticsEvent, CallEvent, ConversationEvent, LatencyEvent

logger = logging.getLogger(__name__)


class EventCollector:
    """
    Collects and buffers analytics events.

    Features:
    - Event validation
    - Buffered ingestion
    - Automatic flushing
    - Back-pressure handling
    """

    def __init__(self):
        """Initialize collector."""
        self.settings = get_settings()
        self.config = self.settings.collector

        # Event buffer
        self._buffer: Deque[AnalyticsEvent] = deque(maxlen=self.config.buffer_size)
        self._buffer_lock = asyncio.Lock()

        # Metrics
        self._events_received = 0
        self._events_processed = 0
        self._events_dropped = 0
        self._events_invalid = 0

        # Callbacks
        self._on_flush: Optional[Callable[[List[AnalyticsEvent]], None]] = None

        # Background flush task
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the collector."""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Event collector started")

    async def stop(self) -> None:
        """Stop the collector and flush remaining events."""
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush()
        logger.info("Event collector stopped")

    def set_flush_callback(
        self,
        callback: Callable[[List[AnalyticsEvent]], None],
    ) -> None:
        """Set callback for when events are flushed."""
        self._on_flush = callback

    async def collect(self, event: AnalyticsEvent) -> bool:
        """
        Collect a single event.

        Args:
            event: Event to collect

        Returns:
            True if event was accepted
        """
        # Validate
        if self.config.validate_events:
            if not self._validate_event(event):
                self._events_invalid += 1
                if self.config.drop_invalid:
                    return False
                logger.warning(f"Invalid event accepted: {event.event_id}")

        self._events_received += 1

        # Add to buffer
        async with self._buffer_lock:
            if len(self._buffer) >= self.config.buffer_size:
                self._events_dropped += 1
                logger.warning("Event buffer full, dropping oldest event")

            self._buffer.append(event)

        # Check if we should flush
        if len(self._buffer) >= self.config.batch_size:
            await self._flush()

        return True

    async def collect_batch(self, events: List[AnalyticsEvent]) -> int:
        """
        Collect multiple events.

        Args:
            events: Events to collect

        Returns:
            Number of events accepted
        """
        accepted = 0
        for event in events:
            if await self.collect(event):
                accepted += 1
        return accepted

    async def collect_raw(
        self,
        event_type: str,
        data: Dict[str, Any],
        organization_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        call_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Collect event from raw data.

        Args:
            event_type: Event type string
            data: Event data
            organization_id: Organization ID
            agent_id: Agent ID
            call_id: Call ID
            session_id: Session ID
            timestamp: Event timestamp

        Returns:
            True if event was accepted
        """
        try:
            etype = EventType(event_type)
        except ValueError:
            logger.warning(f"Unknown event type: {event_type}")
            self._events_invalid += 1
            return False

        event = self._create_event(
            event_type=etype,
            data=data,
            organization_id=organization_id,
            agent_id=agent_id,
            call_id=call_id,
            session_id=session_id,
            timestamp=timestamp,
        )

        return await self.collect(event)

    def _create_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        **kwargs,
    ) -> AnalyticsEvent:
        """Create appropriate event type."""
        common = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": kwargs.get("timestamp") or datetime.utcnow(),
            "organization_id": kwargs.get("organization_id"),
            "agent_id": kwargs.get("agent_id"),
            "call_id": kwargs.get("call_id"),
            "session_id": kwargs.get("session_id"),
            "data": data,
            "source": data.get("source", "api"),
        }

        # Create specific event type
        if event_type in [
            EventType.CALL_STARTED,
            EventType.CALL_CONNECTED,
            EventType.CALL_ENDED,
            EventType.CALL_TRANSFERRED,
            EventType.CALL_FAILED,
        ]:
            return CallEvent(
                **common,
                caller_number=data.get("caller_number"),
                called_number=data.get("called_number"),
                direction=data.get("direction", "inbound"),
                outcome=data.get("outcome"),
                duration_ms=data.get("duration_ms"),
            )

        elif event_type in [
            EventType.SPEECH_STARTED,
            EventType.SPEECH_ENDED,
            EventType.INTENT_DETECTED,
            EventType.ENTITY_EXTRACTED,
            EventType.SENTIMENT_CHANGED,
        ]:
            return ConversationEvent(
                **common,
                transcript=data.get("transcript"),
                intent=data.get("intent"),
                intent_confidence=data.get("intent_confidence"),
                entities=data.get("entities", {}),
                sentiment=data.get("sentiment"),
                sentiment_score=data.get("sentiment_score"),
            )

        elif event_type == EventType.LATENCY_RECORDED:
            from ..config import LatencyComponent

            return LatencyEvent(
                **common,
                component=LatencyComponent(data.get("component", "total")),
                latency_ms=data.get("latency_ms", 0.0),
                success=data.get("success", True),
                error=data.get("error"),
            )

        else:
            return AnalyticsEvent(**common)

    def _validate_event(self, event: AnalyticsEvent) -> bool:
        """Validate an event."""
        # Check required fields
        if not event.event_id:
            return False

        if not event.event_type:
            return False

        if not event.timestamp:
            return False

        # Check timestamp is not in the future
        if event.timestamp > datetime.utcnow():
            logger.warning(f"Event timestamp in future: {event.timestamp}")
            # Allow but log

        return True

    async def _flush(self) -> None:
        """Flush buffered events to processor."""
        async with self._buffer_lock:
            if not self._buffer:
                return

            # Get batch
            batch_size = min(len(self._buffer), self.config.batch_size)
            batch = [self._buffer.popleft() for _ in range(batch_size)]

        # Process batch
        if self._on_flush:
            try:
                await asyncio.to_thread(self._on_flush, batch)
                self._events_processed += len(batch)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Put events back
                async with self._buffer_lock:
                    for event in reversed(batch):
                        self._buffer.appendleft(event)

    async def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            await asyncio.sleep(self.config.flush_interval_ms / 1000)
            await self._flush()

    def get_metrics(self) -> Dict[str, Any]:
        """Get collector metrics."""
        return {
            "events_received": self._events_received,
            "events_processed": self._events_processed,
            "events_dropped": self._events_dropped,
            "events_invalid": self._events_invalid,
            "buffer_size": len(self._buffer),
            "buffer_capacity": self.config.buffer_size,
        }

    def reset_metrics(self) -> None:
        """Reset collector metrics."""
        self._events_received = 0
        self._events_processed = 0
        self._events_dropped = 0
        self._events_invalid = 0
