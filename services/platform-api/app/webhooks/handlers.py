"""
Webhook Handlers

Event handlers and processors:
- Event processing
- Payload transformation
- Filtering
- Batching
"""

from typing import Optional, Dict, Any, List, Callable, Set, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import json
import re
import logging

from .engine import (
    WebhookEngine, WebhookEvent, WebhookEndpoint, WebhookDelivery,
    WebhookEventType, WebhookStatus, get_webhook_engine,
)

logger = logging.getLogger(__name__)


class EventHandler(ABC):
    """Abstract event handler."""

    @property
    @abstractmethod
    def event_types(self) -> Set[str]:
        """Event types this handler processes."""
        pass

    @abstractmethod
    async def handle(self, event: WebhookEvent) -> Optional[WebhookEvent]:
        """Handle event. Return modified event or None to drop."""
        pass


class PayloadTransformer:
    """
    Transform webhook payloads.

    Features:
    - Field mapping
    - Value transformation
    - Schema conversion
    """

    def __init__(self):
        self._transformers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    def register(
        self,
        name: str,
        transformer: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """Register transformer."""
        self._transformers[name] = transformer

    def transform(
        self,
        payload: Dict[str, Any],
        transformer_name: str,
    ) -> Dict[str, Any]:
        """Apply transformer to payload."""
        transformer = self._transformers.get(transformer_name)
        if transformer:
            return transformer(payload)
        return payload

    def map_fields(
        self,
        payload: Dict[str, Any],
        mapping: Dict[str, str],
    ) -> Dict[str, Any]:
        """Map field names."""
        result = {}
        for target_field, source_path in mapping.items():
            value = self._get_nested(payload, source_path)
            if value is not None:
                self._set_nested(result, target_field, value)
        return result

    def _get_nested(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value by dot path."""
        parts = path.split('.')
        value = data
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list):
                try:
                    index = int(part)
                    value = value[index] if 0 <= index < len(value) else None
                except (ValueError, IndexError):
                    value = None
            else:
                return None
        return value

    def _set_nested(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested value by dot path."""
        parts = path.split('.')
        current = data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value


class PayloadFilter:
    """
    Filter webhook payloads.

    Features:
    - Field filtering
    - Sensitive data redaction
    - Schema enforcement
    """

    def __init__(self):
        self._sensitive_fields: Set[str] = {
            "password", "secret", "token", "api_key", "apikey",
            "authorization", "auth", "credential", "ssn", "card_number",
        }

    def add_sensitive_field(self, field: str) -> None:
        """Add sensitive field to redact."""
        self._sensitive_fields.add(field.lower())

    def filter(
        self,
        payload: Dict[str, Any],
        allowed_fields: Optional[Set[str]] = None,
        excluded_fields: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Filter payload fields."""
        result = {}

        for key, value in payload.items():
            # Check allowed/excluded
            if allowed_fields and key not in allowed_fields:
                continue
            if excluded_fields and key in excluded_fields:
                continue

            # Recurse into nested dicts
            if isinstance(value, dict):
                result[key] = self.filter(value, allowed_fields, excluded_fields)
            else:
                result[key] = value

        return result

    def redact_sensitive(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields."""
        result = {}

        for key, value in payload.items():
            if key.lower() in self._sensitive_fields:
                result[key] = "[REDACTED]"
            elif isinstance(value, dict):
                result[key] = self.redact_sensitive(value)
            elif isinstance(value, list):
                result[key] = [
                    self.redact_sensitive(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value

        return result


class EventBatcher:
    """
    Batch multiple events into single deliveries.

    Features:
    - Time-based batching
    - Size-based batching
    - Event grouping
    """

    def __init__(
        self,
        max_batch_size: int = 100,
        max_wait_seconds: float = 5.0,
    ):
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_seconds
        self._batches: Dict[str, List[WebhookEvent]] = {}
        self._batch_timers: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def add(
        self,
        event: WebhookEvent,
        batch_key: str,
    ) -> Optional[List[WebhookEvent]]:
        """Add event to batch. Returns batch if ready."""
        async with self._lock:
            if batch_key not in self._batches:
                self._batches[batch_key] = []
                self._batch_timers[batch_key] = datetime.utcnow()

            self._batches[batch_key].append(event)

            # Check if batch is ready
            if len(self._batches[batch_key]) >= self._max_batch_size:
                return self._flush_batch(batch_key)

            # Check time
            elapsed = (datetime.utcnow() - self._batch_timers[batch_key]).total_seconds()
            if elapsed >= self._max_wait_seconds:
                return self._flush_batch(batch_key)

        return None

    def _flush_batch(self, batch_key: str) -> List[WebhookEvent]:
        """Flush and return batch."""
        batch = self._batches.pop(batch_key, [])
        self._batch_timers.pop(batch_key, None)
        return batch

    async def flush_all(self) -> Dict[str, List[WebhookEvent]]:
        """Flush all pending batches."""
        async with self._lock:
            batches = dict(self._batches)
            self._batches.clear()
            self._batch_timers.clear()
            return batches


class EventRouter:
    """
    Route events to different handlers/endpoints.

    Features:
    - Rule-based routing
    - Dynamic endpoint selection
    - Event transformation
    """

    def __init__(self):
        self._rules: List[Dict[str, Any]] = []

    def add_rule(
        self,
        name: str,
        condition: Callable[[WebhookEvent], bool],
        endpoint_ids: Optional[List[str]] = None,
        transformer: Optional[Callable[[WebhookEvent], WebhookEvent]] = None,
        priority: int = 0,
    ) -> None:
        """Add routing rule."""
        self._rules.append({
            "name": name,
            "condition": condition,
            "endpoint_ids": endpoint_ids,
            "transformer": transformer,
            "priority": priority,
        })
        # Sort by priority (higher first)
        self._rules.sort(key=lambda r: r["priority"], reverse=True)

    def route(self, event: WebhookEvent) -> List[Dict[str, Any]]:
        """Route event and return destinations."""
        destinations = []

        for rule in self._rules:
            if rule["condition"](event):
                transformed_event = event
                if rule["transformer"]:
                    transformed_event = rule["transformer"](event)

                destinations.append({
                    "rule_name": rule["name"],
                    "endpoint_ids": rule["endpoint_ids"],
                    "event": transformed_event,
                })

        return destinations


class WebhookMiddleware(ABC):
    """Abstract webhook middleware."""

    @abstractmethod
    async def before_send(
        self,
        endpoint: WebhookEndpoint,
        delivery: WebhookDelivery,
    ) -> Optional[WebhookDelivery]:
        """Called before sending. Return None to cancel."""
        pass

    @abstractmethod
    async def after_send(
        self,
        endpoint: WebhookEndpoint,
        delivery: WebhookDelivery,
    ) -> None:
        """Called after sending."""
        pass


class LoggingMiddleware(WebhookMiddleware):
    """Middleware for logging webhook deliveries."""

    def __init__(self, log_payloads: bool = False):
        self._log_payloads = log_payloads

    async def before_send(
        self,
        endpoint: WebhookEndpoint,
        delivery: WebhookDelivery,
    ) -> Optional[WebhookDelivery]:
        """Log before sending."""
        logger.info(
            f"Sending webhook: {delivery.event_type} to {endpoint.url} "
            f"(delivery_id={delivery.delivery_id})"
        )
        if self._log_payloads:
            logger.debug(f"Payload: {json.dumps(delivery.payload)}")
        return delivery

    async def after_send(
        self,
        endpoint: WebhookEndpoint,
        delivery: WebhookDelivery,
    ) -> None:
        """Log after sending."""
        if delivery.status == WebhookStatus.SUCCESS:
            logger.info(
                f"Webhook delivered: {delivery.delivery_id} "
                f"(status={delivery.response_status}, "
                f"duration={delivery.duration_ms:.0f}ms)"
            )
        else:
            logger.warning(
                f"Webhook failed: {delivery.delivery_id} "
                f"(status={delivery.response_status}, "
                f"error={delivery.error_message})"
            )


class MetricsMiddleware(WebhookMiddleware):
    """Middleware for collecting webhook metrics."""

    def __init__(self):
        self._metrics: Dict[str, Any] = {
            "total_sent": 0,
            "total_success": 0,
            "total_failed": 0,
            "total_duration_ms": 0,
            "by_endpoint": {},
            "by_event_type": {},
        }

    async def before_send(
        self,
        endpoint: WebhookEndpoint,
        delivery: WebhookDelivery,
    ) -> Optional[WebhookDelivery]:
        """Track before sending."""
        return delivery

    async def after_send(
        self,
        endpoint: WebhookEndpoint,
        delivery: WebhookDelivery,
    ) -> None:
        """Collect metrics after sending."""
        self._metrics["total_sent"] += 1

        if delivery.status == WebhookStatus.SUCCESS:
            self._metrics["total_success"] += 1
        else:
            self._metrics["total_failed"] += 1

        if delivery.duration_ms:
            self._metrics["total_duration_ms"] += delivery.duration_ms

        # By endpoint
        eid = endpoint.endpoint_id
        if eid not in self._metrics["by_endpoint"]:
            self._metrics["by_endpoint"][eid] = {"sent": 0, "success": 0, "failed": 0}
        self._metrics["by_endpoint"][eid]["sent"] += 1
        if delivery.status == WebhookStatus.SUCCESS:
            self._metrics["by_endpoint"][eid]["success"] += 1
        else:
            self._metrics["by_endpoint"][eid]["failed"] += 1

        # By event type
        etype = delivery.event_type
        if etype not in self._metrics["by_event_type"]:
            self._metrics["by_event_type"][etype] = {"sent": 0, "success": 0, "failed": 0}
        self._metrics["by_event_type"][etype]["sent"] += 1
        if delivery.status == WebhookStatus.SUCCESS:
            self._metrics["by_event_type"][etype]["success"] += 1
        else:
            self._metrics["by_event_type"][etype]["failed"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        avg_duration = 0
        if self._metrics["total_sent"] > 0:
            avg_duration = self._metrics["total_duration_ms"] / self._metrics["total_sent"]

        return {
            **self._metrics,
            "average_duration_ms": avg_duration,
            "success_rate": (
                self._metrics["total_success"] / self._metrics["total_sent"]
                if self._metrics["total_sent"] > 0 else 0
            ),
        }

    def reset(self) -> None:
        """Reset metrics."""
        self._metrics = {
            "total_sent": 0,
            "total_success": 0,
            "total_failed": 0,
            "total_duration_ms": 0,
            "by_endpoint": {},
            "by_event_type": {},
        }


class EventEnricher:
    """
    Enrich events with additional data.

    Features:
    - Context injection
    - Data lookup
    - Computed fields
    """

    def __init__(self):
        self._enrichers: List[Callable[[WebhookEvent], WebhookEvent]] = []

    def add_enricher(self, enricher: Callable[[WebhookEvent], WebhookEvent]) -> None:
        """Add enricher function."""
        self._enrichers.append(enricher)

    def enrich(self, event: WebhookEvent) -> WebhookEvent:
        """Apply all enrichers to event."""
        for enricher in self._enrichers:
            try:
                event = enricher(event)
            except Exception as e:
                logger.error(f"Enricher error: {e}")
        return event


class CallEventHandler(EventHandler):
    """Handler for call-related events."""

    @property
    def event_types(self) -> Set[str]:
        return {
            WebhookEventType.CALL_INITIATED.value,
            WebhookEventType.CALL_RINGING.value,
            WebhookEventType.CALL_ANSWERED.value,
            WebhookEventType.CALL_ENDED.value,
            WebhookEventType.CALL_FAILED.value,
            WebhookEventType.CALL_TRANSFERRED.value,
        }

    async def handle(self, event: WebhookEvent) -> Optional[WebhookEvent]:
        """Process call event."""
        # Add call duration for ended calls
        if event.event_type == WebhookEventType.CALL_ENDED.value:
            start_time = event.data.get("start_time")
            end_time = event.data.get("end_time")
            if start_time and end_time:
                # Calculate duration
                duration = (
                    datetime.fromisoformat(end_time) -
                    datetime.fromisoformat(start_time)
                ).total_seconds()
                event.data["duration_seconds"] = duration

        return event


class TranscriptEventHandler(EventHandler):
    """Handler for transcript events."""

    @property
    def event_types(self) -> Set[str]:
        return {
            WebhookEventType.TRANSCRIPT_PARTIAL.value,
            WebhookEventType.TRANSCRIPT_FINAL.value,
        }

    async def handle(self, event: WebhookEvent) -> Optional[WebhookEvent]:
        """Process transcript event."""
        # Add word count
        transcript = event.data.get("transcript", "")
        event.data["word_count"] = len(transcript.split())
        event.data["char_count"] = len(transcript)

        return event


class EventProcessor:
    """
    Central event processor.

    Coordinates handlers, transformers, and enrichers.
    """

    def __init__(self, engine: Optional[WebhookEngine] = None):
        self._engine = engine or get_webhook_engine()
        self._handlers: Dict[str, EventHandler] = {}
        self._enricher = EventEnricher()
        self._transformer = PayloadTransformer()
        self._filter = PayloadFilter()
        self._middlewares: List[WebhookMiddleware] = []

    def register_handler(self, handler: EventHandler) -> None:
        """Register event handler."""
        for event_type in handler.event_types:
            self._handlers[event_type] = handler

    def add_middleware(self, middleware: WebhookMiddleware) -> None:
        """Add middleware."""
        self._middlewares.append(middleware)

    def add_enricher(self, enricher: Callable[[WebhookEvent], WebhookEvent]) -> None:
        """Add event enricher."""
        self._enricher.add_enricher(enricher)

    async def process(self, event: WebhookEvent) -> List[WebhookDelivery]:
        """Process event through the pipeline."""
        # Apply handler
        handler = self._handlers.get(event.event_type)
        if handler:
            event = await handler.handle(event)
            if event is None:
                return []  # Event dropped

        # Enrich event
        event = self._enricher.enrich(event)

        # Redact sensitive data
        event.data = self._filter.redact_sensitive(event.data)

        # Dispatch through engine
        return await self._engine.dispatch(event)


# Factory functions
def create_event_processor() -> EventProcessor:
    """Create configured event processor."""
    processor = EventProcessor()

    # Register default handlers
    processor.register_handler(CallEventHandler())
    processor.register_handler(TranscriptEventHandler())

    # Add logging middleware
    processor.add_middleware(LoggingMiddleware())

    return processor


def create_metrics_middleware() -> MetricsMiddleware:
    """Create metrics middleware."""
    return MetricsMiddleware()
