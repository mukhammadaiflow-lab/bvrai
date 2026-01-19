"""Background worker for processing webhook events."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import redis.asyncio as redis

from .delivery import WebhookDeliveryService
from .models import WebhookEventType

logger = logging.getLogger(__name__)


class WebhookWorker:
    """
    Background worker that processes webhook events from a queue.

    Features:
    - Consumes events from Redis queue
    - Batches events for efficient processing
    - Handles backpressure gracefully
    - Supports graceful shutdown
    """

    QUEUE_KEY = "webhook_events"
    PROCESSING_KEY = "webhook_events_processing"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        batch_size: int = 100,
        poll_interval: float = 0.1,
        max_workers: int = 10,
    ):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.max_workers = max_workers

        self._delivery_service: Optional[WebhookDeliveryService] = None
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def start(self, delivery_service: WebhookDeliveryService) -> None:
        """Start the worker."""
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        self._delivery_service = delivery_service
        self._semaphore = asyncio.Semaphore(self.max_workers)
        self._running = True

        # Start worker tasks
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker_loop(i))
            self._tasks.append(task)

        logger.info(f"Webhook worker started with {self.max_workers} workers")

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        self._running = False

        # Wait for workers to finish
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        if self.redis:
            await self.redis.close()

        logger.info("Webhook worker stopped")

    async def enqueue_event(
        self,
        organization_id: str,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Enqueue an event for processing.

        Args:
            organization_id: The organization ID
            event_type: The event type
            data: The event data
            metadata: Optional metadata

        Returns:
            Event ID
        """
        if not self.redis:
            raise RuntimeError("Worker not started")

        event_id = str(uuid4())

        event = {
            "id": event_id,
            "organization_id": organization_id,
            "event_type": event_type,
            "data": data,
            "metadata": metadata or {},
            "enqueued_at": datetime.utcnow().isoformat(),
        }

        await self.redis.lpush(self.QUEUE_KEY, json.dumps(event))

        logger.debug(f"Enqueued webhook event: {event_type} ({event_id})")
        return event_id

    async def get_queue_length(self) -> int:
        """Get the current queue length."""
        if not self.redis:
            return 0
        return await self.redis.llen(self.QUEUE_KEY)

    async def get_processing_count(self) -> int:
        """Get number of events currently being processed."""
        if not self.redis:
            return 0
        return await self.redis.scard(self.PROCESSING_KEY)

    async def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop."""
        logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                async with self._semaphore:
                    # Get event from queue
                    event = await self._get_event()

                    if event:
                        await self._process_event(event)
                    else:
                        # No events, wait before polling again
                        await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error

        logger.debug(f"Worker {worker_id} stopped")

    async def _get_event(self) -> Optional[Dict[str, Any]]:
        """Get next event from queue."""
        if not self.redis:
            return None

        # Use BRPOPLPUSH for reliable processing
        # Move event to processing set atomically
        result = await self.redis.brpop(self.QUEUE_KEY, timeout=1)

        if result:
            _, event_data = result
            try:
                event = json.loads(event_data)

                # Track in processing set
                await self.redis.sadd(self.PROCESSING_KEY, event["id"])

                return event
            except json.JSONDecodeError:
                logger.warning(f"Invalid event data: {event_data}")

        return None

    async def _process_event(self, event: Dict[str, Any]) -> None:
        """Process a single event."""
        event_id = event.get("id", "unknown")

        try:
            organization_id = event["organization_id"]
            event_type = event["event_type"]
            data = event["data"]
            metadata = event.get("metadata", {})

            # Dispatch to delivery service
            if self._delivery_service:
                await self._delivery_service.dispatch_event(
                    organization_id=organization_id,
                    event_type=event_type,
                    data=data,
                    metadata=metadata,
                )

            logger.debug(f"Processed event: {event_type} ({event_id})")

        except Exception as e:
            logger.error(f"Failed to process event {event_id}: {e}")

            # Could implement dead letter queue here
            await self._handle_failed_event(event, str(e))

        finally:
            # Remove from processing set
            if self.redis:
                await self.redis.srem(self.PROCESSING_KEY, event_id)

    async def _handle_failed_event(self, event: Dict[str, Any], error: str) -> None:
        """Handle a failed event (dead letter queue)."""
        if not self.redis:
            return

        failed_event = {
            **event,
            "error": error,
            "failed_at": datetime.utcnow().isoformat(),
        }

        # Store in dead letter queue
        await self.redis.lpush("webhook_events_failed", json.dumps(failed_event))

        # Trim to last 10000 failed events
        await self.redis.ltrim("webhook_events_failed", 0, 9999)


class WebhookEventEmitter:
    """
    Helper class for emitting webhook events from application code.

    Usage:
        emitter = WebhookEventEmitter(worker)

        # Emit a call started event
        await emitter.emit_call_started(
            organization_id="org_123",
            call_id="call_456",
            agent_id="agent_789",
            to_number="+14155551234",
        )
    """

    def __init__(self, worker: WebhookWorker):
        self.worker = worker

    async def emit(
        self,
        organization_id: str,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Emit a webhook event."""
        return await self.worker.enqueue_event(
            organization_id=organization_id,
            event_type=event_type,
            data=data,
            metadata=metadata,
        )

    # Call events

    async def emit_call_started(
        self,
        organization_id: str,
        call_id: str,
        agent_id: str,
        to_number: str,
        from_number: str,
        direction: str = "outbound",
        **kwargs,
    ) -> str:
        """Emit call started event."""
        return await self.emit(
            organization_id=organization_id,
            event_type=WebhookEventType.CALL_STARTED.value,
            data={
                "call_id": call_id,
                "agent_id": agent_id,
                "to_number": to_number,
                "from_number": from_number,
                "direction": direction,
                **kwargs,
            },
        )

    async def emit_call_ended(
        self,
        organization_id: str,
        call_id: str,
        duration: int,
        end_reason: str,
        status: str = "completed",
        **kwargs,
    ) -> str:
        """Emit call ended event."""
        return await self.emit(
            organization_id=organization_id,
            event_type=WebhookEventType.CALL_ENDED.value,
            data={
                "call_id": call_id,
                "duration": duration,
                "end_reason": end_reason,
                "status": status,
                **kwargs,
            },
        )

    async def emit_call_failed(
        self,
        organization_id: str,
        call_id: str,
        error_code: str,
        error_message: str,
        **kwargs,
    ) -> str:
        """Emit call failed event."""
        return await self.emit(
            organization_id=organization_id,
            event_type=WebhookEventType.CALL_FAILED.value,
            data={
                "call_id": call_id,
                "error_code": error_code,
                "error_message": error_message,
                **kwargs,
            },
        )

    # Transcription events

    async def emit_transcription(
        self,
        organization_id: str,
        call_id: str,
        text: str,
        role: str,
        is_final: bool = True,
        confidence: float = 1.0,
        **kwargs,
    ) -> str:
        """Emit transcription event."""
        event_type = (
            WebhookEventType.TRANSCRIPTION_FINAL.value if is_final
            else WebhookEventType.TRANSCRIPTION_PARTIAL.value
        )
        return await self.emit(
            organization_id=organization_id,
            event_type=event_type,
            data={
                "call_id": call_id,
                "text": text,
                "role": role,
                "is_final": is_final,
                "confidence": confidence,
                **kwargs,
            },
        )

    # Agent events

    async def emit_agent_speech_start(
        self,
        organization_id: str,
        call_id: str,
        agent_id: str,
        **kwargs,
    ) -> str:
        """Emit agent speech start event."""
        return await self.emit(
            organization_id=organization_id,
            event_type=WebhookEventType.AGENT_SPEECH_START.value,
            data={
                "call_id": call_id,
                "agent_id": agent_id,
                **kwargs,
            },
        )

    async def emit_agent_speech_end(
        self,
        organization_id: str,
        call_id: str,
        agent_id: str,
        text: str,
        duration_ms: int,
        **kwargs,
    ) -> str:
        """Emit agent speech end event."""
        return await self.emit(
            organization_id=organization_id,
            event_type=WebhookEventType.AGENT_SPEECH_END.value,
            data={
                "call_id": call_id,
                "agent_id": agent_id,
                "text": text,
                "duration_ms": duration_ms,
                **kwargs,
            },
        )

    # Campaign events

    async def emit_campaign_started(
        self,
        organization_id: str,
        campaign_id: str,
        total_contacts: int,
        **kwargs,
    ) -> str:
        """Emit campaign started event."""
        return await self.emit(
            organization_id=organization_id,
            event_type=WebhookEventType.CAMPAIGN_STARTED.value,
            data={
                "campaign_id": campaign_id,
                "total_contacts": total_contacts,
                **kwargs,
            },
        )

    async def emit_campaign_completed(
        self,
        organization_id: str,
        campaign_id: str,
        total_contacts: int,
        successful: int,
        failed: int,
        **kwargs,
    ) -> str:
        """Emit campaign completed event."""
        return await self.emit(
            organization_id=organization_id,
            event_type=WebhookEventType.CAMPAIGN_COMPLETED.value,
            data={
                "campaign_id": campaign_id,
                "total_contacts": total_contacts,
                "successful": successful,
                "failed": failed,
                **kwargs,
            },
        )

    # Conversation events

    async def emit_conversation_created(
        self,
        organization_id: str,
        conversation_id: str,
        call_id: str,
        agent_id: str,
        **kwargs,
    ) -> str:
        """Emit conversation created event."""
        return await self.emit(
            organization_id=organization_id,
            event_type=WebhookEventType.CONVERSATION_CREATED.value,
            data={
                "conversation_id": conversation_id,
                "call_id": call_id,
                "agent_id": agent_id,
                **kwargs,
            },
        )

    async def emit_conversation_ended(
        self,
        organization_id: str,
        conversation_id: str,
        call_id: str,
        message_count: int,
        summary: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Emit conversation ended event."""
        return await self.emit(
            organization_id=organization_id,
            event_type=WebhookEventType.CONVERSATION_ENDED.value,
            data={
                "conversation_id": conversation_id,
                "call_id": call_id,
                "message_count": message_count,
                "summary": summary,
                **kwargs,
            },
        )
