"""Webhook delivery system with retry and reliability."""

from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import httpx
import hmac
import hashlib
import json
import time
import logging
import uuid

logger = logging.getLogger(__name__)


class DeliveryStatus(str, Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    EXHAUSTED = "exhausted"  # All retries exhausted


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint."""
    id: str
    url: str
    secret: str
    events: List[str]
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    active: bool = True


@dataclass
class DeliveryAttempt:
    """Record of a single delivery attempt."""
    id: str
    webhook_id: str
    attempt_number: int
    status_code: Optional[int]
    response_body: Optional[str]
    error_message: Optional[str]
    duration_ms: int
    timestamp: datetime
    success: bool


@dataclass
class WebhookDelivery:
    """A webhook delivery task."""
    id: str
    webhook_id: str
    event_type: str
    payload: Dict[str, Any]
    status: DeliveryStatus
    attempts: List[DeliveryAttempt] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    next_retry_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class WebhookSigner:
    """
    Sign webhooks for verification.

    Uses HMAC-SHA256 with timestamp to prevent replay attacks.
    """

    @staticmethod
    def sign(
        payload: bytes,
        secret: str,
        timestamp: Optional[int] = None,
    ) -> tuple[str, int]:
        """
        Sign a webhook payload.

        Returns (signature, timestamp).
        """
        ts = timestamp or int(time.time())
        message = f"{ts}.{payload.decode('utf-8')}"
        signature = hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256,
        ).hexdigest()
        return f"v1={signature}", ts

    @staticmethod
    def verify(
        payload: bytes,
        signature: str,
        timestamp: int,
        secret: str,
        tolerance_seconds: int = 300,
    ) -> bool:
        """Verify a webhook signature."""
        # Check timestamp
        now = int(time.time())
        if abs(now - timestamp) > tolerance_seconds:
            return False

        # Compute expected signature
        expected, _ = WebhookSigner.sign(payload, secret, timestamp)

        # Constant-time comparison
        return hmac.compare_digest(expected, signature)


class WebhookDeliveryService:
    """
    Service for delivering webhooks with retry support.

    Features:
    - Async delivery with configurable concurrency
    - Exponential backoff retry
    - Signature verification
    - Delivery tracking and history
    - Dead letter queue for failed deliveries
    """

    def __init__(
        self,
        max_concurrent_deliveries: int = 100,
        batch_size: int = 50,
        default_timeout: int = 30,
    ):
        self.max_concurrent = max_concurrent_deliveries
        self.batch_size = batch_size
        self.default_timeout = default_timeout

        self._queue: asyncio.Queue = asyncio.Queue()
        self._in_flight: Dict[str, WebhookDelivery] = {}
        self._dead_letter: List[WebhookDelivery] = []
        self._delivery_history: Dict[str, WebhookDelivery] = {}
        self._webhooks: Dict[str, WebhookConfig] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_deliveries)
        self._running = False
        self._workers: List[asyncio.Task] = []

        # Callbacks
        self._on_success: Optional[Callable[[WebhookDelivery], Awaitable[None]]] = None
        self._on_failure: Optional[Callable[[WebhookDelivery], Awaitable[None]]] = None
        self._on_exhausted: Optional[Callable[[WebhookDelivery], Awaitable[None]]] = None

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        """Start the delivery service."""
        if self._running:
            return

        self._running = True
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.default_timeout),
            follow_redirects=True,
        )

        # Start worker tasks
        for i in range(min(10, self.max_concurrent)):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        logger.info(f"Webhook delivery service started with {len(self._workers)} workers")

    async def stop(self) -> None:
        """Stop the delivery service."""
        self._running = False

        # Cancel workers
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        if self._client:
            await self._client.aclose()
            self._client = None

        logger.info("Webhook delivery service stopped")

    def register_webhook(self, config: WebhookConfig) -> None:
        """Register a webhook endpoint."""
        self._webhooks[config.id] = config
        logger.info(f"Registered webhook {config.id}: {config.url}")

    def unregister_webhook(self, webhook_id: str) -> None:
        """Unregister a webhook endpoint."""
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            logger.info(f"Unregistered webhook {webhook_id}")

    async def enqueue(
        self,
        webhook_id: str,
        event_type: str,
        payload: Dict[str, Any],
    ) -> str:
        """
        Enqueue a webhook for delivery.

        Returns delivery ID.
        """
        if webhook_id not in self._webhooks:
            raise ValueError(f"Unknown webhook: {webhook_id}")

        webhook = self._webhooks[webhook_id]
        if not webhook.active:
            raise ValueError(f"Webhook {webhook_id} is not active")

        if event_type not in webhook.events and "*" not in webhook.events:
            raise ValueError(f"Webhook {webhook_id} not subscribed to {event_type}")

        delivery = WebhookDelivery(
            id=str(uuid.uuid4()),
            webhook_id=webhook_id,
            event_type=event_type,
            payload=payload,
            status=DeliveryStatus.PENDING,
        )

        await self._queue.put(delivery)
        self._delivery_history[delivery.id] = delivery

        logger.debug(f"Enqueued webhook delivery {delivery.id}")
        return delivery.id

    async def enqueue_to_all(
        self,
        event_type: str,
        payload: Dict[str, Any],
    ) -> List[str]:
        """
        Enqueue event to all subscribed webhooks.

        Returns list of delivery IDs.
        """
        delivery_ids = []

        for webhook_id, webhook in self._webhooks.items():
            if not webhook.active:
                continue
            if event_type not in webhook.events and "*" not in webhook.events:
                continue

            try:
                delivery_id = await self.enqueue(webhook_id, event_type, payload)
                delivery_ids.append(delivery_id)
            except Exception as e:
                logger.error(f"Failed to enqueue to {webhook_id}: {e}")

        return delivery_ids

    async def get_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get a delivery by ID."""
        return self._delivery_history.get(delivery_id)

    async def retry_delivery(self, delivery_id: str) -> bool:
        """Manually retry a failed delivery."""
        delivery = self._delivery_history.get(delivery_id)
        if not delivery:
            return False

        if delivery.status not in (DeliveryStatus.FAILED, DeliveryStatus.EXHAUSTED):
            return False

        delivery.status = DeliveryStatus.PENDING
        delivery.next_retry_at = None
        await self._queue.put(delivery)

        return True

    def on_success(
        self,
        callback: Callable[[WebhookDelivery], Awaitable[None]],
    ) -> None:
        """Register callback for successful deliveries."""
        self._on_success = callback

    def on_failure(
        self,
        callback: Callable[[WebhookDelivery], Awaitable[None]],
    ) -> None:
        """Register callback for failed deliveries."""
        self._on_failure = callback

    def on_exhausted(
        self,
        callback: Callable[[WebhookDelivery], Awaitable[None]],
    ) -> None:
        """Register callback for deliveries with exhausted retries."""
        self._on_exhausted = callback

    async def _worker(self, worker_id: int) -> None:
        """Worker task that processes deliveries."""
        logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                # Get next delivery
                delivery = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )

                # Check if it's ready for retry
                if delivery.next_retry_at and delivery.next_retry_at > datetime.utcnow():
                    # Put back in queue and wait
                    await self._queue.put(delivery)
                    await asyncio.sleep(0.1)
                    continue

                # Process delivery
                async with self._semaphore:
                    await self._deliver(delivery)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.debug(f"Worker {worker_id} stopped")

    async def _deliver(self, delivery: WebhookDelivery) -> None:
        """Attempt to deliver a webhook."""
        webhook = self._webhooks.get(delivery.webhook_id)
        if not webhook:
            delivery.status = DeliveryStatus.FAILED
            return

        delivery.status = DeliveryStatus.IN_PROGRESS
        self._in_flight[delivery.id] = delivery

        try:
            attempt_num = len(delivery.attempts) + 1

            # Build payload with metadata
            full_payload = {
                "id": delivery.id,
                "event_type": delivery.event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": delivery.payload,
                "metadata": {
                    "webhook_id": webhook.id,
                    "attempt": attempt_num,
                },
            }

            payload_bytes = json.dumps(full_payload).encode('utf-8')

            # Sign payload
            signature, timestamp = WebhookSigner.sign(payload_bytes, webhook.secret)

            # Build headers
            headers = {
                "Content-Type": "application/json",
                "X-BVRAI-Signature": signature,
                "X-BVRAI-Timestamp": str(timestamp),
                "X-BVRAI-Delivery-ID": delivery.id,
                "X-BVRAI-Event-Type": delivery.event_type,
                "User-Agent": "BVRAI-Webhook/1.0",
                **webhook.headers,
            }

            # Make request
            start_time = time.monotonic()

            try:
                response = await self._client.post(
                    webhook.url,
                    content=payload_bytes,
                    headers=headers,
                    timeout=webhook.timeout_seconds,
                )

                duration_ms = int((time.monotonic() - start_time) * 1000)

                # Record attempt
                attempt = DeliveryAttempt(
                    id=str(uuid.uuid4()),
                    webhook_id=webhook.id,
                    attempt_number=attempt_num,
                    status_code=response.status_code,
                    response_body=response.text[:1000] if response.text else None,
                    error_message=None,
                    duration_ms=duration_ms,
                    timestamp=datetime.utcnow(),
                    success=200 <= response.status_code < 300,
                )
                delivery.attempts.append(attempt)

                if attempt.success:
                    delivery.status = DeliveryStatus.SUCCESS
                    delivery.completed_at = datetime.utcnow()
                    logger.info(
                        f"Webhook delivery {delivery.id} succeeded "
                        f"(attempt {attempt_num}, {duration_ms}ms)"
                    )
                    if self._on_success:
                        await self._on_success(delivery)
                else:
                    await self._handle_failure(delivery, webhook, attempt)

            except httpx.TimeoutException:
                duration_ms = int((time.monotonic() - start_time) * 1000)
                attempt = DeliveryAttempt(
                    id=str(uuid.uuid4()),
                    webhook_id=webhook.id,
                    attempt_number=attempt_num,
                    status_code=None,
                    response_body=None,
                    error_message="Request timed out",
                    duration_ms=duration_ms,
                    timestamp=datetime.utcnow(),
                    success=False,
                )
                delivery.attempts.append(attempt)
                await self._handle_failure(delivery, webhook, attempt)

            except httpx.RequestError as e:
                duration_ms = int((time.monotonic() - start_time) * 1000)
                attempt = DeliveryAttempt(
                    id=str(uuid.uuid4()),
                    webhook_id=webhook.id,
                    attempt_number=attempt_num,
                    status_code=None,
                    response_body=None,
                    error_message=str(e),
                    duration_ms=duration_ms,
                    timestamp=datetime.utcnow(),
                    success=False,
                )
                delivery.attempts.append(attempt)
                await self._handle_failure(delivery, webhook, attempt)

        finally:
            if delivery.id in self._in_flight:
                del self._in_flight[delivery.id]

    async def _handle_failure(
        self,
        delivery: WebhookDelivery,
        webhook: WebhookConfig,
        attempt: DeliveryAttempt,
    ) -> None:
        """Handle a failed delivery attempt."""
        attempt_num = len(delivery.attempts)

        if attempt_num >= webhook.max_retries:
            # Exhausted all retries
            delivery.status = DeliveryStatus.EXHAUSTED
            delivery.completed_at = datetime.utcnow()
            self._dead_letter.append(delivery)

            logger.warning(
                f"Webhook delivery {delivery.id} exhausted after {attempt_num} attempts"
            )

            if self._on_exhausted:
                await self._on_exhausted(delivery)
        else:
            # Schedule retry
            delay = webhook.retry_backoff_base ** attempt_num
            delivery.status = DeliveryStatus.RETRYING
            delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)

            logger.info(
                f"Webhook delivery {delivery.id} failed (attempt {attempt_num}), "
                f"retrying in {delay:.1f}s"
            )

            # Put back in queue
            await self._queue.put(delivery)

            if self._on_failure:
                await self._on_failure(delivery)

    # Statistics

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def in_flight_count(self) -> int:
        """Get count of in-flight deliveries."""
        return len(self._in_flight)

    @property
    def dead_letter_count(self) -> int:
        """Get count of dead letter deliveries."""
        return len(self._dead_letter)

    def get_stats(self) -> Dict[str, Any]:
        """Get delivery statistics."""
        total = len(self._delivery_history)
        success = sum(
            1 for d in self._delivery_history.values()
            if d.status == DeliveryStatus.SUCCESS
        )
        failed = sum(
            1 for d in self._delivery_history.values()
            if d.status in (DeliveryStatus.FAILED, DeliveryStatus.EXHAUSTED)
        )
        pending = sum(
            1 for d in self._delivery_history.values()
            if d.status in (DeliveryStatus.PENDING, DeliveryStatus.RETRYING)
        )

        return {
            "total_deliveries": total,
            "successful": success,
            "failed": failed,
            "pending": pending,
            "queue_size": self.queue_size,
            "in_flight": self.in_flight_count,
            "dead_letter": self.dead_letter_count,
            "success_rate": success / total if total > 0 else 0,
        }


# Global delivery service
_delivery_service: Optional[WebhookDeliveryService] = None


def get_delivery_service() -> WebhookDeliveryService:
    """Get the global delivery service."""
    global _delivery_service
    if _delivery_service is None:
        _delivery_service = WebhookDeliveryService()
    return _delivery_service


async def send_webhook(
    webhook_id: str,
    event_type: str,
    payload: Dict[str, Any],
) -> str:
    """Send a webhook using the global service."""
    service = get_delivery_service()
    return await service.enqueue(webhook_id, event_type, payload)


async def broadcast_event(
    event_type: str,
    payload: Dict[str, Any],
) -> List[str]:
    """Broadcast an event to all subscribed webhooks."""
    service = get_delivery_service()
    return await service.enqueue_to_all(event_type, payload)
