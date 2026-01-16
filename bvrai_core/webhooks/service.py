"""
Webhook Management Service

Comprehensive webhook system with reliable delivery, retries, and monitoring.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp

logger = logging.getLogger(__name__)


# =============================================================================
# Enums & Constants
# =============================================================================


class WebhookEvent(str, Enum):
    """Supported webhook events."""
    # Call events
    CALL_STARTED = "call.started"
    CALL_ENDED = "call.ended"
    CALL_FAILED = "call.failed"
    CALL_RECORDING_READY = "call.recording_ready"

    # Conversation events
    CONVERSATION_MESSAGE = "conversation.message"
    CONVERSATION_ENDED = "conversation.ended"
    CONVERSATION_TRANSCRIPT_READY = "conversation.transcript_ready"

    # Agent events
    AGENT_UPDATED = "agent.updated"
    AGENT_DEPLOYED = "agent.deployed"

    # Transfer events
    TRANSFER_INITIATED = "transfer.initiated"
    TRANSFER_COMPLETED = "transfer.completed"
    TRANSFER_FAILED = "transfer.failed"

    # Billing events
    USAGE_THRESHOLD_REACHED = "billing.usage_threshold"
    PAYMENT_FAILED = "billing.payment_failed"


class WebhookDeliveryStatus(str, Enum):
    """Status of a webhook delivery attempt."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


# Retry configuration
DEFAULT_RETRY_DELAYS = [5, 30, 120, 600, 3600]  # Seconds: 5s, 30s, 2m, 10m, 1h
MAX_RETRIES = 5
DELIVERY_TIMEOUT = 30  # seconds


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Webhook:
    """Webhook configuration."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = ""
    name: str = ""
    url: str = ""
    secret: str = field(default_factory=lambda: secrets.token_hex(32))
    events: Set[WebhookEvent] = field(default_factory=set)
    is_active: bool = True

    # Configuration
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    retry_enabled: bool = True
    max_retries: int = MAX_RETRIES

    # Statistics
    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    last_triggered_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None
    consecutive_failures: int = 0

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "url": self.url,
            "events": [e.value for e in self.events],
            "is_active": self.is_active,
            "headers": self.headers,
            "timeout_seconds": self.timeout_seconds,
            "retry_enabled": self.retry_enabled,
            "max_retries": self.max_retries,
            "total_deliveries": self.total_deliveries,
            "successful_deliveries": self.successful_deliveries,
            "failed_deliveries": self.failed_deliveries,
            "success_rate": (
                self.successful_deliveries / self.total_deliveries
                if self.total_deliveries > 0 else 0
            ),
            "last_triggered_at": (
                self.last_triggered_at.isoformat() if self.last_triggered_at else None
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    webhook_id: str = ""
    event_type: WebhookEvent = WebhookEvent.CALL_STARTED
    payload: Dict[str, Any] = field(default_factory=dict)

    # Delivery details
    status: WebhookDeliveryStatus = WebhookDeliveryStatus.PENDING
    attempt_count: int = 0
    max_attempts: int = MAX_RETRIES

    # Response
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    response_headers: Dict[str, str] = field(default_factory=dict)

    # Timing
    duration_ms: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    delivered_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None

    # Error info
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "webhook_id": self.webhook_id,
            "event_type": self.event_type.value,
            "status": self.status.value,
            "attempt_count": self.attempt_count,
            "response_status": self.response_status,
            "response_body": (
                self.response_body[:500] if self.response_body else None
            ),
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "delivered_at": (
                self.delivered_at.isoformat() if self.delivered_at else None
            ),
        }


# =============================================================================
# Signature Functions
# =============================================================================


def create_signature(payload: str, secret: str, timestamp: int) -> str:
    """
    Create a webhook signature for payload verification.

    Uses HMAC-SHA256 with timestamp to prevent replay attacks.

    Args:
        payload: JSON payload string
        secret: Webhook secret key
        timestamp: Unix timestamp

    Returns:
        Signature string in format "t={timestamp},v1={signature}"
    """
    message = f"{timestamp}.{payload}"
    signature = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"t={timestamp},v1={signature}"


def verify_signature(
    payload: str,
    signature: str,
    secret: str,
    tolerance_seconds: int = 300,
) -> bool:
    """
    Verify a webhook signature.

    Args:
        payload: JSON payload string
        signature: Signature header value
        secret: Webhook secret key
        tolerance_seconds: Max age of signature in seconds

    Returns:
        True if signature is valid
    """
    try:
        # Parse signature
        parts = dict(p.split("=") for p in signature.split(","))
        timestamp = int(parts.get("t", "0"))
        received_sig = parts.get("v1", "")

        # Check timestamp
        if abs(time.time() - timestamp) > tolerance_seconds:
            return False

        # Compute expected signature
        expected = create_signature(payload, secret, timestamp)
        expected_sig = expected.split(",")[1].split("=")[1]

        # Constant-time comparison
        return hmac.compare_digest(received_sig, expected_sig)
    except Exception:
        return False


# =============================================================================
# Webhook Manager
# =============================================================================


class WebhookManager:
    """
    Manages webhook registrations and delivery.

    Features:
    - CRUD operations for webhooks
    - Event triggering and delivery
    - Automatic retries with exponential backoff
    - Delivery logging and statistics
    - Signature verification
    """

    def __init__(
        self,
        retry_delays: Optional[List[int]] = None,
        delivery_timeout: int = DELIVERY_TIMEOUT,
    ):
        self.retry_delays = retry_delays or DEFAULT_RETRY_DELAYS
        self.delivery_timeout = delivery_timeout

        # In-memory storage (replace with database in production)
        self._webhooks: Dict[str, Webhook] = {}
        self._deliveries: Dict[str, WebhookDelivery] = {}
        self._org_webhooks: Dict[str, Set[str]] = {}  # org_id -> webhook_ids
        self._lock = asyncio.Lock()  # Lock for webhook operations

        # Background retry queue
        self._retry_queue: asyncio.Queue = asyncio.Queue()
        self._retry_task: Optional[asyncio.Task] = None

        # Event handlers for extensibility
        self._pre_delivery_handlers: List[Callable] = []
        self._post_delivery_handlers: List[Callable] = []

    async def start(self) -> None:
        """Start the webhook manager background tasks."""
        if self._retry_task is None:
            self._retry_task = asyncio.create_task(self._process_retries())
            logger.info("Webhook manager started")

    async def stop(self) -> None:
        """Stop the webhook manager."""
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
            self._retry_task = None
            logger.info("Webhook manager stopped")

    # =========================================================================
    # Webhook CRUD
    # =========================================================================

    async def create_webhook(
        self,
        organization_id: str,
        url: str,
        events: List[WebhookEvent],
        name: str = "",
        headers: Optional[Dict[str, str]] = None,
        created_by: Optional[str] = None,
    ) -> Webhook:
        """
        Create a new webhook.

        Args:
            organization_id: Organization ID
            url: Webhook endpoint URL
            events: List of events to subscribe to
            name: Human-readable name
            headers: Custom headers to include
            created_by: User ID who created this

        Returns:
            Created webhook
        """
        webhook = Webhook(
            organization_id=organization_id,
            name=name or f"Webhook {len(self._webhooks) + 1}",
            url=url,
            events=set(events),
            headers=headers or {},
            created_by=created_by,
        )

        async with self._lock:
            self._webhooks[webhook.id] = webhook

            if organization_id not in self._org_webhooks:
                self._org_webhooks[organization_id] = set()
            self._org_webhooks[organization_id].add(webhook.id)

        logger.info(
            f"Created webhook {webhook.id} for org {organization_id} "
            f"with events: {[e.value for e in events]}"
        )

        return webhook

    async def get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        """Get a webhook by ID."""
        return self._webhooks.get(webhook_id)

    async def list_webhooks(
        self,
        organization_id: str,
        include_inactive: bool = False,
    ) -> List[Webhook]:
        """List all webhooks for an organization."""
        webhook_ids = self._org_webhooks.get(organization_id, set())
        webhooks = [
            self._webhooks[wid]
            for wid in webhook_ids
            if wid in self._webhooks
        ]

        if not include_inactive:
            webhooks = [w for w in webhooks if w.is_active]

        return sorted(webhooks, key=lambda w: w.created_at, reverse=True)

    async def update_webhook(
        self,
        webhook_id: str,
        **updates,
    ) -> Optional[Webhook]:
        """Update a webhook."""
        webhook = self._webhooks.get(webhook_id)
        if not webhook:
            return None

        for key, value in updates.items():
            if hasattr(webhook, key):
                if key == "events" and isinstance(value, list):
                    value = set(value)
                setattr(webhook, key, value)

        webhook.updated_at = datetime.utcnow()
        return webhook

    async def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook (thread-safe)."""
        async with self._lock:
            webhook = self._webhooks.pop(webhook_id, None)
            if webhook:
                org_webhooks = self._org_webhooks.get(webhook.organization_id, set())
                org_webhooks.discard(webhook_id)
                logger.info(f"Deleted webhook {webhook_id}")
                return True
            return False

    # =========================================================================
    # Event Triggering
    # =========================================================================

    async def trigger_event(
        self,
        organization_id: str,
        event_type: WebhookEvent,
        payload: Dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> List[WebhookDelivery]:
        """
        Trigger a webhook event.

        Sends the event to all subscribed webhooks for the organization.

        Args:
            organization_id: Organization ID
            event_type: Type of event
            payload: Event data
            idempotency_key: Optional key to prevent duplicate deliveries

        Returns:
            List of delivery records
        """
        # Find subscribed webhooks
        webhook_ids = self._org_webhooks.get(organization_id, set())
        webhooks = [
            self._webhooks[wid]
            for wid in webhook_ids
            if wid in self._webhooks
            and self._webhooks[wid].is_active
            and event_type in self._webhooks[wid].events
        ]

        if not webhooks:
            logger.debug(
                f"No webhooks subscribed to {event_type.value} "
                f"for org {organization_id}"
            )
            return []

        # Create deliveries
        deliveries = []
        for webhook in webhooks:
            delivery = await self._create_delivery(
                webhook=webhook,
                event_type=event_type,
                payload=payload,
            )
            deliveries.append(delivery)

            # Attempt delivery
            asyncio.create_task(self._deliver(webhook, delivery))

        logger.info(
            f"Triggered {event_type.value} for org {organization_id}, "
            f"queued {len(deliveries)} deliveries"
        )

        return deliveries

    async def _create_delivery(
        self,
        webhook: Webhook,
        event_type: WebhookEvent,
        payload: Dict[str, Any],
    ) -> WebhookDelivery:
        """Create a delivery record."""
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event_type=event_type,
            payload=payload,
            max_attempts=webhook.max_retries if webhook.retry_enabled else 1,
        )

        self._deliveries[delivery.id] = delivery
        return delivery

    async def _deliver(
        self,
        webhook: Webhook,
        delivery: WebhookDelivery,
    ) -> bool:
        """
        Attempt to deliver a webhook.

        Returns True if successful, False otherwise.
        """
        delivery.attempt_count += 1
        delivery.status = WebhookDeliveryStatus.PENDING

        # Prepare payload
        event_payload = {
            "id": delivery.id,
            "type": delivery.event_type.value,
            "created_at": delivery.created_at.isoformat(),
            "data": delivery.payload,
        }
        payload_str = json.dumps(event_payload)

        # Create signature
        timestamp = int(time.time())
        signature = create_signature(payload_str, webhook.secret, timestamp)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Timestamp": str(timestamp),
            "X-Webhook-Event": delivery.event_type.value,
            "X-Webhook-Delivery-Id": delivery.id,
            **webhook.headers,
        }

        # Run pre-delivery handlers
        for handler in self._pre_delivery_handlers:
            try:
                await handler(webhook, delivery)
            except Exception as e:
                logger.error(f"Pre-delivery handler error: {e}")

        # Attempt delivery
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook.url,
                    data=payload_str,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=webhook.timeout_seconds),
                ) as response:
                    delivery.response_status = response.status
                    delivery.response_body = await response.text()
                    delivery.response_headers = dict(response.headers)
                    delivery.duration_ms = (time.time() - start_time) * 1000

                    if 200 <= response.status < 300:
                        delivery.status = WebhookDeliveryStatus.SUCCESS
                        delivery.delivered_at = datetime.utcnow()

                        # Update webhook stats
                        webhook.total_deliveries += 1
                        webhook.successful_deliveries += 1
                        webhook.last_triggered_at = datetime.utcnow()
                        webhook.last_success_at = datetime.utcnow()
                        webhook.consecutive_failures = 0

                        logger.info(
                            f"Webhook delivery {delivery.id} succeeded: "
                            f"{response.status}"
                        )
                        return True
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=(),
                            status=response.status,
                            message=f"HTTP {response.status}",
                        )

        except asyncio.TimeoutError:
            delivery.error_message = "Request timed out"
            delivery.duration_ms = (time.time() - start_time) * 1000
        except aiohttp.ClientError as e:
            delivery.error_message = str(e)
            delivery.duration_ms = (time.time() - start_time) * 1000
        except Exception as e:
            delivery.error_message = f"Unexpected error: {str(e)}"
            delivery.duration_ms = (time.time() - start_time) * 1000

        # Handle failure
        webhook.total_deliveries += 1
        webhook.failed_deliveries += 1
        webhook.last_triggered_at = datetime.utcnow()
        webhook.last_failure_at = datetime.utcnow()
        webhook.consecutive_failures += 1

        # Check if we should retry
        if (
            webhook.retry_enabled
            and delivery.attempt_count < delivery.max_attempts
        ):
            delivery.status = WebhookDeliveryStatus.RETRYING
            retry_delay = self._get_retry_delay(delivery.attempt_count)
            delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=retry_delay)

            await self._retry_queue.put((webhook, delivery))
            logger.warning(
                f"Webhook delivery {delivery.id} failed, "
                f"scheduling retry {delivery.attempt_count}/{delivery.max_attempts} "
                f"in {retry_delay}s"
            )
        else:
            delivery.status = WebhookDeliveryStatus.FAILED
            logger.error(
                f"Webhook delivery {delivery.id} failed permanently: "
                f"{delivery.error_message}"
            )

        # Run post-delivery handlers
        for handler in self._post_delivery_handlers:
            try:
                await handler(webhook, delivery)
            except Exception as e:
                logger.error(f"Post-delivery handler error: {e}")

        return False

    def _get_retry_delay(self, attempt: int) -> int:
        """Get retry delay for attempt number (exponential backoff)."""
        if attempt <= len(self.retry_delays):
            return self.retry_delays[attempt - 1]
        return self.retry_delays[-1]

    async def _process_retries(self) -> None:
        """Background task to process retry queue."""
        while True:
            try:
                webhook, delivery = await self._retry_queue.get()

                if delivery.next_retry_at:
                    # Wait until retry time
                    wait_seconds = (
                        delivery.next_retry_at - datetime.utcnow()
                    ).total_seconds()
                    if wait_seconds > 0:
                        await asyncio.sleep(wait_seconds)

                # Retry delivery
                await self._deliver(webhook, delivery)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retry processing error: {e}")
                await asyncio.sleep(1)

    # =========================================================================
    # Delivery History
    # =========================================================================

    async def get_deliveries(
        self,
        webhook_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[WebhookDelivery]:
        """Get delivery history for a webhook."""
        deliveries = [
            d for d in self._deliveries.values()
            if d.webhook_id == webhook_id
        ]
        deliveries.sort(key=lambda d: d.created_at, reverse=True)
        return deliveries[offset:offset + limit]

    async def get_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get a specific delivery."""
        return self._deliveries.get(delivery_id)

    # =========================================================================
    # Testing
    # =========================================================================

    async def test_webhook(self, webhook_id: str) -> WebhookDelivery:
        """
        Send a test event to a webhook.

        Returns the delivery record.
        """
        webhook = self._webhooks.get(webhook_id)
        if not webhook:
            raise ValueError(f"Webhook not found: {webhook_id}")

        # Create test delivery
        delivery = await self._create_delivery(
            webhook=webhook,
            event_type=WebhookEvent.CALL_STARTED,
            payload={
                "test": True,
                "message": "This is a test webhook delivery",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Deliver synchronously for testing
        await self._deliver(webhook, delivery)
        return delivery

    # =========================================================================
    # Extension Points
    # =========================================================================

    def add_pre_delivery_handler(
        self,
        handler: Callable[[Webhook, WebhookDelivery], None],
    ) -> None:
        """Add a handler to run before each delivery attempt."""
        self._pre_delivery_handlers.append(handler)

    def add_post_delivery_handler(
        self,
        handler: Callable[[Webhook, WebhookDelivery], None],
    ) -> None:
        """Add a handler to run after each delivery attempt."""
        self._post_delivery_handlers.append(handler)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "WebhookManager",
    "WebhookEvent",
    "WebhookDeliveryStatus",
    "Webhook",
    "WebhookDelivery",
    "create_signature",
    "verify_signature",
]
