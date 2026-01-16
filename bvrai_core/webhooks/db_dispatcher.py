"""
Database-backed Webhook Event Dispatcher

Integrates webhook delivery with the database repositories for persistent
webhook configurations and delivery logging.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from ..database.repositories import WebhookRepository
from ..database.models import Webhook, WebhookDelivery

logger = logging.getLogger(__name__)


# =============================================================================
# Webhook Event Types
# =============================================================================

class WebhookEvents:
    """All supported webhook event types."""

    # Call events
    CALL_STARTED = "call.started"
    CALL_ENDED = "call.ended"
    CALL_FAILED = "call.failed"
    CALL_TRANSFERRED = "call.transferred"
    CALL_RECORDING_READY = "call.recording_ready"
    CALL_TRANSCRIPT_READY = "call.transcript_ready"

    # Agent events
    AGENT_CREATED = "agent.created"
    AGENT_UPDATED = "agent.updated"
    AGENT_DELETED = "agent.deleted"
    AGENT_DEPLOYED = "agent.deployed"

    # Campaign events
    CAMPAIGN_STARTED = "campaign.started"
    CAMPAIGN_COMPLETED = "campaign.completed"
    CAMPAIGN_PAUSED = "campaign.paused"
    CAMPAIGN_CONTACT_CALLED = "campaign.contact_called"

    # Knowledge base events
    KNOWLEDGE_BASE_UPDATED = "knowledge_base.updated"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_FAILED = "document.failed"

    # Billing events
    USAGE_THRESHOLD = "billing.usage_threshold"
    PAYMENT_FAILED = "billing.payment_failed"

    ALL_EVENTS = [
        CALL_STARTED, CALL_ENDED, CALL_FAILED, CALL_TRANSFERRED,
        CALL_RECORDING_READY, CALL_TRANSCRIPT_READY,
        AGENT_CREATED, AGENT_UPDATED, AGENT_DELETED, AGENT_DEPLOYED,
        CAMPAIGN_STARTED, CAMPAIGN_COMPLETED, CAMPAIGN_PAUSED, CAMPAIGN_CONTACT_CALLED,
        KNOWLEDGE_BASE_UPDATED, DOCUMENT_PROCESSED, DOCUMENT_FAILED,
        USAGE_THRESHOLD, PAYMENT_FAILED,
    ]


# =============================================================================
# Signature Functions
# =============================================================================

def create_webhook_signature(payload: str, secret: str, timestamp: int) -> str:
    """
    Create HMAC-SHA256 signature for webhook payload.

    Format: t={timestamp},v1={signature}
    """
    message = f"{timestamp}.{payload}"
    signature = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"t={timestamp},v1={signature}"


def verify_webhook_signature(
    payload: str,
    signature: str,
    secret: str,
    tolerance_seconds: int = 300,
) -> bool:
    """Verify webhook signature with timestamp tolerance."""
    try:
        parts = dict(p.split("=") for p in signature.split(","))
        timestamp = int(parts.get("t", "0"))
        received_sig = parts.get("v1", "")

        # Check timestamp freshness
        if abs(time.time() - timestamp) > tolerance_seconds:
            return False

        # Compute expected signature
        expected = create_webhook_signature(payload, secret, timestamp)
        expected_sig = expected.split(",")[1].split("=")[1]

        return hmac.compare_digest(received_sig, expected_sig)
    except Exception:
        return False


# =============================================================================
# Webhook Dispatcher
# =============================================================================

class WebhookDispatcher:
    """
    Database-backed webhook event dispatcher.

    Features:
    - Sends webhooks based on database-stored configurations
    - Logs all delivery attempts to database
    - Automatic retries with exponential backoff
    - HMAC signature generation
    - Concurrent delivery with rate limiting
    """

    DEFAULT_TIMEOUT = 30
    MAX_CONCURRENT = 50
    RETRY_DELAYS = [60, 300, 900, 3600]  # 1min, 5min, 15min, 1hr

    def __init__(self, db_session_factory):
        """
        Initialize dispatcher.

        Args:
            db_session_factory: Async session factory for database access
        """
        self.db_session_factory = db_session_factory
        self._http_client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._running = False
        self._retry_queue: asyncio.Queue = asyncio.Queue()
        self._retry_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the dispatcher."""
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.DEFAULT_TIMEOUT, connect=10.0),
            follow_redirects=False,
            http2=True,
        )
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)
        self._running = True
        self._retry_task = asyncio.create_task(self._process_retries())
        logger.info("Webhook dispatcher started")

    async def stop(self) -> None:
        """Stop the dispatcher."""
        self._running = False
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
        if self._http_client:
            await self._http_client.aclose()
        logger.info("Webhook dispatcher stopped")

    async def dispatch(
        self,
        organization_id: str,
        event_type: str,
        data: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """
        Dispatch an event to all matching webhooks.

        Args:
            organization_id: Organization ID
            event_type: Event type (e.g., 'call.started')
            data: Event payload data
            agent_id: Optional agent ID for filtering

        Returns:
            List of delivery IDs
        """
        async with self.db_session_factory() as session:
            repo = WebhookRepository(session)

            # Get matching webhooks
            webhooks = await repo.get_for_event(
                organization_id=organization_id,
                event_type=event_type,
                agent_id=agent_id,
            )

            if not webhooks:
                logger.debug(f"No webhooks for {event_type} in org {organization_id}")
                return []

            delivery_ids = []

            for webhook in webhooks:
                # Create delivery record
                event_id = str(uuid.uuid4())
                delivery = await repo.create_delivery(
                    webhook_id=webhook.id,
                    event_type=event_type,
                    event_id=event_id,
                    request_url=webhook.url,
                    request_headers=webhook.headers if hasattr(webhook, 'headers') else {},
                    request_body=data,
                )
                await session.commit()

                delivery_ids.append(delivery.id)

                # Start async delivery
                asyncio.create_task(
                    self._deliver(webhook, delivery, data, event_type, event_id)
                )

            logger.info(
                f"Dispatched {event_type} to {len(webhooks)} webhooks "
                f"for org {organization_id}"
            )

            return delivery_ids

    async def _deliver(
        self,
        webhook: Webhook,
        delivery: WebhookDelivery,
        data: Dict[str, Any],
        event_type: str,
        event_id: str,
        attempt: int = 1,
    ) -> bool:
        """Deliver webhook with retry logic."""
        if not self._http_client or not self._running:
            return False

        async with self._semaphore:
            # Build payload
            payload = {
                "id": event_id,
                "type": event_type,
                "created_at": datetime.utcnow().isoformat(),
                "data": data,
            }
            payload_str = json.dumps(payload)

            # Create signature
            timestamp = int(time.time())
            signature = create_webhook_signature(
                payload_str,
                webhook.secret or "default-secret",
                timestamp,
            )

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "BVRAI-Webhook/1.0",
                "X-Webhook-Signature": signature,
                "X-Webhook-Timestamp": str(timestamp),
                "X-Webhook-Event": event_type,
                "X-Webhook-Delivery-Id": delivery.id,
            }

            # Add custom headers if present
            if hasattr(webhook, 'headers') and webhook.headers:
                # Don't expose headers dict directly for security
                pass

            # Attempt delivery
            start_time = time.time()
            try:
                timeout = webhook.timeout_seconds if hasattr(webhook, 'timeout_seconds') else self.DEFAULT_TIMEOUT
                response = await self._http_client.post(
                    webhook.url,
                    content=payload_str,
                    headers=headers,
                    timeout=timeout,
                )

                duration_ms = int((time.time() - start_time) * 1000)

                # Read response
                response_body = None
                try:
                    response_body = response.text[:5000]
                except Exception:
                    pass

                # Update delivery record
                async with self.db_session_factory() as session:
                    repo = WebhookRepository(session)

                    if 200 <= response.status_code < 300:
                        await repo.update_delivery(
                            delivery_id=delivery.id,
                            status="success",
                            response_status=response.status_code,
                            response_body=response_body,
                            duration_ms=duration_ms,
                        )
                        await session.commit()

                        logger.info(
                            f"Webhook delivered: {webhook.id} "
                            f"status={response.status_code} "
                            f"duration={duration_ms}ms"
                        )
                        return True
                    else:
                        # Non-2xx response - handle as failure
                        raise httpx.HTTPStatusError(
                            f"HTTP {response.status_code}",
                            request=response.request,
                            response=response,
                        )

            except asyncio.TimeoutError:
                error_msg = "Request timeout"
                duration_ms = int((time.time() - start_time) * 1000)
            except httpx.ConnectError as e:
                error_msg = f"Connection error: {str(e)}"
                duration_ms = int((time.time() - start_time) * 1000)
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code}"
                duration_ms = int((time.time() - start_time) * 1000)
            except Exception as e:
                error_msg = f"Delivery error: {str(e)}"
                duration_ms = int((time.time() - start_time) * 1000)
                logger.exception(f"Webhook delivery error: {e}")

            # Handle failure
            max_retries = webhook.max_retries if hasattr(webhook, 'max_retries') else 3

            if attempt < max_retries:
                # Schedule retry
                async with self.db_session_factory() as session:
                    repo = WebhookRepository(session)
                    await repo.update_delivery(
                        delivery_id=delivery.id,
                        status="retrying",
                        error_message=error_msg,
                        duration_ms=duration_ms,
                    )
                    await session.commit()

                # Queue for retry
                retry_delay = self.RETRY_DELAYS[min(attempt - 1, len(self.RETRY_DELAYS) - 1)]
                await self._retry_queue.put((
                    webhook, delivery, data, event_type, event_id, attempt + 1, retry_delay
                ))

                logger.warning(
                    f"Webhook delivery failed, retrying in {retry_delay}s: "
                    f"{webhook.id} attempt {attempt}/{max_retries}"
                )
            else:
                # Final failure
                async with self.db_session_factory() as session:
                    repo = WebhookRepository(session)
                    await repo.update_delivery(
                        delivery_id=delivery.id,
                        status="failed",
                        error_message=error_msg,
                        duration_ms=duration_ms,
                    )
                    await session.commit()

                logger.error(
                    f"Webhook delivery failed permanently: {webhook.id} "
                    f"after {attempt} attempts: {error_msg}"
                )

            return False

    async def _process_retries(self) -> None:
        """Background task to process retry queue."""
        while self._running:
            try:
                item = await asyncio.wait_for(
                    self._retry_queue.get(),
                    timeout=1.0,
                )
                webhook, delivery, data, event_type, event_id, attempt, delay = item

                # Wait for retry delay
                await asyncio.sleep(delay)

                # Retry delivery
                await self._deliver(
                    webhook, delivery, data, event_type, event_id, attempt
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retry processing error: {e}")
                await asyncio.sleep(1)

    async def test_webhook(
        self,
        webhook_id: str,
        organization_id: str,
    ) -> Dict[str, Any]:
        """
        Send a test event to a webhook.

        Returns delivery result.
        """
        async with self.db_session_factory() as session:
            repo = WebhookRepository(session)
            webhook = await repo.get_by_id(webhook_id)

            if not webhook or webhook.organization_id != organization_id:
                return {
                    "success": False,
                    "error": "Webhook not found",
                }

            # Create test delivery
            event_id = str(uuid.uuid4())
            delivery = await repo.create_delivery(
                webhook_id=webhook.id,
                event_type="test.ping",
                event_id=event_id,
                request_url=webhook.url,
                request_body={"test": True, "message": "Test webhook delivery"},
            )
            await session.commit()

            # Deliver synchronously for test
            success = await self._deliver(
                webhook=webhook,
                delivery=delivery,
                data={"test": True, "message": "Test webhook delivery"},
                event_type="test.ping",
                event_id=event_id,
            )

            return {
                "success": success,
                "delivery_id": delivery.id,
            }


# =============================================================================
# Global Dispatcher Instance
# =============================================================================

_dispatcher: Optional[WebhookDispatcher] = None


def get_dispatcher() -> Optional[WebhookDispatcher]:
    """Get the global dispatcher instance."""
    return _dispatcher


async def init_dispatcher(db_session_factory) -> WebhookDispatcher:
    """Initialize and start the global dispatcher."""
    global _dispatcher
    _dispatcher = WebhookDispatcher(db_session_factory)
    await _dispatcher.start()
    return _dispatcher


async def shutdown_dispatcher() -> None:
    """Shutdown the global dispatcher."""
    global _dispatcher
    if _dispatcher:
        await _dispatcher.stop()
        _dispatcher = None


# =============================================================================
# Convenience Functions
# =============================================================================

async def dispatch_webhook(
    organization_id: str,
    event_type: str,
    data: Dict[str, Any],
    agent_id: Optional[str] = None,
) -> List[str]:
    """
    Dispatch a webhook event using the global dispatcher.

    Usage:
        from bvrai_core.webhooks.db_dispatcher import dispatch_webhook

        await dispatch_webhook(
            organization_id="org-123",
            event_type="call.started",
            data={
                "call_id": "call-456",
                "agent_id": "agent-789",
                "from_number": "+14155551234",
            }
        )
    """
    if _dispatcher:
        return await _dispatcher.dispatch(
            organization_id=organization_id,
            event_type=event_type,
            data=data,
            agent_id=agent_id,
        )
    else:
        logger.warning("Webhook dispatcher not initialized")
        return []


__all__ = [
    "WebhookDispatcher",
    "WebhookEvents",
    "create_webhook_signature",
    "verify_webhook_signature",
    "get_dispatcher",
    "init_dispatcher",
    "shutdown_dispatcher",
    "dispatch_webhook",
]
