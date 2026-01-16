"""Webhook delivery service."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from uuid import uuid4

import httpx
import redis.asyncio as redis

from .models import (
    Webhook,
    WebhookEvent,
    DeliveryLog,
    DeliveryAttempt,
    DeliveryStatus,
    WebhookStatus,
    WebhookStats,
)
from .signing import WebhookSigner, create_signed_payload

logger = logging.getLogger(__name__)


@dataclass
class DeliveryResult:
    """Result of a webhook delivery attempt."""
    success: bool
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[int] = None


class WebhookDeliveryService:
    """
    Service for delivering webhooks with retry support.

    Features:
    - Async HTTP delivery with configurable timeouts
    - Automatic retries with exponential backoff
    - Signature generation for payload verification
    - Delivery logging and statistics
    - Circuit breaker for failing endpoints
    """

    # Circuit breaker settings
    FAILURE_THRESHOLD = 10  # Disable after this many consecutive failures
    CIRCUIT_RESET_SECONDS = 3600  # Re-enable after this duration

    # Retry backoff (exponential)
    RETRY_BASE_DELAY = 60  # First retry after 60 seconds
    RETRY_MAX_DELAY = 3600  # Max retry delay of 1 hour

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_concurrent_deliveries: int = 100,
    ):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.max_concurrent_deliveries = max_concurrent_deliveries

        # In-memory state
        self._webhooks: Dict[str, Webhook] = {}
        self._delivery_semaphore: Optional[asyncio.Semaphore] = None
        self._lock = asyncio.Lock()

        # HTTP client
        self._http_client: Optional[httpx.AsyncClient] = None

        # Callbacks
        self._on_delivery_success: List[Callable[[DeliveryLog], Any]] = []
        self._on_delivery_failure: List[Callable[[DeliveryLog], Any]] = []

    async def start(self) -> None:
        """Start the delivery service."""
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        self._delivery_semaphore = asyncio.Semaphore(self.max_concurrent_deliveries)
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=False,
            http2=True,
        )

        # Load webhooks
        await self._load_webhooks()

        logger.info("Webhook delivery service started")

    async def stop(self) -> None:
        """Stop the delivery service."""
        if self._http_client:
            await self._http_client.aclose()

        if self.redis:
            await self.redis.close()

        logger.info("Webhook delivery service stopped")

    # Webhook management

    async def register_webhook(self, webhook: Webhook) -> Webhook:
        """Register a new webhook endpoint."""
        async with self._lock:
            self._webhooks[webhook.id] = webhook

        # Persist to Redis
        if self.redis:
            await self.redis.hset(
                f"webhooks:{webhook.organization_id}",
                webhook.id,
                json.dumps(webhook.to_dict()),
            )

        logger.info(f"Registered webhook: {webhook.id} -> {webhook.url}")
        return webhook

    async def update_webhook(self, webhook_id: str, updates: Dict[str, Any]) -> Optional[Webhook]:
        """Update a webhook."""
        async with self._lock:
            if webhook_id not in self._webhooks:
                return None

            webhook = self._webhooks[webhook_id]
            for key, value in updates.items():
                if hasattr(webhook, key):
                    setattr(webhook, key, value)
            webhook.updated_at = datetime.utcnow()

            self._webhooks[webhook_id] = webhook

        # Persist to Redis
        if self.redis:
            await self.redis.hset(
                f"webhooks:{webhook.organization_id}",
                webhook.id,
                json.dumps(webhook.to_dict()),
            )

        return webhook

    async def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook."""
        async with self._lock:
            if webhook_id not in self._webhooks:
                return False

            webhook = self._webhooks.pop(webhook_id)

        # Remove from Redis
        if self.redis:
            await self.redis.hdel(f"webhooks:{webhook.organization_id}", webhook_id)

        return True

    async def get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        """Get a webhook by ID."""
        async with self._lock:
            return self._webhooks.get(webhook_id)

    async def list_webhooks(self, organization_id: str) -> List[Webhook]:
        """List webhooks for an organization."""
        async with self._lock:
            return [
                w for w in self._webhooks.values()
                if w.organization_id == organization_id
            ]

    # Event delivery

    async def dispatch_event(
        self,
        organization_id: str,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Dispatch an event to all matching webhooks.

        Args:
            organization_id: The organization ID
            event_type: The event type
            data: The event data
            metadata: Optional event metadata

        Returns:
            List of delivery log IDs
        """
        # Find matching webhooks
        webhooks = await self._get_matching_webhooks(organization_id, event_type)

        if not webhooks:
            return []

        # Create delivery tasks
        delivery_ids = []
        tasks = []

        for webhook in webhooks:
            event = WebhookEvent(
                id=str(uuid4()),
                webhook_id=webhook.id,
                organization_id=organization_id,
                event_type=event_type,
                payload=data,
                metadata=metadata or {},
            )

            delivery_log = DeliveryLog(
                id=str(uuid4()),
                webhook_id=webhook.id,
                event_id=event.id,
                event_type=event_type,
                organization_id=organization_id,
                url=webhook.url,
                status=DeliveryStatus.PENDING,
                payload=data,
            )

            delivery_ids.append(delivery_log.id)
            tasks.append(self._deliver_with_retry(webhook, event, delivery_log))

        # Execute deliveries concurrently (fire-and-forget but ensure tasks are started)
        if tasks:
            # Create background task to avoid blocking the return
            asyncio.create_task(asyncio.gather(*tasks, return_exceptions=True))

        return delivery_ids

    async def _deliver_with_retry(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        delivery_log: DeliveryLog,
    ) -> None:
        """Deliver an event with retry logic."""
        max_attempts = webhook.max_retries + 1 if webhook.retry_enabled else 1

        for attempt_num in range(1, max_attempts + 1):
            # Wait for semaphore
            async with self._delivery_semaphore:
                delivery_log.status = DeliveryStatus.IN_PROGRESS

                # Perform delivery
                result = await self._deliver_event(webhook, event, attempt_num)

                # Record attempt
                attempt = DeliveryAttempt(
                    attempt_number=attempt_num,
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    status_code=result.status_code,
                    response_body=result.response_body[:1000] if result.response_body else None,
                    error_message=result.error_message,
                    response_time_ms=result.response_time_ms,
                )
                delivery_log.attempts.append(attempt)

                if result.success:
                    delivery_log.status = DeliveryStatus.SUCCESS
                    delivery_log.completed_at = datetime.utcnow()

                    # Update webhook stats
                    await self._record_success(webhook)

                    # Store log
                    await self._store_delivery_log(delivery_log)

                    # Notify callbacks
                    for callback in self._on_delivery_success:
                        try:
                            await callback(delivery_log)
                        except Exception as e:
                            logger.error(f"Delivery success callback error: {e}")

                    logger.info(
                        f"Webhook delivered: {webhook.id} ({event.event_type}) "
                        f"status={result.status_code}"
                    )
                    return

                # Delivery failed
                await self._record_failure(webhook)

                if attempt_num < max_attempts:
                    # Schedule retry
                    delay = self._calculate_retry_delay(attempt_num, webhook)
                    delivery_log.status = DeliveryStatus.RETRYING
                    delivery_log.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)

                    await self._store_delivery_log(delivery_log)

                    logger.warning(
                        f"Webhook delivery failed, retrying in {delay}s: "
                        f"{webhook.id} attempt {attempt_num}/{max_attempts}"
                    )

                    await asyncio.sleep(delay)
                else:
                    # Final failure
                    delivery_log.status = DeliveryStatus.FAILED
                    delivery_log.completed_at = datetime.utcnow()

                    await self._store_delivery_log(delivery_log)

                    # Notify callbacks
                    for callback in self._on_delivery_failure:
                        try:
                            await callback(delivery_log)
                        except Exception as e:
                            logger.error(f"Delivery failure callback error: {e}")

                    logger.error(
                        f"Webhook delivery failed permanently: {webhook.id} "
                        f"({event.event_type}) after {max_attempts} attempts"
                    )

    async def _deliver_event(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        attempt_num: int,
    ) -> DeliveryResult:
        """Perform a single delivery attempt."""
        if not self._http_client:
            return DeliveryResult(
                success=False,
                error_message="HTTP client not initialized",
            )

        try:
            # Create signed payload
            payload_json, signature, headers = create_signed_payload(
                secret=webhook.secret,
                event_type=event.event_type,
                data=event.payload,
                metadata={
                    "event_id": event.id,
                    "webhook_id": webhook.id,
                    "attempt": attempt_num,
                },
            )

            # Add custom headers
            headers.update(webhook.headers)
            headers["User-Agent"] = "BuilderEngine-Webhook/1.0"

            # Send request
            start_time = time.time()

            response = await self._http_client.post(
                webhook.url,
                content=payload_json,
                headers=headers,
                timeout=webhook.timeout_seconds,
            )

            response_time_ms = int((time.time() - start_time) * 1000)

            # Read response body
            response_body = None
            try:
                response_body = response.text[:5000]  # Limit response body size
            except Exception:
                pass

            # Check success (2xx status codes)
            if 200 <= response.status_code < 300:
                return DeliveryResult(
                    success=True,
                    status_code=response.status_code,
                    response_body=response_body,
                    response_time_ms=response_time_ms,
                )
            else:
                return DeliveryResult(
                    success=False,
                    status_code=response.status_code,
                    response_body=response_body,
                    error_message=f"HTTP {response.status_code}",
                    response_time_ms=response_time_ms,
                )

        except httpx.TimeoutException:
            return DeliveryResult(
                success=False,
                error_message="Request timeout",
            )
        except httpx.ConnectError as e:
            return DeliveryResult(
                success=False,
                error_message=f"Connection error: {str(e)}",
            )
        except Exception as e:
            logger.exception(f"Webhook delivery error: {e}")
            return DeliveryResult(
                success=False,
                error_message=f"Delivery error: {str(e)}",
            )

    def _calculate_retry_delay(self, attempt: int, webhook: Webhook) -> int:
        """Calculate retry delay using exponential backoff."""
        base_delay = webhook.retry_delay_seconds or self.RETRY_BASE_DELAY

        # Exponential backoff: base * 2^(attempt-1)
        delay = base_delay * (2 ** (attempt - 1))

        # Add jitter (up to 10%)
        import random
        jitter = delay * random.uniform(0, 0.1)

        return min(int(delay + jitter), self.RETRY_MAX_DELAY)

    async def _record_success(self, webhook: Webhook) -> None:
        """Record a successful delivery."""
        async with self._lock:
            webhook.consecutive_failures = 0
            webhook.last_success_at = datetime.utcnow()
            webhook.last_triggered_at = datetime.utcnow()

            # Re-enable if was disabled
            if webhook.status == WebhookStatus.FAILED:
                webhook.status = WebhookStatus.ACTIVE
                logger.info(f"Webhook re-enabled after successful delivery: {webhook.id}")

    async def _record_failure(self, webhook: Webhook) -> None:
        """Record a failed delivery."""
        async with self._lock:
            webhook.consecutive_failures += 1
            webhook.last_triggered_at = datetime.utcnow()

            # Circuit breaker: disable after too many failures
            if webhook.consecutive_failures >= self.FAILURE_THRESHOLD:
                if webhook.status == WebhookStatus.ACTIVE:
                    webhook.status = WebhookStatus.FAILED
                    logger.warning(
                        f"Webhook disabled after {webhook.consecutive_failures} "
                        f"consecutive failures: {webhook.id}"
                    )

    # Delivery log management

    async def _store_delivery_log(self, log: DeliveryLog) -> None:
        """Store a delivery log in Redis."""
        if not self.redis:
            return

        # Store in sorted set by timestamp
        key = f"webhook_logs:{log.webhook_id}"
        await self.redis.zadd(
            key,
            {json.dumps(log.to_dict_full()): log.created_at.timestamp()},
        )

        # Trim to last 1000 logs per webhook
        await self.redis.zremrangebyrank(key, 0, -1001)

        # Also store in organization-wide log
        org_key = f"org_webhook_logs:{log.organization_id}"
        await self.redis.zadd(
            org_key,
            {json.dumps(log.to_dict()): log.created_at.timestamp()},
        )
        await self.redis.zremrangebyrank(org_key, 0, -10001)

    async def get_delivery_logs(
        self,
        webhook_id: str,
        limit: int = 100,
        status: Optional[DeliveryStatus] = None,
    ) -> List[DeliveryLog]:
        """Get delivery logs for a webhook."""
        if not self.redis:
            return []

        key = f"webhook_logs:{webhook_id}"
        results = await self.redis.zrevrange(key, 0, limit - 1)

        logs = []
        for data in results:
            try:
                log_dict = json.loads(data)
                log_dict["status"] = DeliveryStatus(log_dict["status"])
                log = DeliveryLog(**log_dict)

                if status is None or log.status == status:
                    logs.append(log)
            except Exception as e:
                logger.warning(f"Failed to parse delivery log: {e}")

        return logs

    async def get_webhook_stats(self, webhook_id: str) -> WebhookStats:
        """Get statistics for a webhook."""
        logs = await self.get_delivery_logs(webhook_id, limit=1000)

        if not logs:
            return WebhookStats(webhook_id=webhook_id)

        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)

        total = len(logs)
        successful = sum(1 for l in logs if l.status == DeliveryStatus.SUCCESS)
        failed = sum(1 for l in logs if l.status == DeliveryStatus.FAILED)
        pending = sum(1 for l in logs if l.status in (DeliveryStatus.PENDING, DeliveryStatus.RETRYING))

        # Last 24h stats
        recent_logs = [l for l in logs if l.created_at > last_24h]
        last_24h_deliveries = len(recent_logs)
        last_24h_failures = sum(1 for l in recent_logs if l.status == DeliveryStatus.FAILED)

        # Average response time
        response_times = [l.response_time_ms for l in logs if l.response_time_ms]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        return WebhookStats(
            webhook_id=webhook_id,
            total_deliveries=total,
            successful_deliveries=successful,
            failed_deliveries=failed,
            pending_deliveries=pending,
            avg_response_time_ms=avg_response_time,
            success_rate=successful / total if total > 0 else 0,
            last_24h_deliveries=last_24h_deliveries,
            last_24h_failures=last_24h_failures,
        )

    # Test delivery

    async def test_webhook(
        self,
        webhook_id: str,
        event_type: str = "test.ping",
    ) -> DeliveryResult:
        """Send a test event to a webhook."""
        webhook = await self.get_webhook(webhook_id)
        if not webhook:
            return DeliveryResult(success=False, error_message="Webhook not found")

        event = WebhookEvent(
            id=str(uuid4()),
            webhook_id=webhook.id,
            organization_id=webhook.organization_id,
            event_type=event_type,
            payload={
                "message": "This is a test webhook event",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        result = await self._deliver_event(webhook, event, 1)
        return result

    # Retry failed deliveries

    async def retry_delivery(self, log_id: str) -> Optional[DeliveryResult]:
        """Manually retry a failed delivery."""
        # This would need to be implemented with proper log lookup
        # For now, return None
        return None

    # Helper methods

    async def _get_matching_webhooks(
        self,
        organization_id: str,
        event_type: str,
    ) -> List[Webhook]:
        """Get webhooks that should receive an event."""
        async with self._lock:
            return [
                w for w in self._webhooks.values()
                if w.organization_id == organization_id
                and w.should_receive_event(event_type)
            ]

    async def _load_webhooks(self) -> None:
        """Load webhooks from Redis."""
        if not self.redis:
            return

        # Scan for all webhook hashes
        async for key in self.redis.scan_iter(match="webhooks:*"):
            webhooks_data = await self.redis.hgetall(key)
            for webhook_id, data in webhooks_data.items():
                try:
                    webhook_dict = json.loads(data)
                    webhook_dict["status"] = WebhookStatus(webhook_dict["status"])
                    webhook_dict["created_at"] = datetime.fromisoformat(webhook_dict["created_at"])
                    webhook_dict["updated_at"] = datetime.fromisoformat(webhook_dict["updated_at"])

                    if webhook_dict.get("last_triggered_at"):
                        webhook_dict["last_triggered_at"] = datetime.fromisoformat(
                            webhook_dict["last_triggered_at"]
                        )
                    if webhook_dict.get("last_success_at"):
                        webhook_dict["last_success_at"] = datetime.fromisoformat(
                            webhook_dict["last_success_at"]
                        )

                    webhook = Webhook(**webhook_dict)
                    async with self._lock:
                        self._webhooks[webhook.id] = webhook

                except Exception as e:
                    logger.warning(f"Failed to load webhook {webhook_id}: {e}")

        logger.info(f"Loaded {len(self._webhooks)} webhooks")

    # Callbacks

    def on_delivery_success(self, callback: Callable[[DeliveryLog], Any]) -> None:
        """Register callback for successful deliveries."""
        self._on_delivery_success.append(callback)

    def on_delivery_failure(self, callback: Callable[[DeliveryLog], Any]) -> None:
        """Register callback for failed deliveries."""
        self._on_delivery_failure.append(callback)
