"""
Webhook Engine

Core webhook dispatch system:
- Event subscription
- Webhook delivery
- Retry handling
- Signature verification
"""

from typing import Optional, Dict, Any, List, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import asyncio
import aiohttp
import hashlib
import hmac
import json
import time
import logging

logger = logging.getLogger(__name__)


class WebhookStatus(str, Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    SENDING = "sending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class WebhookEventType(str, Enum):
    """Standard webhook event types."""
    # Call events
    CALL_INITIATED = "call.initiated"
    CALL_RINGING = "call.ringing"
    CALL_ANSWERED = "call.answered"
    CALL_ENDED = "call.ended"
    CALL_FAILED = "call.failed"
    CALL_TRANSFERRED = "call.transferred"

    # Conversation events
    CONVERSATION_STARTED = "conversation.started"
    CONVERSATION_ENDED = "conversation.ended"
    CONVERSATION_TURN = "conversation.turn"

    # Speech events
    SPEECH_STARTED = "speech.started"
    SPEECH_ENDED = "speech.ended"
    TRANSCRIPT_PARTIAL = "transcript.partial"
    TRANSCRIPT_FINAL = "transcript.final"

    # Agent events
    AGENT_JOINED = "agent.joined"
    AGENT_LEFT = "agent.left"
    AGENT_SPEAKING = "agent.speaking"
    AGENT_LISTENING = "agent.listening"

    # Workflow events
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_NODE = "workflow.node"

    # Recording events
    RECORDING_STARTED = "recording.started"
    RECORDING_STOPPED = "recording.stopped"
    RECORDING_AVAILABLE = "recording.available"

    # Custom events
    CUSTOM = "custom"


class SignatureMethod(str, Enum):
    """Webhook signature methods."""
    HMAC_SHA256 = "hmac_sha256"
    HMAC_SHA512 = "hmac_sha512"
    NONE = "none"


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration."""
    endpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    name: str = ""
    description: str = ""

    # Authentication
    secret: str = ""
    signature_method: SignatureMethod = SignatureMethod.HMAC_SHA256
    headers: Dict[str, str] = field(default_factory=dict)

    # Events
    events: Set[str] = field(default_factory=set)
    event_filter: Optional[Dict[str, Any]] = None

    # Configuration
    enabled: bool = True
    timeout_seconds: int = 30
    max_retries: int = 5
    retry_delay_seconds: int = 60

    # Rate limiting
    rate_limit: int = 100  # Requests per minute
    rate_limit_window: int = 60  # Seconds

    # Metadata
    tenant_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def subscribes_to(self, event_type: str) -> bool:
        """Check if endpoint subscribes to event type."""
        if not self.enabled:
            return False
        if not self.events:
            return True  # Subscribe to all if no specific events
        return event_type in self.events or "*" in self.events

    def matches_filter(self, payload: Dict[str, Any]) -> bool:
        """Check if payload matches event filter."""
        if not self.event_filter:
            return True

        for key, expected in self.event_filter.items():
            actual = payload.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False

        return True


@dataclass
class WebhookDelivery:
    """Webhook delivery record."""
    delivery_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    endpoint_id: str = ""
    event_type: str = ""

    # Request
    url: str = ""
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: WebhookStatus = WebhookStatus.PENDING
    attempt_count: int = 0
    max_attempts: int = 5

    # Response
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    response_headers: Optional[Dict[str, str]] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None

    # Error
    error_message: Optional[str] = None
    next_retry_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "delivery_id": self.delivery_id,
            "endpoint_id": self.endpoint_id,
            "event_type": self.event_type,
            "status": self.status.value,
            "attempt_count": self.attempt_count,
            "response_status": self.response_status,
            "created_at": self.created_at.isoformat(),
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
        }


@dataclass
class WebhookEvent:
    """Webhook event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Context
    tenant_id: str = ""
    call_id: Optional[str] = None
    conversation_id: Optional[str] = None
    agent_id: Optional[str] = None

    # Data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Convert to webhook payload."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "call_id": self.call_id,
            "conversation_id": self.conversation_id,
            "agent_id": self.agent_id,
            "data": self.data,
            "metadata": self.metadata,
        }


class SignatureGenerator:
    """Generate webhook signatures."""

    @staticmethod
    def generate(
        payload: str,
        secret: str,
        method: SignatureMethod = SignatureMethod.HMAC_SHA256,
        timestamp: Optional[int] = None,
    ) -> Tuple[str, int]:
        """Generate signature and timestamp."""
        ts = timestamp or int(time.time())

        if method == SignatureMethod.NONE:
            return "", ts

        # Create signed payload
        signed_payload = f"{ts}.{payload}"

        if method == SignatureMethod.HMAC_SHA256:
            signature = hmac.new(
                secret.encode(),
                signed_payload.encode(),
                hashlib.sha256,
            ).hexdigest()
        elif method == SignatureMethod.HMAC_SHA512:
            signature = hmac.new(
                secret.encode(),
                signed_payload.encode(),
                hashlib.sha512,
            ).hexdigest()
        else:
            signature = ""

        return signature, ts

    @staticmethod
    def verify(
        payload: str,
        signature: str,
        timestamp: int,
        secret: str,
        method: SignatureMethod = SignatureMethod.HMAC_SHA256,
        tolerance_seconds: int = 300,
    ) -> bool:
        """Verify webhook signature."""
        if method == SignatureMethod.NONE:
            return True

        # Check timestamp tolerance
        now = int(time.time())
        if abs(now - timestamp) > tolerance_seconds:
            return False

        # Generate expected signature
        expected, _ = SignatureGenerator.generate(payload, secret, method, timestamp)
        return hmac.compare_digest(signature, expected)


class WebhookDispatcher:
    """
    Dispatches webhooks to endpoints.

    Features:
    - Async delivery
    - Retry handling
    - Rate limiting
    - Circuit breaker
    """

    def __init__(
        self,
        max_concurrent: int = 100,
        default_timeout: int = 30,
    ):
        self._max_concurrent = max_concurrent
        self._default_timeout = default_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limiters: Dict[str, List[float]] = {}
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}

    async def send(
        self,
        endpoint: WebhookEndpoint,
        delivery: WebhookDelivery,
    ) -> WebhookDelivery:
        """Send webhook to endpoint."""
        async with self._semaphore:
            # Check rate limit
            if not self._check_rate_limit(endpoint):
                delivery.status = WebhookStatus.RETRYING
                delivery.error_message = "Rate limit exceeded"
                delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=60)
                return delivery

            # Check circuit breaker
            if self._is_circuit_open(endpoint):
                delivery.status = WebhookStatus.RETRYING
                delivery.error_message = "Circuit breaker open"
                delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=300)
                return delivery

            return await self._execute_delivery(endpoint, delivery)

    async def _execute_delivery(
        self,
        endpoint: WebhookEndpoint,
        delivery: WebhookDelivery,
    ) -> WebhookDelivery:
        """Execute webhook delivery."""
        delivery.status = WebhookStatus.SENDING
        delivery.sent_at = datetime.utcnow()
        delivery.attempt_count += 1

        start_time = time.time()

        try:
            # Build request
            payload_json = json.dumps(delivery.payload)

            # Generate signature
            signature, timestamp = SignatureGenerator.generate(
                payload_json,
                endpoint.secret,
                endpoint.signature_method,
            )

            # Build headers
            headers = {
                "Content-Type": "application/json",
                "X-Webhook-ID": delivery.delivery_id,
                "X-Webhook-Timestamp": str(timestamp),
                **endpoint.headers,
            }

            if signature:
                headers["X-Webhook-Signature"] = signature

            # Send request
            timeout = aiohttp.ClientTimeout(total=endpoint.timeout_seconds)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    endpoint.url,
                    data=payload_json,
                    headers=headers,
                ) as response:
                    delivery.response_status = response.status
                    delivery.response_headers = dict(response.headers)

                    try:
                        delivery.response_body = await response.text()
                    except:
                        pass

                    delivery.duration_ms = (time.time() - start_time) * 1000
                    delivery.completed_at = datetime.utcnow()

                    if response.status < 400:
                        delivery.status = WebhookStatus.SUCCESS
                        self._record_success(endpoint)
                    else:
                        delivery.status = WebhookStatus.FAILED
                        delivery.error_message = f"HTTP {response.status}"
                        self._record_failure(endpoint)

        except asyncio.TimeoutError:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = "Request timeout"
            delivery.duration_ms = (time.time() - start_time) * 1000
            self._record_failure(endpoint)

        except aiohttp.ClientError as e:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = str(e)
            delivery.duration_ms = (time.time() - start_time) * 1000
            self._record_failure(endpoint)

        except Exception as e:
            logger.error(f"Webhook delivery error: {e}")
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = str(e)
            self._record_failure(endpoint)

        # Schedule retry if needed
        if delivery.status == WebhookStatus.FAILED:
            if delivery.attempt_count < delivery.max_attempts:
                delivery.status = WebhookStatus.RETRYING
                delay = self._calculate_retry_delay(
                    delivery.attempt_count,
                    endpoint.retry_delay_seconds,
                )
                delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)

        return delivery

    def _check_rate_limit(self, endpoint: WebhookEndpoint) -> bool:
        """Check if rate limit allows request."""
        now = time.time()
        window_start = now - endpoint.rate_limit_window

        # Get or create rate limiter for endpoint
        if endpoint.endpoint_id not in self._rate_limiters:
            self._rate_limiters[endpoint.endpoint_id] = []

        # Clean old entries
        self._rate_limiters[endpoint.endpoint_id] = [
            ts for ts in self._rate_limiters[endpoint.endpoint_id]
            if ts > window_start
        ]

        # Check limit
        if len(self._rate_limiters[endpoint.endpoint_id]) >= endpoint.rate_limit:
            return False

        # Record request
        self._rate_limiters[endpoint.endpoint_id].append(now)
        return True

    def _is_circuit_open(self, endpoint: WebhookEndpoint) -> bool:
        """Check if circuit breaker is open."""
        breaker = self._circuit_breakers.get(endpoint.endpoint_id)
        if not breaker:
            return False

        if breaker.get("state") == "open":
            # Check if should half-open
            if time.time() > breaker.get("reset_at", 0):
                breaker["state"] = "half-open"
                return False
            return True

        return False

    def _record_success(self, endpoint: WebhookEndpoint) -> None:
        """Record successful delivery."""
        breaker = self._circuit_breakers.get(endpoint.endpoint_id, {})
        if breaker.get("state") == "half-open":
            breaker["state"] = "closed"
            breaker["failures"] = 0

    def _record_failure(self, endpoint: WebhookEndpoint) -> None:
        """Record failed delivery."""
        if endpoint.endpoint_id not in self._circuit_breakers:
            self._circuit_breakers[endpoint.endpoint_id] = {
                "state": "closed",
                "failures": 0,
                "reset_at": 0,
            }

        breaker = self._circuit_breakers[endpoint.endpoint_id]
        breaker["failures"] = breaker.get("failures", 0) + 1

        # Open circuit after 5 consecutive failures
        if breaker["failures"] >= 5:
            breaker["state"] = "open"
            breaker["reset_at"] = time.time() + 300  # 5 minute reset

    def _calculate_retry_delay(self, attempt: int, base_delay: int) -> int:
        """Calculate exponential backoff delay."""
        return min(base_delay * (2 ** (attempt - 1)), 3600)  # Max 1 hour


class WebhookEngine:
    """
    Main webhook engine.

    Features:
    - Event subscription
    - Endpoint management
    - Delivery tracking
    - Retry queue
    """

    def __init__(self):
        self._endpoints: Dict[str, WebhookEndpoint] = {}
        self._deliveries: Dict[str, WebhookDelivery] = {}
        self._dispatcher = WebhookDispatcher()
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._retry_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._retry_task: Optional[asyncio.Task] = None

    def register_endpoint(self, endpoint: WebhookEndpoint) -> None:
        """Register webhook endpoint."""
        self._endpoints[endpoint.endpoint_id] = endpoint
        logger.info(f"Registered webhook endpoint: {endpoint.endpoint_id}")

    def unregister_endpoint(self, endpoint_id: str) -> bool:
        """Unregister webhook endpoint."""
        return self._endpoints.pop(endpoint_id, None) is not None

    def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get endpoint by ID."""
        return self._endpoints.get(endpoint_id)

    def list_endpoints(
        self,
        tenant_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> List[WebhookEndpoint]:
        """List endpoints with optional filters."""
        endpoints = list(self._endpoints.values())

        if tenant_id:
            endpoints = [e for e in endpoints if e.tenant_id == tenant_id]

        if event_type:
            endpoints = [e for e in endpoints if e.subscribes_to(event_type)]

        return endpoints

    async def dispatch(self, event: WebhookEvent) -> List[WebhookDelivery]:
        """Dispatch event to all subscribed endpoints."""
        deliveries = []

        # Find subscribed endpoints
        endpoints = self.list_endpoints(
            tenant_id=event.tenant_id,
            event_type=event.event_type,
        )

        payload = event.to_payload()

        # Create and send deliveries
        for endpoint in endpoints:
            if not endpoint.matches_filter(payload):
                continue

            delivery = WebhookDelivery(
                endpoint_id=endpoint.endpoint_id,
                event_type=event.event_type,
                url=endpoint.url,
                payload=payload,
                max_attempts=endpoint.max_retries,
            )

            self._deliveries[delivery.delivery_id] = delivery

            # Send asynchronously
            result = await self._dispatcher.send(endpoint, delivery)
            deliveries.append(result)

            # Queue for retry if needed
            if result.status == WebhookStatus.RETRYING:
                await self._retry_queue.put((endpoint.endpoint_id, delivery.delivery_id))

        # Emit internal event
        await self._emit("dispatched", {
            "event": event,
            "deliveries": deliveries,
        })

        return deliveries

    def emit_sync(self, event: WebhookEvent) -> None:
        """Emit event synchronously (fire and forget)."""
        asyncio.create_task(self.dispatch(event))

    async def retry_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Manually retry a delivery."""
        delivery = self._deliveries.get(delivery_id)
        if not delivery:
            return None

        endpoint = self._endpoints.get(delivery.endpoint_id)
        if not endpoint:
            return None

        return await self._dispatcher.send(endpoint, delivery)

    def get_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get delivery by ID."""
        return self._deliveries.get(delivery_id)

    def list_deliveries(
        self,
        endpoint_id: Optional[str] = None,
        status: Optional[WebhookStatus] = None,
        limit: int = 100,
    ) -> List[WebhookDelivery]:
        """List deliveries with optional filters."""
        deliveries = list(self._deliveries.values())

        if endpoint_id:
            deliveries = [d for d in deliveries if d.endpoint_id == endpoint_id]

        if status:
            deliveries = [d for d in deliveries if d.status == status]

        # Sort by created_at descending
        deliveries.sort(key=lambda d: d.created_at, reverse=True)

        return deliveries[:limit]

    async def start(self) -> None:
        """Start the webhook engine."""
        if self._running:
            return

        self._running = True
        self._retry_task = asyncio.create_task(self._retry_worker())
        logger.info("Webhook engine started")

    async def stop(self) -> None:
        """Stop the webhook engine."""
        self._running = False

        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

        logger.info("Webhook engine stopped")

    async def _retry_worker(self) -> None:
        """Worker for processing retry queue."""
        while self._running:
            try:
                endpoint_id, delivery_id = await asyncio.wait_for(
                    self._retry_queue.get(),
                    timeout=10,
                )

                delivery = self._deliveries.get(delivery_id)
                endpoint = self._endpoints.get(endpoint_id)

                if delivery and endpoint and delivery.status == WebhookStatus.RETRYING:
                    # Wait until retry time
                    if delivery.next_retry_at:
                        wait_seconds = (delivery.next_retry_at - datetime.utcnow()).total_seconds()
                        if wait_seconds > 0:
                            await asyncio.sleep(min(wait_seconds, 60))

                    # Retry
                    result = await self._dispatcher.send(endpoint, delivery)

                    if result.status == WebhookStatus.RETRYING:
                        await self._retry_queue.put((endpoint_id, delivery_id))

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Retry worker error: {e}")
                await asyncio.sleep(1)

    def on(self, event: str, handler: Callable) -> None:
        """Register internal event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    async def _emit(self, event: str, data: Any) -> None:
        """Emit internal event."""
        for handler in self._event_handlers.get(event, []):
            try:
                result = handler(event, data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        deliveries = list(self._deliveries.values())

        return {
            "total_endpoints": len(self._endpoints),
            "enabled_endpoints": sum(1 for e in self._endpoints.values() if e.enabled),
            "total_deliveries": len(deliveries),
            "by_status": {
                status.value: sum(1 for d in deliveries if d.status == status)
                for status in WebhookStatus
            },
            "pending_retries": self._retry_queue.qsize(),
        }

    def cleanup_deliveries(self, max_age_hours: int = 24) -> int:
        """Clean up old deliveries."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        to_remove = []

        for delivery_id, delivery in self._deliveries.items():
            if delivery.created_at < cutoff:
                to_remove.append(delivery_id)

        for delivery_id in to_remove:
            del self._deliveries[delivery_id]

        return len(to_remove)


# Singleton engine instance
_engine_instance: Optional[WebhookEngine] = None


def get_webhook_engine() -> WebhookEngine:
    """Get singleton webhook engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = WebhookEngine()
    return _engine_instance


# Helper functions
def create_event(
    event_type: str,
    data: Dict[str, Any],
    tenant_id: str = "",
    call_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> WebhookEvent:
    """Create webhook event."""
    return WebhookEvent(
        event_type=event_type,
        tenant_id=tenant_id,
        call_id=call_id,
        conversation_id=conversation_id,
        agent_id=agent_id,
        data=data,
    )


def create_endpoint(
    url: str,
    events: Optional[List[str]] = None,
    secret: str = "",
    tenant_id: str = "",
    **kwargs,
) -> WebhookEndpoint:
    """Create webhook endpoint."""
    return WebhookEndpoint(
        url=url,
        events=set(events or []),
        secret=secret,
        tenant_id=tenant_id,
        **kwargs,
    )
