"""Webhooks API for Builder Engine."""

from typing import Optional, Dict, List, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hmac
import hashlib
import json
import asyncio

if TYPE_CHECKING:
    from bvrai_sdk.client import BvraiClient


class WebhookEvent(str, Enum):
    """Webhook event types."""
    # Call events
    CALL_STARTED = "call.started"
    CALL_RINGING = "call.ringing"
    CALL_ANSWERED = "call.answered"
    CALL_ENDED = "call.ended"
    CALL_FAILED = "call.failed"
    CALL_TRANSFERRED = "call.transferred"

    # Conversation events
    CONVERSATION_STARTED = "conversation.started"
    CONVERSATION_TURN = "conversation.turn"
    CONVERSATION_ENDED = "conversation.ended"

    # Transcription events
    TRANSCRIPTION_PARTIAL = "transcription.partial"
    TRANSCRIPTION_FINAL = "transcription.final"

    # Agent events
    AGENT_CREATED = "agent.created"
    AGENT_UPDATED = "agent.updated"
    AGENT_DELETED = "agent.deleted"

    # Function events
    FUNCTION_CALLED = "function.called"
    FUNCTION_COMPLETED = "function.completed"

    # Recording events
    RECORDING_READY = "recording.ready"

    # Analysis events
    CALL_ANALYZED = "call.analyzed"
    SENTIMENT_DETECTED = "sentiment.detected"


class WebhookStatus(str, Enum):
    """Webhook delivery status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILING = "failing"
    DISABLED = "disabled"


@dataclass
class Webhook:
    """A webhook endpoint configuration."""
    id: str
    url: str
    events: List[WebhookEvent]
    status: WebhookStatus
    secret: str
    description: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    retry_count: int = 3
    timeout_seconds: int = 30
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_triggered_at: Optional[datetime] = None
    failure_count: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Webhook":
        return cls(
            id=data["id"],
            url=data["url"],
            events=[WebhookEvent(e) for e in data.get("events", [])],
            status=WebhookStatus(data.get("status", "active")),
            secret=data.get("secret", ""),
            description=data.get("description"),
            headers=data.get("headers", {}),
            retry_count=data.get("retry_count", 3),
            timeout_seconds=data.get("timeout_seconds", 30),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            last_triggered_at=datetime.fromisoformat(data["last_triggered_at"]) if data.get("last_triggered_at") else None,
            failure_count=data.get("failure_count", 0),
        )


@dataclass
class WebhookDelivery:
    """A webhook delivery attempt."""
    id: str
    webhook_id: str
    event_type: WebhookEvent
    payload: Dict[str, Any]
    status_code: Optional[int]
    response_body: Optional[str]
    duration_ms: int
    success: bool
    attempt_number: int
    created_at: datetime
    error_message: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookDelivery":
        return cls(
            id=data["id"],
            webhook_id=data["webhook_id"],
            event_type=WebhookEvent(data["event_type"]),
            payload=data.get("payload", {}),
            status_code=data.get("status_code"),
            response_body=data.get("response_body"),
            duration_ms=data.get("duration_ms", 0),
            success=data.get("success", False),
            attempt_number=data.get("attempt_number", 1),
            created_at=datetime.fromisoformat(data["created_at"]),
            error_message=data.get("error_message"),
        )


@dataclass
class WebhookTestResult:
    """Result of a webhook test."""
    success: bool
    status_code: Optional[int]
    response_body: Optional[str]
    duration_ms: int
    error_message: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookTestResult":
        return cls(
            success=data.get("success", False),
            status_code=data.get("status_code"),
            response_body=data.get("response_body"),
            duration_ms=data.get("duration_ms", 0),
            error_message=data.get("error_message"),
        )


class WebhooksAPI:
    """
    Webhooks API client.

    Manage webhook endpoints for receiving real-time events.
    """

    def __init__(self, client: "BvraiClient"):
        self._client = client

    async def create(
        self,
        url: str,
        events: List[WebhookEvent],
        description: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        retry_count: int = 3,
        timeout_seconds: int = 30,
    ) -> Webhook:
        """Create a new webhook endpoint."""
        data = {
            "url": url,
            "events": [e.value for e in events],
            "description": description,
            "headers": headers or {},
            "retry_count": retry_count,
            "timeout_seconds": timeout_seconds,
        }
        response = await self._client.post("/v1/webhooks", data=data)
        return Webhook.from_dict(response)

    async def get(self, webhook_id: str) -> Webhook:
        """Get a webhook by ID."""
        response = await self._client.get(f"/v1/webhooks/{webhook_id}")
        return Webhook.from_dict(response)

    async def list(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Webhook]:
        """List all webhooks."""
        params = {"limit": limit, "offset": offset}
        response = await self._client.get("/v1/webhooks", params=params)
        return [Webhook.from_dict(w) for w in response.get("webhooks", [])]

    async def update(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[WebhookEvent]] = None,
        description: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        retry_count: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        status: Optional[WebhookStatus] = None,
    ) -> Webhook:
        """Update a webhook."""
        data = {}
        if url is not None:
            data["url"] = url
        if events is not None:
            data["events"] = [e.value for e in events]
        if description is not None:
            data["description"] = description
        if headers is not None:
            data["headers"] = headers
        if retry_count is not None:
            data["retry_count"] = retry_count
        if timeout_seconds is not None:
            data["timeout_seconds"] = timeout_seconds
        if status is not None:
            data["status"] = status.value

        response = await self._client.patch(f"/v1/webhooks/{webhook_id}", data=data)
        return Webhook.from_dict(response)

    async def delete(self, webhook_id: str) -> bool:
        """Delete a webhook."""
        await self._client.delete(f"/v1/webhooks/{webhook_id}")
        return True

    async def enable(self, webhook_id: str) -> Webhook:
        """Enable a webhook."""
        return await self.update(webhook_id, status=WebhookStatus.ACTIVE)

    async def disable(self, webhook_id: str) -> Webhook:
        """Disable a webhook."""
        return await self.update(webhook_id, status=WebhookStatus.DISABLED)

    async def rotate_secret(self, webhook_id: str) -> Webhook:
        """Rotate the webhook secret."""
        response = await self._client.post(f"/v1/webhooks/{webhook_id}/rotate-secret")
        return Webhook.from_dict(response)

    async def test(
        self,
        webhook_id: str,
        event_type: Optional[WebhookEvent] = None,
    ) -> WebhookTestResult:
        """Send a test event to a webhook."""
        data = {}
        if event_type:
            data["event_type"] = event_type.value

        response = await self._client.post(f"/v1/webhooks/{webhook_id}/test", data=data)
        return WebhookTestResult.from_dict(response)

    async def get_deliveries(
        self,
        webhook_id: str,
        limit: int = 20,
        offset: int = 0,
        success: Optional[bool] = None,
    ) -> List[WebhookDelivery]:
        """Get delivery history for a webhook."""
        params = {"limit": limit, "offset": offset}
        if success is not None:
            params["success"] = str(success).lower()

        response = await self._client.get(f"/v1/webhooks/{webhook_id}/deliveries", params=params)
        return [WebhookDelivery.from_dict(d) for d in response.get("deliveries", [])]

    async def retry_delivery(self, webhook_id: str, delivery_id: str) -> WebhookDelivery:
        """Retry a failed delivery."""
        response = await self._client.post(
            f"/v1/webhooks/{webhook_id}/deliveries/{delivery_id}/retry"
        )
        return WebhookDelivery.from_dict(response)

    # Event subscriptions helpers

    async def subscribe_to_call_events(
        self,
        url: str,
        description: Optional[str] = None,
    ) -> Webhook:
        """Subscribe to all call-related events."""
        events = [
            WebhookEvent.CALL_STARTED,
            WebhookEvent.CALL_RINGING,
            WebhookEvent.CALL_ANSWERED,
            WebhookEvent.CALL_ENDED,
            WebhookEvent.CALL_FAILED,
            WebhookEvent.CALL_TRANSFERRED,
        ]
        return await self.create(url, events, description=description or "Call events webhook")

    async def subscribe_to_conversation_events(
        self,
        url: str,
        description: Optional[str] = None,
    ) -> Webhook:
        """Subscribe to all conversation events."""
        events = [
            WebhookEvent.CONVERSATION_STARTED,
            WebhookEvent.CONVERSATION_TURN,
            WebhookEvent.CONVERSATION_ENDED,
            WebhookEvent.TRANSCRIPTION_FINAL,
        ]
        return await self.create(url, events, description=description or "Conversation events webhook")

    async def subscribe_to_all_events(
        self,
        url: str,
        description: Optional[str] = None,
    ) -> Webhook:
        """Subscribe to all available events."""
        events = list(WebhookEvent)
        return await self.create(url, events, description=description or "All events webhook")


class WebhookSignatureVerifier:
    """
    Utility for verifying webhook signatures.

    Example:
        verifier = WebhookSignatureVerifier(webhook_secret)

        # In your webhook handler:
        signature = request.headers.get("X-BVRAI-Signature")
        timestamp = request.headers.get("X-BVRAI-Timestamp")
        body = await request.body()

        if verifier.verify(body, signature, timestamp):
            # Process the webhook
            payload = verifier.parse_payload(body)
    """

    def __init__(self, secret: str):
        self.secret = secret

    def compute_signature(self, payload: bytes, timestamp: str) -> str:
        """Compute the expected signature for a payload."""
        message = f"{timestamp}.{payload.decode('utf-8')}"
        signature = hmac.new(
            self.secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"v1={signature}"

    def verify(
        self,
        payload: bytes,
        signature: str,
        timestamp: str,
        tolerance_seconds: int = 300,
    ) -> bool:
        """
        Verify a webhook signature.

        Args:
            payload: Raw request body
            signature: X-BVRAI-Signature header value
            timestamp: X-BVRAI-Timestamp header value
            tolerance_seconds: Maximum age of the request (default 5 minutes)

        Returns:
            True if signature is valid
        """
        # Check timestamp is recent
        try:
            ts = int(timestamp)
            now = int(datetime.now().timestamp())
            if abs(now - ts) > tolerance_seconds:
                return False
        except (ValueError, TypeError):
            return False

        # Compute expected signature
        expected = self.compute_signature(payload, timestamp)

        # Compare signatures (constant-time comparison)
        return hmac.compare_digest(expected, signature)

    def parse_payload(self, payload: bytes) -> Dict[str, Any]:
        """Parse the webhook payload."""
        return json.loads(payload.decode('utf-8'))


@dataclass
class WebhookPayload:
    """Parsed webhook payload."""
    id: str
    event_type: WebhookEvent
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookPayload":
        return cls(
            id=data["id"],
            event_type=WebhookEvent(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_request(
        cls,
        body: bytes,
        signature: str,
        timestamp: str,
        secret: str,
    ) -> Optional["WebhookPayload"]:
        """Parse and verify a webhook request."""
        verifier = WebhookSignatureVerifier(secret)
        if not verifier.verify(body, signature, timestamp):
            return None
        return cls.from_dict(verifier.parse_payload(body))


class WebhookHandler:
    """
    Webhook handler for processing events.

    Example:
        handler = WebhookHandler(webhook_secret)

        @handler.on(WebhookEvent.CALL_STARTED)
        async def handle_call_started(payload: WebhookPayload):
            print(f"Call started: {payload.data['call_id']}")

        @handler.on(WebhookEvent.CALL_ENDED)
        async def handle_call_ended(payload: WebhookPayload):
            print(f"Call ended: {payload.data['call_id']}")

        # In your web framework:
        async def webhook_endpoint(request):
            return await handler.handle(
                request.body,
                request.headers.get("X-BVRAI-Signature"),
                request.headers.get("X-BVRAI-Timestamp"),
            )
    """

    def __init__(self, secret: str):
        self.secret = secret
        self._handlers: Dict[WebhookEvent, List[Callable]] = {}
        self._verifier = WebhookSignatureVerifier(secret)

    def on(self, event: WebhookEvent) -> Callable:
        """Decorator to register a handler for an event."""
        def decorator(func: Callable) -> Callable:
            if event not in self._handlers:
                self._handlers[event] = []
            self._handlers[event].append(func)
            return func
        return decorator

    def add_handler(self, event: WebhookEvent, handler: Callable) -> None:
        """Add a handler for an event."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

    def remove_handler(self, event: WebhookEvent, handler: Callable) -> bool:
        """Remove a handler for an event."""
        if event in self._handlers and handler in self._handlers[event]:
            self._handlers[event].remove(handler)
            return True
        return False

    async def handle(
        self,
        body: bytes,
        signature: str,
        timestamp: str,
    ) -> Dict[str, Any]:
        """
        Handle an incoming webhook request.

        Returns a response dict with status and message.
        """
        # Verify signature
        if not self._verifier.verify(body, signature, timestamp):
            return {"status": "error", "message": "Invalid signature"}

        # Parse payload
        try:
            payload = WebhookPayload.from_dict(self._verifier.parse_payload(body))
        except Exception as e:
            return {"status": "error", "message": f"Invalid payload: {e}"}

        # Get handlers for this event
        handlers = self._handlers.get(payload.event_type, [])
        if not handlers:
            # No handlers registered, but still acknowledge
            return {"status": "ok", "message": "No handlers registered"}

        # Run all handlers
        errors = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(payload)
                else:
                    handler(payload)
            except Exception as e:
                errors.append(str(e))

        if errors:
            return {
                "status": "partial",
                "message": f"Some handlers failed: {', '.join(errors)}",
            }

        return {"status": "ok", "message": "Processed successfully"}


class WebhookBuilder:
    """
    Builder for creating webhook configurations.

    Example:
        webhook = (WebhookBuilder()
            .url("https://api.example.com/webhooks")
            .description("My webhook")
            .on_call_events()
            .with_header("X-Custom", "value")
            .build())
    """

    def __init__(self):
        self._url: Optional[str] = None
        self._events: List[WebhookEvent] = []
        self._description: Optional[str] = None
        self._headers: Dict[str, str] = {}
        self._retry_count: int = 3
        self._timeout_seconds: int = 30

    def url(self, url: str) -> "WebhookBuilder":
        """Set the webhook URL."""
        self._url = url
        return self

    def description(self, description: str) -> "WebhookBuilder":
        """Set the description."""
        self._description = description
        return self

    def event(self, event: WebhookEvent) -> "WebhookBuilder":
        """Add a single event."""
        if event not in self._events:
            self._events.append(event)
        return self

    def events(self, events: List[WebhookEvent]) -> "WebhookBuilder":
        """Add multiple events."""
        for e in events:
            self.event(e)
        return self

    def on_call_events(self) -> "WebhookBuilder":
        """Subscribe to all call events."""
        return self.events([
            WebhookEvent.CALL_STARTED,
            WebhookEvent.CALL_RINGING,
            WebhookEvent.CALL_ANSWERED,
            WebhookEvent.CALL_ENDED,
            WebhookEvent.CALL_FAILED,
            WebhookEvent.CALL_TRANSFERRED,
        ])

    def on_conversation_events(self) -> "WebhookBuilder":
        """Subscribe to all conversation events."""
        return self.events([
            WebhookEvent.CONVERSATION_STARTED,
            WebhookEvent.CONVERSATION_TURN,
            WebhookEvent.CONVERSATION_ENDED,
        ])

    def on_transcription_events(self) -> "WebhookBuilder":
        """Subscribe to transcription events."""
        return self.events([
            WebhookEvent.TRANSCRIPTION_PARTIAL,
            WebhookEvent.TRANSCRIPTION_FINAL,
        ])

    def on_all_events(self) -> "WebhookBuilder":
        """Subscribe to all events."""
        return self.events(list(WebhookEvent))

    def with_header(self, key: str, value: str) -> "WebhookBuilder":
        """Add a custom header."""
        self._headers[key] = value
        return self

    def with_headers(self, headers: Dict[str, str]) -> "WebhookBuilder":
        """Add multiple custom headers."""
        self._headers.update(headers)
        return self

    def retry_count(self, count: int) -> "WebhookBuilder":
        """Set retry count."""
        self._retry_count = count
        return self

    def timeout(self, seconds: int) -> "WebhookBuilder":
        """Set timeout in seconds."""
        self._timeout_seconds = seconds
        return self

    def build(self) -> Dict[str, Any]:
        """Build the webhook configuration."""
        if not self._url:
            raise ValueError("URL is required")
        if not self._events:
            raise ValueError("At least one event is required")

        return {
            "url": self._url,
            "events": self._events,
            "description": self._description,
            "headers": self._headers,
            "retry_count": self._retry_count,
            "timeout_seconds": self._timeout_seconds,
        }

    async def create(self, api: WebhooksAPI) -> Webhook:
        """Create the webhook using the API."""
        config = self.build()
        return await api.create(
            url=config["url"],
            events=config["events"],
            description=config["description"],
            headers=config["headers"],
            retry_count=config["retry_count"],
            timeout_seconds=config["timeout_seconds"],
        )
