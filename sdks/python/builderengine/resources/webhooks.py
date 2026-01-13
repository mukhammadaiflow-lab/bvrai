"""
Builder Engine Python SDK - Webhooks Resource

This module provides methods for managing webhooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List

from builderengine.resources.base import BaseResource, PaginatedResponse
from builderengine.models import Webhook, WebhookEvent
from builderengine.config import Endpoints

if TYPE_CHECKING:
    from builderengine.client import BuilderEngine


class WebhooksResource(BaseResource):
    """
    Resource for managing webhooks.

    Webhooks allow you to receive real-time notifications about events
    in your Builder Engine account. This resource provides methods for
    creating, managing, and testing webhooks.

    Example:
        >>> client = BuilderEngine(api_key="...")
        >>> webhook = client.webhooks.create(
        ...     url="https://api.example.com/webhooks/builderengine",
        ...     events=[WebhookEvent.CALL_STARTED, WebhookEvent.CALL_ENDED]
        ... )
    """

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        enabled: Optional[bool] = None,
    ) -> PaginatedResponse[Webhook]:
        """
        List all webhooks.

        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            enabled: Filter by enabled status

        Returns:
            PaginatedResponse containing Webhook objects
        """
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            enabled=enabled,
        )
        response = self._get(Endpoints.WEBHOOKS, params=params)
        return self._parse_paginated_response(response, Webhook)

    def get(self, webhook_id: str) -> Webhook:
        """
        Get a webhook by ID.

        Args:
            webhook_id: The webhook's unique identifier

        Returns:
            Webhook object
        """
        path = Endpoints.WEBHOOK.format(webhook_id=webhook_id)
        response = self._get(path)
        return Webhook.from_dict(response)

    def create(
        self,
        url: str,
        events: List[WebhookEvent],
        description: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        enabled: bool = True,
        retry_count: int = 3,
        timeout_seconds: int = 30,
    ) -> Webhook:
        """
        Create a new webhook.

        Args:
            url: The webhook endpoint URL
            events: List of events to subscribe to
            description: Description of the webhook
            headers: Custom headers to include in webhook requests
            enabled: Whether the webhook is enabled
            retry_count: Number of retry attempts
            timeout_seconds: Request timeout in seconds

        Returns:
            Created Webhook object with secret

        Example:
            >>> webhook = client.webhooks.create(
            ...     url="https://api.example.com/webhooks",
            ...     events=[
            ...         WebhookEvent.CALL_STARTED,
            ...         WebhookEvent.CALL_ENDED,
            ...         WebhookEvent.TRANSCRIPTION_READY
            ...     ],
            ...     headers={"X-Custom-Header": "value"}
            ... )
            >>> print(f"Secret: {webhook.secret}")  # Save this!
        """
        data: Dict[str, Any] = {
            "url": url,
            "events": [e.value for e in events],
            "enabled": enabled,
            "retry_count": retry_count,
            "timeout_seconds": timeout_seconds,
        }

        if description:
            data["description"] = description
        if headers:
            data["headers"] = headers

        response = self._post(Endpoints.WEBHOOKS, json=data)
        return Webhook.from_dict(response)

    def update(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[WebhookEvent]] = None,
        description: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        enabled: Optional[bool] = None,
        retry_count: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Webhook:
        """
        Update a webhook.

        Args:
            webhook_id: The webhook's unique identifier
            url: New endpoint URL
            events: New list of events
            description: New description
            headers: New custom headers
            enabled: New enabled status
            retry_count: New retry count
            timeout_seconds: New timeout

        Returns:
            Updated Webhook object
        """
        data: Dict[str, Any] = {}

        if url is not None:
            data["url"] = url
        if events is not None:
            data["events"] = [e.value for e in events]
        if description is not None:
            data["description"] = description
        if headers is not None:
            data["headers"] = headers
        if enabled is not None:
            data["enabled"] = enabled
        if retry_count is not None:
            data["retry_count"] = retry_count
        if timeout_seconds is not None:
            data["timeout_seconds"] = timeout_seconds

        path = Endpoints.WEBHOOK.format(webhook_id=webhook_id)
        response = self._patch(path, json=data)
        return Webhook.from_dict(response)

    def delete(self, webhook_id: str) -> None:
        """
        Delete a webhook.

        Args:
            webhook_id: The webhook's unique identifier
        """
        path = Endpoints.WEBHOOK.format(webhook_id=webhook_id)
        self._delete(path)

    def enable(self, webhook_id: str) -> Webhook:
        """
        Enable a webhook.

        Args:
            webhook_id: The webhook's unique identifier

        Returns:
            Updated Webhook object
        """
        return self.update(webhook_id, enabled=True)

    def disable(self, webhook_id: str) -> Webhook:
        """
        Disable a webhook.

        Args:
            webhook_id: The webhook's unique identifier

        Returns:
            Updated Webhook object
        """
        return self.update(webhook_id, enabled=False)

    def test(
        self,
        webhook_id: str,
        event: WebhookEvent = WebhookEvent.CALL_STARTED,
    ) -> Dict[str, Any]:
        """
        Send a test event to a webhook.

        Args:
            webhook_id: The webhook's unique identifier
            event: Event type to simulate

        Returns:
            Test result with response details

        Example:
            >>> result = client.webhooks.test("webhook_abc123")
            >>> print(f"Status: {result['status_code']}")
        """
        path = Endpoints.WEBHOOK_TEST.format(webhook_id=webhook_id)
        response = self._post(path, json={"event": event.value})
        return response

    def rotate_secret(self, webhook_id: str) -> Webhook:
        """
        Rotate the webhook secret.

        This generates a new secret for the webhook. Make sure to
        update your endpoint to use the new secret.

        Args:
            webhook_id: The webhook's unique identifier

        Returns:
            Updated Webhook object with new secret
        """
        path = Endpoints.WEBHOOK_ROTATE_SECRET.format(webhook_id=webhook_id)
        response = self._post(path)
        return Webhook.from_dict(response)

    def get_deliveries(
        self,
        webhook_id: str,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        event: Optional[WebhookEvent] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get delivery history for a webhook.

        Args:
            webhook_id: The webhook's unique identifier
            page: Page number
            page_size: Items per page
            status: Filter by status (success, failed, pending)
            event: Filter by event type
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            Paginated list of deliveries with request/response details
        """
        path = Endpoints.WEBHOOK_DELIVERIES.format(webhook_id=webhook_id)
        params = self._build_pagination_params(
            page=page,
            page_size=page_size,
            status=status,
            event=event.value if event else None,
            start_date=start_date,
            end_date=end_date,
        )
        return self._get(path, params=params)

    def retry_delivery(self, webhook_id: str, delivery_id: str) -> Dict[str, Any]:
        """
        Retry a failed webhook delivery.

        Args:
            webhook_id: The webhook's unique identifier
            delivery_id: The delivery's unique identifier

        Returns:
            Retry result with new delivery details
        """
        path = f"{Endpoints.WEBHOOK_DELIVERIES.format(webhook_id=webhook_id)}/{delivery_id}/retry"
        return self._post(path)

    @staticmethod
    def verify_signature(
        payload: bytes,
        signature: str,
        secret: str,
        tolerance_seconds: int = 300,
    ) -> bool:
        """
        Verify a webhook signature.

        Use this in your webhook endpoint to validate that the
        request came from Builder Engine.

        Args:
            payload: Raw request body bytes
            signature: Value of X-BuilderEngine-Signature header
            secret: Your webhook secret
            tolerance_seconds: Maximum age of the request

        Returns:
            True if signature is valid

        Example:
            >>> @app.route("/webhook", methods=["POST"])
            ... def handle_webhook():
            ...     is_valid = WebhooksResource.verify_signature(
            ...         payload=request.data,
            ...         signature=request.headers.get("X-BuilderEngine-Signature"),
            ...         secret=WEBHOOK_SECRET
            ...     )
            ...     if not is_valid:
            ...         return "Invalid signature", 401
            ...     # Process webhook...
        """
        import hmac
        import hashlib
        import time

        try:
            # Parse signature header: t=timestamp,v1=signature
            parts = dict(part.split("=") for part in signature.split(","))
            timestamp = int(parts.get("t", 0))
            expected_sig = parts.get("v1", "")

            # Check timestamp tolerance
            current_time = int(time.time())
            if abs(current_time - timestamp) > tolerance_seconds:
                return False

            # Compute expected signature
            signed_payload = f"{timestamp}.{payload.decode('utf-8')}"
            computed_sig = hmac.new(
                secret.encode("utf-8"),
                signed_payload.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(computed_sig, expected_sig)

        except Exception:
            return False

    @staticmethod
    def get_event_types() -> List[Dict[str, Any]]:
        """
        Get all available webhook event types.

        Returns:
            List of event types with descriptions
        """
        return [
            {"event": e.value, "description": e.name.replace("_", " ").title()}
            for e in WebhookEvent
        ]
