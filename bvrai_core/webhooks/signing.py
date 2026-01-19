"""Webhook payload signing and verification."""

import hashlib
import hmac
import json
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SignatureComponents:
    """Components of a webhook signature."""
    timestamp: int
    signature: str
    algorithm: str = "sha256"


class WebhookSigner:
    """
    Signs and verifies webhook payloads using HMAC-SHA256.

    The signature scheme is compatible with common webhook verification patterns
    used by Stripe, GitHub, etc.

    Signature format:
        t=<timestamp>,v1=<signature>

    The signature is computed as:
        HMAC-SHA256(secret, "<timestamp>.<payload>")
    """

    SIGNATURE_VERSION = "v1"
    DEFAULT_TOLERANCE_SECONDS = 300  # 5 minutes

    def __init__(self, secret: str):
        """
        Initialize the signer.

        Args:
            secret: The webhook signing secret
        """
        self.secret = secret.encode("utf-8")

    def sign(self, payload: str, timestamp: Optional[int] = None) -> str:
        """
        Sign a payload.

        Args:
            payload: The JSON payload string
            timestamp: Unix timestamp (defaults to current time)

        Returns:
            The signature header value (e.g., "t=1234567890,v1=abc123...")
        """
        if timestamp is None:
            timestamp = int(time.time())

        # Create signature
        signed_payload = f"{timestamp}.{payload}"
        signature = hmac.new(
            self.secret,
            signed_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return f"t={timestamp},{self.SIGNATURE_VERSION}={signature}"

    def verify(
        self,
        payload: str,
        signature_header: str,
        tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a webhook signature.

        Args:
            payload: The raw JSON payload string
            signature_header: The signature header value
            tolerance_seconds: Maximum age of the signature in seconds

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Parse signature header
            components = self._parse_signature_header(signature_header)

            if components is None:
                return False, "Invalid signature format"

            # Check timestamp
            current_time = int(time.time())
            if abs(current_time - components.timestamp) > tolerance_seconds:
                return False, "Signature timestamp out of tolerance"

            # Compute expected signature
            expected = self._compute_signature(payload, components.timestamp)

            # Constant-time comparison
            if not hmac.compare_digest(components.signature, expected):
                return False, "Signature mismatch"

            return True, None

        except Exception as e:
            return False, f"Verification error: {str(e)}"

    def _parse_signature_header(self, header: str) -> Optional[SignatureComponents]:
        """Parse the signature header."""
        try:
            parts = {}
            for item in header.split(","):
                key, value = item.split("=", 1)
                parts[key] = value

            if "t" not in parts or self.SIGNATURE_VERSION not in parts:
                return None

            return SignatureComponents(
                timestamp=int(parts["t"]),
                signature=parts[self.SIGNATURE_VERSION],
            )
        except Exception:
            return None

    def _compute_signature(self, payload: str, timestamp: int) -> str:
        """Compute the expected signature."""
        signed_payload = f"{timestamp}.{payload}"
        return hmac.new(
            self.secret,
            signed_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def generate_secret(length: int = 32) -> str:
        """
        Generate a random webhook secret.

        Args:
            length: Length of the secret in bytes

        Returns:
            Hex-encoded secret string
        """
        import secrets
        return secrets.token_hex(length)


class SignatureVerifier:
    """Helper class for verifying webhook signatures."""

    def __init__(self, tolerance_seconds: int = 300):
        """
        Initialize the verifier.

        Args:
            tolerance_seconds: Maximum age of the signature
        """
        self.tolerance_seconds = tolerance_seconds

    def verify(
        self,
        payload: str,
        signature_header: str,
        secret: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a webhook signature.

        Args:
            payload: The raw JSON payload string
            signature_header: The signature header value
            secret: The webhook secret

        Returns:
            Tuple of (is_valid, error_message)
        """
        signer = WebhookSigner(secret)
        return signer.verify(payload, signature_header, self.tolerance_seconds)


def create_signed_payload(
    secret: str,
    event_type: str,
    data: Dict,
    metadata: Optional[Dict] = None,
) -> Tuple[str, str, Dict[str, str]]:
    """
    Create a signed webhook payload with headers.

    Args:
        secret: The webhook secret
        event_type: The event type
        data: The event data
        metadata: Optional metadata

    Returns:
        Tuple of (payload_json, signature, headers)
    """
    timestamp = int(time.time())

    payload = {
        "id": f"evt_{timestamp}",
        "type": event_type,
        "created": timestamp,
        "data": data,
    }

    if metadata:
        payload["metadata"] = metadata

    payload_json = json.dumps(payload, separators=(",", ":"))

    signer = WebhookSigner(secret)
    signature = signer.sign(payload_json, timestamp)

    headers = {
        "Content-Type": "application/json",
        "X-Webhook-Signature": signature,
        "X-Webhook-Id": payload["id"],
        "X-Webhook-Timestamp": str(timestamp),
    }

    return payload_json, signature, headers
