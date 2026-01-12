"""Exceptions for Builder Engine SDK."""

from typing import Optional, List, Any


class BvraiError(Exception):
    """Base exception for BVRAI SDK errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code

    def __str__(self):
        parts = [self.message]
        if self.status_code:
            parts.append(f"(HTTP {self.status_code})")
        if self.error_code:
            parts.append(f"[{self.error_code}]")
        return " ".join(parts)


class AuthenticationError(BvraiError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401, error_code="AUTH_ERROR")


class NotFoundError(BvraiError):
    """Raised when a resource is not found."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404, error_code="NOT_FOUND")


class RateLimitError(BvraiError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
    ):
        super().__init__(message, status_code=429, error_code="RATE_LIMIT")
        self.retry_after = retry_after


class ValidationError(BvraiError):
    """Raised when request validation fails."""

    def __init__(
        self,
        message: str = "Validation failed",
        errors: Optional[List[dict]] = None,
    ):
        super().__init__(message, status_code=422, error_code="VALIDATION_ERROR")
        self.errors = errors or []


class QuotaExceededError(BvraiError):
    """Raised when account quota is exceeded."""

    def __init__(
        self,
        message: str = "Quota exceeded",
        quota_type: Optional[str] = None,
    ):
        super().__init__(message, status_code=403, error_code="QUOTA_EXCEEDED")
        self.quota_type = quota_type


class ConflictError(BvraiError):
    """Raised when there's a resource conflict."""

    def __init__(self, message: str = "Resource conflict"):
        super().__init__(message, status_code=409, error_code="CONFLICT")


class ServerError(BvraiError):
    """Raised when server encounters an error."""

    def __init__(self, message: str = "Server error"):
        super().__init__(message, status_code=500, error_code="SERVER_ERROR")


class ServiceUnavailableError(BvraiError):
    """Raised when service is temporarily unavailable."""

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        retry_after: int = 30,
    ):
        super().__init__(message, status_code=503, error_code="SERVICE_UNAVAILABLE")
        self.retry_after = retry_after


class TimeoutError(BvraiError):
    """Raised when a request times out."""

    def __init__(self, message: str = "Request timed out"):
        super().__init__(message, error_code="TIMEOUT")


class WebSocketError(BvraiError):
    """Raised when WebSocket connection fails."""

    def __init__(self, message: str = "WebSocket error"):
        super().__init__(message, error_code="WEBSOCKET_ERROR")


class ConnectionError(BvraiError):
    """Raised when connection to API fails."""

    def __init__(self, message: str = "Connection failed"):
        super().__init__(message, error_code="CONNECTION_ERROR")


class RequestError(BvraiError):
    """Raised when request fails."""

    def __init__(self, message: str = "Request failed"):
        super().__init__(message, error_code="REQUEST_ERROR")


class PaymentRequiredError(BvraiError):
    """Raised when payment is required."""

    def __init__(self, message: str = "Payment required"):
        super().__init__(message, status_code=402, error_code="PAYMENT_REQUIRED")
