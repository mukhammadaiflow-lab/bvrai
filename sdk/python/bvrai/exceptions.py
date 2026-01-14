"""
Custom exceptions for the BVRAI SDK.

This module provides a comprehensive exception hierarchy for handling
all types of errors that may occur when interacting with the BVRAI API.
"""

from typing import Any, Dict, Optional


class BVRAIError(Exception):
    """
    Base exception for all BVRAI SDK errors.

    All exceptions raised by the SDK inherit from this class,
    making it easy to catch all SDK-related errors.

    Attributes:
        message: Human-readable error description
        code: Optional error code from the API
        details: Optional additional error details
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, code={self.code!r})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details,
        }


# =============================================================================
# HTTP/API Errors
# =============================================================================

class APIError(BVRAIError):
    """
    Base exception for API-related errors.

    Raised when the API returns an error response.

    Attributes:
        status_code: HTTP status code from the response
        request_id: Unique request ID for debugging
        response_body: Raw response body if available
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        response_body: Optional[str] = None
    ):
        super().__init__(message, code, details)
        self.status_code = status_code
        self.request_id = request_id
        self.response_body = response_body

    def __str__(self) -> str:
        base = f"[HTTP {self.status_code}]"
        if self.code:
            base += f" [{self.code}]"
        base += f" {self.message}"
        if self.request_id:
            base += f" (request_id: {self.request_id})"
        return base

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "status_code": self.status_code,
            "request_id": self.request_id,
        })
        return result


class AuthenticationError(APIError):
    """
    Raised when authentication fails.

    This typically occurs when:
    - API key is missing or invalid
    - API key has been revoked
    - API key has expired

    Example:
        try:
            client = BVRAIClient(api_key="invalid-key")
            client.agents.list()
        except AuthenticationError as e:
            print(f"Auth failed: {e}")
    """

    def __init__(
        self,
        message: str = "Authentication failed. Please check your API key.",
        **kwargs
    ):
        super().__init__(message, status_code=401, code="auth_error", **kwargs)


class AuthorizationError(APIError):
    """
    Raised when the authenticated user lacks permissions.

    This occurs when:
    - User doesn't have access to the requested resource
    - User's role doesn't permit the action
    - Resource belongs to a different organization
    """

    def __init__(
        self,
        message: str = "You don't have permission to access this resource.",
        **kwargs
    ):
        super().__init__(message, status_code=403, code="forbidden", **kwargs)


class NotFoundError(APIError):
    """
    Raised when a requested resource doesn't exist.

    Attributes:
        resource_type: Type of resource that wasn't found
        resource_id: ID of the resource that wasn't found
    """

    def __init__(
        self,
        message: str = "The requested resource was not found.",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        if resource_type and resource_id:
            message = f"{resource_type} with ID '{resource_id}' was not found."

        super().__init__(message, status_code=404, code="not_found", details=details, **kwargs)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ValidationError(APIError):
    """
    Raised when request data fails validation.

    Attributes:
        field_errors: Dictionary mapping field names to error messages
    """

    def __init__(
        self,
        message: str = "The request data was invalid.",
        field_errors: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if field_errors:
            details["field_errors"] = field_errors

        super().__init__(message, status_code=422, code="validation_error", details=details, **kwargs)
        self.field_errors = field_errors or {}

    def get_field_error(self, field: str) -> Optional[str]:
        """Get error message for a specific field."""
        return self.field_errors.get(field)


class ConflictError(APIError):
    """
    Raised when a resource conflict occurs.

    This happens when:
    - Creating a resource that already exists
    - Updating a resource with stale data (optimistic locking)
    - Violating a unique constraint
    """

    def __init__(
        self,
        message: str = "The request conflicts with the current state of the resource.",
        **kwargs
    ):
        super().__init__(message, status_code=409, code="conflict", **kwargs)


class RateLimitError(APIError):
    """
    Raised when API rate limits are exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying
        limit: The rate limit that was exceeded
        remaining: Requests remaining in the current window
        reset_at: Timestamp when the rate limit resets
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded. Please slow down your requests.",
        retry_after: Optional[int] = None,
        limit: Optional[int] = None,
        remaining: Optional[int] = None,
        reset_at: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if retry_after:
            details["retry_after"] = retry_after
        if limit:
            details["limit"] = limit
        if remaining is not None:
            details["remaining"] = remaining
        if reset_at:
            details["reset_at"] = reset_at

        super().__init__(message, status_code=429, code="rate_limit_exceeded", details=details, **kwargs)
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at


class ServerError(APIError):
    """
    Raised when the API returns a 5xx error.

    This indicates a problem on the server side.
    These errors are typically transient and can be retried.
    """

    def __init__(
        self,
        message: str = "An internal server error occurred. Please try again later.",
        status_code: int = 500,
        **kwargs
    ):
        super().__init__(message, status_code=status_code, code="server_error", **kwargs)


class ServiceUnavailableError(APIError):
    """
    Raised when the API is temporarily unavailable.

    This may occur during:
    - Planned maintenance
    - High load conditions
    - Temporary outages

    Attributes:
        retry_after: Seconds to wait before retrying
    """

    def __init__(
        self,
        message: str = "The service is temporarily unavailable. Please try again later.",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(message, status_code=503, code="service_unavailable", details=details, **kwargs)
        self.retry_after = retry_after


# =============================================================================
# Network/Connection Errors
# =============================================================================

class ConnectionError(BVRAIError):
    """
    Raised when a network connection error occurs.

    This may happen due to:
    - Network connectivity issues
    - DNS resolution failures
    - Firewall blocking
    """

    def __init__(
        self,
        message: str = "Failed to connect to the BVRAI API.",
        **kwargs
    ):
        super().__init__(message, code="connection_error", **kwargs)


class TimeoutError(BVRAIError):
    """
    Raised when an API request times out.

    Attributes:
        timeout: The timeout value that was exceeded (in seconds)
    """

    def __init__(
        self,
        message: str = "The request timed out.",
        timeout: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if timeout:
            details["timeout"] = timeout
            message = f"The request timed out after {timeout} seconds."

        super().__init__(message, code="timeout", details=details, **kwargs)
        self.timeout = timeout


# =============================================================================
# SDK Configuration Errors
# =============================================================================

class ConfigurationError(BVRAIError):
    """
    Raised when the SDK is misconfigured.

    This occurs when:
    - Required configuration is missing
    - Configuration values are invalid
    """

    def __init__(
        self,
        message: str = "The SDK is not properly configured.",
        **kwargs
    ):
        super().__init__(message, code="configuration_error", **kwargs)


class InvalidAPIKeyError(ConfigurationError):
    """
    Raised when an API key format is invalid.

    API keys must start with 'bvr_' prefix.
    """

    def __init__(
        self,
        message: str = "Invalid API key format. Keys must start with 'bvr_'.",
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.code = "invalid_api_key"


# =============================================================================
# Resource-Specific Errors
# =============================================================================

class AgentError(BVRAIError):
    """Base exception for agent-related errors."""
    pass


class AgentNotActiveError(AgentError):
    """Raised when trying to use an agent that isn't active."""

    def __init__(
        self,
        agent_id: str,
        status: str,
        **kwargs
    ):
        message = f"Agent '{agent_id}' is not active (current status: {status})."
        super().__init__(message, code="agent_not_active", **kwargs)
        self.agent_id = agent_id
        self.status = status


class AgentVersionError(AgentError):
    """Raised when there's an issue with agent versioning."""

    def __init__(
        self,
        message: str = "Agent version conflict.",
        **kwargs
    ):
        super().__init__(message, code="agent_version_error", **kwargs)


class CallError(BVRAIError):
    """Base exception for call-related errors."""
    pass


class CallNotFoundError(CallError):
    """Raised when a call cannot be found."""

    def __init__(self, call_id: str, **kwargs):
        message = f"Call '{call_id}' was not found."
        super().__init__(message, code="call_not_found", **kwargs)
        self.call_id = call_id


class CallInProgressError(CallError):
    """Raised when an operation cannot be performed on an active call."""

    def __init__(
        self,
        call_id: str,
        message: str = "Cannot perform this operation while the call is in progress.",
        **kwargs
    ):
        super().__init__(message, code="call_in_progress", **kwargs)
        self.call_id = call_id


class ConversationError(BVRAIError):
    """Base exception for conversation-related errors."""
    pass


class ConversationEndedError(ConversationError):
    """Raised when trying to interact with an ended conversation."""

    def __init__(self, conversation_id: str, **kwargs):
        message = f"Conversation '{conversation_id}' has already ended."
        super().__init__(message, code="conversation_ended", **kwargs)
        self.conversation_id = conversation_id


class WebhookError(BVRAIError):
    """Base exception for webhook-related errors."""
    pass


class WebhookDeliveryError(WebhookError):
    """Raised when a webhook delivery fails."""

    def __init__(
        self,
        webhook_id: str,
        delivery_id: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        message = f"Webhook delivery failed for webhook '{webhook_id}'."
        if status_code:
            message += f" (HTTP {status_code})"

        super().__init__(message, code="webhook_delivery_failed", **kwargs)
        self.webhook_id = webhook_id
        self.delivery_id = delivery_id
        self.status_code = status_code


class WebhookSignatureError(WebhookError):
    """Raised when webhook signature verification fails."""

    def __init__(
        self,
        message: str = "Webhook signature verification failed.",
        **kwargs
    ):
        super().__init__(message, code="invalid_signature", **kwargs)


# =============================================================================
# Voice Processing Errors
# =============================================================================

class VoiceError(BVRAIError):
    """Base exception for voice processing errors."""
    pass


class STTError(VoiceError):
    """Raised when speech-to-text processing fails."""

    def __init__(
        self,
        message: str = "Speech-to-text processing failed.",
        provider: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if provider:
            details["provider"] = provider

        super().__init__(message, code="stt_error", details=details, **kwargs)
        self.provider = provider


class TTSError(VoiceError):
    """Raised when text-to-speech processing fails."""

    def __init__(
        self,
        message: str = "Text-to-speech processing failed.",
        provider: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if provider:
            details["provider"] = provider

        super().__init__(message, code="tts_error", details=details, **kwargs)
        self.provider = provider


class AudioFormatError(VoiceError):
    """Raised when audio format is invalid or unsupported."""

    def __init__(
        self,
        message: str = "Invalid or unsupported audio format.",
        format: Optional[str] = None,
        supported_formats: Optional[list] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {}) or {}
        if format:
            details["format"] = format
        if supported_formats:
            details["supported_formats"] = supported_formats

        super().__init__(message, code="invalid_audio_format", details=details, **kwargs)
        self.format = format
        self.supported_formats = supported_formats


# =============================================================================
# Billing Errors
# =============================================================================

class BillingError(BVRAIError):
    """Base exception for billing-related errors."""
    pass


class InsufficientCreditsError(BillingError):
    """Raised when account has insufficient credits."""

    def __init__(
        self,
        required: Optional[float] = None,
        available: Optional[float] = None,
        **kwargs
    ):
        message = "Insufficient credits to complete this operation."
        if required and available is not None:
            message = f"Insufficient credits. Required: {required}, Available: {available}."

        super().__init__(message, code="insufficient_credits", **kwargs)
        self.required = required
        self.available = available


class PaymentRequiredError(BillingError):
    """Raised when payment is required to continue."""

    def __init__(
        self,
        message: str = "Payment required. Please update your billing information.",
        **kwargs
    ):
        super().__init__(message, code="payment_required", **kwargs)


class SubscriptionError(BillingError):
    """Raised when there's an issue with the subscription."""

    def __init__(
        self,
        message: str = "Subscription issue. Please check your subscription status.",
        **kwargs
    ):
        super().__init__(message, code="subscription_error", **kwargs)


class PlanLimitExceededError(BillingError):
    """Raised when a plan limit is exceeded."""

    def __init__(
        self,
        limit_type: str,
        current: int,
        limit: int,
        **kwargs
    ):
        message = f"Plan limit exceeded for {limit_type}. Current: {current}, Limit: {limit}."

        details = kwargs.pop("details", {}) or {}
        details.update({
            "limit_type": limit_type,
            "current": current,
            "limit": limit,
        })

        super().__init__(message, code="plan_limit_exceeded", details=details, **kwargs)
        self.limit_type = limit_type
        self.current = current
        self.limit = limit


# =============================================================================
# Exception Factory
# =============================================================================

def from_response(
    status_code: int,
    response_data: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> APIError:
    """
    Create an appropriate exception from an API response.

    This factory function examines the status code and response data
    to create the most specific exception type.

    Args:
        status_code: HTTP status code
        response_data: Parsed JSON response body
        request_id: Request ID from response headers

    Returns:
        Appropriate APIError subclass instance
    """
    response_data = response_data or {}

    message = response_data.get("message", response_data.get("error", "Unknown error"))
    code = response_data.get("code")
    details = response_data.get("details", {})

    common_kwargs = {
        "message": message,
        "code": code,
        "details": details,
        "request_id": request_id,
    }

    # Map status codes to exception types
    if status_code == 401:
        return AuthenticationError(**common_kwargs)
    elif status_code == 403:
        return AuthorizationError(**common_kwargs)
    elif status_code == 404:
        return NotFoundError(
            resource_type=details.get("resource_type"),
            resource_id=details.get("resource_id"),
            **common_kwargs
        )
    elif status_code == 409:
        return ConflictError(**common_kwargs)
    elif status_code == 422:
        return ValidationError(
            field_errors=details.get("field_errors"),
            **common_kwargs
        )
    elif status_code == 429:
        return RateLimitError(
            retry_after=details.get("retry_after"),
            limit=details.get("limit"),
            remaining=details.get("remaining"),
            reset_at=details.get("reset_at"),
            **common_kwargs
        )
    elif status_code == 503:
        return ServiceUnavailableError(
            retry_after=details.get("retry_after"),
            **common_kwargs
        )
    elif status_code >= 500:
        return ServerError(status_code=status_code, **common_kwargs)
    else:
        return APIError(status_code=status_code, **common_kwargs)


# =============================================================================
# Exception Groups (for Python 3.11+)
# =============================================================================

class BVRAIExceptionGroup(ExceptionGroup, BVRAIError):
    """
    Group of related BVRAI exceptions.

    Useful for reporting multiple errors from batch operations.
    """

    def __init__(self, message: str, exceptions: list):
        ExceptionGroup.__init__(self, message, exceptions)
        BVRAIError.__init__(self, message)


# =============================================================================
# Retry Helpers
# =============================================================================

def is_retryable(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if the error is transient and can be retried
    """
    if isinstance(error, RateLimitError):
        return True
    if isinstance(error, ServiceUnavailableError):
        return True
    if isinstance(error, ServerError) and error.status_code >= 500:
        return True
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    return False


def get_retry_after(error: Exception) -> Optional[int]:
    """
    Get the recommended retry delay for an error.

    Args:
        error: The exception to check

    Returns:
        Recommended retry delay in seconds, or None
    """
    if isinstance(error, RateLimitError):
        return error.retry_after or 60
    if isinstance(error, ServiceUnavailableError):
        return error.retry_after or 30
    if isinstance(error, ServerError):
        return 5
    if isinstance(error, (ConnectionError, TimeoutError)):
        return 2
    return None
