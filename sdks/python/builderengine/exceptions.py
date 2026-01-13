"""
Builder Engine Python SDK - Exceptions

This module contains all custom exceptions used by the SDK.
"""

from typing import Optional, Dict, Any


class BuilderEngineError(Exception):
    """
    Base exception for all Builder Engine SDK errors.

    Attributes:
        message: Human-readable error message
        code: Error code if available
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', code='{self.code}')"


class AuthenticationError(BuilderEngineError):
    """
    Raised when authentication fails.

    This can occur when:
    - API key is invalid or expired
    - API key lacks required permissions
    - Authentication token is malformed
    """

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, code="AUTHENTICATION_ERROR")


class AuthorizationError(BuilderEngineError):
    """
    Raised when the user lacks permission for an action.

    This occurs when the authenticated user doesn't have
    sufficient permissions to perform the requested operation.
    """

    def __init__(self, message: str = "Authorization failed") -> None:
        super().__init__(message, code="AUTHORIZATION_ERROR")


class RateLimitError(BuilderEngineError):
    """
    Raised when the API rate limit is exceeded.

    Attributes:
        retry_after: Number of seconds to wait before retrying
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[str] = None,
    ) -> None:
        super().__init__(message, code="RATE_LIMIT_ERROR")
        self.retry_after = int(retry_after) if retry_after else None

    def __str__(self) -> str:
        base = super().__str__()
        if self.retry_after:
            return f"{base}. Retry after {self.retry_after} seconds."
        return base


class ValidationError(BuilderEngineError):
    """
    Raised when request validation fails.

    This occurs when the request body or parameters
    don't meet the API's validation requirements.

    Attributes:
        field_errors: Dictionary mapping field names to error messages
    """

    def __init__(
        self,
        message: str = "Validation failed",
        field_errors: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(message, code="VALIDATION_ERROR", details=field_errors)
        self.field_errors = field_errors or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.field_errors:
            errors = ", ".join(f"{k}: {v}" for k, v in self.field_errors.items())
            return f"{base} ({errors})"
        return base


class NotFoundError(BuilderEngineError):
    """
    Raised when a requested resource is not found.

    This occurs when attempting to access a resource
    that doesn't exist or has been deleted.

    Attributes:
        resource_type: Type of resource that wasn't found
        resource_id: ID of the resource that wasn't found
    """

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
    ) -> None:
        super().__init__(message, code="NOT_FOUND")
        self.resource_type = resource_type
        self.resource_id = resource_id

    def __str__(self) -> str:
        if self.resource_type and self.resource_id:
            return f"{self.resource_type} with ID '{self.resource_id}' not found"
        return super().__str__()


class ConflictError(BuilderEngineError):
    """
    Raised when there's a resource conflict.

    This can occur when:
    - Creating a resource that already exists
    - Attempting concurrent modifications
    - Violating unique constraints
    """

    def __init__(self, message: str = "Resource conflict") -> None:
        super().__init__(message, code="CONFLICT")


class ServerError(BuilderEngineError):
    """
    Raised when a server error occurs.

    This indicates an unexpected error on the API server.
    These errors are typically transient and can be retried.

    Attributes:
        request_id: Request ID for debugging
    """

    def __init__(
        self,
        message: str = "Server error",
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message, code="SERVER_ERROR")
        self.request_id = request_id

    def __str__(self) -> str:
        base = super().__str__()
        if self.request_id:
            return f"{base} (Request ID: {self.request_id})"
        return base


class TimeoutError(BuilderEngineError):
    """
    Raised when a request times out.

    This can occur when:
    - The server takes too long to respond
    - Network connectivity issues
    - Long-running operations exceed the timeout
    """

    def __init__(
        self,
        message: str = "Request timed out",
        timeout_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(message, code="TIMEOUT")
        self.timeout_seconds = timeout_seconds

    def __str__(self) -> str:
        base = super().__str__()
        if self.timeout_seconds:
            return f"{base} after {self.timeout_seconds}s"
        return base


class WebSocketError(BuilderEngineError):
    """
    Raised when a WebSocket error occurs.

    This can occur when:
    - Connection fails to establish
    - Connection is unexpectedly closed
    - Protocol errors occur
    """

    def __init__(
        self,
        message: str = "WebSocket error",
        close_code: Optional[int] = None,
    ) -> None:
        super().__init__(message, code="WEBSOCKET_ERROR")
        self.close_code = close_code

    def __str__(self) -> str:
        base = super().__str__()
        if self.close_code:
            return f"{base} (Close code: {self.close_code})"
        return base


class CallError(BuilderEngineError):
    """
    Raised when a call operation fails.

    Attributes:
        call_id: ID of the failed call
        call_status: Final status of the call
    """

    def __init__(
        self,
        message: str = "Call operation failed",
        call_id: Optional[str] = None,
        call_status: Optional[str] = None,
    ) -> None:
        super().__init__(message, code="CALL_ERROR")
        self.call_id = call_id
        self.call_status = call_status


class TranscriptionError(BuilderEngineError):
    """
    Raised when transcription fails.

    This can occur when:
    - Audio quality is too poor
    - Unsupported audio format
    - Transcription service is unavailable
    """

    def __init__(self, message: str = "Transcription failed") -> None:
        super().__init__(message, code="TRANSCRIPTION_ERROR")


class VoiceSynthesisError(BuilderEngineError):
    """
    Raised when voice synthesis fails.

    This can occur when:
    - Invalid voice ID
    - Text is too long
    - Voice synthesis service is unavailable
    """

    def __init__(self, message: str = "Voice synthesis failed") -> None:
        super().__init__(message, code="VOICE_SYNTHESIS_ERROR")


class CampaignError(BuilderEngineError):
    """
    Raised when a campaign operation fails.

    Attributes:
        campaign_id: ID of the failed campaign
    """

    def __init__(
        self,
        message: str = "Campaign operation failed",
        campaign_id: Optional[str] = None,
    ) -> None:
        super().__init__(message, code="CAMPAIGN_ERROR")
        self.campaign_id = campaign_id


class WorkflowError(BuilderEngineError):
    """
    Raised when a workflow execution fails.

    Attributes:
        workflow_id: ID of the failed workflow
        action_id: ID of the action that failed
    """

    def __init__(
        self,
        message: str = "Workflow execution failed",
        workflow_id: Optional[str] = None,
        action_id: Optional[str] = None,
    ) -> None:
        super().__init__(message, code="WORKFLOW_ERROR")
        self.workflow_id = workflow_id
        self.action_id = action_id


class BillingError(BuilderEngineError):
    """
    Raised when a billing operation fails.

    This can occur when:
    - Payment method is invalid
    - Insufficient funds
    - Subscription limit reached
    """

    def __init__(self, message: str = "Billing operation failed") -> None:
        super().__init__(message, code="BILLING_ERROR")


class QuotaExceededError(BuilderEngineError):
    """
    Raised when a usage quota is exceeded.

    Attributes:
        quota_type: Type of quota exceeded (e.g., "calls", "minutes")
        current_usage: Current usage amount
        quota_limit: Maximum allowed amount
    """

    def __init__(
        self,
        message: str = "Quota exceeded",
        quota_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        quota_limit: Optional[float] = None,
    ) -> None:
        super().__init__(message, code="QUOTA_EXCEEDED")
        self.quota_type = quota_type
        self.current_usage = current_usage
        self.quota_limit = quota_limit

    def __str__(self) -> str:
        if all([self.quota_type, self.current_usage, self.quota_limit]):
            return (
                f"{self.quota_type} quota exceeded: "
                f"{self.current_usage}/{self.quota_limit}"
            )
        return super().__str__()
