/**
 * Builder Engine TypeScript SDK - Exceptions
 *
 * Custom error classes for API errors.
 */

/**
 * Base error class for all Builder Engine SDK errors.
 */
export class BuilderEngineError extends Error {
  /** Error code */
  public readonly code?: string;
  /** Additional error details */
  public readonly details?: Record<string, any>;

  constructor(message: string, code?: string, details?: Record<string, any>) {
    super(message);
    this.name = 'BuilderEngineError';
    this.code = code;
    this.details = details;

    // Maintains proper stack trace for where error was thrown
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }
}

/**
 * Raised when authentication fails.
 *
 * This can occur when:
 * - API key is invalid or expired
 * - API key lacks required permissions
 * - Authentication token is malformed
 */
export class AuthenticationError extends BuilderEngineError {
  constructor(message: string = 'Authentication failed') {
    super(message, 'AUTHENTICATION_ERROR');
    this.name = 'AuthenticationError';
  }
}

/**
 * Raised when the API rate limit is exceeded.
 *
 * @property retryAfter - Number of seconds to wait before retrying
 */
export class RateLimitError extends BuilderEngineError {
  /** Number of seconds to wait before retrying */
  public readonly retryAfter?: number;

  constructor(message: string = 'Rate limit exceeded', retryAfter?: number) {
    super(message, 'RATE_LIMIT_ERROR');
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

/**
 * Raised when request validation fails.
 *
 * This occurs when the request body or parameters
 * don't meet the API's validation requirements.
 */
export class ValidationError extends BuilderEngineError {
  /** Field-specific error messages */
  public readonly fieldErrors?: Record<string, string>;

  constructor(message: string = 'Validation failed', fieldErrors?: Record<string, string>) {
    super(message, 'VALIDATION_ERROR', fieldErrors);
    this.name = 'ValidationError';
    this.fieldErrors = fieldErrors;
  }
}

/**
 * Raised when a requested resource is not found.
 */
export class NotFoundError extends BuilderEngineError {
  /** Type of resource that wasn't found */
  public readonly resourceType?: string;
  /** ID of the resource that wasn't found */
  public readonly resourceId?: string;

  constructor(
    message: string = 'Resource not found',
    resourceType?: string,
    resourceId?: string
  ) {
    super(message, 'NOT_FOUND');
    this.name = 'NotFoundError';
    this.resourceType = resourceType;
    this.resourceId = resourceId;
  }
}

/**
 * Raised when there's a resource conflict.
 */
export class ConflictError extends BuilderEngineError {
  constructor(message: string = 'Resource conflict') {
    super(message, 'CONFLICT');
    this.name = 'ConflictError';
  }
}

/**
 * Raised when a server error occurs.
 */
export class ServerError extends BuilderEngineError {
  /** Request ID for debugging */
  public readonly requestId?: string;

  constructor(message: string = 'Server error', requestId?: string) {
    super(message, 'SERVER_ERROR');
    this.name = 'ServerError';
    this.requestId = requestId;
  }
}

/**
 * Raised when a request times out.
 */
export class TimeoutError extends BuilderEngineError {
  /** Timeout duration in milliseconds */
  public readonly timeoutMs?: number;

  constructor(message: string = 'Request timed out', timeoutMs?: number) {
    super(message, 'TIMEOUT');
    this.name = 'TimeoutError';
    this.timeoutMs = timeoutMs;
  }
}

/**
 * Raised when a WebSocket error occurs.
 */
export class WebSocketError extends BuilderEngineError {
  /** WebSocket close code */
  public readonly closeCode?: number;

  constructor(message: string = 'WebSocket error', closeCode?: number) {
    super(message, 'WEBSOCKET_ERROR');
    this.name = 'WebSocketError';
    this.closeCode = closeCode;
  }
}

/**
 * Raised when a call operation fails.
 */
export class CallError extends BuilderEngineError {
  /** ID of the failed call */
  public readonly callId?: string;
  /** Final status of the call */
  public readonly callStatus?: string;

  constructor(
    message: string = 'Call operation failed',
    callId?: string,
    callStatus?: string
  ) {
    super(message, 'CALL_ERROR');
    this.name = 'CallError';
    this.callId = callId;
    this.callStatus = callStatus;
  }
}

/**
 * Raised when a usage quota is exceeded.
 */
export class QuotaExceededError extends BuilderEngineError {
  /** Type of quota exceeded */
  public readonly quotaType?: string;
  /** Current usage amount */
  public readonly currentUsage?: number;
  /** Maximum allowed amount */
  public readonly quotaLimit?: number;

  constructor(
    message: string = 'Quota exceeded',
    quotaType?: string,
    currentUsage?: number,
    quotaLimit?: number
  ) {
    super(message, 'QUOTA_EXCEEDED');
    this.name = 'QuotaExceededError';
    this.quotaType = quotaType;
    this.currentUsage = currentUsage;
    this.quotaLimit = quotaLimit;
  }
}
