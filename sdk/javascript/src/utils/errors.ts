/**
 * Error classes for Builder Engine SDK
 */

export class BvraiError extends Error {
  public readonly statusCode?: number;
  public readonly errorCode?: string;
  public readonly details?: Record<string, unknown>;

  constructor(
    message: string,
    options?: {
      statusCode?: number;
      errorCode?: string;
      details?: Record<string, unknown>;
    }
  ) {
    super(message);
    this.name = 'BvraiError';
    this.statusCode = options?.statusCode;
    this.errorCode = options?.errorCode;
    this.details = options?.details;

    // Maintains proper stack trace for where error was thrown
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, BvraiError);
    }
  }
}

export class AuthenticationError extends BvraiError {
  constructor(message = 'Authentication failed') {
    super(message, { statusCode: 401, errorCode: 'AUTH_ERROR' });
    this.name = 'AuthenticationError';
  }
}

export class NotFoundError extends BvraiError {
  constructor(message = 'Resource not found') {
    super(message, { statusCode: 404, errorCode: 'NOT_FOUND' });
    this.name = 'NotFoundError';
  }
}

export class RateLimitError extends BvraiError {
  public readonly retryAfter: number;

  constructor(message = 'Rate limit exceeded', retryAfter = 60) {
    super(message, { statusCode: 429, errorCode: 'RATE_LIMIT' });
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

export class ValidationError extends BvraiError {
  public readonly errors: Array<{ field: string; message: string }>;

  constructor(
    message = 'Validation failed',
    errors: Array<{ field: string; message: string }> = []
  ) {
    super(message, { statusCode: 422, errorCode: 'VALIDATION_ERROR' });
    this.name = 'ValidationError';
    this.errors = errors;
  }
}

export class QuotaExceededError extends BvraiError {
  public readonly quotaType?: string;

  constructor(message = 'Quota exceeded', quotaType?: string) {
    super(message, { statusCode: 403, errorCode: 'QUOTA_EXCEEDED' });
    this.name = 'QuotaExceededError';
    this.quotaType = quotaType;
  }
}

export class ConflictError extends BvraiError {
  constructor(message = 'Resource conflict') {
    super(message, { statusCode: 409, errorCode: 'CONFLICT' });
    this.name = 'ConflictError';
  }
}

export class ServerError extends BvraiError {
  constructor(message = 'Server error') {
    super(message, { statusCode: 500, errorCode: 'SERVER_ERROR' });
    this.name = 'ServerError';
  }
}

export class ServiceUnavailableError extends BvraiError {
  public readonly retryAfter: number;

  constructor(message = 'Service temporarily unavailable', retryAfter = 30) {
    super(message, { statusCode: 503, errorCode: 'SERVICE_UNAVAILABLE' });
    this.name = 'ServiceUnavailableError';
    this.retryAfter = retryAfter;
  }
}

export class TimeoutError extends BvraiError {
  constructor(message = 'Request timed out') {
    super(message, { errorCode: 'TIMEOUT' });
    this.name = 'TimeoutError';
  }
}

export class WebSocketError extends BvraiError {
  constructor(message = 'WebSocket error') {
    super(message, { errorCode: 'WEBSOCKET_ERROR' });
    this.name = 'WebSocketError';
  }
}

export class ConnectionError extends BvraiError {
  constructor(message = 'Connection failed') {
    super(message, { errorCode: 'CONNECTION_ERROR' });
    this.name = 'ConnectionError';
  }
}

/**
 * Create appropriate error from HTTP response
 */
export function createErrorFromResponse(
  statusCode: number,
  body: Record<string, unknown>
): BvraiError {
  const message = (body.message as string) || (body.detail as string) || 'Request failed';

  switch (statusCode) {
    case 401:
      return new AuthenticationError(message);
    case 404:
      return new NotFoundError(message);
    case 409:
      return new ConflictError(message);
    case 422:
      return new ValidationError(
        message,
        (body.errors as Array<{ field: string; message: string }>) || []
      );
    case 429:
      return new RateLimitError(message, (body.retryAfter as number) || 60);
    case 403:
      if (body.errorCode === 'QUOTA_EXCEEDED') {
        return new QuotaExceededError(message, body.quotaType as string);
      }
      return new BvraiError(message, { statusCode, errorCode: body.errorCode as string });
    case 500:
      return new ServerError(message);
    case 503:
      return new ServiceUnavailableError(message, (body.retryAfter as number) || 30);
    default:
      return new BvraiError(message, {
        statusCode,
        errorCode: body.errorCode as string,
        details: body,
      });
  }
}
