package builderengine

import "fmt"

// APIError represents a generic API error
type APIError struct {
	StatusCode int
	Message    string
	Code       string
}

func (e *APIError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("API error %d (%s): %s", e.StatusCode, e.Code, e.Message)
	}
	return fmt.Sprintf("API error %d: %s", e.StatusCode, e.Message)
}

// AuthenticationError represents a 401 error
type AuthenticationError struct {
	APIError
}

func (e *AuthenticationError) Error() string {
	return fmt.Sprintf("Authentication error: %s", e.Message)
}

// PermissionError represents a 403 error
type PermissionError struct {
	APIError
}

func (e *PermissionError) Error() string {
	return fmt.Sprintf("Permission denied: %s", e.Message)
}

// NotFoundError represents a 404 error
type NotFoundError struct {
	APIError
}

func (e *NotFoundError) Error() string {
	return fmt.Sprintf("Not found: %s", e.Message)
}

// ValidationError represents a 422 error
type ValidationError struct {
	APIError
	Errors map[string]interface{}
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("Validation error: %s", e.Message)
}

// RateLimitError represents a 429 error
type RateLimitError struct {
	APIError
	RetryAfter int
}

func (e *RateLimitError) Error() string {
	return fmt.Sprintf("Rate limit exceeded. Retry after %d seconds", e.RetryAfter)
}

// InsufficientCreditsError represents a 402 error
type InsufficientCreditsError struct {
	APIError
}

func (e *InsufficientCreditsError) Error() string {
	return fmt.Sprintf("Insufficient credits: %s", e.Message)
}

// WebSocketError represents a WebSocket error
type WebSocketError struct {
	Message string
	Err     error
}

func (e *WebSocketError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("WebSocket error: %s: %v", e.Message, e.Err)
	}
	return fmt.Sprintf("WebSocket error: %s", e.Message)
}

func (e *WebSocketError) Unwrap() error {
	return e.Err
}

// IsAuthenticationError checks if an error is an authentication error
func IsAuthenticationError(err error) bool {
	_, ok := err.(*AuthenticationError)
	return ok
}

// IsPermissionError checks if an error is a permission error
func IsPermissionError(err error) bool {
	_, ok := err.(*PermissionError)
	return ok
}

// IsNotFoundError checks if an error is a not found error
func IsNotFoundError(err error) bool {
	_, ok := err.(*NotFoundError)
	return ok
}

// IsValidationError checks if an error is a validation error
func IsValidationError(err error) bool {
	_, ok := err.(*ValidationError)
	return ok
}

// IsRateLimitError checks if an error is a rate limit error
func IsRateLimitError(err error) bool {
	_, ok := err.(*RateLimitError)
	return ok
}

// IsInsufficientCreditsError checks if an error is an insufficient credits error
func IsInsufficientCreditsError(err error) bool {
	_, ok := err.(*InsufficientCreditsError)
	return ok
}
