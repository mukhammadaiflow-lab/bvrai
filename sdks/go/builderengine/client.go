// Package builderengine provides a Go SDK for the Builder Engine AI Voice Agent Platform.
package builderengine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const (
	// DefaultBaseURL is the default API base URL
	DefaultBaseURL = "https://api.builderengine.io"
	// DefaultTimeout is the default request timeout
	DefaultTimeout = 30 * time.Second
	// DefaultMaxRetries is the default number of retries
	DefaultMaxRetries = 3
	// Version is the SDK version
	Version = "1.0.0"
)

// Client is the main Builder Engine API client
type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	maxRetries int

	// Resources
	Agents        *AgentsService
	Calls         *CallsService
	Conversations *ConversationsService
	PhoneNumbers  *PhoneNumbersService
	Voices        *VoicesService
	Webhooks      *WebhooksService
	KnowledgeBase *KnowledgeBaseService
	Workflows     *WorkflowsService
	Campaigns     *CampaignsService
	Analytics     *AnalyticsService
	Organizations *OrganizationsService
	Users         *UsersService
	APIKeys       *APIKeysService
	Billing       *BillingService
}

// ClientOption is a function that configures the client
type ClientOption func(*Client)

// WithBaseURL sets a custom base URL
func WithBaseURL(url string) ClientOption {
	return func(c *Client) {
		c.baseURL = strings.TrimSuffix(url, "/")
	}
}

// WithTimeout sets a custom timeout
func WithTimeout(timeout time.Duration) ClientOption {
	return func(c *Client) {
		c.httpClient.Timeout = timeout
	}
}

// WithHTTPClient sets a custom HTTP client
func WithHTTPClient(httpClient *http.Client) ClientOption {
	return func(c *Client) {
		c.httpClient = httpClient
	}
}

// WithMaxRetries sets the maximum number of retries
func WithMaxRetries(retries int) ClientOption {
	return func(c *Client) {
		c.maxRetries = retries
	}
}

// NewClient creates a new Builder Engine API client
func NewClient(apiKey string, opts ...ClientOption) (*Client, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	c := &Client{
		apiKey:     apiKey,
		baseURL:    DefaultBaseURL,
		httpClient: &http.Client{Timeout: DefaultTimeout},
		maxRetries: DefaultMaxRetries,
	}

	for _, opt := range opts {
		opt(c)
	}

	// Initialize resources
	c.Agents = &AgentsService{client: c}
	c.Calls = &CallsService{client: c}
	c.Conversations = &ConversationsService{client: c}
	c.PhoneNumbers = &PhoneNumbersService{client: c}
	c.Voices = &VoicesService{client: c}
	c.Webhooks = &WebhooksService{client: c}
	c.KnowledgeBase = &KnowledgeBaseService{client: c}
	c.Workflows = &WorkflowsService{client: c}
	c.Campaigns = &CampaignsService{client: c}
	c.Analytics = &AnalyticsService{client: c}
	c.Organizations = &OrganizationsService{client: c}
	c.Users = &UsersService{client: c}
	c.APIKeys = &APIKeysService{client: c}
	c.Billing = &BillingService{client: c}

	return c, nil
}

// RequestOptions contains options for an API request
type RequestOptions struct {
	Method  string
	Path    string
	Body    interface{}
	Params  map[string]string
	Headers map[string]string
}

// doRequest performs an HTTP request with retries
func (c *Client) doRequest(ctx context.Context, opts RequestOptions) (*http.Response, error) {
	var lastErr error

	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}

		resp, err := c.executeRequest(ctx, opts)
		if err != nil {
			lastErr = err
			continue
		}

		// Check if we should retry
		if resp.StatusCode >= 500 || resp.StatusCode == 429 {
			lastErr = &APIError{
				StatusCode: resp.StatusCode,
				Message:    fmt.Sprintf("Server error: %d", resp.StatusCode),
			}
			resp.Body.Close()
			continue
		}

		return resp, nil
	}

	return nil, lastErr
}

// executeRequest performs a single HTTP request
func (c *Client) executeRequest(ctx context.Context, opts RequestOptions) (*http.Response, error) {
	// Build URL
	reqURL := c.baseURL + opts.Path
	if len(opts.Params) > 0 {
		params := url.Values{}
		for k, v := range opts.Params {
			params.Set(k, v)
		}
		reqURL += "?" + params.Encode()
	}

	// Build body
	var body io.Reader
	if opts.Body != nil {
		jsonBody, err := json.Marshal(opts.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		body = bytes.NewReader(jsonBody)
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, opts.Method, reqURL, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "builderengine-go/"+Version)

	for k, v := range opts.Headers {
		req.Header.Set(k, v)
	}

	return c.httpClient.Do(req)
}

// Request performs an API request and decodes the response
func (c *Client) Request(ctx context.Context, opts RequestOptions, result interface{}) error {
	resp, err := c.doRequest(ctx, opts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Read body
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response body: %w", err)
	}

	// Check for errors
	if resp.StatusCode >= 400 {
		return parseError(resp.StatusCode, bodyBytes, resp.Header)
	}

	// Decode response
	if result != nil && len(bodyBytes) > 0 {
		if err := json.Unmarshal(bodyBytes, result); err != nil {
			return fmt.Errorf("failed to decode response: %w", err)
		}
	}

	return nil
}

// parseError parses an error response
func parseError(statusCode int, body []byte, headers http.Header) error {
	var errResp struct {
		Error struct {
			Message string                 `json:"message"`
			Code    string                 `json:"code"`
			Details map[string]interface{} `json:"details"`
		} `json:"error"`
	}

	if err := json.Unmarshal(body, &errResp); err != nil {
		return &APIError{
			StatusCode: statusCode,
			Message:    string(body),
		}
	}

	switch statusCode {
	case 401:
		return &AuthenticationError{
			APIError: APIError{
				StatusCode: statusCode,
				Message:    errResp.Error.Message,
				Code:       errResp.Error.Code,
			},
		}
	case 403:
		return &PermissionError{
			APIError: APIError{
				StatusCode: statusCode,
				Message:    errResp.Error.Message,
				Code:       errResp.Error.Code,
			},
		}
	case 404:
		return &NotFoundError{
			APIError: APIError{
				StatusCode: statusCode,
				Message:    errResp.Error.Message,
				Code:       errResp.Error.Code,
			},
		}
	case 422:
		return &ValidationError{
			APIError: APIError{
				StatusCode: statusCode,
				Message:    errResp.Error.Message,
				Code:       errResp.Error.Code,
			},
			Errors: errResp.Error.Details,
		}
	case 429:
		retryAfter := 60
		if ra := headers.Get("Retry-After"); ra != "" {
			if val, err := strconv.Atoi(ra); err == nil {
				retryAfter = val
			}
		}
		return &RateLimitError{
			APIError: APIError{
				StatusCode: statusCode,
				Message:    errResp.Error.Message,
				Code:       errResp.Error.Code,
			},
			RetryAfter: retryAfter,
		}
	case 402:
		return &InsufficientCreditsError{
			APIError: APIError{
				StatusCode: statusCode,
				Message:    errResp.Error.Message,
				Code:       errResp.Error.Code,
			},
		}
	default:
		return &APIError{
			StatusCode: statusCode,
			Message:    errResp.Error.Message,
			Code:       errResp.Error.Code,
		}
	}
}

// Streaming returns a new streaming client
func (c *Client) Streaming() *StreamingClient {
	wsURL := strings.Replace(c.baseURL, "https://", "wss://", 1)
	wsURL = strings.Replace(wsURL, "http://", "ws://", 1)
	return NewStreamingClient(c.apiKey, wsURL)
}
