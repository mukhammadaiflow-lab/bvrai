package builderengine

import (
	"context"
	"fmt"
)

// WebhooksService handles webhook-related API calls
type WebhooksService struct {
	client *Client
}

// WebhookCreateParams represents parameters for creating a webhook
type WebhookCreateParams struct {
	URL         string            `json:"url"`
	Events      []WebhookEvent    `json:"events"`
	Secret      string            `json:"secret,omitempty"`
	Description string            `json:"description,omitempty"`
	Headers     map[string]string `json:"headers,omitempty"`
	RetryPolicy *RetryPolicy      `json:"retry_policy,omitempty"`
}

// WebhookUpdateParams represents parameters for updating a webhook
type WebhookUpdateParams struct {
	URL         *string           `json:"url,omitempty"`
	Events      []WebhookEvent    `json:"events,omitempty"`
	Secret      *string           `json:"secret,omitempty"`
	Description *string           `json:"description,omitempty"`
	Headers     map[string]string `json:"headers,omitempty"`
	RetryPolicy *RetryPolicy      `json:"retry_policy,omitempty"`
	Enabled     *bool             `json:"enabled,omitempty"`
}

// WebhookDelivery represents a webhook delivery attempt
type WebhookDelivery struct {
	ID           string                 `json:"id"`
	WebhookID    string                 `json:"webhook_id"`
	Event        WebhookEvent           `json:"event"`
	Payload      map[string]interface{} `json:"payload"`
	StatusCode   int                    `json:"status_code"`
	ResponseBody string                 `json:"response_body,omitempty"`
	Success      bool                   `json:"success"`
	Attempts     int                    `json:"attempts"`
	CreatedAt    string                 `json:"created_at"`
}

// List returns a list of webhooks
func (s *WebhooksService) List(ctx context.Context, params ListParams) (*PaginatedResponse[Webhook], error) {
	var result PaginatedResponse[Webhook]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/webhooks",
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// Get returns a webhook by ID
func (s *WebhooksService) Get(ctx context.Context, webhookID string) (*Webhook, error) {
	var result Webhook
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/webhooks/%s", webhookID),
	}, &result)
	return &result, err
}

// Create creates a new webhook
func (s *WebhooksService) Create(ctx context.Context, params WebhookCreateParams) (*Webhook, error) {
	var result Webhook
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/webhooks",
		Body:   params,
	}, &result)
	return &result, err
}

// Update updates a webhook
func (s *WebhooksService) Update(ctx context.Context, webhookID string, params WebhookUpdateParams) (*Webhook, error) {
	var result Webhook
	err := s.client.Request(ctx, RequestOptions{
		Method: "PATCH",
		Path:   fmt.Sprintf("/api/v1/webhooks/%s", webhookID),
		Body:   params,
	}, &result)
	return &result, err
}

// Delete deletes a webhook
func (s *WebhooksService) Delete(ctx context.Context, webhookID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/webhooks/%s", webhookID),
	}, nil)
}

// Test sends a test event to a webhook
func (s *WebhooksService) Test(ctx context.Context, webhookID string) (*WebhookDelivery, error) {
	var result WebhookDelivery
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/webhooks/%s/test", webhookID),
	}, &result)
	return &result, err
}

// ListDeliveries returns delivery attempts for a webhook
func (s *WebhooksService) ListDeliveries(ctx context.Context, webhookID string, params ListParams) (*PaginatedResponse[WebhookDelivery], error) {
	var result PaginatedResponse[WebhookDelivery]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/webhooks/%s/deliveries", webhookID),
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// RetryDelivery retries a failed webhook delivery
func (s *WebhooksService) RetryDelivery(ctx context.Context, webhookID string, deliveryID string) (*WebhookDelivery, error) {
	var result WebhookDelivery
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/webhooks/%s/deliveries/%s/retry", webhookID, deliveryID),
	}, &result)
	return &result, err
}

// RotateSecret rotates the webhook signing secret
func (s *WebhooksService) RotateSecret(ctx context.Context, webhookID string) (*Webhook, error) {
	var result Webhook
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/webhooks/%s/rotate-secret", webhookID),
	}, &result)
	return &result, err
}
