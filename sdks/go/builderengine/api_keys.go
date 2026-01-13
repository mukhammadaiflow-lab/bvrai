package builderengine

import (
	"context"
	"fmt"
)

// APIKeysService handles API key-related API calls
type APIKeysService struct {
	client *Client
}

// APIKeyCreateParams represents parameters for creating an API key
type APIKeyCreateParams struct {
	Name        string   `json:"name"`
	Permissions []string `json:"permissions,omitempty"`
	RateLimit   int      `json:"rate_limit,omitempty"`
	ExpiresAt   string   `json:"expires_at,omitempty"`
}

// List returns a list of API keys
func (s *APIKeysService) List(ctx context.Context, params ListParams) (*PaginatedResponse[APIKey], error) {
	var result PaginatedResponse[APIKey]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/api-keys",
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// Get returns an API key by ID
func (s *APIKeysService) Get(ctx context.Context, apiKeyID string) (*APIKey, error) {
	var result APIKey
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/api-keys/%s", apiKeyID),
	}, &result)
	return &result, err
}

// Create creates a new API key
func (s *APIKeysService) Create(ctx context.Context, params APIKeyCreateParams) (*APIKey, error) {
	var result APIKey
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/api-keys",
		Body:   params,
	}, &result)
	return &result, err
}

// Delete deletes an API key
func (s *APIKeysService) Delete(ctx context.Context, apiKeyID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/api-keys/%s", apiKeyID),
	}, nil)
}

// Regenerate regenerates an API key
func (s *APIKeysService) Regenerate(ctx context.Context, apiKeyID string) (*APIKey, error) {
	var result APIKey
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/api-keys/%s/regenerate", apiKeyID),
	}, &result)
	return &result, err
}

// GetUsage returns usage for an API key
func (s *APIKeysService) GetUsage(ctx context.Context, apiKeyID string, period string) (map[string]interface{}, error) {
	var result map[string]interface{}
	params := map[string]string{}
	if period != "" {
		params["period"] = period
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/api-keys/%s/usage", apiKeyID),
		Params: params,
	}, &result)
	return result, err
}
