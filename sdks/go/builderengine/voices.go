package builderengine

import (
	"context"
	"fmt"
)

// VoicesService handles voice-related API calls
type VoicesService struct {
	client *Client
}

// VoiceListParams represents parameters for listing voices
type VoiceListParams struct {
	ListParams
	Provider string `json:"provider,omitempty"`
	Language string `json:"language,omitempty"`
	Gender   string `json:"gender,omitempty"`
}

// ToMap converts VoiceListParams to a map
func (p VoiceListParams) ToMap() map[string]string {
	m := p.ListParams.ToMap()
	if p.Provider != "" {
		m["provider"] = p.Provider
	}
	if p.Language != "" {
		m["language"] = p.Language
	}
	if p.Gender != "" {
		m["gender"] = p.Gender
	}
	return m
}

// VoiceCloneParams represents parameters for cloning a voice
type VoiceCloneParams struct {
	Name        string   `json:"name"`
	Description string   `json:"description,omitempty"`
	AudioFiles  [][]byte `json:"-"` // Audio files to upload
}

// List returns a list of available voices
func (s *VoicesService) List(ctx context.Context, params VoiceListParams) (*PaginatedResponse[Voice], error) {
	var result PaginatedResponse[Voice]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/voices",
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// Get returns a voice by ID
func (s *VoicesService) Get(ctx context.Context, voiceID string) (*Voice, error) {
	var result Voice
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/voices/%s", voiceID),
	}, &result)
	return &result, err
}

// GetPreview returns a preview URL for a voice
func (s *VoicesService) GetPreview(ctx context.Context, voiceID string) (string, error) {
	var result struct {
		URL string `json:"url"`
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/voices/%s/preview", voiceID),
	}, &result)
	return result.URL, err
}

// ListProviders returns available voice providers
func (s *VoicesService) ListProviders(ctx context.Context) ([]map[string]interface{}, error) {
	var result struct {
		Data []map[string]interface{} `json:"data"`
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/voices/providers",
	}, &result)
	return result.Data, err
}

// ListLanguages returns available languages
func (s *VoicesService) ListLanguages(ctx context.Context) ([]map[string]interface{}, error) {
	var result struct {
		Data []map[string]interface{} `json:"data"`
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/voices/languages",
	}, &result)
	return result.Data, err
}

// Clone creates a voice clone (requires audio samples)
func (s *VoicesService) Clone(ctx context.Context, params VoiceCloneParams) (*Voice, error) {
	var result Voice
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/voices/clone",
		Body:   map[string]string{"name": params.Name, "description": params.Description},
	}, &result)
	return &result, err
}

// DeleteClone deletes a cloned voice
func (s *VoicesService) DeleteClone(ctx context.Context, voiceID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/voices/%s", voiceID),
	}, nil)
}
