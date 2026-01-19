package builderengine

import (
	"context"
	"fmt"
)

// PhoneNumbersService handles phone number-related API calls
type PhoneNumbersService struct {
	client *Client
}

// PhoneNumberListParams represents parameters for listing phone numbers
type PhoneNumberListParams struct {
	ListParams
	Country string `json:"country,omitempty"`
	AgentID string `json:"agent_id,omitempty"`
	Status  string `json:"status,omitempty"`
}

// ToMap converts PhoneNumberListParams to a map
func (p PhoneNumberListParams) ToMap() map[string]string {
	m := p.ListParams.ToMap()
	if p.Country != "" {
		m["country"] = p.Country
	}
	if p.AgentID != "" {
		m["agent_id"] = p.AgentID
	}
	if p.Status != "" {
		m["status"] = p.Status
	}
	return m
}

// AvailableNumbersParams represents parameters for searching available numbers
type AvailableNumbersParams struct {
	Country    string `json:"country"`
	AreaCode   string `json:"area_code,omitempty"`
	Contains   string `json:"contains,omitempty"`
	Locality   string `json:"locality,omitempty"`
	VoiceOnly  bool   `json:"voice_only,omitempty"`
	Limit      int    `json:"limit,omitempty"`
}

// ToMap converts AvailableNumbersParams to a map
func (p AvailableNumbersParams) ToMap() map[string]string {
	m := make(map[string]string)
	if p.Country != "" {
		m["country"] = p.Country
	}
	if p.AreaCode != "" {
		m["area_code"] = p.AreaCode
	}
	if p.Contains != "" {
		m["contains"] = p.Contains
	}
	if p.Locality != "" {
		m["locality"] = p.Locality
	}
	if p.VoiceOnly {
		m["voice_only"] = "true"
	}
	if p.Limit > 0 {
		m["limit"] = fmt.Sprintf("%d", p.Limit)
	}
	return m
}

// PurchaseParams represents parameters for purchasing a number
type PurchaseParams struct {
	PhoneNumber  string `json:"phone_number"`
	FriendlyName string `json:"friendly_name,omitempty"`
	AgentID      string `json:"agent_id,omitempty"`
}

// UpdateParams represents parameters for updating a phone number
type PhoneNumberUpdateParams struct {
	FriendlyName     *string `json:"friendly_name,omitempty"`
	AgentID          *string `json:"agent_id,omitempty"`
	VoicemailEnabled *bool   `json:"voicemail_enabled,omitempty"`
	RecordingEnabled *bool   `json:"recording_enabled,omitempty"`
}

// List returns a list of phone numbers
func (s *PhoneNumbersService) List(ctx context.Context, params PhoneNumberListParams) (*PaginatedResponse[PhoneNumber], error) {
	var result PaginatedResponse[PhoneNumber]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/phone-numbers",
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// Get returns a phone number by ID
func (s *PhoneNumbersService) Get(ctx context.Context, phoneNumberID string) (*PhoneNumber, error) {
	var result PhoneNumber
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/phone-numbers/%s", phoneNumberID),
	}, &result)
	return &result, err
}

// ListAvailable returns available phone numbers for purchase
func (s *PhoneNumbersService) ListAvailable(ctx context.Context, params AvailableNumbersParams) ([]AvailableNumber, error) {
	var result struct {
		Data []AvailableNumber `json:"data"`
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/phone-numbers/available",
		Params: params.ToMap(),
	}, &result)
	return result.Data, err
}

// Purchase purchases a phone number
func (s *PhoneNumbersService) Purchase(ctx context.Context, params PurchaseParams) (*PhoneNumber, error) {
	var result PhoneNumber
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/phone-numbers/purchase",
		Body:   params,
	}, &result)
	return &result, err
}

// Update updates a phone number
func (s *PhoneNumbersService) Update(ctx context.Context, phoneNumberID string, params PhoneNumberUpdateParams) (*PhoneNumber, error) {
	var result PhoneNumber
	err := s.client.Request(ctx, RequestOptions{
		Method: "PATCH",
		Path:   fmt.Sprintf("/api/v1/phone-numbers/%s", phoneNumberID),
		Body:   params,
	}, &result)
	return &result, err
}

// Release releases a phone number
func (s *PhoneNumbersService) Release(ctx context.Context, phoneNumberID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/phone-numbers/%s", phoneNumberID),
	}, nil)
}

// GetUsage returns usage for a phone number
func (s *PhoneNumbersService) GetUsage(ctx context.Context, phoneNumberID string, period string) (map[string]interface{}, error) {
	var result map[string]interface{}
	params := map[string]string{}
	if period != "" {
		params["period"] = period
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/phone-numbers/%s/usage", phoneNumberID),
		Params: params,
	}, &result)
	return result, err
}
