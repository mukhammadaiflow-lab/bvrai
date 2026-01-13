package builderengine

import (
	"context"
	"fmt"
)

// CallsService handles call-related API calls
type CallsService struct {
	client *Client
}

// CallCreateParams represents parameters for creating a call
type CallCreateParams struct {
	AgentID       string                 `json:"agent_id"`
	ToNumber      string                 `json:"to_number"`
	FromNumber    string                 `json:"from_number,omitempty"`
	PhoneNumberID string                 `json:"phone_number_id,omitempty"`
	Variables     map[string]interface{} `json:"variables,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	WebhookURL    string                 `json:"webhook_url,omitempty"`
	Record        bool                   `json:"record,omitempty"`
	MaxDuration   int                    `json:"max_duration,omitempty"`
}

// CallListParams represents parameters for listing calls
type CallListParams struct {
	ListParams
	AgentID   string     `json:"agent_id,omitempty"`
	Status    CallStatus `json:"status,omitempty"`
	Direction CallDirection `json:"direction,omitempty"`
	FromDate  string     `json:"from_date,omitempty"`
	ToDate    string     `json:"to_date,omitempty"`
	Search    string     `json:"search,omitempty"`
}

// ToMap converts CallListParams to a map
func (p CallListParams) ToMap() map[string]string {
	m := p.ListParams.ToMap()
	if p.AgentID != "" {
		m["agent_id"] = p.AgentID
	}
	if p.Status != "" {
		m["status"] = string(p.Status)
	}
	if p.Direction != "" {
		m["direction"] = string(p.Direction)
	}
	if p.FromDate != "" {
		m["from_date"] = p.FromDate
	}
	if p.ToDate != "" {
		m["to_date"] = p.ToDate
	}
	if p.Search != "" {
		m["search"] = p.Search
	}
	return m
}

// List returns a list of calls
func (s *CallsService) List(ctx context.Context, params CallListParams) (*PaginatedResponse[Call], error) {
	var result PaginatedResponse[Call]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/calls",
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// Get returns a call by ID
func (s *CallsService) Get(ctx context.Context, callID string) (*Call, error) {
	var result Call
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/calls/%s", callID),
	}, &result)
	return &result, err
}

// Create initiates a new outbound call
func (s *CallsService) Create(ctx context.Context, params CallCreateParams) (*Call, error) {
	var result Call
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/calls",
		Body:   params,
	}, &result)
	return &result, err
}

// End ends an active call
func (s *CallsService) End(ctx context.Context, callID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/calls/%s/end", callID),
	}, nil)
}

// Transfer transfers a call to another number
func (s *CallsService) Transfer(ctx context.Context, callID string, toNumber string, announce string) error {
	body := map[string]string{"to_number": toNumber}
	if announce != "" {
		body["announce"] = announce
	}
	return s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/calls/%s/transfer", callID),
		Body:   body,
	}, nil)
}

// GetTranscript returns the transcript for a call
func (s *CallsService) GetTranscript(ctx context.Context, callID string) (*Transcript, error) {
	var result Transcript
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/calls/%s/transcript", callID),
	}, &result)
	return &result, err
}

// GetRecording returns the recording for a call
func (s *CallsService) GetRecording(ctx context.Context, callID string) (*Recording, error) {
	var result Recording
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/calls/%s/recording", callID),
	}, &result)
	return &result, err
}

// SendDTMF sends DTMF tones to an active call
func (s *CallsService) SendDTMF(ctx context.Context, callID string, digits string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/calls/%s/dtmf", callID),
		Body:   map[string]string{"digits": digits},
	}, nil)
}

// InjectText injects text for the agent to speak
func (s *CallsService) InjectText(ctx context.Context, callID string, text string, interrupt bool) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/calls/%s/inject", callID),
		Body:   map[string]interface{}{"text": text, "interrupt": interrupt},
	}, nil)
}

// Mute mutes or unmutes a call
func (s *CallsService) Mute(ctx context.Context, callID string, muted bool) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/calls/%s/mute", callID),
		Body:   map[string]bool{"muted": muted},
	}, nil)
}

// GetAnalytics returns analytics for a specific call
func (s *CallsService) GetAnalytics(ctx context.Context, callID string) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/calls/%s/analytics", callID),
	}, &result)
	return result, err
}
