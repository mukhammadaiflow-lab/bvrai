package builderengine

import (
	"context"
	"fmt"
)

// AgentsService handles agent-related API calls
type AgentsService struct {
	client *Client
}

// AgentCreateParams represents parameters for creating an agent
type AgentCreateParams struct {
	Name             string            `json:"name"`
	Voice            string            `json:"voice"`
	SystemPrompt     string            `json:"system_prompt"`
	Description      string            `json:"description,omitempty"`
	Model            string            `json:"model,omitempty"`
	FirstMessage     string            `json:"first_message,omitempty"`
	Language         string            `json:"language,omitempty"`
	Temperature      float64           `json:"temperature,omitempty"`
	MaxTokens        int               `json:"max_tokens,omitempty"`
	InterruptionMode string            `json:"interruption_mode,omitempty"`
	EndCallPhrases   []string          `json:"end_call_phrases,omitempty"`
	Functions        []AgentFunction   `json:"functions,omitempty"`
	KnowledgeBaseIDs []string          `json:"knowledge_base_ids,omitempty"`
	VoiceSettings    *VoiceSettings    `json:"voice_settings,omitempty"`
	Metadata         map[string]string `json:"metadata,omitempty"`
}

// AgentUpdateParams represents parameters for updating an agent
type AgentUpdateParams struct {
	Name             *string           `json:"name,omitempty"`
	Voice            *string           `json:"voice,omitempty"`
	SystemPrompt     *string           `json:"system_prompt,omitempty"`
	Description      *string           `json:"description,omitempty"`
	Model            *string           `json:"model,omitempty"`
	FirstMessage     *string           `json:"first_message,omitempty"`
	Language         *string           `json:"language,omitempty"`
	Temperature      *float64          `json:"temperature,omitempty"`
	MaxTokens        *int              `json:"max_tokens,omitempty"`
	InterruptionMode *string           `json:"interruption_mode,omitempty"`
	EndCallPhrases   []string          `json:"end_call_phrases,omitempty"`
	Functions        []AgentFunction   `json:"functions,omitempty"`
	KnowledgeBaseIDs []string          `json:"knowledge_base_ids,omitempty"`
	VoiceSettings    *VoiceSettings    `json:"voice_settings,omitempty"`
	Metadata         map[string]string `json:"metadata,omitempty"`
	Status           *AgentStatus      `json:"status,omitempty"`
}

// AgentListParams represents parameters for listing agents
type AgentListParams struct {
	ListParams
	Status AgentStatus `json:"status,omitempty"`
	Search string      `json:"search,omitempty"`
}

// ToMap converts AgentListParams to a map
func (p AgentListParams) ToMap() map[string]string {
	m := p.ListParams.ToMap()
	if p.Status != "" {
		m["status"] = string(p.Status)
	}
	if p.Search != "" {
		m["search"] = p.Search
	}
	return m
}

// List returns a list of agents
func (s *AgentsService) List(ctx context.Context, params AgentListParams) (*PaginatedResponse[Agent], error) {
	var result PaginatedResponse[Agent]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/agents",
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// Get returns an agent by ID
func (s *AgentsService) Get(ctx context.Context, agentID string) (*Agent, error) {
	var result Agent
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/agents/%s", agentID),
	}, &result)
	return &result, err
}

// Create creates a new agent
func (s *AgentsService) Create(ctx context.Context, params AgentCreateParams) (*Agent, error) {
	var result Agent
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/agents",
		Body:   params,
	}, &result)
	return &result, err
}

// Update updates an agent
func (s *AgentsService) Update(ctx context.Context, agentID string, params AgentUpdateParams) (*Agent, error) {
	var result Agent
	err := s.client.Request(ctx, RequestOptions{
		Method: "PATCH",
		Path:   fmt.Sprintf("/api/v1/agents/%s", agentID),
		Body:   params,
	}, &result)
	return &result, err
}

// Delete deletes an agent
func (s *AgentsService) Delete(ctx context.Context, agentID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/agents/%s", agentID),
	}, nil)
}

// Duplicate duplicates an agent
func (s *AgentsService) Duplicate(ctx context.Context, agentID string, name string) (*Agent, error) {
	var result Agent
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/agents/%s/duplicate", agentID),
		Body:   map[string]string{"name": name},
	}, &result)
	return &result, err
}

// GetAnalytics returns analytics for an agent
func (s *AgentsService) GetAnalytics(ctx context.Context, agentID string, period string) (*Analytics, error) {
	var result Analytics
	params := map[string]string{}
	if period != "" {
		params["period"] = period
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/agents/%s/analytics", agentID),
		Params: params,
	}, &result)
	return &result, err
}

// TestCall initiates a test call to an agent
func (s *AgentsService) TestCall(ctx context.Context, agentID string, phoneNumber string) (*Call, error) {
	var result Call
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/agents/%s/test-call", agentID),
		Body:   map[string]string{"phone_number": phoneNumber},
	}, &result)
	return &result, err
}
