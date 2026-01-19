package builderengine

import (
	"context"
	"fmt"
)

// ConversationsService handles conversation-related API calls
type ConversationsService struct {
	client *Client
}

// ConversationListParams represents parameters for listing conversations
type ConversationListParams struct {
	ListParams
	AgentID string `json:"agent_id,omitempty"`
	Status  string `json:"status,omitempty"`
	Search  string `json:"search,omitempty"`
}

// ToMap converts ConversationListParams to a map
func (p ConversationListParams) ToMap() map[string]string {
	m := p.ListParams.ToMap()
	if p.AgentID != "" {
		m["agent_id"] = p.AgentID
	}
	if p.Status != "" {
		m["status"] = p.Status
	}
	if p.Search != "" {
		m["search"] = p.Search
	}
	return m
}

// List returns a list of conversations
func (s *ConversationsService) List(ctx context.Context, params ConversationListParams) (*PaginatedResponse[Conversation], error) {
	var result PaginatedResponse[Conversation]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/conversations",
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// Get returns a conversation by ID
func (s *ConversationsService) Get(ctx context.Context, conversationID string) (*Conversation, error) {
	var result Conversation
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/conversations/%s", conversationID),
	}, &result)
	return &result, err
}

// ListMessages returns messages for a conversation
func (s *ConversationsService) ListMessages(ctx context.Context, conversationID string, params ListParams) (*PaginatedResponse[Message], error) {
	var result PaginatedResponse[Message]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/conversations/%s/messages", conversationID),
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// AddMessageParams represents parameters for adding a message
type AddMessageParams struct {
	Role     string                 `json:"role"`
	Content  string                 `json:"content"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// AddMessage adds a message to a conversation
func (s *ConversationsService) AddMessage(ctx context.Context, conversationID string, params AddMessageParams) (*Message, error) {
	var result Message
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/conversations/%s/messages", conversationID),
		Body:   params,
	}, &result)
	return &result, err
}

// Delete deletes a conversation
func (s *ConversationsService) Delete(ctx context.Context, conversationID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/conversations/%s", conversationID),
	}, nil)
}

// GetSummary returns a summary of a conversation
func (s *ConversationsService) GetSummary(ctx context.Context, conversationID string) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/conversations/%s/summary", conversationID),
	}, &result)
	return result, err
}

// Export exports a conversation
func (s *ConversationsService) Export(ctx context.Context, conversationID string, format string) (map[string]interface{}, error) {
	var result map[string]interface{}
	params := map[string]string{}
	if format != "" {
		params["format"] = format
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/conversations/%s/export", conversationID),
		Params: params,
	}, &result)
	return result, err
}
