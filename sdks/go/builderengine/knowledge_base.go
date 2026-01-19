package builderengine

import (
	"context"
	"fmt"
)

// KnowledgeBaseService handles knowledge base-related API calls
type KnowledgeBaseService struct {
	client *Client
}

// KnowledgeBaseCreateParams represents parameters for creating a knowledge base
type KnowledgeBaseCreateParams struct {
	Name           string `json:"name"`
	Description    string `json:"description,omitempty"`
	EmbeddingModel string `json:"embedding_model,omitempty"`
}

// KnowledgeBaseUpdateParams represents parameters for updating a knowledge base
type KnowledgeBaseUpdateParams struct {
	Name        *string `json:"name,omitempty"`
	Description *string `json:"description,omitempty"`
}

// DocumentCreateParams represents parameters for adding a document
type DocumentCreateParams struct {
	Title       string                 `json:"title"`
	Content     string                 `json:"content,omitempty"`
	ContentType string                 `json:"content_type,omitempty"`
	SourceURL   string                 `json:"source_url,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// SearchParams represents parameters for searching a knowledge base
type SearchParams struct {
	Query    string `json:"query"`
	Limit    int    `json:"limit,omitempty"`
	MinScore float64 `json:"min_score,omitempty"`
}

// List returns a list of knowledge bases
func (s *KnowledgeBaseService) List(ctx context.Context, params ListParams) (*PaginatedResponse[KnowledgeBase], error) {
	var result PaginatedResponse[KnowledgeBase]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/knowledge-bases",
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// Get returns a knowledge base by ID
func (s *KnowledgeBaseService) Get(ctx context.Context, kbID string) (*KnowledgeBase, error) {
	var result KnowledgeBase
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/knowledge-bases/%s", kbID),
	}, &result)
	return &result, err
}

// Create creates a new knowledge base
func (s *KnowledgeBaseService) Create(ctx context.Context, params KnowledgeBaseCreateParams) (*KnowledgeBase, error) {
	var result KnowledgeBase
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/knowledge-bases",
		Body:   params,
	}, &result)
	return &result, err
}

// Update updates a knowledge base
func (s *KnowledgeBaseService) Update(ctx context.Context, kbID string, params KnowledgeBaseUpdateParams) (*KnowledgeBase, error) {
	var result KnowledgeBase
	err := s.client.Request(ctx, RequestOptions{
		Method: "PATCH",
		Path:   fmt.Sprintf("/api/v1/knowledge-bases/%s", kbID),
		Body:   params,
	}, &result)
	return &result, err
}

// Delete deletes a knowledge base
func (s *KnowledgeBaseService) Delete(ctx context.Context, kbID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/knowledge-bases/%s", kbID),
	}, nil)
}

// ListDocuments returns documents in a knowledge base
func (s *KnowledgeBaseService) ListDocuments(ctx context.Context, kbID string, params ListParams) (*PaginatedResponse[Document], error) {
	var result PaginatedResponse[Document]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/knowledge-bases/%s/documents", kbID),
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// GetDocument returns a document by ID
func (s *KnowledgeBaseService) GetDocument(ctx context.Context, kbID string, documentID string) (*Document, error) {
	var result Document
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/knowledge-bases/%s/documents/%s", kbID, documentID),
	}, &result)
	return &result, err
}

// AddDocument adds a document to a knowledge base
func (s *KnowledgeBaseService) AddDocument(ctx context.Context, kbID string, params DocumentCreateParams) (*Document, error) {
	var result Document
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/knowledge-bases/%s/documents", kbID),
		Body:   params,
	}, &result)
	return &result, err
}

// DeleteDocument deletes a document from a knowledge base
func (s *KnowledgeBaseService) DeleteDocument(ctx context.Context, kbID string, documentID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/knowledge-bases/%s/documents/%s", kbID, documentID),
	}, nil)
}

// Search searches a knowledge base
func (s *KnowledgeBaseService) Search(ctx context.Context, kbID string, params SearchParams) ([]SearchResult, error) {
	var result struct {
		Results []SearchResult `json:"results"`
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/knowledge-bases/%s/search", kbID),
		Body:   params,
	}, &result)
	return result.Results, err
}

// Reindex triggers a reindex of a knowledge base
func (s *KnowledgeBaseService) Reindex(ctx context.Context, kbID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/knowledge-bases/%s/reindex", kbID),
	}, nil)
}
