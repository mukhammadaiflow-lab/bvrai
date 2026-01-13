package builderengine

import (
	"context"
	"fmt"
)

// CampaignsService handles campaign-related API calls
type CampaignsService struct {
	client *Client
}

// CampaignCreateParams represents parameters for creating a campaign
type CampaignCreateParams struct {
	Name        string            `json:"name"`
	AgentID     string            `json:"agent_id"`
	Description string            `json:"description,omitempty"`
	FromNumber  string            `json:"from_number,omitempty"`
	Contacts    []ContactInput    `json:"contacts,omitempty"`
	Schedule    *CampaignSchedule `json:"schedule,omitempty"`
	Settings    *CampaignSettings `json:"settings,omitempty"`
}

// ContactInput represents a contact to add to a campaign
type ContactInput struct {
	PhoneNumber string                 `json:"phone_number"`
	Name        string                 `json:"name,omitempty"`
	Variables   map[string]interface{} `json:"variables,omitempty"`
}

// CampaignUpdateParams represents parameters for updating a campaign
type CampaignUpdateParams struct {
	Name        *string           `json:"name,omitempty"`
	Description *string           `json:"description,omitempty"`
	FromNumber  *string           `json:"from_number,omitempty"`
	Schedule    *CampaignSchedule `json:"schedule,omitempty"`
	Settings    *CampaignSettings `json:"settings,omitempty"`
}

// CampaignListParams represents parameters for listing campaigns
type CampaignListParams struct {
	ListParams
	AgentID string         `json:"agent_id,omitempty"`
	Status  CampaignStatus `json:"status,omitempty"`
}

// ToMap converts CampaignListParams to a map
func (p CampaignListParams) ToMap() map[string]string {
	m := p.ListParams.ToMap()
	if p.AgentID != "" {
		m["agent_id"] = p.AgentID
	}
	if p.Status != "" {
		m["status"] = string(p.Status)
	}
	return m
}

// List returns a list of campaigns
func (s *CampaignsService) List(ctx context.Context, params CampaignListParams) (*PaginatedResponse[Campaign], error) {
	var result PaginatedResponse[Campaign]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/campaigns",
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// Get returns a campaign by ID
func (s *CampaignsService) Get(ctx context.Context, campaignID string) (*Campaign, error) {
	var result Campaign
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/campaigns/%s", campaignID),
	}, &result)
	return &result, err
}

// Create creates a new campaign
func (s *CampaignsService) Create(ctx context.Context, params CampaignCreateParams) (*Campaign, error) {
	var result Campaign
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/campaigns",
		Body:   params,
	}, &result)
	return &result, err
}

// Update updates a campaign
func (s *CampaignsService) Update(ctx context.Context, campaignID string, params CampaignUpdateParams) (*Campaign, error) {
	var result Campaign
	err := s.client.Request(ctx, RequestOptions{
		Method: "PATCH",
		Path:   fmt.Sprintf("/api/v1/campaigns/%s", campaignID),
		Body:   params,
	}, &result)
	return &result, err
}

// Delete deletes a campaign
func (s *CampaignsService) Delete(ctx context.Context, campaignID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/campaigns/%s", campaignID),
	}, nil)
}

// Start starts a campaign
func (s *CampaignsService) Start(ctx context.Context, campaignID string) (*Campaign, error) {
	var result Campaign
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/campaigns/%s/start", campaignID),
	}, &result)
	return &result, err
}

// Pause pauses a campaign
func (s *CampaignsService) Pause(ctx context.Context, campaignID string) (*Campaign, error) {
	var result Campaign
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/campaigns/%s/pause", campaignID),
	}, &result)
	return &result, err
}

// Resume resumes a paused campaign
func (s *CampaignsService) Resume(ctx context.Context, campaignID string) (*Campaign, error) {
	var result Campaign
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/campaigns/%s/resume", campaignID),
	}, &result)
	return &result, err
}

// Cancel cancels a campaign
func (s *CampaignsService) Cancel(ctx context.Context, campaignID string) (*Campaign, error) {
	var result Campaign
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/campaigns/%s/cancel", campaignID),
	}, &result)
	return &result, err
}

// GetProgress returns progress for a campaign
func (s *CampaignsService) GetProgress(ctx context.Context, campaignID string) (*CampaignProgress, error) {
	var result CampaignProgress
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/campaigns/%s/progress", campaignID),
	}, &result)
	return &result, err
}

// ListContacts returns contacts for a campaign
func (s *CampaignsService) ListContacts(ctx context.Context, campaignID string, params ListParams) (*PaginatedResponse[CampaignContact], error) {
	var result PaginatedResponse[CampaignContact]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/campaigns/%s/contacts", campaignID),
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// AddContacts adds contacts to a campaign
func (s *CampaignsService) AddContacts(ctx context.Context, campaignID string, contacts []ContactInput) ([]CampaignContact, error) {
	var result struct {
		Data []CampaignContact `json:"data"`
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/campaigns/%s/contacts", campaignID),
		Body:   map[string][]ContactInput{"contacts": contacts},
	}, &result)
	return result.Data, err
}

// RemoveContact removes a contact from a campaign
func (s *CampaignsService) RemoveContact(ctx context.Context, campaignID string, contactID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/campaigns/%s/contacts/%s", campaignID, contactID),
	}, nil)
}

// GetAnalytics returns analytics for a campaign
func (s *CampaignsService) GetAnalytics(ctx context.Context, campaignID string) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/campaigns/%s/analytics", campaignID),
	}, &result)
	return result, err
}

// ExportResults exports campaign results
func (s *CampaignsService) ExportResults(ctx context.Context, campaignID string, format string) (map[string]interface{}, error) {
	var result map[string]interface{}
	params := map[string]string{}
	if format != "" {
		params["format"] = format
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/campaigns/%s/export", campaignID),
		Params: params,
	}, &result)
	return result, err
}
