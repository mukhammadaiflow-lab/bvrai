package builderengine

import (
	"context"
	"fmt"
)

// OrganizationsService handles organization-related API calls
type OrganizationsService struct {
	client *Client
}

// OrganizationUpdateParams represents parameters for updating an organization
type OrganizationUpdateParams struct {
	Name         *string                `json:"name,omitempty"`
	BillingEmail *string                `json:"billing_email,omitempty"`
	Settings     map[string]interface{} `json:"settings,omitempty"`
}

// GetCurrent returns the current organization
func (s *OrganizationsService) GetCurrent(ctx context.Context) (*Organization, error) {
	var result Organization
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/organizations/current",
	}, &result)
	return &result, err
}

// Get returns an organization by ID
func (s *OrganizationsService) Get(ctx context.Context, orgID string) (*Organization, error) {
	var result Organization
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/organizations/%s", orgID),
	}, &result)
	return &result, err
}

// Update updates an organization
func (s *OrganizationsService) Update(ctx context.Context, orgID string, params OrganizationUpdateParams) (*Organization, error) {
	var result Organization
	err := s.client.Request(ctx, RequestOptions{
		Method: "PATCH",
		Path:   fmt.Sprintf("/api/v1/organizations/%s", orgID),
		Body:   params,
	}, &result)
	return &result, err
}

// ListMembers returns members of an organization
func (s *OrganizationsService) ListMembers(ctx context.Context, orgID string, params ListParams) (*PaginatedResponse[User], error) {
	var result PaginatedResponse[User]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/organizations/%s/members", orgID),
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// InviteMember invites a new member to an organization
func (s *OrganizationsService) InviteMember(ctx context.Context, orgID string, email string, role string) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/organizations/%s/invite", orgID),
		Body:   map[string]string{"email": email, "role": role},
	}, &result)
	return result, err
}

// RemoveMember removes a member from an organization
func (s *OrganizationsService) RemoveMember(ctx context.Context, orgID string, userID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/organizations/%s/members/%s", orgID, userID),
	}, nil)
}

// UpdateMemberRole updates a member's role
func (s *OrganizationsService) UpdateMemberRole(ctx context.Context, orgID string, userID string, role string) (*User, error) {
	var result User
	err := s.client.Request(ctx, RequestOptions{
		Method: "PATCH",
		Path:   fmt.Sprintf("/api/v1/organizations/%s/members/%s", orgID, userID),
		Body:   map[string]string{"role": role},
	}, &result)
	return &result, err
}

// ListPendingInvites returns pending invitations
func (s *OrganizationsService) ListPendingInvites(ctx context.Context, orgID string) ([]map[string]interface{}, error) {
	var result struct {
		Data []map[string]interface{} `json:"data"`
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/organizations/%s/invites", orgID),
	}, &result)
	return result.Data, err
}

// CancelInvite cancels a pending invitation
func (s *OrganizationsService) CancelInvite(ctx context.Context, orgID string, inviteID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/organizations/%s/invites/%s", orgID, inviteID),
	}, nil)
}

// GetUsage returns organization usage
func (s *OrganizationsService) GetUsage(ctx context.Context, orgID string, period string) (*Usage, error) {
	var result Usage
	params := map[string]string{}
	if period != "" {
		params["period"] = period
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/organizations/%s/usage", orgID),
		Params: params,
	}, &result)
	return &result, err
}
