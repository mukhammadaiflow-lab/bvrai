package builderengine

import "context"

// UsersService handles user-related API calls
type UsersService struct {
	client *Client
}

// UserUpdateParams represents parameters for updating a user profile
type UserUpdateParams struct {
	FirstName *string `json:"first_name,omitempty"`
	LastName  *string `json:"last_name,omitempty"`
}

// PasswordChangeParams represents parameters for changing password
type PasswordChangeParams struct {
	CurrentPassword string `json:"current_password"`
	NewPassword     string `json:"new_password"`
}

// GetMe returns the current user
func (s *UsersService) GetMe(ctx context.Context) (*User, error) {
	var result User
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/users/me",
	}, &result)
	return &result, err
}

// UpdateProfile updates the current user's profile
func (s *UsersService) UpdateProfile(ctx context.Context, params UserUpdateParams) (*User, error) {
	var result User
	err := s.client.Request(ctx, RequestOptions{
		Method: "PATCH",
		Path:   "/api/v1/users/me/profile",
		Body:   params,
	}, &result)
	return &result, err
}

// ChangePassword changes the current user's password
func (s *UsersService) ChangePassword(ctx context.Context, params PasswordChangeParams) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/users/me/password",
		Body:   params,
	}, nil)
}

// GetPreferences returns user preferences
func (s *UsersService) GetPreferences(ctx context.Context) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/users/me/preferences",
	}, &result)
	return result, err
}

// UpdatePreferences updates user preferences
func (s *UsersService) UpdatePreferences(ctx context.Context, preferences map[string]interface{}) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "PATCH",
		Path:   "/api/v1/users/me/preferences",
		Body:   preferences,
	}, &result)
	return result, err
}

// GetNotificationSettings returns notification settings
func (s *UsersService) GetNotificationSettings(ctx context.Context) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/users/me/notifications",
	}, &result)
	return result, err
}

// UpdateNotificationSettings updates notification settings
func (s *UsersService) UpdateNotificationSettings(ctx context.Context, settings map[string]interface{}) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "PATCH",
		Path:   "/api/v1/users/me/notifications",
		Body:   settings,
	}, &result)
	return result, err
}
