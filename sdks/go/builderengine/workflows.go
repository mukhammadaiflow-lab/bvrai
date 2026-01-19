package builderengine

import (
	"context"
	"fmt"
)

// WorkflowsService handles workflow-related API calls
type WorkflowsService struct {
	client *Client
}

// WorkflowCreateParams represents parameters for creating a workflow
type WorkflowCreateParams struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Trigger     WorkflowTrigger `json:"trigger"`
	Steps       []WorkflowStep  `json:"steps"`
	Enabled     bool            `json:"enabled,omitempty"`
}

// WorkflowUpdateParams represents parameters for updating a workflow
type WorkflowUpdateParams struct {
	Name        *string          `json:"name,omitempty"`
	Description *string          `json:"description,omitempty"`
	Trigger     *WorkflowTrigger `json:"trigger,omitempty"`
	Steps       []WorkflowStep   `json:"steps,omitempty"`
	Enabled     *bool            `json:"enabled,omitempty"`
}

// WorkflowExecution represents a workflow execution
type WorkflowExecution struct {
	ID         string                 `json:"id"`
	WorkflowID string                 `json:"workflow_id"`
	Status     string                 `json:"status"`
	TriggerData map[string]interface{} `json:"trigger_data"`
	StepResults []StepResult          `json:"step_results,omitempty"`
	Error       string                `json:"error,omitempty"`
	StartedAt   string                `json:"started_at"`
	CompletedAt string                `json:"completed_at,omitempty"`
}

// StepResult represents the result of a workflow step
type StepResult struct {
	StepID   string                 `json:"step_id"`
	Status   string                 `json:"status"`
	Output   map[string]interface{} `json:"output,omitempty"`
	Error    string                 `json:"error,omitempty"`
	Duration int                    `json:"duration"`
}

// List returns a list of workflows
func (s *WorkflowsService) List(ctx context.Context, params ListParams) (*PaginatedResponse[Workflow], error) {
	var result PaginatedResponse[Workflow]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/workflows",
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// Get returns a workflow by ID
func (s *WorkflowsService) Get(ctx context.Context, workflowID string) (*Workflow, error) {
	var result Workflow
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/workflows/%s", workflowID),
	}, &result)
	return &result, err
}

// Create creates a new workflow
func (s *WorkflowsService) Create(ctx context.Context, params WorkflowCreateParams) (*Workflow, error) {
	var result Workflow
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/workflows",
		Body:   params,
	}, &result)
	return &result, err
}

// Update updates a workflow
func (s *WorkflowsService) Update(ctx context.Context, workflowID string, params WorkflowUpdateParams) (*Workflow, error) {
	var result Workflow
	err := s.client.Request(ctx, RequestOptions{
		Method: "PATCH",
		Path:   fmt.Sprintf("/api/v1/workflows/%s", workflowID),
		Body:   params,
	}, &result)
	return &result, err
}

// Delete deletes a workflow
func (s *WorkflowsService) Delete(ctx context.Context, workflowID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/workflows/%s", workflowID),
	}, nil)
}

// Enable enables a workflow
func (s *WorkflowsService) Enable(ctx context.Context, workflowID string) (*Workflow, error) {
	enabled := true
	return s.Update(ctx, workflowID, WorkflowUpdateParams{Enabled: &enabled})
}

// Disable disables a workflow
func (s *WorkflowsService) Disable(ctx context.Context, workflowID string) (*Workflow, error) {
	enabled := false
	return s.Update(ctx, workflowID, WorkflowUpdateParams{Enabled: &enabled})
}

// Trigger manually triggers a workflow
func (s *WorkflowsService) Trigger(ctx context.Context, workflowID string, data map[string]interface{}) (*WorkflowExecution, error) {
	var result WorkflowExecution
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/workflows/%s/trigger", workflowID),
		Body:   data,
	}, &result)
	return &result, err
}

// ListExecutions returns executions for a workflow
func (s *WorkflowsService) ListExecutions(ctx context.Context, workflowID string, params ListParams) (*PaginatedResponse[WorkflowExecution], error) {
	var result PaginatedResponse[WorkflowExecution]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/workflows/%s/executions", workflowID),
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// GetExecution returns a workflow execution by ID
func (s *WorkflowsService) GetExecution(ctx context.Context, workflowID string, executionID string) (*WorkflowExecution, error) {
	var result WorkflowExecution
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/workflows/%s/executions/%s", workflowID, executionID),
	}, &result)
	return &result, err
}
