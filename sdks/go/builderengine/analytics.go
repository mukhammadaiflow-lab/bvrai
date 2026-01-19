package builderengine

import (
	"context"
	"fmt"
)

// AnalyticsService handles analytics-related API calls
type AnalyticsService struct {
	client *Client
}

// AnalyticsParams represents parameters for analytics queries
type AnalyticsParams struct {
	Period    string `json:"period,omitempty"`
	StartDate string `json:"start_date,omitempty"`
	EndDate   string `json:"end_date,omitempty"`
	AgentID   string `json:"agent_id,omitempty"`
	GroupBy   string `json:"group_by,omitempty"`
}

// ToMap converts AnalyticsParams to a map
func (p AnalyticsParams) ToMap() map[string]string {
	m := make(map[string]string)
	if p.Period != "" {
		m["period"] = p.Period
	}
	if p.StartDate != "" {
		m["start_date"] = p.StartDate
	}
	if p.EndDate != "" {
		m["end_date"] = p.EndDate
	}
	if p.AgentID != "" {
		m["agent_id"] = p.AgentID
	}
	if p.GroupBy != "" {
		m["group_by"] = p.GroupBy
	}
	return m
}

// TimeSeriesData represents time series analytics data
type TimeSeriesData struct {
	Timestamp string  `json:"timestamp"`
	Value     float64 `json:"value"`
}

// OverviewMetrics represents overview analytics
type OverviewMetrics struct {
	TotalCalls        int     `json:"total_calls"`
	TotalMinutes      float64 `json:"total_minutes"`
	AverageDuration   float64 `json:"average_duration"`
	SuccessRate       float64 `json:"success_rate"`
	InboundCalls      int     `json:"inbound_calls"`
	OutboundCalls     int     `json:"outbound_calls"`
	UniqueCallers     int     `json:"unique_callers"`
	TotalCost         float64 `json:"total_cost"`
	SentimentScore    float64 `json:"sentiment_score"`
	CallsPerDay       []TimeSeriesData `json:"calls_per_day,omitempty"`
	MinutesPerDay     []TimeSeriesData `json:"minutes_per_day,omitempty"`
}

// GetOverview returns overview analytics
func (s *AnalyticsService) GetOverview(ctx context.Context, params AnalyticsParams) (*OverviewMetrics, error) {
	var result OverviewMetrics
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/analytics/overview",
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// GetUsage returns usage metrics
func (s *AnalyticsService) GetUsage(ctx context.Context, period string) (*Usage, error) {
	var result Usage
	params := map[string]string{}
	if period != "" {
		params["period"] = period
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/analytics/usage",
		Params: params,
	}, &result)
	return &result, err
}

// GetCostBreakdown returns cost breakdown
func (s *AnalyticsService) GetCostBreakdown(ctx context.Context, params AnalyticsParams) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/analytics/costs",
		Params: params.ToMap(),
	}, &result)
	return result, err
}

// GetCallMetrics returns call metrics
func (s *AnalyticsService) GetCallMetrics(ctx context.Context, params AnalyticsParams) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/analytics/calls",
		Params: params.ToMap(),
	}, &result)
	return result, err
}

// GetAgentPerformance returns agent performance metrics
func (s *AnalyticsService) GetAgentPerformance(ctx context.Context, agentID string, params AnalyticsParams) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/analytics/agents/%s", agentID),
		Params: params.ToMap(),
	}, &result)
	return result, err
}

// ExportParams represents parameters for exporting reports
type ExportParams struct {
	ReportType string `json:"report_type"`
	Period     string `json:"period,omitempty"`
	StartDate  string `json:"start_date,omitempty"`
	EndDate    string `json:"end_date,omitempty"`
	Format     string `json:"format,omitempty"`
}

// ExportReport exports an analytics report
func (s *AnalyticsService) Export(ctx context.Context, params ExportParams) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/analytics/export",
		Body:   params,
	}, &result)
	return result, err
}

// GetTimeSeries returns time series data
func (s *AnalyticsService) GetTimeSeries(ctx context.Context, metric string, params AnalyticsParams) ([]TimeSeriesData, error) {
	var result struct {
		Data []TimeSeriesData `json:"data"`
	}
	qParams := params.ToMap()
	qParams["metric"] = metric
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/analytics/timeseries",
		Params: qParams,
	}, &result)
	return result.Data, err
}

// GetTopAgents returns top performing agents
func (s *AnalyticsService) GetTopAgents(ctx context.Context, params AnalyticsParams) ([]map[string]interface{}, error) {
	var result struct {
		Data []map[string]interface{} `json:"data"`
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/analytics/top-agents",
		Params: params.ToMap(),
	}, &result)
	return result.Data, err
}
