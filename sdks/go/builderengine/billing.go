package builderengine

import (
	"context"
	"fmt"
)

// BillingService handles billing-related API calls
type BillingService struct {
	client *Client
}

// PaymentMethod represents a payment method
type PaymentMethod struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Last4     string `json:"last4,omitempty"`
	Brand     string `json:"brand,omitempty"`
	ExpiryMonth int  `json:"expiry_month,omitempty"`
	ExpiryYear  int  `json:"expiry_year,omitempty"`
	IsDefault bool   `json:"is_default"`
}

// BillingPortalSession represents a billing portal session
type BillingPortalSession struct {
	URL string `json:"url"`
}

// GetSubscription returns the current subscription
func (s *BillingService) GetSubscription(ctx context.Context) (*Subscription, error) {
	var result Subscription
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/billing/subscription",
	}, &result)
	return &result, err
}

// ListPlans returns available plans
func (s *BillingService) ListPlans(ctx context.Context) ([]map[string]interface{}, error) {
	var result struct {
		Data []map[string]interface{} `json:"data"`
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/billing/plans",
	}, &result)
	return result.Data, err
}

// ChangePlan changes the subscription plan
func (s *BillingService) ChangePlan(ctx context.Context, planID string) (*Subscription, error) {
	var result Subscription
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/billing/subscription/change",
		Body:   map[string]string{"plan_id": planID},
	}, &result)
	return &result, err
}

// CancelSubscription cancels the subscription
func (s *BillingService) CancelSubscription(ctx context.Context, cancelAtPeriodEnd bool) (*Subscription, error) {
	var result Subscription
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/billing/subscription/cancel",
		Body:   map[string]bool{"cancel_at_period_end": cancelAtPeriodEnd},
	}, &result)
	return &result, err
}

// ResumeSubscription resumes a canceled subscription
func (s *BillingService) ResumeSubscription(ctx context.Context) (*Subscription, error) {
	var result Subscription
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/billing/subscription/resume",
	}, &result)
	return &result, err
}

// ListInvoices returns a list of invoices
func (s *BillingService) ListInvoices(ctx context.Context, params ListParams) (*PaginatedResponse[Invoice], error) {
	var result PaginatedResponse[Invoice]
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/billing/invoices",
		Params: params.ToMap(),
	}, &result)
	return &result, err
}

// GetInvoice returns an invoice by ID
func (s *BillingService) GetInvoice(ctx context.Context, invoiceID string) (*Invoice, error) {
	var result Invoice
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/billing/invoices/%s", invoiceID),
	}, &result)
	return &result, err
}

// DownloadInvoice returns a download URL for an invoice
func (s *BillingService) DownloadInvoice(ctx context.Context, invoiceID string) (string, error) {
	var result struct {
		URL string `json:"url"`
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   fmt.Sprintf("/api/v1/billing/invoices/%s/download", invoiceID),
	}, &result)
	return result.URL, err
}

// ListPaymentMethods returns payment methods
func (s *BillingService) ListPaymentMethods(ctx context.Context) ([]PaymentMethod, error) {
	var result struct {
		Data []PaymentMethod `json:"data"`
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/billing/payment-methods",
	}, &result)
	return result.Data, err
}

// AddPaymentMethod adds a new payment method
func (s *BillingService) AddPaymentMethod(ctx context.Context, setupIntentID string) (*PaymentMethod, error) {
	var result PaymentMethod
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/billing/payment-methods",
		Body:   map[string]string{"setup_intent_id": setupIntentID},
	}, &result)
	return &result, err
}

// SetDefaultPaymentMethod sets the default payment method
func (s *BillingService) SetDefaultPaymentMethod(ctx context.Context, paymentMethodID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   fmt.Sprintf("/api/v1/billing/payment-methods/%s/default", paymentMethodID),
	}, nil)
}

// DeletePaymentMethod deletes a payment method
func (s *BillingService) DeletePaymentMethod(ctx context.Context, paymentMethodID string) error {
	return s.client.Request(ctx, RequestOptions{
		Method: "DELETE",
		Path:   fmt.Sprintf("/api/v1/billing/payment-methods/%s", paymentMethodID),
	}, nil)
}

// GetBalance returns the current credit balance
func (s *BillingService) GetBalance(ctx context.Context) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/billing/balance",
	}, &result)
	return result, err
}

// AddCredits adds credits to the account
func (s *BillingService) AddCredits(ctx context.Context, amount float64) (map[string]interface{}, error) {
	var result map[string]interface{}
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/billing/credits",
		Body:   map[string]float64{"amount": amount},
	}, &result)
	return result, err
}

// CreatePortalSession creates a billing portal session
func (s *BillingService) CreatePortalSession(ctx context.Context) (*BillingPortalSession, error) {
	var result BillingPortalSession
	err := s.client.Request(ctx, RequestOptions{
		Method: "POST",
		Path:   "/api/v1/billing/portal-session",
	}, &result)
	return &result, err
}

// GetUsageSummary returns usage summary for billing
func (s *BillingService) GetUsageSummary(ctx context.Context, period string) (map[string]interface{}, error) {
	var result map[string]interface{}
	params := map[string]string{}
	if period != "" {
		params["period"] = period
	}
	err := s.client.Request(ctx, RequestOptions{
		Method: "GET",
		Path:   "/api/v1/billing/usage",
		Params: params,
	}, &result)
	return result, err
}
