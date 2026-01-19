package builderengine

import "time"

// AgentStatus represents the status of an agent
type AgentStatus string

const (
	AgentStatusActive   AgentStatus = "active"
	AgentStatusInactive AgentStatus = "inactive"
	AgentStatusDraft    AgentStatus = "draft"
)

// CallStatus represents the status of a call
type CallStatus string

const (
	CallStatusQueued     CallStatus = "queued"
	CallStatusRinging    CallStatus = "ringing"
	CallStatusInProgress CallStatus = "in_progress"
	CallStatusCompleted  CallStatus = "completed"
	CallStatusFailed     CallStatus = "failed"
	CallStatusBusy       CallStatus = "busy"
	CallStatusNoAnswer   CallStatus = "no_answer"
	CallStatusCanceled   CallStatus = "canceled"
)

// CallDirection represents the direction of a call
type CallDirection string

const (
	CallDirectionInbound  CallDirection = "inbound"
	CallDirectionOutbound CallDirection = "outbound"
)

// CampaignStatus represents the status of a campaign
type CampaignStatus string

const (
	CampaignStatusDraft     CampaignStatus = "draft"
	CampaignStatusScheduled CampaignStatus = "scheduled"
	CampaignStatusRunning   CampaignStatus = "running"
	CampaignStatusPaused    CampaignStatus = "paused"
	CampaignStatusCompleted CampaignStatus = "completed"
	CampaignStatusCanceled  CampaignStatus = "canceled"
)

// WebhookEvent represents a webhook event type
type WebhookEvent string

const (
	WebhookEventCallStarted         WebhookEvent = "call.started"
	WebhookEventCallRinging         WebhookEvent = "call.ringing"
	WebhookEventCallAnswered        WebhookEvent = "call.answered"
	WebhookEventCallEnded           WebhookEvent = "call.ended"
	WebhookEventCallFailed          WebhookEvent = "call.failed"
	WebhookEventTranscriptionFinal  WebhookEvent = "transcription.final"
	WebhookEventFunctionCall        WebhookEvent = "function.call"
	WebhookEventRecordingReady      WebhookEvent = "recording.ready"
	WebhookEventCampaignCompleted   WebhookEvent = "campaign.completed"
)

// Agent represents an AI voice agent
type Agent struct {
	ID                string            `json:"id"`
	Name              string            `json:"name"`
	Description       string            `json:"description,omitempty"`
	Voice             string            `json:"voice"`
	VoiceSettings     *VoiceSettings    `json:"voice_settings,omitempty"`
	Model             string            `json:"model"`
	SystemPrompt      string            `json:"system_prompt"`
	FirstMessage      string            `json:"first_message,omitempty"`
	Language          string            `json:"language"`
	Temperature       float64           `json:"temperature"`
	MaxTokens         int               `json:"max_tokens,omitempty"`
	InterruptionMode  string            `json:"interruption_mode,omitempty"`
	EndCallPhrases    []string          `json:"end_call_phrases,omitempty"`
	Functions         []AgentFunction   `json:"functions,omitempty"`
	KnowledgeBaseIDs  []string          `json:"knowledge_base_ids,omitempty"`
	Metadata          map[string]string `json:"metadata,omitempty"`
	Status            AgentStatus       `json:"status"`
	TotalCalls        int               `json:"total_calls"`
	TotalMinutes      float64           `json:"total_minutes"`
	SuccessRate       float64           `json:"success_rate"`
	OrganizationID    string            `json:"organization_id"`
	CreatedAt         time.Time         `json:"created_at"`
	UpdatedAt         time.Time         `json:"updated_at"`
}

// VoiceSettings represents voice configuration
type VoiceSettings struct {
	Speed         float64 `json:"speed,omitempty"`
	Pitch         float64 `json:"pitch,omitempty"`
	Stability     float64 `json:"stability,omitempty"`
	SimilarityBoost float64 `json:"similarity_boost,omitempty"`
}

// AgentFunction represents a callable function for an agent
type AgentFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
	Webhook     *FunctionWebhook       `json:"webhook,omitempty"`
}

// FunctionWebhook represents webhook configuration for a function
type FunctionWebhook struct {
	URL     string            `json:"url"`
	Method  string            `json:"method"`
	Headers map[string]string `json:"headers,omitempty"`
}

// Call represents a phone call
type Call struct {
	ID              string                 `json:"id"`
	AgentID         string                 `json:"agent_id"`
	PhoneNumberID   string                 `json:"phone_number_id,omitempty"`
	ToNumber        string                 `json:"to_number"`
	FromNumber      string                 `json:"from_number"`
	Direction       CallDirection          `json:"direction"`
	Status          CallStatus             `json:"status"`
	StartedAt       *time.Time             `json:"started_at,omitempty"`
	AnsweredAt      *time.Time             `json:"answered_at,omitempty"`
	EndedAt         *time.Time             `json:"ended_at,omitempty"`
	Duration        int                    `json:"duration"`
	RecordingURL    string                 `json:"recording_url,omitempty"`
	TranscriptURL   string                 `json:"transcript_url,omitempty"`
	Cost            float64                `json:"cost"`
	EndReason       string                 `json:"end_reason,omitempty"`
	SentimentScore  float64                `json:"sentiment_score,omitempty"`
	Summary         string                 `json:"summary,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
	OrganizationID  string                 `json:"organization_id"`
	CreatedAt       time.Time              `json:"created_at"`
}

// Transcript represents a call transcript
type Transcript struct {
	CallID   string              `json:"call_id"`
	Messages []TranscriptMessage `json:"messages"`
	Summary  string              `json:"summary,omitempty"`
}

// TranscriptMessage represents a single message in a transcript
type TranscriptMessage struct {
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
	Duration  float64   `json:"duration,omitempty"`
}

// Recording represents a call recording
type Recording struct {
	URL       string    `json:"url"`
	Duration  int       `json:"duration"`
	Size      int64     `json:"size"`
	Format    string    `json:"format"`
	ExpiresAt time.Time `json:"expires_at"`
}

// Conversation represents a conversation/thread
type Conversation struct {
	ID             string                 `json:"id"`
	AgentID        string                 `json:"agent_id"`
	CallIDs        []string               `json:"call_ids"`
	Status         string                 `json:"status"`
	MessageCount   int                    `json:"message_count"`
	LastMessageAt  *time.Time             `json:"last_message_at,omitempty"`
	Summary        string                 `json:"summary,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
	OrganizationID string                 `json:"organization_id"`
	CreatedAt      time.Time              `json:"created_at"`
	UpdatedAt      time.Time              `json:"updated_at"`
}

// Message represents a conversation message
type Message struct {
	ID             string                 `json:"id"`
	ConversationID string                 `json:"conversation_id"`
	CallID         string                 `json:"call_id,omitempty"`
	Role           string                 `json:"role"`
	Content        string                 `json:"content"`
	FunctionCall   *FunctionCallData      `json:"function_call,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt      time.Time              `json:"created_at"`
}

// FunctionCallData represents function call information
type FunctionCallData struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
	Result    interface{}            `json:"result,omitempty"`
}

// PhoneNumber represents a phone number
type PhoneNumber struct {
	ID               string            `json:"id"`
	PhoneNumber      string            `json:"phone_number"`
	FriendlyName     string            `json:"friendly_name,omitempty"`
	Country          string            `json:"country"`
	Region           string            `json:"region,omitempty"`
	Capabilities     NumberCapabilities `json:"capabilities"`
	AgentID          string            `json:"agent_id,omitempty"`
	VoicemailEnabled bool              `json:"voicemail_enabled"`
	RecordingEnabled bool              `json:"recording_enabled"`
	Provider         string            `json:"provider"`
	Status           string            `json:"status"`
	MonthlyPrice     float64           `json:"monthly_price"`
	OrganizationID   string            `json:"organization_id"`
	CreatedAt        time.Time         `json:"created_at"`
}

// NumberCapabilities represents phone number capabilities
type NumberCapabilities struct {
	Voice bool `json:"voice"`
	SMS   bool `json:"sms"`
	MMS   bool `json:"mms"`
}

// AvailableNumber represents an available number for purchase
type AvailableNumber struct {
	PhoneNumber  string             `json:"phone_number"`
	Country      string             `json:"country"`
	Region       string             `json:"region,omitempty"`
	Locality     string             `json:"locality,omitempty"`
	Capabilities NumberCapabilities `json:"capabilities"`
	MonthlyPrice float64            `json:"monthly_price"`
	SetupPrice   float64            `json:"setup_price"`
}

// Voice represents a voice option
type Voice struct {
	ID             string   `json:"id"`
	Name           string   `json:"name"`
	Provider       string   `json:"provider"`
	Language       string   `json:"language"`
	LanguageCode   string   `json:"language_code"`
	Gender         string   `json:"gender"`
	Accent         string   `json:"accent,omitempty"`
	Age            string   `json:"age,omitempty"`
	UseCase        string   `json:"use_case,omitempty"`
	Description    string   `json:"description,omitempty"`
	PreviewURL     string   `json:"preview_url,omitempty"`
	SupportedModels []string `json:"supported_models,omitempty"`
	IsCustom       bool     `json:"is_custom"`
}

// Webhook represents a webhook configuration
type Webhook struct {
	ID             string         `json:"id"`
	URL            string         `json:"url"`
	Events         []WebhookEvent `json:"events"`
	Secret         string         `json:"secret,omitempty"`
	Enabled        bool           `json:"enabled"`
	Description    string         `json:"description,omitempty"`
	Headers        map[string]string `json:"headers,omitempty"`
	RetryPolicy    *RetryPolicy   `json:"retry_policy,omitempty"`
	OrganizationID string         `json:"organization_id"`
	CreatedAt      time.Time      `json:"created_at"`
	UpdatedAt      time.Time      `json:"updated_at"`
}

// RetryPolicy represents webhook retry configuration
type RetryPolicy struct {
	MaxRetries    int `json:"max_retries"`
	RetryInterval int `json:"retry_interval"`
}

// KnowledgeBase represents a knowledge base
type KnowledgeBase struct {
	ID             string    `json:"id"`
	Name           string    `json:"name"`
	Description    string    `json:"description,omitempty"`
	DocumentCount  int       `json:"document_count"`
	TotalChunks    int       `json:"total_chunks"`
	TotalTokens    int       `json:"total_tokens"`
	EmbeddingModel string    `json:"embedding_model"`
	Status         string    `json:"status"`
	OrganizationID string    `json:"organization_id"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

// Document represents a knowledge base document
type Document struct {
	ID              string                 `json:"id"`
	KnowledgeBaseID string                 `json:"knowledge_base_id"`
	Title           string                 `json:"title"`
	Content         string                 `json:"content,omitempty"`
	ContentType     string                 `json:"content_type"`
	SourceURL       string                 `json:"source_url,omitempty"`
	ChunkCount      int                    `json:"chunk_count"`
	TokenCount      int                    `json:"token_count"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
	Status          string                 `json:"status"`
	CreatedAt       time.Time              `json:"created_at"`
}

// SearchResult represents a knowledge base search result
type SearchResult struct {
	DocumentID string  `json:"document_id"`
	Title      string  `json:"title"`
	Content    string  `json:"content"`
	Score      float64 `json:"score"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// Workflow represents a workflow
type Workflow struct {
	ID             string         `json:"id"`
	Name           string         `json:"name"`
	Description    string         `json:"description,omitempty"`
	Trigger        WorkflowTrigger `json:"trigger"`
	Steps          []WorkflowStep  `json:"steps"`
	Enabled        bool           `json:"enabled"`
	OrganizationID string         `json:"organization_id"`
	CreatedAt      time.Time      `json:"created_at"`
	UpdatedAt      time.Time      `json:"updated_at"`
}

// WorkflowTrigger represents what triggers a workflow
type WorkflowTrigger struct {
	Type       string                 `json:"type"`
	Conditions map[string]interface{} `json:"conditions,omitempty"`
}

// WorkflowStep represents a step in a workflow
type WorkflowStep struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Name     string                 `json:"name"`
	Config   map[string]interface{} `json:"config"`
	Next     string                 `json:"next,omitempty"`
	OnError  string                 `json:"on_error,omitempty"`
}

// Campaign represents a call campaign
type Campaign struct {
	ID             string          `json:"id"`
	Name           string          `json:"name"`
	Description    string          `json:"description,omitempty"`
	AgentID        string          `json:"agent_id"`
	FromNumber     string          `json:"from_number,omitempty"`
	ContactCount   int             `json:"contact_count"`
	Schedule       *CampaignSchedule `json:"schedule,omitempty"`
	Settings       *CampaignSettings `json:"settings,omitempty"`
	Status         CampaignStatus  `json:"status"`
	Progress       CampaignProgress `json:"progress"`
	OrganizationID string          `json:"organization_id"`
	CreatedAt      time.Time       `json:"created_at"`
	UpdatedAt      time.Time       `json:"updated_at"`
	StartedAt      *time.Time      `json:"started_at,omitempty"`
	CompletedAt    *time.Time      `json:"completed_at,omitempty"`
}

// CampaignSchedule represents campaign scheduling
type CampaignSchedule struct {
	StartTime     string `json:"start_time,omitempty"`
	EndTime       string `json:"end_time,omitempty"`
	Timezone      string `json:"timezone,omitempty"`
	DaysOfWeek    []int  `json:"days_of_week,omitempty"`
	MaxConcurrent int    `json:"max_concurrent,omitempty"`
}

// CampaignSettings represents campaign settings
type CampaignSettings struct {
	MaxAttempts     int     `json:"max_attempts,omitempty"`
	RetryDelay      int     `json:"retry_delay,omitempty"`
	MaxCallDuration int     `json:"max_call_duration,omitempty"`
	LeaveVoicemail  bool    `json:"leave_voicemail,omitempty"`
	VoicemailMessage string `json:"voicemail_message,omitempty"`
}

// CampaignProgress represents campaign progress
type CampaignProgress struct {
	Total       int     `json:"total"`
	Completed   int     `json:"completed"`
	Successful  int     `json:"successful"`
	Failed      int     `json:"failed"`
	Pending     int     `json:"pending"`
	InProgress  int     `json:"in_progress"`
	SuccessRate float64 `json:"success_rate"`
}

// CampaignContact represents a campaign contact
type CampaignContact struct {
	ID          string                 `json:"id"`
	CampaignID  string                 `json:"campaign_id"`
	PhoneNumber string                 `json:"phone_number"`
	Name        string                 `json:"name,omitempty"`
	Variables   map[string]interface{} `json:"variables,omitempty"`
	Status      string                 `json:"status"`
	CallID      string                 `json:"call_id,omitempty"`
	Attempts    int                    `json:"attempts"`
	LastAttempt *time.Time             `json:"last_attempt,omitempty"`
}

// Analytics represents analytics data
type Analytics struct {
	Period          string  `json:"period"`
	TotalCalls      int     `json:"total_calls"`
	TotalMinutes    float64 `json:"total_minutes"`
	AverageDuration float64 `json:"average_duration"`
	SuccessRate     float64 `json:"success_rate"`
	InboundCalls    int     `json:"inbound_calls"`
	OutboundCalls   int     `json:"outbound_calls"`
	UniqueCallers   int     `json:"unique_callers"`
	TotalCost       float64 `json:"total_cost"`
}

// Usage represents usage metrics
type Usage struct {
	Period         string  `json:"period"`
	Minutes        float64 `json:"minutes"`
	Calls          int     `json:"calls"`
	APIRequests    int     `json:"api_requests"`
	StorageGB      float64 `json:"storage_gb"`
	CostBreakdown  map[string]float64 `json:"cost_breakdown,omitempty"`
}

// Organization represents an organization
type Organization struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	Slug          string                 `json:"slug"`
	Plan          string                 `json:"plan"`
	BillingEmail  string                 `json:"billing_email,omitempty"`
	Settings      map[string]interface{} `json:"settings,omitempty"`
	MemberCount   int                    `json:"member_count"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`
}

// User represents a user
type User struct {
	ID             string    `json:"id"`
	Email          string    `json:"email"`
	FirstName      string    `json:"first_name,omitempty"`
	LastName       string    `json:"last_name,omitempty"`
	Role           string    `json:"role"`
	OrganizationID string    `json:"organization_id"`
	EmailVerified  bool      `json:"email_verified"`
	LastLoginAt    *time.Time `json:"last_login_at,omitempty"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

// APIKey represents an API key
type APIKey struct {
	ID          string     `json:"id"`
	Name        string     `json:"name"`
	KeyPrefix   string     `json:"key_prefix"`
	Key         string     `json:"key,omitempty"` // Only returned on creation
	Permissions []string   `json:"permissions,omitempty"`
	RateLimit   int        `json:"rate_limit,omitempty"`
	LastUsedAt  *time.Time `json:"last_used_at,omitempty"`
	ExpiresAt   *time.Time `json:"expires_at,omitempty"`
	CreatedAt   time.Time  `json:"created_at"`
}

// Invoice represents a billing invoice
type Invoice struct {
	ID          string    `json:"id"`
	Number      string    `json:"number"`
	Amount      float64   `json:"amount"`
	Currency    string    `json:"currency"`
	Status      string    `json:"status"`
	DueDate     time.Time `json:"due_date"`
	PaidAt      *time.Time `json:"paid_at,omitempty"`
	PDFURL      string    `json:"pdf_url,omitempty"`
	LineItems   []InvoiceLineItem `json:"line_items,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
}

// InvoiceLineItem represents a line item on an invoice
type InvoiceLineItem struct {
	Description string  `json:"description"`
	Quantity    float64 `json:"quantity"`
	UnitPrice   float64 `json:"unit_price"`
	Amount      float64 `json:"amount"`
}

// Subscription represents a subscription
type Subscription struct {
	ID              string    `json:"id"`
	Plan            string    `json:"plan"`
	Status          string    `json:"status"`
	CurrentPeriodStart time.Time `json:"current_period_start"`
	CurrentPeriodEnd   time.Time `json:"current_period_end"`
	CancelAtPeriodEnd  bool     `json:"cancel_at_period_end"`
	CreatedAt       time.Time `json:"created_at"`
}

// ListParams represents common list parameters
type ListParams struct {
	Limit  int    `json:"limit,omitempty"`
	Offset int    `json:"offset,omitempty"`
	Sort   string `json:"sort,omitempty"`
	Order  string `json:"order,omitempty"`
}

// ToMap converts ListParams to a map for query parameters
func (p ListParams) ToMap() map[string]string {
	m := make(map[string]string)
	if p.Limit > 0 {
		m["limit"] = strconv.Itoa(p.Limit)
	}
	if p.Offset > 0 {
		m["offset"] = strconv.Itoa(p.Offset)
	}
	if p.Sort != "" {
		m["sort"] = p.Sort
	}
	if p.Order != "" {
		m["order"] = p.Order
	}
	return m
}

// PaginatedResponse represents a paginated API response
type PaginatedResponse[T any] struct {
	Data   []T `json:"data"`
	Total  int `json:"total"`
	Limit  int `json:"limit"`
	Offset int `json:"offset"`
}
