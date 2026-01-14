/**
 * Builder Voice AI Platform - TypeScript Types
 *
 * Comprehensive type definitions for the entire platform.
 */

// =============================================================================
// Base Types
// =============================================================================

export interface BaseEntity {
  id: string;
  created_at: string;
  updated_at: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

// =============================================================================
// Organization & Users
// =============================================================================

export interface Organization extends BaseEntity {
  name: string;
  slug: string;
  plan: "free" | "starter" | "professional" | "enterprise";
  is_active: boolean;
}

export interface User extends BaseEntity {
  organization_id: string;
  email: string;
  name: string | null;
  role: "owner" | "admin" | "member" | "viewer";
  is_active: boolean;
  last_login: string | null;
}

export interface ApiKey extends BaseEntity {
  organization_id: string;
  user_id: string | null;
  name: string;
  key_prefix: string;
  scopes: string[];
  last_used: string | null;
  expires_at: string | null;
  is_active: boolean;
}

// =============================================================================
// Agents
// =============================================================================

export type AgentStatus = "draft" | "active" | "paused" | "archived";

export interface AgentLLMConfig {
  provider: string;
  model: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
}

export interface AgentTool {
  id: string;
  name: string;
  description: string;
  type: "function" | "webhook" | "transfer";
  config: Record<string, unknown>;
}

export interface Agent extends BaseEntity {
  organization_id: string;
  name: string;
  description: string | null;
  system_prompt: string | null;
  voice_config_id: string | null;
  llm_config: AgentLLMConfig | null;
  tools: AgentTool[];
  status: AgentStatus;
  version: number;
  created_by: string | null;
  updated_by: string | null;
}

export interface AgentVersion extends BaseEntity {
  agent_id: string;
  version: number;
  config_snapshot: Record<string, unknown>;
  changelog: string | null;
  created_by: string | null;
}

export interface CreateAgentRequest {
  name: string;
  description?: string;
  system_prompt?: string;
  voice_config_id?: string;
  llm_config?: AgentLLMConfig;
  tools?: AgentTool[];
}

export interface UpdateAgentRequest extends Partial<CreateAgentRequest> {
  status?: AgentStatus;
}

// =============================================================================
// Voice Configuration
// =============================================================================

export type STTProvider =
  | "deepgram"
  | "openai"
  | "google"
  | "azure"
  | "aws"
  | "assemblyai"
  | "rev"
  | "speechmatics";

export type TTSProvider =
  | "elevenlabs"
  | "playht"
  | "cartesia"
  | "openai"
  | "google"
  | "azure"
  | "aws"
  | "murf"
  | "resemble"
  | "wellsaid"
  | "lmnt";

export interface VoiceConfiguration extends BaseEntity {
  organization_id: string;
  name: string;
  description: string | null;
  stt_provider: STTProvider;
  stt_config: Record<string, unknown>;
  tts_provider: TTSProvider;
  tts_config: Record<string, unknown>;
  voice_id: string | null;
  language: string;
  speed: number;
  pitch: number;
  is_default: boolean;
}

export interface Voice {
  id: string;
  name: string;
  provider: TTSProvider;
  language: string;
  gender: "male" | "female" | "neutral";
  preview_url?: string;
  description?: string;
  tags?: string[];
}

// =============================================================================
// Calls & Conversations
// =============================================================================

export type CallDirection = "inbound" | "outbound";
export type CallStatus =
  | "initiated"
  | "ringing"
  | "in_progress"
  | "completed"
  | "failed"
  | "busy"
  | "no_answer";
export type ConversationStatus = "active" | "completed" | "transferred" | "failed";
export type MessageRole = "user" | "assistant" | "system" | "tool";

export interface Call extends BaseEntity {
  organization_id: string;
  agent_id: string;
  conversation_id: string | null;
  external_call_id: string | null;
  direction: CallDirection;
  from_number: string | null;
  to_number: string | null;
  status: CallStatus;
  hangup_reason: string | null;
  recording_url: string | null;
  started_at: string | null;
  answered_at: string | null;
  ended_at: string | null;
  duration_seconds: number | null;
  cost_cents: number | null;
  metadata: Record<string, unknown>;
}

export interface Conversation extends BaseEntity {
  organization_id: string;
  agent_id: string;
  call_id: string | null;
  customer_phone: string | null;
  customer_id: string | null;
  channel: "phone" | "web" | "api";
  status: ConversationStatus;
  metadata: Record<string, unknown>;
  started_at: string;
  ended_at: string | null;
  duration_seconds: number | null;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: MessageRole;
  content: string;
  audio_url: string | null;
  metadata: Record<string, unknown>;
  latency_ms: number | null;
  token_count: number | null;
  created_at: string;
}

export interface CallEvent {
  id: string;
  call_id: string;
  event_type: string;
  event_data: Record<string, unknown>;
  created_at: string;
}

// =============================================================================
// Analytics
// =============================================================================

export interface AnalyticsSummary {
  period: string;
  total_calls: number;
  total_duration_minutes: number;
  average_duration_seconds: number;
  successful_calls: number;
  failed_calls: number;
  success_rate: number;
  total_cost_cents: number;
  unique_callers: number;
}

export interface CallsByHour {
  hour: number;
  count: number;
  avg_duration: number;
}

export interface CallsByDay {
  date: string;
  count: number;
  total_duration: number;
  success_rate: number;
}

export interface AgentPerformance {
  agent_id: string;
  agent_name: string;
  total_calls: number;
  avg_duration: number;
  success_rate: number;
  avg_latency_ms: number;
  sentiment_score: number;
}

export interface UsageRecord {
  id: string;
  organization_id: string;
  period_start: string;
  period_end: string;
  metric_type: string;
  metric_value: number;
  unit: string | null;
}

// =============================================================================
// Webhooks
// =============================================================================

export type WebhookEvent =
  | "call.started"
  | "call.ended"
  | "call.failed"
  | "conversation.message"
  | "conversation.ended"
  | "agent.updated"
  | "transfer.initiated"
  | "transfer.completed";

export interface Webhook extends BaseEntity {
  organization_id: string;
  name: string;
  url: string;
  events: WebhookEvent[];
  secret: string;
  is_active: boolean;
  last_triggered_at: string | null;
  failure_count: number;
}

export interface WebhookDelivery {
  id: string;
  webhook_id: string;
  event_type: WebhookEvent;
  payload: Record<string, unknown>;
  response_status: number | null;
  response_body: string | null;
  duration_ms: number | null;
  success: boolean;
  created_at: string;
}

// =============================================================================
// Billing
// =============================================================================

export type BillingPlan = "free" | "starter" | "professional" | "enterprise";

export interface BillingPlanDetails {
  id: BillingPlan;
  name: string;
  price_monthly: number;
  price_yearly: number;
  features: string[];
  limits: {
    calls_per_month: number;
    minutes_per_month: number;
    agents: number;
    team_members: number;
    webhooks: number;
    api_rate_limit: number;
  };
}

export interface Invoice {
  id: string;
  organization_id: string;
  amount_cents: number;
  status: "draft" | "open" | "paid" | "void" | "uncollectible";
  period_start: string;
  period_end: string;
  due_date: string;
  paid_at: string | null;
  pdf_url: string | null;
  created_at: string;
}

export interface UsageSummary {
  period: string;
  calls: number;
  minutes: number;
  stt_characters: number;
  tts_characters: number;
  llm_tokens: number;
  estimated_cost_cents: number;
}

// =============================================================================
// Settings
// =============================================================================

export interface OrganizationSettings {
  default_stt_provider: STTProvider;
  default_tts_provider: TTSProvider;
  default_llm_provider: string;
  default_llm_model: string;
  call_recording_enabled: boolean;
  analytics_enabled: boolean;
  webhook_retry_count: number;
  max_concurrent_calls: number;
}

// =============================================================================
// Real-time Events (WebSocket)
// =============================================================================

export type RealtimeEventType =
  | "call:started"
  | "call:ended"
  | "call:status_changed"
  | "message:received"
  | "agent:speaking"
  | "agent:listening"
  | "transcript:partial"
  | "transcript:final"
  | "error";

export interface RealtimeEvent {
  type: RealtimeEventType;
  call_id?: string;
  conversation_id?: string;
  data: Record<string, unknown>;
  timestamp: string;
}

// =============================================================================
// Dashboard Statistics
// =============================================================================

export interface DashboardStats {
  today: {
    total_calls: number;
    total_minutes: number;
    active_calls: number;
    success_rate: number;
  };
  week: {
    total_calls: number;
    total_minutes: number;
    avg_calls_per_day: number;
    trend_percentage: number;
  };
  agents: {
    total: number;
    active: number;
  };
  usage: {
    calls_used: number;
    calls_limit: number;
    minutes_used: number;
    minutes_limit: number;
  };
}
