/**
 * TypeScript types for Builder Engine SDK
 */

// ============================================================================
// Common Types
// ============================================================================

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
  hasMore: boolean;
}

export interface Metadata {
  [key: string]: string | number | boolean | null;
}

// ============================================================================
// Agent Types
// ============================================================================

export type VoiceProvider = 'elevenlabs' | 'playht' | 'azure' | 'google' | 'aws';
export type LLMProvider = 'openai' | 'anthropic' | 'groq' | 'together' | 'azure';
export type ASRProvider = 'deepgram' | 'assemblyai' | 'google' | 'azure' | 'aws';

export interface VoiceConfig {
  provider: VoiceProvider;
  voiceId: string;
  speed?: number;
  pitch?: number;
  stability?: number;
  similarityBoost?: number;
}

export interface LLMConfig {
  provider: LLMProvider;
  model: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
}

export interface ASRConfig {
  provider: ASRProvider;
  model?: string;
  language?: string;
  punctuate?: boolean;
  profanityFilter?: boolean;
}

export interface ToolParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'array' | 'object';
  description?: string;
  required?: boolean;
  enum?: string[];
  default?: unknown;
}

export interface Tool {
  name: string;
  description: string;
  parameters: ToolParameter[];
  webhookUrl?: string;
}

export interface AgentConfig {
  voice?: VoiceConfig;
  llm?: LLMConfig;
  asr?: ASRConfig;
  interruptionThreshold?: number;
  silenceTimeout?: number;
  maxDuration?: number;
  recordCalls?: boolean;
  transcribeCalls?: boolean;
  tools?: Tool[];
  firstMessage?: string;
  endCallMessage?: string;
  transferNumber?: string;
}

export interface Agent {
  id: string;
  name: string;
  systemPrompt: string;
  config: AgentConfig;
  status: 'active' | 'inactive' | 'draft';
  createdAt: string;
  updatedAt: string;
  metadata?: Metadata;
}

export interface CreateAgentRequest {
  name: string;
  systemPrompt: string;
  config?: AgentConfig;
  metadata?: Metadata;
}

export interface UpdateAgentRequest {
  name?: string;
  systemPrompt?: string;
  config?: AgentConfig;
  status?: 'active' | 'inactive' | 'draft';
  metadata?: Metadata;
}

// ============================================================================
// Call Types
// ============================================================================

export type CallDirection = 'inbound' | 'outbound';
export type CallStatus =
  | 'queued'
  | 'ringing'
  | 'in-progress'
  | 'completed'
  | 'failed'
  | 'busy'
  | 'no-answer'
  | 'canceled';

export interface Call {
  id: string;
  agentId: string;
  direction: CallDirection;
  status: CallStatus;
  fromNumber?: string;
  toNumber?: string;
  startedAt?: string;
  answeredAt?: string;
  endedAt?: string;
  duration?: number;
  recordingUrl?: string;
  transcriptUrl?: string;
  cost?: number;
  metadata?: Metadata;
  createdAt: string;
  updatedAt: string;
}

export interface CreateCallRequest {
  agentId: string;
  toNumber: string;
  fromNumber?: string;
  metadata?: Metadata;
  scheduledFor?: string;
  maxDuration?: number;
}

export interface CallTranscript {
  id: string;
  callId: string;
  turns: TranscriptTurn[];
  summary?: string;
  sentiment?: SentimentAnalysis;
}

export interface TranscriptTurn {
  speaker: 'user' | 'agent';
  text: string;
  startTime: number;
  endTime: number;
  confidence: number;
  words?: TranscriptWord[];
}

export interface TranscriptWord {
  word: string;
  startTime: number;
  endTime: number;
  confidence: number;
}

export interface SentimentAnalysis {
  overall: 'positive' | 'negative' | 'neutral';
  score: number;
  breakdown: {
    positive: number;
    negative: number;
    neutral: number;
  };
}

// ============================================================================
// Knowledge Base Types
// ============================================================================

export type KnowledgeSourceType = 'text' | 'url' | 'file' | 'api';
export type KnowledgeStatus = 'pending' | 'processing' | 'ready' | 'failed';

export interface KnowledgeBase {
  id: string;
  agentId: string;
  name: string;
  description?: string;
  sourceType: KnowledgeSourceType;
  chunkCount: number;
  status: KnowledgeStatus;
  createdAt: string;
  updatedAt: string;
}

export interface CreateKnowledgeBaseRequest {
  agentId: string;
  name: string;
  description?: string;
}

export interface KnowledgeSearchResult {
  content: string;
  score: number;
  metadata: Metadata;
  sourceId?: string;
}

// ============================================================================
// Webhook Types
// ============================================================================

export type WebhookEvent =
  | 'call.started'
  | 'call.ringing'
  | 'call.answered'
  | 'call.ended'
  | 'call.failed'
  | 'call.transferred'
  | 'conversation.started'
  | 'conversation.turn'
  | 'conversation.ended'
  | 'transcription.partial'
  | 'transcription.final'
  | 'agent.created'
  | 'agent.updated'
  | 'agent.deleted'
  | 'function.called'
  | 'function.completed'
  | 'recording.ready'
  | 'call.analyzed'
  | 'sentiment.detected';

export type WebhookStatus = 'active' | 'inactive' | 'failing' | 'disabled';

export interface Webhook {
  id: string;
  url: string;
  events: WebhookEvent[];
  status: WebhookStatus;
  secret: string;
  description?: string;
  headers?: Record<string, string>;
  retryCount: number;
  timeoutSeconds: number;
  createdAt: string;
  updatedAt: string;
  lastTriggeredAt?: string;
  failureCount: number;
}

export interface CreateWebhookRequest {
  url: string;
  events: WebhookEvent[];
  description?: string;
  headers?: Record<string, string>;
  retryCount?: number;
  timeoutSeconds?: number;
}

export interface WebhookDelivery {
  id: string;
  webhookId: string;
  eventType: WebhookEvent;
  payload: Record<string, unknown>;
  statusCode?: number;
  responseBody?: string;
  durationMs: number;
  success: boolean;
  attemptNumber: number;
  createdAt: string;
  errorMessage?: string;
}

export interface WebhookPayload {
  id: string;
  eventType: WebhookEvent;
  timestamp: string;
  data: Record<string, unknown>;
  metadata?: Metadata;
}

// ============================================================================
// Phone Number Types
// ============================================================================

export type PhoneNumberType = 'local' | 'toll_free' | 'mobile';
export type PhoneNumberStatus = 'active' | 'pending' | 'suspended' | 'released';
export type PhoneNumberCapability = 'voice' | 'sms' | 'mms' | 'fax';

export interface PhoneNumber {
  id: string;
  number: string;
  friendlyName?: string;
  type: PhoneNumberType;
  status: PhoneNumberStatus;
  capabilities: PhoneNumberCapability[];
  agentId?: string;
  countryCode: string;
  areaCode?: string;
  provider: string;
  monthlyCost: number;
  createdAt: string;
  updatedAt: string;
  metadata?: Metadata;
}

export interface AvailableNumber {
  number: string;
  friendlyName: string;
  type: PhoneNumberType;
  capabilities: PhoneNumberCapability[];
  countryCode: string;
  areaCode?: string;
  region?: string;
  locality?: string;
  monthlyCost: number;
  setupCost: number;
}

export interface SearchNumbersRequest {
  countryCode?: string;
  type?: PhoneNumberType;
  areaCode?: string;
  contains?: string;
  capabilities?: PhoneNumberCapability[];
  limit?: number;
}

// ============================================================================
// Analytics Types
// ============================================================================

export type MetricType =
  | 'call_volume'
  | 'call_duration'
  | 'success_rate'
  | 'latency'
  | 'cost'
  | 'tokens'
  | 'errors';

export type AggregationType = 'sum' | 'avg' | 'min' | 'max' | 'count' | 'p50' | 'p90' | 'p95' | 'p99';
export type TimeGranularity = 'minute' | 'hour' | 'day' | 'week' | 'month';

export interface TimeSeriesPoint {
  timestamp: string;
  value: number;
  metadata?: Metadata;
}

export interface TimeSeries {
  metric: string;
  granularity: TimeGranularity;
  points: TimeSeriesPoint[];
  aggregation: AggregationType;
}

export interface CallMetrics {
  totalCalls: number;
  successfulCalls: number;
  failedCalls: number;
  averageDurationSeconds: number;
  totalDurationSeconds: number;
  averageLatencyMs: number;
  totalCost: number;
  totalTokens: number;
  successRate: number;
}

export interface AgentMetrics {
  agentId: string;
  agentName: string;
  totalCalls: number;
  averageDurationSeconds: number;
  successRate: number;
  averageSentimentScore: number;
  averageResponseTimeMs: number;
  topIntents: Array<{ intent: string; count: number }>;
  totalCost: number;
}

export interface UsageMetrics {
  periodStart: string;
  periodEnd: string;
  totalMinutes: number;
  totalCalls: number;
  totalTokens: number;
  totalCost: number;
  breakdownByAgent: Record<string, { calls: number; minutes: number; cost: number }>;
  breakdownByService: Record<string, { usage: number; cost: number }>;
}

export interface RealtimeMetrics {
  timestamp: string;
  activeCalls: number;
  callsPerMinute: number;
  averageQueueTimeMs: number;
  errorRate: number;
  activeAgents: number;
  concurrentConnections: number;
}

// ============================================================================
// Streaming Types
// ============================================================================

export type StreamEventType =
  | 'connected'
  | 'disconnected'
  | 'error'
  | 'ping'
  | 'pong'
  | 'call.started'
  | 'call.ringing'
  | 'call.answered'
  | 'call.ended'
  | 'call.failed'
  | 'audio.input'
  | 'audio.output'
  | 'transcript.partial'
  | 'transcript.final'
  | 'turn.started'
  | 'turn.ended'
  | 'agent.thinking'
  | 'agent.speaking'
  | 'function.call'
  | 'function.result'
  | 'state.changed'
  | 'vad.speech_start'
  | 'vad.speech_end'
  | 'latency.report';

export interface StreamEvent {
  type: StreamEventType;
  timestamp: string;
  data: Record<string, unknown>;
  callId?: string;
  sessionId?: string;
}

export interface TranscriptEvent {
  text: string;
  isFinal: boolean;
  confidence: number;
  speaker: 'user' | 'agent';
  startTime: number;
  endTime: number;
  words?: Array<{
    word: string;
    startTime: number;
    endTime: number;
    confidence: number;
  }>;
}

export interface AudioFrame {
  data: ArrayBuffer | Buffer;
  sampleRate?: number;
  channels?: number;
  encoding?: string;
  timestampMs?: number;
}

// ============================================================================
// Client Configuration
// ============================================================================

export interface ClientConfig {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  maxRetries?: number;
  retryDelay?: number;
}

export interface RequestOptions {
  signal?: AbortSignal;
  timeout?: number;
}
