/**
 * Builder Engine Voice AI SDK for JavaScript/TypeScript
 *
 * @packageDocumentation
 */

// Main client
export { BvraiClient, createClient } from './client';

// API modules
export { AgentsAPI, ToolBuilder } from './api/agents';
export { CallsAPI } from './api/calls';
export { KnowledgeAPI } from './api/knowledge';
export { AnalyticsAPI, AnalyticsQueryBuilder, type Report } from './api/analytics';
export {
  WebhooksAPI,
  WebhookSignatureVerifier,
  WebhookHandler,
  WebhookBuilder,
} from './api/webhooks';
export {
  PhoneNumbersAPI,
  PhoneNumberSearchBuilder,
  type PhoneNumberConfig,
} from './api/phone-numbers';

// Streaming
export {
  StreamingConnection,
  StreamingSession,
  parseTranscriptEvent,
  type StreamingConnectionOptions,
  type StreamEventHandler,
} from './streaming';

// HTTP utilities
export { HttpClient, type HttpRequestOptions } from './utils/http';

// Errors
export {
  BvraiError,
  AuthenticationError,
  NotFoundError,
  RateLimitError,
  ValidationError,
  QuotaExceededError,
  ConflictError,
  ServerError,
  ServiceUnavailableError,
  TimeoutError,
  WebSocketError,
  ConnectionError,
  createErrorFromResponse,
} from './utils/errors';

// Types
export type {
  // Common
  PaginatedResponse,
  Metadata,
  ClientConfig,
  RequestOptions,

  // Agents
  Agent,
  CreateAgentRequest,
  UpdateAgentRequest,
  AgentConfig,
  VoiceConfig,
  LLMConfig,
  ASRConfig,
  Tool,
  ToolParameter,
  VoiceProvider,
  LLMProvider,
  ASRProvider,

  // Calls
  Call,
  CreateCallRequest,
  CallTranscript,
  TranscriptTurn,
  TranscriptWord,
  SentimentAnalysis,
  CallDirection,
  CallStatus,

  // Knowledge
  KnowledgeBase,
  CreateKnowledgeBaseRequest,
  KnowledgeSearchResult,
  KnowledgeSourceType,
  KnowledgeStatus,

  // Webhooks
  Webhook,
  CreateWebhookRequest,
  WebhookDelivery,
  WebhookPayload,
  WebhookEvent,
  WebhookStatus,

  // Phone Numbers
  PhoneNumber,
  AvailableNumber,
  SearchNumbersRequest,
  PhoneNumberType,
  PhoneNumberStatus,
  PhoneNumberCapability,

  // Analytics
  CallMetrics,
  AgentMetrics,
  UsageMetrics,
  RealtimeMetrics,
  TimeSeries,
  TimeSeriesPoint,
  MetricType,
  AggregationType,
  TimeGranularity,

  // Streaming
  StreamEvent,
  StreamEventType,
  TranscriptEvent,
  AudioFrame,
} from './types';

// Version
export const VERSION = '1.0.0';

// Default export
export { BvraiClient as default } from './client';
