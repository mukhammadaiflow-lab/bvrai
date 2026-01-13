/**
 * Builder Engine TypeScript SDK - Models
 *
 * Type definitions for all API resources.
 */

// =============================================================================
// Enums
// =============================================================================

export enum CallStatus {
  QUEUED = 'queued',
  RINGING = 'ringing',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  FAILED = 'failed',
  BUSY = 'busy',
  NO_ANSWER = 'no_answer',
  CANCELED = 'canceled',
}

export enum CallDirection {
  INBOUND = 'inbound',
  OUTBOUND = 'outbound',
}

export enum AgentStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  DRAFT = 'draft',
  ARCHIVED = 'archived',
}

export enum VoiceProvider {
  ELEVENLABS = 'elevenlabs',
  OPENAI = 'openai',
  AZURE = 'azure',
  GOOGLE = 'google',
  AWS_POLLY = 'aws_polly',
  DEEPGRAM = 'deepgram',
  PLAYHT = 'playht',
  CARTESIA = 'cartesia',
}

export enum STTProvider {
  DEEPGRAM = 'deepgram',
  OPENAI_WHISPER = 'openai_whisper',
  AZURE = 'azure',
  GOOGLE = 'google',
  AWS_TRANSCRIBE = 'aws_transcribe',
  ASSEMBLY_AI = 'assembly_ai',
}

export enum LLMProvider {
  OPENAI = 'openai',
  ANTHROPIC = 'anthropic',
  AZURE_OPENAI = 'azure_openai',
  GOOGLE = 'google',
  COHERE = 'cohere',
  MISTRAL = 'mistral',
  GROQ = 'groq',
  TOGETHER = 'together',
}

export enum WebhookEvent {
  CALL_STARTED = 'call.started',
  CALL_RINGING = 'call.ringing',
  CALL_ANSWERED = 'call.answered',
  CALL_ENDED = 'call.ended',
  CALL_FAILED = 'call.failed',
  TRANSCRIPTION_READY = 'transcription.ready',
  TRANSCRIPTION_PARTIAL = 'transcription.partial',
  AGENT_RESPONSE = 'agent.response',
  FUNCTION_CALLED = 'function.called',
  RECORDING_READY = 'recording.ready',
  DTMF_RECEIVED = 'dtmf.received',
  CALL_TRANSFERRED = 'call.transferred',
  VOICEMAIL_DETECTED = 'voicemail.detected',
  HUMAN_DETECTED = 'human.detected',
}

export enum CampaignStatus {
  DRAFT = 'draft',
  SCHEDULED = 'scheduled',
  RUNNING = 'running',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  CANCELED = 'canceled',
}

export enum WorkflowTrigger {
  CALL_STARTED = 'call_started',
  CALL_ENDED = 'call_ended',
  KEYWORD_DETECTED = 'keyword_detected',
  INTENT_DETECTED = 'intent_detected',
  SENTIMENT_NEGATIVE = 'sentiment_negative',
  SILENCE_DETECTED = 'silence_detected',
  TRANSFER_REQUESTED = 'transfer_requested',
  SCHEDULE = 'schedule',
  WEBHOOK = 'webhook',
  API = 'api',
}

export enum PhoneNumberType {
  LOCAL = 'local',
  TOLL_FREE = 'toll_free',
  MOBILE = 'mobile',
}

export enum PhoneNumberCapability {
  VOICE = 'voice',
  SMS = 'sms',
  MMS = 'mms',
  FAX = 'fax',
}

// =============================================================================
// Configuration Types
// =============================================================================

export interface VoiceConfig {
  provider?: VoiceProvider;
  voiceId?: string;
  model?: string;
  language?: string;
  speakingRate?: number;
  pitch?: number;
  volumeGainDb?: number;
  stability?: number;
  similarityBoost?: number;
  style?: number;
  useSpeakerBoost?: boolean;
  optimizeStreamingLatency?: number;
}

export interface STTConfig {
  provider?: STTProvider;
  model?: string;
  language?: string;
  punctuate?: boolean;
  profanityFilter?: boolean;
  diarize?: boolean;
  smartFormat?: boolean;
  fillerWords?: boolean;
  interimResults?: boolean;
  endpointing?: number;
  vadEvents?: boolean;
  keywords?: string[];
}

export interface LLMConfig {
  provider?: LLMProvider;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  stopSequences?: string[];
  systemPrompt?: string;
  contextWindow?: number;
  responseFormat?: string;
}

export interface FunctionDefinition {
  name: string;
  description: string;
  parameters: Record<string, any>;
  required?: string[];
  webhookUrl?: string;
  asyncExecution?: boolean;
  timeoutSeconds?: number;
}

export interface AgentConfig {
  voice?: VoiceConfig;
  stt?: STTConfig;
  llm?: LLMConfig;
  firstMessage?: string;
  endCallMessage?: string;
  endCallPhrases?: string[];
  interruptionThreshold?: number;
  maxDurationSeconds?: number;
  silenceTimeoutSeconds?: number;
  responseDelayMs?: number;
  recordingEnabled?: boolean;
  transcriptionEnabled?: boolean;
  voicemailDetection?: boolean;
  voicemailMessage?: string;
  answeringMachineDetection?: boolean;
  backgroundSound?: string;
  backgroundVolume?: number;
  functions?: FunctionDefinition[];
  metadata?: Record<string, any>;
}

// =============================================================================
// Core Models
// =============================================================================

export interface Agent {
  id: string;
  name: string;
  description?: string;
  status: AgentStatus;
  config?: AgentConfig;
  voiceId?: string;
  phoneNumberId?: string;
  knowledgeBaseIds?: string[];
  workflowIds?: string[];
  organizationId?: string;
  createdAt?: string;
  updatedAt?: string;
  metadata?: Record<string, any>;
  totalCalls?: number;
  totalDurationSeconds?: number;
  averageDurationSeconds?: number;
  successRate?: number;
}

export interface Call {
  id: string;
  agentId: string;
  status: CallStatus;
  direction: CallDirection;
  fromNumber?: string;
  toNumber?: string;
  durationSeconds?: number;
  startTime?: string;
  endTime?: string;
  answerTime?: string;
  recordingUrl?: string;
  transcriptUrl?: string;
  cost?: number;
  costBreakdown?: Record<string, number>;
  endedReason?: string;
  errorMessage?: string;
  metadata?: Record<string, any>;
  organizationId?: string;
  createdAt?: string;
  sentimentScore?: number;
  summary?: string;
  detectedIntents?: string[];
  actionItems?: string[];
}

export interface Message {
  id: string;
  conversationId: string;
  role: 'user' | 'assistant' | 'system' | 'function';
  content: string;
  timestamp: string;
  audioUrl?: string;
  durationMs?: number;
  tokensUsed?: number;
  confidence?: number;
  functionCall?: Record<string, any>;
  functionResult?: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface Conversation {
  id: string;
  callId: string;
  agentId: string;
  messages?: Message[];
  totalTokens?: number;
  totalCost?: number;
  startedAt?: string;
  endedAt?: string;
  summary?: string;
  sentiment?: string;
  topics?: string[];
  metadata?: Record<string, any>;
}

export interface PhoneNumber {
  id: string;
  number: string;
  friendlyName?: string;
  type: PhoneNumberType;
  capabilities?: PhoneNumberCapability[];
  countryCode?: string;
  region?: string;
  provider?: string;
  agentId?: string;
  voiceUrl?: string;
  smsUrl?: string;
  statusCallbackUrl?: string;
  monthlyCost?: number;
  status?: string;
  organizationId?: string;
  createdAt?: string;
  metadata?: Record<string, any>;
}

export interface Voice {
  id: string;
  name: string;
  description?: string;
  provider: VoiceProvider;
  providerVoiceId: string;
  language?: string;
  gender?: string;
  age?: string;
  accent?: string;
  previewUrl?: string;
  isCustom?: boolean;
  isCloned?: boolean;
  config?: VoiceConfig;
  organizationId?: string;
  createdAt?: string;
  metadata?: Record<string, any>;
}

export interface Webhook {
  id: string;
  url: string;
  events: WebhookEvent[];
  secret?: string;
  enabled: boolean;
  description?: string;
  headers?: Record<string, string>;
  retryCount?: number;
  timeoutSeconds?: number;
  organizationId?: string;
  createdAt?: string;
  totalDeliveries?: number;
  successfulDeliveries?: number;
  failedDeliveries?: number;
  lastTriggeredAt?: string;
  lastStatusCode?: number;
}

export interface Document {
  id: string;
  knowledgeBaseId: string;
  name: string;
  content?: string;
  contentType?: string;
  fileUrl?: string;
  fileSize?: number;
  chunkCount?: number;
  vectorStatus?: string;
  createdAt?: string;
  updatedAt?: string;
  metadata?: Record<string, any>;
}

export interface KnowledgeBase {
  id: string;
  name: string;
  description?: string;
  documentCount?: number;
  totalChunks?: number;
  embeddingModel?: string;
  chunkSize?: number;
  chunkOverlap?: number;
  organizationId?: string;
  createdAt?: string;
  updatedAt?: string;
  metadata?: Record<string, any>;
}

export interface WorkflowAction {
  id: string;
  type: string;
  config: Record<string, any>;
  condition?: string;
  order?: number;
}

export interface Workflow {
  id: string;
  name: string;
  description?: string;
  trigger: WorkflowTrigger;
  triggerConfig?: Record<string, any>;
  actions: WorkflowAction[];
  enabled: boolean;
  agentIds?: string[];
  organizationId?: string;
  createdAt?: string;
  updatedAt?: string;
  totalExecutions?: number;
  successfulExecutions?: number;
  failedExecutions?: number;
  lastExecutedAt?: string;
}

export interface CampaignContact {
  id: string;
  campaignId: string;
  phoneNumber: string;
  name?: string;
  email?: string;
  customData?: Record<string, any>;
  status?: string;
  callId?: string;
  attempts?: number;
  lastAttemptAt?: string;
  scheduledAt?: string;
}

export interface Campaign {
  id: string;
  name: string;
  description?: string;
  agentId: string;
  status: CampaignStatus;
  contacts?: CampaignContact[];
  totalContacts?: number;
  completedContacts?: number;
  failedContacts?: number;
  skippedContacts?: number;
  maxConcurrentCalls?: number;
  callsPerMinute?: number;
  maxAttempts?: number;
  retryDelayMinutes?: number;
  scheduledStart?: string;
  scheduledEnd?: string;
  timezone?: string;
  callingHoursStart?: string;
  callingHoursEnd?: string;
  callingDays?: string[];
  startedAt?: string;
  completedAt?: string;
  organizationId?: string;
  createdAt?: string;
  metadata?: Record<string, any>;
}

// =============================================================================
// Analytics Models
// =============================================================================

export interface CallMetrics {
  totalCalls?: number;
  completedCalls?: number;
  failedCalls?: number;
  totalDurationSeconds?: number;
  averageDurationSeconds?: number;
  totalCost?: number;
  averageCost?: number;
  successRate?: number;
  answerRate?: number;
}

export interface UsageMetrics {
  periodStart: string;
  periodEnd: string;
  totalCalls?: number;
  totalMinutes?: number;
  sttMinutes?: number;
  ttsCharacters?: number;
  llmTokens?: number;
  storageBytes?: number;
  totalCost?: number;
  costBreakdown?: Record<string, number>;
}

export interface Analytics {
  period: string;
  startDate: string;
  endDate: string;
  callMetrics?: CallMetrics;
  usageMetrics?: UsageMetrics;
  topAgents?: Record<string, any>[];
  callVolumeByHour?: Record<string, number>;
  callVolumeByDay?: Record<string, number>;
  sentimentDistribution?: Record<string, number>;
  topIntents?: Record<string, any>[];
  averageHandleTime?: number;
  firstCallResolutionRate?: number;
}

export interface Usage {
  organizationId: string;
  periodStart: string;
  periodEnd: string;
  callsUsed?: number;
  callsLimit?: number;
  minutesUsed?: number;
  minutesLimit?: number;
  storageUsedBytes?: number;
  storageLimitBytes?: number;
  agentsUsed?: number;
  agentsLimit?: number;
  phoneNumbersUsed?: number;
  phoneNumbersLimit?: number;
  currentSpend?: number;
  spendLimit?: number;
}

// =============================================================================
// Organization & User Models
// =============================================================================

export interface Organization {
  id: string;
  name: string;
  slug: string;
  plan?: string;
  status?: string;
  ownerId?: string;
  memberCount?: number;
  settings?: Record<string, any>;
  billingEmail?: string;
  createdAt?: string;
  updatedAt?: string;
  metadata?: Record<string, any>;
}

export interface User {
  id: string;
  email: string;
  name?: string;
  role?: string;
  organizationId?: string;
  avatarUrl?: string;
  phone?: string;
  timezone?: string;
  emailVerified?: boolean;
  lastLoginAt?: string;
  createdAt?: string;
  metadata?: Record<string, any>;
}

export interface APIKey {
  id: string;
  name: string;
  keyPrefix: string;
  permissions?: string[];
  rateLimit?: number;
  expiresAt?: string;
  lastUsedAt?: string;
  organizationId?: string;
  createdBy?: string;
  createdAt?: string;
}

// =============================================================================
// Billing Models
// =============================================================================

export interface Invoice {
  id: string;
  organizationId: string;
  number: string;
  status: string;
  amount: number;
  currency?: string;
  periodStart: string;
  periodEnd: string;
  dueDate?: string;
  paidAt?: string;
  pdfUrl?: string;
  lineItems?: Record<string, any>[];
  createdAt?: string;
}

export interface Subscription {
  id: string;
  organizationId: string;
  planId: string;
  planName: string;
  status: string;
  currentPeriodStart: string;
  currentPeriodEnd: string;
  cancelAtPeriodEnd?: boolean;
  canceledAt?: string;
  trialEnd?: string;
  monthlyPrice?: number;
  currency?: string;
  createdAt?: string;
}

export interface PaymentMethod {
  id: string;
  type: string;
  isDefault?: boolean;
  cardBrand?: string;
  cardLast4?: string;
  cardExpMonth?: number;
  cardExpYear?: number;
  billingAddress?: Record<string, string>;
  createdAt?: string;
}

// =============================================================================
// Request/Response Types
// =============================================================================

export interface CreateAgentRequest {
  name: string;
  description?: string;
  voiceId?: string;
  phoneNumberId?: string;
  knowledgeBaseIds?: string[];
  workflowIds?: string[];
  config?: AgentConfig;
  voiceConfig?: VoiceConfig;
  sttConfig?: STTConfig;
  llmConfig?: LLMConfig;
  firstMessage?: string;
  systemPrompt?: string;
  functions?: FunctionDefinition[];
  metadata?: Record<string, any>;
}

export interface UpdateAgentRequest {
  name?: string;
  description?: string;
  status?: AgentStatus;
  voiceId?: string;
  phoneNumberId?: string;
  knowledgeBaseIds?: string[];
  workflowIds?: string[];
  config?: AgentConfig;
  metadata?: Record<string, any>;
}

export interface CreateCallRequest {
  agentId: string;
  toNumber: string;
  fromNumber?: string;
  phoneNumberId?: string;
  firstMessage?: string;
  context?: Record<string, any>;
  metadata?: Record<string, any>;
  record?: boolean;
  maxDuration?: number;
  webhookUrl?: string;
  statusCallbackUrl?: string;
  answeringMachineDetection?: boolean;
  voicemailMessage?: string;
}

export interface CreateWebhookRequest {
  url: string;
  events: WebhookEvent[];
  description?: string;
  headers?: Record<string, string>;
  enabled?: boolean;
  retryCount?: number;
  timeoutSeconds?: number;
}

export interface CreateCampaignRequest {
  name: string;
  agentId: string;
  contacts?: Array<{
    phoneNumber: string;
    name?: string;
    email?: string;
    customData?: Record<string, any>;
  }>;
  description?: string;
  maxConcurrentCalls?: number;
  callsPerMinute?: number;
  maxAttempts?: number;
  retryDelayMinutes?: number;
  scheduledStart?: string;
  scheduledEnd?: string;
  timezone?: string;
  callingHoursStart?: string;
  callingHoursEnd?: string;
  callingDays?: string[];
  metadata?: Record<string, any>;
}

export interface ListParams {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}
