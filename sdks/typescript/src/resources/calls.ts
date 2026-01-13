/**
 * Builder Engine TypeScript SDK - Calls Resource
 *
 * Methods for managing voice calls.
 */

import type { BuilderEngine } from '../client';
import type {
  Call,
  CallStatus,
  CallDirection,
  Conversation,
  CreateCallRequest,
  PaginatedResponse,
  ListParams,
} from '../models';
import { TimeoutError } from '../exceptions';

export interface ListCallsParams extends ListParams {
  agentId?: string;
  status?: CallStatus;
  direction?: CallDirection;
  fromNumber?: string;
  toNumber?: string;
  startDate?: string;
  endDate?: string;
  minDuration?: number;
  maxDuration?: number;
}

/**
 * Resource for managing voice calls.
 *
 * Calls represent individual voice interactions between an agent
 * and a phone number. This resource provides methods for initiating
 * outbound calls, managing ongoing calls, and retrieving call history.
 *
 * @example
 * ```typescript
 * const client = new BuilderEngine({ apiKey: '...' });
 *
 * // Make an outbound call
 * const call = await client.calls.create({
 *   agentId: 'agent_abc123',
 *   toNumber: '+1234567890'
 * });
 *
 * // Wait for completion
 * const completed = await client.calls.waitForCompletion(call.id);
 * console.log('Duration:', completed.durationSeconds);
 * ```
 */
export class CallsResource {
  constructor(private readonly client: BuilderEngine) {}

  /**
   * List all calls.
   */
  async list(params: ListCallsParams = {}): Promise<PaginatedResponse<Call>> {
    return this.client.request({
      method: 'GET',
      path: '/api/v1/calls',
      params: {
        page: params.page || 1,
        page_size: params.pageSize || 20,
        agent_id: params.agentId,
        status: params.status,
        direction: params.direction,
        from_number: params.fromNumber,
        to_number: params.toNumber,
        start_date: params.startDate,
        end_date: params.endDate,
        min_duration: params.minDuration,
        max_duration: params.maxDuration,
        sort_by: params.sortBy || 'created_at',
        sort_order: params.sortOrder || 'desc',
      },
    });
  }

  /**
   * Get a call by ID.
   */
  async get(callId: string): Promise<Call> {
    return this.client.request({
      method: 'GET',
      path: `/api/v1/calls/${callId}`,
    });
  }

  /**
   * Create a new outbound call.
   */
  async create(data: CreateCallRequest): Promise<Call> {
    const body: Record<string, any> = {
      agent_id: data.agentId,
      to_number: data.toNumber,
    };

    if (data.fromNumber) body.from_number = data.fromNumber;
    if (data.phoneNumberId) body.phone_number_id = data.phoneNumberId;
    if (data.firstMessage) body.first_message = data.firstMessage;
    if (data.context) body.context = data.context;
    if (data.metadata) body.metadata = data.metadata;
    if (data.record !== undefined) body.record = data.record;
    if (data.maxDuration) body.max_duration = data.maxDuration;
    if (data.webhookUrl) body.webhook_url = data.webhookUrl;
    if (data.statusCallbackUrl) body.status_callback_url = data.statusCallbackUrl;
    if (data.answeringMachineDetection !== undefined) {
      body.answering_machine_detection = data.answeringMachineDetection;
    }
    if (data.voicemailMessage) body.voicemail_message = data.voicemailMessage;

    return this.client.request({
      method: 'POST',
      path: '/api/v1/calls',
      body,
    });
  }

  /**
   * Hang up an active call.
   */
  async hangup(callId: string, reason?: string): Promise<Call> {
    return this.client.request({
      method: 'POST',
      path: `/api/v1/calls/${callId}/hangup`,
      body: reason ? { reason } : undefined,
    });
  }

  /**
   * Transfer an active call.
   */
  async transfer(
    callId: string,
    toNumber: string,
    options: { announce?: string; warmTransfer?: boolean } = {}
  ): Promise<Call> {
    return this.client.request({
      method: 'POST',
      path: `/api/v1/calls/${callId}/transfer`,
      body: {
        to_number: toNumber,
        announce: options.announce,
        warm_transfer: options.warmTransfer || false,
      },
    });
  }

  /**
   * Mute or unmute a call.
   */
  async mute(callId: string, muted: boolean = true): Promise<Call> {
    return this.client.request({
      method: 'POST',
      path: `/api/v1/calls/${callId}/mute`,
      body: { muted },
    });
  }

  /**
   * Put a call on hold or resume.
   */
  async hold(
    callId: string,
    onHold: boolean = true,
    musicUrl?: string
  ): Promise<Call> {
    return this.client.request({
      method: 'POST',
      path: `/api/v1/calls/${callId}/hold`,
      body: { on_hold: onHold, music_url: musicUrl },
    });
  }

  /**
   * Send DTMF tones to a call.
   */
  async sendDtmf(callId: string, digits: string): Promise<Call> {
    return this.client.request({
      method: 'POST',
      path: `/api/v1/calls/${callId}/dtmf`,
      body: { digits },
    });
  }

  /**
   * Inject a message into an active call.
   */
  async injectMessage(
    callId: string,
    message: string,
    options: { role?: string; interrupt?: boolean } = {}
  ): Promise<Call> {
    return this.client.request({
      method: 'POST',
      path: `/api/v1/calls/${callId}/inject`,
      body: {
        message,
        role: options.role || 'assistant',
        interrupt: options.interrupt || false,
      },
    });
  }

  /**
   * Get the recording for a completed call.
   */
  async getRecording(callId: string): Promise<{ url: string; durationSeconds: number }> {
    return this.client.request({
      method: 'GET',
      path: `/api/v1/calls/${callId}/recording`,
    });
  }

  /**
   * Get the transcript for a completed call.
   */
  async getTranscript(
    callId: string,
    options: {
      format?: 'json' | 'text' | 'srt' | 'vtt';
      includeTimestamps?: boolean;
      includeSpeakerLabels?: boolean;
    } = {}
  ): Promise<Record<string, any>> {
    return this.client.request({
      method: 'GET',
      path: `/api/v1/calls/${callId}/transcript`,
      params: {
        format: options.format || 'json',
        include_timestamps: options.includeTimestamps ?? true,
        include_speaker_labels: options.includeSpeakerLabels ?? true,
      },
    });
  }

  /**
   * Get the conversation for a call.
   */
  async getConversation(callId: string): Promise<Conversation> {
    return this.client.request({
      method: 'GET',
      path: `/api/v1/conversations/${callId}`,
    });
  }

  /**
   * Create multiple outbound calls.
   */
  async bulkCreate(
    agentId: string,
    calls: Array<{ toNumber: string; context?: Record<string, any>; metadata?: Record<string, any> }>,
    delayBetweenCallsMs: number = 1000
  ): Promise<Call[]> {
    const response = await this.client.request<{ calls: Call[] }>({
      method: 'POST',
      path: '/api/v1/calls/bulk',
      body: {
        agent_id: agentId,
        calls: calls.map(c => ({
          to_number: c.toNumber,
          context: c.context,
          metadata: c.metadata,
        })),
        delay_between_calls_ms: delayBetweenCallsMs,
      },
    });
    return response.calls;
  }

  /**
   * Get all currently active calls.
   */
  async getActive(agentId?: string): Promise<Call[]> {
    const response = await this.client.request<{ items: Call[] }>({
      method: 'GET',
      path: '/api/v1/calls',
      params: {
        status: 'in_progress',
        agent_id: agentId,
      },
    });
    return response.items;
  }

  /**
   * Wait for a call to complete.
   */
  async waitForCompletion(
    callId: string,
    options: { timeoutSeconds?: number; pollIntervalSeconds?: number } = {}
  ): Promise<Call> {
    const { timeoutSeconds = 300, pollIntervalSeconds = 2 } = options;
    const terminalStatuses = new Set([
      CallStatus.COMPLETED,
      CallStatus.FAILED,
      CallStatus.BUSY,
      CallStatus.NO_ANSWER,
      CallStatus.CANCELED,
    ]);

    const startTime = Date.now();

    while (true) {
      const call = await this.get(callId);

      if (terminalStatuses.has(call.status)) {
        return call;
      }

      const elapsed = (Date.now() - startTime) / 1000;
      if (elapsed >= timeoutSeconds) {
        throw new TimeoutError(
          `Call ${callId} did not complete within ${timeoutSeconds}s`,
          timeoutSeconds * 1000
        );
      }

      await new Promise(resolve => setTimeout(resolve, pollIntervalSeconds * 1000));
    }
  }
}
