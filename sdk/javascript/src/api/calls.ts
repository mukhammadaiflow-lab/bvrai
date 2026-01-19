/**
 * Calls API for Builder Engine SDK
 */

import type { HttpClient } from '../utils/http';
import type {
  Call,
  CreateCallRequest,
  CallTranscript,
  CallDirection,
  CallStatus,
  RequestOptions,
  PaginatedResponse,
  Metadata,
} from '../types';

export class CallsAPI {
  constructor(private readonly http: HttpClient) {}

  /**
   * Create a new outbound call
   */
  async create(request: CreateCallRequest, options?: RequestOptions): Promise<Call> {
    const response = await this.http.post<Call>('/v1/calls', this.transformRequest(request), options);
    return this.transformCall(response);
  }

  /**
   * Create multiple calls in batch
   */
  async batchCreate(
    requests: CreateCallRequest[],
    options?: RequestOptions
  ): Promise<{ calls: Call[]; failed: Array<{ index: number; error: string }> }> {
    const response = await this.http.post<{
      calls: Call[];
      failed: Array<{ index: number; error: string }>;
    }>(
      '/v1/calls/batch',
      { calls: requests.map((r) => this.transformRequest(r)) },
      options
    );

    return {
      calls: response.calls.map((c) => this.transformCall(c)),
      failed: response.failed,
    };
  }

  /**
   * Get a call by ID
   */
  async get(callId: string, options?: RequestOptions): Promise<Call> {
    const response = await this.http.get<Call>(`/v1/calls/${callId}`, undefined, options);
    return this.transformCall(response);
  }

  /**
   * List calls
   */
  async list(
    params?: {
      agentId?: string;
      direction?: CallDirection;
      status?: CallStatus;
      fromDate?: string;
      toDate?: string;
      limit?: number;
      offset?: number;
    },
    options?: RequestOptions
  ): Promise<PaginatedResponse<Call>> {
    const transformedParams = params
      ? {
          agent_id: params.agentId,
          direction: params.direction,
          status: params.status,
          from_date: params.fromDate,
          to_date: params.toDate,
          limit: params.limit,
          offset: params.offset,
        }
      : undefined;

    const response = await this.http.get<{
      calls: Call[];
      total: number;
      limit: number;
      offset: number;
    }>('/v1/calls', transformedParams, options);

    return {
      items: response.calls.map((c) => this.transformCall(c)),
      total: response.total,
      limit: response.limit,
      offset: response.offset,
      hasMore: response.offset + response.calls.length < response.total,
    };
  }

  /**
   * End a call
   */
  async end(callId: string, options?: RequestOptions): Promise<Call> {
    const response = await this.http.post<Call>(
      `/v1/calls/${callId}/end`,
      undefined,
      options
    );
    return this.transformCall(response);
  }

  /**
   * Transfer a call to another number
   */
  async transfer(
    callId: string,
    toNumber: string,
    options?: RequestOptions
  ): Promise<Call> {
    const response = await this.http.post<Call>(
      `/v1/calls/${callId}/transfer`,
      { to_number: toNumber },
      options
    );
    return this.transformCall(response);
  }

  /**
   * Send DTMF tones to a call
   */
  async sendDtmf(
    callId: string,
    digits: string,
    options?: RequestOptions
  ): Promise<void> {
    await this.http.post(`/v1/calls/${callId}/dtmf`, { digits }, options);
  }

  /**
   * Mute/unmute a call
   */
  async setMute(
    callId: string,
    muted: boolean,
    options?: RequestOptions
  ): Promise<void> {
    await this.http.post(`/v1/calls/${callId}/mute`, { muted }, options);
  }

  /**
   * Update call metadata
   */
  async updateMetadata(
    callId: string,
    metadata: Metadata,
    options?: RequestOptions
  ): Promise<Call> {
    const response = await this.http.patch<Call>(
      `/v1/calls/${callId}`,
      { metadata },
      options
    );
    return this.transformCall(response);
  }

  // Recordings

  /**
   * Get recording URL for a call
   */
  async getRecording(callId: string, options?: RequestOptions): Promise<string> {
    const response = await this.http.get<{ url: string }>(
      `/v1/calls/${callId}/recording`,
      undefined,
      options
    );
    return response.url;
  }

  /**
   * Download recording as binary
   */
  async downloadRecording(callId: string, options?: RequestOptions): Promise<ArrayBuffer> {
    return this.http.getRaw(`/v1/calls/${callId}/recording/download`, undefined, options);
  }

  /**
   * Delete recording
   */
  async deleteRecording(callId: string, options?: RequestOptions): Promise<void> {
    await this.http.delete(`/v1/calls/${callId}/recording`, options);
  }

  // Transcripts

  /**
   * Get transcript for a call
   */
  async getTranscript(callId: string, options?: RequestOptions): Promise<CallTranscript> {
    const response = await this.http.get<CallTranscript>(
      `/v1/calls/${callId}/transcript`,
      undefined,
      options
    );
    return this.transformTranscript(response);
  }

  /**
   * Download transcript as text
   */
  async downloadTranscript(
    callId: string,
    format: 'txt' | 'json' | 'srt' = 'txt',
    options?: RequestOptions
  ): Promise<string> {
    const response = await this.http.get<{ content: string }>(
      `/v1/calls/${callId}/transcript/download`,
      { format },
      options
    );
    return response.content;
  }

  // Analysis

  /**
   * Get call analysis
   */
  async getAnalysis(
    callId: string,
    options?: RequestOptions
  ): Promise<{
    summary: string;
    sentiment: { overall: string; score: number };
    intents: string[];
    entities: Array<{ type: string; value: string }>;
    outcome: string;
  }> {
    return this.http.get(`/v1/calls/${callId}/analysis`, undefined, options);
  }

  /**
   * Trigger analysis for a call
   */
  async analyze(callId: string, options?: RequestOptions): Promise<void> {
    await this.http.post(`/v1/calls/${callId}/analyze`, undefined, options);
  }

  // Polling helpers

  /**
   * Wait for a call to reach a specific status
   */
  async waitForStatus(
    callId: string,
    targetStatuses: CallStatus[],
    options?: {
      timeout?: number;
      pollInterval?: number;
    }
  ): Promise<Call> {
    const timeout = options?.timeout ?? 300000; // 5 minutes default
    const pollInterval = options?.pollInterval ?? 1000;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const call = await this.get(callId);

      if (targetStatuses.includes(call.status)) {
        return call;
      }

      // If call is in terminal state but not target, throw
      if (['completed', 'failed', 'busy', 'no-answer', 'canceled'].includes(call.status)) {
        if (!targetStatuses.includes(call.status)) {
          throw new Error(`Call ended with status ${call.status}, expected one of: ${targetStatuses.join(', ')}`);
        }
      }

      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    throw new Error(`Timeout waiting for call ${callId} to reach status ${targetStatuses.join(' or ')}`);
  }

  /**
   * Wait for call to complete
   */
  async waitForCompletion(
    callId: string,
    options?: { timeout?: number; pollInterval?: number }
  ): Promise<Call> {
    return this.waitForStatus(
      callId,
      ['completed', 'failed', 'busy', 'no-answer', 'canceled'],
      options
    );
  }

  /**
   * Wait for call to be answered
   */
  async waitForAnswer(
    callId: string,
    options?: { timeout?: number; pollInterval?: number }
  ): Promise<Call> {
    return this.waitForStatus(callId, ['in-progress'], options);
  }

  // Private helpers

  private transformRequest(request: CreateCallRequest): Record<string, unknown> {
    return {
      agent_id: request.agentId,
      to_number: request.toNumber,
      from_number: request.fromNumber,
      metadata: request.metadata,
      scheduled_for: request.scheduledFor,
      max_duration: request.maxDuration,
    };
  }

  private transformCall(data: Record<string, unknown>): Call {
    return {
      id: data.id as string,
      agentId: (data.agentId || data.agent_id) as string,
      direction: data.direction as CallDirection,
      status: data.status as CallStatus,
      fromNumber: (data.fromNumber || data.from_number) as string | undefined,
      toNumber: (data.toNumber || data.to_number) as string | undefined,
      startedAt: (data.startedAt || data.started_at) as string | undefined,
      answeredAt: (data.answeredAt || data.answered_at) as string | undefined,
      endedAt: (data.endedAt || data.ended_at) as string | undefined,
      duration: data.duration as number | undefined,
      recordingUrl: (data.recordingUrl || data.recording_url) as string | undefined,
      transcriptUrl: (data.transcriptUrl || data.transcript_url) as string | undefined,
      cost: data.cost as number | undefined,
      metadata: data.metadata as Metadata | undefined,
      createdAt: (data.createdAt || data.created_at) as string,
      updatedAt: (data.updatedAt || data.updated_at) as string,
    };
  }

  private transformTranscript(data: Record<string, unknown>): CallTranscript {
    return {
      id: data.id as string,
      callId: (data.callId || data.call_id) as string,
      turns: (data.turns as Array<Record<string, unknown>>).map((turn) => ({
        speaker: turn.speaker as 'user' | 'agent',
        text: turn.text as string,
        startTime: (turn.startTime || turn.start_time) as number,
        endTime: (turn.endTime || turn.end_time) as number,
        confidence: turn.confidence as number,
        words: turn.words as Array<{
          word: string;
          startTime: number;
          endTime: number;
          confidence: number;
        }>,
      })),
      summary: data.summary as string | undefined,
      sentiment: data.sentiment as CallTranscript['sentiment'],
    };
  }
}
